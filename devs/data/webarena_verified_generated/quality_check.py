#!/usr/bin/env python3
"""Report v0.2 candidate gates and preserve pending tasks in quarantine.

Independent runs use reference_runs/<generated_task_id>/reference_receipt.json.
Human decisions are read from append-only human_review_receipts.jsonl; this
command writes a review packet but never creates or changes human decisions.
Canonical stage and target-model export remain the responsibility of their
existing owners and are reported as pending here.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from contract import (
    OFFICIAL_CATALOG_SHA256,
    ContractError,
    assert_public_projection,
    canonical_json,
    normalize_text,
    sha256_file,
    sha256_json,
    validate_candidate_shape,
)

FORBIDDEN_EVIDENCE_MARKERS = (b"x-m2-admin-auto-login",)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=30)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--human-review-receipts", type=Path)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _contains_forbidden_marker(path: Path) -> str | None:
    def scan(stream: Any) -> str | None:
        overlap = b""
        while chunk := stream.read(1024 * 1024):
            lowered = overlap + chunk.lower()
            for marker in FORBIDDEN_EVIDENCE_MARKERS:
                if marker in lowered:
                    return marker.decode()
            overlap = lowered[-64:]
        return None

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                with archive.open(member) as stream:
                    if marker := scan(stream):
                        return marker
        return None
    with path.open("rb") as stream:
        return scan(stream)


def _answer_overlap_flags(public_task: dict[str, Any], response: dict[str, Any]) -> list[dict]:
    """Flag complete answer values in public text for contextual human review."""
    public_text = (
        " "
        + normalize_text(public_task["intent"] + " " + " ".join(public_task["start_urls"]))
        + " "
    )
    flags = []

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, pointer + "/" + str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, pointer + f"/{index}")
        elif value is not None:
            normalized = normalize_text(canonical_json(value))
            if normalized and " " + normalized + " " in public_text:
                flags.append(
                    {
                        "field": pointer,
                        "value": value,
                        "reason": "Answer value also occurs in the public task. It may be a "
                        "legitimate target parameter; review whether the task exposes the "
                        "answer instead of requiring observation or calculation.",
                    }
                )

    visit(response["retrieved_data"], "/retrieved_data")
    return flags


def build_review_report(
    batch_root: Path,
    *,
    expected_count: int,
    reference_root: Path | None = None,
    human_review_receipts: Path | None = None,
) -> dict[str, Any]:
    """Bind gate receipts to each candidate without promoting pending tasks."""
    private = _load_jsonl(batch_root / "private.jsonl")
    public = _load_jsonl(batch_root / "public.jsonl")
    if len(private) != expected_count or len(public) != expected_count:
        raise ContractError(
            f"expected {expected_count} private/public tasks, found {len(private)}/{len(public)}"
        )
    private_by_id = {row["generated_task_id"]: row for row in private}
    public_by_id = {row["generated_task_id"]: row for row in public}
    if len(private_by_id) != expected_count or len(public_by_id) != expected_count:
        raise ContractError("generated task IDs must be unique")
    expected_ids = set(private_by_id)
    if set(public_by_id) != expected_ids:
        raise ContractError("public task IDs do not match the private manifest")

    receipt_maps = {}
    for name in ("replay", "contamination"):
        path = batch_root / f"{name}_receipts.jsonl"
        rows = _load_jsonl(path) if path.is_file() else []
        by_id = {row["generated_task_id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) - expected_ids:
            raise ContractError(f"{name} receipt IDs are duplicate or outside the manifest")
        receipt_maps[name] = by_id
    reference_root = reference_root or batch_root / "reference_runs"
    human_path = human_review_receipts or batch_root / "human_review_receipts.jsonl"
    human_rows = _load_jsonl(human_path) if human_path.is_file() else []
    if any(row["generated_task_id"] not in expected_ids for row in human_rows):
        raise ContractError("human review task IDs are outside the private manifest")
    # The last appended decision owns the current review; history stays untouched.
    human_by_id = {row["generated_task_id"]: row for row in human_rows}

    review_rows = []
    for generated_id in sorted(expected_ids):
        candidate = private_by_id[generated_id]
        public_task = public_by_id[generated_id]
        validate_candidate_shape(candidate)
        assert_public_projection(public_task, candidate)
        bindings = {
            "task_definition_sha256": sha256_json(candidate["task_definition"]),
            "task_spec_sha256": sha256_json(candidate["task_spec"]),
            "baseline_sha256": sha256_json(candidate["baseline"]),
            "grounding_sha256": sha256_json(candidate["grounding"]),
            "public_projection_sha256": sha256_json(public_task),
        }
        artifacts = {}
        evidence_problems = []
        for name, relative in candidate["grounding"]["artifacts"].items():
            path = (batch_root / relative).resolve()
            if not path.is_relative_to(batch_root.resolve()) or not path.is_file():
                evidence_problems.append(f"invalid grounding artifact: {name}")
                continue
            digest = sha256_file(path)
            artifacts[name] = {"path": relative, "sha256": digest}
            if digest != candidate["grounding"]["artifact_sha256"][name]:
                evidence_problems.append(f"grounding artifact hash mismatch: {name}")
            if marker := _contains_forbidden_marker(path):
                evidence_problems.append(f"grounding artifact {name} contains {marker}")
        bindings["artifact_sha256"] = {name: item["sha256"] for name, item in artifacts.items()}
        stages = {
            "contract": {"status": "pass"},
            "grounding_evidence": {
                "status": "quarantine" if evidence_problems else "pass",
                "reasons": evidence_problems,
            },
        }

        replay_receipt = receipt_maps["replay"].get(generated_id)
        if replay_receipt is None:
            stages["author_replay"] = {"status": "pending", "reason": "Receipt is absent."}
        else:
            replay_problems = [
                f"receipt binding mismatch: {key}"
                for key in (
                    "task_definition_sha256",
                    "task_spec_sha256",
                    "baseline_sha256",
                    "grounding_sha256",
                    "artifact_sha256",
                )
                if replay_receipt.get(key) != bindings[key]
            ]
            if replay_receipt.get("reference_response_sha256") != sha256_json(
                candidate["reference_response"]
            ):
                replay_problems.append("reference answer hash mismatch")
            if replay_receipt.get("gate_version") != 2:
                replay_problems.append("author replay gate version is not 2")
            if (
                replay_receipt.get("status") != "pass"
                or not replay_receipt.get("checks")
                or not all(replay_receipt["checks"].values())
            ):
                replay_problems.append("author replay checks did not pass")
            for key in ("webarena_verified_version", "webarena_verified_evaluator_checksum"):
                bindings[key] = replay_receipt.get(key)
                if not bindings[key]:
                    replay_problems.append(f"missing evaluator binding: {key}")
            stages["author_replay"] = {
                "status": "quarantine" if replay_problems else "pass",
                "reasons": replay_problems,
                "checks": replay_receipt.get("checks", {}),
                "evaluator_errors": replay_receipt.get("evaluator_errors", []),
                "evaluator_limitations": replay_receipt.get("evaluator_limitations", []),
            }

        contamination = receipt_maps["contamination"].get(generated_id)
        if contamination is None:
            stages["contamination"] = {"status": "pending", "reason": "Receipt is absent."}
        else:
            contamination_valid = (
                contamination.get("status") == "pass"
                and contamination.get("candidate_sha256") == sha256_json(candidate)
                and contamination.get("official_catalog_sha256") == OFFICIAL_CATALOG_SHA256
            )
            stages["contamination"] = {
                "status": "pass" if contamination_valid else "quarantine",
                "receipt_status": contamination.get("status"),
                "reasons": contamination.get("reasons", []),
            }

        reference_case_root = reference_root / generated_id
        reference_path = reference_case_root / "reference_receipt.json"
        if not reference_path.is_file():
            stages["independent_reference"] = {
                "status": "pending",
                "reason": "Independent model execution is absent.",
            }
        else:
            reference = json.loads(reference_path.read_text())
            bindings["reference_receipt_sha256"] = sha256_json(reference)
            reference_problems = [
                f"reference binding mismatch: {key}"
                for key in (
                    "task_spec_sha256",
                    "baseline_sha256",
                    "public_projection_sha256",
                    "webarena_verified_version",
                    "webarena_verified_evaluator_checksum",
                )
                if reference.get(key) != bindings.get(key)
            ]
            if reference.get("generated_task_id") != generated_id:
                reference_problems.append("reference task ID mismatch")
            if (
                reference.get("reference_kind") != "independent_model_execution"
                or reference.get("public_only") is not True
            ):
                reference_problems.append("execution is not an independent public-only run")
            if (
                reference.get("status") != "pass"
                or reference.get("execution_status") != "completed"
                or reference.get("agent_ended") is not True
            ):
                reference_problems.append("independent execution did not complete and pass")
            if not reference.get("model_id") or not reference.get(
                "execution_baseline_receipt_sha256"
            ):
                reference_problems.append("model or execution-baseline provenance is absent")
            for check in (
                "response_contract_valid",
                "all_evaluators_pass",
                "har_playwright_results_equal",
                "canonical_trajectory_present",
                "final_state_confirmed",
            ):
                if reference.get("checks", {}).get(check) is not True:
                    reference_problems.append(f"reference check did not pass: {check}")
            for source in ("har_results", "playwright_results"):
                results = reference.get(source, [])
                if len(results) != len(candidate["task_spec"]["eval"]) or not all(
                    result["status"] == "success" and result["score"] == 1 for result in results
                ):
                    reference_problems.append(f"reference evaluator vector did not pass: {source}")
            reference_artifacts = reference.get("artifacts", {})
            if not {"har", "trace", "trajectory", "initial_screenshot"}.issubset(
                reference_artifacts
            ):
                reference_problems.append("required execution artifacts are absent")
            for name, relative in reference_artifacts.items():
                path = (reference_case_root / relative).resolve()
                if not path.is_relative_to(reference_case_root.resolve()) or not path.is_file():
                    reference_problems.append(f"invalid execution artifact: {name}")
                elif sha256_file(path) != reference.get("artifact_sha256", {}).get(name):
                    reference_problems.append(f"execution artifact hash mismatch: {name}")
            stages["independent_reference"] = {
                "status": "quarantine" if reference_problems else "pass",
                "reasons": reference_problems,
                "receipt_sha256": bindings["reference_receipt_sha256"],
                "model_id": reference.get("model_id"),
            }

        overlap_flags = _answer_overlap_flags(public_task, candidate["reference_response"])
        human = human_by_id.get(generated_id)
        if human is None:
            stages["human_review"] = {"status": "pending", "reason": "Human decision is absent."}
        else:
            human_problems = []
            if human.get("bindings") != bindings:
                human_problems.append("human review bindings are stale or incomplete")
            if human.get("decision") != "accept":
                human_problems.append("human has not accepted this candidate")
            if not human.get("reviewer") or not human.get("reviewed_at"):
                human_problems.append("reviewer or review time is absent")
            if overlap_flags and human.get("answer_overlap_reviewed") is not True:
                human_problems.append("public answer overlap requires contextual review")
            stages["human_review"] = {
                "status": "quarantine" if human_problems else "pass",
                "reasons": human_problems,
                "decision": human.get("decision"),
            }
        ready_for_stage = all(stage["status"] == "pass" for stage in stages.values())
        stages["canonical_stage"] = {
            "status": "pending",
            "reason": "Run canonical stage with its existing owner.",
        }
        stages["target_model_export"] = {
            "status": "pending",
            "reason": "Target-model SFT export has not been verified.",
        }
        review_rows.append(
            {
                "generated_task_id": generated_id,
                "site": candidate["task_spec"]["sites"][0],
                "intent": candidate["task_spec"]["intent"],
                "task_type": candidate["reference_response"]["task_type"],
                "reference_response": candidate["reference_response"],
                "status": "quarantine",
                "ready_for_stage": ready_for_stage,
                "training_eligible": False,
                "stages": stages,
                "bindings": bindings,
                "answer_overlap_review_flags": overlap_flags,
                "grounding_artifacts": artifacts,
                "nearest_official_task": contamination.get("nearest_official_task")
                if contamination
                else None,
            }
        )

    report = {
        "gate_version": 2,
        "status": "quarantine",
        "task_count": len(review_rows),
        "all_single_site": all(len(row["task_spec"]["sites"]) == 1 for row in private),
        "official_tasks_used_for_generation": False,
        "staged_for_training": False,
        "training_eligible_count": 0,
        "ready_for_stage_count": sum(row["ready_for_stage"] for row in review_rows),
        "stage_counts": {
            name: {
                status: sum(row["stages"][name]["status"] == status for row in review_rows)
                for status in ("pass", "pending", "quarantine")
            }
            for name in review_rows[0]["stages"]
        }
        if review_rows
        else {},
        "tasks": review_rows,
    }
    report_root = batch_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "machine_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# WebArena-Verified v0.2: Human Review Packet",
        "",
        "Candidates remain quarantined until all promotion gates pass.",
        "",
        "Review the public goal, visible evidence, answer or final state, independent "
        "execution, and each pending gate. Record decisions separately in "
        "human_review_receipts.jsonl with the bindings below. This packet is regenerated.",
        "",
    ]
    for row in review_rows:
        lines.extend(
            [
                f"## {row['generated_task_id']} — {row['site']} / {row['task_type']}",
                "",
                f"- Intent: {row['intent']}",
                "- Reference response: " + canonical_json(row["reference_response"]),
                "- Stages: " + canonical_json(row["stages"]),
                "- Answer overlap review: " + canonical_json(row["answer_overlap_review_flags"]),
                "- Bindings: " + canonical_json(row["bindings"]),
                "- Decision: pending unless an independently recorded review is bound above.",
                "",
            ]
        )
        for name, artifact in row["grounding_artifacts"].items():
            link = (Path("..") / artifact["path"]).as_posix()
            lines.append(f"- {name}: [{artifact['path']}]({link})")
        lines.append("")
    (report_root / "human_review_packet.md").write_text("\n".join(lines))
    return report


def main() -> None:
    args = _parse_args()
    report = build_review_report(
        args.batch_root,
        expected_count=args.expected_count,
        reference_root=args.reference_root,
        human_review_receipts=args.human_review_receipts,
    )
    print(
        f"candidates={report['task_count']} ready_for_stage={report['ready_for_stage_count']} "
        f"training_eligible={report['training_eligible_count']} "
        f"report={args.batch_root / 'reports' / 'machine_validation.json'}"
    )
    if report["status"] != "pass":
        raise ContractError("promotion gates remain pending or quarantined; reports were written")


if __name__ == "__main__":
    main()
