#!/usr/bin/env python3
"""Merge the published legacy delivery and a newly verified delivery.

This explicit published-data boundary retains every source execution snapshot
and cleaning receipt. It performs no sampling, cleaning, or student-based
selection. Run in the pinned cleaning environment with ``python -B``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from clean_native_trajectories import sha256_file, verify_execution_evidence
from clean_v3_projection import audit_roundtrip
from export_review_bundle import (
    absolute_image_paths,
    export_absolute_jsonl,
    export_trajectory_files,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-delivery", type=Path, required=True)
    parser.add_argument("--new-delivery", type=Path, required=True)
    parser.add_argument("--legacy-student-report", type=Path, required=True)
    parser.add_argument(
        "--legacy-student-evidence-index",
        type=Path,
        required=True,
        help="Audited normalized student episodes bound to the legacy report SHA",
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--require-trajectories", type=int, default=350)
    return parser.parse_args()


def _copy_evidence(source: Path, target: Path, source_hashes: dict[Path, str]) -> None:
    """Copy immutable evidence bytes and retain hashes for final source readback."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
        files = [
            (path, target / path.relative_to(source))
            for path in source.rglob("*")
            if path.is_file()
        ]
    else:
        shutil.copyfile(source, target)
        files = [(source, target)]
    for original, copied in files:
        digest = sha256_file(original)
        if sha256_file(copied) != digest:
            raise ValueError(f"Copied evidence differs from its source: {original}")
        source_hashes[original] = digest


def main() -> int:
    """Verify each published source, copy its evidence, and export the whole cohort."""
    args = _parse_args()
    legacy = args.legacy_delivery.resolve()
    new = args.new_delivery.resolve()
    target = args.target.resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing merge: {target}")
    legacy_manifest = json.loads((legacy / "delivery_manifest.json").read_text())
    new_manifest = json.loads((new / "review_manifest.json").read_text())
    if new_manifest["scope"] != "v03_formal":
        raise ValueError("The new delivery must be a formally verified review bundle")

    # Freeze complete published membership before reading any student outcomes.
    order = []
    published_hashes = {}
    source_hashes = {}
    for bundle, expected_sha, expected_count in (
        (legacy, legacy_manifest["combined_JSONL_SHA256"], legacy_manifest["combined_rows"]),
        (new, new_manifest["jsonl_sha256"], new_manifest["sample_rows"]),
    ):
        path = bundle / "multimodal_image_text_dialogs_v3.jsonl"
        source_hashes[path] = sha256_file(path)
        if source_hashes[path] != expected_sha:
            raise ValueError("Published JSONL differs from its delivery receipt")
        count = 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                task = row["metadata"]["source_task_id"]
                if task in published_hashes:
                    raise ValueError(f"Duplicate published task ID: {task}")
                if row["metadata"]["scope"] != "v03_formal":
                    raise ValueError("Only formal published trajectories may be merged")
                published_hashes[task] = hashlib.sha256(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                for suffix in ("json", "jsonl"):
                    paired = bundle / (suffix + "_absolute") / (task + "." + suffix)
                    if json.loads(paired.read_text()) != row:
                        raise ValueError("Published trajectory pair differs from its combined row")
                    source_hashes[paired] = sha256_file(paired)
                order.append(task)
                count += 1
        if count != expected_count:
            raise ValueError("Published row count differs from its delivery receipt")
    legacy_ids = order[: legacy_manifest["combined_rows"]]
    new_ids = order[legacy_manifest["combined_rows"] :]
    if (
        legacy_ids != legacy_manifest["formal_order_case_ids"]
        or len(order) != args.require_trajectories
    ):
        raise ValueError("Published membership differs from the requested complete merge")
    pairs = {row["task_id"]: row for row in legacy_manifest["pairs_provenance"]}
    if len(pairs) != legacy_manifest["pairs"] or set(pairs) != set(legacy_ids):
        raise ValueError("Legacy source provenance differs from published membership")
    sources = {}
    task_sources = {}
    for task in legacy_ids:
        pair = pairs[task]
        source = Path(pair["existing_export_audit"]).resolve().parent
        for suffix in ("json", "jsonl"):
            if sha256_file(Path(pair[suffix]["source"])) != pair[suffix]["SHA256"]:
                raise ValueError("Legacy source pair differs from its original export")
            if (
                source_hashes[legacy / (suffix + "_absolute") / (task + "." + suffix)]
                != pair[suffix]["SHA256"]
            ):
                raise ValueError("Legacy delivered pair differs from its source receipt")
        if source not in sources:
            sources[source] = {"source_id": f"legacy-{len(sources):02d}", "tasks": []}
        sources[source]["tasks"].append(task)
        task_sources[task] = source
    if new in sources:
        raise ValueError("New delivery must be distinct from the legacy review sources")
    sources[new] = {"source_id": "new", "tasks": new_ids}
    task_sources.update({task: new for task in new_ids})
    student_index_path = args.legacy_student_evidence_index.resolve()
    student_index = json.loads(student_index_path.read_text())
    if student_index["schema"] != "legacy_student_episode_index_v1":
        raise ValueError("Use the normalized legacy student episode evidence contract")
    student_sources = {row["task"]: row for row in student_index["rows"]}
    if len(student_sources) != student_index["count"] or set(student_sources) != set(legacy_ids):
        raise ValueError("Legacy student evidence must cover every published legacy task once")
    if any(
        target.is_relative_to(source)
        for source in (
            legacy,
            args.legacy_student_report.resolve().parent,
            *sources,
            *(Path(row["source"]).resolve() for row in student_index["rows"]),
        )
    ):
        raise ValueError("Merge target must be outside the published source directories")

    target.mkdir(parents=True, exist_ok=False)
    (target / "audits").mkdir()
    (target / "original").mkdir()
    provenance = target / "provenance"
    provenance.mkdir()
    (target / "merge.intent.json").write_text(
        json.dumps(
            {
                "membership": order,
                "require_trajectories": args.require_trajectories,
                "selection": "all published teacher-qualified rows in original delivery order",
                "student_outcomes_used_for_selection": False,
            },
            indent=2,
        )
        + "\n"
    )

    # Archive each original snapshot and cleaning owner receipt without reserializing it.
    offsets = {}
    snapshots = {}
    for source, info in sources.items():
        archived = provenance / "sources" / info["source_id"]
        archived.mkdir(parents=True)
        for name in (
            "model_execution.json",
            "verification.json",
            "profile.yaml",
            "prepared_plan.json",
            "image_manifest.json",
            "input_manifest.jsonl",
            "review_manifest.json",
            "export_audits.json",
            "copied_bundle_audits.json",
            "multimodal_image_text_dialogs_v3.export_audit.json",
            "json_absolute.export_manifest.json",
            "jsonl_absolute.export_manifest.json",
            "audits",
            "rejected",
        ):
            _copy_evidence(source / name, archived / name, source_hashes)
        verification = json.loads((source / "verification.json").read_text())
        snapshot = source / "model_execution.json"
        digest = sha256_file(snapshot)
        if (
            verification["scope"] != "v03_formal"
            or verification["model_execution_sha256"] != digest
        ):
            raise ValueError("Published cleaning receipt differs from its original snapshot")
        if verification["versions"] != {"internflow": "0.2.0", "internvalidata": "0.4.17"}:
            raise ValueError("Published source did not use the pinned cleaning versions")
        if verification["operations"] != 46 or verification["groups"] != 3:
            raise ValueError("Published source did not pass the fixed46 cleaning pipeline")
        info["model_execution_sha256"] = digest
        info["archived_snapshot"] = str(archived / "model_execution.json")
        published_execution = json.loads(snapshot.read_text())
        info["executions"] = {}
        if "source_executions" in published_execution:
            # This published-data owner supports immutable legacy bundles and
            # the new source-snapshot index. Normal clean/export readers use
            # only the new explicit index contract.
            _copy_evidence(source / "model_executions", archived / "model_executions", source_hashes)
            for source_digest, relative in published_execution["source_executions"].items():
                actual = source / relative
                if sha256_file(actual) != source_digest:
                    raise ValueError("Published source execution snapshot differs from its index")
                archived_snapshot = str(archived / relative)
                info["executions"][source_digest] = {"source_snapshot": str(actual),
                                                     "archived_snapshot": archived_snapshot}
                snapshots.setdefault(source_digest, []).append(archived_snapshot)
        else:
            info["executions"][digest] = {"source_snapshot": str(snapshot),
                                          "archived_snapshot": info["archived_snapshot"]}
            snapshots.setdefault(digest, []).append(info["archived_snapshot"])
        relative = source / "audits/cleaning_projection.relative.jsonl"
        if sha256_file(relative) != verification["published_sha256"]:
            raise ValueError("Relative projection differs from its original cleaning receipt")
        wanted = set(info["tasks"])
        position = 0
        with relative.open("rb") as stream:
            for line in stream:
                row = json.loads(line)
                task = row["metadata"]["source_task_id"]
                if task in wanted:
                    if task in offsets:
                        raise ValueError("Source projection repeats a selected task")
                    offsets[task] = (relative, position)
                position += len(line)
    if set(offsets) != set(order):
        raise ValueError("Original relative projections do not cover published membership")

    # Verify native decisions and real teacher requests against each source's own snapshot.
    row_provenance = []
    relative_output = target / "audits/cleaning_projection.relative.jsonl"
    sites = Counter()
    teacher_models = Counter()
    with relative_output.open("x") as output:
        for task in order:
            source = task_sources[task]
            info = sources[source]
            path, position = offsets[task]
            with path.open("rb") as stream:
                stream.seek(position)
                row = json.loads(stream.readline())
            original_export, _ = absolute_image_paths(row, source)
            fingerprint = hashlib.sha256(
                json.dumps(
                    original_export, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            if fingerprint != published_hashes[task]:
                raise ValueError("Published row differs from its original reversible projection")
            md = row["metadata"]
            row_execution_sha = md["execution_evidence"]["model_execution_sha256"]
            row_execution = info["executions"][row_execution_sha]
            original = source / "original" / task
            native = json.loads((original / "agent_trajectory.jsonl").read_text())
            if sha256_file(original / "agent_trajectory.jsonl") != md["source_trajectory_sha256"]:
                raise ValueError("Native trajectory differs from its published provenance")
            if sha256_file(original / "canonical_tools.json") != md["canonical_tools_sha256"]:
                raise ValueError("Canonical tool schemas differ from published provenance")
            if row["tools"] != native["metadata"]["tools"] or row["tools"] != json.loads(
                (original / "canonical_tools.json").read_text()
            ):
                raise ValueError("Published tools differ from their native owning schemas")
            roundtrip = audit_roundtrip(row, native, source)
            execution = verify_execution_evidence(
                {
                    "task_id": task,
                    "request_dir": str(original / "requests"),
                    "score_receipt": str(original / "reference_receipt.json"),
                    "api_type": md["source_api_type"],
                    "model_execution": row_execution["source_snapshot"],
                    "model_execution_sha256": row_execution_sha,
                },
                native,
                json.loads((original / "public_task.json").read_text()),
                True,
            )
            for key in (
                "model_execution_sha256",
                "request_evidence_sha256",
                "score_receipt_sha256",
            ):
                if execution[key] != md["execution_evidence"][key]:
                    raise ValueError("Published execution evidence differs from its source")
            _copy_evidence(original, target / "original" / task, source_hashes)
            for stage in (
                "001_common_format_fixes",
                "002_common_format_filters",
                "003_gui_behavior_filters",
            ):
                _copy_evidence(
                    source / "processed" / stage / (task + ".json"),
                    target / "processed" / stage / (task + ".json"),
                    source_hashes,
                )
            for reference in md["image_aliases"].values():
                media = Path(reference["url"])
                if media.is_absolute():
                    raise ValueError("Original cleaning projections must use relative media paths")
                if not (target / media).exists():
                    _copy_evidence(source / media, target / media, source_hashes)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            row_provenance.append(
                {
                    "task": task,
                    "source_id": info["source_id"],
                    "original_review_bundle": str(source),
                    "model_execution_sha256": row_execution_sha,
                    "archived_snapshot": row_execution["archived_snapshot"],
                    "source_trajectory_sha256": md["source_trajectory_sha256"],
                    "native_roundtrip": roundtrip,
                    "teacher_score_checks": execution["score_checks"],
                }
            )
            sites.update(md["sites"])
            teacher_models.update([md["source_model_id"]])

    # Keep original new author/student evidence and historical student report bytes.
    _copy_evidence(
        legacy / "delivery_manifest.json",
        provenance / "legacy_delivery_manifest.json",
        source_hashes,
    )
    legacy_report_path = args.legacy_student_report.resolve()
    legacy_report = json.loads(legacy_report_path.read_text())
    report_hashes = json.loads((legacy_report_path.parent / "files_SHA256.json").read_text())[
        "files"
    ]
    if sha256_file(legacy_report_path) != report_hashes[legacy_report_path.name]:
        raise ValueError("Legacy student report differs from its published report manifest")
    if legacy_report["original_report_SHA256"] != legacy_manifest["final_report_SHA256"]:
        raise ValueError("Legacy reassessment is bound to a different original delivery report")
    if student_index["legacy_student_report_sha256"] != sha256_file(legacy_report_path):
        raise ValueError("Legacy student evidence index is bound to a different report")
    for name, reference in legacy_manifest[
        "existing_formal_methods_GLM_and_report_references"
    ].items():
        source = Path(reference["path"])
        if sha256_file(source) != reference["SHA256"]:
            raise ValueError("Historical report differs from the legacy delivery manifest")
        _copy_evidence(source, provenance / "legacy_reports" / name, source_hashes)
    _copy_evidence(legacy_report_path.parent, provenance / "legacy_student_report", source_hashes)
    old_rows = {row["case_id"]: row for row in legacy_report["rows"]}
    reassessed = {row["case_id"] for row in legacy_report["reevaluation_rows"]}
    _copy_evidence(
        student_index_path, provenance / "legacy_student_evidence_index.json", source_hashes
    )
    for task, entry in student_sources.items():
        original = Path(entry["source"]).resolve()
        receipt_path = original / "reference_receipt.json"
        if sha256_file(receipt_path) != entry["receipt_sha256"]:
            raise ValueError("Legacy student receipt differs from its audited index")
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["generated_task_id"] != task
            or receipt["model_id"] != "glm-5.3-flash"
            or receipt["api_type"] != "openai_chat"
        ):
            raise ValueError("Legacy student evidence has a different task or model identity")
        status = (
            "PASS"
            if old_rows[task]["category"] == "pass"
            else "FAIL"
            if old_rows[task]["effective_scored"]
            else "HOLD"
        )
        if (
            entry["reported_category"] != old_rows[task]["category"]
            or entry["status"] != status
            or entry["effective_scored"] != old_rows[task]["effective_scored"]
        ):
            raise ValueError("Legacy student evidence differs from the retained reported outcome")
        if (entry["origin"] == "reassessment21") != (task in reassessed):
            raise ValueError("Legacy student episode origin differs from the reassessment cohort")
        identity = entry["actual_identity"]
        if (
            identity["receipt_model_id"] != receipt["model_id"]
            or identity["api_type"] != receipt["api_type"]
            or identity["native_model_id"] != "glm-5.3-flash"
            or identity["wire_request_models"] != ["glm-5.3-flash"]
        ):
            raise ValueError("Legacy student actual identity differs from its receipt")
        native = json.loads((original / "agent_trajectory.jsonl").read_text())
        if native["metadata"]["model_id"] != identity["native_model_id"]:
            raise ValueError("Legacy student native identity differs from its audited index")
        requests = original / "requests"
        actual_files = {
            str(path.relative_to(requests)): path for path in requests.rglob("*") if path.is_file()
        }
        if set(actual_files) != set(entry["request_evidence_sha256"]):
            raise ValueError("Legacy student request evidence membership differs from its index")
        request_models = set()
        response_models = set()
        for relative, path in actual_files.items():
            if sha256_file(path) != entry["request_evidence_sha256"][relative]:
                raise ValueError("Legacy student request evidence differs from its audited index")
            if path.name.endswith(".request.json"):
                request_models.add(json.loads(path.read_text())["model"])
            elif path.name.endswith(".response.json"):
                response = json.loads(path.read_text())
                if "model" in response:
                    response_models.add(response["model"])
        if (
            sorted(request_models) != identity["wire_request_models"]
            or sorted(response_models) != identity["wire_successful_response_models"]
        ):
            raise ValueError("Legacy student wire identity differs from its audited index")
        artifact_hashes = {
            relative: receipt["artifact_sha256"][name]
            for name, relative in receipt["artifacts"].items()
        }
        if artifact_hashes != entry["artifacts_sha256"]:
            raise ValueError("Legacy student artifact inventory differs from its audited index")
        for relative, digest in artifact_hashes.items():
            if sha256_file(original / relative) != digest:
                raise ValueError("Legacy student artifact differs from its owner receipt")
        _copy_evidence(original, provenance / "legacy_students" / task, source_hashes)
        evidence_report = Path(entry["source_report"])
        if sha256_file(evidence_report) != entry["source_report_sha256"]:
            raise ValueError("Legacy student source report differs from its evidence index")
        archived_report = (
            provenance / "legacy_student_source_reports" / (entry["source_report_sha256"] + ".json")
        )
        if not archived_report.exists():
            _copy_evidence(evidence_report, archived_report, source_hashes)
    for audit in student_index["audits"]:
        source = Path(audit["path"])
        if sha256_file(source) != audit["sha256"]:
            raise ValueError("Legacy supplementary audit differs from its evidence index")
        _copy_evidence(source, provenance / "legacy_audits" / source.name, source_hashes)
    new_evidence = json.loads((new / "evidence_manifest.json").read_text())
    actual_evidence = {
        str(path.relative_to(new / "evidence"))
        for path in (new / "evidence").rglob("*")
        if path.is_file()
    }
    if actual_evidence != set(new_evidence["files_SHA256"]):
        raise ValueError("New author/student evidence membership differs from its manifest")
    for relative, digest in new_evidence["files_SHA256"].items():
        if sha256_file(new / "evidence" / relative) != digest:
            raise ValueError("New author/student evidence differs from its manifest")
    _copy_evidence(new / "evidence", provenance / "new_evidence", source_hashes)
    _copy_evidence(
        new / "evidence_manifest.json", provenance / "new_evidence_manifest.json", source_hashes
    )
    _copy_evidence(
        new / "difficulty_report.json", provenance / "new_difficulty_report.json", source_hashes
    )

    # Aggregate frozen assessments after membership is fixed; HOLD never means FAIL.
    new_report = json.loads((new / "difficulty_report.json").read_text())
    new_rows = {row["task"]: row for row in new_report["rows"]}
    if set(old_rows) != set(legacy_ids) or set(new_rows) != set(new_ids):
        raise ValueError("Student report membership differs from the delivered cohorts")
    old_counts = Counter(
        "PASS" if row["category"] == "pass" else "FAIL" if row["effective_scored"] else "HOLD"
        for row in old_rows.values()
    )
    new_counts = Counter(row["status"] for row in new_rows.values())
    if set(new_counts) - {"PASS", "FAIL", "HOLD"}:
        raise ValueError("New student outcomes must be PASS, FAIL, or HOLD")
    counts = {
        status: old_counts[status] + new_counts[status] for status in ("PASS", "FAIL", "HOLD")
    }
    effective = counts["PASS"] + counts["FAIL"]
    verification = {
        "scope": "v03_formal_published_bundle_merge",
        "accepted": len(order),
        "published_sha256": sha256_file(relative_output),
        "fixed46_reexecuted": False,
        "source_cleaning_receipts_preserved": True,
        "execution_snapshots": snapshots,
        "student_outcomes_used_for_selection": False,
    }
    (target / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    exported = export_absolute_jsonl(
        target, relative_output, target / "multimodal_image_text_dialogs_v3.jsonl"
    )
    export_trajectory_files(
        target / "multimodal_image_text_dialogs_v3.jsonl", target / "json_absolute", "json"
    )
    export_trajectory_files(
        target / "multimodal_image_text_dialogs_v3.jsonl", target / "jsonl_absolute", "jsonl"
    )
    for path, digest in source_hashes.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Source evidence changed during merge: {path}")
    difficulty = {
        "scope": "frozen_legacy_reassessment_plus_independent_new_cohort",
        "delivered_count": len(order),
        "legacy_count": len(legacy_ids),
        "new_count": len(new_ids),
        "legacy_student_counts": {
            status: old_counts[status] for status in ("PASS", "FAIL", "HOLD")
        },
        "new_student_counts": {status: new_counts[status] for status in ("PASS", "FAIL", "HOLD")},
        "student_counts": counts,
        "student_effective": effective,
        "student_success_rate": counts["PASS"] / effective if effective else None,
        "legacy_historical_manifest_counts": {
            "PASS": legacy_manifest["GLM_pass"],
            "FAIL": legacy_manifest["GLM_fail"],
            "HOLD": legacy_manifest["GLM_HOLD"],
        },
        "legacy_reassessment_SHA256": sha256_file(legacy_report_path),
        "new_difficulty_methods_and_rules": new_report["difficulty_methods_and_rules"],
        "new_cohort_report": new_report,
        "site_distribution": dict(sites),
        "teacher_models": dict(teacher_models),
        "student_outcomes_used_for_selection": False,
        "uniform_current_reevaluation": False,
        "legacy_original_student_evidence_references_preserved": True,
        "legacy_student_raw_archived": len(student_sources),
        "legacy_original_valid_episodes": len(student_sources) - len(reassessed),
        "legacy_reassessment_episodes": len(reassessed),
        "training_promotion": False,
    }
    (target / "difficulty_report.json").write_text(
        json.dumps(difficulty, ensure_ascii=False, indent=2) + "\n"
    )
    (target / "row_provenance.json").write_text(json.dumps(row_provenance, indent=2) + "\n")
    manifest = {
        "exported_at": datetime.now(UTC).isoformat(),
        "scope": "v03_formal_published_bundle_merge",
        "sample_rows": len(order),
        "legacy_rows": len(legacy_ids),
        "new_rows": len(new_ids),
        "unique_ids": len(order),
        "jsonl_sha256": exported["jsonl_sha256"],
        "image_paths": "absolute_shared_storage",
        "execution_snapshots": snapshots,
        "snapshot_count": len(snapshots),
        "source_files_unchanged": True,
        "fixed46_reexecuted": False,
        "student_outcomes_used_for_selection": False,
        "review_status": "pending_human_review",
        "training_promotion": "not_promoted",
        "raw_evidence_policy": "Original native/evaluation artifacts and historical references retain their bytes; delivered V3 JSON/JSONL image references point inside this bundle.",
    }
    (target / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (target / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "files_SHA256": {
                    str(path.relative_to(target)): sha256_file(path)
                    for path in sorted(target.rglob("*"))
                    if path.is_file()
                },
                "original_source_files_SHA256": {
                    str(path): digest for path, digest in source_hashes.items()
                },
                "source_files_unchanged": True,
                "credential_files_included": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
