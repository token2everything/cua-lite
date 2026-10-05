#!/usr/bin/env python3
"""Validate one GLM campaign case before publishing its terminal receipt.

Only a complete, run-bound archive can be terminal. Official evaluator errors
are archived as unscored cases, never silently converted into model failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EvidenceInvalid(ValueError):
    """The case is incomplete or cannot be trusted as terminal evidence."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise EvidenceInvalid(reason)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceInvalid(f"cannot read {path.name}: {error}") from error
    _require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def _numeric(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _identity_matches(identity: Any, data: bytes, name: str) -> bool:
    return (
        isinstance(identity, dict)
        and identity.get("sha256") == hashlib.sha256(data).hexdigest()
        and identity.get("size") == len(data)
        and identity.get("path", name) == name
    )


def make_receipt(
    index: Path,
    archive: Path,
    *,
    run_id: str,
    plan_sha256: str,
    worker: str,
    sandbox_id: str,
    task_id: int,
    runner_sha256: str,
    image_lock_sha256: str,
    runner_exit_code: int,
) -> dict[str, Any]:
    """Return a terminal receipt or raise EvidenceInvalid for retry/inspection."""
    _require(runner_exit_code == 0, "runner did not exit successfully")
    _require(archive.is_file(), "case archive is missing")
    case_key = f"task-{task_id:04d}"
    status = _json(index / "status.json")
    evaluation = _json(index / "evaluation.json")
    metadata = _json(index / "run_metadata.json")
    baseline = _json(index / "baseline_receipt.json")
    startup = _json(index / "startup_receipt.json")

    _require(status.get("task_id") == task_id, "status task ID mismatch")
    _require(status.get("model") == "glm-5.3-flash", "wrong model")
    _require(status.get("model_profile") == "glm53flash", "wrong model profile")
    _require(
        status.get("status") in {"completed", "model_protocol_failed", "evaluator_error"},
        "case is not terminal",
    )
    _require(not status.get("error"), "runner reported an execution error")
    parameters = status.get("fixed_parameters") or {}
    _require(parameters.get("base_max_steps") == 150, "step budget mismatch")
    _require(parameters.get("final_response_reserve_steps") == 7, "reserve budget mismatch")
    _require(parameters.get("coordinate_type") == "absolute", "coordinate policy mismatch")
    _require(parameters.get("startup_page_gate") is True, "startup gate was not enabled")
    _require(parameters.get("reasoning_effort") == "max", "reasoning effort mismatch")
    _require(metadata.get("script_sha256") == runner_sha256, "runner SHA mismatch")
    _require(startup.get("result") == "passed", "startup page gate did not pass")
    _require(
        isinstance(startup.get("selected_page_url"), str) and bool(startup["selected_page_url"]),
        "startup page not selected",
    )
    for key, expected in {
        "session_id": run_id,
        "sandbox_id": sandbox_id,
        "worker": worker,
        "task_id": task_id,
        "image_lock_sha256": image_lock_sha256,
    }.items():
        _require(baseline.get(key) == expected, f"baseline {key} mismatch")
    _require(baseline.get("fresh_service_processes") is True, "site baseline is not fresh")

    har = evaluation.get("har") or {}
    trace = evaluation.get("playwright_trace") or {}
    _require(isinstance(har, dict) and isinstance(trace, dict), "official results missing")
    _require(evaluation.get("inputs_agree") is True, "HAR/trace contracts disagree")
    _require(evaluation.get("score_inputs_agree") is True, "HAR/trace scores disagree")
    for field in (
        "task_id",
        "status",
        "score",
        "webarena_verified_version",
        "webarena_verified_evaluator_checksum",
        "webarena_verified_data_checksum",
    ):
        _require(har.get(field) == trace.get(field), f"HAR/trace {field} mismatch")
    _require(har.get("task_id") == task_id, "official task ID mismatch")
    _require(har.get("status") in {"success", "failure", "error"}, "unknown official status")
    _require(
        _numeric(har.get("score")) and evaluation.get("score") == har["score"],
        "official score mismatch",
    )
    _require(status.get("score") == har["score"], "status score mismatch")
    _require(
        (har["status"] == "success" and har["score"] == 1)
        or (har["status"] in {"failure", "error"} and har["score"] == 0),
        "official all-or-nothing status/score mismatch",
    )
    _require(
        bool(har.get("webarena_verified_evaluator_checksum"))
        and bool(har.get("webarena_verified_data_checksum")),
        "official checksums missing",
    )
    _require(
        evaluation.get("integrity_status") in {"passed", "evaluator_error"},
        "evaluation integrity failed",
    )
    evaluator_error = har["status"] == "error" or any(
        item.get("status") == "error" for item in har.get("evaluators_results", [])
    )
    if evaluator_error:
        _require(
            evaluation["integrity_status"] == "evaluator_error"
            and status["status"] == "evaluator_error",
            "evaluator error was not classified",
        )
        terminal_status = "evaluator_error"
        score: float | None = None
    else:
        _require(
            evaluation["integrity_status"] == "passed" and evaluation.get("publishable") is True,
            "score is not publishable",
        )
        _require(
            status["status"] in {"completed", "model_protocol_failed"},
            "scored case status mismatch",
        )
        if status["status"] == "model_protocol_failed":
            _require(
                bool(status.get("model_protocol_error")), "protocol failure lacks error evidence"
            )
        terminal_status = status["status"]
        score = har["score"]

    try:
        with tarfile.open(archive, "r:gz") as bundle:
            names = set(bundle.getnames())
            required = {
                f"{case_key}/{name}"
                for name in (
                    "network.har",
                    "trace.zip",
                    "evaluation.json",
                    "status.json",
                    "run_metadata.json",
                    "trajectory.jsonl",
                    "agent_response_raw.txt",
                    "baseline_receipt.json",
                    "startup_receipt.json",
                )
            }
            _require(required <= names, "archive lacks required evidence")
            stems = []
            for suffix in ("_request.json", "_response.json", "_metadata.json"):
                stems.append(
                    {
                        name.removesuffix(suffix)
                        for name in names
                        if name.startswith(f"{case_key}/requests/") and name.endswith(suffix)
                    }
                )
            _require(
                bool(stems[0]) and stems[0] == stems[1] == stems[2],
                "model request/response evidence is unmatched",
            )
            for request_name in sorted(name + "_request.json" for name in stems[0]):
                request = json.load(bundle.extractfile(request_name))
                expected_request = {
                    "model": "glm-5.3-flash",
                    "max_tokens": 32768,
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "enable_thinking": True,
                    "preserve_thinking": True,
                    "reasoning_effort": "max",
                }
                _require(
                    all(request.get(key) == value for key, value in expected_request.items()),
                    "model request settings mismatch",
                )
            input_receipt = evaluation.get("input_receipt") or {}
            for identity_key, filename in (
                ("network_har", "network.har"),
                ("playwright_trace", "trace.zip"),
                ("startup_receipt", "startup_receipt.json"),
                ("agent_response_raw", "agent_response_raw.txt"),
            ):
                member = bundle.extractfile(f"{case_key}/{filename}")
                _require(member is not None, f"archive member {filename} missing")
                _require(
                    _identity_matches(input_receipt.get(identity_key), member.read(), filename),
                    f"{filename} input identity mismatch",
                )
            for filename, value in (
                ("status.json", status),
                ("evaluation.json", evaluation),
                ("run_metadata.json", metadata),
                ("baseline_receipt.json", baseline),
                ("startup_receipt.json", startup),
            ):
                member = bundle.extractfile(f"{case_key}/{filename}")
                _require(
                    member is not None and json.load(member) == value,
                    f"archive/index {filename} mismatch",
                )
    except (OSError, tarfile.TarError, json.JSONDecodeError) as error:
        raise EvidenceInvalid(f"archive is unreadable: {error}") from error

    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    return {
        "run_id": run_id,
        "plan_sha256": plan_sha256,
        "worker": worker,
        "sandbox_id": sandbox_id,
        "task_id": task_id,
        "archive_sha256": archive_sha256,
        "runner_sha256": runner_sha256,
        "image_lock_sha256": image_lock_sha256,
        "terminal_status": terminal_status,
        "score": score,
        "official_evaluation_status": har["status"],
        "evaluator_checksum": har["webarena_verified_evaluator_checksum"],
        "data_checksum": har["webarena_verified_data_checksum"],
        "completed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--sandbox-id", required=True)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--image-lock-sha256", required=True)
    parser.add_argument("--runner-exit-code", required=True, type=int)
    args = parser.parse_args()
    try:
        receipt = make_receipt(
            args.index,
            args.archive,
            run_id=args.run_id,
            plan_sha256=args.plan_sha256,
            worker=args.worker,
            sandbox_id=args.sandbox_id,
            task_id=args.task_id,
            runner_sha256=args.runner_sha256,
            image_lock_sha256=args.image_lock_sha256,
            runner_exit_code=args.runner_exit_code,
        )
    except EvidenceInvalid as error:
        print(f"non-terminal case evidence: {error}", file=sys.stderr)
        return 2
    json.dump(receipt, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
