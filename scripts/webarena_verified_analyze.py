#!/usr/bin/env python3
"""Analyze published, hash-bound WebArena-Verified campaign evidence.

Usage: uv run python scripts/webarena_verified_analyze.py \
    --run-root /path/to/results/<run> --output-dir /path/to/new-analysis

Reads the campaign plan, official catalog, receipts, and one archive at a time.
Writes factual case records and QWEN3.8MAX-shaped summary tables, not root-cause
judgments or modified scores. Partial coverage and evaluator errors never
produce a complete-run accuracy. Credentials and live services are not needed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tarfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CAMPAIGN_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "devs/exps/eval/browsergym.webarena_verified/campaign/orchestrate.py"
)
_spec = importlib.util.spec_from_file_location("webarena_verified_campaign", CAMPAIGN_SCRIPT)
campaign = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(campaign)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, help="defaults to the campaign's catalog")
    parser.add_argument(
        "--output-dir", type=Path, help="new directory; existing files are not overwritten"
    )
    return parser.parse_args()


def read_case(root: Path, task: dict[str, Any], run_id: str, runner_sha: str) -> dict[str, Any]:
    """Read one published case and extract observable evaluation/rollout facts."""
    task_id = task["task_id"]
    case = f"task-{task_id:04d}"
    index = root / "index" / case
    receipt = json.loads((index / "completed.json").read_text())
    if receipt["run_id"] != run_id or receipt["runner_sha256"] != runner_sha:
        raise ValueError(f"{case}: receipt belongs to another run or runner")
    archive_path = root / "archives" / f"{case}.tar.gz"
    digest = hashlib.sha256()
    with archive_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != receipt["archive_sha256"]:
        raise ValueError(f"{case}: archive differs from the published receipt")

    response_evaluator = next(
        item for item in task["eval"] if item["evaluator"] == "AgentResponseEvaluator"
    )
    with tarfile.open(archive_path, "r:gz") as archive:
        # Read the immutable scored inputs, then summarize provider and GUI turns.
        status = json.load(archive.extractfile(f"{case}/status.json"))
        evaluation = json.load(archive.extractfile(f"{case}/evaluation.json"))
        startup = json.load(archive.extractfile(f"{case}/startup_receipt.json"))
        baseline = json.load(archive.extractfile(f"{case}/baseline_receipt.json"))
        trajectory = [
            json.loads(line)
            for line in archive.extractfile(f"{case}/trajectory.jsonl")
            if line.strip()
        ]
        turns = Counter(row["turn_kind"] for row in trajectory)
        usage = Counter()
        metadata_count = usage_count = structured_request_count = transport_errors = 0
        for member in archive:
            if not member.name.startswith(f"{case}/requests/step_"):
                continue
            if member.name.endswith("_metadata.json"):
                metadata = json.load(archive.extractfile(member))
                metadata_count += 1
                if metadata["usage"] is not None:
                    usage_count += 1
                    usage.update(
                        {
                            key: metadata["usage"][key]
                            for key in ("prompt_tokens", "completion_tokens")
                        }
                    )
            elif member.name.endswith("_request.json"):
                request = json.load(archive.extractfile(member))
                structured_request_count += "response_format" in request
            elif member.name.endswith("_error.json"):
                transport_errors += 1

    assertions = []
    evaluator_failures = []
    failed_evaluators = set()
    for evaluator in evaluation["har"]["evaluators_results"]:
        if evaluator["status"] == "success":
            continue
        failed_evaluators.add(evaluator["evaluator_name"])
        evaluator_failures.append(
            {
                key: evaluator.get(key)
                for key in (
                    "evaluator_name",
                    "status",
                    "error_msg",
                    "actual_normalized",
                    "expected",
                )
            }
        )
        for assertion in evaluator["assertions"] or []:
            if assertion["status"] != "success":
                assertions.append({"evaluator_name": evaluator["evaluator_name"], **assertion})

    # These are evidence signals, not automatic model-vs-infra root causes.
    if receipt["terminal_status"] == "evaluator_error":
        signal = "evaluator_error"
    elif receipt["score"] == 1:
        signal = "official_success"
    elif receipt["terminal_status"] == "model_protocol_failed":
        signal = "model_protocol_failed"
    elif status["truncated"]:
        signal = "step_budget_exhausted"
    elif failed_evaluators == {"AgentResponseEvaluator"}:
        signal = "response_requirement_failed"
    elif failed_evaluators == {"NetworkEventEvaluator"}:
        signal = "network_requirement_failed"
    elif failed_evaluators:
        signal = "multiple_evaluation_requirements_failed"
    else:
        signal = "official_failure_requires_review"

    return {
        "task_id": task_id,
        "sites": task["sites"],
        "task_type": response_evaluator["expected"]["task_type"],
        "intent": task["intent"],
        "intent_template_id": task["intent_template_id"],
        "worker": receipt["worker"],
        "sandbox_id": receipt["sandbox_id"],
        "archive_sha256": receipt["archive_sha256"],
        "terminal_status": receipt["terminal_status"],
        "score": receipt["score"],
        "official_evaluation_status": receipt["official_evaluation_status"],
        "failure_signal": signal,
        "failure_signal_is_root_cause": False,
        "model_protocol_error": status["model_protocol_error"],
        "truncated": status["truncated"],
        "terminal_action": status["terminal_action"],
        "startup_result": startup["result"],
        "startup_page_url": startup["selected_page_url"],
        "fresh_service_processes": baseline["fresh_service_processes"],
        "evaluation_integrity": evaluation["integrity_status"],
        "inputs_agree": evaluation["inputs_agree"],
        "trajectory_steps": len(trajectory),
        "turn_kinds": dict(turns),
        "executed_gui_actions": sum(
            action.startswith("pyautogui.")
            for row in trajectory
            for action in row["executed_actions"]
        ),
        "execution_errors": sum(bool(row["error"]) for row in trajectory),
        "projection_turns": sum("projected_content" in row for row in trajectory),
        "repair_requested_turns": [row["step"] for row in trajectory if "repair_requested" in row],
        "structured_repair_requests": structured_request_count,
        "response_repair_used": status["response_repair_used"],
        "structured_repair_used": status["structured_repair_used"],
        "action_repair_count": status["action_repair_count"],
        "model_requests": metadata_count,
        "usage_recorded_requests": usage_count,
        "usage_missing_requests": metadata_count - usage_count,
        "observed_token_usage": dict(usage),
        "model_transport_retry_errors": transport_errors,
        "failed_assertions": assertions,
        "failed_evaluators": evaluator_failures,
    }


def aggregate_metrics(planned: int, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep missing and unscored cases distinct in every report dimension."""
    scored = [case for case in cases if case["score"] is not None]
    passed = sum(case["score"] == 1 for case in scored)
    return {
        "planned": planned,
        "terminal": len(cases),
        "scored": len(scored),
        "passed": passed,
        "failed": len(scored) - passed,
        "unscored": len(cases) - len(scored),
        "pending": planned - len(cases),
        "official_accuracy": passed / planned if planned and len(scored) == planned else None,
        "scored_subset_rate": passed / len(scored) if scored else None,
    }


def analyze(root: Path, catalog_path: Path) -> dict[str, Any]:
    """Build complete/partial evidence tables without changing recorded scores."""
    config = json.loads((root / "EXPERIMENT_CONFIG.json").read_text())
    manifest = json.loads((root / "VERSION_MANIFEST.json").read_text())
    catalog_bytes = catalog_path.read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != config["dataset_sha256"]:
        raise ValueError("catalog differs from the configured official dataset")
    plan = campaign.Campaign.load_plan(root, catalog_path, config)
    catalog = {record["task_id"]: record for record in json.loads(catalog_bytes)}
    task_ids = sorted(
        int(line.split("\t", 1)[0])
        for worker in plan["workers"]
        for line in (root / "workers" / worker / "tasks.tsv").read_text().splitlines()
    )
    cases = []
    missing = []
    dimension_tasks: dict[str, list[int]] = defaultdict(list)
    for task_id in task_ids:
        task = catalog[task_id]
        task_type = next(
            item["expected"]["task_type"]
            for item in task["eval"]
            if item["evaluator"] == "AgentResponseEvaluator"
        )
        scope = "single_site" if len(task["sites"]) == 1 else "multi_site"
        for dimension in (
            "all",
            scope,
            f"{scope}/{task_type}",
            "sites/" + "+".join(sorted(task["sites"])),
        ):
            dimension_tasks[dimension].append(task_id)
        if not (root / "index" / f"task-{task_id:04d}" / "completed.json").is_file():
            missing.append(task_id)
            continue
        cases.append(read_case(root, task, config["run_id"], manifest["runner_sha256"]))
    by_id = {case["task_id"]: case for case in cases}
    return {
        "run_id": config["run_id"],
        "infra_version": manifest["infra_version"],
        "runner_sha256": manifest["runner_sha256"],
        "dataset_sha256": config["dataset_sha256"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection": plan["selection"],
        "complete": not missing,
        "summary": aggregate_metrics(len(task_ids), cases),
        "missing_task_ids": missing,
        "evaluator_error_task_ids": [
            case["task_id"] for case in cases if case["terminal_status"] == "evaluator_error"
        ],
        "dimensions": {
            name: aggregate_metrics(len(ids), [by_id[task] for task in ids if task in by_id])
            for name, ids in dimension_tasks.items()
        },
        "failure_signals": dict(Counter(case["failure_signal"] for case in cases)),
        "failure_signals_note": (
            "mechanical evidence signals only; "
            "trajectory and business semantics require case review"
        ),
        "cases": cases,
    }


def main() -> int:
    args = _parse_args()
    root = args.run_root.resolve()
    catalog = args.catalog or root.parents[1] / "dataset/webarena-verified.json"
    report = analyze(root, catalog)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / "report.json").open("x") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
        with (args.output_dir / "cases.jsonl").open("x") as output:
            for case in report["cases"]:
                output.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
