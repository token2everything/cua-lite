#!/usr/bin/env python3
"""Analyze hash-bound native WebArena-Verified campaign evidence.

Usage: uv run python scripts/webarena_verified_analyze.py \
    --run-root /path/to/results/<run> --output-dir /path/to/new-analysis \
    --issues-path /path/to/issues.jsonl

Config, plan, source, receipts and official pins bind the experiment. Read each
archive separately, without retaining model text/images. Include recorded failed
HTTP attempts in usage/latency; missing usage is unknown, not zero. Partial
coverage and evaluator errors never produce a complete-run accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
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

SITE_LABELS = {
    "reddit": "Reddit",
    "shopping_admin": "Shopping Admin",
    "shopping": "Shopping",
    "gitlab": "GitLab",
    "map": "Map",
    "wikipedia": "Wikipedia",
}
MULTI_SITE_ORDER = [
    "reddit+shopping",
    "gitlab+reddit",
    "gitlab+wikipedia",
    "map+shopping_admin",
    "map+wikipedia",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, help="defaults to the campaign's official catalog")
    parser.add_argument(
        "--issues-path", type=Path, help="recorded issue JSONL, not inferred causes"
    )
    parser.add_argument("--output-dir", type=Path, help="new directory; never overwrites a report")
    return parser.parse_args()


def latency_metrics(values: list[float]) -> dict[str, Any]:
    """Summarize observed seconds; unobserved latency is unknown, not zero."""
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("recorded API latency must be finite and nonnegative")
    ordered = sorted(values)
    return {
        "count": len(values),
        "sum_seconds": sum(values) if values else None,
        "min_seconds": min(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "mean_seconds": statistics.mean(values) if values else None,
        "p95_seconds": ordered[math.ceil(len(ordered) * 0.95) - 1] if values else None,
        "max_seconds": max(values) if values else None,
    }


def api_metrics(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate observed transport attempts without charging absent usage as zero."""
    attempts = [attempt for item in evidence for attempt in item["api_attempts"]]
    usage = Counter()
    for attempt in attempts:
        for key, value in (attempt["usage"] or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] += value
    return {
        "model_requests": sum(item["model_requests"] for item in evidence),
        "metadata_missing_requests": sum(item["metadata_missing_requests"] for item in evidence),
        "observed_http_attempts": len(attempts),
        "usage_recorded_attempts": sum(attempt["usage"] is not None for attempt in attempts),
        "usage_missing_attempts": sum(attempt["usage"] is None for attempt in attempts),
        "observed_token_usage": dict(usage),
        "token_usage_is_billing_total": False,
        "transport_error_attempts": sum(
            bool(attempt["error_type"])
            or (attempt["status_code"] is not None and not 200 <= attempt["status_code"] < 300)
            for attempt in attempts
        ),
        "http_attempt_latency": latency_metrics(
            [attempt["elapsed_seconds"] for attempt in attempts]
        ),
        "model_request_latency": latency_metrics(
            [value for item in evidence for value in item["request_elapsed_seconds"]]
        ),
        "model_request_wall_time": latency_metrics(
            [value for item in evidence for value in item["request_wall_seconds"]]
        ),
    }


def archive_evidence(
    path: Path, case: str, expected_sha256: str, config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Extract one immutable native attempt's API, message and GUI evidence."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"{case}: archive differs from its published hash")
    facts: dict[str, Any] = {
        "archive_sha256": expected_sha256,
        "model_requests": 0,
        "metadata_missing_requests": 0,
        "api_attempts": [],
        "request_elapsed_seconds": [],
        "request_wall_seconds": [],
        "trajectory_steps": 0,
        "assistant_messages": 0,
        "tool_calls": 0,
        "tool_result_messages": 0,
        "stored_screenshots": 0,
        "executed_gui_actions": 0,
        "execution_error_turns": 0,
        "tool_result_errors": 0,
        "reserve_blocked_turns": [],
        "projection_error_turns": [],
        "projection_error_responses": [],
        "native_trajectory_status": None,
        "environment_timings_seconds": {},
    }
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
        for filename, field in (
            ("status.json", "status"),
            ("evaluation.json", "evaluation"),
            ("run_metadata.json", "run_metadata"),
            ("startup_receipt.json", "startup"),
            ("baseline_receipt.json", "baseline"),
        ):
            if f"{case}/{filename}" in names:
                facts[field] = json.load(archive.extractfile(f"{case}/{filename}"))
        if "run_metadata" in facts:
            metadata = facts["run_metadata"]
            if any(
                metadata[key] != value
                for key, value in {
                    "harness": "native_chat_gateway",
                    "loop": "lite.agents.core.agent.base.AdapterBasedAgent.sample",
                    "dataset_sha256": config["benchmark"]["dataset_sha256"],
                    "script_sha256": manifest["runner_sha256"],
                    "osworld_sha": manifest["osworld_commit"],
                }.items()
            ):
                raise ValueError(f"{case}: archive is not this frozen native benchmark experiment")
            for owner, identity in metadata["dependencies"].items():
                module_file = Path(identity["module_file"])
                if owner == "cursor":
                    if Path("/opt/OSWorld") not in module_file.parents:
                        raise ValueError(f"{case}: cursor imported outside pinned source")
                elif Path("/root/cua-lite") not in module_file.parents or (
                    manifest["source_artifacts"][str(module_file.relative_to("/root/cua-lite"))]
                    != identity["sha256"]
                ):
                    raise ValueError(f"{case}: imported {owner} differs from frozen source")
        native_path = f"{case}/agent_trajectory.jsonl"
        if native_path in names:
            with archive.extractfile(native_path) as source:
                row = json.loads(next(source))
                if any(line.strip() for line in source):
                    raise ValueError(f"{case}: native trajectory must contain exactly one row")
            if row["metadata"]["model_id"] != config["model"]:
                raise ValueError(f"{case}: native trajectory model differs from config")
            facts["native_trajectory_status"] = row["metadata"]["status"]
            facts["assistant_messages"] = sum(m["role"] == "assistant" for m in row["messages"])
            facts["tool_calls"] = sum(len(m.get("tool_calls", [])) for m in row["messages"])
            facts["tool_result_messages"] = sum(m["role"] == "tool" for m in row["messages"])
            facts["stored_screenshots"] = sum(
                part["type"] == "image"
                for message in row["messages"]
                if isinstance(message.get("content"), list)
                for part in message["content"]
            )
            del row
        timings = Counter()
        if f"{case}/trajectory.jsonl" in names:
            for line in archive.extractfile(f"{case}/trajectory.jsonl"):
                if not line.strip():
                    continue
                turn = json.loads(line)
                if turn["step"] != facts["trajectory_steps"]:
                    raise ValueError(f"{case}: environment turns are not contiguous")
                facts["trajectory_steps"] += 1
                if turn.get("prediction_failed") is True:
                    response_path = f"requests/{turn['step'] + 1:05d}.response.json"
                    if (
                        turn["raw_response"] != response_path
                        or f"{case}/{response_path}" not in names
                        or turn["actions"] != []
                        or turn["executed_actions"] != []
                        or turn["feedback"] != []
                        or turn["provider_content"] is not None
                        or turn["timings"] != {}
                        or turn["reserve_blocked"]
                        or turn["failure_phase"] != "provider_projection"
                        or not turn["terminated"]
                        or turn["truncated"]
                    ):
                        raise ValueError(f"{case}: invalid provider-projection failure decision")
                    facts["projection_error_turns"].append(turn["step"])
                    facts["projection_error_responses"].append(response_path)
                facts["executed_gui_actions"] += sum(
                    action["call"] not in {"response", "noop"}
                    for action in turn["executed_actions"]
                )
                errors = sum(bool(result["error"]) for result in turn["feedback"])
                facts["execution_error_turns"] += bool(errors)
                facts["tool_result_errors"] += errors
                if turn["reserve_blocked"]:
                    facts["reserve_blocked_turns"].append(turn["step"])
                timings.update(turn["timings"] or {})
            facts["environment_timings_seconds"] = dict(timings)
        request_names = sorted(
            name
            for name in names
            if re.fullmatch(rf"{case}/requests/[0-9]{{5}}\.request\.json", name)
        )
        expected_request = {
            **config["agent"]["request_params"],
            "model": config["model"],
            "reasoning_effort": config["reasoning_effort"],
        }
        for name in request_names:
            request = json.load(archive.extractfile(name))
            if any(request.get(key) != value for key, value in expected_request.items()) or any(
                key in request for key in config["agent"]["omitted_request_parameters"]
            ):
                raise ValueError(f"{case}: recorded request settings differ from config")
            del request
            facts["model_requests"] += 1
            stem = name.removesuffix(".request.json")
            metadata_path = stem + ".metadata.json"
            if metadata_path not in names:
                facts["metadata_missing_requests"] += 1
                continue
            metadata = json.load(archive.extractfile(metadata_path))
            facts["request_elapsed_seconds"].append(metadata["elapsed_seconds"])
            facts["request_wall_seconds"].append(metadata["wall_seconds"])
            for index, attempt in enumerate(metadata["attempts"], start=1):
                if attempt["attempt"] != index:
                    raise ValueError(f"{case}: HTTP attempts are not contiguous")
                response_path = (
                    stem + ".response.json"
                    if attempt["status_code"] is not None and 200 <= attempt["status_code"] < 300
                    else stem + f".attempt-{index:02d}.response.json"
                )
                usage = None
                if response_path in names:
                    response = json.load(archive.extractfile(response_path))
                    usage = response.get("usage") if isinstance(response, dict) else None
                    del response
                facts["api_attempts"].append(
                    {
                        "request_id": Path(stem).name,
                        **attempt,
                        "usage": usage,
                    }
                )
        if facts["projection_error_turns"]:
            last_response = f"requests/{facts['model_requests']:05d}.response.json"
            if (
                facts["projection_error_turns"] != [facts["trajectory_steps"] - 1]
                or facts["projection_error_responses"] != [last_response]
                or not facts["api_attempts"]
                or facts["api_attempts"][-1]["request_id"] != f"{facts['model_requests']:05d}"
                or facts["api_attempts"][-1]["status_code"] is None
                or not 200 <= facts["api_attempts"][-1]["status_code"] < 300
            ):
                raise ValueError(
                    f"{case}: projection failure lacks a terminal successful HTTP reply"
                )
        if "evaluation" in facts:
            # Hash scored HAR/trace input streams without loading them in memory.
            for key, filename in (
                ("network_har", "network.har"),
                ("playwright_trace", "trace.zip"),
                ("startup_receipt", "startup_receipt.json"),
                ("agent_response_raw", "agent_response_raw.txt"),
            ):
                identity = facts["evaluation"]["input_receipt"][key]
                digest, size = hashlib.sha256(), 0
                with archive.extractfile(f"{case}/{filename}") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                if identity["sha256"] != digest.hexdigest() or identity["size"] != size:
                    raise ValueError(f"{case}: scored input {filename} differs from receipt")
    facts["api"] = api_metrics([facts])
    return facts


def read_case(
    root: Path, task: dict[str, Any], config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Read one published official outcome without inferring model-vs-infra causes."""
    task_id = task["task_id"]
    case = f"task-{task_id:04d}"
    index = root / "index" / case
    receipt = json.loads((index / "completed.json").read_text())
    expected = {
        "run_id": config["run_id"],
        "runner_sha256": manifest["runner_sha256"],
        "source_sha256": manifest["source_sha256"],
        "task_id": task_id,
        "plan_sha256": campaign._sha(root / "PLAN_MANIFEST.sha256"),
        "image_lock_sha256": campaign._sha(root / "image-lock.env"),
        "data_checksum": config["benchmark"]["dataset_sha256"],
        "evaluator_checksum": config["benchmark"]["evaluator_sha256"],
    }
    if any(receipt[key] != value for key, value in expected.items()):
        raise ValueError(f"{case}: receipt belongs to another experiment or official version")
    evidence = archive_evidence(
        root / "archives" / f"{case}.tar.gz", case, receipt["archive_sha256"], config, manifest
    )
    for field, filename in (
        ("status", "status.json"),
        ("evaluation", "evaluation.json"),
        ("run_metadata", "run_metadata.json"),
        ("startup", "startup_receipt.json"),
        ("baseline", "baseline_receipt.json"),
    ):
        if evidence[field] != json.loads((index / filename).read_text()):
            raise ValueError(f"{case}: archive/index {filename} mismatch")
    status, evaluation = evidence["status"], evidence["evaluation"]
    if evidence["run_metadata"]["script_sha256"] != manifest["runner_sha256"]:
        raise ValueError(f"{case}: archive runner differs from manifest")
    projection_failure = status.get("model_protocol_failure_phase") == "provider_projection"
    if projection_failure:
        if (
            status["model_protocol_error_type"] != "ChatGatewayProtocolError"
            or not status["model_protocol_error"]
            or receipt["terminal_status"] not in {"model_protocol_failed", "evaluator_error"}
            or evidence["native_trajectory_status"] != "failed"
            or len(evidence["projection_error_turns"]) != 1
        ):
            raise ValueError(f"{case}: typed provider-projection failure contract is incomplete")
    elif evidence["native_trajectory_status"] != "completed" or evidence["projection_error_turns"]:
        raise ValueError(f"{case}: published native episode lacks its canonical completion")
    if (
        evidence["assistant_messages"] + len(evidence["projection_error_turns"])
        != evidence["trajectory_steps"]
        or evidence["trajectory_steps"] != evidence["model_requests"]
        or evidence["metadata_missing_requests"]
    ):
        raise ValueError(f"{case}: published native episode evidence is incomplete")
    for result in (evaluation["har"], evaluation["playwright_trace"]):
        if any(
            result[key] != value
            for key, value in {
                "task_id": task_id,
                "webarena_verified_version": config["benchmark"]["version"],
                "webarena_verified_data_checksum": expected["data_checksum"],
                "webarena_verified_evaluator_checksum": expected["evaluator_checksum"],
                "status": receipt["official_evaluation_status"],
            }.items()
        ):
            raise ValueError(f"{case}: official result differs from pinned experiment")
    if (
        not evaluation["inputs_agree"]
        or not evaluation["score_inputs_agree"]
        or (
            evaluation["har"]["score"] != evaluation["playwright_trace"]["score"]
            or evaluation["score"] != evaluation["har"]["score"]
            or status["status"] != receipt["terminal_status"]
            or status["score"] != evaluation["har"]["score"]
        )
    ):
        raise ValueError(f"{case}: official contracts/status/score disagree")
    if receipt["terminal_status"] == "evaluator_error":
        if (
            receipt["score"] is not None
            or evaluation["har"]["status"] != "error"
            or evaluation["har"]["score"] != 0
            or evaluation["integrity_status"] != "evaluator_error"
        ):
            raise ValueError(f"{case}: evaluator error must remain unscored")
    elif receipt["terminal_status"] not in {"completed", "model_protocol_failed"} or (
        type(receipt["score"]) not in {int, float}
        or receipt["score"] not in {0, 1}
        or receipt["score"] != evaluation["har"]["score"]
        or evaluation["integrity_status"] != "passed"
        or not evaluation["publishable"]
        or receipt["official_evaluation_status"]
        != ("success" if receipt["score"] == 1 else "failure")
    ):
        raise ValueError(f"{case}: invalid publishable official outcome")
    failures, assertions = [], []
    for evaluator in evaluation["har"]["evaluators_results"]:
        if evaluator["status"] == "success":
            continue
        failures.append(
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
        assertions.extend(
            {"evaluator_name": evaluator["evaluator_name"], **assertion}
            for assertion in evaluator["assertions"] or []
            if assertion["status"] != "success"
        )
    names = {item["evaluator_name"] for item in failures}
    if receipt["terminal_status"] == "evaluator_error":
        signal = "evaluator_error"
    elif receipt["score"] == 1:
        signal = "official_success"
    elif receipt["terminal_status"] == "model_protocol_failed":
        signal = "model_protocol_failed"
    elif status["truncated"]:
        signal = "step_budget_exhausted"
    elif names == {"AgentResponseEvaluator"}:
        signal = "response_requirement_failed"
    elif names == {"NetworkEventEvaluator"}:
        signal = "network_requirement_failed"
    elif names:
        signal = "multiple_evaluation_requirements_failed"
    else:
        signal = "official_failure_requires_review"
    task_type = next(
        item["expected"]["task_type"]
        for item in task["eval"]
        if item["evaluator"] == "AgentResponseEvaluator"
    )
    return {
        "task_id": task_id,
        "sites": task["sites"],
        "task_type": task_type,
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
        "truncated": status["truncated"],
        "terminal_action": status["terminal_action"],
        "model_protocol_error": status["model_protocol_error"],
        "model_protocol_failure_phase": status.get("model_protocol_failure_phase"),
        "model_protocol_error_type": status.get("model_protocol_error_type"),
        "response_repair_used": status["response_repair_used"],
        "startup_result": evidence["startup"]["result"],
        "fresh_service_processes": evidence["baseline"]["fresh_service_processes"],
        "evaluation_integrity": evaluation["integrity_status"],
        "failed_assertions": assertions,
        "failed_evaluators": failures,
        **{
            key: evidence[key]
            for key in (
                "trajectory_steps",
                "assistant_messages",
                "tool_calls",
                "tool_result_messages",
                "stored_screenshots",
                "executed_gui_actions",
                "execution_error_turns",
                "tool_result_errors",
                "reserve_blocked_turns",
                "projection_error_turns",
                "projection_error_responses",
                "native_trajectory_status",
                "environment_timings_seconds",
                "api",
            )
        },
    }


def aggregate_metrics(planned: int, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep pending, unscored and infra outcomes distinct from official failure."""
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


def analyze(root: Path, catalog_path: Path, issues_path: Path | None = None) -> dict[str, Any]:
    """Build reference-shaped tables and all-attempt observed usage from frozen evidence."""
    root = root.resolve()
    config = json.loads((root / "EXPERIMENT_CONFIG.json").read_text())
    manifest = json.loads((root / "VERSION_MANIFEST.json").read_text())
    if campaign._sha(root / "cua-lite-source.tar.gz") != manifest["source_sha256"]:
        raise ValueError("source bundle differs from version manifest")
    with tarfile.open(root / "cua-lite-source.tar.gz", "r:gz") as source:
        if (
            hashlib.sha256(source.extractfile("./" + config["entrypoint"]).read()).hexdigest()
            != (manifest["runner_sha256"])
        ):
            raise ValueError("source entrypoint differs from runner manifest")
        for relative, expected_sha in manifest["source_artifacts"].items():
            if (
                hashlib.sha256(source.extractfile("./" + relative).read()).hexdigest()
                != expected_sha
            ):
                raise ValueError(f"frozen source artifact differs from manifest: {relative}")
    catalog_bytes = catalog_path.read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != config["benchmark"]["dataset_sha256"]:
        raise ValueError("catalog differs from the configured official dataset")
    plan = campaign.Campaign.load_plan(root, catalog_path, config)
    catalog = {record["task_id"]: record for record in json.loads(catalog_bytes)}
    task_ids = sorted(int(task) for task in plan["tasks"])
    cases, missing, attempts = [], [], []
    dimension_tasks: dict[str, list[int]] = defaultdict(list)
    plan_sha, image_sha = (
        campaign._sha(root / "PLAN_MANIFEST.sha256"),
        campaign._sha(root / "image-lock.env"),
    )
    for task_id in task_ids:
        task, case_key = catalog[task_id], f"task-{task_id:04d}"
        task_type = next(
            item["expected"]["task_type"]
            for item in task["eval"]
            if item["evaluator"] == "AgentResponseEvaluator"
        )
        scope = "single_site" if len(task["sites"]) == 1 else "multi_site"
        for dimension in (
            "all",
            f"all/{task_type}",
            scope,
            f"{scope}/{task_type}",
            "sites/" + "+".join(sorted(task["sites"])),
        ):
            dimension_tasks[dimension].append(task_id)
        if (root / "index" / case_key / "completed.json").is_file():
            case = read_case(root, task, config, manifest)
            if case["worker"] not in plan["workers"] or not set(task["sites"]) <= set(
                plan["workers"][case["worker"]]["capabilities"]
            ):
                raise ValueError(f"{case_key}: receipt worker cannot execute this site's task")
            cases.append(case)
        else:
            missing.append(task_id)
        for ledger_path in sorted((root / "attempts" / case_key).glob("*.json")):
            ledger = json.loads(ledger_path.read_text())
            if any(
                ledger[key] != value
                for key, value in {
                    "run_id": config["run_id"],
                    "plan_sha256": plan_sha,
                    "task_id": task_id,
                    "runner_sha256": manifest["runner_sha256"],
                    "image_lock_sha256": image_sha,
                    "source_sha256": manifest["source_sha256"],
                }.items()
            ):
                raise ValueError(f"{case_key}: attempt ledger belongs to another experiment")
            if ledger["worker"] not in plan["workers"] or not set(task["sites"]) <= set(
                plan["workers"][ledger["worker"]]["capabilities"]
            ):
                raise ValueError(f"{case_key}: attempt worker cannot execute this site's task")
            if ledger["local_archive_path"] is None:
                if (
                    ledger["model_started"]
                    or ledger["archive_sha256"] is not None
                    or (ledger["archive_path"] is not None)
                ):
                    raise ValueError(f"{case_key}: model-started attempt lacks retained archive")
                attempts.append(
                    {
                        **ledger,
                        "ledger_path": str(ledger_path.relative_to(root)),
                        "model_requests": 0,
                        "metadata_missing_requests": 0,
                        "api_attempts": [],
                        "request_elapsed_seconds": [],
                        "request_wall_seconds": [],
                        "api": api_metrics([]),
                    }
                )
                continue
            path = Path(ledger["local_archive_path"])
            path = path.resolve() if path.is_absolute() else (root / path).resolve()
            if root not in path.parents:
                raise ValueError(f"{case_key}: local attempt archive is outside result root")
            evidence = archive_evidence(path, case_key, ledger["archive_sha256"], config, manifest)
            if (
                "run_metadata" in evidence
                and evidence["run_metadata"]["script_sha256"] != (manifest["runner_sha256"])
            ):
                raise ValueError(f"{case_key}: attempt runner differs from manifest")
            attempts.append(
                {
                    **ledger,
                    "ledger_path": str(ledger_path.relative_to(root)),
                    **{
                        key: evidence[key]
                        for key in (
                            "model_requests",
                            "metadata_missing_requests",
                            "api_attempts",
                            "request_elapsed_seconds",
                            "request_wall_seconds",
                            "api",
                        )
                    },
                }
            )
    by_id = {case["task_id"]: case for case in cases}
    dimensions = {
        name: aggregate_metrics(len(ids), [by_id[task] for task in ids if task in by_id])
        for name, ids in dimension_tasks.items()
    }
    dimensions.setdefault("all", aggregate_metrics(0, []))
    for scope in ("single_site", "multi_site"):
        for suffix in ("", "/retrieve", "/mutate", "/navigate"):
            dimensions.setdefault(scope + suffix, aggregate_metrics(0, []))
    accounted = {(item["task_id"], item["archive_sha256"]) for item in attempts}
    missing_attempt_ledgers = [
        case["task_id"]
        for case in cases
        if (case["task_id"], case["archive_sha256"]) not in accounted
    ]
    infra_attempts = [
        attempt
        for attempt in attempts
        if attempt["terminal_status"]
        not in {
            "completed",
            "model_protocol_failed",
            "evaluator_error",
        }
    ]
    issues = []
    if issues_path is not None:
        with issues_path.open() as source:
            issues = [json.loads(line) for line in source if line.strip()]
    return {
        "run_id": config["run_id"],
        "model": config["model"],
        "experiment_version": config["experiment_version"],
        "runner_sha256": manifest["runner_sha256"],
        "source_sha256": manifest["source_sha256"],
        "dataset_sha256": config["benchmark"]["dataset_sha256"],
        "evaluator_sha256": config["benchmark"]["evaluator_sha256"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection": plan["selection"],
        "complete": not missing,
        "fully_scored": len(cases) == len(task_ids)
        and all(case["score"] is not None for case in cases),
        "summary": aggregate_metrics(len(task_ids), cases),
        "dimensions": dimensions,
        "missing_task_ids": missing,
        "evaluator_error_task_ids": [
            case["task_id"] for case in cases if case["terminal_status"] == "evaluator_error"
        ],
        "infra_attempt_count": len(infra_attempts),
        "infra_affected_task_ids": sorted({attempt["task_id"] for attempt in infra_attempts}),
        "failure_signals": dict(Counter(case["failure_signal"] for case in cases)),
        "failure_signals_note": "Observable signals only; no automatic model-vs-infra attribution.",
        "all_attempts_api": {
            **api_metrics(attempts),
            "archived_case_attempts": len(attempts),
            "terminal_archives_accounted_for": not missing_attempt_ledgers,
            "terminal_archives_without_attempt_ledger": missing_attempt_ledgers,
            "observed_attempt_usage_complete": bool(attempts)
            and all(
                not attempt["metadata_missing_requests"]
                and all(call["usage"] is not None for call in attempt["api_attempts"])
                for attempt in attempts
            ),
            "coverage_note": (
                "Only synchronized hash-bound attempts are observed. Missing usage, unsynchronized "
                "or in-flight attempts are unknown; recorded tokens are not a billing total."
            ),
        },
        "issues": issues,
        "cases": cases,
        "attempts": attempts,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the three reference tables, separate accounting and issues last."""
    lines = [
        f"# {report['model']} WebArena-Verified 全量结果分析",
        "",
        f"Run ID：`{report['run_id']}`",
        "",
        "以下分母为实验计划范围；仅当该行全部题目都有有效评分时显示成功率。"
        "尚未完成、未评分和基础设施重试不混入模型失败。",
        "",
        "## 全量分析",
        "",
        "| 范围 | 类型 | 通过/总数 | 成功率 |",
        "| --- | --- | --- | --- |",
    ]
    rows = [
        ("全量", "全部", "all"),
        ("单站", "全部", "single_site"),
        ("单站", "retrieve", "single_site/retrieve"),
        ("单站", "mutate", "single_site/mutate"),
        ("单站", "navigate", "single_site/navigate"),
        ("多站", "全部", "multi_site"),
        ("多站", "mutate", "multi_site/mutate"),
        ("多站", "navigate", "multi_site/navigate"),
        ("多站", "retrieve", "multi_site/retrieve"),
    ]
    for scope, task_type, dimension in rows:
        metric = report["dimensions"][dimension]
        rate = metric["official_accuracy"]
        display = (
            f"{rate:.2%}"
            if rate is not None
            else "—"
            if not metric["planned"]
            else "未形成完整评分"
        )
        lines.append(
            f"| {scope} | {task_type} | {metric['passed']}/{metric['planned']} | {display} |"
        )
    for title, groups in (
        ("单站按站点分类", list(SITE_LABELS)),
        ("多站按站点分析", MULTI_SITE_ORDER),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| 站点组合 | 通过/总数 | 成功率 |"
                if title == "多站按站点分析"
                else "| 站点 | 通过/总数 | 成功率 |",
                "| --- | --- | --- |",
            ]
        )
        extras = sorted(
            name.removeprefix("sites/")
            for name in report["dimensions"]
            if name.startswith("sites/")
            and name.removeprefix("sites/") not in groups
            and (("+" in name) == (title == "多站按站点分析"))
        )
        for sites in [*groups, *extras]:
            metric = report["dimensions"].get("sites/" + sites, aggregate_metrics(0, []))
            label = " + ".join(SITE_LABELS[site] for site in sites.split("+"))
            rate = metric["official_accuracy"]
            display = (
                f"{rate:.2%}"
                if rate is not None
                else "—"
                if not metric["planned"]
                else "未形成完整评分"
            )
            lines.append(f"| {label} | {metric['passed']}/{metric['planned']} | {display} |")
    summary, api = report["summary"], report["all_attempts_api"]
    lines.extend(
        [
            "",
            "## 完成与异常状态",
            "",
            "| 计划 | 已归档终态 | 已评分 | 通过 | 失败 | 未评分 | 未完成 | Infra 失败 attempts |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            f"| {summary['planned']} | {summary['terminal']} | {summary['scored']} | "
            f"{summary['passed']} | {summary['failed']} | {summary['unscored']} | "
            f"{summary['pending']} | {report['infra_attempt_count']} |",
            "",
            f"未完成 task IDs：{report['missing_task_ids']}。",
            "",
            f"评分异常 task IDs：{report['evaluator_error_task_ids']}。",
            "",
            "## 模型调用与时延",
            "",
            f"已同步 case attempts：{api['archived_case_attempts']}；"
            f"模型请求：{api['model_requests']}；HTTP attempts：{api['observed_http_attempts']}。",
            "",
            f"API 回包耗时（仅 HTTP 请求计时，不含截图、日志序列化和重试等待）："
            f"{json.dumps(api['http_attempt_latency'], ensure_ascii=False)}。",
            "",
            f"有 usage 的 HTTP attempts：{api['usage_recorded_attempts']}；"
            f"缺失 usage：{api['usage_missing_attempts']}；"
            f"观测 token 用量：{json.dumps(api['observed_token_usage'], ensure_ascii=False)}。",
            "",
            "缺失 usage 不记作 0；以上不是账单总额，也不能用于凭空推断模型或 Infra 根因。",
            "",
            "## 本次实验遇到的问题",
            "",
            "证据统计标签仅指出观察到的现象，不等于已确认的原因。",
            "",
        ]
    )
    if report["issues"]:
        for issue in report["issues"]:
            lines.append("- " + json.dumps(issue, ensure_ascii=False))
    else:
        lines.append("未提供过程问题记录；这不代表实验没有问题。")
    lines.extend(
        [
            "",
            f"评分/轨迹证据信号：{json.dumps(report['failure_signals'], ensure_ascii=False)}。",
            "",
            f"未纳入 all-attempts 账本的终态题目："
            f"{api['terminal_archives_without_attempt_ledger']}。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    root = args.run_root.resolve()
    catalog = args.catalog or root.parents[1] / "dataset/webarena-verified.json"
    report = analyze(root, catalog, args.issues_path)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / "report.json").open("x") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
        for filename, records in (
            ("cases.jsonl", report["cases"]),
            ("attempts.jsonl", report["attempts"]),
        ):
            with (args.output_dir / filename).open("x") as output:
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (args.output_dir / "report.md").open("x") as output:
            output.write(render_markdown(report))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"cases", "attempts"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
