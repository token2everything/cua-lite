from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/webarena_verified_analyze.py"
SPEC = importlib.util.spec_from_file_location("webarena_verified_analyze_test", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

RUN_ID = "wav-glm53flash-infra-v10-full50-20261005"
RUNNER_SHA = "a" * 64


def write_case(
    root: Path,
    task_id: int = 0,
    *,
    terminal_status: str = "completed",
    score: float | None = 1.0,
    failed_evaluators: tuple[str, ...] = (),
    truncated: bool = False,
    missing_usage: bool = False,
) -> dict:
    case = f"task-{task_id:04d}"
    task = {
        "task_id": task_id,
        "sites": ["gitlab"],
        "intent": "Retrieve the project's name",
        "intent_template_id": 1,
        "eval": [{"evaluator": "AgentResponseEvaluator", "expected": {"task_type": "retrieve"}}],
    }
    status = {
        "truncated": truncated,
        "terminal_action": "FINAL_JSON",
        "model_protocol_error": "bad call" if terminal_status == "model_protocol_failed" else None,
        "response_repair_used": False,
        "structured_repair_used": True,
        "action_repair_count": 1,
    }
    evaluators = [
        {
            "evaluator_name": name,
            "status": "error" if terminal_status == "evaluator_error" else "failure",
            "actual_normalized": {"value": "actual"},
            "expected": {"value": "expected"},
            "error_msg": "error" if terminal_status == "evaluator_error" else None,
            "assertions": [
                {
                    "status": "failure",
                    "assertion_name": "value_mismatch",
                    "assertion_msgs": ["different values"],
                }
            ],
        }
        for name in failed_evaluators
    ]
    files = {
        "status.json": status,
        "evaluation.json": {
            "har": {"evaluators_results": evaluators},
            "integrity_status": "evaluator_error" if score is None else "passed",
            "inputs_agree": True,
        },
        "startup_receipt.json": {"result": "passed", "selected_page_url": "http://localhost:8023"},
        "baseline_receipt.json": {"fresh_service_processes": True},
        "requests/step_000_metadata.json": {
            "usage": None if missing_usage else {"prompt_tokens": 10, "completion_tokens": 5}
        },
        "requests/step_000_request.json": {"response_format": {"type": "json_schema"}},
        "requests/step_000_attempt_01_error.json": {"type": "APITimeoutError"},
    }
    trajectory = {
        "step": 0,
        "turn_kind": "final",
        "executed_actions": ["FINAL_JSON"],
        "error": None,
        "repair_requested": True,
    }
    archive_path = root / "archives" / f"{case}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, value in files.items():
            data = json.dumps(value).encode()
            member = tarfile.TarInfo(f"{case}/{name}")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        data = (json.dumps(trajectory) + "\n").encode()
        member = tarfile.TarInfo(f"{case}/trajectory.jsonl")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    index = root / "index" / case
    index.mkdir(parents=True)
    receipt = {
        "run_id": RUN_ID,
        "runner_sha256": RUNNER_SHA,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "worker": "01",
        "sandbox_id": "owned",
        "terminal_status": terminal_status,
        "score": score,
        "official_evaluation_status": "error"
        if score is None
        else "success"
        if score == 1
        else "failure",
    }
    (index / "completed.json").write_text(json.dumps(receipt))
    return task


@pytest.mark.parametrize(
    "planned,scores,expected_accuracy,expected_pending,expected_unscored",
    [
        (3, [1.0, 0.0], None, 1, 0),
        (3, [1.0, 0.0, None], None, 0, 1),
        (2, [0.0, 0.0], 0.0, 0, 0),
        (2, [1.0, 0.0], 0.5, 0, 0),
        (0, [], None, 0, 0),
    ],
)
def test_aggregation_never_converts_missing_or_unscored_into_failure(
    planned, scores, expected_accuracy, expected_pending, expected_unscored
):
    result = analysis.aggregate_metrics(planned, [{"score": score} for score in scores])
    assert result["official_accuracy"] == expected_accuracy
    assert result["pending"] == expected_pending
    assert result["unscored"] == expected_unscored
    assert result["failed"] == scores.count(0.0)


def test_case_reads_actual_usage_assertions_and_repair_without_root_cause_inference(tmp_path):
    task = write_case(tmp_path, score=0.0, failed_evaluators=("AgentResponseEvaluator",))
    case = analysis.read_case(tmp_path, task, RUN_ID, RUNNER_SHA)
    assert case["failure_signal"] == "response_requirement_failed"
    assert case["failure_signal_is_root_cause"] is False
    assert case["failed_assertions"][0]["assertion_name"] == "value_mismatch"
    assert case["failed_evaluators"][0]["actual_normalized"] == {"value": "actual"}
    assert case["observed_token_usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
    assert case["structured_repair_requests"] == case["model_transport_retry_errors"] == 1
    assert case["repair_requested_turns"] == [0]
    assert case["executed_gui_actions"] == case["execution_errors"] == 0


@pytest.mark.parametrize(
    "terminal_status,score,truncated,evaluators,signal",
    [
        ("completed", 1.0, False, (), "official_success"),
        ("evaluator_error", None, False, ("NetworkEventEvaluator",), "evaluator_error"),
        ("model_protocol_failed", 0.0, True, (), "model_protocol_failed"),
        ("completed", 0.0, True, (), "step_budget_exhausted"),
        ("completed", 0.0, False, ("NetworkEventEvaluator",), "network_requirement_failed"),
        (
            "completed",
            0.0,
            False,
            ("NetworkEventEvaluator", "AgentResponseEvaluator"),
            "multiple_evaluation_requirements_failed",
        ),
        ("completed", 0.0, False, (), "official_failure_requires_review"),
    ],
)
def test_evidence_signals_preserve_distinct_outcomes(
    tmp_path, terminal_status, score, truncated, evaluators, signal
):
    task = write_case(
        tmp_path,
        terminal_status=terminal_status,
        score=score,
        truncated=truncated,
        failed_evaluators=evaluators,
    )
    case = analysis.read_case(tmp_path, task, RUN_ID, RUNNER_SHA)
    assert case["failure_signal"] == signal
    assert case["score"] == score


def test_absent_usage_remains_unknown_instead_of_zero_billed_tokens(tmp_path):
    task = write_case(tmp_path, missing_usage=True)
    case = analysis.read_case(tmp_path, task, RUN_ID, RUNNER_SHA)
    assert case["usage_missing_requests"] == 1
    assert case["usage_recorded_requests"] == 0
    assert case["observed_token_usage"] == {}


@pytest.mark.parametrize("change", ["run", "runner", "archive"])
def test_mixed_run_or_changed_archive_cannot_enter_statistics(tmp_path, change):
    task = write_case(tmp_path)
    if change == "archive":
        with (tmp_path / "archives/task-0000.tar.gz").open("ab") as archive:
            archive.write(b"changed")
        run_id, runner_sha = RUN_ID, RUNNER_SHA
    else:
        run_id = "wrong" if change == "run" else RUN_ID
        runner_sha = "wrong" if change == "runner" else RUNNER_SHA
    with pytest.raises(ValueError):
        analysis.read_case(tmp_path, task, run_id, runner_sha)


def write_plan_manifest(root):
    paths = [root / "worker-plan.json", *sorted((root / "workers").glob("*/*"))]
    (root / "PLAN_MANIFEST.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
            for path in paths
        )
    )


@pytest.fixture
def campaign_evidence(tmp_path):
    task0 = write_case(tmp_path)
    task1 = write_case(
        tmp_path,
        1,
        terminal_status="evaluator_error",
        score=None,
        failed_evaluators=("NetworkEventEvaluator",),
    )
    task1["sites"] = ["gitlab", "reddit"]
    task1["eval"][0]["expected"]["task_type"] = "mutate"
    task2 = {**task0, "task_id": 2, "sites": ["reddit"]}
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([task0, task1, task2]))
    (tmp_path / "EXPERIMENT_CONFIG.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "task_count": 3,
                "task_ids": [0, 1, 2],
                "dataset_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
            }
        )
    )
    (tmp_path / "VERSION_MANIFEST.json").write_text(
        json.dumps({"infra_version": "test-version", "runner_sha256": RUNNER_SHA})
    )
    (tmp_path / "worker-plan.json").write_text(
        json.dumps(
            {
                "selection": "canary",
                "task_count": 3,
                "workers": {
                    "01": {"capabilities": ["gitlab", "reddit"], "startup_task": 0, "tasks": 3}
                },
            }
        )
    )
    workers = tmp_path / "workers/01"
    workers.mkdir(parents=True)
    (workers / "tasks.tsv").write_text("0\tgitlab\n1\tgitlab,reddit\n2\treddit\n")
    (workers / "capabilities.txt").write_text("gitlab\nreddit\n")
    write_plan_manifest(tmp_path)
    return tmp_path, catalog


def test_partial_dimensions_keep_planned_denominators_and_missing_case_ids(campaign_evidence):
    root, catalog = campaign_evidence
    report = analysis.analyze(root, catalog)
    assert report["complete"] is False
    assert report["missing_task_ids"] == [2]
    assert report["evaluator_error_task_ids"] == [1]
    assert report["summary"]["official_accuracy"] is None
    assert report["dimensions"]["single_site/retrieve"]["planned"] == 2
    assert report["dimensions"]["single_site/retrieve"]["official_accuracy"] is None
    assert report["dimensions"]["multi_site/mutate"]["unscored"] == 1
    assert report["dimensions"]["sites/gitlab+reddit"]["unscored"] == 1


@pytest.mark.parametrize("rows", ["0\tgitlab\n1\tgitlab,reddit\n", "0\tgitlab\n" * 3])
def test_modified_task_table_cannot_change_statistics(campaign_evidence, rows):
    root, catalog = campaign_evidence
    (root / "workers/01/tasks.tsv").write_text(rows)
    with pytest.raises(ValueError, match="invalid plan manifest entry"):
        analysis.analyze(root, catalog)


def test_rehashed_truncated_plan_still_cannot_shrink_configured_denominator(campaign_evidence):
    root, catalog = campaign_evidence
    (root / "workers/01/tasks.tsv").write_text("0\tgitlab\n1\tgitlab,reddit\n")
    plan_path = root / "worker-plan.json"
    plan = json.loads(plan_path.read_text())
    plan["task_count"] = plan["workers"]["01"]["tasks"] = 2
    plan_path.write_text(json.dumps(plan))
    write_plan_manifest(root)
    with pytest.raises(ValueError, match="configured task count"):
        analysis.analyze(root, catalog)


def test_rehashed_duplicate_assignments_cannot_inflate_statistics(campaign_evidence):
    root, catalog = campaign_evidence
    (root / "workers/01/tasks.tsv").write_text("0\tgitlab\n1\tgitlab,reddit\n0\tgitlab\n")
    write_plan_manifest(root)
    with pytest.raises(ValueError, match="task assignment is incomplete"):
        analysis.analyze(root, catalog)


def test_plan_must_cover_exact_configured_case_ids(campaign_evidence):
    root, catalog = campaign_evidence
    config_path = root / "EXPERIMENT_CONFIG.json"
    config = json.loads(config_path.read_text())
    config["task_ids"] = [0, 1, 99]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configured task IDs"):
        analysis.analyze(root, catalog)
