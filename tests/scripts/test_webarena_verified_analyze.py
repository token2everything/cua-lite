"""Native analysis uses small immutable archives, never live services or keys."""

from __future__ import annotations

import copy
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

RUN_ID = "wav-generic-qwen38max-full128-20261007-v1"
ENTRY = "scripts/webarena_verified_generic.py"
RUNNER = b"native agent.sample entrypoint fixture\n"
RUNNER_SHA = hashlib.sha256(RUNNER).hexdigest()
EVALUATOR_SHA = "e" * 64
OSWORLD_COMMIT = "c" * 40


def write_archive(path: Path, case: str, files: dict[str, bytes]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for filename, data in files.items():
            member = tarfile.TarInfo(f"{case}/{filename}")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_plan_manifest(root: Path) -> None:
    paths = [
        root / "EXPERIMENT_CONFIG.json",
        root / "worker-plan.json",
        *sorted((root / "workers").glob("*/*")),
    ]
    (root / "PLAN_MANIFEST.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
            for path in paths
        )
    )


def request(config: dict) -> dict:
    return {
        **config["agent"]["request_params"],
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "messages": [],
        "tools": [],
    }


def write_case(
    root: Path,
    task: dict,
    *,
    score: float | None = 1.0,
    terminal_status: str = "completed",
    failed_evaluators: tuple[str, ...] = (),
    truncated: bool = False,
    missing_usage: bool = False,
) -> dict:
    config = json.loads((root / "EXPERIMENT_CONFIG.json").read_text())
    task_id, case = task["task_id"], f"task-{task['task_id']:04d}"
    official_status = "error" if score is None else "success" if score == 1 else "failure"
    official_score = 0 if score is None else score
    status = {
        "task_id": task_id,
        "model": config["model"],
        "status": terminal_status,
        "score": official_score,
        "truncated": truncated,
        "terminal_action": "FINAL_JSON",
        "model_protocol_error": "bad answer"
        if terminal_status == "model_protocol_failed"
        else None,
        "response_repair_used": True,
        "model_protocol_failure_phase": None,
        "model_protocol_error_type": None,
    }
    official = {
        "task_id": task_id,
        "status": official_status,
        "score": official_score,
        "webarena_verified_version": "1.2.3",
        "webarena_verified_evaluator_checksum": EVALUATOR_SHA,
        "webarena_verified_data_checksum": config["benchmark"]["dataset_sha256"],
        "evaluators_results": [
            {
                "evaluator_name": name,
                "status": "error" if score is None else "failure",
                "actual_normalized": {"value": "actual"},
                "expected": {"value": "expected"},
                "error_msg": "error" if score is None else None,
                "assertions": [
                    {
                        "status": "failure",
                        "assertion_name": "value_mismatch",
                        "assertion_msgs": ["different values"],
                    }
                ],
            }
            for name in failed_evaluators
        ],
    }
    startup = {"result": "passed", "selected_page_url": "http://localhost:8023"}
    files = {
        "network.har": b'{"log":{}}',
        "trace.zip": b"trace fixture bytes",
        "agent_response_raw.txt": b'{"answer":"fixture"}',
        "startup_receipt.json": json.dumps(startup).encode(),
    }
    evaluation = {
        "har": official,
        "playwright_trace": copy.deepcopy(official),
        "integrity_status": "evaluator_error" if score is None else "passed",
        "inputs_agree": True,
        "score_inputs_agree": True,
        "score": official_score,
        "publishable": score is not None,
        "input_receipt": {
            key: {
                "sha256": hashlib.sha256(files[filename]).hexdigest(),
                "size": len(files[filename]),
            }
            for key, filename in (
                ("network_har", "network.har"),
                ("playwright_trace", "trace.zip"),
                ("startup_receipt", "startup_receipt.json"),
                ("agent_response_raw", "agent_response_raw.txt"),
            )
        },
    }
    metadata = {
        "script_sha256": RUNNER_SHA,
        "harness": "native_chat_gateway",
        "loop": "lite.agents.core.agent.base.AdapterBasedAgent.sample",
        "dataset_sha256": config["benchmark"]["dataset_sha256"],
        "osworld_sha": OSWORLD_COMMIT,
        "dependencies": {"agent": {"module_file": "/root/cua-lite/" + ENTRY, "sha256": RUNNER_SHA}},
    }
    baseline = {"fresh_service_processes": True}
    objects = {
        "status.json": status,
        "evaluation.json": evaluation,
        "run_metadata.json": metadata,
        "baseline_receipt.json": baseline,
    }
    for number, seconds in ((1, 5), (2, 4)):
        prefix = f"requests/{number:05d}"
        objects[prefix + ".request.json"] = request(config)
        attempts = [
            {"attempt": 1, "elapsed_seconds": seconds, "status_code": 200, "error_type": None}
        ]
        if number == 1:
            attempts = [
                {"attempt": 1, "elapsed_seconds": 2, "status_code": 503, "error_type": None},
                {"attempt": 2, "elapsed_seconds": 3, "status_code": 200, "error_type": None},
            ]
            objects[prefix + ".attempt-01.response.json"] = {
                "error": "temporarily unavailable",
                "usage": None if missing_usage else {"prompt_tokens": 2, "completion_tokens": 1},
            }
        objects[prefix + ".response.json"] = {
            "usage": None
            if missing_usage
            else {
                "prompt_tokens": 10 + 2 * (number - 1),
                "completion_tokens": 5 + number - 1,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
            "choices": [{"message": {"content": "fixture"}}],
        }
        objects[prefix + ".metadata.json"] = {
            "elapsed_seconds": seconds,
            "wall_seconds": seconds + 5,
            "attempts": attempts,
        }
    files.update({name: json.dumps(value).encode() for name, value in objects.items()})
    native = {
        "metadata": {"model_id": config["model"], "status": "completed"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Task."},
                    {"type": "image", "image": {"url": "images/0.png"}},
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": "fixture reasoning",
                "tool_calls": [
                    {"id": "call-original", "function": {"name": "computer", "arguments": {}}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-original",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "image", "image": {"url": "images/1.png"}},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Final answer."}]},
        ],
    }
    files["agent_trajectory.jsonl"] = (json.dumps(native) + "\n").encode()
    turns = [
        {
            "step": index,
            "executed_actions": [{"call": call, "args": {}}],
            "feedback": [{"tool_call_id": "call-original", "text": "ok", "error": None}],
            "timings": {"observe": 0.1, "predict": 10.0, "act": 0.5},
            "reserve_blocked": False,
        }
        for index, call in enumerate(("click", "response"))
    ]
    files["trajectory.jsonl"] = "".join(json.dumps(turn) + "\n" for turn in turns).encode()
    archive = root / "archives" / f"{case}.tar.gz"
    digest = write_archive(archive, case, files)
    index = root / "index" / case
    index.mkdir(parents=True)
    for filename in (
        "status.json",
        "evaluation.json",
        "run_metadata.json",
        "baseline_receipt.json",
        "startup_receipt.json",
    ):
        (index / filename).write_bytes(files[filename])
    receipt = {
        "run_id": config["run_id"],
        "runner_sha256": RUNNER_SHA,
        "task_id": task_id,
        "archive_sha256": digest,
        "worker": "01",
        "sandbox_id": "owned",
        "plan_sha256": analysis.campaign._sha(root / "PLAN_MANIFEST.sha256"),
        "image_lock_sha256": analysis.campaign._sha(root / "image-lock.env"),
        "terminal_status": terminal_status,
        "score": score,
        "official_evaluation_status": official_status,
        "evaluator_checksum": EVALUATOR_SHA,
        "data_checksum": config["benchmark"]["dataset_sha256"],
        "source_sha256": analysis.campaign._sha(root / "cua-lite-source.tar.gz"),
    }
    (index / "completed.json").write_text(json.dumps(receipt))
    ledger = {
        key: receipt[key]
        for key in (
            "run_id",
            "runner_sha256",
            "task_id",
            "archive_sha256",
            "worker",
            "sandbox_id",
            "plan_sha256",
            "image_lock_sha256",
            "terminal_status",
            "source_sha256",
        )
    }
    ledger.update(
        started_at="2026-10-07T01:00:00Z",
        finished_at="2026-10-07T01:01:00Z",
        runner_exit_code=0,
        archive_path=f"/mnt/results/archives/{case}.tar.gz",
        local_archive_path=str(archive),
        model_started=True,
        launch_id=f"final-{task_id}",
    )
    directory = root / "attempts" / case
    directory.mkdir(parents=True)
    (directory / "final.json").write_text(json.dumps(ledger))
    return files


@pytest.fixture
def campaign_evidence(tmp_path):
    tasks = [
        {
            "task_id": task_id,
            "sites": sites,
            "intent": "Public task goal",
            "intent_template_id": 1,
            "eval": [{"evaluator": "AgentResponseEvaluator", "expected": {"task_type": task_type}}],
        }
        for task_id, sites, task_type in (
            (0, ["gitlab"], "retrieve"),
            (1, ["gitlab", "reddit"], "mutate"),
            (2, ["reddit"], "navigate"),
        )
    ]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(tasks))
    config = {
        "run_id": RUN_ID,
        "task_count": 3,
        "task_ids": [0, 1, 2],
        "entrypoint": ENTRY,
        "experiment_version": "native-test",
        "model": "qwen3.8-max",
        "reasoning_effort": "xhigh",
        "benchmark": {
            "version": "1.2.3",
            "evaluator_sha256": EVALUATOR_SHA,
            "dataset_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        },
        "agent": {
            "request_params": {
                "max_tokens": 32768,
                "enable_thinking": True,
                "preserve_thinking": True,
                "stream": False,
            },
            "omitted_request_parameters": ["temperature", "top_p"],
        },
    }
    (tmp_path / "EXPERIMENT_CONFIG.json").write_text(json.dumps(config))
    source_path = tmp_path / "cua-lite-source.tar.gz"
    with tarfile.open(source_path, "w:gz") as archive:
        member = tarfile.TarInfo("./" + ENTRY)
        member.size = len(RUNNER)
        archive.addfile(member, io.BytesIO(RUNNER))
    (tmp_path / "VERSION_MANIFEST.json").write_text(
        json.dumps(
            {
                "runner_sha256": RUNNER_SHA,
                "source_sha256": analysis.campaign._sha(source_path),
                "osworld_commit": OSWORLD_COMMIT,
                "source_artifacts": {ENTRY: RUNNER_SHA},
            }
        )
    )
    (tmp_path / "image-lock.env").write_text("immutable image fixture\n")
    plan = {
        "selection": "canary",
        "task_count": 3,
        "workers": {"01": {"capabilities": ["gitlab", "reddit"], "startup_task": 0, "tasks": 3}},
        "tasks": {str(task["task_id"]): {"sites": task["sites"]} for task in tasks},
    }
    (tmp_path / "worker-plan.json").write_text(json.dumps(plan))
    workers = tmp_path / "workers/01"
    workers.mkdir(parents=True)
    (workers / "tasks.tsv").write_text("0\tgitlab\n1\tgitlab,reddit\n2\treddit\n")
    (workers / "capabilities.txt").write_text("gitlab,reddit\n")
    write_plan_manifest(tmp_path)
    return tmp_path, catalog, tasks


@pytest.mark.parametrize(
    "planned,scores,accuracy,pending,unscored",
    [
        (3, [1.0, 0.0], None, 1, 0),
        (3, [1.0, 0.0, None], None, 0, 1),
        (2, [0.0, 0.0], 0.0, 0, 0),
        (2, [1.0, 0.0], 0.5, 0, 0),
        (0, [], None, 0, 0),
    ],
)
def test_aggregation_never_converts_missing_or_unscored_into_failure(
    planned,
    scores,
    accuracy,
    pending,
    unscored,
):
    result = analysis.aggregate_metrics(planned, [{"score": score} for score in scores])
    assert result["official_accuracy"] == accuracy
    assert (result["pending"], result["unscored"], result["failed"]) == (
        pending,
        unscored,
        scores.count(0.0),
    )


def test_native_case_reads_usage_retries_gui_turns_and_assertions(campaign_evidence):
    root, _, tasks = campaign_evidence
    write_case(root, tasks[0], score=0.0, failed_evaluators=("AgentResponseEvaluator",))
    case = analysis.read_case(
        root,
        tasks[0],
        json.loads((root / "EXPERIMENT_CONFIG.json").read_text()),
        json.loads((root / "VERSION_MANIFEST.json").read_text()),
    )
    assert case["failure_signal"] == "response_requirement_failed"
    assert not case["failure_signal_is_root_cause"]
    assert case["failed_assertions"][0]["assertion_name"] == "value_mismatch"
    assert case["failed_evaluators"][0]["actual_normalized"] == {"value": "actual"}
    assert case["trajectory_steps"] == case["assistant_messages"] == 2
    assert case["executed_gui_actions"] == case["tool_calls"] == case["tool_result_messages"] == 1
    assert case["stored_screenshots"] == 2
    assert case["api"]["observed_token_usage"] == {"prompt_tokens": 24, "completion_tokens": 12}
    assert case["api"]["observed_http_attempts"] == 3
    assert case["api"]["transport_error_attempts"] == 1
    assert case["api"]["http_attempt_latency"]["sum_seconds"] == 9
    assert case["api"]["model_request_wall_time"]["sum_seconds"] == 19


def write_projection_failure(root: Path, task: dict, failed_step: int) -> dict[str, bytes]:
    files = write_case(root, task, score=0.0, terminal_status="model_protocol_failed")
    status = json.loads(files["status.json"])
    status.update(
        model_protocol_failure_phase="provider_projection",
        model_protocol_error_type="ChatGatewayProtocolError",
    )
    files["status.json"] = json.dumps(status).encode()
    native = json.loads(files["agent_trajectory.jsonl"])
    native["metadata"]["status"] = "failed"
    native["messages"] = native["messages"][:1] if failed_step == 0 else native["messages"][:-1]
    files["agent_trajectory.jsonl"] = (json.dumps(native) + "\n").encode()
    turns = [json.loads(line) for line in files["trajectory.jsonl"].splitlines()][:failed_step]
    raw_response = f"requests/{failed_step + 1:05d}.response.json"
    turns.append(
        {
            "step": failed_step,
            "prediction_failed": True,
            "raw_response": raw_response,
            "provider_content": None,
            "actions": [],
            "executed_actions": [],
            "feedback": [],
            "timings": {},
            "reserve_blocked": False,
            "terminated": True,
            "truncated": False,
            "failure_phase": "provider_projection",
        }
    )
    files["trajectory.jsonl"] = "".join(json.dumps(turn) + "\n" for turn in turns).encode()
    for name in list(files):
        if name.startswith("requests/00002") and failed_step == 0:
            del files[name]
    response = json.loads(files[raw_response])
    response["choices"][0]["message"] = {
        "tool_calls": [
            {
                "id": "bad-call",
                "type": "function",
                "function": {"name": "computer", "arguments": "not JSON"},
            }
        ]
    }
    files[raw_response] = json.dumps(response).encode()
    files["agent_response_raw.txt"] = b""
    evaluation = json.loads(files["evaluation.json"])
    evaluation["input_receipt"]["agent_response_raw"] = {
        "sha256": hashlib.sha256(b"").hexdigest(),
        "size": 0,
    }
    files["evaluation.json"] = json.dumps(evaluation).encode()
    publish_changed_case(root, files)
    return files


def publish_changed_case(root: Path, files: dict[str, bytes]) -> None:
    """Rebind small test evidence after an intentional source-contract mutation."""
    case = "task-0000"
    digest = write_archive(root / "archives" / f"{case}.tar.gz", case, files)
    for filename in ("status.json", "evaluation.json", "run_metadata.json"):
        (root / "index" / case / filename).write_bytes(files[filename])
    for path in (root / "index" / case / "completed.json", root / "attempts" / case / "final.json"):
        receipt = json.loads(path.read_text())
        receipt["archive_sha256"] = digest
        path.write_text(json.dumps(receipt))


@pytest.mark.parametrize("failed_step", [0, 1])
def test_typed_projection_failure_is_one_real_decision_without_invented_assistant(
    campaign_evidence,
    failed_step,
):
    root, catalog, tasks = campaign_evidence
    files = write_projection_failure(root, tasks[0], failed_step)
    report = analysis.analyze(root, catalog)
    case = report["cases"][0]
    assert case["terminal_status"] == case["failure_signal"] == "model_protocol_failed"
    assert case["score"] == 0 and report["infra_attempt_count"] == 0
    assert case["native_trajectory_status"] == "failed"
    assert case["assistant_messages"] == failed_step
    assert case["trajectory_steps"] == case["api"]["model_requests"] == failed_step + 1
    assert case["projection_error_turns"] == [failed_step]
    assert case["projection_error_responses"] == [f"requests/{failed_step + 1:05d}.response.json"]
    assert case["executed_gui_actions"] == failed_step
    assert (root / "index/task-0000/status.json").read_bytes() == files["status.json"]
    assert "not JSON" not in json.dumps(report)  # Raw model output remains in its bound archive.


def test_non_object_successful_http_body_is_retained_as_typed_failure_with_unknown_usage(
    campaign_evidence,
):
    root, catalog, tasks = campaign_evidence
    files = write_projection_failure(root, tasks[0], 1)
    files["requests/00002.response.json"] = b"[]"
    publish_changed_case(root, files)
    report = analysis.analyze(root, catalog)
    assert report["cases"][0]["failure_signal"] == "model_protocol_failed"
    assert report["all_attempts_api"]["usage_missing_attempts"] == 1


@pytest.mark.parametrize(
    "damage",
    [
        "phase",
        "type",
        "missing_wire",
        "wrong_wire",
        "invented_action",
        "missing_marker",
        "completed_canonical",
        "unrelated_missing_assistant",
        "unsuccessful_http",
    ],
)
def test_projection_marker_does_not_hide_missing_or_fabricated_native_evidence(
    campaign_evidence,
    damage,
):
    root, catalog, tasks = campaign_evidence
    files = write_projection_failure(root, tasks[0], 1)
    if damage in {"phase", "type"}:
        status = json.loads(files["status.json"])
        status[
            "model_protocol_failure_phase" if damage == "phase" else "model_protocol_error_type"
        ] = "other_phase" if damage == "phase" else "ChatGatewayError"
        files["status.json"] = json.dumps(status).encode()
    elif damage in {"completed_canonical", "unrelated_missing_assistant"}:
        native = json.loads(files["agent_trajectory.jsonl"])
        if damage == "completed_canonical":
            native["metadata"]["status"] = "completed"
        else:
            native["messages"] = [m for m in native["messages"] if m["role"] != "assistant"]
        files["agent_trajectory.jsonl"] = (json.dumps(native) + "\n").encode()
    elif damage == "missing_wire":
        del files["requests/00002.response.json"]
    elif damage == "unsuccessful_http":
        metadata = json.loads(files["requests/00002.metadata.json"])
        metadata["attempts"][0]["status_code"] = 503
        files["requests/00002.metadata.json"] = json.dumps(metadata).encode()
    else:
        turns = [json.loads(line) for line in files["trajectory.jsonl"].splitlines()]
        failure = turns[-1]
        if damage == "wrong_wire":
            failure["raw_response"] = "requests/00001.response.json"
        elif damage == "invented_action":
            failure["actions"] = [{"name": "response", "arguments": {"text": "invented"}}]
        else:
            del failure["prediction_failed"]
        files["trajectory.jsonl"] = "".join(json.dumps(turn) + "\n" for turn in turns).encode()
    publish_changed_case(root, files)
    with pytest.raises(ValueError):
        analysis.analyze(root, catalog)


def test_failed_canonical_ordinary_case_is_still_incomplete_evidence(campaign_evidence):
    root, catalog, tasks = campaign_evidence
    files = write_case(root, tasks[0], score=0.0)
    native = json.loads(files["agent_trajectory.jsonl"])
    native["metadata"]["status"] = "failed"
    files["agent_trajectory.jsonl"] = (json.dumps(native) + "\n").encode()
    publish_changed_case(root, files)
    with pytest.raises(ValueError, match="canonical completion"):
        analysis.analyze(root, catalog)


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
    campaign_evidence,
    terminal_status,
    score,
    truncated,
    evaluators,
    signal,
):
    root, catalog, tasks = campaign_evidence
    write_case(
        root,
        tasks[0],
        terminal_status=terminal_status,
        score=score,
        truncated=truncated,
        failed_evaluators=evaluators,
    )
    case = analysis.analyze(root, catalog)["cases"][0]
    assert case["failure_signal"] == signal and case["score"] == score


def test_absent_usage_is_unknown_and_raw_detail_usage_is_preserved(campaign_evidence):
    root, catalog, tasks = campaign_evidence
    write_case(root, tasks[0], missing_usage=True)
    report = analysis.analyze(root, catalog)
    assert report["all_attempts_api"]["usage_missing_attempts"] == 3
    assert report["all_attempts_api"]["usage_recorded_attempts"] == 0
    assert report["all_attempts_api"]["observed_token_usage"] == {}
    assert report["all_attempts_api"]["token_usage_is_billing_total"] is False


def test_full_partial_dimensions_and_reference_markdown_keep_denominators(campaign_evidence):
    root, catalog, tasks = campaign_evidence
    write_case(root, tasks[0])
    write_case(
        root,
        tasks[1],
        terminal_status="evaluator_error",
        score=None,
        failed_evaluators=("NetworkEventEvaluator",),
    )
    issues = root / "issues.jsonl"
    issues.write_text(json.dumps({"issue": "recorded timeout", "status": "recorded"}) + "\n")
    report = analysis.analyze(root, catalog, issues)
    assert report["complete"] is False and report["fully_scored"] is False
    assert report["missing_task_ids"] == [2]
    assert report["evaluator_error_task_ids"] == [1]
    assert report["summary"]["official_accuracy"] is None
    assert report["dimensions"]["single_site"]["planned"] == 2
    assert report["dimensions"]["multi_site/mutate"]["unscored"] == 1
    assert report["dimensions"]["sites/gitlab+reddit"]["unscored"] == 1
    assert report["all_attempts_api"]["archived_case_attempts"] == 2
    assert report["all_attempts_api"]["terminal_archives_accounted_for"] is True
    assert report["attempts"][0]["api_attempts"][1]["usage"]["completion_tokens_details"] == {
        "reasoning_tokens": 3,
    }
    markdown = analysis.render_markdown(report)
    assert "| 范围 | 类型 | 通过/总数 | 成功率 |" in markdown
    assert "| 全量 | 全部 | 1/3 | 未形成完整评分 |" in markdown
    assert "| 站点 | 通过/总数 | 成功率 |" in markdown
    assert "| 站点组合 | 通过/总数 | 成功率 |" in markdown
    assert markdown.rsplit("## ", 1)[1].startswith("本次实验遇到的问题")
    assert "recorded timeout" in markdown


def test_failed_case_attempt_usage_is_included_once_without_scoring_as_model_failure(
    campaign_evidence,
):
    root, catalog, tasks = campaign_evidence
    write_case(root, tasks[0])
    ledger_path = root / "attempts/task-0000/final.json"
    ledger = json.loads(ledger_path.read_text())
    config = json.loads((root / "EXPERIMENT_CONFIG.json").read_text())
    files = {
        "requests/00001.request.json": json.dumps(request(config)).encode(),
        "requests/00001.metadata.json": json.dumps(
            {
                "elapsed_seconds": 600.0,
                "wall_seconds": 601.0,
                "attempts": [
                    {
                        "attempt": 1,
                        "elapsed_seconds": 600.0,
                        "status_code": None,
                        "error_type": "ReadTimeout",
                    }
                ],
            }
        ).encode(),
    }
    archive = ledger_path.parent / "failed.tar.gz"
    ledger.update(
        archive_sha256=write_archive(archive, "task-0000", files),
        local_archive_path=str(archive),
        runner_exit_code=1,
        terminal_status="infrastructure_failed",
        launch_id="failed-0",
    )
    (ledger_path.parent / "failed.json").write_text(json.dumps(ledger))
    report = analysis.analyze(root, catalog)
    assert report["summary"]["passed"] == 1 and report["summary"]["failed"] == 0
    assert report["infra_attempt_count"] == 1 and report["infra_affected_task_ids"] == [0]
    assert report["all_attempts_api"]["archived_case_attempts"] == 2
    assert report["all_attempts_api"]["observed_http_attempts"] == 4
    assert report["all_attempts_api"]["http_attempt_latency"]["sum_seconds"] == 609.0
    assert report["all_attempts_api"]["usage_missing_attempts"] == 1
    assert report["all_attempts_api"]["observed_token_usage"] == {
        "prompt_tokens": 24,
        "completion_tokens": 12,
    }


@pytest.mark.parametrize("model_started", [False, True])
def test_no_model_attempt_can_be_archive_free_but_started_model_cannot(
    campaign_evidence,
    model_started,
):
    root, catalog, tasks = campaign_evidence
    write_case(root, tasks[0])
    path = root / "attempts/task-0000/final.json"
    ledger = json.loads(path.read_text())
    ledger.update(
        archive_sha256=None,
        archive_path=None,
        local_archive_path=None,
        model_started=model_started,
        terminal_status="infrastructure_failed",
        runner_exit_code=1,
        launch_id="no-model-attempt",
    )
    (path.parent / "no-model.json").write_text(json.dumps(ledger))
    if model_started:
        with pytest.raises(ValueError, match="model-started attempt"):
            analysis.analyze(root, catalog)
    else:
        report = analysis.analyze(root, catalog)
        assert report["infra_attempt_count"] == 1
        assert report["all_attempts_api"]["model_requests"] == 2
        assert report["all_attempts_api"]["observed_http_attempts"] == 3


def test_glm_model_and_effort_are_read_from_same_native_config_contract(campaign_evidence):
    root, catalog, tasks = campaign_evidence
    path = root / "EXPERIMENT_CONFIG.json"
    config = json.loads(path.read_text())
    config.update(
        model="glm-5.3-flash",
        reasoning_effort="max",
        run_id="wav-generic-glm53flash-full128-20261007-v1",
    )
    path.write_text(json.dumps(config))
    write_plan_manifest(root)
    write_case(root, tasks[0])
    report = analysis.analyze(root, catalog)
    assert report["model"] == "glm-5.3-flash"
    assert report["run_id"] == config["run_id"]
    assert report["summary"]["passed"] == 1


def test_missing_attempt_ledger_is_explicit_not_a_final_only_cost_total(campaign_evidence):
    root, catalog, tasks = campaign_evidence
    write_case(root, tasks[0])
    (root / "attempts/task-0000/final.json").unlink()
    report = analysis.analyze(root, catalog)
    assert report["all_attempts_api"]["terminal_archives_without_attempt_ledger"] == [0]
    assert report["all_attempts_api"]["terminal_archives_accounted_for"] is False
    assert report["all_attempts_api"]["http_attempt_latency"]["sum_seconds"] is None
    assert report["cases"][0]["api"]["http_attempt_latency"]["sum_seconds"] == 9


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "runner_sha256",
        "plan_sha256",
        "image_lock_sha256",
        "source_sha256",
        "data_checksum",
        "evaluator_checksum",
        "archive_sha256",
    ],
)
def test_mixed_version_or_changed_archive_cannot_enter_statistics(campaign_evidence, field):
    root, catalog, tasks = campaign_evidence
    write_case(root, tasks[0])
    path = root / "index/task-0000/completed.json"
    receipt = json.loads(path.read_text())
    receipt[field] = "wrong"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        analysis.analyze(root, catalog)


@pytest.mark.parametrize("change", ["source", "index", "attempt_identity", "attempt_path"])
def test_source_index_and_attempt_scope_mismatches_are_rejected(campaign_evidence, change):
    root, catalog, tasks = campaign_evidence
    write_case(root, tasks[0])
    if change == "source":
        with (root / "cua-lite-source.tar.gz").open("ab") as source:
            source.write(b"different source")
    elif change == "index":
        (root / "index/task-0000/status.json").write_text("{}")
    else:
        path = root / "attempts/task-0000/final.json"
        ledger = json.loads(path.read_text())
        ledger["run_id" if change == "attempt_identity" else "local_archive_path"] = "wrong"
        if change == "attempt_path":
            ledger["local_archive_path"] = "/outside/result-root/archive.tar.gz"
        path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        analysis.analyze(root, catalog)


@pytest.mark.parametrize("change", ["temperature", "top_p", "native_shape", "input_hash"])
def test_rehashed_wrong_request_shape_or_scored_input_is_rejected(campaign_evidence, change):
    root, catalog, tasks = campaign_evidence
    files = write_case(root, tasks[0])
    if change in {"temperature", "top_p"}:
        body = json.loads(files["requests/00001.request.json"])
        body[change] = 0.0
        files["requests/00001.request.json"] = json.dumps(body).encode()
    elif change == "native_shape":
        files["agent_trajectory.jsonl"] *= 2
    else:
        files["network.har"] = b"changed HAR"
    digest = write_archive(root / "archives/task-0000.tar.gz", "task-0000", files)
    path = root / "index/task-0000/completed.json"
    receipt = json.loads(path.read_text())
    receipt["archive_sha256"] = digest
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        analysis.analyze(root, catalog)


@pytest.mark.parametrize("change", ["import_path", "import_hash", "official_pin", "official_score"])
def test_rehashed_runtime_or_official_contract_drift_cannot_enter_statistics(
    campaign_evidence,
    change,
):
    root, catalog, tasks = campaign_evidence
    files = write_case(root, tasks[0])
    if change.startswith("import"):
        metadata = json.loads(files["run_metadata.json"])
        identity = metadata["dependencies"]["agent"]
        identity["module_file" if change == "import_path" else "sha256"] = (
            "/other/editable/source.py" if change == "import_path" else "wrong"
        )
        files["run_metadata.json"] = json.dumps(metadata).encode()
        (root / "index/task-0000/run_metadata.json").write_bytes(files["run_metadata.json"])
    else:
        evaluation = json.loads(files["evaluation.json"])
        for field in ("har", "playwright_trace"):
            evaluation[field][
                "webarena_verified_evaluator_checksum" if change == "official_pin" else "status"
            ] = "wrong" if change == "official_pin" else "failure"
        files["evaluation.json"] = json.dumps(evaluation).encode()
        (root / "index/task-0000/evaluation.json").write_bytes(files["evaluation.json"])
    digest = write_archive(root / "archives/task-0000.tar.gz", "task-0000", files)
    path = root / "index/task-0000/completed.json"
    receipt = json.loads(path.read_text())
    receipt["archive_sha256"] = digest
    if change == "official_score":
        receipt["official_evaluation_status"] = "failure"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        analysis.analyze(root, catalog)


@pytest.mark.parametrize("rows", ["0\tgitlab\n1\tgitlab,reddit\n", "0\tgitlab\n" * 3])
def test_modified_task_table_cannot_change_statistics(campaign_evidence, rows):
    root, catalog, _ = campaign_evidence
    (root / "workers/01/tasks.tsv").write_text(rows)
    with pytest.raises(ValueError, match="invalid plan manifest entry"):
        analysis.analyze(root, catalog)


def test_rehashed_truncated_plan_cannot_shrink_configured_denominator(campaign_evidence):
    root, catalog, _ = campaign_evidence
    (root / "workers/01/tasks.tsv").write_text("0\tgitlab\n1\tgitlab,reddit\n")
    path = root / "worker-plan.json"
    plan = json.loads(path.read_text())
    plan["task_count"] = plan["workers"]["01"]["tasks"] = 2
    path.write_text(json.dumps(plan))
    write_plan_manifest(root)
    with pytest.raises(ValueError, match="configured task count"):
        analysis.analyze(root, catalog)


def test_full_plan_cannot_claim_full_accuracy_for_three_tasks(campaign_evidence):
    root, catalog, _ = campaign_evidence
    path = root / "worker-plan.json"
    plan = json.loads(path.read_text())
    plan["selection"] = "full"
    path.write_text(json.dumps(plan))
    write_plan_manifest(root)
    with pytest.raises(ValueError, match="exactly 0..811"):
        analysis.analyze(root, catalog)


def test_all_scored_plan_produces_accuracy_and_cli_files_without_overwriting(
    campaign_evidence,
    monkeypatch,
    capsys,
):
    root, catalog, tasks = campaign_evidence
    for task in tasks:
        write_case(root, task, score=0.0 if task["task_id"] == 1 else 1.0)
    output = root / "analysis"
    monkeypatch.setattr(
        analysis,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "run_root": root,
                "catalog": catalog,
                "issues_path": None,
                "output_dir": output,
            },
        )(),
    )
    assert analysis.main() == 0
    capsys.readouterr()
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] and report["fully_scored"]
    assert report["summary"]["official_accuracy"] == 2 / 3
    assert len((output / "cases.jsonl").read_text().splitlines()) == 3
    assert len((output / "attempts.jsonl").read_text().splitlines()) == 3
    assert "| 全量 | 全部 | 2/3 | 66.67% |" in (output / "report.md").read_text()
    with pytest.raises(FileExistsError):
        analysis.main()


@pytest.mark.parametrize("values", [[-1], [float("nan")], [float("inf")]])
def test_unusable_recorded_latency_is_not_reported_as_provider_speed(values):
    with pytest.raises(ValueError, match="latency"):
        analysis.latency_metrics(values)
