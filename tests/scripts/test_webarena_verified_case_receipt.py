from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/webarena_verified_case_receipt.py"
SPEC = importlib.util.spec_from_file_location("webarena_verified_case_receipt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
receipt_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receipt_module)

RUN_ID = "wav-generic-qwen38max-full128-20261007-v1"
RUNNER_SHA = "a" * 64
PLAN_SHA = "b" * 64
IMAGE_SHA = "c" * 64
SANDBOX = "owned-canary"
TASK_ID = 63
CONFIG = {
    "run_id": RUN_ID,
    "model": "qwen3.8-max",
    "model_profile": "qwen",
    "reasoning_effort": "xhigh",
    "environment": {
        "max_steps": 150,
        "final_response_reserve_steps": 7,
        "coordinates": "canonical_normalized_0_1000",
    },
    "agent": {
        "image_budget_including_current": 20,
        "image_fold_size": 10,
        "request_params": {
            "max_tokens": 32768,
            "enable_thinking": True,
            "preserve_thinking": True,
            "stream": False,
        },
        "omitted_request_parameters": ["temperature", "top_p"],
    },
    "benchmark": {
        "version": "1.2.3",
        "evaluator_sha256": "evaluator-checksum",
        "dataset_sha256": "data-checksum",
    },
}
VERSION = {
    "source_sha256": "d" * 64,
    "osworld_commit": "b138d348256078fa634fc3b73567a7337c793e6b",
    "source_artifacts": {"lite/agent.py": "agent-source-hash"},
}


def _identity(data: bytes, *, path: str | None = None) -> dict:
    value = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    if path is not None:
        value["path"] = path
    return value


def case_files(
    *,
    status_name: str = "completed",
    official_status: str = "success",
    score: float = 1.0,
) -> dict[str, bytes]:
    status = {
        "task_id": TASK_ID,
        "status": status_name,
        "score": score,
        "model": CONFIG["model"],
        "model_profile": CONFIG["model_profile"],
        "model_protocol_error": "invalid computer_use action"
        if status_name == "model_protocol_failed"
        else None,
        "fixed_parameters": {
            "base_max_steps": 150,
            "final_response_reserve_steps": 7,
            "coordinate_type": "canonical_normalized_0_1000",
            "startup_page_gate": True,
            "reasoning_effort": "xhigh",
            "text_history": "all",
            "image_max": 20,
            "fold_size": 10,
        },
    }
    official = {
        "task_id": TASK_ID,
        "status": official_status,
        "score": score,
        "webarena_verified_version": "1.2.3",
        "webarena_verified_evaluator_checksum": "evaluator-checksum",
        "webarena_verified_data_checksum": "data-checksum",
        "evaluators_results": [
            {"evaluator_name": "AgentResponseEvaluator", "status": official_status, "score": score}
        ],
    }
    files = {
        "network.har": b"har-evidence",
        "trace.zip": b"trace-evidence",
        "agent_response_raw.txt": b"result",
        "trajectory.jsonl": b"{}\n",
        "agent_trajectory.jsonl": b"{}\n",
        "startup_receipt.json": json.dumps(
            {"result": "passed", "selected_page_url": "http://localhost:7780/admin"}
        ).encode(),
        "status.json": json.dumps(status).encode(),
        "run_metadata.json": json.dumps(
            {
                "script_sha256": RUNNER_SHA,
                "harness": "native_chat_gateway",
                "osworld_sha": VERSION["osworld_commit"],
                "dependencies": {
                    owner: {
                        "module_file": "/root/cua-lite/lite/agent.py",
                        "sha256": "agent-source-hash",
                    }
                    for owner in (
                        "agent",
                        "loop",
                        "prompt",
                        "action_space",
                        "env_ingress",
                        "desktop_action_projection",
                        "model_input_projection",
                        "desktop_scoring_helpers",
                        "catalog_trace_loader",
                    )
                }
                | {
                    "cursor": {
                        "module_file": "/opt/OSWorld/desktop_env/server/pyxcursor.py",
                        "sha256": "cursor-hash",
                    }
                },
            }
        ).encode(),
        "baseline_receipt.json": json.dumps(
            {
                "session_id": RUN_ID,
                "sandbox_id": SANDBOX,
                "worker": "01",
                "task_id": TASK_ID,
                "image_lock_sha256": IMAGE_SHA,
                "source_sha256": VERSION["source_sha256"],
                "fresh_service_processes": True,
            }
        ).encode(),
    }
    evaluation = {
        "score": score,
        "har": official,
        "playwright_trace": official,
        "inputs_agree": True,
        "score_inputs_agree": True,
        "integrity_status": "evaluator_error" if official_status == "error" else "passed",
        "publishable": official_status != "error",
        "input_receipt": {
            "network_har": _identity(files["network.har"], path="network.har"),
            "playwright_trace": _identity(files["trace.zip"], path="trace.zip"),
            "startup_receipt": _identity(
                files["startup_receipt.json"], path="startup_receipt.json"
            ),
            "agent_response_raw": _identity(files["agent_response_raw.txt"]),
        },
    }
    files["evaluation.json"] = json.dumps(evaluation).encode()
    request = {
        **CONFIG["agent"]["request_params"],
        "model": CONFIG["model"],
        "reasoning_effort": CONFIG["reasoning_effort"],
        "tools": [{"type": "function", "function": {"name": "computer"}}],
    }
    files["requests/00001.request.json"] = json.dumps(request).encode()
    files["requests/00001.response.json"] = b"{}"
    files["requests/00001.metadata.json"] = b"{}"
    return files


def write_case(tmp_path: Path, files: dict[str, bytes]) -> tuple[Path, Path]:
    index = tmp_path / "index" / f"task-{TASK_ID:04d}"
    index.mkdir(parents=True)
    for name, data in files.items():
        if "/" not in name:
            (index / name).write_bytes(data)
    archive = tmp_path / f"task-{TASK_ID:04d}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in files.items():
            member = tarfile.TarInfo(f"task-{TASK_ID:04d}/{name}")
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    return index, archive


def validate(index: Path, archive: Path, runner_exit_code: int = 0) -> dict:
    return receipt_module.make_receipt(
        index,
        archive,
        run_id=RUN_ID,
        plan_sha256=PLAN_SHA,
        worker="01",
        sandbox_id=SANDBOX,
        task_id=TASK_ID,
        runner_sha256=RUNNER_SHA,
        image_lock_sha256=IMAGE_SHA,
        runner_exit_code=runner_exit_code,
        config=CONFIG,
        version_manifest=VERSION,
    )


def test_completed_case_has_scored_terminal_receipt(tmp_path: Path) -> None:
    index, archive = write_case(tmp_path, case_files())
    receipt = validate(index, archive)
    assert receipt["terminal_status"] == "completed"
    assert receipt["score"] == 1.0
    assert receipt["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_model_protocol_failure_is_terminal_and_scored_once(tmp_path: Path) -> None:
    index, archive = write_case(
        tmp_path,
        case_files(status_name="model_protocol_failed", official_status="failure", score=0.0),
    )
    receipt = validate(index, archive)
    assert receipt["terminal_status"] == "model_protocol_failed"
    assert receipt["score"] == 0.0


def test_official_evaluator_error_is_terminal_but_unscored(tmp_path: Path) -> None:
    index, archive = write_case(
        tmp_path, case_files(status_name="evaluator_error", official_status="error", score=0.0)
    )
    receipt = validate(index, archive)
    assert receipt["terminal_status"] == "evaluator_error"
    assert receipt["score"] is None


@pytest.mark.parametrize(
    "change",
    [
        "failed_status",
        "wrong_runner",
        "bad_startup",
        "missing_response",
        "tampered_har",
        "evaluator_as_model_failure",
        "score_only_agreement",
    ],
)
def test_nonterminal_evidence_never_publishes_receipt(tmp_path: Path, change: str) -> None:
    files = case_files()
    if change == "failed_status":
        value = json.loads(files["status.json"])
        value["status"] = "failed"
        files["status.json"] = json.dumps(value).encode()
    elif change == "wrong_runner":
        files["run_metadata.json"] = json.dumps({"script_sha256": "old-runner"}).encode()
    elif change == "bad_startup":
        files["startup_receipt.json"] = b'{"result":"failed"}'
    elif change == "missing_response":
        del files["requests/00001.response.json"]
    elif change == "tampered_har":
        files["network.har"] = b"tampered"
    elif change == "evaluator_as_model_failure":
        files = case_files(official_status="error", score=0.0)
    elif change == "score_only_agreement":
        value = json.loads(files["evaluation.json"])
        value["inputs_agree"] = False
        files["evaluation.json"] = json.dumps(value).encode()
    index, archive = write_case(tmp_path, files)
    with pytest.raises(receipt_module.EvidenceInvalid):
        validate(index, archive)


def test_nonzero_runner_exit_cannot_be_terminal(tmp_path: Path) -> None:
    index, archive = write_case(tmp_path, case_files())
    with pytest.raises(receipt_module.EvidenceInvalid, match="runner did not exit"):
        validate(index, archive, runner_exit_code=1)


@pytest.mark.parametrize("key", ["temperature", "top_p"])
def test_native_request_must_leave_sampling_defaults_unspecified(tmp_path, key):
    files = case_files()
    request = json.loads(files["requests/00001.request.json"])
    request[key] = 0.9
    files["requests/00001.request.json"] = json.dumps(request).encode()
    index, archive = write_case(tmp_path, files)
    with pytest.raises(receipt_module.EvidenceInvalid, match="sampling defaults"):
        validate(index, archive)


def test_native_failed_transport_response_is_retained_not_mistaken_for_extra_success(tmp_path):
    files = case_files()
    files["requests/00001.attempt-01.response.json"] = b'{"error":"timeout"}'
    files["requests/00001.attempt-01.error.json"] = b'{"elapsed_seconds":600}'
    index, archive = write_case(tmp_path, files)
    assert validate(index, archive)["score"] == 1


def test_wrong_official_dataset_pin_prevents_publication(tmp_path):
    files = case_files()
    evaluation = json.loads(files["evaluation.json"])
    evaluation["har"]["webarena_verified_data_checksum"] = "wrong-published-dataset"
    evaluation["playwright_trace"]["webarena_verified_data_checksum"] = "wrong-published-dataset"
    files["evaluation.json"] = json.dumps(evaluation).encode()
    index, archive = write_case(tmp_path, files)
    with pytest.raises(receipt_module.EvidenceInvalid, match="pinned config"):
        validate(index, archive)


@pytest.mark.parametrize("change", ["outside_source", "wrong_hash", "missing_loop"])
def test_actual_imported_code_must_match_frozen_source(tmp_path, change):
    files = case_files()
    metadata = json.loads(files["run_metadata.json"])
    if change == "outside_source":
        metadata["dependencies"]["agent"]["module_file"] = "/root/editable-old-source/lite/agent.py"
    elif change == "wrong_hash":
        metadata["dependencies"]["agent"]["sha256"] = "different-old-source"
    else:
        del metadata["dependencies"]["loop"]
    files["run_metadata.json"] = json.dumps(metadata).encode()
    index, archive = write_case(tmp_path, files)
    with pytest.raises(receipt_module.EvidenceInvalid, match="imported|dependency"):
        validate(index, archive)
