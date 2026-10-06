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

RUN_ID = "wav-glm53flash-infra-v2-canary-20261005"
RUNNER_SHA = "a" * 64
PLAN_SHA = "b" * 64
IMAGE_SHA = "c" * 64
CAPTURE = {"har/harTracer.js": "d" * 64, "trace/recorder/tracing.js": "e" * 64}
SANDBOX = "owned-canary"
TASK_ID = 63


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
        "model": "glm-5.3-flash",
        "model_profile": "glm53flash",
        "model_protocol_error": "invalid computer_use action"
        if status_name == "model_protocol_failed"
        else None,
        "fixed_parameters": {
            "base_max_steps": 150,
            "final_response_reserve_steps": 7,
            "coordinate_type": "absolute",
            "startup_page_gate": True,
            "reasoning_effort": "max",
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
        "startup_receipt.json": json.dumps(
            {"result": "passed", "selected_page_url": "http://localhost:7780/admin"}
        ).encode(),
        "status.json": json.dumps(status).encode(),
        "run_metadata.json": json.dumps(
            {"script_sha256": RUNNER_SHA, "playwright_capture": CAPTURE}
        ).encode(),
        "baseline_receipt.json": json.dumps(
            {
                "session_id": RUN_ID,
                "sandbox_id": SANDBOX,
                "worker": "01",
                "task_id": TASK_ID,
                "image_lock_sha256": IMAGE_SHA,
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
        "model": "glm-5.3-flash",
        "max_tokens": 32768,
        "temperature": 0.0,
        "top_p": 0.9,
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "max",
    }
    files["requests/step_000_request.json"] = json.dumps(request).encode()
    files["requests/step_000_response.json"] = b"{}"
    files["requests/step_000_metadata.json"] = b"{}"
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
        playwright_capture=CAPTURE,
        runner_exit_code=runner_exit_code,
    )


def test_completed_case_has_scored_terminal_receipt(tmp_path: Path) -> None:
    index, archive = write_case(tmp_path, case_files())
    receipt = validate(index, archive)
    assert receipt["terminal_status"] == "completed"
    assert receipt["score"] == 1.0
    assert receipt["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert receipt["playwright_capture"] == CAPTURE


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
        "wrong_capture_producer",
        "missing_capture_producer",
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
        del files["requests/step_000_response.json"]
    elif change == "tampered_har":
        files["network.har"] = b"tampered"
    elif change == "evaluator_as_model_failure":
        files = case_files(official_status="error", score=0.0)
    elif change == "score_only_agreement":
        value = json.loads(files["evaluation.json"])
        value["inputs_agree"] = False
        files["evaluation.json"] = json.dumps(value).encode()
    elif change in {"wrong_capture_producer", "missing_capture_producer"}:
        value = json.loads(files["run_metadata.json"])
        if change == "wrong_capture_producer":
            value["playwright_capture"] = {**CAPTURE, "har/harTracer.js": "f" * 64}
        else:
            del value["playwright_capture"]
        files["run_metadata.json"] = json.dumps(value).encode()
    index, archive = write_case(tmp_path, files)
    with pytest.raises(receipt_module.EvidenceInvalid):
        validate(index, archive)


def test_nonzero_runner_exit_cannot_be_terminal(tmp_path: Path) -> None:
    index, archive = write_case(tmp_path, case_files())
    with pytest.raises(receipt_module.EvidenceInvalid, match="runner did not exit"):
        validate(index, archive, runner_exit_code=1)
