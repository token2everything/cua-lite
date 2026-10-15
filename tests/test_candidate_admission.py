"""Exercise candidate admission through durable commands and real byte archives.

Gate reports are offline boundary fixtures, not native replay or model evidence.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from sandbox import _atomic_json, _sha

import pipeline
from pipeline import Coordinator


def admission_fixture(tmp_path, monkeypatch, replay, validate=None):
    owner = Coordinator.__new__(Coordinator)
    owner.run = tmp_path / "run"
    owner.args = SimpleNamespace(run="admissionfixture")
    owner.site = "map"
    record = {"id": "owned-sandbox", "metadata": {"session_nonce": "owned-nonce"}}
    owner.owned = lambda: record
    tasks = [f"wavgen-admissionfixture-{6107101 + index}" for index in range(len(replay))]
    original = {}
    for index, task in enumerate(tasks):
        directory = owner.run / "candidates" / task
        directory.mkdir(parents=True)
        for name in ("private.jsonl", "public.jsonl"):
            # Preserve noncanonical whitespace, UTF-8 and CRLF at the byte boundary.
            raw = (json.dumps({"generated_task_id": task, "value": index, "text": "café"},
                              ensure_ascii=False, separators=(", ", ": ")) + "\r\n").encode()
            (directory / name).write_bytes(raw)
            original[directory / name] = raw
    raw_reports = {}
    for gate, statuses in (("replay", replay), ("validate", validate or ["pass"] * len(tasks))):
        raw_reports[gate] = b"".join(
            (json.dumps({"generated_task_id": task, "status": status, "raw_evidence": "café"},
                        ensure_ascii=False, separators=(", ", ": ")) + "\r\n").encode()
            for task, status in zip(tasks, statuses, strict=True)
        )
    outputs = tmp_path / "remote/results"
    remote = tmp_path / "remote/infra"
    events = []
    controls = {"submit_error": {}, "observe": {}, "transfer_error": None}
    handles = {}

    def local(path):
        return Path(path.replace("/var/lib/wav-v03/results", str(outputs))
                        .replace("/mnt/results/v03", str(remote)))

    def upload(sandbox, path, data):
        assert sandbox == record["id"]
        target = local(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        events.append(("upload", path, data))

    def command(sandbox, command, *, label, background, timeout_ms):
        assert sandbox == record["id"] and background is True
        assert timeout_ms == 259200000
        events.append(("submit", label, command))
        intent = {"sandbox_id": sandbox, "command": command}
        path = owner.run / "commands" / (label + ".intent.json")
        if path.exists():
            assert json.loads(path.read_text()) == intent
            if label not in handles:
                raise RuntimeError("Original submission outcome unknown; no second POST")
            return handles[label]
        _atomic_json(path, intent)
        gate = label.split("-")[1]
        if label.endswith("-archive"):
            body = shlex.split(command)[2].replace("/var/lib/wav-v03/results", str(outputs))
            body = body.replace("/mnt/results/v03", str(remote))
            subprocess.run(["bash", "-c", body], check=True, capture_output=True)
            code = 0
        else:
            if gate in controls["submit_error"]:
                raise controls["submit_error"][gate]
            body = shlex.split(command)[2]
            argv = shlex.split(body.splitlines()[-1])
            receipt = local(argv[argv.index("--receipts") + 1].replace("/outputs", "/var/lib/wav-v03/results"))
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_bytes(raw_reports[gate])
            parsed = [json.loads(line) for line in raw_reports[gate].splitlines()]
            code = int(any(row["status"] != "pass" for row in parsed))
        handle = "original-handle-" + label
        handles[label] = handle
        _atomic_json(owner.run / "commands" / (label + ".handle.json"), {"command_id": handle})
        controls["observe"].setdefault(label, [{"id": handle, "running": False, "exit_code": code}])
        return handle

    def command_status(sandbox, handle):
        assert sandbox == record["id"]
        label = next(key for key, value in handles.items() if value == handle)
        events.append(("observe", label, handle))
        states = controls["observe"].get(label, controls["observe"].get(label.split("-")[1]))
        # Gate-wide overrides are configured before the actual label is known.
        if not label.endswith("-archive") and label.split("-")[1] in controls["observe"]:
            states = controls["observe"][label.split("-")[1]]
        state = states.pop(0) if len(states) > 1 else states[0]
        if isinstance(state, Exception):
            raise state
        return {"id": handle, **state}

    def download(sandbox, path):
        assert sandbox == record["id"]
        return local(path).read_bytes()

    def download_large(sandbox, path, destination, digest):
        assert sandbox == record["id"]
        if controls["transfer_error"]:
            raise controls["transfer_error"]
        source = local(path)
        assert _sha(source) == digest
        shutil.copyfile(source, destination)
        events.append(("transfer", destination.name))

    owner.client = SimpleNamespace(upload=upload, command=command, command_status=command_status,
                                   download=download, download_large_file=download_large)
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: events.append(("sleep", seconds)))
    return owner, tasks, original, raw_reports, events, controls


@pytest.mark.parametrize("replay", [
    ["pass", "pass", "pass", "quarantine"],
    ["pass", "quarantine", "pass", "quarantine", "pass"],
])
def test_mixed_native_reports_admit_only_passes_and_preserve_all_raw_bytes(tmp_path, monkeypatch, replay):
    owner, tasks, original, reports, events, _controls = admission_fixture(tmp_path, monkeypatch, replay)
    owner.check()
    admitted = json.loads((owner.run / "admitted.json").read_text())
    assert admitted["tasks"] == [task for task, status in zip(tasks, replay) if status == "pass"]
    assert admitted["private_SHA256"] == _sha(owner.run / "private.jsonl")
    assert (owner.run / "private.jsonl").read_bytes() == b"".join(
        raw for path, raw in original.items() if path.name == "private.jsonl")
    assert (owner.run / "public.jsonl").read_bytes() == b"".join(
        raw for path, raw in original.items() if path.name == "public.jsonl")
    assert all(path.read_bytes() == raw for path, raw in original.items())
    for gate, name in (("replay", "replay_receipts.jsonl"), ("validate", "contamination_receipts.jsonl")):
        assert (owner.run / name).read_bytes() == reports[gate]
        archive = next((owner.run / "archives").glob("check-" + gate + "-*.tar.gz"))
        with tarfile.open(archive) as packet:
            assert packet.extractfile(packet.getnames()[0]).read() == reports[gate]
        label = "check-" + gate + "-v02-" + _sha(owner.run / "private.jsonl")[:12]
        status = json.loads((owner.run / "commands" / (label + ".status.json")).read_text())
        assert status["exit_code"] == (1 if gate == "replay" else 0)
    submissions = [event[1] for event in events if event[0] == "submit"]
    assert submissions[0].startswith("check-replay-")
    assert submissions[1].startswith("check-replay-") and submissions[1].endswith("-archive")
    assert submissions[2].startswith("check-validate-")
    private_hash = hashlib.sha256((owner.run / "private.jsonl").read_bytes()).hexdigest()[:12]
    assert submissions[0] == "check-replay-v02-" + private_hash


def test_native_pass_cannot_replace_catalog_pass(tmp_path, monkeypatch):
    owner, tasks, *_ = admission_fixture(tmp_path, monkeypatch,
        ["pass", "pass", "quarantine", "pass"], ["pass", "reject", "pass", "review"])
    owner.check()
    assert json.loads((owner.run / "admitted.json").read_text())["tasks"] == tasks[:1]


def test_all_pass_keeps_existing_admission_and_sampling_contract(tmp_path, monkeypatch):
    owner, tasks, *_ = admission_fixture(tmp_path, monkeypatch, ["pass", "pass"])
    owner.check()
    admitted_bytes = (owner.run / "admitted.json").read_bytes()
    owner.check()
    assert (owner.run / "admitted.json").read_bytes() == admitted_bytes
    admitted = json.loads(admitted_bytes)
    assert admitted == {"tasks": tasks, "private_SHA256": _sha(owner.run / "private.jsonl")}


def test_new_rejected_candidate_keeps_prior_admitted_task_bytes_and_all_raw_rows(tmp_path, monkeypatch):
    owner, tasks, original, reports, _events, _controls = admission_fixture(tmp_path, monkeypatch, ["pass", "pass"])
    owner.check()
    previous_private = (owner.run / "private.jsonl").read_bytes()
    rejected = "wavgen-admissionfixture-6107103"
    directory = owner.run / "candidates" / rejected
    directory.mkdir()
    raw = (json.dumps({"generated_task_id": rejected, "value": 2}) + "\r\n").encode()
    for name in ("private.jsonl", "public.jsonl"):
        (directory / name).write_bytes(raw)
    reports["replay"] += (json.dumps({"generated_task_id": rejected, "status": "quarantine"}) + "\n").encode()
    reports["validate"] += (json.dumps({"generated_task_id": rejected, "status": "pass"}) + "\n").encode()
    owner.check()
    admitted = json.loads((owner.run / "admitted.json").read_text())
    assert admitted["tasks"] == tasks
    assert admitted["private_SHA256"] == _sha(owner.run / "private.jsonl")
    assert (owner.run / "private.jsonl").read_bytes() == previous_private + raw
    assert all(path.read_bytes() == before for path, before in original.items())
    assert len(list((owner.run / "archives").glob("check-*.tar.gz"))) == 4


@pytest.mark.parametrize("previous_policy", ["combined", "split_v01"])
def test_new_gate_labels_preserve_accepted_previous_command_intents(tmp_path, monkeypatch, previous_policy):
    owner, _tasks, _original, _reports, events, _controls = admission_fixture(tmp_path, monkeypatch, ["pass", "reject"])
    private = b"".join(path.read_bytes() for path in sorted((owner.run / "candidates").glob("*/private.jsonl")))
    digest = hashlib.sha256(private).hexdigest()[:12]
    old_label = "check-" + ("replay-" if previous_policy == "split_v01" else "") + digest
    old = {}
    for suffix, value in (("intent", {"command": "original combined replay && validate"}),
                          ("handle", {"command_id": "failed-legacy-handle"}),
                          ("status", {"id": "failed-legacy-handle", "running": False, "exit_code": 1})):
        path = owner.run / "commands" / (old_label + "." + suffix + ".json")
        _atomic_json(path, value)
        old[path] = path.read_bytes()
    owner.check()
    assert all(path.read_bytes() == raw for path, raw in old.items())
    assert not any(event[0] == "submit" and event[1] == old_label for event in events)
    assert any(event[0] == "submit" and event[1] == "check-replay-v02-" + digest for event in events)


def test_zero_pass_still_collects_both_reports_and_fails_clearly(tmp_path, monkeypatch):
    owner, _tasks, _original, reports, *_ = admission_fixture(tmp_path, monkeypatch,
        ["quarantine", "reject"])
    with pytest.raises(ValueError, match="No candidate passed both"):
        owner.check()
    assert (owner.run / "replay_receipts.jsonl").read_bytes() == reports["replay"]
    assert (owner.run / "contamination_receipts.jsonl").read_bytes() == reports["validate"]
    assert not (owner.run / "admitted.json").exists()


@pytest.mark.parametrize("gate", ["replay", "validate"])
@pytest.mark.parametrize("failure", ["duplicate", "missing", "foreign"])
def test_each_gate_must_cover_every_input_id_exactly_once(tmp_path, monkeypatch, gate, failure):
    owner, tasks, _original, reports, *_ = admission_fixture(tmp_path, monkeypatch, ["pass", "pass"])
    ids = {"duplicate": [tasks[0], tasks[0]], "missing": tasks[:1],
           "foreign": [tasks[0], "wavgen-foreign"]}[failure]
    reports[gate] = b"".join((json.dumps({"generated_task_id": task, "status": "pass"}) + "\n").encode()
                             for task in ids)
    with pytest.raises(ValueError, match="exactly once"):
        owner.check()
    name = "replay_receipts.jsonl" if gate == "replay" else "contamination_receipts.jsonl"
    assert (owner.run / name).read_bytes() == reports[gate]
    assert not (owner.run / "admitted.json").exists()


@pytest.mark.parametrize("exit_code", [2, 127, None, False, 0.0])
def test_other_or_unknown_terminal_codes_are_never_candidate_rejections(tmp_path, monkeypatch, exit_code):
    owner, _tasks, _original, _reports, events, controls = admission_fixture(tmp_path, monkeypatch, ["pass"])
    controls["observe"]["replay"] = [{"running": False, "exit_code": exit_code}]
    with pytest.raises(RuntimeError, match="not terminal exit 0/1"):
        owner.check()
    assert not any(event[0] == "transfer" for event in events)
    assert not (owner.run / "admitted.json").exists()
    assert len(list((owner.run / "commands").glob("*.handle.json"))) == 1


@pytest.mark.parametrize("phase", ["submit", "observe", "transfer"])
def test_transport_errors_propagate_without_admission_or_second_gate(tmp_path, monkeypatch, phase):
    owner, _tasks, _original, _reports, events, controls = admission_fixture(tmp_path, monkeypatch, ["pass"])
    error = requests.HTTPError("actual transport failed")
    if phase == "submit":
        controls["submit_error"]["replay"] = error
    elif phase == "observe":
        controls["observe"]["replay"] = [error]
    else:
        controls["transfer_error"] = error
    with pytest.raises(requests.HTTPError) as raised:
        owner.check()
    assert raised.value is error
    assert not any(event[0] == "submit" and event[1].startswith("check-validate-") for event in events)
    assert not (owner.run / "admitted.json").exists()


def test_unknown_submission_is_preserved_and_cannot_be_reposted(tmp_path, monkeypatch):
    owner, _tasks, _original, _reports, _events, controls = admission_fixture(tmp_path, monkeypatch, ["pass"])
    controls["submit_error"]["replay"] = RuntimeError("submission unknown")
    with pytest.raises(RuntimeError, match="submission unknown"):
        owner.check()
    intent = next((owner.run / "commands").glob("*.intent.json"))
    original = intent.read_bytes()
    with pytest.raises(RuntimeError, match="no second POST"):
        owner.check()
    assert intent.read_bytes() == original
    assert not list((owner.run / "commands").glob("*.handle.json"))
    assert not (owner.run / "admitted.json").exists()


def test_running_original_handle_is_observed_until_terminal_or_observation_error(tmp_path, monkeypatch):
    owner, _tasks, _original, _reports, events, controls = admission_fixture(tmp_path, monkeypatch, ["pass"])
    error = requests.Timeout("original still running; observation unavailable")
    controls["observe"]["replay"] = [{"running": True, "exit_code": None}, error]
    with pytest.raises(requests.Timeout) as raised:
        owner.check()
    assert raised.value is error
    handles = [event[2] for event in events if event[0] == "observe"]
    assert len(handles) == 2 and handles[0] == handles[1]
    assert ("sleep", 10) in events
    assert not any(event[0] == "transfer" for event in events)
    assert not (owner.run / "admitted.json").exists()


def test_status_must_belong_to_actual_original_handle(tmp_path, monkeypatch):
    owner, _tasks, _original, _reports, _events, controls = admission_fixture(tmp_path, monkeypatch, ["pass"])
    controls["observe"]["replay"] = [{"id": "foreign-handle", "running": False, "exit_code": 1}]
    with pytest.raises(ValueError, match="original handle"):
        owner.check()
    assert not (owner.run / "admitted.json").exists()


@pytest.mark.parametrize("report_status,exit_code", [("pass", 1), ("reject", 0)])
def test_terminal_code_must_agree_with_complete_report(tmp_path, monkeypatch, report_status, exit_code):
    owner, _tasks, _original, _reports, _events, controls = admission_fixture(tmp_path, monkeypatch, [report_status])
    controls["observe"]["replay"] = [{"running": False, "exit_code": exit_code}]
    with pytest.raises(ValueError, match="differs from its complete receipts"):
        owner.check()
    assert (owner.run / "replay_receipts.jsonl").exists()
    assert not (owner.run / "admitted.json").exists()


def test_unknown_gate_status_is_not_silently_treated_as_rejection(tmp_path, monkeypatch):
    owner, *_ = admission_fixture(tmp_path, monkeypatch, ["unrecognized"])
    with pytest.raises(ValueError, match="unknown status"):
        owner.check()
    assert not (owner.run / "admitted.json").exists()


@pytest.mark.parametrize("change", ["GT", "bytes", "removed"])
def test_existing_private_task_bytes_cannot_be_changed(tmp_path, monkeypatch, change):
    owner, _tasks, original, _reports, events, _controls = admission_fixture(tmp_path, monkeypatch, ["pass"])
    owner.check()
    private = (owner.run / "private.jsonl").read_bytes()
    admitted = (owner.run / "admitted.json").read_bytes()
    path = next(path for path in original if path.name == "private.jsonl")
    if change == "GT":
        path.write_bytes(path.read_bytes().replace(b'"value": 0', b'"value": 42'))
    elif change == "bytes":
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    else:
        path.unlink()
    previous_events = len(events)
    with pytest.raises(ValueError, match="changed|No formally grounded"):
        owner.check()
    assert len(events) == previous_events
    assert (owner.run / "private.jsonl").read_bytes() == private
    assert (owner.run / "admitted.json").read_bytes() == admitted


def test_previously_admitted_membership_cannot_be_replaced_by_new_passes(tmp_path, monkeypatch):
    owner, tasks, _original, reports, _events, _controls = admission_fixture(tmp_path, monkeypatch, ["reject", "pass"])
    _atomic_json(owner.run / "admitted.json", {"tasks": tasks[:1], "private_SHA256": "old-proof"})
    original = (owner.run / "admitted.json").read_bytes()
    with pytest.raises(ValueError, match="previously admitted task"):
        owner.check()
    assert (owner.run / "admitted.json").read_bytes() == original
    assert (owner.run / "replay_receipts.jsonl").read_bytes() == reports["replay"]
