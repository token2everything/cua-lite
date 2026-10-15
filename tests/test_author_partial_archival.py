"""Preserve terminal author failures through the real host archive boundary."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import map_case_storage
import pytest
from sandbox import _atomic_json

from pipeline import Coordinator

TASK = "wavgen-partial-map-6107991"


def author_fixture(tmp_path, monkeypatch, state="failed", existing=("author", "baselines")):
    owner = Coordinator.__new__(Coordinator)
    owner.run = tmp_path / "run"
    owner.site = "map"
    owner.config = {"helper_SHA256": "b" * 64}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"generated_task_id": TASK}))
    owner.args = SimpleNamespace(run="partialfixture", task=TASK, config=config,
                                 author_module="author_map.py", author_args="[]")
    _atomic_json(owner.run / "infra/author_map.py", {})
    record = {"id": "owned-sandbox", "metadata": {"session_nonce": "owned-nonce"}}
    owner.owned = lambda: record
    outputs = tmp_path / "sandbox/results"
    remote = tmp_path / "sandbox/infra"
    for directory in existing:
        _atomic_json(outputs / directory / TASK / "actual-partial.json", {"source": directory})
    _atomic_json(outputs / "private" / TASK / "state.json", {"credential": "fixture-only"})
    events = []
    error = RuntimeError("ground command failed with exit 35; original handle retained")
    handle = "original-author-handle"
    status = {"id": handle, "running": state == "running",
              "exit_code": None if state in {"running", "unknown_status"} else (0 if state == "success" else 35)}

    def local(path):
        return Path(path.replace("/var/lib/wav-v03/results", str(outputs))
                        .replace("/mnt/results/v03", str(remote)))

    def command(label, body):
        events.append(("command", label, body))
        if label == "ground-" + TASK:
            if state != "unknown_submission":
                _atomic_json(owner.run / "commands" / (label + ".handle.json"), {"command_id": handle})
            if state != "success":
                raise error
            return handle
        body = body.replace("/var/lib/wav-v03/results", str(outputs)).replace("/mnt/results/v03", str(remote))
        subprocess.run(["bash", "-c", body], check=True, capture_output=True, text=True)
        return "artifact-handle-" + label

    def status_call(sandbox, command_id):
        events.append(("status", sandbox, command_id))
        assert sandbox == record["id"] and command_id == handle
        return status

    def download_large(sandbox, path, destination, digest):
        source = local(path)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        shutil.copyfile(source, destination)
        events.append(("transfer", str(destination)))

    def reclaim(coordinator, label):
        receipt = json.loads((owner.run / "archives" / (label + ".receipt.json")).read_text())
        archive = Path(receipt["archive"])
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt["SHA256"]
        with tarfile.open(archive) as packet:
            assert all(not member.name.startswith("private/") for member in packet)
        events.append(("reclaim", label))
        return {"status": "reaped"}

    owner.command = command
    owner.client = SimpleNamespace(upload=lambda *args: None, command_status=status_call,
                                   download=lambda sandbox, path: local(path).read_bytes(),
                                   download_large_file=download_large)
    monkeypatch.setattr(map_case_storage, "reap_collected_map_case", reclaim)
    return owner, events, error, outputs


def test_terminal_failure_collects_existing_outputs_and_reclaims_before_reraising(tmp_path, monkeypatch):
    owner, events, error, outputs = author_fixture(tmp_path, monkeypatch, existing=("author", "author_baselines", "baselines"))
    with pytest.raises(RuntimeError) as raised:
        owner.author(False)
    assert raised.value is error
    failure = json.loads((owner.run / "commands" / ("ground-" + TASK + ".author_failure.json")).read_text())
    assert failure["partial_archive"] == "collected"
    assert failure["existing_output_directories"] == ["author/" + TASK, "author_baselines/" + TASK, "baselines/" + TASK]
    assert failure["actual_command_status"]["exit_code"] == 35
    assert events[-1] == ("reclaim", "ground-" + TASK)
    assert next(i for i, row in enumerate(events) if row[0] == "transfer") < len(events) - 1
    assert (owner.run / "author" / TASK / "actual-partial.json").read_bytes() == (outputs / "author" / TASK / "actual-partial.json").read_bytes()
    assert not (owner.run / "candidates" / TASK).exists()
    assert not (owner.run / "private.jsonl").exists()
    assert not (owner.run / "private" / TASK).exists()
    discovery = next(row for row in events if row[0] == "command" and row[1].endswith("-partial-output-discovery"))
    assert "/private/" not in discovery[2]


@pytest.mark.parametrize("state", ["running", "unknown_status", "unknown_submission"])
def test_unknown_or_running_author_retains_original_error_and_does_not_archive(tmp_path, monkeypatch, state):
    owner, events, error, outputs = author_fixture(tmp_path, monkeypatch, state=state)
    with pytest.raises(RuntimeError) as raised:
        owner.author(False)
    assert raised.value is error
    assert not any(row[0] in {"transfer", "reclaim"} for row in events)
    assert not any(row[0] == "command" and "partial-output-discovery" in row[1] for row in events)
    assert not (owner.run / "archives").exists()
    assert (outputs / "baselines" / TASK / "actual-partial.json").exists()


def test_no_existing_output_does_not_manufacture_an_empty_success_archive(tmp_path, monkeypatch):
    owner, events, error, _outputs = author_fixture(tmp_path, monkeypatch, existing=())
    with pytest.raises(RuntimeError) as raised:
        owner.author(False)
    assert raised.value is error
    assert not (owner.run / "archives").exists()
    failure = json.loads((owner.run / "commands" / ("ground-" + TASK + ".author_failure.json")).read_text())
    assert failure["partial_archive"] == "no_existing_outputs"
    assert failure["existing_output_directories"] == []
    assert not any(row[0] == "reclaim" for row in events)


def test_reclamation_error_is_reported_and_never_hides_original_author_error(tmp_path, monkeypatch):
    owner, _events, error, _outputs = author_fixture(tmp_path, monkeypatch)
    def refuse(coordinator, label):
        raise ValueError("actual case root is mounted; databases retained")
    monkeypatch.setattr(map_case_storage, "reap_collected_map_case", refuse)
    with pytest.raises(RuntimeError) as raised:
        owner.author(False)
    assert raised.value is error
    assert error.__notes__ == ["Partial author archival/reclamation failed: actual case root is mounted; databases retained"]
    failure = json.loads((owner.run / "commands" / ("ground-" + TASK + ".author_failure.json")).read_text())
    assert failure["partial_archive"] == "failed"
    assert failure["archival_error_type"] == "ValueError"
    assert failure["original_error"] == str(error)
    assert (owner.run / "archives" / ("ground-" + TASK + ".receipt.json")).exists()


def test_successful_author_collection_is_synchronous_before_next_case_can_start(tmp_path, monkeypatch):
    owner, events, _error, _outputs = author_fixture(tmp_path, monkeypatch, state="success", existing=("author", "candidates", "author_baselines", "baselines"))
    marker = tmp_path / "first-case-database.bin"
    marker.write_bytes(b"one large owned writable copy")
    def reclaim(coordinator, label):
        assert (owner.run / "archives" / (label + ".receipt.json")).exists()
        marker.unlink()
        events.append(("reclaim", label))
    monkeypatch.setattr(map_case_storage, "reap_collected_map_case", reclaim)
    owner.author(False)
    assert not marker.exists()
    assert events[-1] == ("reclaim", "ground-" + TASK)
    assert not (owner.run / "commands" / ("ground-" + TASK + ".author_failure.json")).exists()


def test_map_explore_collects_its_readonly_reusable_baseline_without_reclaiming(tmp_path, monkeypatch):
    reclaim = map_case_storage.reap_collected_map_case
    owner, events, _error, _outputs = author_fixture(tmp_path, monkeypatch, state="success")
    label = "explore-" + TASK
    monkeypatch.setattr(map_case_storage, "reap_collected_map_case", reclaim)
    owner.collect(label, ["baselines/" + TASK])
    assert not any(row[0] == "reclaim" for row in events)
    assert (owner.run / "baselines" / TASK / "actual-partial.json").exists()
