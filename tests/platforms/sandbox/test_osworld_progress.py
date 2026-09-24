"""Read bounded production-shaped artifacts without loading model requests or images."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lite.platforms.sandbox.client import AFSCommandError
from lite.platforms.sandbox.osworld_2 import batch


@pytest.fixture
def progress(monkeypatch):
    clock = SimpleNamespace(now=1000)
    monkeypatch.setattr(batch, "time", SimpleNamespace(time=lambda: clock.now, monotonic=lambda: 0))
    item = {"task_id": "069", "experiment": "batch-task-069", "state": "running", "started_at": 0}
    record = {"release": "test-release", "submission": {"results_prefix": "outputs/test"}}
    run = {"run_id": "run-069", "task_id": "069", "release": "test-release",
           "started_at": "1970-01-01T00:00:00Z", "turns": 3,
           "phase_count": 2, "status": "running"}
    files = {
        "run.json": run,
        "events.jsonl": 'cut fragment\n' + json.dumps({
            "method": "agent_step", "episode_index": 0, "turn": 3,
            "time": "1970-01-01T00:00:20Z",
        }) + '\n{"partial":',
        "model/episode_000/00004.request.json": "MUST NOT READ",
        "model/episode_000/00004.attempt-01.response.json": "MUST NOT READ",
        "model/episode_000/00004.attempt-02.error.json": "MUST NOT READ",
    }
    client = Mock(secrets=["private-key"])

    def afs(*command, timeout=None):
        assert timeout and timeout <= 20
        if command[0] == "lsf":
            if "--dirs-only" in command:
                return "run-069/\n"
            return "\n".join(files)
        name = command[1].split("/run-069/", 1)[1]
        assert name in {"run.json", "events.jsonl"}
        if name == "events.jsonl":
            assert command[2:4] == ("--tail", "65536")
            return files[name]
        return json.dumps(files[name])

    client.afs.side_effect = afs
    adapter = batch.OSWorldBatch(None, {}, client, {"job": {"max_steps": 500}})
    return SimpleNamespace(
        adapter=adapter, item=item, record=record, run=run, files=files, client=client, clock=clock,
    )


def test_pending_request_retries_do_not_count_as_progress(progress):
    p = progress
    p.adapter.observe_progress(p.item, p.record)
    data = p.item["progress"]
    assert data["turns"] == 3 and data["phase_turn"] == 3
    assert data["last_progress_at"] == 20 and data["observed_at"] == 1000
    assert data["activity"] == "waiting_model" and data["retries"] == 2
    p.clock.now += 60
    p.files["model/episode_000/00004.attempt-03.error.json"] = "MUST NOT READ"
    p.adapter.observe_progress(p.item, p.record)
    assert p.item["progress"]["last_progress_at"] == 20
    p.files["model/episode_000/00004.response.json"] = "MUST NOT READ"
    p.adapter.observe_progress(p.item, p.record)
    assert p.item["progress"]["activity"] == "executing"
    assert p.item["progress"]["turns"] == 3


def test_phase_change_keeps_cumulative_turns_but_resets_phase_turn(progress):
    p = progress
    p.files["model/episode_001/00001.request.json"] = "MUST NOT READ"
    p.adapter.observe_progress(p.item, p.record)
    assert p.item["progress"]["phase_index"] == 1
    assert p.item["progress"]["phase_turn"] == 0
    assert p.item["progress"]["turns"] == 3
    p.run["turns"] = 4
    p.files["events.jsonl"] += '\n' + json.dumps({"method": "agent_step",
        "episode_index": 1, "turn": 1, "time": "1970-01-01T00:17:00Z"})
    p.adapter.observe_progress(p.item, p.record)
    assert p.item["progress"]["turns"] == 4
    assert p.item["progress"]["phase_turn"] == 1
    assert p.item["progress"]["last_progress_at"] == 1020


def test_read_error_preserves_last_progress_and_recovery_clears_error(progress):
    p = progress
    p.adapter.observe_progress(p.item, p.record)
    previous = dict(p.item["progress"])
    reader = p.client.afs.side_effect
    p.client.afs.side_effect = AFSCommandError("unavailable private-key", 7)
    p.adapter.observe_progress(p.item, p.record)
    assert p.item["progress"] == previous and p.item["state"] == "running"
    assert p.item["progress_error"] == "unavailable [redacted]"
    p.client.afs.side_effect = reader
    p.adapter.observe_progress(p.item, p.record)
    assert "progress_error" not in p.item
    assert p.item["progress"]["last_progress_at"] == 20


def test_initialization_without_model_or_events_is_not_a_read_failure(progress):
    p = progress
    p.run["turns"] = 0
    p.files.clear()
    p.files["run.json"] = p.run
    p.adapter.observe_progress(p.item, p.record)
    assert "progress_error" not in p.item
    assert p.item["progress"]["activity"] == "starting"
    assert p.item["progress"]["last_progress_at"] == 0


def test_task_identity_mismatch_is_reported_without_using_foreign_progress(progress):
    p = progress
    p.run["task_id"] = "054"
    p.adapter.observe_progress(p.item, p.record)
    assert "progress" not in p.item
    assert "different task" in p.item["progress_error"]
