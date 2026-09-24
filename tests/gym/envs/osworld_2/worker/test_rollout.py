"""Benchmark coordination with local Worker and generic Agent episode fakes.

Provider requests, screenshots, tools and Agent logging are exercised by the
provider and generic episode tests; this file owns task/phase lifecycle behavior.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lite.gym.envs.osworld_2 import rollout
from lite.gym.envs.osworld_2.catalog import load_release, release_names

_MODEL = "test-gui-model"
_WORKER_TOKEN = "test-only-private-worker-key"


@pytest.fixture
def excluded_release(monkeypatch):
    # Exercise exclusion handling independently of live qualification results.
    monkeypatch.setattr(
        rollout,
        "load_release",
        lambda name: replace(load_release(name), exclusions={"007": "Test protocol exclusion"}),
    )


@pytest.fixture
def harness(tmp_path, monkeypatch, excluded_release):
    args = SimpleNamespace(
        release="osworld-v2.1",
        task_set=None,
        plan_only=False,
        skip_unsupported=False,
        verify_protocols=False,
        cleanup_timeout=0.1,
        environment_results_root=None,
        job=tmp_path / "job.json",
        model_profile="gui",
        connection=tmp_path / "connection.json",
        tasks=["054", "055"],
        max_steps=3,
        model_timeout=10,
        history_turns=None,
        full_history=False,
        results_root=tmp_path / "results",
        experiment="rollout-test",
    )
    args.job.write_text(
        json.dumps(
            {
                "model": {
                    "model_id": _MODEL,
                    "api_type": "openai_responses",
                }
            }
        )
    )
    args.connection.write_text(
        json.dumps(
            {
                "endpoint": "http://fake-worker.test",
                "token": _WORKER_TOKEN,
                "worker_id": "fake-worker",
            }
        )
    )
    state = SimpleNamespace(
        args=args,
        modes={},
        phase_modes={},
        phase_scores={},
        gate_phases={},
        simulator_tasks=set(),
        episodes=[],
        events=[],
        metadata_calls=[],
        tasks={},
        initial_traces=[],
        completed_traces=[],
        active_run_id=None,
        last_closed_run_id=None,
        worker_clients=0,
    )

    def metadata(env_id, task_id):
        release = load_release(args.release)
        assert env_id == release.env_id
        state.metadata_calls.append(task_id)
        others = {"release_manifest_sha256": release.manifest_sha256}
        if state.modes.get(task_id) == "excluded":
            others["exclude_reason"] = "Unsupported protocol"
        elif state.modes.get(task_id) == "wrong_manifest":
            others["release_manifest_sha256"] = "different-manifest"
        return SimpleNamespace(others=others)

    def unexpected_make(*args, **kwargs):
        raise AssertionError("The benchmark coordinator must not create an Agent environment")

    monkeypatch.setattr(rollout.gym.registry, "task_metadata", metadata)
    monkeypatch.setattr(rollout.gym, "make", unexpected_make)

    class Worker:
        def __init__(self, endpoint, token):
            assert endpoint == "http://fake-worker.test" and token == _WORKER_TOKEN
            state.worker_clients += 1

        def request(self, method, path):
            assert (method, path) == ("GET", "/healthz")
            task = state.tasks.get(state.active_run_id)
            return {
                "run_id": state.active_run_id,
                "last_closed_run_id": state.last_closed_run_id,
                "state": task.state if task else "idle",
                "busy": False,
                "worker_id": "fake-worker",
                "release": args.release,
            }

        def reset(self, task_id, run_id):
            state.events.append(("worker_reset", task_id))
            assert state.active_run_id is None, "Previous task was not cleaned up"
            if state.modes.get(task_id) == "reset_error":
                raise RuntimeError("Reset could not begin")
            if state.modes.get(task_id) == "reset_private_error":
                raise RuntimeError(f"Reset rejected token {_WORKER_TOKEN}")
            state.tasks[run_id] = SimpleNamespace(
                task_id=task_id,
                phase_index=0,
                state="active",
                evaluation=None,
                scores=state.phase_scores.get(task_id, [1.0]),
            )
            state.active_run_id = run_id
            if state.modes.get(task_id) == "reset_lost":
                raise RuntimeError("Worker reset response was lost")
            return {
                "phase_count": len(state.tasks[run_id].scores),
                "has_user_simulator": task_id in state.simulator_tasks,
            }

        def task(self, run_id):
            task = state.tasks[run_id]
            state.events.append(("task", task.task_id))
            snapshot = {
                "phase_index": task.phase_index,
                "state": task.state,
                "evaluation": task.evaluation,
                "busy": False,
            }
            mode = state.phase_modes.get(
                (task.task_id, task.phase_index), state.modes.get(task.task_id)
            )
            if mode == "busy_evaluation":
                snapshot["busy"] = True
            elif mode == "wrong_phase":
                snapshot["phase_index"] += 1
            elif mode == "unevaluated_state":
                snapshot["state"] = "active"
            elif mode == "missing_evaluation":
                snapshot["evaluation"] = None
            return snapshot

        def advance(self, run_id, *, phase_index):
            task = state.tasks[run_id]
            state.events.append(("advance", task.task_id))
            assert phase_index == task.phase_index
            assert task.state == "phase_evaluated" and not task.evaluation["complete"]
            if state.modes.get(task.task_id) == "advance_error":
                raise RuntimeError("Worker could not advance the phase")
            task.phase_index += 1
            task.state, task.evaluation = "active", None

        def close(self, run_id):
            task = state.tasks[run_id]
            state.events.append(("worker_close", task.task_id))
            assert state.active_run_id == run_id
            if state.modes.get(task.task_id) == "cleanup_error":
                task.state = "broken"
                raise RuntimeError("Worker cleanup failed")
            state.active_run_id = None
            state.last_closed_run_id = run_id

        def wait_closed(self, run_id, worker_id, release, *, timeout):
            task = state.tasks[run_id]
            state.events.append(("receipt", task.task_id))
            assert (worker_id, release, timeout) == (
                "fake-worker",
                args.release,
                args.cleanup_timeout,
            )
            assert state.last_closed_run_id == run_id and state.active_run_id is None
            output = args.results_root / args.experiment / ("task_" + task.task_id) / run_id
            trace = output / "agent_trajectory.jsonl"
            if trace.exists():
                state.completed_traces.append(json.loads(trace.read_text()))
            return {"last_closed_run_id": run_id, "state": "idle", "busy": False}

    monkeypatch.setattr(rollout, "WorkerClient", Worker)

    async def execute_episode(specification, artifacts, details):
        task_id = specification["metadata"]["task_id"]
        phase_index = specification["episode_index"]
        assert specification["env"] == f"{load_release(args.release).env_id}@{task_id}"
        assert specification["env_kwargs"]["run_id"] == state.active_run_id == artifacts.run_id
        task = state.tasks[artifacts.run_id]
        assert specification["env_kwargs"]["phase_index"] == task.phase_index == phase_index
        assert specification["metadata"] == {
            "task_id": task_id,
            "phase_index": phase_index,
            "run_id": artifacts.run_id,
            "release": args.release,
        }
        assert specification["output_dir"] == str(artifacts.path)
        assert "connection_file" not in specification["env_kwargs"]
        assert _WORKER_TOKEN not in json.dumps(specification)
        state.events.append(("execute", task_id))
        episode = SimpleNamespace(
            task_id=task_id,
            phase_index=phase_index,
            run_id=artifacts.run_id,
            output=artifacts.path,
            specification=copy.deepcopy(specification),
            closed=False,
        )
        state.episodes.append(episode)
        trace_path = artifacts.path / "agent_trajectory.jsonl"
        trace = (
            json.loads(trace_path.read_text())
            if trace_path.exists()
            else {
                "messages": [],
                "metadata": {
                    "status": "running",
                    "episodes": [],
                    "tools": [{"name": "sentinel-from-generic-episode"}],
                },
            }
        )
        start = len(trace["messages"])
        trace["messages"].append({"role": "user", "content": f"Task {task_id} phase {phase_index}"})
        boundary = {
            "episode_index": phase_index,
            "context_reset": True,
            "message_start": start,
            "message_end": start + 1,
            "status": "running",
        }
        trace["metadata"]["episodes"].append(boundary)
        trace_path.write_text(json.dumps(trace) + "\n")
        state.initial_traces.append(copy.deepcopy(trace))
        mode = state.phase_modes.get((task_id, phase_index), state.modes.get(task_id))
        try:
            if mode == "cancel":
                boundary["status"] = "interrupted"
                raise asyncio.CancelledError
            if mode == "episode_error":
                boundary["status"] = "failed"
                raise RuntimeError("Agent episode failed")
            if mode in ("returned_failure", "unclosed_episode"):
                boundary["status"] = "failed" if mode == "returned_failure" else "completed"
                return {
                    "status": boundary["status"],
                    "closed": mode == "returned_failure",
                    "error": "Episode handle close failed",
                }
            trace["messages"].append(
                {
                    "role": "assistant",
                    "content": f"phase-{phase_index}-answer",
                }
            )
            boundary.update(status="completed", message_end=len(trace["messages"]))
            if mode != "evaluation_error":
                gate_failed = state.gate_phases.get(task_id) == phase_index
                complete = gate_failed or phase_index == len(task.scores) - 1
                if mode == "incomplete_final_evaluation":
                    complete = False
                task.evaluation = {
                    "phase_index": phase_index,
                    "phase_score": task.scores[phase_index],
                    "score": sum(task.scores[: phase_index + 1]),
                    "complete": complete,
                    "gate_failed": gate_failed,
                }
                task.state = "evaluated" if complete else "phase_evaluated"
            details["turns"] = details.get("turns", 0) + 1
            artifacts.update(**details)
            return {
                "status": "completed",
                "closed": True,
                "turns": 1,
                "message_start": start,
                "message_end": len(trace["messages"]),
                "terminated": mode != "truncated",
                "truncated": mode == "truncated",
            }
        finally:
            episode.closed = mode != "unclosed_episode"
            state.events.append(
                (
                    "episode_close" if episode.closed else "episode_close_failed",
                    task_id,
                )
            )
            trace_path.write_text(json.dumps(trace) + "\n")

    state.execute_episode = execute_episode
    return state


@pytest.mark.parametrize("release", release_names())
async def test_tasks_reset_once_sequentially_and_use_one_run_directory(harness, release):
    harness.args.release = release
    summaries = await rollout.run(harness.args, harness.execute_episode)
    assert [item["task_id"] for item in summaries] == ["054", "055"]
    assert harness.events == [
        (operation, task)
        for task in ("054", "055")
        for operation in (
            "worker_reset",
            "execute",
            "episode_close",
            "task",
            "worker_close",
            "receipt",
        )
    ]
    assert harness.worker_clients == 1 and harness.active_run_id is None
    for episode, summary, before_final in zip(
        harness.episodes,
        summaries,
        harness.completed_traces,
        strict=True,
    ):
        assert episode.closed and episode.run_id == summary["run_id"]
        assert episode.output.parent.name == "task_" + summary["task_id"]
        assert list(episode.output.parent.iterdir()) == [episode.output]
        assert summary["status"] == "completed" and summary["closed"] is True
        assert summary["model_id"] == _MODEL and summary["api_type"] == "openai_responses"
        assert summary["release"] == release and summary["turns"] == 1
        assert summary["score"] == 1.0
        assert episode.specification["max_steps"] == 3
        assert episode.specification["model_timeout"] == 10
        assert episode.specification["model_profile"] == "gui"
        assert "history_turns" not in episode.specification
        trace_path = episode.output / "agent_trajectory.jsonl"
        assert len(trace_path.read_text().splitlines()) == 1
        trace = json.loads(trace_path.read_text())
        assert trace["messages"] == before_final["messages"]
        assert trace["metadata"]["tools"] == before_final["metadata"]["tools"]
        assert trace["metadata"]["episodes"] == before_final["metadata"]["episodes"]
        assert trace["metadata"]["status"] == "completed"
        assert trace["metadata"]["environment_closed"] is True
        assert trace["metadata"]["run_id"] == summary["run_id"]
        assert trace["metadata"]["release"] == release
        assert trace["metadata"]["phases"] == summary["phases"]


async def test_multiple_phases_share_task_identity_and_keep_native_cumulative_score(harness):
    harness.args.tasks = ["054"]
    harness.phase_scores["054"] = [0.2, 0.3]
    summaries = await rollout.run(harness.args, harness.execute_episode)
    assert harness.events == [
        (operation, "054")
        for operation in (
            "worker_reset",
            "execute",
            "episode_close",
            "task",
            "advance",
            "execute",
            "episode_close",
            "task",
            "worker_close",
            "receipt",
        )
    ]
    (summary,) = summaries
    first, second = harness.episodes
    assert first is not second and first.closed and second.closed
    assert first.output == second.output and first.run_id == second.run_id == summary["run_id"]
    assert [episode.specification["episode_index"] for episode in harness.episodes] == [0, 1]
    assert summary["score"] == 0.5 and summary["turns"] == 2
    assert [phase["score"] for phase in summary["phases"]] == [0.2, 0.3]
    assert all(phase["context_reset"] for phase in summary["phases"])
    trace = json.loads((first.output / "agent_trajectory.jsonl").read_text())
    assert trace["metadata"]["phases"] == summary["phases"]
    assert [(p["message_start"], p["message_end"]) for p in summary["phases"]] == [(0, 2), (2, 4)]
    assert [(p["message_start"], p["message_end"]) for p in trace["metadata"]["episodes"]] == [
        (0, 2),
        (2, 4),
    ]
    assert len(trace["messages"]) == 4
    assert json.loads((first.output / "result.json").read_text()) == {
        "phase_index": 1,
        "phase_score": 0.3,
        "score": 0.5,
        "complete": True,
        "gate_failed": False,
    }


async def test_gate_failure_completes_task_without_advancing(harness):
    harness.args.tasks = ["054"]
    harness.phase_scores["054"] = [0.0, 0.8]
    harness.gate_phases["054"] = 0
    (summary,) = await rollout.run(harness.args, harness.execute_episode)
    assert summary["status"] == "completed" and summary["score"] == 0.0
    assert len(summary["phases"]) == len(harness.episodes) == 1
    assert ("advance", "054") not in harness.events
    assert harness.events.count(("worker_close", "054")) == 1
    result = json.loads((harness.episodes[0].output / "result.json").read_text())
    assert result["complete"] is True and result["gate_failed"] is True


@pytest.mark.parametrize("phase_index", [0, 1])
@pytest.mark.parametrize("mode", ["episode_error", "cancel"])
async def test_phase_failure_retains_trace_and_closes_before_stopping_batch(
    harness,
    phase_index,
    mode,
):
    harness.phase_scores["054"] = [0.2, 0.3]
    harness.phase_modes[("054", phase_index)] = mode
    exception = asyncio.CancelledError if mode == "cancel" else RuntimeError
    with pytest.raises(exception):
        await rollout.run(harness.args, harness.execute_episode)
    assert len(harness.episodes) == phase_index + 1
    assert all(episode.closed for episode in harness.episodes)
    assert harness.events.count(("advance", "054")) == phase_index
    assert harness.events.count(("worker_close", "054")) == 1
    assert harness.events[-1] == ("receipt", "054")
    output = harness.episodes[0].output
    record = json.loads((output / "run.json").read_text())
    expected_status = "interrupted" if mode == "cancel" else "failed"
    assert record["status"] == expected_status and record["closed"] is True
    assert not (output.parent.parent / "task_055").exists()
    trace_path = output / "agent_trajectory.jsonl"
    assert len(trace_path.read_text().splitlines()) == 1
    trace = json.loads(trace_path.read_text())
    episodes = trace["metadata"]["episodes"]
    assert [episode["episode_index"] for episode in episodes] == list(range(phase_index + 1))
    assert episodes[0]["message_start"] == 0
    assert episodes[-1]["message_end"] == len(trace["messages"])
    assert episodes[-1]["status"] == expected_status
    assert trace["metadata"]["status"] == expected_status
    assert trace["metadata"]["environment_closed"] is True
    assert len(trace["metadata"]["phases"]) == phase_index
    if phase_index:
        assert record["score"] == 0.2
        assert trace["metadata"]["phases"] == record["phases"]
        assert trace["messages"][:2] == harness.initial_traces[1]["messages"][:2]
    else:
        assert "score" not in record
        assert trace["metadata"]["phases"] == []


@pytest.mark.parametrize(
    "mode",
    [
        "evaluation_error",
        "busy_evaluation",
        "wrong_phase",
        "unevaluated_state",
        "missing_evaluation",
    ],
)
async def test_authoritative_worker_evaluation_is_required_before_advance(harness, mode):
    harness.phase_scores["054"] = [0.2, 0.3]
    harness.modes["054"] = mode
    with pytest.raises(RuntimeError, match="confirm.*evaluation"):
        await rollout.run(harness.args, harness.execute_episode)
    assert len(harness.episodes) == 1 and harness.episodes[0].closed
    assert ("advance", "054") not in harness.events
    assert harness.events[-2:] == [("worker_close", "054"), ("receipt", "054")]
    record = json.loads((harness.episodes[0].output / "run.json").read_text())
    assert record["status"] == "failed" and record["closed"] is True
    assert "score" not in record


@pytest.mark.parametrize("mode", ["returned_failure", "unclosed_episode"])
async def test_failed_or_unclosed_episode_does_not_claim_worker_evaluation(harness, mode):
    harness.modes["054"] = mode
    with pytest.raises(RuntimeError, match="Episode handle close failed"):
        await rollout.run(harness.args, harness.execute_episode)
    assert len(harness.episodes) == 1
    assert ("task", "054") not in harness.events and ("advance", "054") not in harness.events
    assert harness.events[-2:] == [("worker_close", "054"), ("receipt", "054")]
    record = json.loads((harness.episodes[0].output / "run.json").read_text())
    assert record["status"] == "failed" and record["closed"] is True
    assert record["error"] == "Episode handle close failed" and "score" not in record
    assert not (harness.episodes[0].output.parent.parent / "task_055").exists()


async def test_truncated_episode_keeps_stop_flags_and_native_score(harness):
    harness.args.tasks = ["054"]
    harness.phase_scores["054"] = [0.25]
    harness.modes["054"] = "truncated"
    (summary,) = await rollout.run(harness.args, harness.execute_episode)
    assert summary["status"] == "completed" and summary["score"] == 0.25
    assert summary["terminated"] is False and summary["truncated"] is True
    assert summary["phases"][0]["terminated"] is False
    assert summary["phases"][0]["truncated"] is True


async def test_failed_advance_keeps_completed_phase_and_closes_task(harness):
    harness.phase_scores["054"] = [0.2, 0.3]
    harness.modes["054"] = "advance_error"
    with pytest.raises(RuntimeError, match="could not advance"):
        await rollout.run(harness.args, harness.execute_episode)
    assert len(harness.episodes) == 1 and harness.episodes[0].closed
    assert harness.events.count(("advance", "054")) == 1
    assert harness.events.count(("worker_close", "054")) == 1
    record = json.loads((harness.episodes[0].output / "run.json").read_text())
    assert record["status"] == "failed" and record["closed"] is True
    assert record["score"] == 0.2 and len(record["phases"]) == 1


async def test_exhausted_phases_do_not_claim_task_completion(harness):
    harness.modes["054"] = "incomplete_final_evaluation"
    with pytest.raises(RuntimeError, match="exhausted phases"):
        await rollout.run(harness.args, harness.execute_episode)
    assert len(harness.episodes) == 1
    assert harness.events[-2:] == [("worker_close", "054"), ("receipt", "054")]
    record = json.loads((harness.episodes[0].output / "run.json").read_text())
    assert record["status"] == "failed" and record["closed"] is True


async def test_lost_reset_response_still_closes_accepted_task(harness):
    harness.modes["054"] = "reset_lost"
    with pytest.raises(RuntimeError, match="reset response was lost"):
        await rollout.run(harness.args, harness.execute_episode)
    assert harness.episodes == []
    assert harness.events == [("worker_reset", "054"), ("worker_close", "054"), ("receipt", "054")]
    (record_path,) = (harness.args.results_root / harness.args.experiment).glob(
        "task_054/*/run.json"
    )
    record = json.loads(record_path.read_text())
    assert record["status"] == "failed" and record["closed"] is True
    assert "score" not in record


@pytest.mark.parametrize("operation", ["reset", "advance"])
async def test_cancellation_waits_for_pending_worker_operation_before_cleanup(
    harness,
    monkeypatch,
    operation,
):
    harness.phase_scores["054"] = [0.2, 0.3]
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    original = getattr(rollout.WorkerClient, operation)

    def delayed(self, *args, **kwargs):
        entered.set()
        assert release.wait(5), "Test did not release the Worker operation"
        try:
            return original(self, *args, **kwargs)
        finally:
            completed.set()

    monkeypatch.setattr(rollout.WorkerClient, operation, delayed)
    task = asyncio.create_task(rollout.run(harness.args, harness.execute_episode))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        event_name = "worker_reset" if operation == "reset" else "advance"
        assert (event_name, "054") not in harness.events
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=0.05)
        assert not done and ("worker_close", "054") not in harness.events
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert completed.is_set()
    finally:
        release.set()
        if not task.done():
            await asyncio.gather(task, return_exceptions=True)
    assert harness.active_run_id is None
    assert harness.events.count(("worker_reset", "054")) == 1
    assert harness.events.count(("worker_close", "054")) == 1
    assert harness.events.count(("receipt", "054")) == 1
    assert len(harness.episodes) == (1 if operation == "advance" else 0)
    assert harness.events.count(("advance", "054")) == (1 if operation == "advance" else 0)
    (record_path,) = (harness.args.results_root / harness.args.experiment).glob(
        "task_054/*/run.json"
    )
    record = json.loads(record_path.read_text())
    assert record["status"] == "interrupted" and record["closed"] is True
    if operation == "advance":
        assert record["score"] == 0.2
    assert not (record_path.parent.parent.parent / "task_055").exists()


async def test_cancellation_settles_executor_handles_before_closing_worker(harness):
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def delayed_executor(specification, artifacts, details):
        assert specification["env_kwargs"]["run_id"] == harness.active_run_id
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
            harness.events.append(("executor_settled", "054"))
            raise

    task = asyncio.create_task(rollout.run(harness.args, delayed_executor))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        await asyncio.wait_for(cancelled.wait(), timeout=5)
        done, _ = await asyncio.wait({task}, timeout=0.05)
        assert not done and harness.events == [("worker_reset", "054")]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    finally:
        release.set()
        if not task.done():
            await asyncio.gather(task, return_exceptions=True)
    assert harness.events == [
        ("worker_reset", "054"),
        ("executor_settled", "054"),
        ("worker_close", "054"),
        ("receipt", "054"),
    ]
    assert harness.active_run_id is None
    (record_path,) = (harness.args.results_root / harness.args.experiment).glob(
        "task_054/*/run.json"
    )
    record = json.loads(record_path.read_text())
    assert record["status"] == "interrupted" and record["closed"] is True
    assert "score" not in record
    assert not (record_path.parent.parent.parent / "task_055").exists()


@pytest.mark.parametrize(
    "history_turns,full_history,expected",
    [
        (None, False, 4),
        (0, False, 0),
        (2, False, 2),
        (None, True, None),
    ],
)
async def test_chat_screenshot_history_is_forwarded_and_recorded(
    harness,
    history_turns,
    full_history,
    expected,
):
    harness.args.job.write_text(
        json.dumps(
            {
                "model": {"model_id": _MODEL, "api_type": "openai_chat"},
            }
        )
    )
    harness.args.history_turns, harness.args.full_history = history_turns, full_history
    summaries = await rollout.run(harness.args, harness.execute_episode)
    assert [episode.specification["history_turns"] for episode in harness.episodes] == [
        expected,
        expected,
    ]
    assert [summary["history_turns"] for summary in summaries] == [expected, expected]
    assert [summary["history_scope"] for summary in summaries] == ["screenshots", "screenshots"]


@pytest.mark.parametrize("history_turns,full_history", [(0, False), (None, True)])
async def test_responses_rejects_chat_history_flags_before_worker_access(
    harness,
    history_turns,
    full_history,
):
    harness.args.history_turns, harness.args.full_history = history_turns, full_history
    with pytest.raises(ValueError, match="Chat gateway only"):
        await rollout.run(harness.args, harness.execute_episode)
    assert harness.worker_clients == 0 and harness.events == []


async def test_worker_cleanup_failure_keeps_score_and_stops_batch(harness):
    harness.modes["054"] = "cleanup_error"
    with pytest.raises(RuntimeError, match="Worker cleanup failed"):
        await rollout.run(harness.args, harness.execute_episode)
    assert len(harness.episodes) == 1 and harness.episodes[0].closed
    assert harness.events[-1] == ("worker_close", "054")
    record = json.loads((harness.episodes[0].output / "run.json").read_text())
    trace = json.loads((harness.episodes[0].output / "agent_trajectory.jsonl").read_text())
    assert record["status"] == trace["metadata"]["status"] == "failed"
    assert record["closed"] is trace["metadata"]["environment_closed"] is False
    assert record["score"] == 1.0
    assert not (harness.episodes[0].output.parent.parent / "task_055").exists()


async def test_reset_failure_before_acceptance_records_failure_without_unowned_close(harness):
    harness.modes["054"] = "reset_error"
    with pytest.raises(RuntimeError, match="Reset could not begin"):
        await rollout.run(harness.args, harness.execute_episode)
    assert harness.episodes == [] and harness.events == [("worker_reset", "054")]
    (record_path,) = (harness.args.results_root / harness.args.experiment).glob(
        "task_054/*/run.json"
    )
    record = json.loads(record_path.read_text())
    assert record["status"] == "failed" and record["closed"] is False
    assert "score" not in record
    assert not (record_path.parent.parent.parent / "task_055").exists()


async def test_worker_credentials_are_redacted_from_errors_and_artifacts(harness, capsys):
    harness.modes["054"] = "reset_private_error"
    with pytest.raises(RuntimeError) as error:
        await rollout.run(harness.args, harness.execute_episode)
    assert _WORKER_TOKEN not in str(error.value) and "[redacted]" in str(error.value)
    assert _WORKER_TOKEN not in capsys.readouterr().out
    for path in (harness.args.results_root / harness.args.experiment).rglob("*.json"):
        assert _WORKER_TOKEN not in path.read_text()


async def test_episode_spec_uses_server_log_path_without_worker_admin_credentials(harness):
    harness.args.environment_results_root = "/env-server/results"
    summaries = await rollout.run(harness.args, harness.execute_episode)
    for episode, summary in zip(harness.episodes, summaries, strict=True):
        assert episode.specification["env_kwargs"] == {
            "max_steps": 3,
            "experiment": "rollout-test",
            "run_id": summary["run_id"],
            "phase_index": 0,
            "results_root": "/env-server/results",
            "extra_tools": ["terminate", "response", "report_infeasible"],
        }
        assert str(harness.args.connection) not in json.dumps(episode.specification)
        assert _WORKER_TOKEN not in json.dumps(episode.specification)


@pytest.mark.parametrize(
    "mode,error",
    [
        ("excluded", "Task 055: Unsupported protocol"),
        ("wrong_manifest", "release manifest differs"),
    ],
)
async def test_all_task_metadata_are_checked_before_first_reset(harness, mode, error):
    harness.modes["055"] = mode
    with pytest.raises(ValueError, match=error):
        await rollout.run(harness.args, harness.execute_episode)
    assert harness.metadata_calls == ["054", "055"]
    assert harness.episodes == [] and harness.events == [] and harness.worker_clients == 0


async def test_execution_requires_sandbox_episode_executor(harness):
    with pytest.raises(ValueError, match="Sandbox coordinator"):
        await rollout.run(harness.args)
    assert harness.worker_clients == 0 and harness.events == []


async def test_offline_plan_needs_no_model_worker_or_remote_server(monkeypatch, excluded_release):
    def unexpected(*args, **kwargs):
        raise AssertionError("Planning must not contact an Agent, Worker or env-server")

    monkeypatch.setattr(rollout, "WorkerClient", unexpected)
    monkeypatch.setattr(rollout.gym.registry, "task_metadata", unexpected)
    args = rollout._parse_args(["--task-set", "offline", "--plan-only"])
    plan = await rollout.run(args)
    assert plan["requested_count"] == 82
    assert len(plan["scheduled_task_ids"]) == 81
    assert [item["task_id"] for item in plan["unsupported"]] == ["007"]
    assert plan["skipped"] == []


@pytest.mark.parametrize("selection", [[], ["--task-set", "full"], ["--task-set", "offline"]])
async def test_protocol_qualification_requires_explicit_task_ids(selection):
    args = rollout._parse_args([*selection, "--verify-protocols", "--plan-only"])
    with pytest.raises(ValueError, match="explicit --tasks"):
        await rollout.run(args)


async def test_protocol_qualification_cannot_skip_unsupported_tasks(harness):
    harness.args.tasks = ["007"]
    harness.args.verify_protocols = harness.args.skip_unsupported = True
    with pytest.raises(ValueError, match="qualification or skipping"):
        await rollout.run(harness.args, harness.execute_episode)
    assert harness.metadata_calls == [] and harness.events == []
    assert not harness.args.results_root.exists()


async def test_protocol_qualification_plan_retains_exclusions_without_skipping(excluded_release):
    args = rollout._parse_args(["--tasks", "007", "054", "--verify-protocols", "--plan-only"])
    plan = await rollout.run(args)
    assert plan["verification_mode"] is True
    assert plan["scheduled_task_ids"] == plan["requested_task_ids"] == ["007", "054"]
    assert [item["task_id"] for item in plan["unsupported"]] == ["007"]
    assert plan["skipped"] == []


async def test_protocol_qualification_binds_user_tool_only_when_worker_declares_it(harness):
    harness.args.tasks = ["007", "054"]
    harness.args.verify_protocols = True
    harness.modes["007"] = "excluded"
    harness.simulator_tasks.add("007")
    summaries = await rollout.run(harness.args, harness.execute_episode)
    assert [record["task_id"] for record in summaries] == ["007", "054"]
    assert [episode.specification["env_kwargs"]["extra_tools"] for episode in harness.episodes] == [
        ["terminate", "response", "report_infeasible", "ask_user"],
        ["terminate", "response", "report_infeasible"],
    ]
    assert [record["has_user_simulator"] for record in summaries] == [True, False]
    record = json.loads(
        (harness.args.results_root / harness.args.experiment / "experiment.json").read_text()
    )
    assert record["config"]["selection"]["verification_mode"] is True
    assert record["config"]["selection"]["skipped"] == []


async def test_protocol_exclusions_require_explicit_partial_run_opt_in(harness):
    harness.args.tasks = ["007", "054"]
    with pytest.raises(ValueError, match="--skip-unsupported"):
        await rollout.run(harness.args, harness.execute_episode)
    assert harness.metadata_calls == [] and harness.episodes == []
    harness.args.skip_unsupported = True
    summaries = await rollout.run(harness.args, harness.execute_episode)
    assert [summary["task_id"] for summary in summaries] == ["054"]
    root = harness.args.results_root / harness.args.experiment
    config = json.loads((root / "experiment.json").read_text())["config"]
    assert config["selection"]["requested_task_ids"] == ["007", "054"]
    coverage = json.loads((root / "summary.json").read_text())["coverage"]
    assert coverage["requested_count"] == 2 and coverage["completed_count"] == 1
    assert coverage["skipped"][0]["task_id"] == "007"
    assert coverage["complete_requested_set"] is False


async def test_fully_skipped_selection_preserves_coverage_without_creating_runs(harness):
    harness.args.tasks = ["007"]
    harness.args.skip_unsupported = True
    assert await rollout.run(harness.args, harness.execute_episode) == []
    assert harness.episodes == [] and harness.events == []
    root = harness.args.results_root / harness.args.experiment
    summary = json.loads((root / "summary.json").read_text())
    assert summary["runs"] == []
    assert summary["coverage"]["requested_count"] == 1
    assert summary["coverage"]["scheduled_count"] == 0
    assert summary["coverage"]["skipped"][0]["task_id"] == "007"
    assert summary["coverage"]["complete_requested_set"] is False


async def test_close_receipt_failure_retains_score_but_stops_batch(harness, monkeypatch):
    calls = []

    def fail(self, run_id, *args, **kwargs):
        calls.append(run_id)
        record = json.loads((harness.episodes[0].output / "run.json").read_text())
        assert record["status"] == "running" and record["closed"] is False
        assert record["score"] == 1.0
        raise TimeoutError("No matching cleanup receipt")

    monkeypatch.setattr(rollout.WorkerClient, "wait_closed", fail)
    with pytest.raises(RuntimeError, match="cleanup receipt"):
        await rollout.run(harness.args, harness.execute_episode)
    assert calls == [harness.episodes[0].run_id] and len(harness.episodes) == 1
    summary = json.loads((harness.episodes[0].output / "run.json").read_text())
    assert summary["status"] == "failed" and summary["closed"] is False
    assert summary["score"] == 1.0
    assert not (harness.episodes[0].output.parent.parent / "task_055").exists()


def test_release_selection_does_not_infer_versions_from_shared_images():
    august = load_release("osworld-v2-2026.08.08")
    assert august.select(tasks=["054"]) == ["054"]
    assert august.env_id == "osworld_2.v2026_08_08"
    with pytest.raises(ValueError, match="Unknown task set"):
        august.select(task_set="offline")
    release = load_release()
    with pytest.raises(ValueError, match="not both"):
        release.select(task_set="full", tasks=["054"])
    with pytest.raises(ValueError, match="duplicates"):
        release.select(tasks=["054", "054"])
    with pytest.raises(ValueError, match="outside"):
        release.select(tasks=["109"])


async def test_june_full_plan_keeps_pending_protocols_and_all_task_ids():
    args = rollout._parse_args(
        [
            "--release",
            "osworld-v2-2026.06.24",
            "--task-set",
            "full",
            "--plan-only",
        ]
    )
    plan = await rollout.run(args)
    assert plan["requested_task_ids"] == list(load_release(args.release).tasks)
    assert plan["requested_count"] == 108 and len(plan["scheduled_task_ids"]) == 100
    assert {item["task_id"] for item in plan["unsupported"]} == {
        "007",
        "023",
        "024",
        "026",
        "034",
        "069",
        "095",
        "098",
    }
    assert plan["skipped"] == []


async def test_june_protocol_qualification_plan_does_not_claim_prior_verification():
    tasks = ["007", "023", "024", "026", "034", "054", "069", "095", "098"]
    args = rollout._parse_args(
        [
            "--release",
            "osworld-v2-2026.06.24",
            "--tasks",
            *tasks,
            "--verify-protocols",
            "--plan-only",
        ]
    )
    plan = await rollout.run(args)
    assert plan["verification_mode"] is True
    assert plan["scheduled_task_ids"] == plan["requested_task_ids"] == tasks
    assert {item["task_id"] for item in plan["unsupported"]} == set(tasks) - {"054"}
    assert plan["skipped"] == []
