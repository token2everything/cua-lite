"""Live model traces retain complete feedback and portable images in one row."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from agents.models.gpt._support import _fake_response, _FakeEnv, _stub_gpt_echoed_dim_fetch
from PIL import Image

from lite.agents.core.agent.jsonl_logger import JsonlTrajectoryLogger, project_viewer_messages
from lite.agents.models.gpt.agent import GPTDesktopUseAgent
from lite.core import LiteCUAMetadata, LiteSample


def test_atomic_trace_preserves_model_text_and_relative_images(tmp_path):
    sample = LiteSample(
        metadata=LiteCUAMetadata(),
        images=[Image.new("RGB", (20, 10))],
        messages=[
            {"role": "user", "content": [{"type": "image", "index": 0}]},
            {"role": "assistant", "content": "model says private-key"},
        ],
    )
    original = copy.deepcopy(sample.messages)
    logger = JsonlTrajectoryLogger(tmp_path, model_id="test-model", secrets=["private-key"])
    logger.on_trajectory(sample)
    for turn in range(2):
        sample.messages.append({"role": "assistant", "content": f"turn {turn}"})
        logger.on_trajectory(sample)
        lines = (tmp_path / "agent_trajectory.jsonl").read_text().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert len(row["messages"]) == turn + 3
        assert "private-key" not in lines[0]
        image = row["messages"][0]["content"][0]["image"]
        assert image["image_wh"] == [20, 10]
        assert (tmp_path / image["url"]).is_file()
    assert sample.messages[:2] == original
    logger.on_complete(None)
    assert (
        json.loads((tmp_path / "agent_trajectory.jsonl").read_text())["metadata"]["status"]
        == "failed"
    )


def test_viewer_projection_preserves_annotations_reasoning_and_tool_pairing(tmp_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Task instruction"},
                {"type": "metadata", "data": {"benchmark_release": "osworld-v2.1"}},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "inline_reasoning", "text": "Supplied summary."},
                {"type": "text", "text": "Next action."},
                {"type": "action_description", "text": "Screenshot"},
                {"type": "history_summary", "text": "Task history"},
            ],
            "reasoning_content": "Supplied provider reasoning.",
            "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {
                    "name": "computer", "arguments": {"actions": [{"action": "screenshot"}]},
                },
            }],
        },
        {
            "role": "tool", "tool_call_id": "call-1",
            "content": [
                {"type": "image", "index": 0},
                {"type": "metadata", "data": {"is_error": False}},
            ],
        },
    ]
    original = copy.deepcopy(messages)
    sample = LiteSample(
        metadata=LiteCUAMetadata(), images=[Image.new("RGB", (20, 10))], messages=messages,
    )
    logger = JsonlTrajectoryLogger(tmp_path, model_id="test", absolute_image_paths=True)
    logger.on_trajectory(sample)
    row = json.loads((tmp_path / "agent_trajectory.jsonl").read_text())
    exported = row["messages"]
    assert sample.messages == original
    assert [m["role"] for m in exported] == ["user", "assistant", "tool"]
    for source, target in zip(original, exported, strict=True):
        assert {p["type"] for p in target["content"]} <= {"text", "image"}
        assert target["content_annotations"] == [
            {"index": index, "content": part}
            for index, part in enumerate(source["content"])
            if part["type"] not in {"text", "image"}
        ]
    image_path = Path(exported[0]["content"][0]["image"]["url"])
    assert image_path.parent == tmp_path / "images"
    assert image_path.name == f"{hashlib.sha256(image_path.read_bytes()).hexdigest()}.png"
    assert exported[1]["reasoning_content"] == "Supplied provider reasoning."
    assert exported[1]["tool_calls"] == original[1]["tool_calls"]
    assert exported[2]["tool_call_id"] == "call-1"
    assert project_viewer_messages(exported) == exported


def test_duplicate_images_share_files_without_removing_trajectory_positions(tmp_path):
    sample = LiteSample(
        metadata=LiteCUAMetadata(),
        images=[Image.new("RGB", (8, 6), "blue")],
        messages=[{"role": "user", "content": [{"type": "image", "index": 0}]}],
    )
    logger = JsonlTrajectoryLogger(tmp_path, model_id="test")
    logger.on_trajectory(sample)
    first_path = next((tmp_path / "images").glob("*.png"))
    initial_stat = first_path.stat()
    for image in (sample.images[0].copy(), Image.new("RGB", (8, 6), "red")):
        index = len(sample.images)
        sample.images.append(image)
        sample.messages.append({"role": "user", "content": [{"type": "image", "index": index}]})
        logger.on_trajectory(sample)

    row = json.loads((tmp_path / "agent_trajectory.jsonl").read_text())
    refs = [message["content"][0]["image"]["url"] for message in row["messages"]]
    assert len(sample.images) == len(row["messages"]) == 3
    assert [m["content"][0]["index"] for m in sample.messages] == [0, 1, 2]
    assert refs[0] == refs[1] and refs[2] != refs[0]
    assert len(list((tmp_path / "images").glob("*.png"))) == 2
    assert first_path.stat().st_ino == initial_stat.st_ino
    assert first_path.stat().st_mtime_ns == initial_stat.st_mtime_ns
    for image, ref in zip(sample.images, refs, strict=True):
        path = tmp_path / ref
        assert path.name == f"{hashlib.sha256(path.read_bytes()).hexdigest()}.png"
        with Image.open(path) as stored:
            assert stored.size == image.size and stored.tobytes() == image.tobytes()


@pytest.mark.asyncio
async def test_responses_loop_publishes_feedback_before_requesting_next_turn(tmp_path, monkeypatch):
    _stub_gpt_echoed_dim_fetch(monkeypatch)
    snapshots = []
    logger = JsonlTrajectoryLogger(tmp_path, model_id="test-model")
    responses = [
        _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call-1",
                    "actions": [{"type": "click", "x": 20, "y": 30}],
                }
            ]
        ),
        _fake_response(),
    ]

    async def respond(**kwargs):
        path = tmp_path / "agent_trajectory.jsonl"
        if path.exists():
            snapshots.append(json.loads(path.read_text()))
        return responses.pop(0)

    monkeypatch.setattr("litellm.aresponses", AsyncMock(side_effect=respond))
    env = _FakeEnv(terminate_after=3)
    result = await GPTDesktopUseAgent().sample(env, max_steps=3, hooks=[logger])
    assert env.closed
    assert len(snapshots) == 2
    assert [m["role"] for m in snapshots[0]["messages"]] == ["user"]
    assert [m["role"] for m in snapshots[1]["messages"]] == ["user", "assistant", "tool"]
    assert snapshots[1]["messages"][-1]["content"][0]["type"] == "image"
    row = json.loads((tmp_path / "agent_trajectory.jsonl").read_text())
    assert row["metadata"]["status"] == "completed"
    assert row["metadata"]["episode_return"] == result.episode_return
    assert sum(message["role"] == "user" for message in row["messages"]) == 1


@pytest.mark.asyncio
async def test_failed_environment_step_keeps_last_model_turn_without_inventing_feedback(
    tmp_path,
    monkeypatch,
):
    _stub_gpt_echoed_dim_fetch(monkeypatch)
    responses = [
        _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "first",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        ),
        _fake_response(
            [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Last reasoning."}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Last action intent."}],
                },
                {"type": "computer_call", "call_id": "last", "actions": [{"type": "screenshot"}]},
            ]
        ),
    ]
    model = AsyncMock(side_effect=responses)
    monkeypatch.setattr("litellm.aresponses", model)

    class StepFails(_FakeEnv):
        async def step(self, actions):
            if self._step_count == 1:
                raise RuntimeError("Worker action failed")
            return await super().step(actions)

    env = StepFails(terminate_after=3)
    with pytest.raises(RuntimeError, match="Worker action failed"):
        await GPTDesktopUseAgent().sample(
            env,
            max_steps=3,
            hooks=[JsonlTrajectoryLogger(tmp_path, model_id="test-model")],
        )
    assert env.closed and model.await_count == 2
    lines = (tmp_path / "agent_trajectory.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["metadata"]["status"] == "failed"
    assert [message["role"] for message in row["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    last = row["messages"][-1]
    assert last["tool_calls"]
    assert {"type": "text", "text": "Last action intent."} in last["content"]
    assert last["reasoning_content"] == "Last reasoning."
    assert any(
        item["content"] == {"type": "inline_reasoning", "text": "Last reasoning."}
        for item in last["content_annotations"]
    )
    for message in row["messages"]:
        for part in message.get("content", []):
            if part["type"] == "image":
                assert (tmp_path / part["image"]["url"]).is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_first_request_failure_leaves_initial_trace_and_closes_environment(
    cancelled,
    tmp_path,
    monkeypatch,
):
    seen_initial = []

    async def fail(**kwargs):
        del kwargs
        seen_initial.append(json.loads((tmp_path / "agent_trajectory.jsonl").read_text()))
        if cancelled:
            raise asyncio.CancelledError
        raise RuntimeError("Model request failed")

    monkeypatch.setattr("litellm.aresponses", AsyncMock(side_effect=fail))
    env = _FakeEnv()
    expected = asyncio.CancelledError if cancelled else RuntimeError
    with pytest.raises(expected):
        await GPTDesktopUseAgent(api_retry_max=0).sample(
            env,
            max_steps=2,
            hooks=[JsonlTrajectoryLogger(tmp_path, model_id="test-model")],
        )
    assert env.closed and env._step_count == 0
    assert len(seen_initial) == 1
    assert seen_initial[0]["metadata"]["status"] == "running"
    row = json.loads((tmp_path / "agent_trajectory.jsonl").read_text())
    assert row["metadata"]["status"] == "failed"
    assert [message["role"] for message in row["messages"]] == ["user"]
    assert row["messages"] == seen_initial[0]["messages"]


@pytest.mark.asyncio
async def test_live_export_coexists_with_existing_duck_typed_hooks(tmp_path, monkeypatch):
    _stub_gpt_echoed_dim_fetch(monkeypatch)
    monkeypatch.setattr("litellm.aresponses", AsyncMock(return_value=_fake_response()))
    events = []

    class ExistingHook:
        def on_step(self, data):
            events.append(("step", data.step_idx))

        def on_complete(self, result):
            events.append(("complete", result is not None))

    env = _FakeEnv()
    await GPTDesktopUseAgent().sample(
        env,
        max_steps=2,
        hooks=[ExistingHook(), JsonlTrajectoryLogger(tmp_path, model_id="test-model")],
    )
    assert events == [("step", 0), ("complete", True)]
    assert env.closed
    row = json.loads((tmp_path / "agent_trajectory.jsonl").read_text())
    assert row["metadata"]["status"] == "completed"
