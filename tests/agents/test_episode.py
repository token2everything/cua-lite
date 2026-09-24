"""Single-episode CLI tests exercise real Agent/client code over mocked HTTP."""

from __future__ import annotations

import asyncio
import base64
import copy
import importlib
import io
import json
import sys
import threading
from pathlib import Path

import httpx
import pytest
from PIL import Image

from lite.agents import episode
from lite.core import LiteCUAMetadata
from lite.core.tools.calls import tool_call_id, tool_call_name
from lite.core.tools.results import make_tool_result
from lite.gym.remote.frame import encode_reset_observation, encode_step_result
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

TOKEN = "private-test-server-token"
API_KEY = "private-test-api-key"
ENV_KEY = "episode_test@demo"


@pytest.fixture
def job(tmp_path):
    config = tmp_path / "model.json"
    config.write_text(json.dumps({
        "api_key": API_KEY, "base_url": "https://model.test/v1",
        "models": {
            "chat": {"model": "native-chat-model", "api_type": "openai_chat"},
            "responses": {"model": "native-responses-model", "api_type": "openai_responses"},
        },
        "default_model": "chat",
    }))
    return {
        "env": ENV_KEY, "env_kwargs": {"phase_index": 0},
        "server": {"url": "https://env.test", "token": TOKEN},
        "model_config": str(config), "output_dir": str(tmp_path / "run"),
        "episode_index": 0, "max_steps": 12, "model_timeout": 30,
        "metadata": {"phase_index": 0, "run_id": "run-example", "release": "example-v1"},
    }


class _Wire:
    """Native env-server frames plus gateway responses, without a socket/server."""

    def __init__(self):
        self.requests = []
        self.models = []
        self.actions = []
        self.creates = []
        self.deletes = []
        self.close_failures = 0
        self.model_error = False
        self.reset_error = False
        self.step_error = False
        self.turns_before_finish = 1
        self.instance = 0
        self.turn = 0
        self.create_started = threading.Event()
        self.create_release = None
        self.metadata = LiteCUAMetadata(
            dims=("desktop", "use"), valid_actions=["click", "screenshot"],
            others={"env_id": "episode_test", "task_id": "demo"},
        )

    def image(self):
        stream = io.BytesIO()
        Image.new("RGB", (32, 24), (self.instance * 30, self.turn * 20, 70)).save(
            stream, format="PNG"
        )
        return stream.getvalue()

    def handle(self, request):
        self.requests.append((request.method, request.url.path))
        if request.url.host == "model.test":
            assert request.headers["Authorization"] == f"Bearer {API_KEY}"
            body = json.loads(request.content)
            self.models.append(body)
            if self.model_error:
                return httpx.Response(400, json={"error": f"Rejected {API_KEY}"})
            if request.url.path == "/v1/responses":
                if self.turn < self.turns_before_finish:
                    output = [{
                        "type": "computer_call", "call_id": f"call-{self.turn}",
                        "action": {"type": "click", "x": 16, "y": 12, "button": "left"},
                    }]
                else:
                    output = [{
                        "type": "message", "content": [{"type": "output_text", "text": "Done."}],
                    }]
                return httpx.Response(200, json={
                    "id": "response-test", "status": "completed", "output": output, "usage": {},
                })
            if request.method == "GET":
                return httpx.Response(404, json={"error": "stateless gateway"})
            message = {"role": "assistant", "content": "Done."}
            if self.turn < self.turns_before_finish:
                message = {
                    "role": "assistant", "content": f"Inspect screen {self.turn}.",
                    "tool_calls": [{
                        "id": f"provider-{self.turn}", "type": "function",
                        "function": {"name": "computer", "arguments": json.dumps({
                            "actions": [{"action": "click", "coordinate": [500, 500]}],
                        })},
                    }],
                }
            return httpx.Response(200, json={
                "id": "chat-test", "model": "native-chat-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            })
        assert request.url.host == "env.test"
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        if request.url.path == "/envs/episode_test/tasks":
            return httpx.Response(200, json={
                "splits": {"eval": ["demo"]},
                "metadata": {"demo": self.metadata.to_dict()},
                "env_supported_kwargs": ["phase_index"],
            })
        if request.url.path == "/instances":
            self.creates.append(json.loads(request.content))
            self.create_started.set()
            if self.create_release is not None:
                assert self.create_release.wait(5), "test did not release pending create"
            self.instance += 1
            self.turn = 0
            return httpx.Response(201, json={
                "id": f"handle-{self.instance}", "metadata": self.metadata.to_dict(),
            })
        if request.method == "DELETE":
            self.deletes.append(request.url.path)
            if len(self.deletes) <= self.close_failures:
                return httpx.Response(500, text=f"close failed {TOKEN} {API_KEY}")
            return httpx.Response(204)
        if request.url.path.endswith("/reset"):
            if self.reset_error:
                return httpx.Response(400, text=f"reset failed {TOKEN} {API_KEY}")
            payload = encode_reset_observation(LiteEnvObservation(
                image=self.image(), text=f"Complete episode {self.instance}.",
            ))
        elif request.url.path.endswith("/step"):
            actions = json.loads(request.content)["actions"]
            self.actions.append(actions)
            if self.step_error:
                return httpx.Response(400, text=f"step failed {TOKEN} {API_KEY}")
            self.turn += 1
            done = any(tool_call_name(action) in {"response", "terminate"} for action in actions)
            payload = encode_step_result(LiteEnvStepResult(
                terminated=done, reward=0.75 if done else 0.0,
                results=[make_tool_result(
                    tool_call_id(action), text=f"screen {self.turn}", images=[self.image()],
                ) for action in actions],
            ))
        else:
            raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")
        return httpx.Response(
            200, content=payload, headers={"Content-Type": "application/octet-stream"}
        )


@pytest.fixture
def wire(monkeypatch):
    wire = _Wire()
    transport = httpx.MockTransport(wire.handle)
    sync_client, async_client = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(httpx, "Client", lambda **kw: sync_client(transport=transport, **kw))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: async_client(transport=transport, **kw))

    def get(url, **kwargs):
        with sync_client(transport=transport) as client:
            return client.get(url, **kwargs)

    monkeypatch.setattr(httpx, "get", get)
    registry = importlib.import_module("lite.gym.registry")
    registry._clear_env_registration("episode_test")
    yield wire
    registry._clear_env_registration("episode_test")


def _receipt(job):
    directory = Path(job["output_dir"]) / "model" / f"episode_{job['episode_index']:03d}"
    return directory / "episode.json"


def _trace(job):
    path = Path(job["output_dir"]) / "agent_trajectory.jsonl"
    assert len(path.read_text().splitlines()) == 1
    return json.loads(path.read_text())


async def test_remote_episode_uses_real_proxy_without_local_env_imports(
    job, wire, monkeypatch, capsys,
):
    before = set(sys.modules)
    original_import = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(("lite.gym.envs", "lite.platforms")), name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "https://unrelated.test")
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_TOKEN", "unrelated-token")
    result = await episode.run(job)

    assert result == json.loads(_receipt(job).read_text())
    assert result["status"] == "completed" and result["closed"]
    assert result["turns"] == 2 and result["episode_return"] == 0.75
    assert result["terminated"] and not result["truncated"]
    assert result["history_turns"] == 4
    assert result["model_id"] == "native-chat-model" and result["api_type"] == "openai_chat"
    assert len(wire.creates) == len(wire.deletes) == 1
    assert wire.creates[0]["env_key"] == ENV_KEY
    assert wire.creates[0]["env_kwargs"]["phase_index"] == 0
    assert wire.actions[0][0]["function"]["arguments"] == {
        "actions": [{"action": "click", "coordinate": [500, 500]}],
    }
    assert not any(
        name.startswith(("lite.gym.envs", "lite.platforms")) for name in set(sys.modules) - before
    )
    assert episode.os.environ["CUA_LITE_ENV_SERVER_URL"] == "https://unrelated.test"
    assert episode.os.environ["CUA_LITE_ENV_SERVER_TOKEN"] == "unrelated-token"
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == ["agent_step", "agent_step", "episode_complete"]
    assert all(event["episode_index"] == 0 for event in events[:-1])
    assert all(set(event["seconds"]) == {"observe", "predict", "act"} for event in events[:-1])
    trace = _trace(job)
    assert trace["metadata"]["status"] == "running"  # coordinator owns whole-task finalization
    assert trace["metadata"]["tools"] == wire.models[0]["tools"]
    assert trace["metadata"]["episodes"] == [{
        **job["metadata"], "episode_index": 0, "context_reset": True,
        "message_start": 0, "message_end": len(trace["messages"]),
    }]
    assert not (Path(job["output_dir"]) / "run.json").exists()


@pytest.mark.parametrize("history", [4, None])
async def test_history_retains_all_text_and_archives_original_images(job, wire, history):
    wire.turns_before_finish = 6
    job["history_turns"] = history
    await episode.run(job)
    trace = _trace(job)
    last = wire.models[-1]
    image_urls = [part["image_url"]["url"] for message in last["messages"]
                  for part in message.get("content", []) if isinstance(part, dict)
                  and part["type"] == "image_url"]
    assert len(image_urls) == (7 if history is None else 5)
    assert "Complete episode 1." in json.dumps(last["messages"])
    assert all(f"Inspect screen {turn}." in json.dumps(last["messages"]) for turn in range(6))
    archived = [part["image"]["url"] for message in trace["messages"]
                for part in message.get("content", []) if part["type"] == "image"]
    assert len(archived) >= 7
    assert base64.b64decode(image_urls[0].split(",", 1)[1]) == Path(
        archived[0 if history is None else 2]
    ).read_bytes()
    assert trace["metadata"]["tools"] == last["tools"]
    for request in wire.models:
        assert request["tools"] == last["tools"]
    model_dir = _receipt(job).parent
    logged = json.loads((model_dir / "00001.request.json").read_text())
    logged_image = next(part["image_url"]["url"] for message in logged["messages"]
                        for part in message.get("content", []) if isinstance(part, dict)
                        and part["type"] == "image_url")
    assert (model_dir / logged_image).resolve() == Path(archived[0])
    assert not (model_dir / "images").exists()


async def test_second_episode_resets_context_and_keeps_first_logs_and_messages(job, wire):
    first_result = await episode.run(job)
    first = _trace(job)
    first_logs = {path.name: path.read_bytes() for path in _receipt(job).parent.iterdir()}
    first_requests = len(wire.models)
    job.update(episode_index=1, env_kwargs={"phase_index": 1}, metadata={"phase_index": 1})
    second_result = await episode.run(job)
    trace = _trace(job)
    assert trace["messages"][:len(first["messages"])] == first["messages"]
    assert trace["metadata"]["episodes"][0] == first["metadata"]["episodes"][0]
    assert second_result["message_start"] == first_result["message_end"] == len(first["messages"])
    assert second_result["message_end"] == len(trace["messages"])
    second_request = wire.models[first_requests]
    assert "Complete episode 1." not in json.dumps(second_request)
    assert "Complete episode 2." in json.dumps(second_request)
    assert len(second_request["messages"]) == 2
    first_dir = Path(job["output_dir"]) / "model" / "episode_000"
    assert first_logs == {path.name: path.read_bytes() for path in first_dir.iterdir()}
    assert (_receipt(job).parent / "00001.request.json").exists()
    prior_images = {part["image"]["url"] for message in first["messages"]
                    for part in message.get("content", []) if part["type"] == "image"}
    new_images = {part["image"]["url"] for message in trace["messages"][len(first["messages"]):]
                  for part in message.get("content", []) if part["type"] == "image"}
    assert prior_images.isdisjoint(new_images)


@pytest.mark.parametrize("index", [0, 2])
async def test_non_contiguous_episode_refuses_before_http_or_overwrite(job, wire, index):
    await episode.run(job)
    before = _receipt(job).read_bytes()
    job["episode_index"] = index
    with pytest.raises(ValueError, match="continue"):
        await episode.run(job)
    assert len(wire.creates) == 1
    original = Path(job["output_dir"]) / "model" / "episode_000" / "episode.json"
    assert original.read_bytes() == before


@pytest.mark.parametrize("failure", ["model_error", "reset_error", "step_error"])
async def test_failure_preserves_trace_and_closes_real_remote_handle(
    job, wire, failure, capsys, caplog,
):
    setattr(wire, failure, True)
    result = await episode.run(job)
    assert result["status"] == "failed" and result["closed"]
    assert result["error"]
    assert len(wire.deletes) == 1
    trace = _trace(job)
    assert len(trace["metadata"]["episodes"]) == 1
    if failure != "reset_error":
        assert trace["messages"][0]["role"] == "system"
    if failure == "step_error":
        assert any(message.get("tool_calls") for message in trace["messages"])
    public = capsys.readouterr().out + caplog.text + _receipt(job).read_text()
    assert API_KEY not in public and TOKEN not in public


async def test_unacknowledged_close_is_failed_and_retried(job, wire, capsys, caplog):
    wire.close_failures = 5
    result = await episode.run(job)
    assert result["status"] == "failed" and not result["closed"]
    assert len(wire.deletes) == 3  # Agent finalizer, explicit confirmation, failure cleanup
    public = capsys.readouterr().out + caplog.text + _receipt(job).read_text()
    assert API_KEY not in public and TOKEN not in public


async def test_sample_close_failure_is_recovered_by_explicit_confirmation(job, wire):
    wire.close_failures = 1
    result = await episode.run(job)
    assert result["status"] == "completed" and result["closed"]
    assert len(wire.deletes) == 2


async def test_cancelled_make_waits_for_remote_handle_and_releases_it(job, wire):
    wire.create_release = threading.Event()
    task = asyncio.create_task(episode.run(job))
    assert await asyncio.to_thread(wire.create_started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not wire.deletes
    wire.create_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = json.loads(_receipt(job).read_text())
    assert result["status"] == "failed" and result["closed"]
    assert result["error"] == "Episode was cancelled"
    assert len(wire.deletes) == 1
    assert _trace(job)["metadata"]["episodes"][0]["message_end"] == 0


async def test_cancel_before_make_thread_submission_still_drains_handle(job, wire, monkeypatch):
    queued, release = asyncio.Event(), asyncio.Event()
    real_to_thread = asyncio.to_thread

    async def delayed(func, *args, **kwargs):
        if func is episode.gym.make:
            queued.set()
            await release.wait()
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", delayed)
    task = asyncio.create_task(episode.run(job))
    await queued.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not wire.creates and not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(wire.creates) == len(wire.deletes) == 1


async def test_responses_uses_existing_full_history_gateway(job, wire):
    job.update(model_profile="responses", history_turns=None)
    result = await episode.run(job)
    assert result["status"] == "completed" and result["closed"]
    assert result["history_turns"] is None
    assert result["model_id"] == "native-responses-model"
    assert all(body["tools"] == [{"type": "computer"}] for body in wire.models)
    assert all("previous_response_id" not in body for body in wire.models)
    assert _trace(job)["metadata"]["tools"] == [{"type": "computer"}]


async def test_coordinator_logs_can_preexist_and_are_never_changed(job, wire):
    directory = _receipt(job).parent
    directory.mkdir(parents=True)
    for name in ("agent.log", "env-server.log"):
        (directory / name).write_text("coordinator owns this log\n")
    result = await episode.run(job)
    assert result["status"] == "completed"
    for name in ("agent.log", "env-server.log"):
        assert (directory / name).read_text() == "coordinator owns this log\n"


async def test_model_config_failure_keeps_empty_episode_boundary(job, wire):
    job["model_profile"] = "not-configured"
    result = await episode.run(job)
    assert result["status"] == "failed" and not result["closed"]
    assert not wire.requests
    trace = _trace(job)
    assert trace["messages"] == []
    assert trace["metadata"]["episodes"][0]["message_end"] == 0


async def test_existing_episode_receipt_refuses_reuse_without_trace(job, wire):
    receipt = _receipt(job)
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"status":"completed"}\n')
    before = receipt.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        await episode.run(job)
    assert receipt.read_bytes() == before and not wire.requests


@pytest.mark.parametrize("patch", [
    {"server": {}}, {"server": {"url": "", "token": TOKEN}},
    {"server": {"url": "https://env.test", "token": ""}},
    {"env_kwargs": {"env_server_url": "https://other.test"}},
    {"env": "missing-task"}, {"episode_index": True}, {"history_turns": -1},
    {"model_timeout": float("inf")}, {"output_dir": "relative"},
])
async def test_invalid_jobs_never_discover_or_create_env(job, wire, patch):
    job.update(copy.deepcopy(patch))
    with pytest.raises(ValueError):
        await episode.run(job)
    assert not wire.requests


async def test_sigterm_handler_cancels_episode_without_duplicate_cancel(job, monkeypatch):
    handlers = {}
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        loop, "add_signal_handler", lambda sig, callback: handlers.update({sig: callback})
    )
    monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: handlers.pop(sig))
    entered = asyncio.Event()
    finished = []

    async def fake_run(_job):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            finished.append(True)

    monkeypatch.setattr(episode, "run", fake_run)
    task = asyncio.create_task(episode._run_cli(job))
    await entered.wait()
    handler = handlers[episode.signal.SIGTERM]
    handler()
    handler()
    handler()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished == [True] and not handlers
