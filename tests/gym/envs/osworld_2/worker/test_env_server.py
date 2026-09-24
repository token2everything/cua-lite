"""Native env-server HTTP integration over a Worker with a tiny fake runtime.

The server owns a separate process so client-side remote registry registration
cannot replace the server's local entries. All sockets and processes belong to
this fixture; no VM, model, platform service, or external credentials are used.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

import lite.gym as gym
from lite.core.tools.calls import make_tool_call
from lite.gym.envs.osworld_2.catalog import load_release
from lite.gym.envs.osworld_2.worker_client import WorkerClient
from lite.gym.remote.client import LiteEnvClient

TOKEN = "local-env-server-test-token"


class _FakeRuntime:
    """Exercise the real Worker's episode process and persistence without a VM."""

    def __init__(self, config):
        self.output = Path(config["output"])
        self.task_id = None
        self.actions = 0
        self.closed = False
        self.phase_index = 0
        self.phase_count = 1
        self.evaluated = False

    def _record(self, method, **values):
        with (self.output / "fake_runtime.jsonl").open("a") as stream:
            stream.write(json.dumps({"method": method, "time": time.time(), **values}) + "\n")

    def _observation(self):
        image = io.BytesIO()
        Image.new("RGB", (96, 64), (self.actions, 20, 30)).save(image, format="PNG")
        instruction = f"Task {self.task_id}"
        if self.phase_count > 1:
            instruction += f" phase {self.phase_index}"
        return {"screenshot": image.getvalue(), "instruction": instruction}

    def reset(self, task_id):
        self.task_id = task_id
        self.phase_count = 2 if task_id == "069" else 1
        self._record("reset", task_id=task_id)
        return {
            "observation": self._observation(),
            "phase_count": self.phase_count,
            "phase_index": 0,
            "has_user_simulator": task_id == "007",
            "task_current_date": None,
            "runtime": {"kind": "fake"},
        }

    def action(self, action, pause=0.2):
        self.actions += 1
        self._record("action", action=action, phase_index=self.phase_index)
        return {"observation": self._observation()}

    def observe(self):
        return {"observation": self._observation()}

    def ask_user(self, question):
        assert self.task_id == "007"
        self._record("ask_user", question=question, phase_index=self.phase_index)
        return {
            "question": question, "answer": 'Keep the original; use “Budget.xlsx”.\n谢谢。',
            "observation": self._observation(),
        }

    def evaluate(self):
        self._record("evaluate", phase_index=self.phase_index)
        self.evaluated = True
        phase_score = (0.25, 0.5)[self.phase_index] if self.phase_count > 1 else 0.75
        return {
            "complete": self.phase_index == self.phase_count - 1,
            "score": 0.25 if self.phase_count > 1 and self.phase_index == 0 else 0.75,
            "phase_score": phase_score,
            "phase_index": self.phase_index,
            "phase_count": self.phase_count,
            "result": {"checked": self.task_id},
        }

    def advance(self):
        assert self.evaluated and self.phase_index < self.phase_count - 1
        self._record("advance", phase_index=self.phase_index)
        self.phase_index += 1
        self.evaluated = False
        return {
            "phase_index": self.phase_index, "phase_count": self.phase_count,
            "has_user_simulator": False, "observation": self._observation(),
            "task_current_date": None,
        }

    def close(self):
        if not self.closed:
            # Keep cleanup visibly asynchronous relative to native DELETE.
            time.sleep(0.25)
            self._record("close")
            self.closed = True
        return {"closed": True}


def _serve(root: Path, release_name: str) -> None:
    """Run the fixture's real Worker and native server on private local ports."""
    import uvicorn

    from lite.gym.envs.osworld_2 import worker_client
    from lite.gym.remote.admission import AdmissionConfig, AdmissionGate
    from lite.gym.remote.alive import SERVER_KEEP_ALIVE_TIMEOUT_SEC
    from lite.gym.remote.server import State, make_app

    sys.path.insert(0, str(Path(worker_client.__file__).parent / "worker"))
    release = load_release(release_name)
    env_id = release.env_id
    os.environ["OSWORLD_RELEASE_DIR"] = str(release.path)
    service = importlib.import_module("worker_service")
    worker = service.Worker(
        {"results": str(root / "worker"), "worker_id": "http-test-worker"},
        runtime_factory=_FakeRuntime,
    )
    worker_server = service.make_server(worker, ("127.0.0.1", 0))
    worker_thread = threading.Thread(target=worker_server.serve_forever, daemon=True)
    worker_thread.start()
    worker_url = f"http://127.0.0.1:{worker_server.server_port}"
    connection = root / "connection.json"
    connection.write_text(json.dumps({
        "endpoint": worker_url, "token": worker.admin_token, "worker_id": worker.worker_id,
    }))
    connection.chmod(0o600)

    registry = importlib.import_module("lite.gym.registry")
    registry._import_env(env_id)
    os.environ["CUA_LITE_RESET_JITTER_S"] = "0"
    os.environ["CUA_LITE_BOOT_JITTER_S"] = "0"
    admission = AdmissionGate(AdmissionConfig(
        max_live_envs=1, emergency_ram_pct=100,
        emergency_ram_free_min_bytes=1, emergency_load_per_cpu=9999,
        docker_create_concurrency=1,
    ))
    state = State.for_env_server(
        admission=admission, idle_ttl_sec=120, allowed_env_ids={env_id},
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    app = make_app(state, token=TOKEN, port=port)
    (root / "ready.json").write_text(json.dumps({"url": f"http://127.0.0.1:{port}"}))
    server = uvicorn.Server(uvicorn.Config(
        app, log_level="warning", timeout_keep_alive=SERVER_KEEP_ALIVE_TIMEOUT_SEC,
    ))
    try:
        server.run(sockets=[listener])
    finally:
        listener.close()
        worker_server.shutdown()
        worker_server.server_close()
        worker_thread.join(timeout=5)
        worker.shutdown()


@pytest.fixture(scope="module", params=["osworld-v2.1", "osworld-v2-2026.08.08"])
def env_server(tmp_path_factory, request):
    release = load_release(request.param)
    root = tmp_path_factory.mktemp("osworld-native-server")
    log_path = root / "server.log"
    environment = dict(os.environ)
    for key in ("CUA_LITE_ENV_SERVER_URL", "CUA_LITE_ENV_SERVER_TOKEN"):
        environment.pop(key, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--serve", str(root), release.name],
            env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 30
        with httpx.Client(trust_env=False, timeout=0.5) as client:
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text()
                if (root / "ready.json").exists():
                    url = json.loads((root / "ready.json").read_text())["url"]
                    try:
                        response = client.get(
                            f"{url}/envs/{release.env_id}/tasks",
                            headers={"Authorization": f"Bearer {TOKEN}"},
                        )
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                time.sleep(0.05)
            else:
                pytest.fail(f"Native test server did not become ready:\n{log_path.read_text()}")
        connection = json.loads((root / "connection.json").read_text())
        yield SimpleNamespace(
            root=root, url=url, connection=root / "connection.json",
            worker=WorkerClient(connection["endpoint"], connection["token"], timeout=10),
            worker_id=connection["worker_id"], release=release.name, env_id=release.env_id,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.fixture
def remote_env(env_server, monkeypatch):
    registry = importlib.import_module("lite.gym.registry")
    snapshot = registry._snapshot_env_registration(env_server.env_id)
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_TOKEN", TOKEN)

    def make(task_id, run_id, **kwargs):
        env = gym.make(
            f"{env_server.env_id}@{task_id}", env_server_url=env_server.url,
            connection_file=str(env_server.connection), run_id=run_id,
            experiment="native-http", results_root=str(env_server.root / "environment"),
            post_action_delay=0, step_timeout=10, reset_timeout=10,
            **kwargs,
        )
        assert isinstance(env.unwrapped, LiteEnvClient)
        return env

    yield make
    registry._restore_env_registration(env_server.env_id, snapshot)


def _runtime_calls(server, run_id):
    path = server.root / "worker" / "runs" / run_id / "fake_runtime.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_remote_catalog_keeps_release_full_and_offline_metadata(env_server):
    release = load_release(env_server.release)
    with httpx.Client(trust_env=False) as client:
        response = client.get(
            f"{env_server.url}/envs/{env_server.env_id}/tasks",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    response.raise_for_status()
    catalog = response.json()
    assert catalog["splits"] == {"eval": list(release.tasks)}
    assert len(catalog["splits"]["eval"]) == 108
    assert "run_id" in catalog["env_supported_kwargs"]
    assert "phase_index" in catalog["env_supported_kwargs"]
    for task_id, metadata in catalog["metadata"].items():
        others = metadata["others"]
        assert others["env_id"] == env_server.env_id
        assert others["benchmark_release"] == env_server.release
        assert others["release_manifest_sha256"] == release.manifest_sha256
        assert "full" in others["task_sets"]
        assert ("offline" in others["task_sets"]) == (
            task_id in release.task_sets.get("offline", ())
        )
        assert others.get("exclude_reason") == release.exclusions.get(task_id)


@pytest.mark.asyncio
async def test_remote_gui_evaluation_and_sequential_task_cleanup(env_server, remote_env):
    previous_closed_at = 0
    for task_id in ("054", "055"):
        run_id = f"http-{uuid.uuid4().hex}"
        env = remote_env(task_id, run_id)
        try:
            assert env.metadata.others["benchmark_release"] == env_server.release
            observation = await env.reset()
            assert observation.text == f"Task {task_id}"
            assert observation.metadata["run_id"] == run_id
            with Image.open(io.BytesIO(observation.image)) as image:
                assert image.size == (96, 64)
                assert image.getpixel((0, 0)) == (0, 20, 30)
            stepped = await env.step([
                make_tool_call("click", {"coordinate": [500, 250]}, call_id="click-1")
            ])
            assert not stepped.terminated
            assert stepped.results[0].tool_call_id == "click-1"
            with Image.open(io.BytesIO(stepped.results[0].images[0])) as image:
                assert image.getpixel((0, 0)) == (1, 20, 30)
            terminal = await env.step([
                make_tool_call("terminate", {"status": "success"}, call_id="done-1")
            ])
            assert terminal.terminated and not terminal.truncated
            assert terminal.reward == 0.75
            assert terminal.info["run_id"] == run_id
            assert terminal.info["benchmark_release"] == env_server.release
            assert terminal.info["worker_id"] == env_server.worker_id
            assert terminal.info["evaluate_payload"] == {
                "complete": True, "score": 0.75, "phase_score": 0.75,
                "phase_index": 0, "phase_count": 1, "result": {"checked": task_id},
            }
        finally:
            await env.close()
        receipt = await asyncio.to_thread(
            env_server.worker.wait_closed, run_id, env_server.worker_id, env_server.release,
        )
        assert receipt["last_closed_run_id"] == run_id
        calls = _runtime_calls(env_server, run_id)
        assert [call["method"] for call in calls] == [
            "reset", "action", "action", "evaluate", "close",
        ]
        assert calls[1]["action"] == "pyautogui.click(48, 16, button='left')"
        assert calls[2]["action"] == "DONE"
        assert calls[0]["time"] >= previous_closed_at
        previous_closed_at = calls[-1]["time"]


@pytest.mark.asyncio
async def test_remote_two_phase_handles_share_one_worker_lifecycle(env_server, remote_env):
    run_id = f"phase-{uuid.uuid4().hex}"
    session = await asyncio.to_thread(env_server.worker.reset, "069", run_id)
    agent_client = WorkerClient(env_server.worker.endpoint, session["agent_token"], timeout=10)
    env = None
    try:
        for phase_index, expected_reward in enumerate((0.25, 0.5)):
            env = remote_env("069", run_id, phase_index=phase_index)
            observation = await env.reset()
            assert observation.text == f"Task 069 phase {phase_index}"
            assert observation.metadata["run_id"] == run_id
            assert observation.metadata["phase_index"] == phase_index
            assert observation.metadata["phase_count"] == 2
            await env.step([
                make_tool_call("click", {"coordinate": [500, 250]}, call_id="click")
            ])
            terminal = await env.step([
                make_tool_call("terminate", {"status": "success"}, call_id="done")
            ])
            assert terminal.terminated and not terminal.truncated
            assert terminal.reward == expected_reward
            evaluation = terminal.info["evaluate_payload"]
            assert evaluation["phase_index"] == phase_index
            assert evaluation["complete"] == (phase_index == 1)
            assert evaluation["score"] == (0.25 if phase_index == 0 else 0.75)
            await env.close()
            env = None
            health = await asyncio.to_thread(env_server.worker.request, "GET", "/healthz")
            assert health["run_id"] == run_id
            assert health["state"] == ("phase_evaluated" if phase_index == 0 else "evaluated")
            assert health["last_closed_run_id"] != run_id
            assert not any(call["method"] == "close" for call in _runtime_calls(env_server, run_id))
            if phase_index == 0:
                await asyncio.to_thread(env_server.worker.advance, run_id, phase_index=0)
                with pytest.raises(RuntimeError, match="phase_index"):
                    await asyncio.to_thread(
                        agent_client.action, run_id, "pyautogui.click()", phase_index=0,
                    )
    finally:
        if env is not None:
            await env.close()
        await asyncio.to_thread(env_server.worker.close, run_id)
        await asyncio.to_thread(
            env_server.worker.wait_closed, run_id, env_server.worker_id, env_server.release,
        )
    calls = _runtime_calls(env_server, run_id)
    assert [call["method"] for call in calls] == [
        "reset", "action", "action", "evaluate", "advance",
        "action", "action", "evaluate", "close",
    ]
    assert [call["phase_index"] for call in calls if call["method"] == "action"] == [0, 0, 1, 1]


@pytest.mark.asyncio
async def test_remote_ask_user_returns_exact_paired_reply_without_finishing(env_server, remote_env):
    run_id = f"ask-{uuid.uuid4().hex}"
    env = remote_env("007", run_id, extra_tools=["ask_user", "terminate"])
    try:
        await env.reset()
        result = await env.step([
            make_tool_call("ask_user", {"text": "Which file?"}, call_id="ask-1"),
            make_tool_call("screenshot", {}, call_id="screen-1"),
        ])
        assert not result.terminated and not result.truncated
        assert result.reward is None
        assert [item.tool_call_id for item in result.results] == ["ask-1", "screen-1"]
        reply = result.results[0]
        assert reply.text == 'Keep the original; use “Budget.xlsx”.\n谢谢。'
        assert reply.error is None
        assert reply.metadata["phase_index"] == 0
        assert reply.images[0].startswith(b"\x89PNG")
        assert [call["method"] for call in _runtime_calls(env_server, run_id)] == [
            "reset", "ask_user",
        ]
        terminal = await env.step([
            make_tool_call("terminate", {"status": "success"}, call_id="done")
        ])
        assert terminal.terminated
    finally:
        await env.close()
    await asyncio.to_thread(
        env_server.worker.wait_closed, run_id, env_server.worker_id, env_server.release,
    )


class _CutFirstReset(httpx.AsyncBaseTransport):
    """Lose a real successful reset frame after the server has executed it."""

    def __init__(self):
        self.transport = httpx.AsyncHTTPTransport()
        self.reset_calls = 0

    async def handle_async_request(self, request):
        response = await self.transport.handle_async_request(request)
        if request.url.path.endswith("/reset"):
            self.reset_calls += 1
            if self.reset_calls == 1 and response.status_code == 200:
                content = await response.aread()
                await response.aclose()
                return httpx.Response(
                    200, content=content[:-1],
                    headers={"Content-Type": response.headers["Content-Type"]},
                )
        return response

    async def aclose(self):
        await self.transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase_bound", [False, True])
async def test_native_reset_retry_reuses_one_episode(
    env_server, remote_env, monkeypatch, phase_bound,
):
    client_module = importlib.import_module("lite.gym.remote.client")
    monkeypatch.setattr(client_module, "_client_retry_delay_s", lambda _: 0.01)
    run_id = f"retry-{uuid.uuid4().hex}"
    if phase_bound:
        await asyncio.to_thread(env_server.worker.reset, "054", run_id)
    env = remote_env("054", run_id, **({"phase_index": 0} if phase_bound else {}))
    transport = _CutFirstReset()
    await env.unwrapped._client.aclose()
    env.unwrapped._client = httpx.AsyncClient(
        transport=transport, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10,
    )
    try:
        observation = await env.reset()
        assert transport.reset_calls == 2
        assert observation.metadata["run_id"] == run_id
        repeated = await asyncio.gather(env.reset(), env.reset())
        assert all(item == observation for item in repeated)
        calls = _runtime_calls(env_server, run_id)
        assert [call["method"] for call in calls] == ["reset"]
    finally:
        await env.close()
        if phase_bound:
            assert [call["method"] for call in _runtime_calls(env_server, run_id)] == ["reset"]
            await asyncio.to_thread(env_server.worker.close, run_id)
    await asyncio.to_thread(
        env_server.worker.wait_closed, run_id, env_server.worker_id, env_server.release,
    )
    calls = _runtime_calls(env_server, run_id)
    assert [call["method"] for call in calls] == ["reset", "close"]


if __name__ == "__main__":
    assert sys.argv[1] == "--serve"
    _serve(Path(sys.argv[2]), sys.argv[3])
