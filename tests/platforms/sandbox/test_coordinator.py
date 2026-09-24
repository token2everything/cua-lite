"""Coordinate phases with fake subprocesses/HTTP; never contact Docker or a Worker."""

from __future__ import annotations

import asyncio
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.platforms.sandbox.osworld_2 import coordinator

IMAGE = "registry.example/agent@sha256:" + "a" * 64


class _Artifacts:
    def __init__(self, path):
        self.path = path
        self.run_id = "owned-run-123"
        self.events = []
        self.updates = []

    def event(self, name, **fields):
        self.events.append((name, copy.deepcopy(fields)))

    def update(self, **fields):
        self.updates.append(copy.deepcopy(fields))


class _Process:
    def __init__(self, harness, kind, *, code=None, output=b""):
        self.harness = harness
        self.kind = kind
        self.returncode = code
        self.output = output
        self.waited = 0
        self.terminated = False
        self.killed = False
        self.stdout = self.lines() if kind == "agent" else None

    async def lines(self):
        harness = self.harness
        harness.stream_entered.set()
        if harness.hold_stream:
            await harness.stream_release.wait()
        yield b"Agent diagnostic line\n"
        if harness.emit_progress:
            for turn in range(1, 3):
                yield (json.dumps({
                    "event": "agent_step",
                    "episode_index": harness.spec["episode_index"] + harness.progress_offset,
                    "turn": turn, "reward": 0.0, "terminated": turn == 2,
                    "truncated": False, "seconds": {"predict": float(turn), "act": 0.2},
                }) + "\n").encode()
        if harness.write_receipt:
            path = harness.artifacts.path / "model" / (
                f"episode_{harness.spec['episode_index']:03d}"
            ) / "episode.json"
            path.write_text(json.dumps(harness.result))
        yield (json.dumps({"event": "episode_complete", **harness.result}) + "\n").encode()

    async def wait(self):
        self.waited += 1
        harness = self.harness
        harness.events.append(("wait", self.kind))
        if self.kind == "agent":
            self.returncode = harness.agent_exit if self.returncode is None else self.returncode
            harness.owned_live = False
            if harness.server_exit_during_agent is not None:
                harness.server.returncode = harness.server_exit_during_agent
        elif self.kind == "server":
            if harness.hang_server_close and not self.killed:
                await asyncio.Event().wait()
            if self.returncode is None:
                self.returncode = harness.server_close_exit
        return self.returncode

    async def communicate(self, input=None):
        if self.kind == self.harness.block_communicate:
            self.harness.communicate_entered.set()
            await self.harness.communicate_release.wait()
        await self.wait()
        return self.output, b""

    def terminate(self):
        if self.kind == "server" and self.harness.server_terminate_race:
            self.returncode = -coordinator.signal.SIGTERM
            raise ProcessLookupError("env-server exited before termination")
        self.terminated = True
        self.harness.events.append(("terminate", self.kind))
        if self.kind == "server" and not self.harness.hang_server_close:
            self.returncode = self.harness.server_close_exit
        elif self.kind == "agent":
            self.returncode = 143

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.harness.events.append(("kill", self.kind))


class _Harness:
    def __init__(self, tmp_path):
        self.artifacts = _Artifacts(tmp_path / "results" / "task" / "run")
        self.args = SimpleNamespace(
            connection=tmp_path / "connection.json", release="osworld-v2.1",
            results_root=tmp_path / "results", model_timeout=600, cleanup_timeout=0.05,
        )
        self.spec = {
            "env": "osworld_2.v2_1@054", "env_kwargs": {"phase_index": 0},
            "episode_index": 0, "output_dir": str(self.artifacts.path),
            "max_steps": 10, "model_timeout": 600,
            "metadata": {"phase_index": 0, "run_id": self.artifacts.run_id},
        }
        self.result = {
            "status": "completed", "closed": True, "turns": 2,
            "episode_return": 0.5, "terminated": True, "truncated": False,
            "message_start": 0, "message_end": 5,
            "model_id": "test-model", "api_type": "openai_chat", "history_turns": 4,
            "timing": {"predict": 3.0, "act": 0.4, "total": 5.0},
        }
        self.commands = []
        self.events = []
        self.requests = []
        self.jobs = []
        self.server = self.agent = None
        self.owned_name = None
        self.owned_id = "b" * 64
        self.owned_live = False
        self.container_present = False
        self.other_container_live = True
        self.available = True
        self.readiness_error = None
        self.server_start_exit = None
        self.server_exit_during_agent = None
        self.server_close_exit = 0
        self.server_terminate_race = False
        self.agent_exit = 0
        self.stop_exit = 0
        self.create_exit = 0
        self.hang_server_close = False
        self.write_receipt = True
        self.block_spawn = None
        self.spawn_entered = asyncio.Event()
        self.spawn_release = asyncio.Event()
        self.hold_stream = False
        self.stream_entered = asyncio.Event()
        self.stream_release = asyncio.Event()
        self.block_communicate = None
        self.communicate_entered = asyncio.Event()
        self.communicate_release = asyncio.Event()
        self.emit_progress = True
        self.progress_offset = 0
        self.fail_spawn = None

    def open(self, request, timeout):
        self.requests.append(request)
        assert request.full_url.endswith("/envs/osworld_2.v2_1")
        assert request.get_header("Authorization").startswith("Bearer ")
        if self.readiness_error is not None:
            raise self.readiness_error
        return io.BytesIO(json.dumps({"available": self.available}).encode())

    async def spawn(self, *command, **kwargs):
        command = list(command)
        self.commands.append((command, kwargs))
        if command[:2] == ["docker", self.fail_spawn]:
            raise OSError(f"test Docker {self.fail_spawn} process could not start")
        if "lite.gym.remote.serve" in command:
            kind = "server"
            process = self.server = _Process(self, kind, code=self.server_start_exit)
        elif command[:2] == ["docker", "create"]:
            kind = "create"
            self.owned_name = command[command.index("--name") + 1]
            self.container_present = self.create_exit == 0
            mounts = [command[i + 1] for i, part in enumerate(command) if part == "--volume"]
            episode_mount = next(item for item in mounts if ":/private/episode.json:" in item)
            self.jobs.append(json.loads(Path(episode_mount.split(":", 1)[0]).read_text()))
            process = _Process(
                self, kind, code=self.create_exit,
                output=(self.owned_id + "\n").encode() if self.create_exit == 0 else b"",
            )
        elif command[:2] == ["docker", "start"]:
            kind = "agent"
            assert command[-1] == self.owned_id and self.container_present
            self.owned_live = True
            process = self.agent = _Process(self, kind)
        elif command[:2] in (["docker", "stop"], ["docker", "rm"]):
            assert command[-1] == self.owned_id, command
            kind = "stop" if command[1] == "stop" else "remove"
            code = self.stop_exit if kind == "stop" else 0
            if code == 0:
                self.owned_live = False
                if kind == "remove":
                    self.container_present = False
                if self.agent is not None and self.agent.returncode is None:
                    self.agent.returncode = 143
            process = _Process(self, kind, code=code)
        elif command[:2] in (["docker", "inspect"], ["docker", "container"]):
            assert command[-1] in {self.owned_name, self.owned_id}, command
            kind = "inspect"
            process = _Process(
                self, kind, code=0 if self.owned_live else 1,
                output=(json.dumps([{"Id": self.owned_id, "State": {
                    "Running": self.owned_live,
                }}]) + "\n").encode() if self.owned_live else b"",
            )
        else:
            raise AssertionError(f"Unexpected subprocess {command}")
        self.events.append(("spawn", kind))
        if kind == self.block_spawn:
            self.spawn_entered.set()
            await self.spawn_release.wait()
        return process

    async def run(self, details=None):
        return await coordinator.execute_episode(
            self.spec, self.artifacts, {} if details is None else details,
            args=self.args, image=IMAGE,
        )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    harness = _Harness(tmp_path)
    private_root = tmp_path / "private"
    original_path = Path

    def scoped_path(value):
        return private_root if str(value) == "/run/cua-episodes" else original_path(value)

    monkeypatch.setattr(coordinator, "Path", scoped_path)
    monkeypatch.setattr(coordinator.asyncio, "create_subprocess_exec", harness.spawn)
    monkeypatch.setattr(coordinator.urllib.request, "build_opener", lambda *args: harness)
    monkeypatch.setattr(
        coordinator, "load_release", lambda name: SimpleNamespace(env_id="osworld_2.v2_1")
    )
    return harness


@pytest.mark.parametrize("server_exit", [0, -coordinator.signal.SIGTERM])
async def test_returns_after_agent_and_server_exit_and_accumulates_phase_progress(
    harness, capsys, server_exit,
):
    harness.server_close_exit = server_exit
    details = {"turns": 3, "timing": {"predict": 7.0}}
    result = await harness.run(details)
    assert result == harness.result
    assert harness.agent.returncode is not None and harness.server.waited
    assert harness.server.terminated and harness.server.returncode == server_exit
    assert details["turns"] == 5 and details["timing"]["predict"] == 10.0
    assert details["timing"]["act"] == pytest.approx(0.4)
    assert [event[0] for event in harness.artifacts.events] == ["agent_step", "agent_step"]
    assert len(harness.artifacts.updates) >= 2
    assert harness.events.index(("wait", "agent")) < harness.events.index(("terminate", "server"))
    assert not harness.owned_live and harness.other_container_live
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(event["event"] == "agent_step" for event in output)
    assert not list((harness.artifacts.path.parents[2] / "private").glob("*/episode.json"))


async def test_generic_agent_gets_only_episode_model_and_results_mounts(harness, monkeypatch):
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "http://unrelated.test")
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_TOKEN", "unrelated-token")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-forward-as-an-agent-env")
    await harness.run()
    server_command, server_options = harness.commands[0]
    assert server_options["env"]["OSWORLD_WORKER_CONNECTION"] == str(harness.args.connection)
    assert "CUA_LITE_ENV_SERVER_URL" not in server_options["env"]
    assert "CUA_LITE_ENV_SERVER_TOKEN" not in server_options["env"]
    assert server_command[server_command.index("--max-live-envs") + 1] == "1"
    command, _ = next(item for item in harness.commands if item[0][:2] == ["docker", "create"])
    mounts = [command[i + 1] for i, part in enumerate(command) if part == "--volume"]
    assert len(mounts) == 3
    assert mounts[0].endswith(":/private/episode.json:ro")
    assert mounts[1] == "/opt/osworld-job/model-api.json:/private/model-api.json:ro"
    assert mounts[2] == f"/results/agent-results:{harness.args.results_root}:rw"
    assert not any(part in command for part in ("--privileged", "--device", "/dev/kvm"))
    assert "docker.sock" not in " ".join(command)
    assert "control.json" not in " ".join(command)
    assert "/opt/osworld-data" not in " ".join(command)
    assert "OPENAI_API_KEY" not in command
    assert command[-3:] == [IMAGE, "--job", "/private/episode.json"]
    assert harness.jobs[0]["model_config"] == "/private/model-api.json"
    assert harness.jobs[0]["server"]["url"] == "http://127.0.0.1:30100"


async def test_two_phases_have_disjoint_jobs_and_no_overlapping_servers(harness):
    await harness.run()
    first_server, first_agent = harness.server, harness.agent
    first_log = (harness.artifacts.path / "model/episode_000/agent.log").read_bytes()
    harness.spec.update(episode_index=1, env_kwargs={"phase_index": 1})
    harness.result.update(message_start=5, message_end=10)
    await harness.run()
    assert first_server.returncode == first_agent.returncode == 0
    assert harness.server is not first_server and harness.agent is not first_agent
    assert (harness.artifacts.path / "model/episode_000/agent.log").read_bytes() == first_log
    assert (harness.artifacts.path / "model/episode_001/episode.json").exists()
    assert harness.jobs[0]["server"]["token"] != harness.jobs[1]["server"]["token"]
    assert harness.jobs[1]["env_kwargs"]["phase_index"] == 1


async def test_unavailable_catalog_stops_server_without_starting_agent(harness):
    harness.available = False
    with pytest.raises(RuntimeError, match="unavailable"):
        await harness.run()
    assert harness.agent is None
    assert harness.server.terminated and harness.server.waited


async def test_readiness_timeout_stops_server_and_never_launches_agent(harness, monkeypatch):
    harness.readiness_error = OSError("offline readiness failure")

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(coordinator.asyncio, "sleep", no_sleep)
    with pytest.raises(TimeoutError, match="readiness"):
        await harness.run()
    assert harness.agent is None and harness.server.terminated and harness.server.waited


@pytest.mark.parametrize("failure", ["missing_receipt", "nonzero_completed"])
async def test_agent_receipt_and_exit_status_must_agree(harness, failure):
    if failure == "missing_receipt":
        harness.write_receipt = False
    else:
        harness.agent_exit = 7
    with pytest.raises(RuntimeError):
        await harness.run()
    assert harness.server.terminated and harness.server.waited
    assert not harness.owned_live


@pytest.mark.parametrize("exit_code", [0, 7, -coordinator.signal.SIGTERM])
async def test_server_exit_during_agent_cannot_report_success(harness, exit_code):
    harness.server_exit_during_agent = exit_code
    with pytest.raises(RuntimeError, match="env-server"):
        await harness.run()
    assert not harness.server.terminated


@pytest.mark.parametrize("exit_code", [3, -coordinator.signal.SIGINT, -coordinator.signal.SIGKILL])
async def test_nonzero_server_close_fails_episode(harness, exit_code):
    harness.server_close_exit = exit_code
    with pytest.raises(RuntimeError, match="env-server"):
        await harness.run()
    assert harness.server.waited


async def test_sigterm_exit_is_not_accepted_when_our_termination_was_not_sent(harness):
    harness.server_terminate_race = True
    with pytest.raises(RuntimeError, match="env-server cleanup failed"):
        await harness.run()
    assert harness.server.waited and not harness.server.terminated


@pytest.mark.parametrize("cleanup_exit", [-coordinator.signal.SIGTERM, 3])
async def test_agent_error_survives_server_shutdown(harness, cleanup_exit):
    harness.result.update(status="failed", error="Model request rejected with status 403")
    harness.agent_exit = 1
    harness.server_close_exit = cleanup_exit
    if cleanup_exit == -coordinator.signal.SIGTERM:
        assert await harness.run() == harness.result
    else:
        with pytest.raises(RuntimeError) as caught:
            await harness.run()
        assert harness.result["error"] in str(caught.value)
        assert "env-server cleanup failed (3)" in str(caught.value)
    assert harness.server.waited and harness.server.terminated


@pytest.mark.parametrize("cleanup_exit", [-coordinator.signal.SIGTERM, 3])
async def test_runtime_error_survives_server_shutdown(harness, cleanup_exit):
    harness.progress_offset = 1
    harness.server_close_exit = cleanup_exit
    with pytest.raises(RuntimeError) as caught:
        await harness.run()
    assert "Agent progress belongs to a different episode" in str(caught.value)
    if cleanup_exit == 3:
        assert "env-server cleanup failed (3)" in str(caught.value)
    else:
        assert "cleanup failed" not in str(caught.value)
    assert harness.server.waited and not harness.container_present


async def test_server_close_timeout_kills_only_owned_process(harness):
    harness.hang_server_close = True
    with pytest.raises(RuntimeError, match="env-server"):
        await harness.run()
    assert harness.server.killed and harness.server.waited >= 2
    assert harness.other_container_live


@pytest.mark.parametrize("server_exit", [0, -coordinator.signal.SIGTERM])
async def test_cancellation_stops_agent_before_server(harness, server_exit):
    harness.server_close_exit = server_exit
    harness.hold_stream = True
    task = asyncio.create_task(harness.run())
    await harness.stream_entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not harness.owned_live
    assert harness.agent.returncode is not None and harness.server.waited
    assert harness.events.index(("spawn", "stop")) < harness.events.index(("terminate", "server"))
    assert harness.other_container_live


@pytest.mark.parametrize("kind", ["server", "create", "agent"])
async def test_cancellation_during_process_creation_drains_then_closes_handle(harness, kind):
    harness.block_spawn = kind
    task = asyncio.create_task(harness.run())
    await harness.spawn_entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    harness.spawn_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert harness.server.waited and harness.server.returncode is not None
    if kind == "agent":
        assert harness.agent.returncode is not None and not harness.owned_live
    assert not harness.container_present
    assert harness.other_container_live


async def test_stop_failure_does_not_skip_server_cleanup(harness):
    harness.hold_stream = True
    harness.stop_exit = 1
    task = asyncio.create_task(harness.run())
    await harness.stream_entered.wait()
    task.cancel()
    with pytest.raises((asyncio.CancelledError, RuntimeError)):
        await task
    assert harness.server.waited
    assert not harness.owned_live
    assert not harness.container_present
    assert harness.other_container_live


async def test_cancelled_create_communication_still_obtains_and_removes_owned_cid(harness):
    harness.block_communicate = "create"
    task = asyncio.create_task(harness.run())
    await harness.communicate_entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    harness.communicate_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert harness.agent is None
    assert not harness.container_present
    assert harness.server.waited


async def test_failed_create_does_not_stop_an_existing_name(harness):
    harness.create_exit = 125
    with pytest.raises(RuntimeError):
        await harness.run()
    assert not any(command[:2] in (["docker", "stop"], ["docker", "rm"])
                   for command, _ in harness.commands)
    assert harness.agent is None and harness.other_container_live
    assert harness.server.waited


async def test_receipt_supplies_final_statistics_when_stdout_progress_is_missing(harness):
    harness.emit_progress = False
    details = {"turns": 5, "timing": {"predict": 2.0}}
    await harness.run(details)
    assert details["turns"] == 7
    assert details["timing"]["predict"] == 5.0
    assert details["timing"]["act"] == 0.4


async def test_wrong_episode_progress_is_rejected_and_resources_are_released(harness):
    harness.progress_offset = 1
    with pytest.raises(RuntimeError, match="episode"):
        await harness.run()
    assert harness.artifacts.events == []
    assert not harness.owned_live and not harness.container_present
    assert harness.server.waited


@pytest.mark.parametrize("operation", ["stop", "rm"])
async def test_cleanup_spawn_failure_still_settles_agent_and_server(harness, operation):
    harness.fail_spawn = operation
    harness.hold_stream = True
    task = asyncio.create_task(harness.run())
    await harness.stream_entered.wait()
    task.cancel()
    with pytest.raises(RuntimeError):
        await task
    assert harness.server.waited
    assert harness.agent.returncode is not None and not harness.owned_live
    if operation == "stop":
        assert not harness.container_present
    assert harness.other_container_live


async def test_repeated_sigterm_cannot_interrupt_main_cleanup(tmp_path, monkeypatch):
    job_path = tmp_path / "job.json"
    control_path = tmp_path / "control.json"
    connection_path = tmp_path / "connection.json"
    job_path.write_text(json.dumps({
        "rollout_args": ["--release", "osworld-v2.1", "--tasks", "054"],
    }))
    control_path.write_text(json.dumps({"token": "test-only-worker-token"}))
    replacements = {
        "/opt/osworld-job/job.json": job_path,
        "/run/osworld-worker/control.json": control_path,
        "/tmp/osworld-connection.json": connection_path,
    }
    original_path = Path
    monkeypatch.setattr(
        coordinator, "Path", lambda value: replacements.get(str(value), original_path(value))
    )
    monkeypatch.setenv("CUA_AGENT_IMAGE", IMAGE)
    monkeypatch.setenv("OSWORLD_AGENT_RESULTS_ROOT", str(tmp_path / "results"))
    handlers = {}
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        loop, "add_signal_handler", lambda signum, callback: handlers.update({signum: callback})
    )
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    completed = []

    async def run(_args, _execute):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            completed.append(True)

    monkeypatch.setattr(coordinator.rollout, "run", run)
    task = asyncio.create_task(coordinator.main())
    await entered.wait()
    handlers[coordinator.signal.SIGTERM]()
    await cleaning.wait()
    handlers[coordinator.signal.SIGTERM]()
    handlers[coordinator.signal.SIGINT]()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed == [True]
