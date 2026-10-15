"""Run benchmark phases beside the Worker, using a separate generic Agent image.

The Sandbox launcher mounts job.json and the Worker control record here. The
model configuration remains on the outer host: only the Agent container receives
it. This process owns the native env-server and every Agent container it starts.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import secrets
import signal
import sys
import tempfile
import urllib.request
from pathlib import Path

from lite.gym.envs.osworld_2 import rollout
from lite.gym.envs.osworld_2.catalog import load_release
from lite.platforms.sandbox.client import NoRedirect, write_json


async def execute_episode(specification, artifacts, details, *, args, image):
    """Settle the Agent and its phase handle before returning control to OSWorld."""
    index = specification["episode_index"]
    name = f"cua-agent-{artifacts.run_id}-{index}"
    token = secrets.token_urlsafe(32)
    endpoint = "http://127.0.0.1:30100"
    server_env = dict(os.environ, OSWORLD_WORKER_CONNECTION=str(args.connection))
    for key in ("CUA_LITE_ENV_SERVER_URL", "CUA_LITE_ENV_SERVER_TOKEN"):
        server_env.pop(key, None)
    command = [
        sys.executable, "-m", "lite.gym.remote.serve", "--host", "127.0.0.1",
        "--port", "30100", "--token", token, "--env-ids", load_release(args.release).env_id,
        "--max-live-envs", "1", "--idle-ttl-sec", str(max(3600, args.model_timeout * 3 + 300)),
    ]
    log_dir = artifacts.path / "model" / f"episode_{index:03d}"
    log_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path("/run/cua-episodes")
    private_root.mkdir(exist_ok=True, mode=0o700)
    turn_offset = details.get("turns", 0)
    timing_before = dict(details.get("timing", {}))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    server = agent = None
    container_id = None
    pending = None
    pending_kind = None
    primary_error = None
    with tempfile.TemporaryDirectory(dir=private_root) as private, (
        log_dir / "env-server.log"
    ).open("a") as server_log:
        episode_file = Path(private) / "episode.json"
        write_json(episode_file, {
            **specification,
            "server": {"url": endpoint, "token": token},
            "model_config": "/private/model-api.json",
        })
        try:
            pending_kind = "server"
            pending = asyncio.create_task(asyncio.create_subprocess_exec(
                *command, env=server_env, stdout=server_log, stderr=server_log,
            ))
            server = await asyncio.shield(pending)
            pending = None
            for _ in range(120):
                if server.returncode is not None:
                    raise RuntimeError("Native env-server failed; inspect env-server.log")
                request = urllib.request.Request(
                    endpoint + "/envs/" + load_release(args.release).env_id,
                    headers={"Authorization": "Bearer " + token},
                )
                try:
                    response = await asyncio.to_thread(opener.open, request, timeout=2)
                    with response:
                        available = json.load(response)["available"]
                    if not available:
                        raise RuntimeError("Requested benchmark is unavailable on the env-server")
                    break
                except OSError:
                    await asyncio.sleep(0.5)
            else:
                raise TimeoutError("Native env-server readiness timed out")

            # Sources refer to the outer Docker host. The trusted coordinator
            # does not mount the model file or task assets in its own filesystem.
            command = [
                "docker", "create", "--pull", "never", "--name", name,
                "--network", "host", "--memory", "8g", "--memory-swap", "8g",
                "--volume", f"{episode_file}:/private/episode.json:ro",
                "--volume", "/opt/osworld-job/model-api.json:/private/model-api.json:ro",
                "--volume", f"/results/agent-results:{args.results_root}:rw",
                "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONUNBUFFERED=1",
                "--env", "NO_PROXY=localhost,127.0.0.1,::1",
                "--env", "no_proxy=localhost,127.0.0.1,::1",
            ]
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                        "http_proxy", "https_proxy", "all_proxy"):
                if key in os.environ:
                    command.extend(("--env", key))
            command.extend((image, "--job", "/private/episode.json"))
            async def create_agent():
                process = await asyncio.create_subprocess_exec(
                    *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                output, _ = await process.communicate()
                if process.returncode:
                    raise RuntimeError(f"Agent container creation failed ({process.returncode})")
                identifier = output.decode().strip()
                if not re.fullmatch(r"[a-f0-9]{64}", identifier):
                    raise RuntimeError("Docker did not return the created Agent container ID")
                return identifier

            # Docker process creation and container creation are separate. Wait
            # for the actual ID even on cancellation; never clean up by name.
            pending_kind = "container"
            pending = asyncio.create_task(create_agent())
            container_id = await asyncio.shield(pending)
            pending = None
            pending_kind = "agent"
            pending = asyncio.create_task(asyncio.create_subprocess_exec(
                "docker", "start", "--attach", container_id,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                limit=1024 * 1024,
            ))
            agent = await asyncio.shield(pending)
            pending = None
            with (log_dir / "agent.log").open("a") as log:
                async for line in agent.stdout:
                    text = line.decode(errors="replace")
                    log.write(text)
                    log.flush()
                    try:
                        event = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and event.get("event") == "agent_step":
                        if event["episode_index"] != index:
                            raise RuntimeError("Agent progress belongs to a different episode")
                        progress = {key: value for key, value in event.items() if key != "event"}
                        artifacts.event("agent_step", **progress)
                        details["turns"] = turn_offset + progress["turn"]
                        timing = details.setdefault("timing", {})
                        for key, seconds in (progress.get("seconds") or {}).items():
                            timing[key] = timing.get(key, 0.0) + seconds
                        artifacts.update(**details)
                        print(json.dumps(event), flush=True)
            code = await agent.wait()
            result_file = log_dir / "episode.json"
            if not result_file.exists():
                raise RuntimeError(
                    f"Agent exited ({code}) without an episode result; see agent.log"
                )
            result = json.loads(result_file.read_text())
            if result["status"] != "completed" or not result["closed"]:
                primary_error = result.get("error", "Agent did not close its episode")
            if code and result["status"] == "completed":
                raise RuntimeError(f"Agent reported completion but exited with status {code}")
            details["turns"] = turn_offset + result["turns"]
            details["timing"] = dict(timing_before)
            for key, seconds in result["timing"].items():
                details["timing"][key] = timing_before.get(key, 0.0) + seconds
            artifacts.update(**details)
            return result
        except BaseException as error:
            primary_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            async def cleanup():
                nonlocal server, agent, container_id
                errors = []
                # A cancelled process creation can still complete. Drain it
                # before stopping anything so no late-created handle escapes.
                if pending is not None:
                    try:
                        process = await pending
                        if pending_kind == "server":
                            server = process
                        elif pending_kind == "container":
                            container_id = process
                        else:
                            agent = process
                    except Exception as error:
                        errors.append(f"{pending_kind} creation failed: {type(error).__name__}")
                if container_id is not None:
                    # Stop the exact container this phase created, then wait for
                    # its Docker CLI. A CLI exit alone is not a close receipt.
                    try:
                        stop = await asyncio.create_subprocess_exec(
                            "docker", "stop", "--time", "45", container_id,
                            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                        )
                        code = await asyncio.wait_for(stop.wait(), timeout=60)
                        if code:
                            errors.append(f"Agent container stop failed ({code})")
                    except TimeoutError:
                        stop.kill()
                        await stop.wait()
                        errors.append("Agent container stop timed out")
                    except OSError as error:
                        errors.append(f"Agent container stop failed: {type(error).__name__}")
                    if agent is not None and agent.returncode is None:
                        try:
                            await asyncio.wait_for(agent.wait(), timeout=15)
                        except TimeoutError:
                            agent.kill()
                            await agent.wait()
                            errors.append("Agent Docker CLI did not settle after container stop")
                    try:
                        remove = await asyncio.create_subprocess_exec(
                            "docker", "rm", "--force", container_id,
                            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                        )
                        code = await asyncio.wait_for(remove.wait(), timeout=30)
                        if code:
                            errors.append(f"Agent container removal failed ({code})")
                    except TimeoutError:
                        remove.kill()
                        await remove.wait()
                        errors.append("Agent container removal timed out")
                    except OSError as error:
                        errors.append(f"Agent container removal failed: {type(error).__name__}")
                if server is not None:
                    already_exited = server.returncode is not None
                    termination_sent = False
                    if not already_exited:
                        try:
                            server.terminate()
                        except ProcessLookupError:
                            pass
                        else:
                            termination_sent = True
                        try:
                            await asyncio.wait_for(server.wait(), timeout=args.cleanup_timeout)
                        except TimeoutError:
                            server.kill()
                            await server.wait()
                            errors.append("env-server did not confirm phase cleanup")
                    if already_exited:
                        errors.append(
                            f"env-server exited before phase cleanup ({server.returncode})"
                        )
                    elif server.returncode and not (
                        termination_sent and server.returncode == -signal.SIGTERM
                    ):
                        # Uvicorn may re-raise the signal after graceful shutdown.
                        # Only our own SIGTERM is an expected process exit.
                        errors.append(f"env-server cleanup failed ({server.returncode})")
                if errors:
                    raise RuntimeError("; ".join(errors))

            # Repeated signals must not interrupt cleanup or permit VM reuse
            # while an Agent or env-server still owns this phase.
            closing = asyncio.create_task(cleanup())
            cancelled = False
            while True:
                try:
                    await asyncio.shield(closing)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                except Exception as error:
                    if primary_error is not None:
                        raise RuntimeError(
                            f"{primary_error}; episode cleanup failed: {error}"
                        ) from error
                    raise
            if cancelled:
                raise asyncio.CancelledError


async def main():
    """Read the mounted job, validate the live Worker, and coordinate its tasks."""
    image = os.environ["CUA_AGENT_IMAGE"]
    if not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", image):
        raise ValueError("CUA_AGENT_IMAGE must be a published image digest")
    job_path = Path("/opt/osworld-job/job.json")
    job = json.loads(job_path.read_text())
    control = json.loads(Path("/run/osworld-worker/control.json").read_text())
    connection = Path("/tmp/osworld-connection.json")
    write_json(connection, {**control, "endpoint": "http://127.0.0.1:8800"})
    root = Path(os.environ["OSWORLD_AGENT_RESULTS_ROOT"])
    args = rollout._parse_args([
        "--job", str(job_path), "--connection", str(connection),
        "--results-root", str(root), "--environment-results-root", str(root / "_environment"),
        *job["rollout_args"],
    ])
    # This process uses its local catalog; only the generic Agent is a client.
    for key in ("CUA_LITE_ENV_SERVER_URL", "CUA_LITE_ENV_SERVER_TOKEN"):
        os.environ.pop(key, None)
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    def cancel():
        if not task.cancelling():
            task.cancel()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, cancel)
    await rollout.run(args, functools.partial(execute_episode, args=args, image=image))


if __name__ == "__main__":
    asyncio.run(main())
