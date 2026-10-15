"""Run one API Agent episode against a native remote Gym environment.

Usage: python -m lite.agents.episode --job /private/episode.json

The job supplies the remote env key, connection, model profile and output
directory. Benchmark coordination and resource ownership stay with the caller.
Only the active episode's images enter memory; earlier viewer messages retain
their archived image references when the task's single JSONL row is refreshed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit

import lite.gym as gym
from lite.agents.core.agent.jsonl_logger import JsonlTrajectoryLogger
from lite.agents.core.agent.utils.request_logs import redact, write_log
from lite.agents.extensions.api_config import load_profile
from lite.agents.extensions.chat_gateway.agent import ChatGatewayAgent
from lite.agents.extensions.responses_gateway.agent import ResponsesGatewayAgent


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    return parser.parse_args(argv)


def _validate_job(job: dict) -> None:
    required = {
        "env", "server", "model_config", "output_dir", "episode_index",
        "max_steps", "model_timeout",
    }
    optional = {"env_kwargs", "model_profile", "history_turns", "metadata"}
    if not isinstance(job, dict) or required - job.keys() or job.keys() - required - optional:
        raise ValueError("Episode job must contain the documented required and optional fields")
    key = job["env"]
    if not isinstance(key, str) or key.count("@") != 1 or not all(key.split("@")):
        raise ValueError("env must be a complete env_id@task_id string")
    server = job["server"]
    if not isinstance(server, dict) or set(server) != {"url", "token"}:
        raise ValueError("server must contain a nonempty url and token")
    if not all(isinstance(value, str) and value.strip() for value in server.values()):
        raise ValueError("server must contain a nonempty url and token")
    url = urlsplit(server["url"])
    if (url.scheme not in {"http", "https"} or not url.netloc or url.username
            or url.password or url.query or url.fragment):
        raise ValueError("server.url must be an HTTP(S) URL without embedded credentials")
    for name in ("model_config", "output_dir"):
        if not isinstance(job[name], str) or not Path(job[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute container path")
    for name in ("env_kwargs", "metadata"):
        if not isinstance(job.get(name, {}), dict):
            raise ValueError(f"{name} must be a JSON object")
    if {"env_server_url", "token"} & job.get("env_kwargs", {}).keys():
        raise ValueError("Connection routing belongs in server, not env_kwargs")
    if "model_profile" in job and job["model_profile"] is not None and (
        not isinstance(job["model_profile"], str) or not job["model_profile"].strip()
    ):
        raise ValueError("model_profile must be a nonempty string or null")
    for name, minimum in (("episode_index", 0), ("max_steps", 1)):
        if type(job[name]) is not int or job[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    timeout = job["model_timeout"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("model_timeout must be finite and positive")
    history = job.get("history_turns", 4)
    if history is not None and (type(history) is not int or history < 0):
        raise ValueError("history_turns must be a nonnegative integer or null")


def _previous_trace(directory: Path, index: int) -> dict:
    path = directory / "agent_trajectory.jsonl"
    if not path.exists():
        if index != 0:
            raise ValueError("The first episode_index must be zero")
        return {"messages": [], "metadata": {"episodes": []}}
    lines = path.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError("agent_trajectory.jsonl must contain exactly one task row")
    row = json.loads(lines[0])
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
        raise ValueError("Task trajectory must contain a messages array")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("episodes"), list):
        raise ValueError("Existing task trajectory must contain metadata.episodes")
    episodes = metadata["episodes"]
    offset = 0
    for number, episode in enumerate(episodes):
        if (not isinstance(episode, dict) or episode.get("episode_index") != number
                or episode.get("message_start") != offset
                or type(episode.get("message_end")) is not int
                or not offset <= episode["message_end"] <= len(row["messages"])):
            raise ValueError("Existing episode boundaries must be contiguous")
        offset = episode["message_end"]
    if index != len(episodes) or offset != len(row["messages"]):
        raise ValueError("episode_index must continue the existing task trajectory")
    return row


class _EpisodeLogger(JsonlTrajectoryLogger):
    """Append an isolated episode's messages to the task's live viewer row."""

    def __init__(self, directory, previous, job, details, secrets):
        super().__init__(directory, model_id="", secrets=secrets, absolute_image_paths=True)
        self.previous = previous
        self.index = job["episode_index"]
        self.context = job.get("metadata", {})
        self.details = details
        self.lock = threading.RLock()
        self.sealed = False
        self._row = {
            "messages": [], "for_agent": True, "source": ["cua-lite"],
            "metadata": {"status": "running"},
        }

    def _publish(self):
        with self.lock:
            current = self._row
            offset = len(self.previous["messages"])
            end = offset + len(current["messages"])
            boundary = {
                **self.context, "episode_index": self.index, "context_reset": True,
                "message_start": offset, "message_end": end,
            }
            self.details.update(message_start=offset, message_end=end)
            self._row = {
                **current,
                "messages": self.previous["messages"] + current["messages"],
                "metadata": {
                    **self.previous["metadata"], **current["metadata"],
                    "status": "running", "trace_scope": "task_model_messages",
                    "episodes": [*self.previous["metadata"]["episodes"], boundary],
                },
            }
            try:
                super()._publish()
            finally:
                self._row = current

    def on_trajectory(self, trajectory):
        with self.lock:
            if not self.sealed:
                super().on_trajectory(trajectory)

    def on_step(self, data):
        with self.lock:
            if self.sealed:
                return
            progress = {
                "event": "agent_step", "episode_index": self.index,
                "turn": data.step_idx + 1, "reward": data.step_result.reward,
                "terminated": data.step_result.terminated,
                "truncated": data.step_result.truncated, "seconds": data.timings,
            }
            self.details.update(
                turns=data.step_idx + 1, terminated=data.step_result.terminated,
                truncated=data.step_result.truncated,
            )
            self.details["episode_return"] += data.step_result.reward or 0.0
            for name, seconds in (data.timings or {}).items():
                timing = self.details["timing"]
                timing[name] = timing.get(name, 0.0) + seconds
            print(json.dumps(redact(progress, self.secrets), allow_nan=False), flush=True)

    def on_complete(self, sample):
        with self.lock:
            if not self.sealed:
                super().on_complete(sample)


async def run(job: dict) -> dict:
    """Run one remote episode, persist its receipt, and drain cancelled creation."""
    _validate_job(job)
    directory = Path(job["output_dir"])
    previous = _previous_trace(directory, job["episode_index"])
    model_dir = directory / "model" / f"episode_{job['episode_index']:03d}"
    if (model_dir / "episode.json").exists():
        raise ValueError("episode.json already exists; choose the next episode_index")
    # The coordinator may already own agent.log/env-server.log in this directory.
    model_dir.mkdir(parents=True, exist_ok=True)
    details = {
        "status": "running", "closed": False, "turns": 0, "episode_return": 0.0,
        "terminated": False, "truncated": False,
        "message_start": len(previous["messages"]), "message_end": len(previous["messages"]),
        "model_id": None, "api_type": None, "timing": {},
        "history_turns": job.get("history_turns", 4),
    }
    secrets = [job["server"]["token"]]
    # Catalog discovery and construction must use the same remote route. The
    # CLI owns this process; restore its caller's routing for in-process tests.
    routing = {
        "CUA_LITE_ENV_SERVER_URL": job["server"]["url"],
        "CUA_LITE_ENV_SERVER_TOKEN": job["server"]["token"],
    }
    original_routing = {key: os.environ.get(key) for key in routing}
    original_factory = logging.getLogRecordFactory()

    def safe_record(*args, **kwargs):
        record = original_factory(*args, **kwargs)
        record.msg = redact(record.getMessage(), secrets)
        record.args = ()
        if record.exc_info:
            record.exc_text = redact("".join(traceback.format_exception(*record.exc_info)), secrets)
            record.exc_info = None
        if record.stack_info:
            record.stack_info = redact(record.stack_info, secrets)
        return record

    logging.setLogRecordFactory(safe_record)
    os.environ.update(routing)
    env = pending = trajectory = None
    cancelled = False
    started = time.monotonic()
    try:
        trajectory = _EpisodeLogger(directory, previous, job, details, secrets)
        trajectory._publish()
        # Resolve the private model without logging the job or profile object.
        config = load_profile(job["model_config"], job.get("model_profile"))
        secrets.append(config["api_key"])
        trajectory.secrets = tuple(secrets)
        trajectory.model_id = config["model"]
        trajectory._row["metadata"]["model_id"] = config["model"]
        details.update(model_id=config["model"], api_type=config["api_type"])
        if config["api_type"] == "openai_chat":
            agent_class = ChatGatewayAgent
            history = {"history_turns": details["history_turns"]}
        else:
            if job.get("history_turns") is not None:
                raise ValueError(
                    "Responses gateway requires full history; omit history_turns or use null"
                )
            agent_class, history = ResponsesGatewayAgent, {}
            details["history_turns"] = None

        # A cancelled await cannot abandon an HTTP create still running in its
        # thread. Cleanup below drains this shielded task and closes its handle.
        pending = asyncio.create_task(asyncio.to_thread(
            gym.make, job["env"], env_server_url=job["server"]["url"],
            **job.get("env_kwargs", {}),
        ))
        env = await asyncio.shield(pending)
        pending = None
        agent = agent_class(
            model_id=config["model"], api_base=config["base_url"], api_key=config["api_key"],
            metadata=env.metadata, request_params=config.get("params", {}),
            request_timeout=job["model_timeout"], api_retry_max=2,
            request_log_dir=model_dir, request_image_dir=directory / "images", **history,
        )
        trajectory.get_tools = agent._build_tools
        trajectory._row["metadata"].update(
            environment=env.metadata.to_dict(), tools=agent._build_tools(),
        )
        trajectory._publish()
        result = await agent.sample(env, max_steps=job["max_steps"], hooks=[trajectory])
        details.update(
            turns=len(result.steps), episode_return=result.episode_return,
            terminated=result.terminated, truncated=result.truncated,
        )
        # Sampling attempts close but may log its failure. Require an explicit
        # acknowledgement before the coordinator can reuse the remote slot.
        await env.close()
        details.update(status="completed", closed=True)
    except asyncio.CancelledError:
        cancelled = True
        details.update(status="failed", error="Episode was cancelled")
    except Exception as error:
        details.update(status="failed", error=redact(str(error), secrets))
    finally:
        async def finish():
            nonlocal env
            if pending is not None:
                try:
                    env = await pending
                except Exception as error:
                    details.update(status="failed", error=redact(str(error), secrets))
            if env is not None and not details["closed"]:
                try:
                    await env.close()
                    details["closed"] = True
                except Exception as error:
                    details.update(
                        status="failed",
                        error=redact(
                            f"{details.get('error', 'Episode cleanup failed')}; close: {error}",
                            secrets,
                        ),
                    )
            if trajectory is not None:
                with trajectory.lock:
                    trajectory.sealed = True
                    trajectory._publish()
            details["timing"]["total"] = time.monotonic() - started
            write_log(model_dir, "episode.json", details, secrets)
            print(json.dumps(redact({"event": "episode_complete", **details}, secrets),
                             ensure_ascii=False, allow_nan=False), flush=True)

        cleanup = asyncio.create_task(finish())
        try:
            while True:
                try:
                    await asyncio.shield(cleanup)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if not cleanup.done():
                        details.update(status="failed", error="Episode was cancelled")
        finally:
            for key, value in original_routing.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            logging.setLogRecordFactory(original_factory)
    if cancelled:
        raise asyncio.CancelledError
    return redact(details, secrets)


async def _run_cli(job: dict) -> dict:
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()

    def cancel():
        if not task.cancelling():
            task.cancel()

    loop.add_signal_handler(signal.SIGTERM, cancel)
    try:
        return await run(job)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main(argv=None) -> int:
    """Read the private job and emit public JSON events, never its contents."""
    args = _parse_args(argv)
    try:
        job = json.loads(args.job.read_text())
        result = asyncio.run(_run_cli(job))
    except asyncio.CancelledError:
        return 130
    except (OSError, ValueError) as error:
        print(json.dumps({"event": "episode_error", "error": str(error)}), flush=True)
        return 1
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
