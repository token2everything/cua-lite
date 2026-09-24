#!/usr/bin/env python3
"""Exercise CUA-Lite GUI/reset/evaluate/close against an idle OSWorld Worker.

This checks environment transport, not model performance or task success.
The caller owns the Sandbox lifecycle. No model endpoint or shell tool is used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import lite.gym as gym
from lite.core.tools.calls import make_tool_call
from lite.utils.image import decode_image

from .artifacts import DEFAULT_ROOT, Experiment, redact
from .catalog import DEFAULT_RELEASE, load_release
from .worker_client import WorkerClient


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--connection", required=True, help="Private Worker connection JSON")
    parser.add_argument("--task", default="054", help="Single-phase task in the selected release")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--results-root", default=str(DEFAULT_ROOT))
    return parser.parse_args(argv)


async def run(args):
    release = load_release(args.release)
    artifacts = Experiment(
        args.experiment, args.results_root, config={"kind": "smoke"}, release=release.name
    ).new_run(args.task)
    report = {"transport_passed": False, "task_solution_attempted": False}
    artifacts.update(closed=False, smoke=report)
    connection_kwargs = (
        {} if os.environ.get("CUA_LITE_ENV_SERVER_URL") else {"connection_file": args.connection}
    )
    env = None
    cleanup_attempted = False
    started = time.monotonic()
    print(json.dumps({"status": "resetting", "env": f"{release.env_id}@{args.task}"}), flush=True)
    try:
        env = gym.make(
            f"{release.env_id}@{args.task}",
            **connection_kwargs,
            max_steps=5,
            post_action_delay=1.0,
            experiment=artifacts.experiment.name,
            run_id=artifacts.run_id,
        )
        observation = await env.reset()
        print(
            json.dumps(
                {
                    "status": "ready",
                    "run_id": artifacts.run_id,
                    "size": decode_image(observation.image).size,
                    "reset_seconds": round(time.monotonic() - started, 2),
                }
            ),
            flush=True,
        )
        for index, keys in enumerate((["meta"], ["esc"]), start=1):
            result = await env.step([make_tool_call("key", {"keys": keys}, call_id=f"key-{index}")])
            if result.terminated or result.truncated or any(r.error for r in result.results):
                raise RuntimeError("GUI smoke failed; inspect the recorded tool feedback")
            print(
                json.dumps(
                    {
                        "status": "gui_step",
                        "keys": keys,
                        "screenshots": len(result.results[0].images),
                    }
                ),
                flush=True,
            )
        result = await env.step(
            [
                make_tool_call(
                    "terminate",
                    {
                        "status": "failure",
                        "reason": "Environment smoke only; no task solution attempted",
                    },
                    call_id="finish",
                )
            ]
        )
        if not result.terminated or result.reward is None:
            raise RuntimeError("Worker did not produce a final evaluation")
        report.update(
            transport_passed=True,
            elapsed_seconds=round(time.monotonic() - started, 2),
        )
        artifacts.update(score=result.reward, smoke=report)
        cleanup_attempted = True
        await env.close()
        connection = json.loads(Path(args.connection).read_text())
        artifacts.secrets.append(connection["token"])
        client = WorkerClient(connection["endpoint"], connection["token"])
        await asyncio.to_thread(
            client.wait_closed, artifacts.run_id, connection["worker_id"], release.name
        )
    except BaseException as error:
        closed = False
        cleanup_error = None
        if env is not None and not cleanup_attempted:
            try:
                await env.close()
                connection = json.loads(Path(args.connection).read_text())
                artifacts.secrets.append(connection["token"])
                client = WorkerClient(connection["endpoint"], connection["token"])
                health = await asyncio.to_thread(client.request, "GET", "/healthz")
                if health["run_id"] != artifacts.run_id:
                    raise RuntimeError(
                        "Worker did not accept this run; no cleanup receipt is available"
                    )
                await asyncio.to_thread(
                    client.wait_closed, artifacts.run_id, connection["worker_id"], release.name
                )
                closed = True
            except Exception as cleanup:
                cleanup_error = redact(str(cleanup), artifacts.secrets)
                error.add_note(f"Worker cleanup also failed: {cleanup_error}")
        artifacts.finish(
            "interrupted" if isinstance(error, asyncio.CancelledError) else "failed",
            closed=closed,
            error=redact(str(error), artifacts.secrets),
            cleanup_error=cleanup_error,
        )
        raise
    record = artifacts.finish("completed", closed=True)
    print(
        json.dumps({**record, "run_dir": str(artifacts.path)}, ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
