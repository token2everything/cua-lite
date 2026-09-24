"""Persistent, bounded Sandbox queue shared by benchmark submission entrypoints.

prepare resolves an offline plan or reloads a saved batch. run owns scheduling,
model snapshots and Sandbox lifetime; the adapter owns submission inputs and
benchmark completion/artifact semantics. No benchmark package is imported here.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

from lite.platforms.sandbox import lifecycle
from lite.platforms.sandbox.client import (
    SandboxAPIError,
    SandboxClient,
    redact,
    state_of,
    write_json,
)
from lite.platforms.sandbox.monitor import BatchMonitor

_FINISHED = {"completed", "failed"}


class BatchAdapter(Protocol):
    """Three benchmark operations; implementations need no inheritance or registration."""

    owner: str

    def submit(self, item: dict) -> dict:
        """Journal inputs, create once and return a Feishu-verified ownership receipt."""
        ...

    def observe(self, item: dict, record: dict) -> bool:
        """Update benchmark progress; return whether a durable exit receipt exists."""
        ...

    def collect(self, item: dict, record: dict) -> None:
        """Mirror artifacts and set workflow_completed, score and any failure details."""
        ...


def prepare(
    args, deployment: dict, build_plan: Callable[[], dict], *, monitor: dict | None = None,
) -> dict:
    """Resolve queue limits and resume identity without reading secrets or contacting Sandbox."""
    if args.resume:
        path = args.state_dir / "batches" / args.resume / "batch.json"
        state = json.loads(path.read_text())
        if state["batch_id"] != args.resume or state["deployment"] != deployment:
            raise ValueError("Resume requires the original batch deployment and state directory")
    else:
        state = {**build_plan(), "batch_run": uuid.uuid4().hex, "deployment": deployment}
    for key, default in (("concurrency", 4), ("poll_interval", 30), ("task_timeout", 14400)):
        value = getattr(args, key)
        if value is None:
            value = state.get(key, default)
        if value < 1:
            raise ValueError(f"{key.replace('_', '-')} must be positive")
        state[key] = value
    defaults = {"enabled": False, "interval": 300, "stall_after": 900}
    if monitor is None:
        monitor = {}
    if not isinstance(monitor, dict) or set(monitor) - set(defaults):
        raise ValueError("monitor accepts only enabled, interval and stall_after")
    settings = {**defaults, **monitor}
    if not isinstance(settings["enabled"], bool):
        raise ValueError("monitor.enabled must be a boolean")
    for key in ("interval", "stall_after"):
        if type(settings[key]) is not int or settings[key] < 60:
            raise ValueError(f"monitor.{key} must be an integer of at least 60 seconds")
    # Runtime reporting settings can change on resume; execution progress remains intact.
    state["monitor"] = settings
    return state


def run(
    args, client: SandboxClient, state: dict, adapter: BatchAdapter,
    *, monitor: BatchMonitor | None = None,
) -> dict:
    """Schedule owned Sandboxes, preserving progress across interruption and API failures."""
    from lite.agents.extensions.api_config import load_profile

    if not state["tasks"]:
        raise ValueError("No tasks remain to submit")
    config = state["deployment"]
    directory = args.state_dir / "batches" / state["batch_id"]
    directory.mkdir(parents=True, exist_ok=bool(args.resume), mode=0o700)
    path = directory / "batch.json"
    # Atomic directory creation also works on AFS. Never steal another scheduler's lock.
    lock = directory / ".scheduler.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f"Batch scheduler already locked: {lock}") from None
    try:
        # Queued tasks retain one selected API profile across restarts/config edits.
        model = load_profile(Path(state["job"]["model_config"]), state["job"].get("model_profile"))
        client.secrets.append(model["api_key"])
        if not args.resume:
            profile = state["job"].get("model_profile")
            snapshot = (
                model
                if profile is None
                else {
                    "default_model": profile,
                    "models": {profile: model},
                }
            )
            model_path = directory / "model-api.json"
            write_json(model_path, snapshot)
            state["job"]["model_config"] = str(model_path)
        write_json(path, state)
        print(
            f"Batch {state['batch_id']}: {len(state['tasks'])} tasks, "
            f"concurrency={state['concurrency']}; state: {path}",
            file=sys.stderr,
            flush=True,
        )

        # Recover only this batch's verified creates. The submission journal is
        # written before upload/POST, so its absence proves preflight never reached create.
        for item in state["tasks"]:
            if item["state"] != "submitting":
                continue
            records = [
                json.loads(file.read_text()) for file in args.state_dir.glob("worker-*.json")
            ]
            matches = [
                r
                for r in records
                if r.get("submission", {}).get("experiment") == item["experiment"]
                and r["submission"].get("batch_run") == state["batch_run"]
            ]
            journal = args.state_dir / "jobs" / item["experiment"] / "submission.json"
            if not matches and not journal.exists():
                item["state"] = "pending"
            elif len(matches) == 1 and matches[0].get("feishu_webhook_verified"):
                item.update(sandbox_id=matches[0]["sandbox_id"], state="running")
            else:
                raise RuntimeError(
                    f"Uncertain submission {item['experiment']}; inspect its worker_run in "
                    f"{journal} and Portal before continuing. No duplicate create was attempted."
                )
            write_json(path, state)

        state.pop("scheduler_error", None)
        if monitor:
            monitor.report(state, time.time(), event="resumed" if args.resume else "started")
            write_json(path, state)
        while any(item["state"] not in _FINISHED for item in state["tasks"]):
            # Settle existing resources before filling slots. Deletion still consumes capacity.
            for item in state["tasks"]:
                if item["state"] in _FINISHED | {"pending"}:
                    continue
                record = lifecycle.read_record(
                    args.state_dir,
                    item["sandbox_id"],
                    client,
                    project=config["project"],
                    owner=adapter.owner,
                )
                if (
                    record["submission"]["experiment"] != item["experiment"]
                    or record["submission"].get("batch_run") != state["batch_run"]
                    or record["submission"]["afs_url"] != client.afs_url
                    or record["submission"]["results_prefix"] != config["results_prefix"]
                ):
                    raise ValueError(
                        "Batch ownership record does not match the original submission"
                    )
                try:
                    current = client.request("GET", "/v1/sandboxes/" + item["sandbox_id"])
                except SandboxAPIError as error:
                    if error.status_code != 404:
                        raise
                    current = None
                if current is not None:
                    lifecycle.check_remote(record, current)
                item["platform_state"] = state_of(current) if current is not None else "absent"
                if current is None and not item.get("collected"):
                    item["state"] = "collecting"
                # A durable benchmark receipt can survive Sandbox expiry. Read it
                # before platform failure/absence is interpreted by the queue.
                finished = adapter.observe(item, record)
                if current is None:
                    item["sandbox_deleted"] = True
                    if item.get("collected"):
                        item["state"] = "completed" if item["workflow_completed"] else "failed"
                        write_json(path, state)
                        print(
                            f"Task {item['task_id']}: {item['state']}; score={item.get('score')}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    item["state"] = "collecting"
                if item["state"] == "running":
                    if finished or item["platform_state"] in {
                        "failed",
                        "terminated",
                        "stopped",
                        "expired",
                    }:
                        item["state"] = "collecting"
                    elif time.time() - item["started_at"] >= state["task_timeout"]:
                        # Stop an unfinished writer before collecting its final artifacts.
                        item.update(state="deleting", error="Task exceeded task-timeout")
                    elif current.get("expiresAt"):
                        expiration = datetime.fromisoformat(
                            current["expiresAt"].replace("Z", "+00:00")
                        )
                        if expiration.timestamp() - time.time() < 600:
                            client.request(
                                "POST",
                                f"/v1/sandboxes/{item['sandbox_id']}/renew-expiration",
                                {"timeout": 7200},
                            )
                write_json(path, state)

                # A collection error stops the scheduler without deleting the resource.
                if item["state"] == "collecting":
                    adapter.collect(item, record)
                    item["collected"] = True
                    item["state"] = (
                        ("completed" if item["workflow_completed"] else "failed")
                        if item.get("sandbox_deleted")
                        else "deleting"
                    )
                    write_json(path, state)
                if item["state"] == "deleting":
                    write_json(path, state)
                    if not item.get("delete_requested"):
                        lifecycle.delete(
                            args.state_dir,
                            item["sandbox_id"],
                            client,
                            project=config["project"],
                            owner=adapter.owner,
                        )
                        item["delete_requested"] = True
                        write_json(path, state)

            # Persist intent before submission; never fill a still-deleting slot.
            active = sum(item["state"] not in _FINISHED | {"pending"} for item in state["tasks"])
            for item in state["tasks"]:
                if active >= state["concurrency"]:
                    break
                if item["state"] != "pending":
                    continue
                if (args.state_dir / "jobs" / item["experiment"]).exists():
                    raise ValueError(f"Child experiment already exists: {item['experiment']}")
                item.update(state="submitting", started_at=time.time())
                write_json(path, state)
                record = adapter.submit(item)
                item.update(sandbox_id=record["sandbox_id"], state="running")
                write_json(path, state)
                active += 1
                print(f"Task {item['task_id']}: {item['sandbox_id']}", file=sys.stderr, flush=True)
            if any(item["state"] not in _FINISHED for item in state["tasks"]):
                if monitor:
                    monitor.report(state, time.time())
                    write_json(path, state)
                time.sleep(state["poll_interval"])

        state.pop("scheduler_error", None)
        if monitor:
            monitor.report(state, time.time(), event="finished")
        write_json(path, state)
        return {
            "batch_id": state["batch_id"],
            "state_file": str(path),
            "completed": sum(item["state"] == "completed" for item in state["tasks"]),
            "failed": sum(item["state"] == "failed" for item in state["tasks"]),
            "tasks": state["tasks"],
        }
    except BaseException as error:
        state["scheduler_error"] = redact(str(error) or type(error).__name__, client.secrets)
        if monitor:
            monitor.report(state, time.time(), event="stopped")
        write_json(path, state)
        print(
            f"Batch stopped; saved state: {path}. Resume with batch --resume {state['batch_id']}",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        lock.rmdir()
