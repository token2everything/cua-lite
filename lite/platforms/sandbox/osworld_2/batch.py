"""OSWorld task planning and completion adapter for the shared Sandbox batch queue."""

from __future__ import annotations

import copy
import json
import re
import time
import urllib.error
from datetime import datetime
from pathlib import Path

from lite.gym.envs.osworld_2.catalog import Release, load_release
from lite.platforms.sandbox import batch
from lite.platforms.sandbox.client import AFSCommandError, SandboxAPIError, redact, safe_component
from lite.platforms.sandbox.osworld_2 import manage
from lite.platforms.sandbox.osworld_2.job import Job


def prepare(args, config: dict, release: Release) -> dict:
    """Project the release/task selection into an offline queue plan."""
    deployment = {
        key: config[key]
        for key in ("project", "base_url", "afs_url", "environment_id", "results_prefix")
    }
    deployment["release"] = release.name
    deployment["task_resources"] = manage.task_resources(config, release)

    def build_plan():
        job = Job.from_config(config.get("job"), args.config.parent, release)
        return {
            "batch_id": job.experiment,
            "selection": job.selection,
            "job": {
                **config["job"],
                "model_config": str(job.model_config),
                "results_root": str(job.results_root),
            },
            "tasks": [
                {
                    "task_id": task,
                    "experiment": safe_component(f"{job.experiment}-task-{task}"),
                    "state": "pending",
                }
                for task in job.selection["scheduled_task_ids"]
            ],
        }

    state = batch.prepare(args, deployment, build_plan, monitor=config.get("monitor"))
    selection = state["selection"]
    if (
        not args.dry_run
        and selection["unsupported"]
        and not (selection["skipped"] or selection["verification_mode"])
    ):
        raise ValueError(
            "Unsupported tasks in batch; select supported tasks or set skip_unsupported"
        )
    return state


class OSWorldBatch:
    """Supply OSWorld launch and receipt semantics without owning the scheduling loop."""

    owner = manage._OWNER

    def __init__(self, args, config: dict, client, state: dict):
        self.args, self.config, self.client, self.state = args, config, client, state

    def submit(self, item: dict) -> dict:
        """Build one release-matched task job and use the existing submission entry."""
        child_config = {
            **self.state["job"],
            "tasks": [item["task_id"]],
            "experiment": item["experiment"],
        }
        child_config.pop("task_set", None)
        release = load_release(self.state["deployment"]["release"])
        directory = self.args.state_dir / "batches" / self.state["batch_id"]
        job = Job.from_config(child_config, directory, release)
        return manage.submit(
            copy.copy(self.args),
            {**self.config, "task_resources": self.state["deployment"]["task_resources"]},
            self.client,
            release.name,
            job,
            batch_run=self.state["batch_run"],
        )

    def observe(self, item: dict, record: dict) -> bool:
        """Discover the Worker and read its durable coordinator exit receipt."""
        if record["release"] != self.state["deployment"]["release"]:
            raise ValueError("Batch ownership record does not match the original submission")
        if (
            item["state"] == "running"
            and item["platform_state"] == "running"
            and "worker_id" not in item
        ):
            try:
                _, health = manage.worker_health(item["sandbox_id"], self.client)
            except (SandboxAPIError, urllib.error.HTTPError) as error:
                code = error.status_code if isinstance(error, SandboxAPIError) else error.code
                if code not in (404, 502, 503, 504):
                    raise
            except (TimeoutError, urllib.error.URLError):
                pass  # Worker boot is bounded by the queue's task timeout.
            else:
                if health["release"] != record["release"]:
                    raise ValueError("Live Worker release differs from batch release")
                item["worker_id"] = safe_component(health["worker_id"])
        if item["state"] in {"running", "collecting"} and "worker_id" in item:
            if "agent_exit_code" not in item:
                source = (
                    f"afs:{record['submission']['results_prefix']}/_workers/{item['worker_id']}"
                )
                if self.client.afs(
                    "lsf", source, "--files-only", "--include", "/agent-exit-code"
                ).strip():
                    receipt = self.client.afs("cat", source + "/agent-exit-code").strip()
                    if receipt:  # The launcher writes this small file non-atomically.
                        item["agent_exit_code"] = int(receipt)
        if self.state["monitor"]["enabled"] and item["state"] == "running":
            self.observe_progress(item, record)
        return "agent_exit_code" in item

    def observe_progress(self, item: dict, record: dict) -> None:
        """Read only small metadata and log tails; retries never count as Agent progress."""
        now = time.time()
        deadline = time.monotonic() + 20

        def read(*command):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Agent progress observation timed out")
            return self.client.afs(
                *command, "--retries", "1", "--low-level-retries", "1", timeout=remaining,
            )

        root = (
            f"afs:{record['submission']['results_prefix']}/agent-results/"
            f"{item['experiment']}/task_{item['task_id']}"
        )
        try:
            if "run_id" not in item:
                directories = read("lsf", root, "--dirs-only").splitlines()
                if not directories:
                    item.pop("progress_error", None)
                    return
                if len(directories) != 1:
                    raise ValueError("More than one run found for this batch task")
                item["run_id"] = safe_component(directories[0].rstrip("/"))
            root += "/" + item["run_id"]
            filenames = set(read(
                "lsf", root, "--recursive", "--files-only",
                "--include", "/run.json", "--include", "/events.jsonl",
                "--include", "/model/episode_*/*.json",
            ).splitlines())
            if "run.json" not in filenames:
                item.pop("progress_error", None)
                return
            saved = json.loads(read("cat", root + "/run.json"))
            if (saved["run_id"], saved["task_id"], saved["release"]) != (
                item["run_id"], item["task_id"], record["release"],
            ):
                raise ValueError("Progress record belongs to a different task/release")
            # AFS writes are not a cross-file snapshot. Events supply step times,
            # while the cumulative counter remains owned by the coordinator.
            events = (
                read("cat", root + "/events.jsonl", "--tail", "65536")
                if "events.jsonl" in filenames else ""
            )
            step = None
            for line in events.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A tail can begin/end in a concurrently appended partial line.
                if isinstance(event, dict) and event.get("method") == "agent_step":
                    step = event
            requests = sorted(name for name in filenames if re.fullmatch(
                r"model/episode_\d{3}/\d{5}\.request\.json", name,
            ))
            previous = item.get("progress", {})
            turns = saved.get("turns", 0)
            progress = {
                "turns": turns, "observed_at": now, "activity": "starting",
                "last_progress_at": previous.get(
                    "last_progress_at", datetime.fromisoformat(saved["started_at"]).timestamp(),
                ),
                "phase_count": saved.get("phase_count", 1),
                "max_steps": self.state["job"].get("max_steps", "?"),
            }
            if step:
                progress.update(
                    last_progress_at=datetime.fromisoformat(step["time"]).timestamp(),
                    phase_index=step["episode_index"], phase_turn=step["turn"],
                )
            elif turns > previous.get("turns", 0):
                # If the bounded tail no longer contains a step, this poll proves
                # advancement but only bounds its time. Never refresh an unchanged count.
                progress["last_progress_at"] = now
            if requests:
                latest = requests[-1].removesuffix(".request.json")
                phase = int(latest.split("/")[1].removeprefix("episode_"))
                if phase > progress.get("phase_index", -1):
                    progress["phase_turn"] = 0
                progress.update(
                    phase_index=max(phase, progress.get("phase_index", 0)),
                    activity=(
                        "executing" if latest + ".response.json" in filenames else "waiting_model"
                    ),
                    retries=len({name.split(".")[1] for name in filenames
                                 if name.startswith(latest + ".attempt-")
                                 and name.endswith((".response.json", ".error.json"))}),
                )
                if f"model/episode_{phase:03d}/episode.json" in filenames:
                    progress["activity"] = "between_phases"
            if saved["status"] != "running":
                progress["activity"] = "finished"
            item["progress"] = progress
            item.pop("progress_error", None)
        except AFSCommandError as error:
            if error.returncode == 3 and "run_id" not in item:
                item.pop("progress_error", None)  # No task directory during initial boot.
            else:
                item["progress_error"] = redact(str(error), self.client.secrets)[:500]
        except (OSError, ValueError, KeyError, TypeError) as error:
            item["progress_error"] = redact(str(error), self.client.secrets)[:500]

    def collect(self, item: dict, record: dict) -> None:
        """Mirror OSWorld artifacts and distinguish completed workflow from model score."""
        client = self.client
        try:
            item.update(manage.sync(record, client))
        except AFSCommandError as error:
            # A failed boot can exit before creating the result directory. Other
            # transfer errors preserve the Sandbox for a later collection attempt.
            if (
                error.returncode != 3
                or "agent_exit_code" in item
                or item["platform_state"]
                not in {"failed", "terminated", "stopped", "expired", "absent"}
            ):
                raise
            item.update(
                results=str(Path(record["submission"]["results_root"]) / item["experiment"]),
                error="Sandbox ended before an Agent result directory was created",
            )
        if "worker_id" in item:
            client.afs(
                "copy",
                f"afs:{record['submission']['results_prefix']}/_workers/{item['worker_id']}",
                str(Path(item["results"]) / "_launcher"),
                "--include",
                "/*.log",
                "--include",
                "/agent-exit-code",
                "--ignore-times",
                "--ignore-size",
                "--multi-thread-streams",
                "0",
            )
        files = list((Path(item["results"]) / f"task_{item['task_id']}").glob("*/run.json"))
        item["workflow_completed"] = False
        if len(files) == 1:
            run_record = json.loads(files[0].read_text())
            if (
                run_record["task_id"] != item["task_id"]
                or run_record["release"] != record["release"]
            ):
                raise ValueError("Mirrored run does not match batch task/release")
            item.update(
                run_id=run_record["run_id"],
                run_status=run_record["status"],
                closed=run_record.get("closed", False),
                score=run_record.get("score"),
            )
            item["workflow_completed"] = (
                not item.get("error")
                and item.get("agent_exit_code") == 0
                and run_record["status"] == "completed"
                and item["closed"]
            )
            if run_record.get("error"):
                item["error"] = redact(run_record["error"], client.secrets)
        if not item["workflow_completed"] and "error" not in item:
            item["error"] = "Agent exit or completed/closed task receipt missing or unsuccessful"
