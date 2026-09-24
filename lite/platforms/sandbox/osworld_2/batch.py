"""OSWorld task planning and completion adapter for the shared Sandbox batch queue."""

from __future__ import annotations

import copy
import json
import urllib.error
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

    state = batch.prepare(args, deployment, build_plan)
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
            self.config,
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
        return "agent_exit_code" in item

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
