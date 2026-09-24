"""Resolve a local run YAML into the existing Sandbox Agent job contract.

Used by ``manage plan`` and ``manage submit``. Resolution reads only the packaged
task catalog; credentials, private model files and platform access are deferred
until submission.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lite.gym.envs.osworld_2.catalog import Release
from lite.platforms.sandbox.client import safe_component


@dataclass(frozen=True)
class Job:
    experiment: str
    model_config: Path
    model_profile: str | None
    results_root: Path
    rollout_args: list[str]
    selection: dict

    @property
    def afs_path(self) -> str:
        return f"private/osworld-agent-jobs/{self.experiment}"

    @classmethod
    def from_config(cls, value: dict, directory: Path, release: Release) -> Job:
        """Validate YAML types before translating values into runner arguments."""
        allowed = {
            "model_config",
            "model_profile",
            "results_root",
            "experiment",
            "tasks",
            "task_set",
            "max_steps",
            "model_timeout",
            "cleanup_timeout",
            "history_turns",
            "full_history",
            "skip_unsupported",
            "verify_protocols",
        }
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError("job must be a mapping with only documented run settings")
        if not isinstance(value.get("model_config"), str) or not value["model_config"].strip():
            raise ValueError("job.model_config must be a nonempty path string")
        for key in ("model_profile", "results_root", "experiment", "task_set"):
            if key in value and (not isinstance(value[key], str) or not value[key].strip()):
                raise ValueError(f"job.{key} must be a nonempty string")
        tasks = value.get("tasks")
        if "tasks" in value and (
            not isinstance(tasks, list) or any(not isinstance(task, str) for task in tasks)
        ):
            raise ValueError('job.tasks must contain quoted strings, for example ["054"]')
        for key in ("full_history", "skip_unsupported", "verify_protocols"):
            if key in value and not isinstance(value[key], bool):
                raise ValueError(f"job.{key} must be true or false")
        for key in ("max_steps", "history_turns", "model_timeout", "cleanup_timeout"):
            if key not in value:
                continue
            number = value[key]
            integer = key in ("max_steps", "history_turns")
            if (
                isinstance(number, bool)
                or not isinstance(number, int if integer else (int, float))
                or not math.isfinite(number)
                or (number < 0 if key == "history_turns" else number <= 0)
            ):
                raise ValueError(f"Invalid job.{key}")
        if value.get("full_history") and "history_turns" in value:
            raise ValueError("Choose job.history_turns or job.full_history")
        selection = release.plan(
            tasks=tasks,
            task_set=value.get("task_set"),
            skip_unsupported=value.get("skip_unsupported", False),
            verify_protocols=value.get("verify_protocols", False),
        )
        experiment = safe_component(
            value.get("experiment")
            or (f"{release.name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}")
        )
        paths = {}
        for key, default in (("model_config", None), ("results_root", "results")):
            path = Path(value.get(key, default)).expanduser()
            paths[key] = (directory / path).resolve()
        if str(paths["results_root"]) == "/" or ":" in str(paths["results_root"]):
            raise ValueError("job.results_root must be a directory other than '/' without ':'")
        argv = ["--release", release.name, "--experiment", experiment]
        if "task_set" in value:
            argv.extend(("--task-set", value["task_set"]))
        else:
            argv.extend(("--tasks", *selection["requested_task_ids"]))
        for key in (
            "model_profile",
            "max_steps",
            "model_timeout",
            "cleanup_timeout",
            "history_turns",
        ):
            if key in value:
                argv.extend(("--" + key.replace("_", "-"), str(value[key])))
        for key in ("full_history", "skip_unsupported", "verify_protocols"):
            if value.get(key):
                argv.append("--" + key.replace("_", "-"))
        return cls(
            experiment=experiment,
            model_config=paths["model_config"],
            model_profile=value.get("model_profile"),
            results_root=paths["results_root"],
            rollout_args=argv,
            selection=selection,
        )

    def summary(self) -> dict:
        """Public plan and submission facts, without opening the private model file."""
        return {
            **self.selection,
            "experiment": self.experiment,
            "agent_job": self.afs_path,
            "model_config_file": str(self.model_config),
            "model_profile": self.model_profile,
            "results_root": str(self.results_root),
            "rollout_args": self.rollout_args,
        }
