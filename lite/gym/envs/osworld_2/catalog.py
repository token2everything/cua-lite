"""OSWorld release catalog: version data, runtime bindings and evaluation sets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

DEFAULT_RELEASE = "osworld-v2.1"
_ROOT = Path(__file__).with_name("releases")


@dataclass(frozen=True)
class Release:
    """One official release bound to the shared OSWorld Worker adapter."""

    name: str
    env_id: str
    path: Path
    manifest: dict
    tasks: tuple[str, ...]
    task_sets: dict[str, tuple[str, ...]]
    exclusions: dict[str, str]
    manifest_sha256: str

    @property
    def task_revision(self) -> str:
        """Use the official commit pin when provided, otherwise its release tag."""
        tasks = self.manifest["tasks"]
        return tasks.get("commit") or tasks["tag"]

    def select(self, *, task_set: str | None = None, tasks: list[str] | None = None) -> list[str]:
        """Resolve exactly one requested set or explicit list, without filtering it."""
        if task_set is not None and tasks is not None:
            raise ValueError("Choose task-set or tasks, not both")
        if task_set is not None:
            if task_set not in self.task_sets:
                raise ValueError(f"Unknown task set {task_set!r} for {self.name}")
            return list(self.task_sets[task_set])
        selected = list(tasks if tasks is not None else ["054"])
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Tasks must be a nonempty list without duplicates")
        unknown = set(selected) - set(self.tasks)
        if unknown:
            raise ValueError(f"Tasks outside {self.name}: {sorted(unknown)}")
        return selected

    def plan(self, *, task_set=None, tasks=None, skip_unsupported=False, verify_protocols=False):
        """Describe the same task selection for submission and the Agent runner."""
        if verify_protocols and not tasks:
            raise ValueError("Protocol qualification requires explicit --tasks")
        if verify_protocols and skip_unsupported:
            raise ValueError("Choose protocol qualification or skipping unsupported tasks")
        requested = self.select(task_set=task_set, tasks=tasks)
        excluded = [
            {"task_id": task, "reason": self.exclusions[task]}
            for task in requested if task in self.exclusions
        ]
        return {
            "benchmark": "osworld_2",
            "release": self.name,
            "env_id": self.env_id,
            "release_manifest_sha256": self.manifest_sha256,
            "task_revision": self.task_revision,
            "task_set": task_set or "explicit",
            "requested_task_ids": requested,
            "requested_count": len(requested),
            "unsupported": excluded,
            "scheduled_task_ids": [
                task for task in requested if verify_protocols or task not in self.exclusions
            ],
            "skipped": excluded if skip_unsupported else [],
            "verification_mode": verify_protocols,
        }


def release_names() -> tuple[str, ...]:
    """List the releases available through the shared Worker adapter."""
    return tuple(json.loads((_ROOT / "index.json").read_text()))


@cache
def load_release(name: str = DEFAULT_RELEASE) -> Release:
    """Read version facts without importing a runtime or accessing gated assets."""
    index = json.loads((_ROOT / "index.json").read_text())
    if name not in index:
        raise ValueError(f"Unknown OSWorld release {name!r}; available: {', '.join(index)}")
    binding = index[name]
    path = _ROOT / name
    manifest_bytes = (path / "release.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["release"] != name:
        raise ValueError("Release directory and manifest identity differ")
    hashes = path / "task_hashes.json"
    digest = "sha256:" + hashlib.sha256(hashes.read_bytes()).hexdigest()
    if digest != manifest["task_hash_manifest"]["sha256"]:
        raise ValueError("Packaged task hashes differ from the release manifest")
    tasks = tuple(json.loads((path / "tasks.json").read_text())["tasks"])
    if len(set(tasks)) != len(tasks) or len(tasks) != manifest["task_hash_manifest"]["task_count"]:
        raise ValueError(f"Task catalog differs from {name}'s manifest")
    task_sets = {"full": tasks}
    for file in sorted((path / "task_sets").glob("*.json")):
        selected = tuple(json.loads(file.read_text())["tasks"])
        if file.stem == "full" or len(set(selected)) != len(selected) or set(selected) - set(tasks):
            raise ValueError(f"Invalid task set: {file}")
        task_sets[file.stem] = selected
    return Release(
        name=name,
        env_id=binding["env_id"],
        path=path,
        manifest=manifest,
        tasks=tasks,
        task_sets=task_sets,
        exclusions=json.loads((path / "protocol_exclusions.json").read_text()),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
