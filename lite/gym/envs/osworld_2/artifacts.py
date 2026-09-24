"""OSWorld episode state, diagnostic events and experiment summaries."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import stat
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .catalog import DEFAULT_RELEASE, load_release

DEFAULT_ROOT = Path(".sandbox/results")


def timestamp():
    return datetime.now(UTC).isoformat()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Experiment and run names must be simple identifiers, not paths")
    return value


def task_identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{3}", value):
        raise ValueError("OSWorld task IDs must contain three digits")
    return value


def redact(value, secrets=()):
    """Strip credential fields and known secret values before writing local results."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            sensitive = normalized in {
                "ak",
                "sk",
                "headers",
                "authorization",
                "credentials",
                "cookie",
            } or normalized.endswith(
                ("apikey", "accesskey", "secretkey", "password", "token", "secret")
            )
            result[key] = "[redacted]" if sensitive else redact(item, secrets)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
    return value


def atomic_text(path, content, *, sync=False):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name + "-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            if sync:
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class Experiment:
    def __init__(
        self, name=None, root=DEFAULT_ROOT, config=None, *, release=DEFAULT_RELEASE, existing=False
    ):
        self.release = load_release(release)
        self.name = identifier(
            name or datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        )
        self.path = Path(root).expanduser().resolve() / self.name
        self._lock_path = self.path / ".experiment.lock"
        manifest = self.path / "experiment.json"
        if existing and not manifest.is_file():
            raise FileNotFoundError(f"Experiment not found: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        with self.lock():
            if manifest.exists():
                saved = json.loads(manifest.read_text())
                if (
                    saved.get("experiment_id") != self.name
                    or saved.get("release") != self.release.name
                ):
                    raise ValueError("Existing experiment metadata does not match")
                if saved.get("release_manifest_sha256") != self.release.manifest_sha256:
                    raise ValueError("Existing experiment release manifest differs")
                if config is not None and saved.get("config") != redact(config):
                    raise ValueError("Experiment configuration differs; use a new experiment name")
            else:
                if set(p.name for p in self.path.iterdir()) - {self._lock_path.name}:
                    raise FileExistsError(
                        "Refusing to adopt a nonempty directory without experiment.json"
                    )
                write_json(
                    manifest,
                    {
                        "schema_version": 1,
                        "experiment_id": self.name,
                        "created_at": timestamp(),
                        "release": self.release.name,
                        "release_manifest_sha256": self.release.manifest_sha256,
                        "config": redact(config or {}),
                    },
                )

    @contextmanager
    def lock(self, *, timeout=60):
        """Serialize shared writers with atomic mkdir; never take over an existing lock."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._lock_path.mkdir()
            except FileExistsError:
                try:
                    mode = self._lock_path.lstat().st_mode
                except FileNotFoundError:
                    pass  # The owner released between mkdir and lstat.
                else:
                    if not stat.S_ISDIR(mode):
                        raise FileExistsError(
                            f"Experiment lock must be a directory: {self._lock_path}. "
                            "File-based locks are unsupported; verify no writer remains "
                            "before removing it, or use a new experiment name."
                        ) from None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for experiment lock: {self._lock_path}. "
                        "Verify whether a writer is active before removing a stale lock, "
                        "or use a new experiment name."
                    ) from None
                time.sleep(min(0.05, remaining))
            else:
                break
        try:
            yield
        finally:
            # Only a successful mkdir reaches here; contenders never remove it.
            self._lock_path.rmdir()

    def new_run(self, task_id, *, run_id=None, secrets=(), existing=False):
        task_id = task_identifier(task_id)
        self.release.select(tasks=[task_id])
        run_id = (
            identifier(run_id) if run_id is not None else "run-" + task_id + "-" + uuid.uuid4().hex
        )
        with self.lock():
            path = self.path / ("task_" + task_id) / run_id
            if existing:
                saved = json.loads((path / "run.json").read_text())
                if (saved["run_id"], saved["task_id"], saved["release"]) != (
                    run_id, task_id, self.release.name
                ):
                    raise ValueError("Existing task artifacts do not match this phase")
                return RunArtifacts(self, path, secrets)
            path.mkdir(parents=True, exist_ok=False)
            write_json(
                path / "run.json",
                {
                    "experiment_id": self.name,
                    "task_id": task_id,
                    "run_id": run_id,
                    "release": self.release.name,
                    "release_manifest_sha256": self.release.manifest_sha256,
                    "status": "running",
                    "started_at": timestamp(),
                },
            )
            self._summarize()
        return RunArtifacts(self, path, secrets)

    def _summarize(self):
        runs = [
            json.loads(path.read_text())
            for path in sorted(self.path.glob("task_[0-9][0-9][0-9]/*/run.json"))
        ]
        counts = {
            status: sum(run["status"] == status for run in runs)
            for status in ("running", "completed", "failed", "interrupted")
        }
        summary = {
            "experiment_id": self.name,
            "updated_at": timestamp(),
            "task_count": len({run["task_id"] for run in runs}),
            "run_count": len(runs),
            "counts": counts,
            "runs": runs,
        }
        selection = json.loads((self.path / "experiment.json").read_text())["config"].get(
            "selection"
        )
        if selection is not None:
            completed = {run["task_id"] for run in runs if run["status"] == "completed"}
            executed = {run["task_id"] for run in runs}
            summary["coverage"] = {
                "task_set": selection["task_set"],
                "requested_count": selection["requested_count"],
                "scheduled_count": len(selection["scheduled_task_ids"]),
                "executed_count": len(executed),
                "completed_count": len(completed),
                "skipped": selection["skipped"],
                "pending_task_ids": [
                    task for task in selection["scheduled_task_ids"] if task not in executed
                ],
                "complete_requested_set": len(completed) == selection["requested_count"],
            }
        write_json(self.path / "summary.json", summary)
        output = io.StringIO()
        columns = ["task_id", "run_id", "status", "score", "started_at", "finished_at", "error"]
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
        atomic_text(self.path / "summary.csv", output.getvalue())
        return summary

    def summarize(self):
        with self.lock():
            return self._summarize()


class RunArtifacts:
    def __init__(self, experiment, path, secrets=()):
        self.experiment, self.path = experiment, path
        self.run_id = path.name
        self.secrets = list(secrets)

    def save_json(self, name, value):
        write_json(self.path / identifier(name), redact(value, self.secrets))

    def event(self, method, **values):
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    redact({"time": timestamp(), "method": method, **values}, self.secrets),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    def observation(self, value):
        """Validate the PNG and journal metadata; the Agent logger owns image files."""
        screenshot = value["screenshot"]
        if not isinstance(screenshot, bytes) or not screenshot.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Expected PNG screenshot bytes")
        observation = {
            key: item
            for key, item in value.items()
            if key not in ("screenshot", "screenshot_encoding")
        }
        self.event("observation", observation=observation)

    def update(self, **values):
        """Publish live run state; aggregate snapshots are refreshed separately."""
        with self.experiment.lock():
            path = self.path / "run.json"
            value = json.loads(path.read_text())
            value.update(redact(values, self.secrets))
            write_json(path, value)
        return value

    def finish(self, status, **values):
        if status not in ("completed", "failed", "interrupted"):
            raise ValueError("Invalid final run status")
        record = self.update(status=status, finished_at=timestamp(), **values)
        self.experiment.summarize()
        return record


def main():
    parser = argparse.ArgumentParser(
        description="Refresh experiment summary JSON/CSV from per-run records"
    )
    parser.add_argument("experiment", type=identifier)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    manifest = args.results_root.expanduser() / args.experiment / "experiment.json"
    release = json.loads(manifest.read_text())["release"]
    experiment = Experiment(args.experiment, args.results_root, release=release, existing=True)
    summary = experiment.summarize()
    print(
        json.dumps(
            {
                "directory": str(experiment.path),
                "run_count": summary["run_count"],
                "counts": summary["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
