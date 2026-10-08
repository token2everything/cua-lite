#!/usr/bin/env python3
"""Generate a config-bound compatible-worker plan without private prior results.

Usage: uv run python devs/exps/eval/browsergym.webarena_verified/campaign/generate_plan.py
    --root /path/to/results/<run> --config /path/to/results/<run>/EXPERIMENT_CONFIG.json
    --catalog /path/to/results/<run>/webarena-verified.json
    [--timing /path/to/task-duration-estimates.jsonl]

Optional timing rows contain exactly task_id and estimated_seconds. Otherwise
every task receives the explicit scheduling cost of one second. Estimates
determine scheduling only, never model inputs or scores. Execution dynamically
assigns pending cases to workers whose prepared sites cover the case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--timing", type=Path)
    return parser.parse_args()


def assign_full(
    catalog: dict[int, set[str]],
    profiles: dict[str, dict[str, object]],
    durations: dict[int, float],
) -> dict[str, list[int]]:
    """Seed each capability profile, then greedily balance longest cases first."""
    assignments: dict[str, list[int]] = {worker: [] for worker in profiles}
    loads = dict.fromkeys(profiles, 0.0)
    unassigned = set(catalog)
    for worker, profile in profiles.items():
        capabilities = set(profile["capabilities"])
        candidates = [task for task in unassigned if catalog[task] == capabilities]
        if not candidates:
            raise ValueError(f"no exact-capability startup case for worker {worker}")
        # Qualification should not put the longest model trajectory ahead of a
        # global release barrier. Long remaining cases lead the full queue.
        task = min(candidates, key=lambda current: (durations[current], current))
        assignments[worker].append(task)
        loads[worker] += durations[task]
        unassigned.remove(task)
    for task in sorted(unassigned, key=lambda current: (-durations[current], current)):
        eligible = [
            worker
            for worker, profile in profiles.items()
            if catalog[task] <= set(profile["capabilities"])
        ]
        if not eligible:
            raise ValueError(f"no compatible worker for task {task}: {catalog[task]}")
        worker = min(eligible, key=lambda current: (loads[current], current))
        assignments[worker].append(task)
        loads[worker] += durations[task]
    return assignments


def generate_plan(
    root: Path, config_path: Path, catalog_path: Path, timing_path: Path | None = None
) -> dict[str, object]:
    """Publish one immutable full plan from the official catalog and public estimates."""
    root = root.resolve()
    if config_path.resolve() != root / "EXPERIMENT_CONFIG.json":
        raise ValueError("config must be the campaign's EXPERIMENT_CONFIG.json")
    if (root / "PLAN_MANIFEST.sha256").exists():
        raise ValueError("refusing to replace an already-published campaign plan")
    if (root / "worker-plan.json").exists() or (root / "workers").exists():
        raise ValueError("refusing to overwrite existing worker plan artifacts")
    config = json.loads(config_path.read_text())
    payload = catalog_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != config["benchmark"]["dataset_sha256"]:
        raise ValueError("catalog differs from the configured official dataset")
    records = json.loads(payload)
    if not isinstance(records, list) or [row["task_id"] for row in records] != list(range(812)):
        raise ValueError("full dataset must contain exactly ordered task IDs 0..811")
    catalog = {row["task_id"]: set(row["sites"]) for row in records}
    if set(catalog) != set(range(config["task_count"])) or config["task_count"] != 812:
        raise ValueError("full dataset must contain exactly task IDs 0..811")
    durations = dict.fromkeys(catalog, 1.0)
    timing_source = {"mode": "uniform_cost", "estimated_seconds": 1.0, "sha256": None}
    if timing_path is not None:
        durations = {}
        timing_payload = timing_path.read_bytes()
        for line in timing_payload.splitlines():
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"task_id", "estimated_seconds"}:
                raise ValueError("timing rows must contain exactly task_id and estimated_seconds")
            task, duration = row["task_id"], row["estimated_seconds"]
            if type(task) is not int or task not in catalog or task in durations:
                raise ValueError(f"duplicate or unknown timing task ID: {task}")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(f"invalid estimated_seconds for task {task}")
            durations[task] = float(duration)
        if set(durations) != set(catalog):
            raise ValueError("timing estimates must cover all 812 tasks exactly")
        timing_source = {
            "mode": "provided_estimates",
            "estimated_seconds": None,
            "sha256": hashlib.sha256(timing_payload).hexdigest(),
        }
    profiles: dict[str, dict[str, object]] = {}
    for pool in config["scheduling"]["pool_allocation"]:
        for _ in range(pool["workers"]):
            worker = f"{len(profiles) + 1:02d}"
            profiles[worker] = {
                "pool": pool["pool"],
                "role": pool["role"],
                "capabilities": sorted(pool["capabilities"]),
            }
    if len(profiles) != config["scheduling"]["workers_per_model"]:
        raise ValueError("pool allocation differs from configured worker count")
    assignments = assign_full(catalog, profiles, durations)
    workers = {}
    for worker, tasks in assignments.items():
        profile = profiles[worker]
        workers[worker] = {
            **profile,
            "startup_task": tasks[0],
            "tasks": len(tasks),
            "estimated_case_seconds": sum(durations[task] for task in tasks),
        }
        folder = root / "workers" / worker
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "tasks.tsv").write_text(
            "".join(f"{task}\t{','.join(sorted(catalog[task]))}\n" for task in tasks)
        )
        (folder / "capabilities.txt").write_text(",".join(profile["capabilities"]) + "\n")
    plan = {
        "selection": "full",
        "strategy": (
            "estimated-cost LPT with shortest exact-profile qualification; "
            "dynamic compatible dispatch"
        ),
        "task_count": len(catalog),
        "timing_source": timing_source,
        "startup_tasks": {worker: tasks[0] for worker, tasks in assignments.items()},
        "tasks": {
            str(task): {
                "sites": sorted(catalog[task]),
                "estimated_seconds": durations[task],
            }
            for task in sorted(catalog)
        },
        "workers": workers,
    }
    (root / "worker-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    paths = [root / "EXPERIMENT_CONFIG.json", root / "worker-plan.json"]
    for worker in workers:
        paths.extend(
            (
                root / "workers" / worker / "tasks.tsv",
                root / "workers" / worker / "capabilities.txt",
            )
        )
    (root / "PLAN_MANIFEST.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
            for path in paths
        )
    )
    return plan


def main() -> None:
    args = _parse_args()
    plan = generate_plan(args.root, args.config, args.catalog, args.timing)
    print(json.dumps({"workers": len(plan["workers"]), "tasks": plan["task_count"]}))


if __name__ == "__main__":
    main()
