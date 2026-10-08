#!/usr/bin/env python3
"""Generate a config-bound compatible-worker plan using historical durations.

Usage: uv run python devs/exps/eval/browsergym.webarena_verified/campaign/generate_plan.py
    --root /path/to/results/<run> --config /path/to/results/<run>/EXPERIMENT_CONFIG.json
    --historical-timing /path/to/V17/qa-analysis-v17-20261007/case-timing.jsonl

Historical timings determine scheduling only; neither scores nor private task
answers are supplied to the agent. Full execution dynamically assigns pending
cases to idle workers whose prepared site capabilities cover the case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "dataset/webarena-verified.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--historical-timing", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
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


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    if args.config.resolve() != root / "EXPERIMENT_CONFIG.json":
        raise ValueError("config must be the campaign's EXPERIMENT_CONFIG.json")
    if (root / "PLAN_MANIFEST.sha256").exists():
        raise ValueError("refusing to replace an already-published campaign plan")
    config = json.loads(args.config.read_text())
    catalog = {
        int(row["task_id"]): set(row["sites"]) for row in json.loads(args.dataset.read_text())
    }
    if set(catalog) != set(range(config["task_count"])) or config["task_count"] != 812:
        raise ValueError("full dataset must contain exactly task IDs 0..811")
    durations: dict[int, float] = {}
    with args.historical_timing.open() as source:
        for line in source:
            row = json.loads(line)
            task = int(row["task_id"])
            if task in durations:
                raise ValueError(f"duplicate historical timing for task {task}")
            index = Path(row["origin_result_root"]) / "index" / f"task-{task:04d}"
            baseline = json.loads((index / "baseline_receipt.json").read_text())
            completed = json.loads((index / "completed.json").read_text())
            duration = (
                datetime.fromisoformat(completed["completed_at"].replace("Z", "+00:00"))
                - datetime.fromisoformat(baseline["created_at"].replace("Z", "+00:00"))
            ).total_seconds()
            if not 0 < duration < float("inf"):
                raise ValueError(f"invalid historical timing for task {task}")
            durations[task] = duration
    if set(durations) != set(catalog):
        raise ValueError("historical timings must cover all 812 tasks exactly")
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
            "historical LPT with shortest exact-profile qualification; "
            "dynamic compatible dispatch"
        ),
        "task_count": len(catalog),
        "historical_timing_sha256": hashlib.sha256(args.historical_timing.read_bytes()).hexdigest(),
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
    print(json.dumps({"run_id": config["run_id"], "workers": len(workers), "tasks": len(catalog)}))


if __name__ == "__main__":
    main()
