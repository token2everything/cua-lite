#!/usr/bin/env python3
"""Generate serial worker shards for a full or selected WebArena run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parents[1] / "dataset" / "webarena-verified.json"
FULL_WORKER_CAPABILITIES = {
    **{f"{worker:02d}": {"gitlab", "reddit"} for worker in range(1, 3)},
    "03": {"gitlab", "wikipedia"},
    "04": {"map", "shopping_admin"},
    **{f"{worker:02d}": {"map", "wikipedia"} for worker in range(5, 7)},
    "07": {"reddit", "shopping"},
    **{f"{worker:02d}": {"gitlab"} for worker in range(8, 17)},
    **{f"{worker:02d}": {"map"} for worker in range(17, 25)},
    **{f"{worker:02d}": {"reddit"} for worker in range(25, 31)},
    **{f"{worker:02d}": {"shopping"} for worker in range(31, 41)},
    **{f"{worker:02d}": {"shopping_admin"} for worker in range(41, 51)},
}
# Duplicate the proven 50-worker capability layout for workers 51-100.
FULL_WORKER_CAPABILITIES.update(
    {
        f"{worker:02d}": set(FULL_WORKER_CAPABILITIES[f"{worker - 50:02d}"])
        for worker in range(51, 101)
    }
)
# Additional workers use common single-site profiles. Rare two-site profiles
# have only 2-18 exact-match tasks and must not be duplicated a third time.
EXTRA_SINGLE_SITE_PROFILES = (
    ["shopping"] * 6
    + ["shopping_admin"] * 6
    + ["gitlab"] * 6
    + ["map"] * 5
    + ["reddit"] * 5
)
FULL_WORKER_CAPABILITIES.update(
    {
        f"{worker:02d}": {site}
        for worker, site in enumerate(EXTRA_SINGLE_SITE_PROFILES, start=101)
    }
)


def parse_task_spec(value: str) -> list[int]:
    """Parse comma-separated task IDs and inclusive ranges."""
    task_ids: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            raise ValueError("task list contains an empty item")
        if "-" in token:
            start_text, separator, end_text = token.partition("-")
            if not separator or not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"invalid task range: {token}")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending task range: {token}")
            task_ids.extend(range(start, end + 1))
        elif token.isdigit():
            task_ids.append(int(token))
        else:
            raise ValueError(f"invalid task ID: {token}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task list contains duplicate IDs")
    return task_ids


def task_weight(sites: set[str]) -> float:
    """Estimate service cost for balanced deterministic sharding."""
    return 1.0 + (0.3 if "map" in sites else 0.0) + 0.1 * (len(sites) - 1)


def load_catalog() -> dict[int, set[str]]:
    rows = json.loads(DATASET.read_text())
    catalog = {int(row["task_id"]): set(row["sites"]) for row in rows}
    if len(catalog) != len(rows):
        raise RuntimeError("dataset contains duplicate task IDs")
    return catalog


def assign_full(catalog: dict[int, set[str]], worker_ids: set[str]) -> dict[str, list[int]]:
    capabilities_by_worker = {
        worker: capabilities
        for worker, capabilities in FULL_WORKER_CAPABILITIES.items()
        if worker in worker_ids
    }
    assignments = {worker: [] for worker in capabilities_by_worker}
    loads = {worker: 0.0 for worker in capabilities_by_worker}
    unassigned = set(catalog)

    # Seed every worker with a task whose exact site set matches its profile. This
    # makes the startup gate cover all profiles and keeps declared capabilities
    # identical to the union of sites actually assigned to that worker.
    for worker, capabilities in capabilities_by_worker.items():
        candidates = sorted(
            task_id for task_id in unassigned if catalog[task_id] == capabilities
        )
        if not candidates:
            raise RuntimeError(
                f"no exact-profile startup task for worker {worker}: {capabilities}"
            )
        task_id = candidates[0]
        assignments[worker].append(task_id)
        loads[worker] += task_weight(catalog[task_id])
        unassigned.remove(task_id)

    remaining = sorted(
        unassigned,
        key=lambda task_id: (-task_weight(catalog[task_id]), task_id),
    )
    for task_id in remaining:
        sites = catalog[task_id]
        eligible = [
            worker
            for worker, capabilities in capabilities_by_worker.items()
            if sites <= capabilities
        ]
        if not eligible:
            raise RuntimeError(f"no full-run worker can serve task {task_id}: {sites}")
        worker = min(
            eligible,
            key=lambda candidate: (
                loads[candidate],
                len(assignments[candidate]),
                candidate,
            ),
        )
        assignments[worker].append(task_id)
        loads[worker] += task_weight(sites)
    return assignments


def assign_subset(task_ids: list[int], worker_count: int) -> dict[str, list[int]]:
    assignments = {f"{worker:02d}": [] for worker in range(1, worker_count + 1)}
    for index, task_id in enumerate(task_ids):
        worker = f"{index % worker_count + 1:02d}"
        assignments[worker].append(task_id)
    return assignments


def write_plan(
    catalog: dict[int, set[str]],
    assignments: dict[str, list[int]],
    selection: str,
    strategy: str,
) -> None:
    workers: dict[str, dict[str, object]] = {}
    startup_tasks: dict[str, int] = {}
    for worker, task_ids in assignments.items():
        if not task_ids:
            raise RuntimeError(f"worker {worker} has no tasks")
        capabilities = sorted(set().union(*(catalog[task_id] for task_id in task_ids)))
        startup_task = task_ids[0]
        startup_tasks[worker] = startup_task
        workers[worker] = {
            "capabilities": capabilities,
            "startup_task": startup_task,
            "tasks": len(task_ids),
            "weighted_load": round(sum(task_weight(catalog[task_id]) for task_id in task_ids), 1),
        }
        worker_dir = ROOT / "workers" / worker
        worker_dir.mkdir(parents=True, exist_ok=True)
        task_lines = [f"{task_id}\t{','.join(sorted(catalog[task_id]))}" for task_id in task_ids]
        (worker_dir / "tasks.tsv").write_text("\n".join(task_lines) + "\n")
        (worker_dir / "capabilities.txt").write_text(",".join(capabilities) + "\n")

    plan = {
        "selection": selection,
        "startup_tasks": startup_tasks,
        "strategy": strategy,
        "task_count": sum(len(task_ids) for task_ids in assignments.values()),
        "workers": workers,
    }
    (ROOT / "worker-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    plan_paths = [ROOT / "worker-plan.json"]
    for worker in assignments:
        plan_paths.extend(
            [
                ROOT / "workers" / worker / "tasks.tsv",
                ROOT / "workers" / worker / "capabilities.txt",
            ]
        )
    manifest_lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}"
        for path in plan_paths
    ]
    (ROOT / "PLAN_MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate worker-plan.json and serial task shards."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--full", action="store_true", help="all 812 tasks")
    selection.add_argument(
        "--tasks", help="comma-separated IDs/ranges, for example 0,16,97,400-405"
    )
    parser.add_argument(
        "--workers", type=int, help="worker count for a subset or full campaign, max 128"
    )
    parser.add_argument(
        "--existing-cluster", type=Path,
        help="reassign the full dataset to the existing cluster's worker IDs",
    )
    parser.add_argument(
        "--disk-aware-50",
        action="store_true",
        help="full 812-task plan with 50 workers and four map database copies",
    )
    args = parser.parse_args()

    catalog = load_catalog()
    if args.full:
        if set(catalog) != set(range(812)):
            parser.error("full dataset must contain exactly task IDs 0-811")
        if args.disk_aware_50:
            if args.workers is not None or args.existing_cluster is not None:
                parser.error("--disk-aware-50 cannot be combined with worker overrides")
            # Keep the rare map+admin and map+Wikipedia profiles, then spread
            # 128 map tasks over four database copies instead of 11-25.
            profile_ids = {
                f"{worker:02d}" for worker in range(1, 51) if worker not in range(18, 25)
            }
            profile_ids.update(f"{worker:02d}" for worker in range(75, 82))
            by_profile = assign_full(catalog, profile_ids)
            assignments = {
                f"{index:02d}": by_profile[profile]
                for index, profile in enumerate(sorted(profile_ids, key=int), start=1)
            }
            strategy = "disk-aware capability balance; four map workers"
        elif args.existing_cluster is not None:
            if args.workers is not None:
                parser.error("--existing-cluster and --workers are mutually exclusive")
            cluster = json.loads(args.existing_cluster.read_text())
            worker_ids = {f"{int(item['worker']):02d}" for item in cluster["workers"]}
            if not worker_ids or len(worker_ids) != len(cluster["workers"]):
                parser.error("existing cluster must contain unique worker IDs")
            if not worker_ids <= FULL_WORKER_CAPABILITIES.keys():
                parser.error("existing cluster contains an unknown worker profile")
            assignments = assign_full(catalog, worker_ids)
            strategy = "capability-constrained greedy weighted balance"
        else:
            if args.workers is not None and args.workers not in {50, 100, 128}:
                parser.error("--workers for --full must be 50, 100, or 128")
            worker_ids = {f"{worker:02d}" for worker in range(1, (args.workers or 128) + 1)}
            assignments = assign_full(catalog, worker_ids)
            strategy = "capability-constrained greedy weighted balance"
    else:
        if args.existing_cluster is not None or args.disk_aware_50:
            parser.error("--existing-cluster and --disk-aware-50 require --full")
        try:
            task_ids = parse_task_spec(args.tasks)
        except ValueError as error:
            parser.error(str(error))
        unknown = sorted(set(task_ids) - set(catalog))
        if unknown:
            parser.error(f"unknown task IDs: {unknown}")
        worker_count = args.workers or min(128, len(task_ids))
        if not 1 <= worker_count <= min(128, len(task_ids)):
            parser.error("--workers must be between 1 and min(128, selected task count)")
        assignments = assign_subset(task_ids, worker_count)
        strategy = "ordered round-robin subset"

    write_plan(catalog, assignments, "full" if args.full else "subset", strategy)
    print(
        f"wrote {sum(map(len, assignments.values()))} tasks "
        f"across {len(assignments)} workers to {ROOT}"
    )


if __name__ == "__main__":
    main()
