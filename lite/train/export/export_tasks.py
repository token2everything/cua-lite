"""CLI: create a parquet task list from the gym task registry.

The output is the rollout input shared by every consumer that turns tasks into
trajectories — RL training (GRPO / REINFORCE / DAgger) and plain rollout /
evaluation (``lite.infer.rollout``). It carries the task catalog (env_key, split,
optional per-row env_kwargs), NOT trajectories — it is GRPO-agnostic.

Usage:
    python -m lite.train.export.export_tasks --env-id lite.demo -o train.parquet
    python -m lite.train.export.export_tasks --env-id lite.osworld \
        --split train -o data/train.parquet
    python -m lite.train.export.export_tasks --env-id lite.osworld \
        --split eval -o data/eval.parquet

Parquet schema:
    problem  (str)  — Instruction text placeholder (required by slime _build_messages,
                       not used by generate(); real instruction comes from env.reset())
    metadata (dict) — passed to gym.make() / rollout pipeline. Fields:
        env_key    (str, required)        — "<env_id>@<task_id>" (e.g. "lite.demo@create_file")
        split      (str, required)        — "train" / "eval" / ...
        env_kwargs (dict, optional)       — per-row env_kwargs override; deep-merges over args/yaml
                                            env_kwargs per-leaf when present (prompt_data > args > yaml).
                                            Written when ``--env-kwargs '{...}'`` is passed (same
                                            dict on every row). Common use: pin ``{"seed": 42}`` so an
                                            otherwise-random ``train`` split becomes a deterministic
                                            eval yardstick (the env salts seed with task_id, so one
                                            fixed seed still yields distinct per-task instances).
"""

from __future__ import annotations

import argparse
import json
import random

import lite.gym as gym
from lite.utils.parquet import write_records_to_parquet
from lite.core.utils.filters import parse_filter


def main():
    parser = argparse.ArgumentParser(
        description="Create a parquet task list from the gym task registry "
                    "(rollout input for RL training + eval)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-id", nargs="+", required=True,
                        help="Env IDs (e.g. lite.demo lite.osworld)")
    parser.add_argument("--split", default="train", help="Split name")
    parser.add_argument("--head", type=int, default=None,
                        help="Keep first N tasks (after filtering)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N tasks (after filtering, seeded by --seed)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for --sample (default 42 for reproducibility).")
    parser.add_argument("--filter", type=str, default=None, dest="filter_expr",
                        help='Python lambda on Lite task metadata to filter tasks. '
                             'Example: "lambda m: not m.others.get(\'exclude_reason\')"')
    parser.add_argument("--env-kwargs", type=json.loads, default=None,
                        help='JSON dict written into each row\'s metadata.env_kwargs (overrides '
                             'yaml/args env_kwargs per-leaf, deep, at rollout). Example: \'{"seed": 42}\' '
                             'to make a random train split deterministic for eval. tuples → JSON '
                             'lists (e.g. "resolution": [1280, 720]); finalize_env_kwargs normalizes.')
    parser.add_argument("-o", "--output", required=True, help="Output parquet path")
    args = parser.parse_args()
    random.seed(args.seed)

    filter_fn = parse_filter(args.filter_expr) if args.filter_expr else None

    records = []
    for env_id in args.env_id:
        task_ids = gym.registry.task_ids(env_id, split=args.split)
        for task_id in task_ids:
            meta = gym.registry.task_metadata(env_id, task_id)
            if filter_fn and not filter_fn(meta):
                continue
            records.append({
                "problem": f"Complete the task: {task_id}",
                "metadata": {
                    "env_key": f"{env_id}@{task_id}", "split": args.split,
                    **({"env_kwargs": args.env_kwargs} if args.env_kwargs else {}),
                },
            })

    if args.head is not None:
        records = records[:args.head]
    elif args.sample is not None:
        records = random.sample(records, min(args.sample, len(records)))

    if not records:
        print(f"No tasks found for env_ids={args.env_id}, split={args.split}")
        return

    write_records_to_parquet(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")

if __name__ == "__main__":
    main()
