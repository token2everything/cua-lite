#!/usr/bin/env python3
"""Generate cua-lite's WAA task catalog from a pinned upstream checkout.

Run (maintainers): uv run python lite/gym/envs/waa/scripts/utils/sync_tasks.py \\
       --waa-root /path/to/WindowsAgentArena   # regenerates data/tasks.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

WAA_COMMIT = "6d39ed88c545a0d40a7a02e39b928e278df7332b"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--waa-root",
        type=Path,
        # Sibling-checkout convenience default (repo-parent / WindowsAgentArena),
        # NOT a this-repo-root derivation; override on other layouts.
        default=Path(__file__).resolve().parents[7] / "WindowsAgentArena",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "tasks.json",
    )
    args = parser.parse_args()

    try:
        head = subprocess.check_output(
            ["git", "-C", str(args.waa_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Unable to verify WindowsAgentArena checkout at {args.waa_root}"
        ) from exc
    if head != WAA_COMMIT:
        raise RuntimeError(f"WindowsAgentArena checkout must be at {WAA_COMMIT}, got {head}")

    data_root = (
        args.waa_root / "src" / "win-arena-container" / "client" / "evaluation_examples_windows"
    )
    index = json.loads((data_root / "test_all.json").read_text(encoding="utf-8"))
    tasks: list[dict] = []
    for variant_dir, split, variant, prefix in (
        ("examples", "eval", "standard", ""),
        ("examples_noctxt", "eval_noctxt", "no_context", "noctxt_"),
    ):
        for domain, task_ids in index.items():
            for upstream_id in task_ids:
                path = data_root / variant_dir / domain / f"{upstream_id}.json"
                config = json.loads(path.read_text(encoding="utf-8"))
                tasks.append(
                    {
                        "split": split,
                        "task_id": f"{prefix}{upstream_id}",
                        "domain": domain,
                        "variant": variant,
                        "config": config,
                    }
                )

    task_ids = [task["task_id"] for task in tasks]
    if len(tasks) != 308 or len(set(task_ids)) != 308:
        raise RuntimeError(
            f"Expected 308 unique WAA task variants, got {len(tasks)} rows "
            f"and {len(set(task_ids))} unique ids"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(tasks)} tasks to {args.output}")


if __name__ == "__main__":
    main()
