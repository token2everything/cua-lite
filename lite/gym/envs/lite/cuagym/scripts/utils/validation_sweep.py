#!/usr/bin/env python3
"""Validate pinned CUA-Gym tasks and maintain ``validation_excludes.json``.

Offline mode verifies every committed finding against the current asset and
summarizes importer-detected defects. Live mode resets tasks in real containers,
takes no agent action, terminates, and records setup/reward failures or a
non-zero baseline.

Examples:
    # Offline audit only.
    uv run python .../validation_sweep.py

    # Review-sized live sample. This writes a report, never the lock file.
    uv run python .../validation_sweep.py --live --limit 20 --concurrency 4

    # Full live-eligible sweep, intentionally explicit. Do not run before review.
    uv run python .../validation_sweep.py --live --all --write --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "pyproject.toml").is_file():
        sys.path.insert(0, str(_candidate))
        break

from lite.core.tools import make_tool_call  # noqa: E402
from lite.utils.path import project_root  # noqa: E402

ROOT = project_root()
ENV_DIR = ROOT / "lite/gym/envs/lite/cuagym"
WEB_CATALOG = ENV_DIR / ".cache/web/lite.cuagym_tasks/train.jsonl"
DESKTOP_CATALOG = ENV_DIR / ".cache/desktop/lite.cuagym_desktop_tasks/train.jsonl"


def _rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for catalog in (WEB_CATALOG, DESKTOP_CATALOG):
        if not catalog.is_file():
            raise RuntimeError(
                f"{catalog} is missing; run lite.cuagym/scripts/install.sh provision"
            )
        for line in catalog.read_text().splitlines():
            row = json.loads(line)
            task_id = row["task_id"]
            if task_id in rows:
                raise RuntimeError(f"duplicate task id across catalogs: {task_id}")
            rows[task_id] = row
    return rows


def _digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_lock(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from lite.gym.envs.lite.cuagym.src.utils import dataset

    lock = json.loads(dataset.VALIDATION_EXCLUDES_PATH.read_text())
    findings = {key: value for key, value in lock.items() if key != "_meta"}
    for task_id, finding in findings.items():
        row = rows.get(task_id)
        if row is None:
            raise RuntimeError(f"validation lock references absent task {task_id}")
        if finding.get("layer") != "duplicate_bundle_audit":
            if finding.get("layer") == "mock_runtime_audit":
                app = finding.get("app")
                apps = (row["metadata"].get("others") or {}).get("apps", [])
                if app not in apps:
                    raise RuntimeError(
                        f"{task_id}: validation app {app!r} not in {apps!r}"
                    )
            continue
        peer_id = finding.get("paired_task_id")
        peer = rows.get(peer_id)
        if peer is None:
            raise RuntimeError(f"{task_id}: paired task {peer_id!r} is absent")
        for field in ("setup", "reward"):
            if _digest(row["metadata"][field]) != _digest(peer["metadata"][field]):
                raise RuntimeError(
                    f"{task_id}: {field} no longer matches paired task {peer_id}"
                )
        if row["instruction"] == peer["instruction"]:
            raise RuntimeError(f"{task_id}: paired instructions are unexpectedly equal")
    return lock


def _task_failure(
    task_id: str,
    phase: str,
    exc: Exception,
) -> dict[str, Any]:
    from lite.gym.errors import is_retryable

    retryable = is_retryable(exc)
    result = {
        "task_id": task_id,
        "outcome": f"{'transient_' if retryable else ''}{phase}_error",
        "detail": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()[-2000:],
    }
    if not retryable:
        result["reason"] = f"broken_{phase}:live_validation_error"
    return result


def _bypass_exclusion_guard(env) -> None:
    from lite.gym.envs.lite.cuagym.src.browser import scripts as browser
    from lite.gym.envs.lite.cuagym.src.desktop import scripts as desktop

    inner = env.unwrapped
    is_browser_task = bool((inner._task.metadata.get("others") or {}).get("apps"))
    inner._setup_fn = browser.setup_fn if is_browser_task else desktop.setup_fn


async def _live_once(task_id: str, *, bypass_exclude: bool) -> dict[str, Any]:
    import lite.gym as gym

    env = gym.make(f"lite.cuagym@{task_id}", max_steps=1)
    if bypass_exclude:
        _bypass_exclusion_guard(env)
    try:
        try:
            await env.reset()
        except Exception as exc:
            return _task_failure(task_id, "setup", exc)
        try:
            result = await env.step(
                [make_tool_call("terminate", {})]
            )
        except Exception as exc:
            return _task_failure(task_id, "reward", exc)
        reward = float(result.reward or 0.0)
        if reward > 0.0:
            return {
                "task_id": task_id,
                "outcome": "nonzero_baseline",
                "reason": "broken_reward:nonzero_baseline",
                "reward": reward,
            }
        return {"task_id": task_id, "outcome": "clean_zero", "reward": reward}
    finally:
        await env.close()


async def _live_one(
    task_id: str,
    semaphore: asyncio.Semaphore,
    attempts: int,
    bypass_exclude: bool,
) -> dict[str, Any]:
    async with semaphore:
        runs = []
        valid_runs = []
        for _ in range(attempts * 3):
            try:
                run = await _live_once(task_id, bypass_exclude=bypass_exclude)
            except Exception as exc:
                run = {
                    "task_id": task_id,
                    "outcome": "harness_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-2000:],
                }
            runs.append(run)
            if not run["outcome"].startswith("transient_"):
                if run["outcome"] == "clean_zero":
                    result = dict(run)
                    result["attempts"] = runs
                    return result
                valid_runs.append(run)
                if len(valid_runs) == attempts:
                    break
    if len(valid_runs) < attempts:
        return {
            "task_id": task_id,
            "outcome": "transient_exhausted",
            "attempts": runs,
        }
    outcomes = {run["outcome"] for run in valid_runs}
    if len(outcomes) == 1:
        result = dict(valid_runs[-1])
        result["attempts"] = runs
        return result
    return {
        "task_id": task_id,
        "outcome": "unstable",
        "attempts": runs,
    }


async def _live(
    ids: list[str],
    concurrency: int,
    attempts: int,
    bypass_exclude: bool,
    on_result=None,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            _live_one(task_id, semaphore, attempts, bypass_exclude)
        )
        for task_id in ids
    ]
    results = []
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        if on_result is not None:
            on_result(result)
    order = {task_id: index for index, task_id in enumerate(ids)}
    return sorted(results, key=lambda result: order[result["task_id"]])


def _checkpoint_meta(ids: list[str], attempts: int) -> dict[str, Any]:
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return {"task_ids_sha256": digest, "tasks": len(ids), "attempts": attempts}


def _load_checkpoint(
    path: Path,
    ids: list[str],
    attempts: int,
) -> list[dict[str, Any]]:
    lines = path.read_text().splitlines()
    if not lines or json.loads(lines[0]).get("_meta") != _checkpoint_meta(ids, attempts):
        raise RuntimeError(f"{path} does not match this validation run")
    results = [json.loads(line) for line in lines[1:] if line.strip()]
    if len({result["task_id"] for result in results}) != len(results):
        raise RuntimeError(f"{path} contains duplicate task results")
    return results


def _write_live(lock: dict[str, Any], results: list[dict[str, Any]]) -> None:
    from lite.gym.envs.lite.cuagym.src.utils import dataset

    retained = {
        key: value
        for key, value in lock.items()
        if key == "_meta" or value.get("layer") != "live_noop"
    }
    for result in results:
        if reason := result.get("reason"):
            retained[result["task_id"]] = {
                "reason": reason,
                "layer": "live_noop",
                "outcome": result["outcome"],
                **({"reward": result["reward"]} if "reward" in result else {}),
                **({"detail": result["detail"]} if "detail" in result else {}),
            }
    meta = retained["_meta"]
    meta["total"] = len(retained) - 1
    meta["layers"]["live_noop"] = {
        "runs": "reset + terminate in a real lite.cuagym container with no agent action",
        "status": "complete",
        "tasks": len(results),
        "findings": sum("reason" in result for result in results),
    }
    dataset.VALIDATION_EXCLUDES_PATH.write_text(
        json.dumps(retained, indent=2, sort_keys=True) + "\n"
    )


def _live_ids(
    rows: dict[str, dict[str, Any]],
    lock: dict[str, Any],
) -> list[str]:
    """Select clean rows plus prior live findings, excluding static defects."""
    prior_live = {
        task_id
        for task_id, finding in lock.items()
        if task_id != "_meta" and finding.get("layer") == "live_noop"
    }
    return [
        task_id
        for task_id, row in sorted(rows.items())
        if not (row["metadata"].get("others") or {}).get("exclude_reason")
        or task_id in prior_live
    ]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--all", action="store_true", help="validate every live task")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/lite-cuagym-validation-report.json"),
    )
    args = parser.parse_args()
    if args.write and not (args.live and args.all):
        parser.error("--write requires the explicit full-sweep flags --live --all")
    if args.all and args.task_id:
        parser.error("--all and --task-id are mutually exclusive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.attempts < 2:
        parser.error("--attempts must be at least 2 to avoid one-shot exclusions")
    if args.resume and not args.live:
        parser.error("--resume requires --live")
    return args


def main() -> None:
    args = _args()
    rows = _rows()
    lock = audit_lock(rows)
    counts = Counter(
        (row["metadata"].get("others") or {}).get("exclude_reason")
        for row in rows.values()
    )
    report: dict[str, Any] = {
        "tasks": len(rows),
        "eligible_rows": counts[None],
        "catalog_excludes": {
            reason: count
            for reason, count in sorted(counts.items(), key=str)
            if reason is not None
        },
        "locked_findings": len(lock) - 1,
    }
    if args.live:
        live_ids = _live_ids(rows, lock)
        if args.task_id:
            requested = set(args.task_id)
            missing = requested - rows.keys()
            if missing:
                raise RuntimeError(f"unknown task ids: {sorted(missing)}")
            live_ids = sorted(requested)
        elif not args.all:
            live_ids = live_ids[: args.limit]
        checkpoint = Path(f"{args.report}.jsonl")
        prior = _load_checkpoint(checkpoint, live_ids, args.attempts) if args.resume else []
        if not args.resume:
            checkpoint.write_text(
                json.dumps({"_meta": _checkpoint_meta(live_ids, args.attempts)}) + "\n"
            )
        completed = {result["task_id"] for result in prior}
        pending = [task_id for task_id in live_ids if task_id not in completed]
        stream = checkpoint.open("a")

        def checkpoint_result(result: dict[str, Any]) -> None:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()

        try:
            new_results = asyncio.run(
                _live(
                    pending,
                    args.concurrency,
                    args.attempts,
                    bypass_exclude=bool(args.task_id),
                    on_result=checkpoint_result,
                )
            )
        finally:
            stream.close()
        by_id = {result["task_id"]: result for result in prior + new_results}
        results = [by_id[task_id] for task_id in live_ids]
        report["live"] = {
            "tasks": len(results),
            "outcomes": dict(Counter(result["outcome"] for result in results)),
            "results": results,
        }
        if args.write:
            _write_live(lock, results)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "live"}, indent=2))
    if "live" in report:
        print(json.dumps(report["live"]["outcomes"], indent=2, sort_keys=True))
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
