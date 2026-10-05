#!/usr/bin/env python3
"""Run WebArena-Verified tasks through the Chat Completions gateway agent.

This coordinator keeps model execution on the current host and assigns each
episode to an already-running remote CUA-Lite env-server.  It is intentionally
limited to WebArena-Verified: the benchmark's read/write classification owns
the scheduling policy and the worker cluster is supplied explicitly.

Example:
    uv run python scripts/webarena_verified_gateway.py \
        --cluster-json /path/to/sandbox-cluster.json \
        --results-root /path/to/results/read \
        --phase read --ramp 0:1,10:2,30:4

``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` are read from the process
environment.  The key is written only to a mode-0600 temporary model profile
that is removed when the coordinator exits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lite.gym.envs.browsergym import webarena_verified


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-json", type=Path, required=True)
    parser.add_argument(
        "--worker-layout",
        type=Path,
        help="Optional JSON object mapping worker numbers to their real site names",
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("read", "write"), required=True)
    parser.add_argument("--model-id", default="qwen3.8-27b")
    parser.add_argument("--sandbox-api-base", default=os.environ.get("BASE"))
    parser.add_argument(
        "--ramp",
        default="0:1,10:2,30:4",
        help="Settled-episode thresholds and per-worker limits, e.g. 0:1,10:2,30:4",
    )
    parser.add_argument(
        "--global-ramp",
        default="0:1",
        help="Settled-episode thresholds and cluster-wide limits, e.g. 0:1,10:2",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--model-timeout", type=float, default=180.0)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--task-ids",
        help="Optional comma-separated official numeric task IDs",
    )
    parser.add_argument(
        "--fast-write-shards",
        action="store_true",
        help=(
            "Shard catalog-ordered writers across independent stacks for a fast first pass. "
            "Without this flag, all writers run serially on worker 1."
        ),
    )
    return parser.parse_args(argv)


def _write_json(path: Path, value: Any, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    if mode is not None:
        temporary.chmod(mode)
    temporary.replace(path)


def _parse_ramp(value: str) -> tuple[tuple[int, int], ...]:
    try:
        ramp = tuple(
            (int(threshold), int(limit))
            for threshold, limit in (item.split(":", 1) for item in value.split(","))
        )
    except (TypeError, ValueError) as error:
        raise ValueError("--ramp must look like 0:1,10:2,30:4") from error
    if not ramp or ramp[0][0] != 0 or any(
        threshold < 0 or limit < 1 for threshold, limit in ramp
    ):
        raise ValueError("--ramp must start at threshold 0 and use nonnegative thresholds")
    if any(current[0] >= following[0] for current, following in zip(ramp, ramp[1:])):
        raise ValueError("--ramp thresholds must be strictly increasing")
    if any(current[1] > following[1] for current, following in zip(ramp, ramp[1:])):
        raise ValueError("--ramp limits must not decrease")
    return ramp


def _limit_for(ramp: tuple[tuple[int, int], ...], settled: int) -> int:
    limit = ramp[0][1]
    for threshold, candidate in ramp:
        if settled < threshold:
            break
        limit = candidate
    return limit


def _is_mutating(record: dict[str, Any]) -> bool:
    response = [
        item for item in record["eval"] if item.get("evaluator") == "AgentResponseEvaluator"
    ]
    if len(response) != 1:
        raise ValueError(f"Task {record['task_id']} has an invalid response evaluator")
    return response[0]["expected"]["task_type"] == "mutate"


@dataclass(frozen=True)
class Task:
    numeric_id: int
    browsergym_id: str
    sites: tuple[str, ...]
    worker: int


@dataclass
class ActiveEpisode:
    task: Task
    attempt: int
    process: asyncio.subprocess.Process
    log_file: Any
    job_path: Path
    output_dir: Path
    started_at: float


def _load_tasks(
    args: argparse.Namespace,
    worker_numbers: tuple[int, ...],
    layout: dict[int, frozenset[str]] | None = None,
) -> list[Task]:
    selected = None
    if args.task_ids:
        selected = {int(item) for item in args.task_ids.split(",") if item.strip()}
    records = list(webarena_verified.task_records())
    catalog_ids = list(webarena_verified.task_ids())
    candidates: list[tuple[dict[str, Any], str]] = []
    for record, catalog_id in zip(records, catalog_ids, strict=True):
        mutating = _is_mutating(record)
        if mutating != (args.phase == "write"):
            continue
        if selected is not None and record["task_id"] not in selected:
            continue
        candidates.append((record, catalog_id))

    tasks: list[Task] = []
    assigned = {worker: 0 for worker in worker_numbers}
    replicas = worker_numbers[1:] or worker_numbers
    for index, (record, catalog_id) in enumerate(candidates):
        sites = tuple(record["sites"])
        required_sites = frozenset(sites)
        if args.phase == "write" and not args.fast_write_shards:
            worker = worker_numbers[0]
        elif layout is not None:
            compatible = [
                candidate
                for candidate in worker_numbers
                if required_sites <= layout.get(candidate, frozenset())
            ]
            if not compatible:
                raise ValueError(
                    f"No worker layout entry can serve task {record['task_id']} sites={sites}"
                )
            worker = min(compatible, key=lambda candidate: (assigned[candidate], candidate))
        elif args.phase == "read":
            # Map's 187 GB writable database is provisioned only on the leader.
            worker = worker_numbers[0] if "map" in sites else replicas[index % len(replicas)]
        elif args.fast_write_shards:
            # Each worker queue preserves catalog order.  Cross-worker state is
            # intentionally independent and the run metadata records this mode.
            worker = worker_numbers[index % len(worker_numbers)]
        else:  # pragma: no cover - the serial write case is selected above
            worker = worker_numbers[0]
        assigned[worker] += 1
        task_suffix = catalog_id.removeprefix("webarena_verified.")
        tasks.append(
            Task(
                numeric_id=record["task_id"],
                browsergym_id=f"browsergym.webarena_verified@{task_suffix}",
                sites=sites,
                worker=worker,
            )
        )
    return tasks


def _completed_attempt(task_dir: Path) -> dict[str, Any] | None:
    for path in sorted(task_dir.glob("attempt_*/model/episode_000/episode.json")):
        details = json.loads(path.read_text())
        if details.get("status") == "completed" and details.get("closed") is True:
            return details
    return None


def _next_attempt(task_dir: Path) -> int:
    attempts = [int(path.name.removeprefix("attempt_")) for path in task_dir.glob("attempt_*")]
    return max(attempts, default=-1) + 1


async def _start_episode(
    task: Task,
    attempt: int,
    *,
    args: argparse.Namespace,
    worker_url: str,
    model_config: Path,
    private_dir: Path,
) -> ActiveEpisode:
    task_dir = args.results_root / "tasks" / f"{task.numeric_id:03d}"
    output_dir = task_dir / f"attempt_{attempt:02d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    job = {
        "env": task.browsergym_id,
        "server": {"url": worker_url, "token": "sandbox-proxy-stripped"},
        "model_config": str(model_config),
        "output_dir": str(output_dir.resolve()),
        "episode_index": 0,
        "max_steps": args.max_steps,
        "model_timeout": args.model_timeout,
        "history_turns": args.history_turns,
        "metadata": {
            "official_task_id": task.numeric_id,
            "worker": task.worker,
            "attempt": attempt,
            "sites": list(task.sites),
        },
    }
    job_path = private_dir / f"task-{task.numeric_id:03d}-attempt-{attempt:02d}.json"
    _write_json(job_path, job, mode=0o600)
    log_file = (output_dir / "process.log").open("wb")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "lite.agents.episode",
        "--job",
        str(job_path),
        stdout=log_file,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    return ActiveEpisode(
        task=task,
        attempt=attempt,
        process=process,
        log_file=log_file,
        job_path=job_path,
        output_dir=output_dir,
        started_at=time.time(),
    )


def _public_task(task: Task, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "task_id": task.numeric_id,
        "env": task.browsergym_id,
        "sites": list(task.sites),
        "worker": task.worker,
        "status": status,
        **extra,
    }


async def _run(args: argparse.Namespace) -> int:
    if not args.sandbox_api_base:
        raise ValueError("Set BASE or pass --sandbox-api-base")
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not api_base:
        raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL are required")
    if args.max_steps < 1 or args.max_attempts < 1 or args.history_turns < 0:
        raise ValueError("Step, attempt, and history limits must be valid positive values")

    ramp = _parse_ramp(args.ramp)
    global_ramp = _parse_ramp(args.global_ramp)
    cluster = json.loads(args.cluster_json.read_text())
    workers = tuple(sorted(cluster["workers"], key=lambda item: item["worker"]))
    worker_numbers = tuple(item["worker"] for item in workers)
    layout = None
    if args.worker_layout:
        raw_layout = json.loads(args.worker_layout.read_text())
        if not isinstance(raw_layout, dict):
            raise ValueError("--worker-layout must contain a JSON object")
        layout = {}
        for worker, sites in raw_layout.items():
            number = int(worker)
            if number not in worker_numbers or not isinstance(sites, list) or not all(
                isinstance(site, str) for site in sites
            ):
                raise ValueError("Worker layout has an unknown worker or invalid site list")
            layout[number] = frozenset(sites)
    tasks = _load_tasks(args, worker_numbers, layout)
    args.results_root.mkdir(parents=True, exist_ok=True)

    worker_urls = {
        item["worker"]: (
            f"{args.sandbox_api_base.rstrip('/')}/v1/sandboxes/{item['id']}/proxy/30100"
        )
        for item in workers
    }
    queues = {number: [] for number in worker_numbers}
    public: dict[int, dict[str, Any]] = {}
    completed = 0
    for task in tasks:
        task_dir = args.results_root / "tasks" / f"{task.numeric_id:03d}"
        details = _completed_attempt(task_dir)
        if details is not None:
            public[task.numeric_id] = _public_task(
                task,
                "completed",
                episode_return=details.get("episode_return"),
                turns=details.get("turns"),
                resumed=True,
            )
            completed += 1
        else:
            queues[task.worker].append(task)

    run = {
        "benchmark": "webarena-verified",
        "version": webarena_verified.PACKAGE_VERSION,
        "dataset_sha256": webarena_verified.DATASET_SHA256,
        "model": args.model_id,
        "phase": args.phase,
        "scheduling": (
            "independent-stack-fast-first-pass"
            if args.phase == "write" and args.fast_write_shards
            else "catalog-ordered-single-writer"
            if args.phase == "write"
            else "read-parallel-map-on-leader"
        ),
        "ramp": [{"settled": threshold, "per_worker": limit} for threshold, limit in ramp],
        "global_ramp": [
            {"settled": threshold, "cluster_wide": limit}
            for threshold, limit in global_ramp
        ],
        "task_count": len(tasks),
        "workers": [
            {
                "worker": item["worker"],
                "sandbox_id": item["id"],
                "sites": sorted(layout.get(item["worker"], ())) if layout else None,
            }
            for item in workers
        ],
        "started_at": time.time(),
    }
    _write_json(args.results_root / "run.json", run)

    state_path = args.results_root / "state.json"
    active: dict[int, ActiveEpisode] = {}
    attempts: dict[int, int] = {}
    settled_this_run = failed_this_run = 0
    worker_cursor = 0
    stopping = False
    loop = asyncio.get_running_loop()

    def stop() -> None:
        nonlocal stopping
        stopping = True

    for name in ("SIGINT", "SIGTERM"):
        loop.add_signal_handler(getattr(signal, name), stop)

    def publish() -> None:
        _write_json(
            state_path,
            {
                **run,
                "updated_at": time.time(),
                "completed": sum(item["status"] == "completed" for item in public.values()),
                "failed": sum(item["status"] == "failed" for item in public.values()),
                "running": len(active),
                "pending": sum(len(queue) for queue in queues.values()),
                "current_per_worker_limit": _limit_for(ramp, settled_this_run),
                "current_global_limit": _limit_for(global_ramp, settled_this_run),
                "tasks": [public[key] for key in sorted(public)],
            },
        )

    with tempfile.TemporaryDirectory(prefix="wav-model-") as private:
        private_dir = Path(private)
        private_dir.chmod(0o700)
        model_config = private_dir / "model-api.json"
        _write_json(
            model_config,
            {
                "base_url": api_base,
                "api_key": api_key,
                "model": args.model_id,
                "api_type": "openai_chat",
                "params": {
                    "temperature": 0,
                    "enable_thinking": False,
                    "max_tokens": 4096,
                },
            },
            mode=0o600,
        )

        publish()
        while any(queues.values()) or active:
            per_worker_limit = _limit_for(ramp, settled_this_run)
            global_limit = _limit_for(global_ramp, settled_this_run)
            if settled_this_run >= 10 and failed_this_run / settled_this_run > 0.3:
                stopping = True

            if not stopping:
                while len(active) < global_limit:
                    selected = None
                    for offset in range(len(worker_numbers)):
                        index = (worker_cursor + offset) % len(worker_numbers)
                        worker = worker_numbers[index]
                        running = sum(
                            item.task.worker == worker for item in active.values()
                        )
                        if queues[worker] and running < per_worker_limit:
                            selected = (index, worker)
                            break
                    if selected is None:
                        break
                    index, worker = selected
                    worker_cursor = (index + 1) % len(worker_numbers)
                    task = queues[worker].pop(0)
                    attempt = attempts.setdefault(
                        task.numeric_id,
                        _next_attempt(args.results_root / "tasks" / f"{task.numeric_id:03d}"),
                    )
                    episode = await _start_episode(
                        task,
                        attempt,
                        args=args,
                        worker_url=worker_urls[worker],
                        model_config=model_config,
                        private_dir=private_dir,
                    )
                    active[task.numeric_id] = episode
                    public[task.numeric_id] = _public_task(
                        task, "running", attempt=attempt, started_at=episode.started_at
                    )
                    print(
                        f"start task={task.numeric_id} worker={worker} attempt={attempt} "
                        f"per_worker_limit={per_worker_limit} global_limit={global_limit}",
                        flush=True,
                    )
                publish()

            if not active:
                break
            waiters = [asyncio.create_task(item.process.wait()) for item in active.values()]
            done, pending = await asyncio.wait(
                waiters,
                timeout=5,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for waiter in done:
                waiter.result()
            for waiter in pending:
                waiter.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task_id, episode in list(active.items()):
                if episode.process.returncode is None:
                    continue
                episode.log_file.close()
                episode.job_path.unlink(missing_ok=True)
                details_path = episode.output_dir / "model" / "episode_000" / "episode.json"
                details = json.loads(details_path.read_text()) if details_path.exists() else {}
                success = episode.process.returncode == 0 and details.get("status") == "completed"
                settled_this_run += 1
                if success:
                    completed += 1
                    public[task_id] = _public_task(
                        episode.task,
                        "completed",
                        attempt=episode.attempt,
                        episode_return=details.get("episode_return"),
                        turns=details.get("turns"),
                        duration=time.time() - episode.started_at,
                    )
                else:
                    failed_this_run += 1
                    next_attempt = episode.attempt + 1
                    attempts[task_id] = next_attempt
                    if next_attempt < args.max_attempts and not stopping:
                        queues[episode.task.worker].append(episode.task)
                        public[task_id] = _public_task(
                            episode.task,
                            "retrying",
                            attempt=episode.attempt,
                            error=details.get(
                                "error", f"process exit {episode.process.returncode}"
                            ),
                        )
                    else:
                        public[task_id] = _public_task(
                            episode.task,
                            "failed",
                            attempt=episode.attempt,
                            error=details.get(
                                "error", f"process exit {episode.process.returncode}"
                            ),
                        )
                print(
                    f"finish task={task_id} worker={episode.task.worker} "
                    f"status={public[task_id]['status']}",
                    flush=True,
                )
                del active[task_id]
                publish()

            if stopping:
                break

        if stopping and active:
            for episode in active.values():
                try:
                    os.killpg(episode.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            await asyncio.gather(*(item.process.wait() for item in active.values()))
            for episode in active.values():
                episode.log_file.close()
                episode.job_path.unlink(missing_ok=True)

    publish()
    unfinished = any(queues.values()) or active
    return 0 if not unfinished and not any(
        item["status"] == "failed" for item in public.values()
    ) else 1


def main(argv: list[str] | None = None) -> int:
    """Parse operator inputs, validate the run, and execute the resumable queue."""
    args = _parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
