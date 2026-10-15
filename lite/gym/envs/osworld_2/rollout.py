"""Coordinate OSWorld tasks, phase transitions, evaluation and final cleanup.

python -m lite.gym.envs.osworld_2.rollout --release osworld-v2.1 \
    --task-set offline --plan-only
Execution is supplied by the Sandbox coordinator; each phase runs an independent
generic Agent episode through the native env-server. Model credentials are never
loaded by this benchmark coordinator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import lite.gym as gym

from .artifacts import DEFAULT_ROOT, Experiment, atomic_text, identifier, redact
from .catalog import DEFAULT_RELEASE, load_release
from .worker_client import WorkerClient
from .worker_env import DEFAULT_EXTRA_TOOLS


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-set", choices=("full", "offline"))
    selection.add_argument("--tasks", nargs="+")
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Explicitly skip protocol exclusions and record partial coverage",
    )
    parser.add_argument(
        "--plan-only", action="store_true", help="Show selection without network or credentials"
    )
    parser.add_argument(
        "--verify-protocols", action="store_true",
        help="Qualify explicitly selected candidate tasks before removing release exclusions",
    )
    parser.add_argument("--job", type=Path, help="Submitted job.json with public model metadata")
    parser.add_argument("--model-profile")
    parser.add_argument(
        "--connection",
        type=Path,
        help="Benchmark-local Worker connection; never mounted into the Agent",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--model-timeout", type=float, default=600)
    parser.add_argument("--cleanup-timeout", type=float, default=2100)
    history = parser.add_mutually_exclusive_group()
    history.add_argument(
        "--history-turns", type=int,
        help="Chat: past screenshots plus current screenshot; keep all text (default: 4)",
    )
    history.add_argument(
        "--full-history", action="store_true", help="Chat: send every historical screenshot"
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--environment-results-root",
        help="Environment log directory on the env-server host (separate from Agent logs)",
    )
    parser.add_argument("--experiment", type=identifier)
    return parser.parse_args(argv)


async def run(args, execute_episode=None):
    """Run sequentially; require a run-specific close receipt before Worker reuse."""
    release = load_release(args.release)
    plan = release.plan(
        task_set=args.task_set, tasks=args.tasks,
        skip_unsupported=args.skip_unsupported, verify_protocols=args.verify_protocols,
    )
    excluded = plan["unsupported"]
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return plan
    if excluded and not args.skip_unsupported and not args.verify_protocols:
        raise ValueError(
            "Unsupported tasks: "
            + ", ".join(item["task_id"] for item in excluded)
            + "; inspect --plan-only or explicitly pass --skip-unsupported"
        )
    if execute_episode is None or not args.job or not args.connection:
        raise ValueError("Execute tasks with the Sandbox coordinator; use --plan-only locally")
    if args.max_steps < 1 or args.model_timeout <= 0 or args.cleanup_timeout <= 0:
        raise ValueError("max-steps and timeouts must be positive")
    if args.history_turns is not None and args.history_turns < 0:
        raise ValueError("history-turns must be nonnegative")
    config = json.loads(args.job.read_text())["model"]
    if config["api_type"] != "openai_chat" and (
        args.history_turns is not None or args.full_history
    ):
        raise ValueError("history-turns/full-history apply to the Chat gateway only")
    # Validate the selected server catalog before allocating any episodes.
    for task_id in plan["scheduled_task_ids"]:
        metadata = gym.registry.task_metadata(release.env_id, task_id)
        if (reason := metadata.others.get("exclude_reason")) and not args.verify_protocols:
            raise ValueError(f"Task {task_id}: {reason}")
        if metadata.others["release_manifest_sha256"] != release.manifest_sha256:
            raise ValueError("Env-server release manifest differs from the requested release")
    connection = json.loads(args.connection.expanduser().read_text())
    worker = WorkerClient(connection["endpoint"], connection["token"])
    health = await asyncio.to_thread(worker.request, "GET", "/healthz")
    if "last_closed_run_id" not in health:
        raise ValueError(
            "Worker lacks close receipts; rebuild/deploy the migrated Worker before running"
        )
    if health["worker_id"] != connection["worker_id"] or health["release"] != release.name:
        raise ValueError("Worker identity/release differs from the selected connection")
    if health["state"] != "idle" or health["busy"]:
        raise ValueError("Allocate an idle Worker exclusively to this sequential runner")
    experiment = args.experiment or time.strftime("gateway-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    experiment_record = Experiment(
        experiment,
        args.results_root,
        release=release.name,
        config={
            "kind": "agent",
            "selection": plan,
            "model_id": config["model_id"],
            "api_type": config["api_type"],
        },
    )
    records = []
    for task_id in plan["scheduled_task_ids"]:
        # The coordinator assigns identity before crossing the env-server boundary.
        artifacts = experiment_record.new_run(
            task_id, secrets=[connection["token"]]
        )
        details = {
            "model_id": config["model_id"],
            "api_type": config["api_type"],
            "environment": release.env_id,
            "task_set": plan["task_set"],
            "closed": False,
        }
        artifacts.update(**details)
        options = dict(max_steps=args.max_steps, experiment=experiment, run_id=artifacts.run_id)
        if args.environment_results_root:
            options["results_root"] = args.environment_results_root
        status = "running"
        task_started = False
        pending = None
        pending_method = None
        phases = []
        try:
            # The task owns one VM; every Agent sample binds one phase of it.
            pending_method = "reset"
            pending = asyncio.create_task(asyncio.to_thread(
                worker.reset, task_id, artifacts.run_id,
            ))
            initial = await asyncio.shield(pending)
            pending = None
            task_started = True
            details.update(phase_count=initial["phase_count"],
                           has_user_simulator=initial["has_user_simulator"])
            options["extra_tools"] = [
                *DEFAULT_EXTRA_TOOLS,
                *(["ask_user"] if initial["has_user_simulator"] else []),
            ]
            if config["api_type"] == "openai_chat":
                details["history_scope"] = "screenshots"
                details["history_turns"] = (
                    None if args.full_history else
                    (4 if args.history_turns is None else args.history_turns)
                )
            artifacts.update(**details)
            print(
                json.dumps(
                    {"task_id": task_id, "output": str(artifacts.path), "status": "starting"}
                ),
                flush=True,
            )
            for phase_index in range(initial["phase_count"]):
                # Only the episode specification crosses into the Agent image.
                # The executor owns its env-server/Agent handles and settles both
                # before this coordinator advances or closes the desktop VM.
                specification = {
                    "env": f"{release.env_id}@{task_id}",
                    "env_kwargs": {**options, "phase_index": phase_index},
                    "model_profile": args.model_profile,
                    "output_dir": str(artifacts.path),
                    "episode_index": phase_index,
                    "max_steps": args.max_steps,
                    "model_timeout": args.model_timeout,
                    "metadata": {
                        "phase_index": phase_index, "run_id": artifacts.run_id,
                        "release": release.name, "task_id": task_id,
                    },
                }
                if config["api_type"] == "openai_chat":
                    specification["history_turns"] = details["history_turns"]
                result = await execute_episode(specification, artifacts, details)
                if result["status"] != "completed" or not result["closed"]:
                    raise RuntimeError(result.get("error", "Agent did not close its episode"))
                # Model stop flags alone cannot prove evaluation has completed;
                # native env close is asynchronous and its errors are logged.
                snapshot = await asyncio.to_thread(worker.task, artifacts.run_id)
                evaluation = snapshot["evaluation"]
                if (snapshot["busy"] or snapshot["phase_index"] != phase_index
                        or snapshot["state"] not in ("phase_evaluated", "evaluated")
                        or evaluation is None):
                    raise RuntimeError("Worker did not confirm the completed phase evaluation")
                phase = {
                    "phase_index": phase_index, "context_reset": True,
                    "message_start": result["message_start"],
                    "message_end": result["message_end"],
                    "score": evaluation["phase_score"], "turns": result["turns"],
                    "terminated": result["terminated"], "truncated": result["truncated"],
                }
                phases.append(phase)
                details.update(score=evaluation["score"], phases=phases,
                               terminated=result["terminated"], truncated=result["truncated"])
                artifacts.save_json("result.json", evaluation)
                artifacts.update(**details)
                if evaluation["complete"]:
                    break
                pending_method = "advance"
                pending = asyncio.create_task(asyncio.to_thread(
                    worker.advance, artifacts.run_id, phase_index=phase_index,
                ))
                await asyncio.shield(pending)
                pending = None
            else:
                raise RuntimeError("Worker exhausted phases without completing the task")
            status = "completed"
        except asyncio.CancelledError:
            status = "interrupted"
            details["error"] = "Model rollout was cancelled"
            raise
        except Exception as error:
            status = "failed"
            details["error"] = redact(str(error), artifacts.secrets)
            raise RuntimeError(details["error"]) from None
        finally:
            # The executor has settled its handles. Always close this task,
            # including cancellation and a lost reset response.
            if pending is not None:
                try:
                    await asyncio.shield(pending)
                    if pending_method == "reset":
                        task_started = True
                except Exception as error:
                    artifacts.event("pending_operation", method_name=pending_method,
                                    error=redact(str(error), artifacts.secrets))
            try:
                health = await asyncio.to_thread(worker.request, "GET", "/healthz")
                if health["run_id"] == artifacts.run_id:
                    deadline = asyncio.get_running_loop().time() + args.cleanup_timeout
                    while health["busy"] and health["state"] != "broken":
                        if asyncio.get_running_loop().time() >= deadline:
                            raise TimeoutError(
                                "Worker task operation did not settle before cleanup"
                            )
                        await asyncio.sleep(1)
                        health = await asyncio.to_thread(worker.request, "GET", "/healthz")
                    if health["state"] != "idle":
                        await asyncio.to_thread(worker.close, artifacts.run_id)
                    await asyncio.to_thread(
                        worker.wait_closed,
                        artifacts.run_id,
                        connection["worker_id"],
                        release.name,
                        timeout=args.cleanup_timeout,
                    )
                    details["closed"] = True
                elif task_started:
                    raise RuntimeError("Worker lost this runner's task before cleanup confirmation")
            except Exception as error:
                status = "failed"
                details["cleanup_error"] = redact(str(error), artifacts.secrets)
            trace_path = artifacts.path / "agent_trajectory.jsonl"
            if trace_path.exists():
                trace = json.loads(trace_path.read_text())
                trace["metadata"].update(
                    status=status,
                    environment_closed=details["closed"],
                    run_id=artifacts.run_id,
                    release=release.name,
                    task_set=plan["task_set"],
                    phases=phases,
                )
                atomic_text(
                    trace_path,
                    json.dumps(trace, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    + "\n",
                    sync=True,
                )
            record = artifacts.finish(status, **details)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        records.append(record)
        if details.get("cleanup_error"):
            raise RuntimeError(details["cleanup_error"])
    if not records:
        experiment_record.summarize()
    return records


if __name__ == "__main__":
    asyncio.run(run(_parse_args()))
