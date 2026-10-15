#!/usr/bin/env python3
"""Run one frozen generated task through an independent public-only GUI agent.

The coordinator supplies a fresh baseline and optional private preauthenticated
storage state. This entrypoint starts only its own browser context, never a site
server or container. Its output includes canonical trajectory.parquet, HAR,
Playwright trace, and reference_receipt.json; semantic failures are preserved.

Example:
  PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python run_reference.py \
    --private /batch/private.jsonl --generated-task-id wavgen-v04-batch-0001 \
    --site-urls /private/site-urls.json --baseline-receipt /private/baseline.json \
    --model-config /private/model.json --model-id <actual-model-id> \
    --output-dir /batch/reference_runs/wavgen-v04-batch-0001
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
from typing import Any

from contract import (
    public_projection,
    sha256_file,
    sha256_json,
    validate_candidate_shape,
    validate_upstream_task_spec,
)
from generated_env import (
    ENV_ID,
    build_public_instruction,
    register_generated_task,
    resolve_public_task,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--generated-task-id", required=True)
    parser.add_argument("--site-urls", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--model-profile")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path)
    parser.add_argument("--lock-dir", type=Path, default=Path("/tmp/wavgen-v04-sandbox-locks"))
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--model-timeout", type=float, default=180)
    parser.add_argument("--history-turns", type=int, default=20)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate frozen inputs and print the public prompt without executing "
            "or consuming the baseline."
        ),
    )
    args = parser.parse_args()
    if args.max_steps < 1 or args.model_timeout <= 0 or args.history_turns < 0:
        parser.error("steps and timeout must be positive; history turns must be nonnegative")
    return args


def validate_execution_baseline(
    receipt: dict[str, Any], candidate: dict[str, Any], site_urls: dict[str, str]
) -> None:
    """Bind one coordinator-owned fresh snapshot to one frozen public task."""
    required = {
        "session_id",
        "sandbox_id",
        "generated_task_id",
        "created_at",
        "sites",
        "fresh_baseline",
    }
    if required - receipt.keys() or receipt.get("fresh_baseline") is not True:
        raise ValueError("baseline receipt must identify a fresh coordinator-owned snapshot")
    if receipt["generated_task_id"] != candidate["generated_task_id"]:
        raise ValueError("baseline receipt is for a different generated task")
    for key in ("session_id", "sandbox_id", "created_at"):
        if not isinstance(receipt[key], str) or not receipt[key]:
            raise ValueError(f"baseline receipt requires nonempty {key}")
    if not receipt["sandbox_id"].replace("-", "").replace("_", "").isalnum():
        raise ValueError("sandbox ID must be a simple identifier")
    sites = set(candidate["task_spec"]["sites"])
    if set(receipt["sites"]) != sites or set(site_urls) != sites:
        raise ValueError("execution baseline must identify exactly the task's sites")
    if receipt["authentication_only_preparation"] is not True:
        raise ValueError("execution preparation must contain only normal authentication")
    for site in sites:
        for key in ("snapshot", "image_digest"):
            if receipt["sites"][site][key] != candidate["baseline"]["sites"][site][key]:
                raise ValueError(f"execution baseline differs from grounded {site}.{key}")
        if receipt["sites"][site]["site_url"].rstrip("/") != site_urls[site].rstrip("/"):
            raise ValueError(f"execution baseline endpoint differs for {site}")
    if candidate["baseline_sha256"] != sha256_json(candidate["baseline"]):
        raise ValueError("candidate baseline hash mismatch")
    if candidate["grounding"]["task_spec_sha256"] != sha256_json(candidate["task_spec"]):
        raise ValueError("candidate task differs from the frozen grounding")


def build_agent(
    args: argparse.Namespace, profile: dict[str, Any], metadata: Any, output: Path
) -> Any:
    """Use the configured gateway and preserve its explicit model parameters."""
    from lite.agents.extensions.chat_gateway.agent import ChatGatewayAgent
    from lite.agents.extensions.responses_gateway.agent import ResponsesGatewayAgent

    agent_class = (
        ChatGatewayAgent if profile["api_type"] == "openai_chat" else ResponsesGatewayAgent
    )
    history = (
        {"history_turns": profile.get("history_turns", args.history_turns)}
        if profile["api_type"] == "openai_chat"
        else {}
    )
    return agent_class(
        model_id=args.model_id,
        api_key=profile["api_key"],
        api_base=profile["base_url"],
        metadata=metadata,
        request_params=profile.get("params", {}),
        request_timeout=args.model_timeout,
        api_retry_max=2,
        request_log_dir=output / "requests",
        request_image_dir=output / "images",
        **history,
    )


async def run_reference(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    site_urls: dict[str, str],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Persist a full Lite episode and bind its scoring receipt to the real model."""
    from importlib.metadata import version

    from webarena_verified.core.utils.checksum import compute_evaluator_checksum

    import lite.gym as gym
    from lite.agents.core.agent.logger import TrajectoryLogger
    from lite.agents.core.agent.utils.request_logs import redact
    from lite.agents.extensions.api_config import load_profile

    profile = load_profile(args.model_config, args.model_profile)
    if profile["model"] != args.model_id:
        raise ValueError("--model-id must match the actual selected model profile")
    if os.environ.get("CUA_LITE_ENV_SERVER_URL"):
        raise ValueError(
            "reference execution uses a new local browser context; unset remote env routing"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "private_task.json").write_text(json.dumps(candidate["task_spec"]))
    (output / "execution_baseline.json").write_text(json.dumps(receipt, indent=2))
    (output / "public_task.json").write_text(
        json.dumps(public_projection(candidate), ensure_ascii=False, indent=2)
    )
    with args.baseline_receipt.with_suffix(".consumed.json").open("x") as stream:
        json.dump(
            {"generated_task_id": candidate["generated_task_id"], "output_dir": str(output)}, stream
        )
    logger = TrajectoryLogger(
        output,
        env_id=ENV_ID,
        task_id=candidate["generated_task_id"],
        provenance={
            "reference_kind": "independent_model_execution",
            "model_id": args.model_id,
            "public_projection_sha256": sha256_json(public_projection(candidate)),
        },
    )
    result = None
    execution_error = None
    env = None
    try:
        key = register_generated_task(
            candidate, site_urls, output, max_steps=args.max_steps, storage_state=args.storage_state
        )
        env = gym.make(key)
        agent = build_agent(args, profile, env.metadata, output)
        result = await agent.sample(env, max_steps=args.max_steps, hooks=[logger])
    except Exception as error:
        # Preserve a failed external model/browser attempt with redacted diagnostics.
        execution_error = redact(str(error), (profile["api_key"],))
    finally:
        if env is not None:
            await env.close()
        for path in output.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
    result_path = output / "reference_receipt.json"
    if result_path.is_file():
        result_receipt = json.loads(result_path.read_text())
    else:
        result_receipt = {
            "gate_version": 2,
            "reference_kind": "independent_model_execution",
            "generated_task_id": candidate["generated_task_id"],
            "status": "error",
            "execution_status": "failed",
            "agent_ended": False,
            "public_only": True,
            "task_spec_sha256": sha256_json(candidate["task_spec"]),
            "baseline_sha256": candidate["baseline_sha256"],
            "public_projection_sha256": sha256_json(public_projection(candidate)),
            "checks": {"scoring_receipt_present": False, "final_state_confirmed": False},
            "artifacts": {},
            "artifact_sha256": {},
        }
    result_receipt.update(
        model_id=args.model_id,
        api_type=profile["api_type"],
        webarena_verified_version=version("webarena-verified"),
        webarena_verified_evaluator_checksum=compute_evaluator_checksum(),
        execution_baseline_receipt_sha256=sha256_file(args.baseline_receipt),
        model_public_projection_sha256=sha256_json(resolve_public_task(candidate, site_urls)),
        steps=len(result.steps) if result is not None else None,
        terminated=result.terminated if result is not None else False,
        truncated=result.truncated if result is not None else False,
    )
    if execution_error is not None:
        result_receipt.update(
            status="error", execution_status="failed", execution_error=execution_error
        )
    trajectory = output / "trajectory.parquet"
    result_receipt["checks"]["canonical_trajectory_present"] = trajectory.is_file()
    if trajectory.is_file():
        result_receipt["artifacts"]["trajectory"] = "trajectory.parquet"
        result_receipt["artifact_sha256"]["trajectory"] = sha256_file(trajectory)
    for name, relative in {
        "har": "network.har",
        "trace": "trace.zip",
        "initial_screenshot": "initial.png",
        "terminal_screenshot": "terminal.png",
        "initial_observation": "initial_observation.json",
        "final_state": "final_state.json",
        "state_audit_screenshot": "state_audit.png",
        "execution_baseline": "execution_baseline.json",
        "public_task": "public_task.json",
        "agent_response": "agent_response.txt",
    }.items():
        if (output / relative).is_file():
            result_receipt["artifacts"][name] = relative
            result_receipt["artifact_sha256"][name] = sha256_file(output / relative)
    if result_receipt["status"] == "pass" and not all(result_receipt["checks"].values()):
        result_receipt["status"] = "fail"
    (output / "reference_receipt.json").write_text(
        json.dumps(result_receipt, ensure_ascii=False, indent=2) + "\n"
    )
    return result_receipt


def main() -> None:
    args = _parse_args()
    candidates = [
        json.loads(line) for line in args.private.read_text().splitlines() if line.strip()
    ]
    matches = [row for row in candidates if row["generated_task_id"] == args.generated_task_id]
    if len(matches) != 1:
        raise ValueError("select exactly one generated candidate")
    candidate = matches[0]
    validate_candidate_shape(candidate)
    validate_upstream_task_spec(candidate["task_spec"])
    site_urls = json.loads(args.site_urls.read_text())
    public = resolve_public_task(candidate, site_urls)
    receipt = json.loads(args.baseline_receipt.read_text())
    validate_execution_baseline(receipt, candidate, site_urls)
    if args.dry_run:
        print(build_public_instruction(public))
        return
    args.lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (args.lock_dir / f"{receipt['sandbox_id']}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = asyncio.run(run_reference(args, candidate, site_urls, receipt))
    print(
        json.dumps(
            {
                "generated_task_id": result["generated_task_id"],
                "status": result["status"],
                "steps": result["steps"],
            }
        )
    )
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
