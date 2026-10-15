#!/usr/bin/env python3
"""Run one frozen generated task with the teacher/GLM shared desktop contract.

The coordinator restores the site and supplies its consumed-once baseline.
Deploy this directory, the frozen desktop helper, and the pinned OSWorld cursor
checkout in the desktop runtime; site servers and Docker remain coordinator
owned. Both configured gateways see 1920x1080 screenshots and the same canonical
GUI actions. Provider tool declarations are archived separately from canonical
trajectory tools. Private model parameters, including reasoning, are preserved.

Example (inside the configured desktop container):
  /opt/venv/bin/python -B run_desktop_reference.py \
    --private /batch/private.jsonl --generated-task-id <generated-id> \
    --site-urls /private/site-urls.json --baseline-receipt /private/baseline.json \
    --model-config /private/models.json --model-profile teacher \
    --model-id gpt-5.6-sol --output-dir /batch/reference_runs/<generated-id> \
    --desktop-helpers /root/cua-lite/scripts/webarena_verified_qwen.py \
    --desktop-helpers-sha256 <frozen-source-sha256> --osworld-root /opt/OSWorld
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
from desktop_env import SCREEN_SIZE, GeneratedDesktopEnv, load_desktop_helpers
from generated_env import build_public_instruction, resolve_public_task
from run_reference import build_agent, validate_execution_baseline


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--generated-task-id", required=True)
    parser.add_argument("--site-urls", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--model-profile")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--desktop-helpers", type=Path, required=True)
    parser.add_argument("--desktop-helpers-sha256", required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path)
    parser.add_argument("--display", default=":99")
    parser.add_argument("--lock-dir", type=Path, default=Path("/tmp/wavgen-desktop-locks"))
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--model-timeout", type=float, default=600)
    parser.add_argument("--history-turns", type=int, default=20)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs without consuming the baseline"
    )
    args = parser.parse_args(argv)
    if args.max_steps < 1 or args.model_timeout <= 0 or args.history_turns < 0:
        parser.error("steps and timeout must be positive; history turns must be nonnegative")
    return args


async def run_desktop_reference(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    site_urls: dict[str, str],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Run native sampling and bind its complete artifacts to generated scoring."""
    from importlib.metadata import version

    from webarena_verified.core.utils.checksum import compute_evaluator_checksum

    from lite.agents.core.agent.jsonl_logger import JsonlTrajectoryLogger
    from lite.agents.core.agent.logger import TrajectoryLogger
    from lite.agents.core.agent.utils.request_logs import redact
    from lite.agents.extensions.api_config import load_profile
    from lite.core.tools.action_space import LiteDesktopActionSet

    profile = load_profile(args.model_config, args.model_profile)
    if profile["model"] != args.model_id:
        raise ValueError("--model-id must match the actual selected model profile")
    if os.environ.get("CUA_LITE_ENV_SERVER_URL"):
        raise ValueError("desktop reference owns a local browser; unset remote env routing")
    helpers = load_desktop_helpers(args.desktop_helpers, args.desktop_helpers_sha256)
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
    env = GeneratedDesktopEnv(args, candidate, site_urls, output, helpers)
    canonical_tools = [
        *LiteDesktopActionSet.get_tool_schemas(include=env.metadata.valid_actions),
        *env.metadata.extra_tool_schemas,
    ]
    (output / "canonical_tools.json").write_text(
        json.dumps(canonical_tools, ensure_ascii=False, indent=2)
    )
    result = None
    execution_errors = []
    secrets = (profile["api_key"], profile["base_url"])
    try:
        agent = build_agent(args, profile, env.metadata, output)
        (output / "provider_tools.json").write_text(
            json.dumps(agent._build_tools(), ensure_ascii=False, indent=2)
        )
        (output / "run_metadata.json").write_text(
            json.dumps(
                {
                    "entrypoint": "run_desktop_reference.py",
                    "model_id": args.model_id,
                    "api_type": profile["api_type"],
                    "screen_size": list(SCREEN_SIZE),
                    "coordinates": "normalized_0_1000",
                    "browser_context": "per_case_persistent_profile",
                    "accept_downloads": False,
                    "max_steps": args.max_steps,
                    "chat_history_turns": (
                        agent.history_turns if profile["api_type"] == "openai_chat" else None
                    ),
                    "settle_seconds": {
                        "initial": helpers.RESET_SETTLE_SECONDS,
                        "post_action": helpers.POST_ACTION_DELAY_SECONDS,
                        "final": helpers.EVALUATION_SETTLE_SECONDS,
                    },
                    "model_parameters": redact(profile.get("params", {}), secrets),
                    "source_sha256": {
                        "entrypoint": sha256_file(Path(__file__)),
                        "desktop_env": sha256_file(Path(__file__).with_name("desktop_env.py")),
                        "desktop_helpers": args.desktop_helpers_sha256,
                        "cursor": sha256_file(
                            args.osworld_root / "desktop_env/server/pyxcursor.py"
                        ),
                    },
                    "canonical_tools": "canonical_tools.json",
                    "provider_tools": "provider_tools.json",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        hooks = [
            TrajectoryLogger(
                output,
                env_id="wavgen_v04_desktop",
                task_id=candidate["generated_task_id"],
                provenance={
                    "reference_kind": "independent_model_execution",
                    "model_id": args.model_id,
                    "public_projection_sha256": sha256_json(public_projection(candidate)),
                },
            ),
            JsonlTrajectoryLogger(
                output, model_id=args.model_id, secrets=secrets, get_tools=lambda: canonical_tools
            ),
        ]
        result = await agent.sample(env, max_steps=args.max_steps, hooks=hooks)
    except Exception as error:
        execution_errors.append(redact(str(error), secrets))
    finally:
        try:
            await env.close()
        except Exception as error:
            execution_errors.append(redact(str(error), secrets))
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
        screen_size=list(SCREEN_SIZE),
        coordinates="normalized_0_1000",
        steps=len(result.steps) if result is not None else None,
        terminated=result.terminated if result is not None else False,
        truncated=result.truncated if result is not None else env.truncated,
    )
    if execution_errors:
        result_receipt.update(
            status="error", execution_status="failed", execution_errors=execution_errors
        )
    result_receipt["checks"].update(
        canonical_trajectory_present=(output / "trajectory.parquet").is_file(),
        native_jsonl_present=(output / "agent_trajectory.jsonl").is_file(),
        canonical_tools_present=(output / "canonical_tools.json").is_file(),
    )
    for name, relative in {
        "trajectory": "trajectory.parquet",
        "native_jsonl": "agent_trajectory.jsonl",
        "canonical_tools": "canonical_tools.json",
        "provider_tools": "provider_tools.json",
        "execution_baseline": "execution_baseline.json",
        "public_task": "public_task.json",
        "agent_response": "agent_response.txt",
        "run_metadata": "run_metadata.json",
        "desktop_actions": "desktop_actions.jsonl",
        "terminal_state": "terminal_state.json",
        "instruction": "instruction.txt",
    }.items():
        path = output / relative
        if path.is_file():
            result_receipt["artifacts"][name] = relative
            result_receipt["artifact_sha256"][name] = sha256_file(path)
    if result_receipt["status"] == "pass" and not all(result_receipt["checks"].values()):
        result_receipt["status"] = "fail"
    result_path.write_text(json.dumps(result_receipt, ensure_ascii=False, indent=2) + "\n")
    for path in output.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    return result_receipt


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    matches = [
        row
        for line in args.private.read_text().splitlines()
        if line.strip() and (row := json.loads(line))["generated_task_id"] == args.generated_task_id
    ]
    if len(matches) != 1:
        raise ValueError("select exactly one generated candidate")
    candidate = matches[0]
    validate_candidate_shape(candidate)
    validate_upstream_task_spec(candidate["task_spec"])
    site_urls = json.loads(args.site_urls.read_text())
    receipt = json.loads(args.baseline_receipt.read_text())
    validate_execution_baseline(receipt, candidate, site_urls)
    if args.dry_run:
        print(build_public_instruction(resolve_public_task(candidate, site_urls)))
        return
    args.lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (args.lock_dir / f"{receipt['sandbox_id']}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = asyncio.run(run_desktop_reference(args, candidate, site_urls, receipt))
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
