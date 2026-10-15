"""Unified inference-entry orchestration: run an agent (local OR API) on an env's tasks.

One driver for the older split local/API and sample/rollout entrypoints. The
two axes are just flags:

  * **task count** — ``--task-id T`` pins a single task (the old ``sample``); omit it
    to run every task (the old ``rollout``). Single-task is a degenerate rollout.
  * **backend** — ``make(model_id)`` routes local-vs-API off the id
    (``AGENTS = {**LOCAL_AGENTS, **API_AGENTS}``). Local models build a
    ``generate_fn`` (sglang server or in-process hf) via
    :func:`lite.infer.serving.local_model`; API models (gpt/claude) don't.

The CLI shell is ``scripts/rollout.py``; everything reusable/testable lives here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lite.agents.factory import AGENTS, LOCAL_AGENTS
from lite.infer.rollout import (
    _merge_agent_kwargs,
    check_ci_gates,
    make_rollout_parser,
    run_rollout,
)
from lite.infer.serving import extract_sampling_kwargs, local_model


def _log_contract_roots(log_root: Path, splits: list[str] | None) -> list[Path]:
    """Return the rollout subtrees owned by the current command."""
    if splits:
        return [log_root / split for split in splits]
    return [log_root]


def make_infer_parser() -> argparse.ArgumentParser:
    """The unified parser: shared rollout flags + model/backend + single-task pin."""
    parser = argparse.ArgumentParser(
        parents=[make_rollout_parser()],
        description="Run an agent (local or API) on an env's tasks. "
                    "--task-id pins one task (sample mode); else all tasks (rollout mode).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", choices=list(AGENTS), required=True,
                        help="Model id (key in AGENTS). Local (sglang/hf) or API (gpt/claude) "
                             "— make routes on the id.")
    parser.add_argument("--task-id", default=None,
                        help="Pin a single task (sample mode). Omit to run all tasks.")

    # --- local-model-only flags (ignored for API models) ---
    parser.add_argument("--model-path", default=None,
                        help="[local] Override model path (e.g. a local checkpoint).")
    parser.add_argument("--backend", choices=("sglang", "hf"), default="sglang",
                        help="[local] sglang (server, default) or hf (in-process).")
    parser.add_argument("--sglang-server-url", default=os.environ.get("SGLANG_SERVER_URL"),
                        help="[local] Connect to an existing sglang server instead of "
                             "auto-starting one (default: $SGLANG_SERVER_URL). Distinct from "
                             "$CUA_LITE_ENV_SERVER_URL (the env-server, lite/gym/remote/).")
    parser.add_argument("--sampling-kwargs", type=json.loads, default=None,
                        help='[local] JSON sampling params, e.g. \'{"max_new_tokens": 2048}\'.')
    parser.add_argument("--engine-kwargs", type=json.loads, default=None,
                        help='[local] JSON sglang Engine overrides, e.g. \'{"tp_size": 4}\'.')

    # --- API-model-only flag ---
    parser.add_argument("--api-kwargs", type=json.loads, default=None, dest="api_kwargs",
                        help='[api] JSON provider params (max_tokens etc.), merged into '
                             'agent_kwargs["api_kwargs"].')
    parser.set_defaults(save_video=True)
    return parser


async def _rollout_with_retries(args: argparse.Namespace, agent_kwargs: dict) -> None:
    """run_rollout up to ``--max-attempts`` (each attempt resumes missing samples),
    then the opt-in CI gates. This block used to be copy-pasted in both rollout scripts.
    """
    model_path = args.model_path or args.model_id
    for attempt in range(1, args.max_attempts + 1):
        if attempt > 1:
            print(f"\n=== Retry attempt {attempt}/{args.max_attempts} ===\n")
        all_done, resolved_log_root = await run_rollout(
            model_id=args.model_id, model_path=model_path, env_id=args.env_id,
            agent_kwargs=agent_kwargs, env_kwargs=args.env_kwargs, seed=args.seed,
            group_size=args.group_size, concurrency=args.concurrency,
            log_root=args.log_root, prompt_data=args.prompt_data, splits=args.splits,
            head=args.head, sample=args.sample, filter_expr=args.filter_expr,
            task_id=args.task_id, config_path=args.config_path,
            save_data=args.save_data, save_video=args.save_video, save_gif=args.save_gif,
            debug=args.debug,
            render_instruction_banner=args.render_instruction_banner,
            group_shared_seed=args.group_shared_seed,
        )
        args.log_root = str(resolved_log_root)  # resume subsequent attempts in-place
        if all_done:
            if attempt > 1:
                print(f"\n=== All tasks complete (attempt {attempt}/{args.max_attempts}) ===\n")
            break
        if attempt == args.max_attempts:
            print(f"\n=== Max attempts ({args.max_attempts}) reached, some tasks incomplete ===\n")

    if args.min_valid_frac is not None or args.min_mean_return is not None:
        reasons = check_ci_gates(
            Path(args.log_root) / "summary.json",
            min_valid_frac=args.min_valid_frac,
            min_mean_return=args.min_mean_return,
        )
        if reasons:
            print("\n=== CI gate FAILED ===", file=sys.stderr)
            for r in reasons:
                print(f"  {r}", file=sys.stderr)
            sys.exit(1)
        print("\n=== CI gates pass ===\n")

    if args.debug:
        from lite.infer.debug.log_contract import check_log_contract

        reasons: list[str] = []
        for root in _log_contract_roots(Path(args.log_root), args.splits):
            reasons.extend(check_log_contract(root))
        if reasons:
            print("\n=== Log contract gate FAILED ===", file=sys.stderr)
            for r in reasons:
                print(f"  {r}", file=sys.stderr)
            sys.exit(1)
        print("\n=== Log contract gate pass ===\n")


async def run_infer(args: argparse.Namespace) -> None:
    """Dispatch local/API, build agent_kwargs, then run the (retrying) rollout."""
    from lite.utils.config import load_config

    # CLI agent_kwargs; run_rollout deep-merges yaml UNDER this (args > yaml,
    # per-leaf) — the one place the merge happens, so it is DEEP for every path
    # (the older sample entrypoint used a shallow merge that dropped nested yaml keys).
    cli_agent_kwargs = dict(args.agent_kwargs or {})

    if args.model_id in LOCAL_AGENTS:
        supported = LOCAL_AGENTS[args.model_id].get("backends")
        if supported is not None and args.backend not in supported:
            raise ValueError(
                f"Model '{args.model_id}' only supports backends {supported}, "
                f"got '{args.backend}'"
            )
        # sampling_kwargs feeds the generate_fn (not the adapter). Resolve it from the
        # DEEP-merged (yaml < CLI) agent_kwargs so a yaml-only sampling_kwargs is honored.
        file_cfg = load_config(args.config_path) if args.config_path else {}
        # `or {}`: a yaml `agent_kwargs:` with only comments parses to None, and
        # `.get(k, {})` returns that None (key present) → deep_merge(None, …) crashes.
        merged_ak = _merge_agent_kwargs(file_cfg.get("agent_kwargs") or {}, cli_agent_kwargs)
        sampling_kwargs = extract_sampling_kwargs(merged_ak, args.sampling_kwargs)
        async with local_model(
            args.model_id, model_path=args.model_path, backend=args.backend,
            engine_kwargs=args.engine_kwargs, server_url=args.sglang_server_url,
            sampling_kwargs=sampling_kwargs,
        ) as (processor, generate_fn):
            cli_agent_kwargs["processor"] = processor
            cli_agent_kwargs["generate_fn"] = generate_fn
            await _rollout_with_retries(args, cli_agent_kwargs)
    else:
        # API model (gpt/claude): no generate_fn; provider params live under api_kwargs.
        if args.api_kwargs is not None:
            cli_agent_kwargs.setdefault("api_kwargs", {}).update(args.api_kwargs)
        await _rollout_with_retries(args, cli_agent_kwargs)
