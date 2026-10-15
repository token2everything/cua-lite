"""Evaluation for lite_osworld tasks.

Reward is the raw OSWorld aggregate (partial credit preserved, not binarized):
"and" → 0.0 if any sub-metric is 0.0 else the mean; "or" → 1.0 if any is 1.0
else the max; a single metric reduces to its raw value. The infeasible path is
still binary (1.0 iff the agent correctly reported infeasible).
Uses computer.interface for most operations, Flask server only for
/accessibility (AT-SPI) and /terminal. OSWorld metrics run on host.

Usage:
    from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
"""

from __future__ import annotations

import logging
from typing import Any

from lite.core.tools.calls import (
    tool_call_arguments,
    tool_call_name,
    validate_lite_tool_call,
)
from lite.gym.envs.lite.osworld.src.eval.runner import evaluate_osworld_task
from lite.gym.sandbox.types import SandboxTaskConfig

logger = logging.getLogger(__name__)


def _action_call(action: Any) -> tuple[str, dict[str, Any]]:
    """Read ``(name, arguments)`` off a final action. NEVER raises.

    THE CONTRACT IS UPHELD; THIS READER IS NOT WHAT UPHOLDS IT. On rollout,
    every element of ``actions`` is an ``accepted_actions`` entry from
    ``SandboxBaseEnv._finalize_step_result`` — an env-internal accepted action
    ``{"name": str, "arguments": dict}``, already validated by
    ``lite.gym.utils.feedback.ingress.prepare_env_tool_calls``. Canonical test
    and replay callers may hand this reward boundary a Lite tool-call envelope;
    those are read only through ``lite.core.tools.calls`` canonical accessors.
    Top-level ``name`` / ``arguments`` reads below are for the env-internal
    accepted-action shape, not a second Lite protocol spelling.

    Unlike the ScaleCUA twin, lite.osworld has NO direct out-of-env caller that
    supplies OpenAI-style provider actions: both harnesses under
    ``devs/envs/lite.osworld/validate/`` call ``evaluate_final_fn(task,
    computer)`` with ``actions=None``, so no provider passthrough can hand
    ``arguments`` over as a JSON string here — and this reader deliberately does
    NOT decode that shape.

    SO WHY IS IT TOTAL? Because of where it sits, not what it distrusts. This is
    the REWARD boundary. A raise here does not degrade the score, it errors the
    episode out of the eval DENOMINATOR entirely — the task silently disappears
    from the mean instead of earning the ``0.0`` an unrecognized final action is
    supposed to earn. That asymmetry (lose the sample vs. score it) is what buys
    the totality; a hypothetical malformed producer is not. Nothing upstream may
    treat this tolerance as licence to relax the canonical shape: the
    ``prepare_env_tool_calls`` gate remains the contract, and the osworld tests
    pin it.
    """
    if not isinstance(action, dict):
        return "", {}
    if validate_lite_tool_call(action, "action", require_id=False) is None:
        return tool_call_name(action), tool_call_arguments(action)
    name = action.get("name")
    args = action.get("arguments")
    if not isinstance(name, str) or not isinstance(args, dict):
        return "", {}
    return name, args


async def evaluate_final_fn(
    task: SandboxTaskConfig, computer, actions: list | None = None, debug: bool = False,
) -> float | tuple[float, dict]:
    """Raw OSWorld aggregate reward (partial credit preserved). Infeasible tasks
    are binary: 1.0 iff the agent correctly reported infeasible, else 0.0."""
    evaluator = task.metadata.get("evaluator", {})
    if not evaluator:
        logger.warning("Task %s has no evaluator", task.task_id)
        return (0.0, {"error": "no evaluator"}) if debug else 0.0

    # Infeasible tasks: reward 1.0 if agent correctly reported infeasible
    func = evaluator.get("func", "")
    if func == "infeasible" and actions:
        for action in actions:
            name, args = _action_call(action)
            if name == "report_infeasible":
                return (1.0, {"infeasible": True}) if debug else 1.0
            if name == "terminate" and args.get("status") == "failure":
                return (1.0, {"infeasible": True}) if debug else 1.0
            if name == "response" and "[infeasible]" in str(args.get("text", "")).lower():
                return (1.0, {"infeasible": True}) if debug else 1.0
        return (0.0, {"infeasible": False}) if debug else 0.0

    return await evaluate_osworld_task(computer, evaluator, debug=debug)
