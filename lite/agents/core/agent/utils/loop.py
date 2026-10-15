"""Provider-independent lifecycle helpers shared by the CUA-Lite agent loops.

Scope: RL-step construction, Lite turn execution, post-step accounting, hook
dispatch, and finalization. Provider request payloads, provider history, and
parse/provenance records stay with each owning loop.

``execute_lite_turn`` / ``record_lite_env_result`` / ``dispatch_lite_step_hooks``
/ ``build_lite_rl_sample`` are provider-native lifecycle steps; the adapter loop
in ``lite/agents/core/agent/base.py`` keeps those inline because it also carries
processed-image state and next-turn feedback. ``mark_steps_truncated`` and
``finalize_lite_sample`` are shared by every loop.

Run:
    uv run pytest tests/agents/core/agent/test_rollout_loop_role_tool.py -q
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Container, Sequence
from dataclasses import dataclass
from typing import Any

from lite.agents.core.agent.hooks import SampleHook, SampleStepData
from lite.agents.core.agent.utils.final import (
    begin_no_tool_call_final,
    mark_no_tool_call_final_result,
)
from lite.agents.core.agent.utils.tool_results import (
    align_tool_results_to_tool_calls,
    require_model_visible_unpaired_response_feedback,
)
from lite.agents.types import PredictResult
from lite.core import (
    STATUS_COMPLETED,
    STATUS_TRUNCATED,
    LiteMessage,
    LiteRLSample,
    LiteRLStep,
    LiteSample,
)
from lite.gym.base import LiteBaseEnv
from lite.gym.types import LiteEnvStepResult
from lite.utils.timer import timed


def append_lite_rl_step(
    *,
    steps: list[LiteRLStep],
    prompt: str,
    image_indices: Sequence[int],
    response: str,
    finish_reason: str | None,
    truncation_finish_reasons: Container[str],
) -> LiteRLStep:
    """Build and append the provider-loop RL record for one model turn.

    The provider's own budget-stop spelling is projected onto the canonical
    status here: a finish reason in ``truncation_finish_reasons`` records
    ``STATUS_TRUNCATED``, everything else records ``STATUS_COMPLETED``.
    """
    rl_step = LiteRLStep(
        prompt=prompt,
        image_indices=tuple(image_indices),
        response=response,
        response_tokens=[],
        response_log_probs=[],
        reward=0.0,
        status=(
            STATUS_TRUNCATED
            if (finish_reason or "").lower() in truncation_finish_reasons
            else STATUS_COMPLETED
        ),
    )
    steps.append(rl_step)
    return rl_step


@dataclass(frozen=True)
class ExecutedLiteTurn:
    """One executed assistant turn, with hook and persistence actions kept apart.

    ``hook_actions`` and ``persisted_actions`` are equal for an ordinary tool
    turn and deliberately differ for a zero-tool-call response/final: the env executes
    (and hooks/loggers report) a synthetic ``response(text=...)`` action that is
    never stored as an assistant tool call, so no unpaired ``role:"tool"``
    message is written. Env state lives on ``step_result``:
    ``terminated`` / ``truncated`` plus any stop reason in ``step_result.info``.
    """

    #: Assistant message persisted in the Lite trajectory for this turn.
    message: LiteMessage
    #: Actions actually passed to ``env.step()``; what hooks and loggers see.
    hook_actions: list[dict[str, Any]]
    #: Canonical assistant tool calls stored in the Lite sample; empty for a
    #: zero-tool-call response/final.
    persisted_actions: list[dict[str, Any]]
    #: Env result for the executed actions, already marked for no-tool response/finals.
    step_result: LiteEnvStepResult
    #: Wall-clock seconds spent inside ``env.step()``.
    act_s: float


async def execute_lite_turn(
    *,
    env: LiteBaseEnv,
    trajectory: LiteSample,
    lite_message: LiteMessage,
    model_output_error: str | None,
    rl_step: LiteRLStep,
    tool_execution_timeout_s: float | None,
) -> ExecutedLiteTurn:
    """Execute a Lite assistant turn, including the zero-tool-call response path."""
    persisted_actions = lite_message.get("tool_calls", [])
    final = None
    if persisted_actions:
        hook_actions = persisted_actions
    else:
        # Zero parsed tool calls go through the shared final helper. Deliberate
        # content-only answers follow the env's terminal flags; malformed
        # outputs still force terminal.
        final = begin_no_tool_call_final(
            lite_message,
            model_output_error=model_output_error,
            step=rl_step,
        )
        trajectory.messages[-1] = final.message
        lite_message = final.message
        # The runtime response action runs in the env and reaches hooks, but is
        # never persisted as an assistant tool call.
        hook_actions = final.actions

    step_coro = env.step(hook_actions)
    if tool_execution_timeout_s is not None:
        step_coro = asyncio.wait_for(step_coro, timeout=tool_execution_timeout_s)
    step_result, act_s = await timed(step_coro)
    if final is not None:
        step_result = mark_no_tool_call_final_result(
            step_result,
            final,
        )

    return ExecutedLiteTurn(
        message=lite_message,
        hook_actions=hook_actions,
        persisted_actions=persisted_actions,
        step_result=step_result,
        act_s=act_s,
    )


async def record_lite_env_result(
    *,
    trajectory: LiteSample,
    persisted_actions: list[dict[str, Any]],
    step_result: LiteEnvStepResult,
    rl_step: LiteRLStep,
    step: int,
    max_steps: int,
    episode_return: float,
    append_terminal_tool_feedback: Callable[..., Awaitable[None]],
) -> tuple[float, bool, bool]:
    """Apply reward/status accounting and persist terminal tool feedback.

    This is the env-result alignment boundary for provider-native loops: env
    results are paired to the persisted assistant calls once and written back
    onto ``step_result.results``, so provider history and message builders
    downstream consume the aligned order directly.

    ``append_terminal_tool_feedback`` is the provider's env-result persistence
    call, invoked with ``trajectory``, ``step_result``, and ``tool_calls``
    keywords once the episode has reached its last step.
    """
    if step_result.reward is not None:
        episode_return += step_result.reward
        rl_step.reward = float(step_result.reward)

    env_terminated = step_result.terminated
    env_truncated = step_result.truncated
    terminated = env_terminated
    agent_truncated = step + 1 >= max_steps and not env_terminated and not env_truncated
    truncated = env_truncated or agent_truncated
    if agent_truncated:
        step_result.truncated = True
    if truncated and rl_step.status == STATUS_COMPLETED:
        rl_step.status = STATUS_TRUNCATED

    require_tool_results = not (env_terminated or env_truncated)
    if persisted_actions:
        step_result.results = align_tool_results_to_tool_calls(
            persisted_actions,
            list(step_result.results),
            require_all=require_tool_results,
        )
    elif require_tool_results and not agent_truncated:
        require_model_visible_unpaired_response_feedback(step_result.results)
    if persisted_actions and (terminated or truncated):
        await append_terminal_tool_feedback(
            trajectory=trajectory,
            step_result=step_result,
            tool_calls=persisted_actions,
        )

    return episode_return, terminated, truncated


async def dispatch_lite_step_hooks(
    *,
    hooks: list[SampleHook],
    step_idx: int,
    image: Any,
    lite_message: LiteMessage,
    rl_step: LiteRLStep,
    step_result: LiteEnvStepResult,
    hook_actions: list[dict[str, Any]],
    timings: dict[str, float],
) -> None:
    """Emit the per-step hook payload off the event loop.

    Hooks are synchronous and may encode images and write files, so they run in
    a worker thread exactly as the adapter-backed loop runs them.
    """
    step_data = SampleStepData(
        step_idx=step_idx,
        image=image,
        predict_result=PredictResult(
            lite_message=lite_message,
            agent_message=lite_message,
            step=rl_step,
        ),
        step_result=step_result,
        actions=hook_actions,
        current_image_index=(rl_step.image_indices[-1] if rl_step.image_indices else None),
        timings=timings,
    )
    for hook in hooks:
        await asyncio.to_thread(hook.on_step, step_data)


def mark_steps_truncated(steps: list[LiteRLStep]) -> None:
    """Project a truncated episode onto its trailing step.

    Truncation is a trajectory-level fact; the per-step record that carries it
    is the last one. Only a ``STATUS_COMPLETED`` last step is restamped, so a
    step that already recorded its own worse outcome (``TRUNCATED`` from a
    provider budget stop, ``FAILED`` from a parse failure) keeps it.
    """
    if steps and steps[-1].status == STATUS_COMPLETED:
        steps[-1].status = STATUS_TRUNCATED


def build_lite_rl_sample(
    *,
    trajectory: LiteSample,
    steps: list[LiteRLStep],
    episode_return: float,
    terminated: bool,
    truncated: bool,
) -> LiteRLSample:
    """Build the provider-native loop result from its accumulated state.

    ``processed_images`` is the sparse projection of ``trajectory.images``.
    Provider-native loops use raw screenshots as their model-ready image form,
    so only frames a recorded step actually bound stay populated; stored but
    invisible action-batch frames remain ``None`` placeholders.
    """
    step_image_indices = {image_index for step in steps for image_index in step.image_indices}
    return LiteRLSample(
        lite_sample=trajectory,
        processed_images=[
            image if image_index in step_image_indices else None
            for image_index, image in enumerate(trajectory.images)
        ],
        steps=steps,
        episode_return=episode_return,
        terminated=terminated,
        truncated=truncated,
    )


async def finalize_lite_sample(
    *,
    hooks: list[SampleHook],
    result: LiteRLSample | None,
    completed: bool,
    env: LiteBaseEnv,
    logger: logging.Logger,
) -> None:
    """Run completion hooks off the event loop, then close the env.

    Hooks are synchronous and may encode images and write files, so completion
    runs in a worker thread. ``result`` reaches the hooks only when the loop
    finished; a failed loop passes ``None``.

    ``logger`` is the calling loop's logger so a teardown warning is attributed
    to the loop that owned the env.
    """
    try:
        for hook in hooks:
            await asyncio.to_thread(hook.on_complete, result if completed else None)
    finally:
        try:
            await env.close()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Env teardown is a transport/container boundary that can fail for
            # reasons unrelated to the rollout. Callers invoke this from a
            # ``finally``, so raising here would replace the sample (or the real
            # failure) with a teardown error; log it instead.
            logger.warning("env.close() failed: %s", e)


__all__ = [
    "ExecutedLiteTurn",
    "append_lite_rl_step",
    "build_lite_rl_sample",
    "dispatch_lite_step_hooks",
    "execute_lite_turn",
    "finalize_lite_sample",
    "mark_steps_truncated",
    "record_lite_env_result",
]
