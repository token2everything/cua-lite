"""Runtime no-tool response/final handling for agent sampling loops.

Run:
    uv run pytest tests/agents/core/agent/test_rollout_loop_role_tool.py -q
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from lite.core import STATUS_FAILED, STATUS_TRUNCATED, LiteRLStep
from lite.core.messages.final import (
    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY,
    MODEL_OUTPUT_ERROR_KEY,
    PARSE_FAILURE_FINAL_REASON,
    STOP_REASON_INFO_KEY,
    make_no_tool_call_final_actions,
    summarize_no_tool_call_final,
)
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvStepResult,
    LiteExecutedAction,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoToolCallFinal:
    """A turn that parsed to zero tool calls, classified once for the loops.

    ``stop_reason`` is the internal classification: ``parse_failure`` means
    malformed model output; ``content_only_final`` / ``empty`` mean a deliberate
    response attempt with no parsed tool call. Ordinary assistant tool-call
    turns never instantiate this type, so loops keep the action vs no-action
    split at the parse boundary and never re-derive it.
    """

    #: Assistant turn to persist in place of the live one.
    message: dict[str, Any]
    #: Transient internal response call to step the env with (never persisted).
    actions: list[dict[str, Any]]
    #: Internal no-tool-call classification, owned by ``lite.core.messages``.
    stop_reason: str


def begin_no_tool_call_final(
    message: dict[str, Any],
    *,
    model_output_error: str | None,
    step: LiteRLStep,
) -> NoToolCallFinal:
    """Classify a zero-tool-call turn and stage its env response step.

    No parseable tool call is classified HERE, once, off the producer's own
    output: MALFORMED when ``predict`` reported a parse error, DELIBERATE
    RESPONSE otherwise. The visible text may be "" (reasoning-only, refusal-only,
    empty output): empty content, not a failure. The env evaluates the response
    and decides whether the rollout terminates, except malformed outputs always
    stay terminal.

    Being malformed is what demotes ``step.status``, so that consequence lives
    with the classification instead of being re-derived at each sampling loop.
    A truncated turn is WHY the tool call did not parse, so the truncation
    signal wins: the cause is never overwritten with the symptom.

    ``Done.`` is a structural data-preproc/migration marker, not a universal
    live-rollout replacement. Runtime still sends the visible text to the env's
    response path, but the persisted assistant turn keeps the exact Lite
    content produced by the adapter so raw logs do not misrepresent the model.
    """
    summary = summarize_no_tool_call_final({**message, MODEL_OUTPUT_ERROR_KEY: model_output_error})
    stop_reason = summary.stop_reason
    if stop_reason == PARSE_FAILURE_FINAL_REASON:
        logger.warning(
            "Model output parse failed; treating as terminal parse-failure final: %s",
            model_output_error,
        )
        if step.status != STATUS_TRUNCATED:
            step.status = STATUS_FAILED
    logger.info(
        "No tool_calls in model response; treating turn as %s response/final (text=%d chars)",
        stop_reason,
        len(summary.visible_text),
    )
    persisted = {
        key: value
        for key, value in dict(message).items()
        if key not in ("tool_calls", "raw_response", MODEL_OUTPUT_ERROR_KEY)
    }
    persisted["role"] = "assistant"
    persisted.setdefault("content", [])
    if model_output_error:
        persisted[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY] = summary.diagnostic()
    return NoToolCallFinal(
        message=persisted,
        actions=make_no_tool_call_final_actions(summary.visible_text),
        stop_reason=stop_reason,
    )


def mark_no_tool_call_final_result(
    step_result: LiteEnvStepResult,
    final: NoToolCallFinal,
) -> LiteEnvStepResult:
    """Attach zero-tool-call response metadata, without changing reward.

    Single writer for every fact a zero-call response/final adds to the env
    result: the executed transient action and, once terminal, a default stop
    reason if the env did not already provide one. Deliberate content-only
    answers follow the env's own terminal flags and stop reason; malformed
    outputs are forced terminal here because they are not valid answer attempts.
    """
    parse_failure = final.stop_reason == PARSE_FAILURE_FINAL_REASON
    if parse_failure:
        step_result.terminated = True
        step_result.truncated = False
    info = dict(step_result.info or {})
    executed: list[LiteExecutedAction] = [
        {"call": tool_call_name(action), "args": dict(tool_call_arguments(action))}
        for action in final.actions
    ]
    info[EXECUTED_ACTIONS_INFO_KEY] = executed
    if parse_failure:
        info[STOP_REASON_INFO_KEY] = final.stop_reason
    elif step_result.terminated or step_result.truncated:
        info.setdefault(STOP_REASON_INFO_KEY, final.stop_reason)
    step_result.info = info
    return step_result


__all__ = [
    "NoToolCallFinal",
    "begin_no_tool_call_final",
    "mark_no_tool_call_final_result",
]
