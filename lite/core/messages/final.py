"""Core no-tool-call assistant final-turn message helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lite.core.messages.content import (
    ACTION_DESCRIPTION_PART,
    ASSISTANT_ROLE,
    HISTORY_SUMMARY_PART,
    INLINE_REASONING_PART,
    TEXT_PART,
)
from lite.core.tools.calls import (
    RuntimeEnvAction,
    make_tool_call,
    with_runtime_internal_stop_reason,
)

CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY = "_lite_content_only_final"
CONTENT_ONLY_FINAL_REASON = "content_only_final"
CONTENT_ONLY_FINAL_TEXT = "Done."
PARSE_FAILURE_FINAL_REASON = "parse_failure"
EMPTY_FINAL_REASON = "empty"
MODEL_OUTPUT_ERROR_KEY = "_lite_model_output_error"

# --- Episode stop-reason vocabulary --------------------------------------
#
# "Why did this episode end" is produced by the env, read by the agent loop and
# filtered into the durable record. This file owns the vocabulary: the single
# ``info`` key it is published under, and every value that is not derived from a
# finish tool name. Anything that needs one of these imports it from here — do
# not re-declare a value or a reason frozenset in a consumer package.
#
# The remaining values are finish tool names (``terminate``, ``response``,
# ``report_infeasible``, …), owned by ``core/tools/action_space`` and read off
# the accepted call.

#: The one ``LiteEnvStepResult.info`` key carrying the episode stop reason.
STOP_REASON_INFO_KEY = "stop_reason"

#: Injected by ``LoopDetectWrapper`` when the agent repeats a fingerprint.
LOOP_DETECT_TERMINATE_REASON = "REPETITIVE_LOOP"
#: An env terminating itself below the model tool surface.
ENV_INTERNAL_TERMINATE_REASON = "ENV_INTERNAL_TERMINATE"
#: Step-budget truncation.
MAX_STEPS_STOP_REASON = "max_steps"

#: Reasons stamped on transient finish calls created below the model surface.
#: A call carrying one of these is env/runtime-internal, never model output.
INTERNAL_STOP_REASONS = frozenset({
    LOOP_DETECT_TERMINATE_REASON,
    CONTENT_ONLY_FINAL_REASON,
    ENV_INTERNAL_TERMINATE_REASON,
})

# Which of these values reach the durable record is PUBLICATION policy, not
# message vocabulary. It lives with the durable containers in
# ``lite/core/samples.py`` (``PERSISTED_FINAL_STOP_REASONS``) and is
# deliberately kept out of the core message API.

#: Text-bearing assistant part types counted in the final-turn diagnostic,
#: in emitted key order. Which parts are counted is this module's policy;
#: the tags themselves are owned by ``lite.core.messages.content``.
_DIAGNOSTIC_PART_TYPES = (
    TEXT_PART,
    INLINE_REASONING_PART,
    ACTION_DESCRIPTION_PART,
    HISTORY_SUMMARY_PART,
)


@dataclass(frozen=True)
class NoToolCallFinalSummary:
    """Structural summary for a no-tool-call assistant final candidate."""

    visible_text: str
    content_types: tuple[str, ...]
    part_counts: Mapping[str, int]
    has_native_reasoning: bool
    has_raw_response: bool
    model_output_error: str | None

    def count(self, part_type: str) -> int:
        """Return how many content parts of ``part_type`` were seen."""
        return int(self.part_counts.get(part_type, 0))

    @property
    def has_model_output_error(self) -> bool:
        """Whether this no-tool final carries model-output-error feedback."""
        return bool(self.model_output_error)

    @property
    def stop_reason(self) -> str:
        """Why this turn carried no tool call, for internal diagnostics.

        The same answer the durable stop reason gives
        (:func:`no_tool_call_final_stop_reason`), read off the two facts this
        summary already carries. Which invisible channel a discarded turn used is
        in ``content_types`` / the part counts.
        """
        return no_tool_call_final_stop_reason(
            final_text=self.visible_text,
            model_output_error=self.model_output_error,
        )

    def diagnostic(self) -> dict[str, Any]:
        """Return the compact diagnostic sidecar for normalized no-tool finals."""
        out: dict[str, Any] = {
            "version": 1,
            "stop_reason": self.stop_reason,
            "content_types": list(self.content_types),
        }
        for part_type in _DIAGNOSTIC_PART_TYPES:
            out[f"{part_type}_part_count"] = self.count(part_type)
        out.update({
            "has_native_reasoning": self.has_native_reasoning,
            "has_raw_response": self.has_raw_response,
            "has_model_output_error": self.has_model_output_error,
            "visible_text": bool(self.visible_text),
        })
        return out


def no_tool_call_final_content() -> list[dict[str, str]]:
    """Return the canonical persisted no-tool-call final content."""
    return [{"type": TEXT_PART, "text": CONTENT_ONLY_FINAL_TEXT}]


def is_canonical_no_tool_call_final_message(message: dict[str, Any]) -> bool:
    """Whether *message* is the persisted structural no-tool-call final."""
    return (
        message.get("role") == ASSISTANT_ROLE
        and not message.get("tool_calls")
        and message.get("content") == no_tool_call_final_content()
    )


def canonicalize_no_tool_call_final_message(
    message: dict[str, Any],
) -> dict[str, Any]:
    """Return the canonical persisted form of a no-tool-call assistant final.

    Runtime may use the original visible prose as ``response(text=...)``. The
    stored Lite message stays structural: a single ``Done.`` text part plus a
    compact diagnostic, with no ``raw_response`` sidecar to replay later.
    """
    return {
        "role": ASSISTANT_ROLE,
        "content": no_tool_call_final_content(),
        CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY: summarize_no_tool_call_final(message).diagnostic(),
    }


def no_tool_call_final_text(message: dict[str, Any]) -> str:
    """Return the visible final text of a no-tool-call assistant turn.

    Only plain ``{"type": "text"}`` parts count. Internal channels such as
    ``action_description``, ``inline_reasoning``, ``history_summary`` and
    ``metadata`` are deliberately invisible to the final-text extractor.
    """
    return summarize_no_tool_call_final(message).visible_text


def summarize_no_tool_call_final(message: dict[str, Any]) -> NoToolCallFinalSummary:
    """Summarize final-turn content shape once for text extraction and diagnostics."""
    content = message.get("content")
    content_types: list[str] = []
    part_counts: dict[str, int] = {}
    visible_parts: list[str] = []

    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                part_type = str(part.get("type") or "<missing>")
                content_types.append(part_type)
                part_counts[part_type] = part_counts.get(part_type, 0) + 1
                if part_type == TEXT_PART and part.get("text"):
                    visible_parts.append(str(part["text"]).strip())
            else:
                content_types.append(type(part).__name__)
    elif content is not None:
        content_types.append(type(content).__name__)

    visible_text = "\n".join(part for part in visible_parts if part)
    return NoToolCallFinalSummary(
        visible_text=visible_text,
        content_types=tuple(content_types),
        part_counts=part_counts,
        has_native_reasoning=_has_native_reasoning(message),
        has_raw_response="raw_response" in message,
        model_output_error=message.get(MODEL_OUTPUT_ERROR_KEY) or None,
    )



def no_tool_call_final_stop_reason(
    *,
    final_text: str,
    model_output_error: str | None,
) -> str:
    """Durable stop reason for a zero-parsed-tool-call terminal turn."""
    if model_output_error:
        return PARSE_FAILURE_FINAL_REASON
    if final_text:
        return CONTENT_ONLY_FINAL_REASON
    return EMPTY_FINAL_REASON


def make_no_tool_call_final_actions(final_text: str) -> list[RuntimeEnvAction]:
    """Transient env action for a no-tool-call final turn.

    The action is not model output and is not persisted in the LiteSample. Its
    ``CONTENT_ONLY_FINAL_REASON`` sidecar is what marks it internal at env
    ingress, so the env scores the visible final text through the shared
    ``response`` shape while the stored assistant message stays structural.
    """
    action = make_tool_call("response", {"text": final_text})
    return [with_runtime_internal_stop_reason(action, CONTENT_ONLY_FINAL_REASON)]


def mark_model_output_error(message: dict[str, Any], error: str) -> None:
    """Attach a transient parser/runtime error to an adapter message."""
    message[MODEL_OUTPUT_ERROR_KEY] = error


def pop_model_output_error(message: dict[str, Any]) -> str | None:
    """Remove and return a transient parser/runtime error from a message."""
    error = message.pop(MODEL_OUTPUT_ERROR_KEY, None)
    return str(error) if error else None


def _has_native_reasoning(message: dict[str, Any]) -> bool:
    return bool(str(message.get("reasoning_content") or "").strip())


__all__ = [
    "CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY",
    "CONTENT_ONLY_FINAL_TEXT",
    "ENV_INTERNAL_TERMINATE_REASON",
    "INTERNAL_STOP_REASONS",
    "LOOP_DETECT_TERMINATE_REASON",
    "MAX_STEPS_STOP_REASON",
    "MODEL_OUTPUT_ERROR_KEY",
    "PARSE_FAILURE_FINAL_REASON",
    "STOP_REASON_INFO_KEY",
    "NoToolCallFinalSummary",
    "canonicalize_no_tool_call_final_message",
    "is_canonical_no_tool_call_final_message",
    "make_no_tool_call_final_actions",
    "mark_model_output_error",
    "no_tool_call_final_content",
    "no_tool_call_final_text",
    "pop_model_output_error",
    "summarize_no_tool_call_final",
]
