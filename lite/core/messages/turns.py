"""Message turn grouping.

This module owns the structural definition of a *turn*: a consecutive run of
observation messages (``role:"user"`` query/observation or ``role:"tool"``
tool-io result) followed by at most one ``role:"assistant"`` action.

Consumers must derive from :func:`group_into_turns` (or :func:`turn_spans`)
rather than re-implement the walk. Roles come from the vocabulary in
``lite.core.messages.content`` -- never from string literals.

Run (self-check):
    uv run pytest tests/core/messages/test_message_turns.py -q
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from lite.core.messages.content import (
    ASSISTANT_ROLE,
    TOOL_ROLE,
    USER_ROLE,
    LiteMessage,
    peel_system_message,
)

if TYPE_CHECKING:
    from lite.core.samples import LiteSample

#: Roles that carry observation input for a turn. Membership is the total
#: predicate: any object outside this set answers ``False``.
OBSERVATION_ROLES = frozenset({USER_ROLE, TOOL_ROLE})


@dataclass(frozen=True)
class TurnSpan:
    """Positional span of one turn inside a message list.

    ``observations`` holds the indices of the turn's consecutive observation
    messages (possibly empty for an assistant-only turn); ``assistant`` is the
    index of the turn's action message, or ``None`` for a predict-time partial
    turn whose action has not been produced yet.
    """

    observations: tuple[int, ...]
    assistant: int | None

    @property
    def start(self) -> int:
        """Index of the turn's first message (its first observation, or action)."""
        return self.observations[0] if self.observations else self.assistant


def turn_spans(messages: list[LiteMessage]) -> list[TurnSpan]:
    """Return the positional :class:`TurnSpan` of every turn in *messages*.

    Messages are expected to be system-peeled. Consecutive observation roles fold
    into one turn, an assistant action closes it, and a system message carries
    neither observation nor action.
    """
    spans: list[TurnSpan] = []
    observations: list[int] = []
    for i, message in enumerate(messages):
        role = message["role"]
        if role in OBSERVATION_ROLES:
            observations.append(i)
        elif role == ASSISTANT_ROLE:
            spans.append(TurnSpan(observations=tuple(observations), assistant=i))
            observations = []
        elif observations:  # SYSTEM_ROLE: ends the block, takes no action
            spans.append(TurnSpan(observations=tuple(observations), assistant=None))
            observations = []
    if observations:
        spans.append(TurnSpan(observations=tuple(observations), assistant=None))
    return spans


def group_into_turns(messages: list[LiteMessage]) -> list[dict[str, Any]]:
    """Group *messages* into observation→assistant turns.

    Args:
        messages: list of messages (excluding system)

    Returns:
        list of turn dicts with ``"observations"`` and ``"assistant"``.
        ``"observations"`` is the canonical observation block; callers that need
        task text should use ``lite.core.messages.instruction_text`` over that
        block instead of assuming the first observation is role:"user".
    """
    return [
        {
            "observations": [messages[i] for i in span.observations],
            "assistant": None if span.assistant is None else messages[span.assistant],
        }
        for span in turn_spans(messages)
    ]


def count_sample_turns(sample: LiteSample) -> int:
    """Return the number of observation→assistant turns in ``sample``."""
    _, content = peel_system_message(sample.messages)
    return len(turn_spans(content))


def observation_cut_index(messages: list[LiteMessage], k: int) -> int:
    """Index just past turn ``k``'s observation block (1-indexed ``k``).

    ``messages[:observation_cut_index(messages, k)]`` is turn ``k``'s
    *pre-action* context: turns ``1..k-1`` complete, plus turn ``k``'s WHOLE
    observation block, and nothing of turn ``k``'s action or later. A ``k``
    beyond the last turn keeps everything.
    """
    if k < 1:
        raise ValueError("turn index k must be >= 1")
    spans = turn_spans(messages)
    if k > len(spans):
        return len(messages)
    span = spans[k - 1]
    if span.observations:
        return span.observations[-1] + 1
    return span.assistant  # assistant-only turn: cut before its action


def truncate_to_observation(messages: list[LiteMessage], k: int) -> list[LiteMessage]:
    """Return turn ``k``'s pre-action message prefix (see :func:`observation_cut_index`)."""
    return messages[: observation_cut_index(messages, k)]


def turn_cut_index(messages: list[LiteMessage], k: int) -> int:
    """Index just past turn ``k`` (1-indexed), action INCLUDED.

    The complement of :func:`observation_cut_index`:
    ``messages[:turn_cut_index(messages, k)]`` is turns ``1..k`` complete. The
    cut lands on the first message of turn ``k+1``, so anything between the two
    turns (a leaked ``system`` message, which belongs to no turn) stays. A ``k``
    at or beyond the last turn keeps everything.
    """
    if k < 1:
        raise ValueError("turn index k must be >= 1")
    spans = turn_spans(messages)
    if k >= len(spans):
        return len(messages)
    return spans[k].start


def truncate_to_turn(messages: list[LiteMessage], k: int) -> list[LiteMessage]:
    """Return turns ``1..k`` in full (see :func:`turn_cut_index`)."""
    return messages[: turn_cut_index(messages, k)]


def truncate_sample_to_turn(sample: LiteSample, k: int) -> LiteSample:
    """Return a shallow copy of ``sample`` whose messages cover turns ``1..k``.

    Preserves a leading system message if present. This is the sample-level
    wrapper around :func:`truncate_to_turn`.
    """
    system_message, content = peel_system_message(sample.messages)
    truncated = truncate_to_turn(content, k)
    final = ([system_message] + truncated) if system_message else truncated
    return replace(sample, messages=final)


__all__ = [
    "count_sample_turns",
    "group_into_turns",
    "truncate_sample_to_turn",
    "truncate_to_observation",
]
