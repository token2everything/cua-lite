"""Claude terminal feedback routing for the shared provider-native loop."""

from __future__ import annotations

from typing import Any

from lite.agents.models.claude.utils.history import append_canonical_step_feedback_messages
from lite.core import LiteSample
from lite.gym.types import LiteEnvStepResult


async def _append_claude_terminal_tool_feedback(
    *,
    trajectory: LiteSample,
    step_result: LiteEnvStepResult,
    tool_calls: list[dict[str, Any]],
) -> None:
    """Persist terminal Claude feedback without preparing another provider turn.

    ``step_result.results`` arrives already aligned to ``tool_calls`` from the
    shared loop's env-result boundary.
    """
    await append_canonical_step_feedback_messages(
        trajectory,
        step_result,
        tool_calls,
    )


__all__ = [
    "_append_claude_terminal_tool_feedback",
]
