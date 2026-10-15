"""GPT terminal feedback routing for the shared provider-native loop."""

from __future__ import annotations

from typing import Any

from lite.agents.core.agent.utils.tool_results import (
    append_tool_result_images,
    canonical_tool_result_messages,
)
from lite.core import LiteSample
from lite.gym.types import LiteEnvStepResult


async def _append_gpt_terminal_tool_feedback(
    *,
    trajectory: LiteSample,
    step_result: LiteEnvStepResult,
    tool_calls: list[dict[str, Any]],
) -> None:
    """Persist terminal GPT feedback without preparing another Responses turn.

    ``step_result.results`` arrives already aligned to ``tool_calls`` from the
    shared loop's env-result boundary.
    """
    stored_result_images = await append_tool_result_images(
        trajectory,
        step_result.results,
    )
    trajectory.messages.extend(
        canonical_tool_result_messages(
            step_result.results,
            tool_calls,
            result_image_indices_by_call_id=stored_result_images.by_call_id,
        )
    )


__all__ = [
    "_append_gpt_terminal_tool_feedback",
]
