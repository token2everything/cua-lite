"""Pure GPT provider-output conversion helpers.

Kept separate from response/image helpers so provenance code can parse/merge tool
calls without importing HTTP/image dependencies.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from lite.core.tools.action_space import merge_adjacent_lite_action_batches

if TYPE_CHECKING:
    from lite.agents.models.gpt.action_space import (
        GPTDesktopActionSpace,
        GPTMobileActionSpace,
    )


def parse_provider_function_args(item: dict[str, Any]) -> dict[str, Any] | None:
    """A function-call item's arguments as a dict, or ``None`` when malformed."""
    raw = item.get("arguments")
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return raw if isinstance(raw, dict) else None


def lite_calls_from_gpt_computer_actions(
    action_space: GPTDesktopActionSpace,
    provider_actions: list[dict[str, Any]],
    resolution: tuple[int, int],
) -> list[dict[str, Any]]:
    """Convert GPT native computer actions into canonical desktop action-batch calls."""
    lite_tool_calls: list[dict[str, Any]] = []
    for provider_action in provider_actions:
        lite_tool_calls.extend(
            action_space.convert_tool_calls_from_agent(
                [provider_action],
                resolution=resolution,
            )
        )
    return merge_adjacent_lite_action_batches(lite_tool_calls)


def lite_calls_from_gpt_mobile_function_calls(
    action_space: GPTMobileActionSpace,
    provider_function_calls: list[dict[str, Any]],
    resolution: tuple[int, int],
) -> list[dict[str, Any]]:
    """Convert GPT mobile function calls into canonical mobile action-batch calls."""
    return action_space.convert_tool_calls_from_agent(
        provider_function_calls,
        resolution=resolution,
    )
