"""UI-TARS history protocol (``ui_tars.history``).

Also the base for ``core.ui_tars_15_v1`` (it
inherit the UI-TARS agent, adapter and this protocol) and for
``core.mai_ui.protocol.MAIUIHistoryProtocol`` (different structural
default but same windowing logic).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from lite.agents.core.protocol import TurnWindowProtocol
from lite.agents.core.protocol.window import (
    append_with_boundary_tool_projection,
    keep_images_and_tool_text,
)
from lite.core import (
    LiteMessage,
)
from lite.core.messages import USER_ROLE


@dataclasses.dataclass
class UITarsHistoryProtocol(TurnWindowProtocol, key="ui_tars.history"):
    """
    Protocol matching OSWorld's UITars history management.

    For old turns (beyond the window): keeps only assistant text (drops user/image messages).
    For recent ``full_history_size`` turns: keeps full user(image) + assistant pairs.
    The first user message (with instruction) is always preserved.

    This differs from :class:`Qwen3VLHistoryProtocol` in that it does NOT
    generate "Previous actions: Step 1: ..." summaries. Instead, old assistant
    responses are kept verbatim as consecutive assistant messages.

    Example with full_history_size=3, 6 total turns:
        Original: [u0+img, a0, u1+img, a1, u2+img, a2, u3+img, a3, u4+img, a4, u5+img, a5]
        Output:
          - user: instruction text (from u0, no image)
          - assistant: a0 (old, no preceding image)
          - assistant: a1 (old, no preceding image)
          - assistant: a2 (old, no preceding image)
          - user: img3
          - assistant: a3
          - user: img4
          - assistant: a4
          - user: img5
          - assistant: a5

    Attributes:
        full_history_size: Number of recent turns to keep with images (default: 5)
    """

    full_history_size: int = 5

    def _select_messages(
        self,
        content: list[LiteMessage],
        turns: list[dict[str, Any]],
    ) -> list[LiteMessage]:
        if len(turns) <= self.full_history_size:
            # All turns fit in the window, return as-is
            return content
        else:
            return self._apply_windowing(turns)

    def _apply_windowing(
        self,
        turns: list[dict[str, LiteMessage | None]],
    ) -> list[LiteMessage]:
        """Apply UITars-style windowing: old turns keep assistant-only, recent turns keep full."""
        num_turns = len(turns)
        window_start = num_turns - self.full_history_size

        result: list[LiteMessage] = []

        # First user message: instruction text only (strip images)
        first_user = next(
            (obs for obs in turns[0]["observations"] if obs.get("role") == USER_ROLE),
            None,
        )
        if first_user:
            first_user_stripped = copy.deepcopy(first_user)
            if "content" in first_user_stripped:
                first_user_stripped["content"] = [
                    item for item in first_user_stripped["content"]
                    if item.get("type") != "image"
                ]
            result.append(first_user_stripped)

        # Old turns (0 to window_start): assistant text only
        for i in range(num_turns):
            if i < window_start:
                # Old turn: only keep assistant response
                assistant_msg = turns[i].get("assistant")
                if assistant_msg:
                    result.append(assistant_msg)
            else:
                # Recent turn: keep full user(image) + assistant
                assistant_msg = turns[i].get("assistant")
                observations = [
                    keep_images_and_tool_text(user_msg)
                    for user_msg in turns[i]["observations"]
                ]

                # For windowed turns, keep images only (strip text)
                append_with_boundary_tool_projection(result, observations)
                if assistant_msg:
                    result.append(assistant_msg)

        return result
