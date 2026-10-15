"""MAI-UI history protocol (``mai_ui.history``).

Same shape as :class:`UITarsHistoryProtocol` but always splits the first
user message into text-only and image-only bubbles (even when the trajectory
fits inside the window).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from lite.agents.core.protocol.window import (
    append_with_boundary_tool_projection,
    evicted_tool_result_summary_text,
    keep_images_and_tool_text,
)
from lite.agents.models.ui_tars.protocol import UITarsHistoryProtocol
from lite.core import (
    LiteMessage,
)
from lite.core.messages import USER_ROLE


@dataclasses.dataclass
class MAIUIHistoryProtocol(UITarsHistoryProtocol, key="mai_ui.history"):
    """
    History layout matching Tongyi-MAI MAI-UI's ``mai_naivigation_agent._build_messages``.

    Same shape as :class:`UITarsHistoryProtocol` (instruction text alone first,
    then assistant-only old turns, then user(image-only) + assistant pairs in
    the window) with one structural difference:

    UITars only splits the first user message's text from its image when
    windowing actually kicks in (num_turns > full_history_size). When the
    trajectory fits in the window, UITars returns the original messages
    unchanged, so the first user keeps its image and text in the same bubble.

    MAI-UI's reference ``_build_messages`` ALWAYS splits the first user
    message into separate text-only and image-only user bubbles, regardless
    of history length. We mirror that here by always going through the
    UITars windowing helper (with the same instruction-strip-images +
    iterate-turns logic), even on short trajectories.

    The default ``full_history_size=3`` matches MAI-UI's ``history_n=3``.

    Reference: ${CUA_LITE_REFERENCES_ROOT}/MAI-UI/src/mai_naivigation_agent.py:395

    Attributes:
        full_history_size: Number of recent turns kept with images (window).
            Default 3 matches MAI-UI's ``history_n=3``.
        summary_history_size: Optional cap on how many old assistant turns
            (turns BEFORE the image window) survive in the text history.
            Older assistants beyond this cap are dropped completely. The
            most recent ``summary_history_size`` old assistants are kept.
            Default ``None`` means unbounded (matches upstream MAI-UI:
            ``_build_messages`` keeps every historical assistant verbatim,
            which is fine for short tasks but inflates prompt length on
            deep multi-turn trajectories — at 30+ turns a single Sample's
            ``total_length`` can grow past 7000 tokens, and the per-Sample
            logits buffer at fwd time becomes the OOM trigger). Set this
            on RL configs where the policy may produce deep trajectories
            and you'd rather drop the oldest assistant text than blow the
            train-side activation budget.
    """

    full_history_size: int = 3   # MAI-UI default history_n=3
    summary_history_size: int | None = None  # None = unlimited (upstream MAI-UI behavior)

    def _select_messages(
        self,
        content: list[LiteMessage],
        turns: list[dict[str, Any]],
    ) -> list[LiteMessage]:
        if not turns:
            return content
        else:
            # Always go through windowing — even when num_turns <= full_history_size —
            # so the first user message's text and image are split into two
            # separate user bubbles. UITars's process_messages skips windowing
            # in the fits-in-window branch; we don't want that here.
            return self._apply_windowing(turns)

    def _apply_windowing(
        self,
        turns: list[dict[str, LiteMessage | None]],
    ) -> list[LiteMessage]:
        """Same as UITars windowing but caps old assistant turns by
        ``summary_history_size`` (None = unlimited).

        Layout (with full_history_size=W, summary_history_size=S, num_turns=N):

            [system]
            [first user: instruction text only]
            [assistant turns from max(0, N-W-S) to N-W-1]   ← S most recent old assistants
            [user(image) + assistant for turns N-W to N-1]   ← W image-window turns

        When ``summary_history_size`` is None (default), no cap is applied
        — all turns from 0 to N-W-1 contribute their assistant text, which
        matches the upstream MAI-UI behavior.
        """
        num_turns = len(turns)
        window_start = self._window_start_idx(turns)

        if self.summary_history_size is None:
            summary_start = 0
        else:
            summary_start = max(0, window_start - self.summary_history_size)

        result: list[LiteMessage] = []

        # First user message: instruction text only (strip images).
        # Kept regardless of summary_history_size — it carries the task
        # instruction, dropping it would lose the goal.
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

        for i in range(num_turns):
            if i < summary_start:
                # Dropped by summary_history_size cap.
                continue
            if i < window_start:
                # Old turn within summary cap: assistant text only.
                assistant_msg = turns[i].get("assistant")
                if assistant_msg:
                    result.append(assistant_msg)
                    self._append_old_tool_result_text(
                        result,
                        turns[i + 1] if i + 1 < window_start else None,
                    )
            else:
                # Image-window turn: keep user(image-only) + assistant.
                assistant_msg = turns[i].get("assistant")
                observations = [
                    keep_images_and_tool_text(user_msg)
                    for user_msg in turns[i]["observations"]
                ]
                append_with_boundary_tool_projection(result, observations)
                if assistant_msg:
                    result.append(assistant_msg)

        return result

    def _window_start_idx(self, turns: list[dict[str, LiteMessage | None]]) -> int:
        """Return the first kept turn index, budgeting by image-bearing turns.

        MAI-UI has a real MCP extra-tool prompt surface, so text-only tool
        turns must not spend the screenshot history budget.
        """
        if self.full_history_size <= 0:
            return len(turns)
        image_indices = self.image_turn_indices(turns)
        if len(image_indices) > self.full_history_size:
            return image_indices[-self.full_history_size]
        return 0

    def _append_old_tool_result_text(
        self,
        result: list[LiteMessage],
        result_turn: dict[str, LiteMessage | None] | None,
    ) -> None:
        result_text = self._tool_result_summary_text(result_turn)
        if result_text:
            result.append({
                "role": "user",
                "content": [{"type": "text", "text": result_text}],
            })

    def _tool_result_summary_text(
        self,
        result_turn: dict[str, LiteMessage | None] | None,
    ) -> str:
        """MAI-UI MCP/text-tool output that survives assistant-only summaries."""
        if result_turn is None:
            return ""
        return evicted_tool_result_summary_text(result_turn["observations"])
