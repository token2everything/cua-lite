"""Qwen3-VL history protocol.

:class:`Qwen3VLHistoryProtocol` (``qwen3_vl.history``) — Qwen-style rolling
summary: last ``full_history_size`` turns keep images, older turns collapse
into a text summary injected into the first in-window user bubble. Default
for all Qwen3-VL adapters (desktop / browser / mobile) and EvoCUA, which shares
the Qwen3-VL dialect.

By default, windowed image observations preserve adjacent text
(``keep_text_with_images=True``), so image+text env feedback and projected
tool-result errors follow the same model-visible path. Set
``keep_text_with_images=False`` only for configs that intentionally reproduce an
older pure-GUI screenshot-only reference.

The mobile adapter also uses this canonical protocol. The cookbook-specific
single-bubble mobile layout is intentionally not registered.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from typing import Any

from lite.agents.core.protocol import TurnWindowProtocol
from lite.agents.core.protocol.window import (
    append_with_boundary_tool_projection,
    evicted_tool_result_summary_text,
    filter_history_content,
)
from lite.core import (
    LiteMessage,
)

# =============================================================================
# Qwen3-VL History Protocol (Qwen-style summarized rolling window)
# =============================================================================

@dataclasses.dataclass
class Qwen3VLHistoryProtocol(TurnWindowProtocol, key="qwen3_vl.history"):
    """
    Qwen-style summarized history: rolling image window + text summary of
    older turns.

    - Keep the last ``full_history_size`` turns (each turn = user image + assistant response)
    - All earlier turns are summarized as "Previous actions: Step 1: ..."
    - The summary + original instruction is placed in the FIRST user message of the window
    - Total images in output = ``full_history_size`` (one per kept turn)

    Example with full_history_size=2, total 5 turns:
        Original: [turn0, turn1, turn2, turn3, turn4]
        Output:
          - user: img3 + "Previous actions: Step 1-3...\\n\\nInstruction: ..."
          - assistant: resp3
          - user: img4 (image only)
          - assistant: resp4 (if complete; omitted during inference)

    Attributes:
        full_history_size: Number of recent turns to keep in full (default: 4)
        summary_history_size: Maximum number of action descriptions to include in
            the summarized text history. Older actions beyond this cap are
            dropped from the summary (the most recent ``summary_history_size``
            actions before the image window are kept). Default 100, which is
            effectively "unlimited" for most trajectories — set lower for
            very long trajectories where the text summary would otherwise
            exceed the model's effective context.
        keep_text_with_images: If True, keep ordinary user text in windowed
            messages that also carry images. Text-only messages and role:"tool"
            result text are always preserved; this flag only controls ordinary
            image+text history compression. Set False only for explicit legacy
            pure-GUI reproduction configs.
        summary_template: Template for generating the summary
        action_format_func: Optional function to format actions for summary.
            Receives an assistant ``LiteMessage`` and returns a string
            description of the action. If None, uses the default
            ``_format_action`` method.
    """

    full_history_size: int = 4
    summary_history_size: int = 100
    keep_text_with_images: bool = True
    summary_template: str = (
        "Please generate the next move according to the UI screenshot, "
        "instruction and previous actions.\n\n"
        "Instruction: {instruction}\n\n"
        "Previous actions:\n{previous_actions_str}"
    )
    action_format_func: Callable[[LiteMessage], str] | None = None

    def _select_messages(
        self,
        content: list[LiteMessage],
        turns: list[dict[str, Any]],
    ) -> list[LiteMessage]:
        """Apply Qwen-style summarized history.

        The surrounding peel/group/re-attach frame lives on
        :class:`TurnWindowProtocol`; this hook owns steps 1-2 below.

        1. Compute ``(kept_turns, summary_text, instruction)`` — which turns
           survive in the image window, and what the dropped older turns
           collapse into as a text summary.
        2. Hand off to :meth:`_render_user_messages` to build the final
           user/assistant message list. **Subclasses that need a different
           output structure (e.g. compress everything into a single user
           message with image-in-middle) override only this hook.**

        Args:
            content: system-stripped messages.
            turns: ``content`` grouped into observation/assistant turns.

        Returns:
            The non-system messages to keep, with summarized history.
        """
        if not turns:
            return list(content)
        else:
            kept_turns, summary_text, instruction = self._compute_summary_and_window(turns)
            return self._render_user_messages(kept_turns, summary_text, instruction)

    # -------------------------------------------------------------------------
    # Helpers (called by process_messages, overridable by subclasses)
    # -------------------------------------------------------------------------

    def _compute_summary_and_window(
        self,
        turns: list[dict[str, LiteMessage | None]],
    ) -> tuple[list[dict[str, LiteMessage | None]], str, str]:
        """Compute the image-window split + text summary.

        Returns:
            kept_turns:    the most recent ``full_history_size`` turns
                           (these still have their image bubbles).
            summary_text:  ``"\\n"``-joined ``Step N: <action>`` lines for
                           the most recent ``summary_history_size`` actions
                           BEFORE the image window. ``"None"`` if empty.
            instruction:   the goal text from the first turn.
        """
        # The window budgets IMAGES, so it is spent by image-bearing turns only.
        # A text-only turn (a ``bash`` result today) contributes no image and
        # therefore must not push a screenshot out of the window.
        #
        # GUI-only identity: every turn carries an image, so ``image_indices`` is
        # ``range(len(turns))`` and the window start below is
        # ``max(0, len(turns) - full_history_size)``.
        #
        # Counted FORWARD (``n_images - full_history_size``) off a list with a
        # past-the-end sentinel. The equivalent-looking
        # ``image_indices[-full_history_size]`` differs at ``full_history_size ==
        # 0``: it wraps to ``image_indices[0]`` and keeps the WHOLE trajectory, the
        # inverse of the "keep none" every sibling protocol means by 0.
        image_indices = self.image_turn_indices(turns)
        n_images = len(image_indices)
        window_bounds = [*image_indices, len(turns)]
        window_start_idx = (
            window_bounds[n_images - self.full_history_size]
            if n_images > self.full_history_size
            else 0
        )

        instruction = self._extract_instruction(turns[0])

        # Drop oldest beyond summary_history_size; absolute step numbering.
        text_start_idx = max(0, window_start_idx - self.summary_history_size)
        summary_lines = []
        for i in range(text_start_idx, window_start_idx):
            turn = turns[i]
            if turn["assistant"]:
                line = self._format_action(turn["assistant"], i + 1)
                # A text tool's RESULT lands in the NEXT turn's observations
                # (observations precede the assistant that consumes them). It is
                # one-shot information -- no later screenshot re-reveals it -- so
                # carry it into the summary instead of dropping it with the
                # window. Empty for an image-bearing turn, so a GUI-only
                # trajectory renders exactly as before.
                result_text = self._tool_result_summary_text(
                    turns[i + 1] if i + 1 < len(turns) else None
                )
                if result_text:
                    line = f"{line}\n{result_text}"
                summary_lines.append(line)
        summary_text = "\n".join(summary_lines) if summary_lines else "None"

        kept_turns = turns[window_start_idx:]
        return kept_turns, summary_text, instruction

    def _render_user_messages(
        self,
        kept_turns: list[dict[str, LiteMessage | None]],
        summary_text: str,
        instruction: str,
    ) -> list[LiteMessage]:
        """Hook: render kept_turns + summary into the final message list.

        Default: produces a multi user/assistant sequence (Qwen-style):

        - The FIRST kept user message gets the ``summary_template`` injected
          (instruction + summary text).
        - Subsequent user messages keep only their image (and optionally text
          if ``keep_text_with_images=True``).
        - Each kept assistant message is appended as-is.

        Subclasses can override to produce a different structure; the
        registered ones (``lite.agents.extensions``) override rendering only.
        """
        full_prompt = self.summary_template.format(
            instruction=instruction,
            previous_actions_str=summary_text,
        )

        result: list[LiteMessage] = []
        for i, turn in enumerate(kept_turns):
            assistant_msg = turn["assistant"]
            observations: list[LiteMessage] = []

            for j, user_msg in enumerate(turn["observations"]):
                if i == 0 and j == 0:
                    user_msg = copy.deepcopy(user_msg)
                    self._inject_text(user_msg, full_prompt)
                else:
                    user_msg = filter_history_content(
                        user_msg,
                        strip_regular_image_text=not self.keep_text_with_images,
                    )
                observations.append(user_msg)

            append_with_boundary_tool_projection(result, observations)
            if assistant_msg:
                result.append(assistant_msg)

        return result

    def _tool_result_summary_text(
        self,
        turn: dict[str, LiteMessage | None] | None,
    ) -> str:
        """Text-tool output that survives when the result turn leaves the window.

        This is Qwen history policy, not a base-protocol rule: text-only
        role:"tool" results are one-shot outputs and image-bearing errors are
        correction feedback marked by structured metadata or cua-lite's rendered
        error header. Ordinary image+text observation prose is controlled by
        ``keep_text_with_images`` in the visible window and is not copied into
        old summaries.
        """
        if turn is None:
            return ""
        return evicted_tool_result_summary_text(turn["observations"])
