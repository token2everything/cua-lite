"""Fara-1.0 history protocol.

:class:`FaraHistoryProtocol` (``fara.history``) — full text history with a
sliding image window: **all** user/assistant turns are kept verbatim (text,
tool_calls, observations), but only the newest ``max_n_images`` screenshots
survive; older image parts are stripped from their messages (text preserved).

This mirrors the reference ``FaraAgent.maybe_remove_old_screenshots``
(``fara/src/fara/fara_agent.py``): it walks the chat history newest-first,
keeps images while the running count is below ``max_n_images``, and drops the
image (but never the text) once the budget is spent. The first turn (the task
instruction) always keeps its text; only its screenshot may be dropped.

Unlike :class:`Qwen3VLHistoryProtocol`, older turns are NOT collapsed into a
rolling text summary — Fara feeds the model every prior thought / action /
observation as-is, only trimming screenshots.
"""

from __future__ import annotations

import copy
import dataclasses

from lite.agents.core.protocol import BaseProtocol
from lite.core import (
    LiteMessage,
)
from lite.core.messages import peel_system_message


@dataclasses.dataclass
class FaraHistoryProtocol(BaseProtocol, key="fara.history"):
    """Full text history + newest-``max_n_images`` screenshot window.

    Attributes:
        max_n_images: Number of most-recent screenshots to keep. Images in
            older messages are stripped (text kept). ``<= 0`` disables the
            window (all images kept), matching the reference's early return.
            Default 3 (the reference ``FaraAgent`` default).
    """

    max_n_images: int = 3

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs,
    ) -> list[LiteMessage]:
        """Keep all messages; strip image parts beyond the newest ``max_n_images``.

        Args:
            messages: list of cua-lite messages.

        Returns:
            Deep-copied messages with older screenshots removed.
        """
        if messages is None or len(messages) == 0:
            return []

        messages = copy.deepcopy(messages)
        if self.max_n_images <= 0:
            return messages

        system_message, content = peel_system_message(messages)

        # Walk newest-first, counting screenshots. Once the budget is spent,
        # strip image parts from older messages (text and everything else stay).
        n_images = 0
        for msg in reversed(content):
            parts = msg.get("content")
            if not isinstance(parts, list):
                continue
            img_parts = [p for p in parts if isinstance(p, dict) and p.get("type") == "image"]
            if not img_parts:
                continue
            if n_images < self.max_n_images:
                # The reference checks ``n_images < max_n_images`` BEFORE adding,
                # so a message is kept whole when the running count is still
                # under budget (even if its own images push past it).
                n_images += len(img_parts)
            else:
                msg["content"] = [
                    p for p in parts
                    if not (isinstance(p, dict) and p.get("type") == "image")
                ]

        result = ([system_message] + content) if system_message else content
        return result
