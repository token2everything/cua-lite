"""Lite message assembly for agent sampling loops."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lite.core.messages.content import (
    IMAGE_PART,
    METADATA_PART,
    TEXT_PART,
    TOOL_ROLE,
    USER_ROLE,
)
from lite.core.tools.results import project_tool_result_text


def build_initial_user_message(
    instruction: str,
    *,
    has_image: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn-0 user message: optional image (index 0) + instruction text."""
    content: list[dict[str, Any]] = []
    if has_image:
        content.append({"type": IMAGE_PART, "index": 0})
    content.append({"type": TEXT_PART, "text": instruction})
    if metadata:
        content.append({"type": METADATA_PART, "data": metadata})
    return {"role": USER_ROLE, "content": content}


def build_tool_result_message(
    call_id: str | None,
    image_indices: Sequence[int] | None,
    text: str | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    error: str | None = None,
) -> dict[str, Any] | None:
    """Turn-N per-call env feedback.

    ``call_id=None`` is reserved for unpaired env feedback that is not
    answering an assistant tool call, so it remains a user observation.
    Empty-string text is preserved because it is a real tool result payload.
    """
    content: list[dict[str, Any]] = []
    for image_index in image_indices or ():
        content.append({"type": IMAGE_PART, "index": image_index})
    projected_text = project_tool_result_text(text, error)
    if projected_text is not None:
        content.append({"type": TEXT_PART, "text": projected_text})
    if metadata:
        content.append({"type": METADATA_PART, "data": metadata})
    if not content:
        return None
    if call_id is None:
        return {"role": USER_ROLE, "content": content}
    return {"role": TOOL_ROLE, "tool_call_id": call_id, "content": content}


__all__ = [
    "build_initial_user_message",
    "build_tool_result_message",
]
