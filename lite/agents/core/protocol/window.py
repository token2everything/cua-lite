"""Protocol-owned history-window content policy."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

from lite.core import messages as _core_messages
from lite.core.messages import (
    ASSISTANT_ROLE,
    IMAGE_PART,
    METADATA_PART,
    TEXT_PART,
    TOOL_ROLE,
    USER_ROLE,
)

# ``tool_message_text_parts`` is a content-owner helper rather than a
# package-level front-door contract, so it is reached through its owning module.
from lite.core.messages import content as _core_content


def filter_history_content(
    message: dict[str, Any],
    *,
    strip_regular_image_text: bool = False,
) -> dict[str, Any]:
    """Filter message content under the shared history-window policy.

    Pure text messages are always preserved. By default, ordinary text alongside
    images is preserved too. Legacy pure-GUI reproduction paths can explicitly
    strip ordinary image-adjacent text, while role:"tool" text stays because it
    is a per-call result/error payload rather than optional page prose.
    Metadata is retained for every role: it is never optional page prose.
    """
    if not strip_regular_image_text or not _core_messages.message_has_image(message):
        return keep_images_and_text(message)
    return keep_images_and_tool_text(message)


def keep_images_and_tool_text(message: dict[str, Any]) -> dict[str, Any]:
    """Keep images and metadata always, plus text for per-call role:"tool" results.

    Dropping non-image/metadata parts is correct for ordinary screenshot
    observations when a protocol intentionally suppresses page/AXTree text. It
    is wrong for role:"tool" text: that text is the per-call result/error
    payload, not ambient observation text, and dropping it blinds the next
    decision — hence the ``role`` check below. Metadata is kept for every role.
    """
    message = copy.deepcopy(message)
    content = message.get("content")
    if isinstance(content, list):
        keep_tool_text = message.get("role") == TOOL_ROLE
        message["content"] = [
            item
            for item in content
            if isinstance(item, dict)
            and (
                item.get("type") == IMAGE_PART
                or item.get("type") == METADATA_PART
                or (
                    keep_tool_text
                    and item.get("type") == TEXT_PART
                    and "text" in item
                )
            )
        ]
    return message


def keep_images_and_text(message: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy with image, text, and metadata items kept.

    Ordinary user observations keep only non-empty text. Per-call role:"tool"
    results keep text by key presence, because an empty string can be tool
    output.
    """
    message = copy.deepcopy(message)
    content = message.get("content")
    if isinstance(content, list):
        keep_empty_text = message.get("role") == TOOL_ROLE
        message["content"] = [
            item
            for item in content
            if isinstance(item, dict)
            and (
                item.get("type") == IMAGE_PART
                or (
                    item.get("type") == TEXT_PART
                    and ("text" in item if keep_empty_text else item.get("text"))
                )
                or item.get("type") == METADATA_PART
            )
        ]
    return message


def collapse_image_observation(
    message: dict[str, Any],
    collapse_text: str,
    *,
    strip_regular_image_text: bool,
) -> dict[str, Any]:
    """Replace image parts with ``collapse_text`` under window-fold policy."""
    message = copy.deepcopy(message)
    content = message.get("content") or []
    collapsed: list[dict[str, Any]] = [{"type": TEXT_PART, "text": collapse_text}]
    if message.get("role") == TOOL_ROLE:
        collapsed.extend(
            item for item in content
            if isinstance(item, dict)
            and (
                (item.get("type") == TEXT_PART and "text" in item)
                or item.get("type") == METADATA_PART
            )
        )
    elif not strip_regular_image_text:
        collapsed.extend(
            item for item in content
            if isinstance(item, dict)
            and (
                (item.get("type") == TEXT_PART and item.get("text"))
                or item.get("type") == METADATA_PART
            )
        )
    else:
        collapsed.extend(
            item for item in content
            if isinstance(item, dict) and item.get("type") == METADATA_PART
        )
    message["content"] = collapsed
    return message


def join_role_tool_text(
    messages: Iterable[dict[str, Any]],
    *,
    sep: str = "\n\n",
) -> str | None:
    """Join text payloads from role:"tool" result messages.

    Some compact adapter protocols render one current user prompt from the last
    observation block. They still need per-call tool result/error text from
    role:"tool" messages, but should not reopen ordinary user observation text.
    """
    parts: list[str] = []
    for message in messages:
        if message.get("role") != TOOL_ROLE:
            continue
        parts.extend(_core_content.tool_message_text_parts(message))
    return sep.join(parts) if parts else None


def inject_summary_text(message: dict[str, Any], text: str) -> None:
    """Inject a history summary without erasing role:"tool" result text.

    Summary prompts replace ordinary user observation text. A role:"tool"
    message's text is a result/error carrier, so the summary is inserted as a
    separate leading text part instead of overwriting the existing payload.
    """
    if message.get("role") == TOOL_ROLE:
        content = message.get("content", [])
        if isinstance(content, str):
            content = [{"type": TEXT_PART, "text": content}]
        elif not isinstance(content, list):
            content = []
        content.insert(0, {"type": TEXT_PART, "text": text})
        message["content"] = content
        return
    _core_messages.set_or_append_text(message, text)


def append_with_boundary_tool_projection(
    result: list[dict[str, Any]],
    observations: Iterable[dict[str, Any]],
) -> None:
    """Append observations, projecting only window-leading role:"tool" results.

    A tool result is projected only while the next appended observation would be
    at a model-visible boundary with no immediately preceding assistant call in
    the rendered prompt. Once a non-projected observation is appended, later
    role:"tool" messages remain structural.
    """
    project_leading_tools = not result or result[-1].get("role") != ASSISTANT_ROLE
    for message in observations:
        if project_leading_tools and message.get("role") == TOOL_ROLE:
            projected = copy.deepcopy(message)
            projected["role"] = USER_ROLE
            projected.pop("tool_call_id", None)
            result.append(projected)
            continue
        result.append(message)
        project_leading_tools = False


def evicted_tool_result_summary_text(
    observations: Iterable[dict[str, Any]],
) -> str:
    """Return role:"tool" text that should survive an evicted observation block.

    Text-only tool output is one-shot state, so history summaries keep it
    verbatim. Image-bearing tool results are copied only when marked as error
    feedback; ordinary screenshots are represented by the surrounding protocol's
    visual/window policy instead.
    """
    parts: list[str] = []
    for observation in observations:
        if observation.get("role") != TOOL_ROLE:
            continue
        if (
            _core_messages.message_has_image(observation)
            and not _core_messages.message_has_error_feedback(observation)
        ):
            continue
        parts.extend(_core_content.tool_message_text_parts(observation))
    return "\n".join(parts)


__all__ = [
    "append_with_boundary_tool_projection",
    "collapse_image_observation",
    "evicted_tool_result_summary_text",
    "filter_history_content",
    "inject_summary_text",
    "join_role_tool_text",
    "keep_images_and_tool_text",
]
