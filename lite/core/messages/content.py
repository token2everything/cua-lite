"""Lite message and content-part shapes and provider-free content helpers."""

from __future__ import annotations

import copy
from typing import Any, Literal, NotRequired, TypedDict, get_args, get_type_hints

from lite.core.errors import LiteContractError
from lite.core.tools.calls import LiteToolCall
from lite.core.tools.results import text_has_projected_tool_result_error


class TextContent(TypedDict):
    """Plain text part. Used by all roles."""

    type: Literal["text"]
    text: str


class ImageContent(TypedDict):
    """Image referenced by index into ``LiteSample.images``."""

    type: Literal["image"]
    index: int


class MetadataContent(TypedDict):
    """Structured side-channel content, not directly rendered to the model."""

    type: Literal["metadata"]
    data: dict[str, Any]


class ActionDescriptionContent(TypedDict):
    """Natural-language description of an action being taken."""

    type: Literal["action_description"]
    text: str


class InlineReasoningContent(TypedDict):
    """Reasoning text embedded inside the content list."""

    type: Literal["inline_reasoning"]
    text: str


class HistorySummaryContent(TypedDict):
    """Cumulative trajectory-history summary."""

    type: Literal["history_summary"]
    text: str


class LiteSystemMessage(TypedDict):
    """System message: instructions / persona / tool list narrative."""

    role: Literal["system"]
    content: str | list[TextContent]


class LiteUserMessage(TypedDict):
    """User message: human turn or observation feedback."""

    role: Literal["user"]
    content: str | list[TextContent | ImageContent | MetadataContent]


class RawAssistantTextSidecar(TypedDict):
    """Opaque sidecar with the model's raw assistant text and producer key.

    Core treats this as an uninterpreted payload. Adapter replay and mutation
    policy belong to the agent layer.
    """

    text: str
    adapter_key: str


class LiteAssistantMessage(TypedDict):
    """Assistant message: model turn."""

    role: Literal["assistant"]
    content: NotRequired[list[
        TextContent
        | ActionDescriptionContent
        | InlineReasoningContent
        | HistorySummaryContent
        | MetadataContent
    ]]
    tool_calls: NotRequired[list[LiteToolCall]]
    reasoning_content: NotRequired[str]
    raw_response: NotRequired[RawAssistantTextSidecar]


class LiteToolMessage(TypedDict):
    """Tool-result message paired by ``tool_call_id``."""

    role: Literal["tool"]
    tool_call_id: str
    content: list[TextContent | ImageContent | MetadataContent]


LiteMessage = LiteSystemMessage | LiteUserMessage | LiteAssistantMessage | LiteToolMessage


# --- Canonical part-type and role vocabulary ---------------------------------
#
# The ``Literal[...]`` fields declared above ARE the vocabulary. The names below
# are *derived* from them, so a tag cannot be renamed on the TypedDict and go on
# living in a hand-written copy. This module is the single home: consumers
# import these names instead of re-listing tags. The boundary validators below
# consume the same derived sets, and the message-vocabulary tests pin that
# derivation from source.


def _literal_tag(shape: type, field: str) -> str:
    """Return the one tag a shape's ``Literal[...]`` field declares."""
    (tag,) = get_args(get_type_hints(shape)[field])
    return tag


#: ``content`` part-type tags, one per content part shape.
TEXT_PART = _literal_tag(TextContent, "type")
IMAGE_PART = _literal_tag(ImageContent, "type")
METADATA_PART = _literal_tag(MetadataContent, "type")
ACTION_DESCRIPTION_PART = _literal_tag(ActionDescriptionContent, "type")
INLINE_REASONING_PART = _literal_tag(InlineReasoningContent, "type")
HISTORY_SUMMARY_PART = _literal_tag(HistorySummaryContent, "type")

#: Every part type a Lite ``content`` list may carry.
CONTENT_PART_TYPES = frozenset({
    TEXT_PART,
    IMAGE_PART,
    METADATA_PART,
    ACTION_DESCRIPTION_PART,
    INLINE_REASONING_PART,
    HISTORY_SUMMARY_PART,
})

#: ``role`` tags, one per message shape.
SYSTEM_ROLE = _literal_tag(LiteSystemMessage, "role")
USER_ROLE = _literal_tag(LiteUserMessage, "role")
ASSISTANT_ROLE = _literal_tag(LiteAssistantMessage, "role")
TOOL_ROLE = _literal_tag(LiteToolMessage, "role")

#: Every role a Lite message may carry.
MESSAGE_ROLES = frozenset({SYSTEM_ROLE, USER_ROLE, ASSISTANT_ROLE, TOOL_ROLE})

#: Part types the model itself sees. Everything else is a cua-lite side channel.
MODEL_VISIBLE_PART_TYPES = frozenset({TEXT_PART, IMAGE_PART})

def _content_parts(message: LiteMessage) -> list[dict[str, Any]]:
    content = message.get("content")
    return content if isinstance(content, list) else []


def _is_model_visible_content(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    return part.get("type") in MODEL_VISIBLE_PART_TYPES


def keep_model_visible_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy with internal side-channel content parts removed.

    This structural boundary keeps canonical model-facing Lite content kinds
    (``text``/``image``) immediately before model serialization. Provider-native
    no-``type`` blocks are not Lite content and must be preserved, if needed, by
    the provider/template projection owner. Persistent Lite history remains
    untouched so env/protocol layers can still consume metadata, action
    descriptions, reasoning, and summaries.
    """
    out = copy.deepcopy(messages)
    for message in out:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                part for part in content if _is_model_visible_content(part)
            ]
    return out


def message_has_image(message: LiteMessage) -> bool:
    """Return whether ``message.content`` contains an image part."""
    return any(
        isinstance(item, dict) and item.get("type") == IMAGE_PART
        for item in _content_parts(message)
    )


def message_metadata(message: LiteMessage | None) -> dict[str, Any]:
    """Return this message's metadata payload, merged across parts, or ``{}``.

    The per-step observation metadata an env attaches to a turn (``page_title``,
    ``web_text``, ``is_error``, …) travels as a
    ``{"type": METADATA_PART, "data": {...}}`` content part. All metadata parts
    are merged in content order, with later keys winning. Empty and absent
    metadata both return ``{}`` so callers can use ``.get(key, default)``
    directly.
    """
    merged: dict[str, Any] = {}
    for item in _content_parts(message or {}):
        if isinstance(item, dict) and item.get("type") == METADATA_PART:
            data = item.get("data")
            if isinstance(data, dict):
                merged.update(data)
    return merged


def message_has_error_feedback(message: LiteMessage) -> bool:
    """Return whether content is marked as tool-result error feedback.

    Reads the model-visible text layer. The owner-rendered error header is the
    durable signal; metadata remains a side channel.
    """
    return any(
        isinstance(item, dict)
        and item.get("type") == TEXT_PART
        and isinstance(item.get("text"), str)
        and text_has_projected_tool_result_error(item["text"])
        for item in _content_parts(message)
    )


def tool_message_text_parts(message: LiteMessage) -> list[str]:
    """Text payloads from one per-call role:"tool" result message."""
    if message.get("role") != TOOL_ROLE:
        return []
    return [
        str(item["text"])
        for item in _content_parts(message)
        if (
            isinstance(item, dict)
            and item.get("type") == TEXT_PART
            and "text" in item
        )
    ]


def extract_first_text(message: LiteMessage | None) -> str:
    """Return the first ``{type: text}`` item from ``message.content``, or empty.

    Used to grab the instruction/goal from a turn's user message: the first
    text item in a user turn is conventionally the task description.
    Absent, empty, or malformed text all mean "this message has no instruction
    text".
    """
    if not message:
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    for item in _content_parts(message):
        if isinstance(item, dict) and item.get("type") == TEXT_PART:
            return item.get("text", "")
    return ""


def first_image_content_part(message: LiteMessage | None) -> dict | None:
    """Return a deep copy of the first ``{type: image}`` item, or ``None``."""
    if not message:
        return None
    for item in _content_parts(message):
        if isinstance(item, dict) and item.get("type") == IMAGE_PART:
            return copy.deepcopy(item)
    return None


def set_or_append_text(message: LiteMessage, text: str) -> None:
    """In-place: replace the first text item with ``text``, or append one if none exist.

    Used to swap a user message's instruction text while preserving its
    image/other content items.
    """
    content = message.get("content", [])
    # Non-list content is replaced with canonical text-list content.
    if not isinstance(content, list):
        message["content"] = [{"type": TEXT_PART, "text": text}]
        return
    for item in content:
        if isinstance(item, dict) and item.get("type") == TEXT_PART:
            item["text"] = text
            return
    content.append({"type": TEXT_PART, "text": text})
    message["content"] = content


def get_inline_reasoning(msg: dict[str, Any]) -> str:
    """Return prompted ``InlineReasoningContent`` text joined by newline."""
    parts: list[str] = []
    for part in msg.get("content") or []:
        if part["type"] == INLINE_REASONING_PART and part["text"]:
            parts.append(part["text"])
    return "\n".join(parts)


def make_assistant_content(
    *,
    text: str = "",
    inline_reasoning: str = "",
    action_description: str = "",
    history_summary: str = "",
) -> list[dict[str, Any]]:
    """Build an assistant ``content`` list in canonical part order."""
    parts: list[dict[str, Any]] = []
    if inline_reasoning:
        parts.append({"type": INLINE_REASONING_PART, "text": inline_reasoning})
    if action_description:
        parts.append({"type": ACTION_DESCRIPTION_PART, "text": action_description})
    if text:
        parts.append({"type": TEXT_PART, "text": text})
    if history_summary:
        parts.append({"type": HISTORY_SUMMARY_PART, "text": history_summary})
    return parts


def get_action_description(msg: dict[str, Any]) -> str:
    """Return the first non-empty action-description part text."""
    for part in msg.get("content") or []:
        if part["type"] == ACTION_DESCRIPTION_PART and part["text"]:
            return part["text"]
    return ""


def validate_message_roles(messages: Any, *, where: str = "messages") -> None:
    """Reject any message whose ``role`` is outside :data:`MESSAGE_ROLES`.

    Canonical row and trajectory boundaries validate the closed role vocabulary
    before downstream turn walkers consume messages.
    """
    if not isinstance(messages, list):
        raise LiteContractError(f"{where} must be a list")
    for index, message in enumerate(messages):
        role = message.get("role") if isinstance(message, dict) else None
        if role not in MESSAGE_ROLES:
            raise LiteContractError(
                f"{where}[{index}]: role {role!r} is outside the Lite role "
                f"vocabulary {sorted(MESSAGE_ROLES)}"
            )


def validate_message_content_parts(messages: Any, *, where: str = "messages") -> None:
    """Reject non-canonical content parts at the row/stage boundary.

    The content part vocabulary is closed for canonical Lite rows. Runtime
    builders may construct messages under their own tighter contracts; this
    boundary handles deserialized row data before downstream walkers consume it.
    """
    if not isinstance(messages, list):
        raise LiteContractError(f"{where} must be a list")
    role_part_types = {
        SYSTEM_ROLE: frozenset({TEXT_PART}),
        USER_ROLE: frozenset({TEXT_PART, IMAGE_PART, METADATA_PART}),
        ASSISTANT_ROLE: frozenset({
            TEXT_PART,
            ACTION_DESCRIPTION_PART,
            INLINE_REASONING_PART,
            HISTORY_SUMMARY_PART,
            METADATA_PART,
        }),
        TOOL_ROLE: frozenset({TEXT_PART, IMAGE_PART, METADATA_PART}),
    }
    text_part_types = frozenset({
        TEXT_PART,
        ACTION_DESCRIPTION_PART,
        INLINE_REASONING_PART,
        HISTORY_SUMMARY_PART,
    })
    for mi, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == TOOL_ROLE and "content" not in message:
            raise LiteContractError(f"{where}[{mi}]: role:tool content is required")
        if role == TOOL_ROLE and content == []:
            raise LiteContractError(f"{where}[{mi}]: role:tool content must be non-empty")
        if role in {SYSTEM_ROLE, USER_ROLE} and "content" in message:
            if not isinstance(content, (str, list)):
                raise LiteContractError(
                    f"{where}[{mi}]: role:{role} content must be a string or list"
                )
        if role in {ASSISTANT_ROLE, TOOL_ROLE} and "content" in message:
            if not isinstance(content, list):
                raise LiteContractError(
                    f"{where}[{mi}]: role:{role} content must be a list"
                )
        if not isinstance(content, list):
            continue
        for pi, part in enumerate(content):
            if not isinstance(part, dict):
                raise LiteContractError(
                    f"{where}[{mi}].content[{pi}]: content part must be a dict"
                )
            part_type = part.get("type")
            if part_type not in CONTENT_PART_TYPES:
                raise LiteContractError(
                    f"{where}[{mi}].content[{pi}]: content part type "
                    f"{part_type!r} is outside the Lite content part vocabulary "
                    f"{sorted(CONTENT_PART_TYPES)}"
                )
            allowed = role_part_types.get(role)
            if allowed is not None and part_type not in allowed:
                raise LiteContractError(
                    f"{where}[{mi}].content[{pi}]: role:{role} messages cannot "
                    f"carry {part_type!r} content parts"
                )
            if part_type in text_part_types and not isinstance(part.get("text"), str):
                raise LiteContractError(
                    f"{where}[{mi}].content[{pi}].text must be a string"
                )
            if part_type == METADATA_PART and not isinstance(part.get("data"), dict):
                raise LiteContractError(
                    f"{where}[{mi}].content[{pi}].data must be a dict"
                )


def require_message_list(messages: Any, *, where: str) -> None:
    """Reject a message window that is not a concrete list.

    ``[]`` is the one spelling of "no messages" that message-window and protocol
    boundaries accept. ``None`` is caller-code breakage, so every boundary states
    that here by name instead of letting it surface downstream as an incidental
    ``len(None)`` / "not iterable" ``TypeError`` — or, worse, as a second empty
    value travelling on unnoticed.
    """
    if not isinstance(messages, list):
        raise TypeError(
            f"{where} expects a list of messages, got {type(messages).__name__}"
        )


def peel_system_message(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Split messages into optional leading system message and conversation.

    ``messages`` must be a concrete list: an empty window is ``[]``, and ``None``
    is not a second spelling of it.
    """
    require_message_list(messages, where="peel_system_message")
    if len(messages) == 0:
        return (None, [])
    if messages[0].get("role") == SYSTEM_ROLE:
        return (messages[0], messages[1:])
    return (None, list(messages))


__all__ = [
    "ACTION_DESCRIPTION_PART",
    "ASSISTANT_ROLE",
    "HISTORY_SUMMARY_PART",
    "IMAGE_PART",
    "INLINE_REASONING_PART",
    "MESSAGE_ROLES",
    "METADATA_PART",
    "SYSTEM_ROLE",
    "TEXT_PART",
    "TOOL_ROLE",
    "USER_ROLE",
    "ActionDescriptionContent",
    "HistorySummaryContent",
    "ImageContent",
    "InlineReasoningContent",
    "LiteAssistantMessage",
    "LiteMessage",
    "LiteSystemMessage",
    "LiteToolMessage",
    "LiteUserMessage",
    "MetadataContent",
    "RawAssistantTextSidecar",
    "TextContent",
    "first_image_content_part",
    "extract_first_text",
    "get_action_description",
    "get_inline_reasoning",
    "keep_model_visible_content",
    "make_assistant_content",
    "message_has_error_feedback",
    "message_has_image",
    "message_metadata",
    "peel_system_message",
    "require_message_list",
    "tool_message_text_parts",
    "set_or_append_text",
    "validate_message_content_parts",
    "validate_message_roles",
]
