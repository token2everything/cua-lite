"""Lite message protocol shapes and production-facing helpers.

The submodules own lower-level helpers. This package facade keeps only the
message shapes, shared vocabulary tags, and helpers imported by production
consumers as the public front door; tests that need owner internals import the
owning submodule directly.
"""

from lite.core.messages.content import (
    ACTION_DESCRIPTION_PART,
    ASSISTANT_ROLE,
    HISTORY_SUMMARY_PART,
    IMAGE_PART,
    INLINE_REASONING_PART,
    MESSAGE_ROLES,
    METADATA_PART,
    SYSTEM_ROLE,
    TEXT_PART,
    TOOL_ROLE,
    USER_ROLE,
    ActionDescriptionContent,
    HistorySummaryContent,
    ImageContent,
    InlineReasoningContent,
    LiteAssistantMessage,
    LiteMessage,
    LiteSystemMessage,
    LiteToolMessage,
    LiteUserMessage,
    MetadataContent,
    TextContent,
    extract_first_text,
    get_action_description,
    get_inline_reasoning,
    keep_model_visible_content,
    make_assistant_content,
    message_has_error_feedback,
    message_has_image,
    message_metadata,
    peel_system_message,
    set_or_append_text,
)
from lite.core.messages.final import (
    no_tool_call_final_text,
)
from lite.core.messages.selectors import (
    instruction_text,
)
from lite.core.messages.turns import (
    group_into_turns,
)

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
    "TextContent",
    "extract_first_text",
    "get_action_description",
    "get_inline_reasoning",
    "group_into_turns",
    "instruction_text",
    "keep_model_visible_content",
    "make_assistant_content",
    "message_has_error_feedback",
    "message_has_image",
    "message_metadata",
    "no_tool_call_final_text",
    "peel_system_message",
    "set_or_append_text",
]
