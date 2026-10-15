"""Stable cross-layer Lite protocol contracts."""

from lite.core.messages.content import (
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
)
from lite.core.metadata import (
    LiteBaseMetadata,
    LiteCUAMetadata,
    LiteGenericMetadata,
    metadata_from_dict,
)
from lite.core.tools.calls import LiteToolCall
from lite.core.tools.results import LiteToolResult
from lite.core.tools.schemas import LiteToolSchema

_SAMPLE_EXPORTS = frozenset({
    "LiteRLSample",
    "LiteRLStep",
    "LiteSample",
    "STATUS_ABORTED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_TRUNCATED",
    "STEP_STATUSES_BY_SEVERITY",
})


def __getattr__(name: str) -> object:
    """Load sample containers only when the core facade user asks for them."""
    if name in _SAMPLE_EXPORTS:
        from lite.core import samples

        value = getattr(samples, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionDescriptionContent",
    "HistorySummaryContent",
    "ImageContent",
    "InlineReasoningContent",
    "LiteAssistantMessage",
    "LiteBaseMetadata",
    "LiteCUAMetadata",
    "LiteGenericMetadata",
    "LiteMessage",
    "LiteRLSample",
    "LiteRLStep",
    "LiteSample",
    "LiteSystemMessage",
    "LiteToolCall",
    "LiteToolMessage",
    "LiteToolResult",
    "LiteToolSchema",
    "LiteUserMessage",
    "MetadataContent",
    "STATUS_ABORTED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_TRUNCATED",
    "STEP_STATUSES_BY_SEVERITY",
    "TextContent",
    "metadata_from_dict",
]
