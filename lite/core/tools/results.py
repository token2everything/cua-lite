"""How a Lite tool RETURNS: the per-call result carrier and text projection.

The projection renders a native ``(text, error)`` pair into the single
model-visible string; the two readers below parse that same rendering back out.
They are a parser of this file's own output, so they live beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LiteToolResult:
    """Per-call env result paired to a canonical ``LiteToolCall``.

    ``tool_call_id`` is the pairing key when the result answers an assistant
    tool call. ``None`` is reserved for unpaired env feedback that renders as a
    user observation, never as a role:tool row. There is deliberately no category
    field: image presence is determined by how the env produced the result.
    ``text`` is the observation/output channel. ``error`` is a separate
    correction channel, so a failed tool call can still carry observation text
    and images back to the model. ``images`` may contain any env-captured frames
    for this call, including multiple captures from one action-batch call;
    whether a producer captures only the final frame or additional in-turn
    frames is an env/runtime policy.
    """

    tool_call_id: str | None
    images: list[bytes] = field(default_factory=list)
    text: str | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None


def make_tool_result(
    tool_call_id: str | None,
    *,
    images: list[bytes] | None = None,
    text: str | None = None,
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
) -> LiteToolResult:
    """Build a result and apply the standard error metadata convention."""
    if error is None:
        result_metadata = metadata
    else:
        result_metadata = dict(metadata or {})
        result_metadata["is_error"] = True
    return LiteToolResult(
        tool_call_id=tool_call_id,
        images=list(images or []),
        text=text,
        metadata=result_metadata,
        error=error,
    )


TOOL_RESULT_ERROR_SECTION_HEADER = "## Error from previous action:"


def project_tool_result_text(text: str | None, error: str | None) -> str | None:
    """Project a native ``(text, error)`` result into model-visible text."""
    if error is None:
        return text

    error_section = f"{TOOL_RESULT_ERROR_SECTION_HEADER}\n{error}"
    if text is None:
        return error_section
    return f"{text}\n\n{error_section}"


def text_has_projected_tool_result_error(text: str) -> bool:
    """Return whether ``text`` contains an exact projected error section."""
    return extract_projected_tool_result_error(text) is not None


def extract_projected_tool_result_error(text: str | None) -> str | None:
    """Extract projected tool-result error text from model-facing content.

    Accepts exactly the two placements produced by
    :func:`project_tool_result_text`: error-only text starting with the header,
    or observation text followed by a blank line and the header.
    """
    if not text:
        return None
    if text.startswith(f"{TOOL_RESULT_ERROR_SECTION_HEADER}\n"):
        header_idx = 0
    else:
        idx = text.find(f"\n\n{TOOL_RESULT_ERROR_SECTION_HEADER}\n")
        if idx < 0:
            return None
        header_idx = idx + 2
    after = text[header_idx + len(TOOL_RESULT_ERROR_SECTION_HEADER):]
    return after.lstrip("\n")


__all__ = [
    "LiteToolResult",
    "TOOL_RESULT_ERROR_SECTION_HEADER",
    "extract_projected_tool_result_error",
    "make_tool_result",
    "project_tool_result_text",
    "text_has_projected_tool_result_error",
]
