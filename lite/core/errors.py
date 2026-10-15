"""Core exceptions for Lite contract violations.

This file owns errors raised when shared Lite data contracts are violated:
malformed messages, out-of-frame normalized coordinates, malformed tool schemas,
invalid task metadata fields, and unpairable tool-call envelopes. Runtime
failure categories belong to ``lite.gym.errors``; model-facing parse-error prose
belongs to the agent layer.

``TypeError`` remains the signal for caller-code type mistakes. These named
errors are for invalid Lite data crossing a core boundary.
"""

from __future__ import annotations


class LiteContractError(Exception):
    """A Lite protocol contract was violated by caller-supplied data.

    Callers should catch this named type when they handle malformed Lite data.
    Incidental ``ValueError`` exceptions remain separate caller-code mistakes.
    """


class ToolCallValidationError(LiteContractError):
    """Unpairable model-output error in a Lite tool-call envelope.

    Gym maps this exception type to its model-output failure category without
    importing gym vocabulary into core.
    """


__all__ = ["LiteContractError", "ToolCallValidationError"]
