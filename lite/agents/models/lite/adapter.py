"""
Canonical Lite Adapter (Pass-through with History Management)

This is the canonical replay/export adapter, not a model family: tool calls
are already in Lite shape, so these adapters keep CUA-Lite format unchanged
and only apply history management protocols for multi-turn data.

    LiteBaseAdapter (shared logic)
    ├── LiteDesktopGroundingActionAdapter  (full history; SFT replay; desktop+browser)
    ├── LiteDesktopUseAdapter       (full history; desktop+browser)
    ├── LiteMobileGroundingActionAdapter   (mobile schema; SFT replay)
    └── LiteMobileUseAdapter        (full history; mobile ``use``)

The ``Desktop`` adapters register under the ``r"lite@(desktop|browser)@..."``
regex, so the same class is reachable via both ``lite@desktop@...`` and
``lite@browser@...`` keys.

``understanding`` / ``grounding.bbox`` / ``grounding.point`` route through
:class:`AsIsAdapter` (lite is canonical so tool calls already in lite shape).

Usage:
    from lite.agents.models.lite.adapter import (
        LiteDesktopGroundingActionAdapter,
        LiteDesktopUseAdapter,
    )
    adapter = LiteDesktopUseAdapter()

See ``lite/agents/models/README.md`` for model-family usage notes.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import re
from typing import Any

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.action_space.base import LiteDesktopActionSpace, LiteMobileActionSpace
from lite.agents.core.adapter import BaseAgentAdapter
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.types import AgentMessage, AgentStep
from lite.core import (
    LiteMessage,
    LiteSample,
)
from lite.core.messages.turns import truncate_sample_to_turn

logger = logging.getLogger(__name__)

# =============================================================================
# System Prompts
# =============================================================================

USE_SYSTEM_PROMPT = """\
Perform actions on the desktop interface. Coordinates are normalized to the range [0, 1000].

# Response format

Response format for every step:
1) Action: A short imperative describing what to do in the UI.
2) Tool_call: A single tool_call block.

Rules:
- Output exactly in the order: action, tool_call.
- Be brief: one sentence for action.
- Do not output anything else outside those parts.
- If finishing, use action=response for a question-answering task, otherwise use action=terminate in the tool call."""

# =============================================================================
# CUA-Lite Base Adapter
# =============================================================================

@dataclasses.dataclass
class LiteBaseAdapter(BaseAgentAdapter):
    """
    Base adapter for CUA-lite format (pass-through with protocol support).
    
    This adapter keeps the CUA-lite format unchanged, only applying
    the configured protocol for history management.
    
    Subclasses must set action_space and protocol.
    """
    
    system_prompt: str | None = None

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render turn ``k``: protocol-process truncated history + system prompt.

        ``processed`` is the trajectory-level processed-image list (unused
        here because messages keep ImageContent indices; consumers
        recover images via ``referenced_images_in_message_order``).
        """
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        result_messages: list[AgentMessage] = []
        if self.system_prompt:
            result_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            })
        # LiteBaseAdapter is a SFT-data adapter — keep messages structural.
        # Per-message canonical conversion still goes through the public
        # convert_message_to_agent so raw_response sidecars are honored.
        for m in messages:
            result_messages.append(self.convert_message_to_agent(m))
        return result_messages

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> LiteMessage:
        """Convert LiteAssistantMessage → agent-side text. ``USE_SYSTEM_PROMPT``
        expects ``Action: <description>`` followed by ``<tool_call>{json}</tool_call>``.
        Whitelist-pick the ``action_description`` content parts and render them
        as ``"Action: <text>"``; drop every other kind (``inline_reasoning``,
        ``history_summary``, extra text) since the SFT distribution has no slot
        for them. NATIVE top-level ``reasoning_content`` (Qwen-family
        ``<think>`` channel) is preserved — the chat_template's Thinking-mode
        branch renders it directly.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result
        actions = [
            p["text"] for p in message.get("content") or []
            if p.get("type") == "action_description" and p.get("text")
        ]
        lines = [f"Action: {a}" for a in actions]
        if lines:
            result["content"] = [{"type": "text", "text": "\n".join(lines)}]
        elif not message.get("tool_calls"):
            # Content-only final turn: no action, so the ``action_description``
            # whitelist above is empty by construction. Keep ONLY the plain
            # ``text`` parts (the canonical "Done.") so the turn is not an empty
            # SFT target, while non-``text`` kinds this format has no slot for
            # still drop. Same three-branch shape as Qwen3_5UseAdapter.
            result["content"] = [
                p for p in (message.get("content") or [])
                if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
            ]
        else:
            result["content"] = []
        return result

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """Convert AgentMessage → LiteMessage. For role=assistant, parse
        the system-prompt-level ``Action: ...`` line out of the text
        content into ``ActionDescriptionContent``. Chat-template tokens
        (``<think>``, ``<tool_call>``) were already extracted by
        :meth:`parse_raw_assistant_response`; this method only handles
        prompt conventions.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result
        if not result.get("tool_calls"):
            return result

        raw_text = ""
        for part in message.get("content") or []:
            if part.get("type") == "text" and part.get("text"):
                raw_text = part["text"]
                break

        parts: list[dict[str, Any]] = []
        if raw_text:
            m = re.search(r"Action:\s*(.*?)(?:\n|$)", raw_text, re.DOTALL)
            action_text = m.group(1).strip() if m else raw_text
            if action_text:
                parts.append({"type": "action_description", "text": action_text})
        result["content"] = parts
        return result

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """LiteBaseAdapter is a data-processing helper for SFT — it operates
        on pre-structured LiteSample data, not raw model output. Live
        inference goes through model-specific adapters (Qwen3VL, etc.)."""
        raise NotImplementedError(
            "LiteBaseAdapter is for SFT data processing, not live inference. "
            "Use a model-specific adapter (e.g. Qwen3VLBaseAdapter) for "
            "parse_raw_assistant_response."
        )

# =============================================================================
# Desktop + Browser Adapters
# =============================================================================
# Desktop and browser share one adapter class per task type — same action
# space and protocol. The (desktop|browser) regex on each class's key
# registers the same body under both ``lite@desktop@...`` and
# ``lite@browser@...``.

@dataclasses.dataclass
class LiteDesktopGroundingActionAdapter(
    LiteBaseAdapter, key=r"lite@(desktop|browser)@grounding\.action"
):
    """Desktop+browser grounding/action adapter (canonical-format pass-through).

    Used for SFT-data replay of ``grounding.action`` parquets (multi-action
    shape: ``click + type + key + ...``). lite is canonical so render is
    identity; full history is preserved.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=LiteDesktopActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )


@dataclasses.dataclass
class LiteDesktopUseAdapter(
    LiteBaseAdapter, key=r"lite@(desktop|browser)@use"
):
    """Desktop+browser ``use`` adapter.

    Uses the raw full-history protocol — every turn is replayed verbatim
    (image + text), with no rolling window or text summarization. This is
    the canonical CUA-Lite dialect; downstream agent dialects (e.g.
    ``qwen3_vl``) layer their own summarized protocols on top.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=LiteDesktopActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    system_prompt: str | None = USE_SYSTEM_PROMPT


# =============================================================================
# Mobile Adapters
# =============================================================================

@dataclasses.dataclass
class LiteMobileGroundingActionAdapter(LiteBaseAdapter, key="lite@mobile@grounding.action"):
    """Mobile grounding/action: same shape as desktop, mobile action_space."""
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=LiteMobileActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )


@dataclasses.dataclass
class LiteMobileUseAdapter(LiteBaseAdapter, key="lite@mobile@use"):
    """Mobile ``use`` adapter. Uses raw full-history protocol (see
    :class:`LiteDesktopUseAdapter`)."""
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=LiteMobileActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    system_prompt: str | None = USE_SYSTEM_PROMPT

# =============================================================================
# Pass-through Registration for Grounding / Understanding
# =============================================================================
#
# Lite is the canonical CUA-Lite dialect, so ``understanding``,
# ``grounding.bbox``, and ``grounding.point`` route through
# :class:`AsIsAdapter` (pure pass-through; tool calls are already in lite
# shape). ``grounding.action`` is served by concrete classes above
# (multi-action SFT replay needs full-history protocol + per-message
# canonical conversion).

from lite.agents.core.adapter import AgentAdapterRegistry, AsIsAdapter

AgentAdapterRegistry.register(r"lite@(desktop|browser|mobile)@understanding", AsIsAdapter)
AgentAdapterRegistry.register(r"lite@(desktop|browser|mobile)@grounding\.bbox", AsIsAdapter)
AgentAdapterRegistry.register(r"lite@(desktop|browser|mobile)@grounding\.point", AsIsAdapter)
