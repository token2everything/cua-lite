"""Provider-independent protocol registry and message-history base classes.

Usage:
    from lite.agents.core.protocol import BaseProtocol

A protocol turns canonical Lite messages into the subset and order a model sees
at one inference call. ``BaseProtocol.process_messages`` is the main hook.
``TurnWindowProtocol`` supplies the common turn-window frame for families that
select complete observation/assistant turns; ``FullHistoryProtocol`` keeps the
full message list. Shared window helpers live in ``protocol/window.py``.

Protocol keys use ``.`` for named flavors such as ``qwen3_vl.history`` and
``browsergym.generic``; a trailing ``@mobile`` is reserved for the rare
platform-specific protocol.
"""

from __future__ import annotations

import copy
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from lite.agents.core.protocol.window import inject_summary_text
from lite.core import (
    LiteMessage,
)
from lite.core.messages import (
    USER_ROLE,
    get_action_description,
    group_into_turns,
    instruction_text,
    message_has_image,
    peel_system_message,
)

# ``require_message_list`` is the content owner's boundary check rather than a
# package-level front-door contract, so it is imported from its owning module.
from lite.core.messages.content import require_message_list
from lite.utils.registry import BaseRegistry, RegistryKeyed

# =============================================================================
# Public API: Registry
# =============================================================================

class ProtocolRegistry(BaseRegistry["BaseProtocol"], cache_by_default=False):
    """
    Registry for protocol classes.

    Supports exact matching and regex pattern matching.
    Uses cache_by_default=False because protocols may have instance state
    (e.g. full_history_size); callers pass kwargs to get().

    Example:
        ProtocolRegistry.get("lite.history")
        ProtocolRegistry.get("ui_tars.history", full_history_size=5)
        ProtocolRegistry.list()
        ProtocolRegistry.list_patterns()
    """
    pass

# =============================================================================
# Base Protocol
# =============================================================================

class BaseProtocol(RegistryKeyed, ABC):
    """Base class for model-visible message-history protocols.

    Subclasses implement ``process_messages`` to select, order, summarize, or
    rewrite canonical Lite messages for one model call.

    Subclasses can auto-register via the key parameter:

        @dataclasses.dataclass
        class MyProtocol(BaseProtocol, key="my_protocol"):
            ...

    Key naming: a protocol key is ``<name>.<flavor>`` — the flavor (``history``,
    ``generic``) is an author-fixed modifier, so it uses ``.`` not ``@``
    (``qwen3_5.history``, ``browsergym.generic``). ``@`` is reserved for
    structured coordinate dimensions (platform / task_type / action-format),
    which a protocol key normally has none of; a platform-specific protocol
    appends one as a trailing dim — ``my_protocol.history@mobile``. See
    ``lite.utils.registry.BaseRegistry`` for the full key grammar.

    ``lite.core.messages.group_into_turns`` owns the grouped-turn dict shape:
    protocol code reads ``turn["observations"]`` for observation blocks and uses
    instruction extraction helpers for task text.
    """

    _registry = ProtocolRegistry

    #: Optional ``(LiteMessage) -> str`` override for ``_format_action``.
    #: Declared here as the seam's default so ``_format_action`` can live on the
    #: base; dataclass subclasses that expose it as a constructor kwarg simply
    #: re-declare it as a field.
    action_format_func: ClassVar[Callable[[LiteMessage], str] | None] = None

    @abstractmethod
    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs,
    ) -> list[LiteMessage]:
        """
        Process messages according to the protocol.

        Args:
            messages: list of CUA-lite messages
            **kwargs: Additional protocol-specific parameters

        Returns:
            Processed messages
        """
        raise NotImplementedError

    @classmethod
    def turn_has_image(cls, turn: dict[str, Any]) -> bool:
        """Does any observation of this turn carry an image?

        THE predicate the history windows budget against. Every image-budget
        knob a protocol exposes exists to bound **images**, so a turn that
        contributes none costs nothing and must not be evicted or collapsed by
        an image budget.

        Deliberately NOT a tool-name list. True standalone text-tool surfaces
        such as ``bash`` carry no image, while ``goto`` / ``open_app`` DO move
        pixels and a screenshot follows them. Asking the observation what it
        actually contains avoids both drift and over-generalizing text-tool
        history to GUI-only protocols.
        """
        return any(message_has_image(obs) for obs in turn["observations"])

    @classmethod
    def image_turn_indices(cls, turns: list[dict[str, Any]]) -> list[int]:
        """Absolute 0-based indices of the turns that carry an image.

        On GUI-only trajectories this is ``list(range(len(turns)))``, so image
        budgets reduce to ordinary turn-count budgets.
        """
        return [i for i, turn in enumerate(turns) if cls.turn_has_image(turn)]

    # -- Summarization seams ---------------------------------------------------
    #
    # Shared summarization hooks. Protocols with a different convention override
    # only the needed method. ``_tool_result_summary_text`` remains
    # protocol-local because callers use different units.

    def _extract_instruction(self, turn: dict[str, Any]) -> str:
        """First user turn's leading text (the task goal)."""
        observations = turn["observations"]
        if not any(obs.get("role") == USER_ROLE for obs in observations):
            raise ValueError("first grouped turn has no role:user instruction message")
        return instruction_text(observations)

    def _format_action(self, message: LiteMessage, step_num: int) -> str:
        """Render one summary line for an assistant message.

        Strict ``action_description`` lookup — understanding/QA turns tag their
        assistant output as plain ``text``, which is not the right source for a
        trajectory summary and must not be picked up here.
        """
        if self.action_format_func:
            return self.action_format_func(message)
        text = get_action_description(message)
        return f"Step {step_num}: {text}" if text else f"Step {step_num}: (action)"

    def _inject_text(self, message: LiteMessage, text: str) -> None:
        """In-place set/append a ``type=text`` part on ``message``."""
        inject_summary_text(message, text)


class TurnWindowProtocol(BaseProtocol):
    """Shared frame for protocols that select whole conversation turns.

    Subclasses implement :meth:`_select_messages`; this class handles system
    peeling, turn grouping, and system-message reattachment. Protocols that keep
    every message or rewrite message content directly implement
    ``process_messages`` on their own instead.
    """

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs,
    ) -> list[LiteMessage]:
        """Peel the system message, group the rest into turns, let the family
        select what survives, then put the system message back in front.

        ``messages`` must be a concrete list: an empty window is ``[]``, and
        ``None`` is a caller-code mistake rather than a second spelling of it.
        """
        require_message_list(messages, where="process_messages")
        if len(messages) == 0:
            return []
        messages = copy.deepcopy(messages)
        system_message, content = peel_system_message(messages)
        turns = group_into_turns(content)
        result = self._select_messages(content, turns)
        if system_message:
            result.insert(0, system_message)
        return result

    @abstractmethod
    def _select_messages(
        self,
        content: list[LiteMessage],
        turns: list[dict[str, Any]],
    ) -> list[LiteMessage]:
        """Family hook: return the kept non-system messages.

        ``content`` is the system-stripped deep copy and ``turns`` is the
        grouped view over that same list. The caller may insert the system
        message into the returned list.
        """
        raise NotImplementedError


@dataclasses.dataclass
class FullHistoryProtocol(BaseProtocol, key="lite.history"):
    """
    Protocol that preserves the full conversation history.

    This is the simplest protocol — all messages are kept as-is.
    """

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs,
    ) -> list[LiteMessage]:
        """Return a deep copy of every message.

        ``messages`` must be a concrete list: an empty history is ``[]``. A
        ``None`` history would otherwise deep-copy to ``None`` and travel on as a
        second spelling of "no messages", so it is rejected here under the same
        contract every other message window states.
        """
        require_message_list(messages, where="process_messages")
        return copy.deepcopy(messages)
