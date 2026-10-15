"""
Fara-1.0 Adapters

Native pixel-coordinate adapter family for Fara-1.0, a Qwen2.5-VL fine-tune for
web browsing. Subclasses :class:`Qwen2_5VLBaseAdapter` (same JSON
``<tool_call>`` chat template, same factor-28 smart_resize, same
[0,1000]↔pixel-in-resized coordinate rescaling and
``{display_width_px}x{display_height_px}`` substitution) and overrides only the
Fara deltas:

  * **System prompt template** — the reference ``FN_CALL_TEMPLATE``
    (``fara/src/fara/qwen_helpers/fncall_prompt.py``): a "web automation agent …
    Critical Points" preamble instead of the plain ``# Tools`` header. Prefixed
    with ``"You are a helpful assistant."`` (the ``NousFnCallPrompt`` system
    preface), producing byte-for-byte the reference system message.
  * **smart_resize max_pixels** — Fara's full ``16384·28·28`` cap
    (``MAX_PIXELS`` in ``_prompts.py``), vs Qwen2.5-VL's SFT-trimmed 2M.
  * **Wire format** — free-form ``thoughts`` prose followed by a single
    ``<tool_call>{json}</tool_call>`` block (the reference
    ``_parse_thoughts_and_action`` splits on ``<tool_call>``). This is exactly
    the inherited :class:`Qwen2_5VLBaseAdapter` behaviour (text parts + tool_call
    text), so there is NO ``Action:`` / ``Thought:`` navigation wire format —
    :class:`FaraDesktopUseAdapter` subclasses the base, not the Use adapter.

Subclass tree::

    FaraBaseAdapter (workflow-agnostic; FaraDesktopActionSpace + FullHistoryProtocol)
    ├── FaraDesktopGroundingActionAdapter  (SFT-replay, full schema, single turn)
    └── FaraDesktopUseAdapter              (multi-turn web navigation; Fara history)

Usage::

    from lite.agents.models.fara.adapter import (
        FaraBaseAdapter,
        FaraDesktopGroundingActionAdapter,
        FaraDesktopUseAdapter,
    )
"""

from __future__ import annotations

import dataclasses
import json
import logging

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.models.fara.action_space import (
    FaraDesktopActionSpace,
    FaraDesktopGroundingPointActionSpace,
)
from lite.agents.models.fara.protocol import FaraHistoryProtocol
from lite.agents.models.qwen2_5_vl.adapter import (
    GROUNDING_POINT_SYSTEM_PROMPT,
    Qwen2_5VLBaseAdapter,
)
from lite.agents.types import AgentMessage, AgentStep
from lite.core import (
    LiteCUAMetadata,
    LiteMessage,
    LiteSample,
)
from lite.core.messages.turns import truncate_sample_to_turn
from lite.core.tools.calls import tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.schemas import validate_extra_tool_schemas

logger = logging.getLogger(__name__)


# =============================================================================
# Smart-resize constants (Fara / Qwen2.5-VL image processor)
# =============================================================================

# Fara's MLM processor (``_prompts.py``): patch_size=14, merge_size=2 → factor 28,
# max_pixels = 16384·28·28 = 12,845,056 (the reference ``MAX_PIXELS`` and
# ``MLM_PROCESSOR_IM_CFG['max_pixels']``). This is the FULL cookbook cap — larger
# than the Qwen2.5-VL sibling's SFT-trimmed 2M. min_pixels = 56·56 = 3136 is
# already the cua-lite ``smart_resize`` default (matches Fara's ``MIN_PIXELS``),
# so it needs no override.
_FACTOR = 28
_MAX_PIXELS = 16384 * 28 * 28


# =============================================================================
# System prompt template (Fara FN_CALL_TEMPLATE)
# =============================================================================

# The reference ``FN_CALL_TEMPLATE`` preamble (fncall_prompt.py). Rendered
# before the ``<tools>`` block, in place of Qwen's plain ``# Tools`` header.
_FARA_TOOLS_PREAMBLE = (
    "You are a web automation agent that performs actions on websites to fulfill user requests by calling various tools.\n"
    "* You should stop execution at Critical Points. A Critical Point would be encountered in tasks like 'Checkout', 'Book', 'Purchase', 'Call', 'Email', 'Order', etc where a binding transaction/agreement would require the user's permission/personal or sensitive information (name, email, credit card, address, payment information, resume, etc) in order to complete a transaction (purchase, reservation, sign-up etc), or to communicate in a way that a human would be expected to do (call, email, apply to a job, etc).\n"
    "* Solve the task as far as you can up until a Critical Point:\n"
    "    - For example, if the task is to \"call a restaurant to make a reservation\", you should not actually make the call but should navigate to the restaurant's page and find the phone number.\n"
    "    - Similarly, if the task is to \"order new size 12 running shoes\" you should not actually place the order but should instead search for the right shoes that meet the criteria and add them to the cart.\n"
    "    - Some tasks, like answering questions, may not encounter a Critical Point at all."
)


# =============================================================================
# Fara Base Adapter
# =============================================================================

@dataclasses.dataclass
class FaraBaseAdapter(
    Qwen2_5VLBaseAdapter,
    key=r"fara\.base(@(desktop|browser)@(use|grounding\.action|grounding\.point))?",
):
    """Base adapter for the Fara-1.0 wire format.

    Inherits the Qwen2.5-VL ``<tool_call>{json}</tool_call>`` text serialization,
    [0,1000]↔pixel-in-resized coordinate rescaling, and
    ``parse_raw_assistant_response`` (no ``<think>`` channel). Overrides the
    smart_resize cap, the ``<tools>`` template header, ``render_step`` (to
    cache the resized image size per message for coordinate rescaling), and
    ``convert_message_from_agent`` (to expose Fara's terminate-turn answer).

    Subclasses set ``action_space`` and ``protocol``.
    """

    action_space: BaseActionSpace = dataclasses.field(
        default_factory=FaraDesktopActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )

    # No leading "# Response format" block — the tools section (with the web
    # preamble) is the whole system message body, after the inherited
    # "You are a helpful assistant." prefix.
    system_prompt: str | None = None

    # Fara's full smart_resize cap (vs the Qwen2.5-VL sibling's trimmed 2M).
    smart_resize_factor: int = _FACTOR
    smart_resize_max_pixels: int = _MAX_PIXELS

    def _build_tools_section(self, image_size: tuple[int, int] | None = None) -> str:
        """Format tool schemas into the Fara ``FN_CALL_TEMPLATE`` tools section.

        Same ``<tools>`` / ``<tool_call>`` framing as the Qwen parent but with
        the web-automation "Critical Points" preamble instead of the ``# Tools``
        header. When ``image_size`` is provided, substitutes the
        ``{display_width_px}`` / ``{display_height_px}`` placeholders in the tool
        descriptions with the resized image dims.
        """
        metadata = self._cua_metadata()
        tool_schemas = self.action_space.get_tool_schemas()
        if metadata.valid_actions is not None:
            tool_schemas = (
                type(self.action_space).filter_tool_schemas_for_valid_actions(
                    tool_schemas, metadata.valid_actions
                )
            )
        # Fara did not render standalone env extras into the prompt. Keep only
        # the native action values whose canonical extra tool is active.
        tool_schemas = (
            type(self.action_space).filter_fara_action_values_for_active_extra_tools(
                tool_schemas,
                self.active_extra_tool_names(),
            )
        )
        # Same canonical nested schema shape as Qwen; Fara's reference differs
        # only in byte policy, dumping tool JSON with ``ensure_ascii=False``.
        validate_extra_tool_schemas(
            tool_schemas,
            where="FaraBaseAdapter._build_tools_section.tool_schemas",
        )
        tools_json = "\n".join(
            json.dumps(schema, ensure_ascii=False) for schema in tool_schemas
        )
        if image_size is not None:
            W, H = image_size
            tools_json = (
                tools_json
                .replace("{display_width_px}", str(W))
                .replace("{display_height_px}", str(H))
            )
        return (
            f"{_FARA_TOOLS_PREAMBLE}\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tools_json}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render turn ``k``: protocol on truncated history + Fara system message.

        Updates ``self._current_image_size`` before converting each message so
        coordinate rescaling uses that message's screenshot dims. Fara's viewport
        is fixed, so all screenshots resize to the same size; the per-message
        update keeps it correct even if that ever changes. The tools-section
        placeholder is substituted with the last in-window image's dims.
        """
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        # Last in-window image → placeholder substitution for the tools section.
        last_image_size: tuple[int, int] | None = None
        for msg in reversed(messages):
            for part in (msg.get("content") or []):
                if isinstance(part, dict) and part.get("type") == "image":
                    idx = part.get("index")
                    if idx is not None and 0 <= idx < len(processed):
                        img = processed[idx]
                        if hasattr(img, "size"):
                            last_image_size = img.size
                            break
            if last_image_size is not None:
                break
        if last_image_size is None and processed:
            img = processed[-1]
            if hasattr(img, "size"):
                last_image_size = img.size

        result_messages: list[AgentMessage] = []
        parts_sys: list[str] = [self._SYSTEM_PREFIX]
        if self.system_prompt:
            parts_sys.append(self.system_prompt)
        parts_sys.append(self._build_tools_section(image_size=last_image_size))
        system_text = "\n\n".join(parts_sys)
        result_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        })

        for msg in messages:
            # Update _current_image_size per-message for correct coord rescaling.
            for part in (msg.get("content") or []):
                if isinstance(part, dict) and part.get("type") == "image":
                    idx = part.get("index")
                    if idx is not None and 0 <= idx < len(processed):
                        img = processed[idx]
                        if hasattr(img, "size"):
                            self._current_image_size = img.size
            result_messages.append(self.convert_message_to_agent(msg))

        return result_messages

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """Convert Fara's terminate-turn thoughts into the canonical answer call.

        Fara has no native ``response`` tool. On terminating turns, its final
        answer is the free-form thoughts text before the ``<tool_call>`` block.
        When the environment advertised ``response``, prepend a canonical
        ``response(text=<thoughts>)`` call before ``terminate`` so browser envs
        that read answers from ``response`` can score the episode. Render replay
        still uses the assistant ``raw_response`` sidecar, so this projection
        stays off Fara's native wire.
        """
        result = super().convert_message_from_agent(message, **kwargs)
        if result.get("role") != "assistant":
            return result
        if "response" not in self.active_extra_tool_names():
            return result

        tool_calls = result.get("tool_calls") or []
        if not any(tool_call_name(tool_call) == "terminate" for tool_call in tool_calls):
            return result

        thoughts = next(
            (
                part["text"]
                for part in (result.get("content") or [])
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
            ),
            "",
        )
        if not thoughts:
            return result

        result["tool_calls"] = [LiteFinishToolSet.response(text=thoughts), *tool_calls]
        return result


# =============================================================================
# Concrete Adapters (desktop/browser)
# =============================================================================

@dataclasses.dataclass
class FaraDesktopGroundingActionAdapter(
    FaraBaseAdapter, key=r"fara@(desktop|browser)@grounding\.action"
):
    """Fara grounding/action: full ``computer_use`` vocabulary, single turn.

    For SFT-data replay / offline rendering of Fara-format grounding parquets
    under the family-native ``<tool_call>`` wire format. Single-step content is
    structured tool_calls; the assistant ``thoughts`` text flows through the base
    text path. Mirrors :class:`Qwen2_5VLDesktopGroundingActionAdapter`.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=FaraDesktopActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )


@dataclasses.dataclass
class FaraDesktopUseAdapter(
    FaraBaseAdapter, key=r"fara@(desktop|browser)@use"
):
    """Fara web navigation: full action vocab + Fara sliding-image history.

    Uses :class:`FaraHistoryProtocol` (keep all text; newest ``max_n_images=3``
    screenshots) — matching the reference ``FaraAgent`` default. No
    ``Action:`` / ``Thought:`` wire format: the assistant turn is free-form
    thoughts prose + a single ``<tool_call>`` block, produced by the inherited
    base converter.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=FaraDesktopActionSpace
    )
    protocol: FaraHistoryProtocol = dataclasses.field(
        default_factory=lambda: FaraHistoryProtocol(max_n_images=3)
    )
    system_prompt: str | None = None


@dataclasses.dataclass
class FaraDesktopGroundingPointAdapter(
    Qwen2_5VLBaseAdapter, key=r"fara@(desktop|browser)@grounding\.point"
):
    """Fara grounding (single-step click) for ScreenSpot-Pro-style benches.

    Fara has no native grounding mode, so this subclasses
    :class:`Qwen2_5VLBaseAdapter` directly (standard ``# Tools`` header +
    ``GROUNDING_POINT_SYSTEM_PROMPT`` rules block — NOT Fara's web-automation
    ``FN_CALL_TEMPLATE`` preamble, which is meant for multi-step web tasks). It
    keeps Fara's factor-28 grid and full ``max_pixels`` cap, and uses the
    trimmed :class:`FaraDesktopGroundingPointActionSpace` (``left_click`` only).
    The adapter round-trips ``computer_use(left_click, coord)`` ↔ cua-lite
    ``point(coord)`` with pixel↔[0,1000] rescaling.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=FaraDesktopGroundingPointActionSpace
    )
    metadata: LiteCUAMetadata = dataclasses.field(
        default_factory=lambda: LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.DESKTOP.value,
                LiteCUAMetadata.TaskType.GROUNDING_POINT.value,
            )
        )
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    system_prompt: str | None = GROUNDING_POINT_SYSTEM_PROMPT
    # Fara's full processor cap (vs the Qwen2.5-VL parent's SFT-trimmed 2M).
    smart_resize_max_pixels: int = _MAX_PIXELS
