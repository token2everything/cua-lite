"""
Qwen2.5-VL Adapters

Native pixel-coordinate adapter family for Qwen2.5-VL-{3B,7B}-Instruct. Mirrors
:mod:`lite.agents.models.qwen3_vl.adapter` in structure but matches Qwen2.5-VL's
pretraining distribution byte-for-byte:

  * 11-action ``computer_use`` enum (no ``triple_click`` / ``hscroll`` / ``answer``).
  * smart_resize ``factor=28`` (= patch_size 14 × spatial_merge 2), ``max_pixels=2_007_040``
    (``14*14*4*2560`` — deliberately below the 12_845_056 the other families use; see
    the SFT activation-memory note beside ``_MAX_PIXELS`` below).
  * System prompt is the standard Qwen-Agent tools template prefixed with
    ``"You are a helpful assistant."`` (matches upstream Jedi eval's
    ``FN_CALL_TEMPLATE``); no separate "# Response format" rules block.
  * Wire-format coordinates are **pixel coords in the smart-resized image space**.
    The adapter rescales between cua-lite [0, 1000] and pixel-in-resized using
    ``self._current_image_size`` cached during :meth:`render_step`.
  * The action-space tool descriptions contain ``{display_width_px}x{display_height_px}``;
    the adapter substitutes the actual resized dims at render time.

Protocols are reused from :mod:`lite.agents.models.qwen3_vl.protocol` (model-agnostic).

Subclass tree:

    Qwen2_5VLBaseAdapter (workflow-agnostic; identity action_space + FullHistoryProtocol)
    ├── Qwen2_5VLUseAdapter              (intermediate; Action:/Thought: wire format)
    │   └── Qwen2_5VLDesktopUseAdapter   (desktop+browser ``use``; summarized history)
    ├── Qwen2_5VLDesktopGroundingPointAdapter   (env-eval grounding, trimmed schema; desktop+browser)
    ├── Qwen2_5VLDesktopGroundingActionAdapter  (SFT-replay grounding, full schema; desktop+browser)
    ├── Qwen2_5VLMobileGroundingPointAdapter    (mobile env-eval grounding)
    └── Qwen2_5VLMobileGroundingActionAdapter   (mobile SFT-replay grounding)
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import math
import re
from typing import Any

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.adapter import (
    AgentAdapterRegistry,
    AsIsAdapter,
)
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.models.qwen2_5_vl.action_space import (
    Qwen2_5VLDesktopActionSpace,
    Qwen2_5VLDesktopGroundingPointActionSpace,
    Qwen2_5VLMobileActionSpace,
    Qwen2_5VLMobileGroundingPointActionSpace,
)
from lite.agents.models.qwen3_vl.adapter import (
    Qwen3VLBaseAdapter,
)
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.agents.types import AgentMessage, AgentStep
from lite.core import (
    LiteCUAMetadata,
    LiteMessage,
    LiteSample,
)
from lite.core.messages import make_assistant_content
from lite.core.messages.final import mark_model_output_error
from lite.core.messages.turns import truncate_sample_to_turn
from lite.core.tools.action_space import pixel_to_norm
from lite.core.tools.action_space.geometry import strict_norm_to_pixel
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name, validate_extra_tool_schemas

logger = logging.getLogger(__name__)

# =============================================================================
# Smart-resize constants (Qwen2.5-VL)
# =============================================================================

# Qwen2.5-VL HF processor defaults: patch_size=14, spatial_merge_size=2 → factor=28.
# max_pixels cap = 14·14·4·2560 = 2,007,040 → at most 2560 image tokens per image.
# Tuned down from the upstream cookbook default (14·14·4·16384 = 12.85M) so that
# the SFT prompt fits the activation memory budget on 2× 80 GB H100 with
# MAX_TOKENS_PER_GPU=3072 / TP=1 / DP=2. The 4M intermediate that we tried
# earlier left p99 ≈ 5400 tokens, which OOM'd at iter 0 (single 5K-token sample
# in a 4096-token micro-batch exceeds the ~42 GB activation budget once you
# account for the 1.6 GB vocab projection per sample plus vision tower
# intermediates). 2M gives p99 ≈ 2870 tokens — fits 3072-budget micro-batches
# with margin and keeps the bimodal-distribution tail intact (vs filtering it
# with MAX_PROMPT_LEN, which would drop ~50% of samples).
#
# Override via ``agent_kwargs.smart_resize_max_pixels`` in the yaml if you need
# native-fidelity 4K+ images (eval-only; will OOM under the SFT config).
#
# The resize body lives on :class:`Qwen3VLBaseAdapter._smart_resize_image`; this
# subclass only rebinds the ``smart_resize_factor`` / ``smart_resize_max_pixels``
# class attrs to these constants.
_FACTOR = 28
_MAX_PIXELS = 14 * 14 * 4 * 2560


# =============================================================================
# System Prompts
# =============================================================================

# Grounding-only system prompt. Single-step click; no Thought / Action prose.
# Mirrors qwen3_vl's GROUNDING_POINT_SYSTEM_PROMPT — the coordinate space
# (pixel-in-resized) is already declared in the tools section via the
# ``{display_width_px}x{display_height_px}`` substitution, so we only need
# the response-format block here.
GROUNDING_POINT_SYSTEM_PROMPT = """# Response format

For each grounding instruction, return a single <tool_call>...</tool_call> block.

Rules:
- Output exactly one <tool_call> block, nothing else.
- Click the (x, y) pixel coordinate of the target element in the screen.
- Do not produce any prose, explanation, or extra tool calls."""

# The default desktop ``use`` adapter uses system_prompt=None (no extra rules
# block), matching the official Qwen2.5-VL OSWorld reference script which
# relies solely on the "You are a helpful assistant" prefix + tools section.
# Callers experimenting with ``enable_inline_reasoning=True`` must supply their
# own ``system_prompt`` that asks for the ``Thought:`` line.


# =============================================================================
# Qwen2.5-VL Base Adapter
# =============================================================================

@dataclasses.dataclass
class Qwen2_5VLBaseAdapter(
    Qwen3VLBaseAdapter,
    key=(
        r"qwen2_5_vl\.base"
        r"(@(desktop|browser|mobile)"
        r"@(use|understanding|grounding\.action|grounding\.point|grounding\.bbox))?"
    ),
):
    """Base adapter for the Qwen2.5-VL chat-template format.

    Subclasses :class:`Qwen3VLBaseAdapter` (same JSON ``<tool_call>`` chat
    template) and overrides only the Qwen2.5-VL deltas:

      * ``smart_resize_factor`` / ``smart_resize_max_pixels`` (28-px grid,
        smaller pixel cap) — the resize body is inherited.
      * ``render_step`` — caches ``_current_image_size`` and passes it to the
        (inherited) ``_build_tools_section`` for ``{display_width_px}`` /
        ``{display_height_px}`` substitution.
      * ``_convert_message_to_agent`` / ``convert_message_from_agent`` —
        Qwen2.5-VL's chat_template drops structured ``tool_calls``, so they are
        serialized into ``<tool_call>{json}</tool_call>`` content text, with
        cua-lite [0,1000] ↔ pixel-in-resized coordinate rescaling.
      * ``parse_raw_assistant_response`` — reuses the parent's shared
        ``<tool_call>`` extraction but drops ``<think>`` parsing (Qwen2.5-VL
        has no native thinking channel).

    Subclasses must set ``action_space`` and ``protocol``.
    """

    action_space: BaseActionSpace = dataclasses.field(default_factory=BaseActionSpace)
    protocol: FullHistoryProtocol = dataclasses.field(default_factory=FullHistoryProtocol)

    # No leading system_prompt block by default — the tools section already
    # carries the action schema (including the screen-resolution sentence).
    # Subclasses may set this to inject additional rules.
    system_prompt: str | None = None
    # NOTE: extra_tool_schemas / valid_actions are NOT adapter fields — they are
    # read from ``self.metadata`` (the single source of truth, forwarded by
    # make at rollout and export_sft at training), like every other
    # adapter. Do not reintroduce them as fields: a field here would shadow
    # metadata for this adapter alone, so an SFT prompt could render a tool
    # surface different from the one the rollout data was generated under
    # while every metadata-based adapter still read metadata.

    # Qwen2.5-VL-specific resize knobs (28-px grid; smaller pixel cap than the
    # Qwen3-VL parent's 32 / large cap). ``smart_resize_enabled`` +
    # ``enable_thinking`` inherit the parent's defaults (True / False).
    smart_resize_factor: int = _FACTOR
    smart_resize_max_pixels: int = _MAX_PIXELS

    # The standard Qwen-Agent system-message prefix used by every Qwen2.5-VL
    # cookbook example (Jedi eval, computer_use.ipynb, mobile_agent.ipynb).
    # Keep this distinct from ``system_prompt`` so subclasses can still add a
    # task-specific rules block without losing the "helpful assistant" preface.
    _SYSTEM_PREFIX: str = "You are a helpful assistant."

    # Cached during render_step from the current step's resized image dims;
    # used by _convert_message_to_agent / convert_message_from_agent to rescale
    # coordinates between cua-lite [0,1000] and pixel-in-resized.
    _current_image_size: tuple[int, int] | None = dataclasses.field(default=None, repr=False)

    # __post_init__ inherited from BaseAgentAdapter (extra_tool_schemas collision
    # check lives there; nothing qwen2_5-specific to add).

    # ``_process_image_after_target`` (smart_resize gated on
    # ``smart_resize_enabled``) is inherited from :class:`Qwen3VLBaseAdapter`;
    # Qwen2.5-VL keeps its trained native tool grammar fixed. Subclasses can
    # expose narrowly supported standalone extras alongside that native schema.

    def _extra_tool_schemas_for_prompt(self) -> list[dict[str, Any]]:
        return []

    def _build_tools_section(self, image_size: tuple[int, int] | None = None) -> str:
        """Format the native Qwen2.5-VL tool schema block.

        Env extra tools gate runtime acceptance and persisted Lite calls; by
        default they are not rendered into this open-source model's trained
        prompt grammar.
        """
        metadata = self._cua_metadata()
        tools = self.action_space.get_tool_schemas()
        if metadata.valid_actions is not None:
            tools = (
                type(self.action_space).filter_tool_schemas_for_valid_actions(
                    tools, metadata.valid_actions
                )
            )
        # Second, ORTHOGONAL gate: the Qwen action values ``terminate`` /
        # ``open`` are the wire spelling of standalone canonical tools, so they
        # appear only while the matching schema is active. Only the two full
        # Qwen2.5-VL surfaces declare such action values; every other action
        # space paired with this base renders its wrapper enum unchanged.
        if isinstance(
            self.action_space,
            (Qwen2_5VLDesktopActionSpace, Qwen2_5VLMobileActionSpace),
        ):
            tools = type(self.action_space).filter_qwen_action_values_for_active_extra_tools(
                tools,
                self.active_extra_tool_names(),
            )
        tools = tools + self._extra_tool_schemas_for_prompt()
        # Byte policy (this family owns it; there is no shared Qwen prompt
        # helper): one JSON object per line, ``json.dumps`` defaults, so
        # non-ASCII escapes as ``\uXXXX``.
        validate_extra_tool_schemas(
            tools,
            where="Qwen2_5VLBaseAdapter._build_tools_section.tool_schemas",
        )
        tools_json = "\n".join(json.dumps(schema) for schema in tools)
        if image_size is not None:
            W, H = image_size
            tools_json = (
                tools_json
                .replace("{display_width_px}", str(W))
                .replace("{display_height_px}", str(H))
            )
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
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

    # -------------------------------------------------------------------------
    # render_step (overridden to cache image size + substitute placeholders)
    # -------------------------------------------------------------------------

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render turn ``k``: protocol on truncated history + system prompt.

        Caches ``self._current_image_size`` from the last image referenced in
        the in-window messages (single-turn grounding has exactly one image;
        use would need per-turn rescaling, which is out of scope here).
        """
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        # Find the resized image size for the current step. Walk messages
        # backward to find the latest ImageContent.index; fall back to the
        # last entry in `processed` if no explicit index is present.
        image_size: tuple[int, int] | None = None
        for msg in reversed(messages):
            for part in (msg.get("content") or []):
                if isinstance(part, dict) and part.get("type") == "image":
                    idx = part.get("index")
                    if idx is not None and 0 <= idx < len(processed):
                        img = processed[idx]
                        if hasattr(img, "size"):
                            image_size = img.size
                            break
            if image_size is not None:
                break
        if image_size is None and processed:
            img = processed[-1]
            if hasattr(img, "size"):
                image_size = img.size
        # Cache for _convert_message_to_agent / convert_message_from_agent.
        self._current_image_size = image_size

        result_messages: list[AgentMessage] = []
        parts: list[str] = [self._SYSTEM_PREFIX]
        if self.system_prompt:
            parts.append(self.system_prompt)
        parts.append(self._build_tools_section(image_size=image_size))
        system_text = "\n\n".join(parts)
        result_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        })

        for msg in messages:
            result_messages.append(self.convert_message_to_agent(msg))

        return result_messages

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """LiteMessage → AgentMessage with [0,1000]→pixel rescaling on coords.

        Unlike Qwen3-VL's chat_template (which renders structured
        ``tool_calls`` natively), **Qwen2.5-VL's chat_template only renders
        ``content`` text and silently drops structured ``tool_calls``** —
        leaving the assistant turn empty between ``<|im_start|>assistant\\n``
        and ``<|im_end|>``. The upstream OSWorld-G Jedi eval works around this
        by literally writing ``<tool_call>\\n{"name": ..., "arguments": ...}\\n</tool_call>``
        as the assistant ``content`` text (see ``FN_CALL_TEMPLATE`` /
        ``qwen25_vllm_osworld_g_jedi.py``).

        We mirror that wire format here: after coord rescaling, every tool
        call is serialized as a ``<tool_call>{json}</tool_call>`` line and
        merged into ``content`` (after any pre-existing text). Structured
        ``tool_calls`` is dropped from the agent-side message. This makes
        SFT replay actually train on the JSON tokens (without it,
        ``response_text`` is just ``<|im_end|>\\n`` ≈ 2 tokens, and loss
        collapses to numerical zero by step ~1k because the model only
        learns the EOS token it already emits perfectly). At inference, the
        model emits the same string and :meth:`parse_raw_assistant_response`
        recovers ``tool_calls`` from the text — symmetric round-trip.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result
        tool_call_text_lines: list[str] = []
        if "tool_calls" in result:
            self._reject_unsupported_canonical_tools(result["tool_calls"])
            passthrough_extra_names = self.active_extra_tool_names() - {"terminate"}
            agent_calls: list[dict[str, Any]] = []
            for tc in result["tool_calls"]:
                if self._admits_active_extra_tool_call(
                    tc,
                    allowed_names=passthrough_extra_names,
                ):
                    agent_calls.append(
                        {
                            "name": tool_call_name(tc),
                            "arguments": tool_call_arguments(tc),
                        }
                    )
                else:
                    agent_calls.extend(self.action_space.convert_tool_calls_to_agent([tc]))
            # Rescale [0,1000]-normalized coords → pixel-in-resized.
            if self._current_image_size is not None:
                W, H = self._current_image_size
                for tc in agent_calls:
                    self._rescale_coords_in_args(
                        tc["arguments"],
                        scale=lambda x, y: list(
                            strict_norm_to_pixel([x, y], W, H, clamp=False)
                        ),
                    )
            # Serialize each tool_call as a <tool_call>{json}</tool_call>
            # literal string for inclusion in content. Matches the upstream
            # Jedi eval's wire format byte-for-byte (incl. literal newlines
            # inside the block).
            for tc in agent_calls:
                tc_obj = {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }
                tool_call_text_lines.append(
                    f"<tool_call>\n{json.dumps(tc_obj, ensure_ascii=False)}\n</tool_call>"
                )
            # Drop structured tool_calls — content text now carries them
            # so Qwen2.5-VL's chat_template actually renders the assistant turn.
            result.pop("tool_calls", None)
        content = message.get("content") or []
        texts = [
            p["text"] for p in content
            if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
        ]
        all_text_parts = texts + tool_call_text_lines
        result["content"] = (
            [{"type": "text", "text": "\n".join(all_text_parts)}] if all_text_parts else []
        )
        return result

    def _reject_unsupported_canonical_tools(self, tool_calls: list[dict[str, Any]]) -> None:
        for tc in tool_calls:
            if tool_call_name(tc) == "response":
                raise ValueError(
                    f"{type(self).__name__} cannot render canonical tool 'response'"
                )

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """AgentMessage → LiteMessage with pixel→[0,1000] rescaling on coords."""
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        if "tool_calls" in result:
            # Rescale pixel-in-resized → [0,1000] BEFORE action_space translates,
            # because action_space copies coords through unchanged. Doing it on
            # the agent-format dicts is equivalent (action_space is identity on
            # coords) but matches the "wire format normalization" semantics.
            if self._current_image_size is not None:
                W, H = self._current_image_size
                for tc in result["tool_calls"]:
                    self._rescale_coords_in_args(
                        tc["arguments"],
                        scale=lambda x, y: pixel_to_norm(x, y, W, H),
                    )
            self._set_converted_tool_calls_from_agent(result, result["tool_calls"])

        raw_text = ""
        for part in message.get("content") or []:
            if part.get("type") == "text" and part.get("text"):
                raw_text = part["text"]
                break
        result["content"] = (
            [{"type": "text", "text": raw_text}] if raw_text else []
        )
        return result

    @staticmethod
    def _rescale_coords_in_args(args: dict[str, Any], *, scale) -> None:
        """Mutate ``args[coordinate]`` and ``args[coordinate2]`` in place.

        ``scale`` is a callable ``(x, y) -> [x', y']``. Non-list / non-2-element
        values are left untouched.
        """
        for key in ("coordinate", "coordinate2"):
            c = args.get(key)
            if isinstance(c, list) and len(c) >= 2:
                try:
                    x_float, y_float = float(c[0]), float(c[1])
                    if not math.isfinite(x_float) or not math.isfinite(y_float):
                        raise ValueError("non-finite coordinate")
                    x, y = int(x_float), int(y_float)
                except (TypeError, ValueError, OverflowError):
                    continue
                args[key] = scale(x, y)

    # -------------------------------------------------------------------------
    # Chat-template token parsing (identical to qwen3_vl base — Qwen2.5-VL
    # uses the same <tool_call>...</tool_call> tag format).
    # -------------------------------------------------------------------------

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """Parse raw model output into an AgentMessage.

        Qwen2.5-VL emits ``<tool_call>{json}</tool_call>`` blocks identical
        to Qwen3-VL — reuses the parent's shared
        :meth:`Qwen3VLBaseAdapter._extract_json_tool_calls` (same two paths:
        normal + sglang-VLM fallback where the open token id was stripped) —
        but Qwen2.5-VL has NO native ``<think>`` channel, so this drops the
        parent's ``<think>``→``reasoning_content`` parsing.
        """
        result: AgentMessage = {"role": "assistant"}

        tool_calls = self._extract_json_tool_calls(response)
        if tool_calls:
            result["tool_calls"] = tool_calls
        elif "<tool_call>" in response or "</tool_call>" in response:
            # The chat-template marker is present but nothing parsed out of it:
            # a grammar failure, NOT a deliberate content-only final. Signal it
            # so the shared loop records a terminal parse-failure final instead
            # of a clean content-only final.
            mark_model_output_error(result, "malformed <tool_call> JSON")

        clean = response
        clean = re.sub(r"<tool_call>.*?</tool_call>", "", clean, flags=re.DOTALL)
        clean = re.sub(r"\{[^<]*\}\s*</tool_call>", "", clean)
        clean = clean.strip()
        if clean:
            result["content"] = [{"type": "text", "text": clean}]

        return result


# =============================================================================
# ``use`` adapter (intermediate)
# =============================================================================

@dataclasses.dataclass
class Qwen2_5VLUseAdapter(Qwen2_5VLBaseAdapter):
    """Intermediate adapter that adds the ``use`` wire format on top of
    :class:`Qwen2_5VLBaseAdapter`.

    Wire format (to-agent, SFT training data):
      * 2-part (default, ``enable_inline_reasoning=False``):
        ``Action: <text>\\n<tool_call>{json}</tool_call>``
      * 3-part (``enable_inline_reasoning=True``):
        ``Thought: <text>\\nAction: <text>\\n<tool_call>{json}</tool_call>``

    On the to-agent side, structured ``action_description`` /
    ``inline_reasoning`` content parts are rendered as ``Action:`` /
    ``Thought:`` prefix lines before the ``<tool_call>`` text that
    :class:`Qwen2_5VLBaseAdapter` already produces. On the from-agent side,
    symmetric regexes parse those lines back into structured parts.

    Decomposed only — same contract as :class:`Qwen3VLUseAdapter`
    (opaque/verbatim turns use a passthrough adapter, never a branch here).

    The official Qwen2.5-VL OSWorld reference script uses NO "# Response
    format" rules block — only "You are a helpful assistant" + tools section.
    The default ``system_prompt=None`` matches this. The model learns the
    Action: wire format from SFT data, not from system instructions.
    """

    enable_inline_reasoning: bool = False

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Render ``action_description`` (+ optional ``inline_reasoning``) parts
        as ``Action:`` / ``Thought:`` prefix lines before the ``<tool_call>``
        text that base already produces."""
        result = super()._convert_message_to_agent(message, **kwargs)
        if result.get("role") != "assistant":
            return result
        content = message.get("content") or []
        nav_lines: list[str] = []
        if self.enable_inline_reasoning:
            nav_lines += [
                f"Thought: {p['text']}" for p in content
                if p.get("type") == "inline_reasoning" and p.get("text")
            ]
        nav_lines += [
            f"Action: {p['text']}" for p in content
            if p.get("type") == "action_description" and p.get("text")
        ]
        if nav_lines:
            existing_text = ""
            for part in result.get("content") or []:
                if part.get("type") == "text" and part.get("text"):
                    existing_text = part["text"]
                    break
            combined = "\n".join(nav_lines)
            if existing_text:
                combined = combined + "\n" + existing_text
            result["content"] = [{"type": "text", "text": combined}]
        return result

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """Parse the ``Action:`` line into ``action_description`` and, when
        :attr:`enable_inline_reasoning`, the ``Thought:`` line into
        ``inline_reasoning``.

        The retag applies only to a turn that actually carries ``tool_calls``.
        A no-tool-call turn is the text-final path, so its prose must remain a
        plain ``text`` part for ``no_tool_call_final_text`` and DAgger round-trip
        replay.

        Tool calls are already parsed by ``parse_raw_assistant_response``
        (inherited from base) and handled by base's ``convert_message_from_agent``.
        """
        result = super().convert_message_from_agent(message, **kwargs)
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
            inline_reasoning = ""
            if self.enable_inline_reasoning:
                # Stop at the ``\nAction:`` line, not the first ``\n`` — the inline
                # reasoning is multi-line (matches the to-agent render above, which
                # emits the full body verbatim as ``Thought: <text>``). ``\Z`` (not
                # ``$``) is the fallback so a multi-line body is not clipped.
                m = re.search(r"Thought:\s*(.*?)(?:\n(?=Action:)|\Z)", raw_text, re.DOTALL)
                if m:
                    inline_reasoning = m.group(1).strip()
            # Capture the action body from the LAST ``Action:`` line to end-of-
            # string (symmetric with Thought, so a multi-line action_description
            # round-trips intact). The greedy ``.*\n`` skips any ``Action:``-prefixed
            # line nested inside the thought body and anchors on the FINAL one.
            m = re.search(r"(?:.*\n)?Action:\s*(.*)\Z", raw_text, re.DOTALL)
            if m:
                action_text = m.group(1).strip()
            else:
                action_text = next(
                    (ln.strip() for ln in raw_text.splitlines() if ln.strip()),
                    raw_text.strip(),
                )
            parts = make_assistant_content(
                inline_reasoning=inline_reasoning, action_description=action_text,
            )
        result["content"] = parts
        return result

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Use-aware render_step with per-message image size caching.

        Unlike the base ``render_step`` which caches ``_current_image_size``
        once from the last image, use has multiple turns each with its
        own screenshot. We update ``_current_image_size`` before converting
        each message so coordinate rescaling uses the correct image dims.
        """
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        # Build system message (same as base).
        # Use the last image in the window for the tools section placeholder substitution.
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


@dataclasses.dataclass
class Qwen2_5VLDesktopUseAdapter(
    Qwen2_5VLUseAdapter, key=r"qwen2_5_vl@(desktop|browser)@use"
):
    """Desktop+browser ``use`` (multi-step rollout): summarized history, full action vocab.

    Matches the official Qwen2.5-VL OSWorld reference script:
    - 11-action ``computer_use`` enum (same as reference's ``action_space="pyautogui"``)
    - ``history_n=4`` rolling window with text summary of older turns
    - ``system_prompt=None`` — no "# Response format" block (reference uses
      only "You are a helpful assistant" + tools section)

    Reference: upstream OSWorld ``mm_agents/qwen25vl_agent.py``.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen2_5VLDesktopActionSpace
    )
    protocol: Qwen3VLHistoryProtocol = dataclasses.field(
        default_factory=lambda: Qwen3VLHistoryProtocol(full_history_size=4)
    )
    system_prompt: str | None = None


# =============================================================================
# Desktop + Browser Grounding Adapters
# =============================================================================

@dataclasses.dataclass
class Qwen2_5VLDesktopGroundingPointAdapter(
    Qwen2_5VLBaseAdapter, key=r"qwen2_5_vl@(desktop|browser)@grounding\.point"
):
    """Desktop+browser grounding (single-step click) for Qwen2.5-VL."""
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen2_5VLDesktopGroundingPointActionSpace
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


@dataclasses.dataclass
class Qwen2_5VLDesktopGroundingActionAdapter(
    Qwen2_5VLBaseAdapter, key=r"qwen2_5_vl@(desktop|browser)@grounding\.action"
):
    """Desktop+browser grounding/action (full schema, single turn) for SFT replay."""
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen2_5VLDesktopActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )


# =============================================================================
# Mobile Adapters
# =============================================================================

@dataclasses.dataclass
class Qwen2_5VLMobileGroundingPointAdapter(
    Qwen2_5VLBaseAdapter, key="qwen2_5_vl@mobile@grounding.point"
):
    """Mobile grounding (single-step tap)."""
    system_prompt: str | None = GROUNDING_POINT_SYSTEM_PROMPT
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen2_5VLMobileGroundingPointActionSpace
    )
    metadata: LiteCUAMetadata = dataclasses.field(
        default_factory=lambda: LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.MOBILE.value,
                LiteCUAMetadata.TaskType.GROUNDING_POINT.value,
            )
        )
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )


@dataclasses.dataclass
class Qwen2_5VLMobileGroundingActionAdapter(
    Qwen2_5VLBaseAdapter, key="qwen2_5_vl@mobile@grounding.action"
):
    """Mobile grounding/action (full schema, single turn) for SFT replay."""
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen2_5VLMobileActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )


@dataclasses.dataclass
class Qwen2_5VLMobileUseAdapter(
    Qwen2_5VLUseAdapter, key="qwen2_5_vl@mobile@use"
):
    """Mobile ``use`` adapter for Qwen2.5-VL.

    Mirrors :class:`Qwen3VLMobileUseAdapter` in shape but uses the
    Qwen2.5-VL mobile wire format:

      * 9-action ``mobile_use`` tool (``key``, ``click``, ``long_press``,
        ``swipe``, ``type``, ``system_button``, ``open``, ``wait``,
        ``terminate``) — matches upstream ``MobileUse`` byte-for-byte.
      * Pixel coords in smart-resized image space (adapter rescales to/from
        cua-lite [0, 1000]).
      * ``<tool_call>{json}</tool_call>`` text serialization (Qwen2.5-VL
        chat_template drops structured ``tool_calls``).
      * ``system_prompt=None`` — no "# Response format" block, same as
        the desktop ``use`` adapter.

    ``smart_resize_enabled=True`` (matching the base + desktop adapters): the
    adapter snaps each screenshot to 28-px multiples *before* it reaches sglang,
    so the dims told to the model (``{display_width_px}x{display_height_px}``)
    and used for [0,1000]↔pixel rescaling **exactly equal** the grid Qwen2.5-VL's
    own image processor feeds the vision tower (which always smart_resizes to
    28-px multiples internally). Disabling it would tell the model e.g.
    ``1080x2400`` while the processor actually sees ``1092x2408`` — a small but
    real told-vs-perceived coordinate skew.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen2_5VLMobileActionSpace
    )
    protocol: Qwen3VLHistoryProtocol = dataclasses.field(
        default_factory=lambda: Qwen3VLHistoryProtocol(full_history_size=4)
    )
    system_prompt: str | None = None
    smart_resize_enabled: bool = True

    def _extra_tool_schemas_for_prompt(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(schema)
            for schema in self.metadata.extra_tool_schemas
            if tool_schema_name(schema) == "open_app"
        ]


# =============================================================================
# Pass-through Adapters (understanding, bbox)
# =============================================================================

AgentAdapterRegistry.register(r"qwen2_5_vl@(desktop|browser|mobile)@understanding", AsIsAdapter)
AgentAdapterRegistry.register(r"qwen2_5_vl@(desktop|browser|mobile)@grounding\.bbox", AsIsAdapter)
