"""
MAI-UI Adapter

Provides two MAI-UI adapters, one per task type (each a single concrete class —
MAI-UI needs no per-platform base/subclass split):

    MAIUIMobileUseAdapter   (mobile ``use``, MAIUIHistoryProtocol)
    MAIUIGroundingPointAdapter     (desktop/browser/mobile grounding, single-step)

Reference: ${CUA_LITE_REFERENCES_ROOT}/MAI-UI/src/prompt.py and
${CUA_LITE_REFERENCES_ROOT}/MAI-UI/src/mai_naivigation_agent.py.

Notable design points (verified by inspecting the local Tongyi-MAI checkpoints):

1. **Thinking tags are PLAIN BPE TEXT, not the special `<think>` token.**
   The model emits ``<thinking>`` and ``</thinking>`` (note the ``ing``)
   as 3-token BPE pieces, not as Qwen3's native reasoning sentinel
   ``<think>`` (token id 151667). The adapter parses ``<thinking>`` content
   into an ``InlineReasoningContent`` part (non-native, prompted CoT — per the
   LiteAssistantMessage type contract, only Qwen3-VL's native ``<think>``
   uses the top-level ``reasoning_content`` slot). ``MAIUIMobileUseAdapter.
   convert_message_to_agent`` folds it back into a ``<thinking>...</thinking>``
   text block at render time, so the agent class itself needs no override
   (``MAIUIMobileAgent`` body is ``pass``).

2. **No `tools=` to apply_chat_template.** MAI-UI's chat_template has a
   ``{% if tools %}`` branch that would inject a Qwen-style ``<tools>``
   schema block. MAI-UI was SFT'd with the action space hand-baked into the
   system prompt as JSON examples, not via chat_template tools injection.
   The agent override forces the ``else`` branch by never passing ``tools=``.

3. **Scoped extra-tool prompt slot.** ``open`` / ``answer`` / ``terminate``
   are always advertised by the native Action Space block, so an active extra
   of that name never duplicates into the MCP block. Non-native env extras use
   the upstream MCP block.

4. **Swipe Option A** — see :mod:`lite.agents.models.mai_ui.action_space` for the
   direction-only forward / synthesized-endpoints reverse mapping.

Usage:
    from lite.agents.models.mai_ui.adapter import MAIUIMobileUseAdapter

    adapter = MAIUIMobileUseAdapter()
    agent_sample = adapter.unroll(sample)
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import re
from typing import Any, ClassVar

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.adapter import (
    BaseAgentAdapter,
)
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.models.mai_ui.action_space import (
    MAIUIGroundingPointActionSpace,
    MAIUIMobileActionSpace,
)
from lite.agents.models.mai_ui.protocol import MAIUIHistoryProtocol
from lite.agents.types import AgentMessage, AgentStep
from lite.core import (
    LiteCUAMetadata,
    LiteMessage,
    LiteSample,
)
from lite.core.messages import get_inline_reasoning, make_assistant_content
from lite.core.messages.final import MODEL_OUTPUT_ERROR_KEY, mark_model_output_error
from lite.core.messages.turns import truncate_sample_to_turn
from lite.core.tools.calls import (
    tool_call_arguments,
    tool_call_name,
)
from lite.core.tools.extra_tools import (
    LiteAppLaunchToolSet,
    LiteFinishToolSet,
    open_app_names_from_metadata,
)
from lite.core.tools.schemas import tool_schema_name
from lite.utils.image import smart_resize

logger = logging.getLogger(__name__)


def _loads_object(raw: str) -> dict[str, Any] | None:
    """``json.loads`` narrowed to a JSON *object*: anything else is ``None``."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_mai_tool_call_json(raw: str) -> dict[str, Any] | None:
    """Parse a MAI-UI ``<tool_call>`` JSON body, with a repair pass for the
    model's well-known coordinate-array quirk: it closes ``"coordinate":[N,N``
    with ``}`` (+ arbitrary trailing junk like ``}"}}``, ``})}``, ``}}|}``)
    instead of ``]}}``.

    Returns the parsed dict, or ``None`` if even the repair fails.

    The return type is a CONTRACT, not a hint: a ``<tool_call>`` body is a JSON
    *object* or it is not a tool call. A bare ``json.loads`` also accepts every
    other JSON top level (``[1,2]``, ``"s"``, ``3``, ``null``), so the
    non-object cases are folded into the ``None`` failure result HERE rather
    than being re-checked at each call site.
    """
    parsed = _loads_object(raw)
    if parsed is not None:
        return parsed
    # Coordinate repair: truncate after the second number and close cleanly.
    repaired = re.sub(
        r'("coordinate"\s*:\s*\[\s*-?\d+\s*,\s*-?\d+).*$',
        r"\1]}}",
        raw,
        flags=re.DOTALL,
    )
    if repaired != raw:
        parsed = _loads_object(repaired)
        if parsed is not None:
            logger.info("Repaired MAI-UI tool_call JSON: %s -> %s", raw, repaired)
            return parsed
    logger.warning("Failed to parse MAI-UI tool_call JSON: %s", raw)
    return None


# =============================================================================
# System prompt — derived from MAI-UI reference (prompt.py:18-50)
# =============================================================================
#
# Template is byte-faithful to MAI-UI's SFT distribution
# (``${CUA_LITE_REFERENCES_ROOT}/MAI-UI/src/prompt.py:18-50``) EXCEPT
# for the "Available Apps" hint — that pair of lines is env-specific and gets
# injected by ``_render_mai_mobile_sys_prompt`` from the active ``open_app``
# schema enum (see ``MAIUIMobileUseAdapter.__post_init__``).
# The adapter itself stays env-agnostic — it never hard-codes an
# androidworld / mobilegym / androidlab default. ``_render`` uses
# ``json.dumps(separators=(",", ":"))`` so when an env supplies the
# original androidworld apps list, the rendered prompt is byte-equal
# to MAI-UI's SFT distribution.
#
# The ``## Action Space`` rows are the SFT text itself, not a rendered tools
# section, so no gate edits them — neither the sample's active extra tools nor
# ``metadata.valid_actions``. Every row stays in the block whatever
# ``metadata.extra_tool_schemas`` and ``metadata.valid_actions`` hold.
# Withholding a trained row moves the model off the distribution it was
# fine-tuned on, and reachability is env ingress's question to answer — a call
# to an inactive tool comes back as model-visible feedback keyed to its call id.

_MAI_MOBILE_SYS_PROMPT_BASE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>
```

## Action Space

{"action": "click", "coordinate": [x, y]}
{"action": "long_press", "coordinate": [x, y]}
{"action": "type", "text": ""}
{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]} # "coordinate" is optional. Use the "coordinate" if you want to swipe a specific UI element.
{"action": "open", "text": "app_name"}
{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
{"action": "system_button", "button": "button_name"} # Options: back, home, menu, enter
{"action": "wait"}
{"action": "terminate", "status": "success or fail"}
{"action": "answer", "text": "xxx"} # Use escape characters \\', \\", and \\n in text part to ensure we can parse the text in normal python string format.


## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part."""

_MAI_MOBILE_SYS_PROMPT_APPS_LINE = (
    "- Available Apps: `{apps_list}`.\n"
    "You should use the `open` action to open the app as possible as you can,"
    " because it is the fast way to open the app."
)

_MAI_MOBILE_SYS_PROMPT_FOOTER = (
    "- You must follow the Action Space strictly, and return the correct"
    " json object within <thinking> </thinking> and <tool_call></tool_call> XML tags."
)


def _render_mai_mobile_sys_prompt(apps: list[str] | None) -> str:
    """Render MAI-UI's mobile system prompt with the given apps list.

    ``apps`` is the ``open_app`` enum the env advertises for this sample. It is
    per-sample DATA rather than trained text — the two "Available Apps" lines
    are the one part of the template that is not byte-faithful to the SFT
    distribution — so they render exactly when the env supplied an enum and are
    simply absent when it did not. Everything else, the ``## Action Space`` rows
    included, is substituted whole.

    Uses ``json.dumps(separators=(",", ":"))`` so that when an env
    supplies the original androidworld apps list the rendered prompt
    is byte-equal to MAI-UI's SFT distribution.
    """
    parts = [_MAI_MOBILE_SYS_PROMPT_BASE]
    if apps:
        apps_repr = json.dumps(list(apps), ensure_ascii=False, separators=(",", ":"))
        parts.append(_MAI_MOBILE_SYS_PROMPT_APPS_LINE.format(apps_list=apps_repr))
    parts.append(_MAI_MOBILE_SYS_PROMPT_FOOTER)
    return "\n".join(parts)


# The env-agnostic render: full trained Action Space block, no apps lines. This
# is what a default-constructed adapter carries, and what every adapter whose
# env advertises no ``open_app`` enum renders regardless of its active extras.
MAI_MOBILE_SYS_PROMPT = _render_mai_mobile_sys_prompt(None)


# =============================================================================
# MAI-UI Mobile Base Adapter
# =============================================================================

@dataclasses.dataclass
class MAIUIMobileUseAdapter(BaseAgentAdapter, key="mai_ui@mobile@use"):
    """
    Adapter for MAI-UI mobile, ``use`` mode (multi-step rollout).

    Default protocol: ``MAIUIHistoryProtocol(full_history_size=3)`` — matches
    upstream ``history_n=3``. MAI-UI is a mobile-only ``use`` model, so this
    is a single concrete adapter (no per-platform base/subclass split — unlike
    Qwen, which shares one base across mobile/desktop/browser ``use`` keys).

    Responsibilities:
      - Build a system message containing ``MAI_MOBILE_SYS_PROMPT`` plus the
        env-specific app-name hint and non-native MCP extras, if any.
      - Apply the protocol (default: MAIUIHistoryProtocol).
      - Translate cua-lite mobile tool_calls <-> MAI-UI ``mobile_use`` shape
        via ``MAIUIMobileActionSpace`` (the action_space handles the
        coordinate rescale 999<->1000 and the swipe direction synthesis).
      - Parse raw model output (``<thinking>`` + ``<tool_call>``) into a
        cua-lite assistant message, storing the thinking text as an
        ``InlineReasoningContent`` part (non-native CoT) and the parsed
        tool call in ``tool_calls``.

    MAI-UI's native Action Space JSON block is trained text: every line is
    advertised whatever ``metadata.valid_actions`` and
    ``metadata.extra_tool_schemas`` hold. Non-native ``extra_tool_schemas``
    render through MAI-UI's existing MCP prompt slot; the natively-advertised
    ``open`` / ``answer`` / ``terminate`` never duplicate into it.
    """

    system_prompt: str | None = None
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=MAIUIMobileActionSpace
    )
    protocol: MAIUIHistoryProtocol = dataclasses.field(
        default_factory=lambda: MAIUIHistoryProtocol(full_history_size=3)
    )
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = (
        LiteAppLaunchToolSet.get_tool_names() | LiteFinishToolSet.get_tool_names()
    )
    # ``system_prompt`` defaults to None so ``__post_init__`` renders the prompt
    # with the env-specific apps list; yaml callers can pin ``agent_kwargs.system_prompt``.

    def __post_init__(self):
        super().__post_init__()  # extra_tool_schemas collision check lives there
        # Render system_prompt from the active ``open_app`` schema enum unless
        # caller pinned one explicitly via ``agent_kwargs.system_prompt``.
        if self.system_prompt is None:
            apps = open_app_names_from_metadata(self.metadata)
            self.system_prompt = _render_mai_mobile_sys_prompt(apps)

    # -- system prompt assembly ----------------------------------------------

    def _standalone_extra_names(self) -> frozenset[str]:
        """Active extras that still need the MCP block.

        The Action Space block always advertises ``open`` / ``answer`` /
        ``terminate``, so the extras those rows spell are already on the model's
        surface and must not be duplicated into the MCP block.
        """
        return self.active_extra_tool_names() - self.natively_rendered_extra_tool_names

    def _mcp_extra_tool_schemas(self) -> list[dict[str, Any]]:
        standalone_extra_names = self._standalone_extra_names()
        return [
            schema for schema in self.metadata.extra_tool_schemas
            if tool_schema_name(schema) in standalone_extra_names
        ]

    def _build_tools_section(self) -> str:
        """Render non-native extras through MAI-UI's existing MCP block."""
        extra_tools = self._mcp_extra_tool_schemas()
        if not extra_tools:
            return ""
        tools_json = "\n".join(json.dumps(schema, ensure_ascii=False) for schema in extra_tools)
        return (
            "## MCP Tools\n"
            "You are also provided with MCP tools, you can use them to complete the task.\n"
            f"{tools_json}\n\n"
            "If you want to use MCP tools, you must output as the following format:\n"
            "<thinking>\n"
            "...\n"
            "</thinking>\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

    def _build_system_text(self) -> str:
        """Concatenate the native prompt with non-native MCP extras."""
        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        mcp = self._build_tools_section()
        if mcp:
            parts.append(mcp)
        return "\n\n".join(parts)

    # -- sample <-> agent ----------------------------------------------------

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render turn ``k`` in MAI-UI wire format.

        Steps:
          1. Truncate sample to turn ``k``.
          2. Apply protocol (default MAIUIHistoryProtocol — splits first
             user into text-only + image-only and applies windowing).
          3. Prepend the MAI-UI system prompt (+ MCP block if any).
          4. Translate each remaining message via convert_message_to_agent.

        ``process_image`` defaults to identity — MAI-UI's reference sends
        raw PNGs (mai_naivigation_agent.py:350-393, no resize step).
        """
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        result_messages: list[AgentMessage] = []
        system_text = self._build_system_text()
        result_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        })

        for msg in messages:
            result_messages.append(self.convert_message_to_agent(msg))

        return result_messages

    # -- per-message conversion ----------------------------------------------

    def _tool_calls_to_agent_ordered(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        standalone_extra_names = self._standalone_extra_names()
        out: list[dict[str, Any]] = []
        for tc in tool_calls:
            if self._admits_active_extra_tool_call(
                tc,
                allowed_names=standalone_extra_names,
            ):
                out.append({
                    "name": tool_call_name(tc),
                    "arguments": tool_call_arguments(tc),
                })
            else:
                out.extend(self.action_space.convert_tool_calls_to_agent([tc]))
        return out

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert a single cua-lite message to MAI-UI's wire format.

        For assistants: fold reasoning + content + tool_calls into the
        SFT-trained ``<thinking>...</thinking>\\n<tool_call>{json}</tool_call>``
        plain text and drop the structured ``tool_calls`` / ``reasoning_content``
        fields. The Qwen2-VL chat_template would otherwise (a) silently drop
        ``reasoning_content``, and (b) render ``tool_calls`` with
        ``tojson`` (spaces between separators) — MAI-UI's upstream
        ``mem2response`` uses ``json.dumps(..., separators=(",", ":"))``
        (no spaces), so we serialize ourselves to match the SFT distribution
        byte-for-byte.

        ``<thinking>`` is PLAIN BPE TEXT, NOT the special tokens
        ``<think>``/``</think>`` (ids 151667/151668) used by Qwen3-VL's native
        reasoning channel. Keep the ``ing`` suffix.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        thinking = get_inline_reasoning(message)
        tool_calls = self._tool_calls_to_agent_ordered(
            message.get("tool_calls") or []
        )

        # MAI_MOBILE_SYS_PROMPT specifies <thinking>...</thinking> +
        # <tool_call>{json}</tool_call>. No prose / Action slot exists in
        # the SFT distribution, so we do not render plain text or
        # action_description content.
        blocks: list[str] = []
        if thinking:
            blocks.append(f"<thinking>\n{thinking}\n</thinking>")
        for tc in tool_calls:
            tc_json = json.dumps(
                {"name": tc["name"], "arguments": tc["arguments"]},
                separators=(",", ":"),
            )
            blocks.append(f"<tool_call>\n{tc_json}\n</tool_call>")

        had_tool_calls = bool(message.get("tool_calls"))
        full_text = "\n".join(blocks)
        result.pop("reasoning_content", None)
        result.pop("tool_calls", None)
        if full_text:
            result["content"] = [{"type": "text", "text": full_text}]
        elif not had_tool_calls:
            # Content-only final turn: no action to narrate. Keep ONLY the plain
            # ``text`` parts -- the canonical "Done." this policy writes -- so the
            # turn does not become an empty SFT target. Non-``text`` parts still
            # drop: this format has no prose slot for them, and the data layer
            # guarantees a no-tool-call final carries exactly one ``text`` part.
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
        """Convert AgentMessage → LiteMessage. Parses the system-prompt-level
        ``<thinking>...</thinking>`` tag (SFT-trained BPE tag; NOT Qwen's
        native ``<think>`` channel) into an ``InlineReasoningContent`` part.
        Chat-template tokens (``<tool_call>``) were already extracted by
        :meth:`parse_raw_assistant_response`.

        Tool calls go through ``BaseAgentAdapter._route_agent_tool_calls_to_lite``:
        an active ``extra_tool_schemas`` name whose arguments match that schema's
        key shape passes through as a standalone tool, and everything else is
        converted by ``MAIUIMobileActionSpace``.

        Tolerance: accepts ``</think>`` as a synonym for ``</thinking>``
        (matches mai_naivigation_agent.py:76-79 — tolerates a native-thinking
        base model under this harness).
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        if "tool_calls" in result:
            # An active env extra claims a call by name AND argument shape;
            # every other call goes to ``MAIUIMobileActionSpace``, which owns the
            # ``mobile_use`` dispatch, the flat-native reading (wrapper dropped,
            # action value used as the tool name) and the standalone fallback.
            converted = self._route_agent_tool_calls_to_lite(result["tool_calls"])
            if result["tool_calls"] and not converted:
                mark_model_output_error(
                    result,
                    "tool call did not satisfy the active tool schema or native action grammar",
                )
            result["tool_calls"] = converted

        raw_text = ""
        for part in message.get("content") or []:
            if part.get("type") == "text" and part.get("text"):
                raw_text = part["text"]
                break

        if raw_text:
            if not result.get("tool_calls") and MODEL_OUTPUT_ERROR_KEY not in result:
                result["content"] = [{"type": "text", "text": raw_text}]
                return result

            text = raw_text
            if "</think>" in text and "</thinking>" not in text:
                text = text.replace("</think>", "</thinking>")
                if "<thinking>" not in text:
                    text = "<thinking>" + text
            m = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
            thinking_text = m.group(1).strip().strip('"') if m else ""
            result["content"] = make_assistant_content(inline_reasoning=thinking_text)
        else:
            result["content"] = []
        return result

    # -- raw response parser -------------------------------------------------

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """Parse a raw MAI-UI assistant response, extracting only the
        chat-template-level ``<tool_call>`` token (rendered by Qwen2.5-VL's
        chat_template from the ``tool_calls`` field, and baked into MAI-UI's
        SFT distribution). The prose remainder (including the SFT-protocol
        ``<thinking>`` tag) goes verbatim into a single text content part;
        system-prompt parsing happens in :meth:`convert_message_from_agent`.
        """
        result: AgentMessage = {"role": "assistant"}

        tool_calls: list[dict[str, Any]] = []
        for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL):
            raw = m.group(1).strip()
            tc_json = _parse_mai_tool_call_json(raw)
            if tc_json is None:
                continue
            # No ``isinstance(tc_json, dict)`` rescue here: the parser's
            # ``dict | None`` return type is now enforced, so a non-object body
            # (``[1,2]``, ``"s"``, ``3``) already came back as ``None`` above.
            name = tc_json.get("name")
            arguments = tc_json.get("arguments")
            # This check is a DIFFERENT contract and stays: the parser
            # guarantees a JSON object, not that the object carries a ``str``
            # name and a ``dict`` arguments (``{"foo": 1}`` parses fine).
            if not isinstance(name, str) or not isinstance(arguments, dict):
                logger.warning("Ignoring non-flat MAI-UI tool_call JSON: %s", tc_json)
                continue
            tool_calls.append({"name": name, "arguments": arguments})
        if tool_calls:
            result["tool_calls"] = tool_calls
        elif "<tool_call>" in response or "</tool_call>" in response:
            mark_model_output_error(result, "malformed <tool_call> JSON")

        clean = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL).strip()
        if clean:
            result["content"] = [{"type": "text", "text": clean}]

        return result


# =============================================================================
# Grounding (single-step click) system prompt + adapter
# =============================================================================
#
# Verbatim from MAI-UI src/prompt.py:137-148 (``MAI_MOBILE_SYS_PROMPT_GROUNDING``).
# Despite the ``MOBILE`` in the upstream symbol name, the cookbook uses this
# for desktop + mobile + browser grounding evals (ScreenSpot-Pro / OSWorld-G /
# UI-Vision / MMBench), so we register it under all 3 platform keys.
MAI_GROUNDING_SYS_PROMPT = """You are a GUI grounding agent.
## Task
Given a screenshot and the user's grounding instruction. Your task is to accurately locate a UI element based on the user's instructions.
First, you should carefully examine the screenshot and analyze the user's instructions,  translate the user's instruction into a effective reasoning process, and then provide the final coordinate.
## Output Format
Return a json object with a reasoning process in <grounding_think></grounding_think> tags, a [x,y] format coordinate within <answer></answer> XML tags:
<grounding_think>...</grounding_think>
<answer>
{"coordinate": [x,y]}
</answer>""".strip()


@dataclasses.dataclass
class MAIUIGroundingPointAdapter(
    BaseAgentAdapter, key=r"mai_ui@(desktop|browser|mobile)@grounding\.point"
):
    """MAI-UI grounding adapter (single-step click) — desktop / browser / mobile.

    MAI-UI ships ONE grounding harness shared across all three platforms (the
    upstream cookbook applies the same prompt to ScreenSpot-Pro / OSWorld-G /
    mobile benchmarks), so this is a single concrete adapter under a regex key —
    no base/subclass split.

    Wire format is **completely different** from the ``use``
    ``<thinking>`` + ``<tool_call>`` shape:

      * Reasoning lives in ``<grounding_think>...</grounding_think>``
        (note ``grounding_think``, NOT ``thinking``).
      * The output is a JSON literal inside ``<answer>...</answer>``,
        not a Qwen-style ``<tool_call>`` block.
      * No tool schemas are advertised — the format lives entirely in the
        system prompt; the chat_template's ``{% if tools %}`` branch is
        intentionally skipped (same approach as use; see the agent
        override on :class:`MAIUIMobileAgent` for the rationale).

    Single-turn full history; one regex-registered class serves all platforms.
    """

    system_prompt: str | None = MAI_GROUNDING_SYS_PROMPT
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=MAIUIGroundingPointActionSpace
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
    # Smart-resize the screenshot to a fixed grounding pixel budget BEFORE sending.
    # Without it the served model's image processor applies its own default
    # (MAI-UI ``preprocessor_config`` ``longest_edge`` = max_pixels = 16_777_216 ≈
    # 16.78MP), so a 4K ScreenSpot-Pro shot is processed near-native. That is fine
    # in isolation, but the resulting score isn't comparable to the upstream eval,
    # which caps the budget. The resize MATH follows the official MAI-UI grounding
    # harness (``evaluation/grounding/models/MAI_UI.py``): ``factor = patch16 ×
    # merge2 = 32``, ``min_pixels = 16²×4``. ``max_pixels`` has NO single official
    # value (``MAI_UI.py`` code default is unlimited; ``eval_local.py`` 2.1MP;
    # README ``--max_pixels 6553600``) — we take the README's 6.5MP; override via
    # ``agent_kwargs.smart_resize_max_pixels``.
    # Coords are RELATIVE [0, 999] (see action_space), so the resize never shifts
    # the answer frame — it only controls how much detail the model sees.
    # Use needs no equivalent: phone screenshots are well under the budget.
    smart_resize_factor: int = 16 * 2
    smart_resize_min_pixels: int = 16 * 16 * 4
    smart_resize_max_pixels: int = 6_553_600

    def _process_image_after_target(self, img):
        """Stage-2 hook: cap the screenshot at the MAI-UI grounding pixel budget.

        Downscales on a 32-px grid to ``smart_resize_max_pixels`` so a high-res
        (e.g. 4K) grounding image matches the upstream eval budget instead of the
        served processor's 16.78MP default. Safe for coords (relative [0, 999]).
        Override the cap via ``agent_kwargs.smart_resize_max_pixels`` in the yaml.
        """
        w, h = img.size
        new_h, new_w = smart_resize(
            height=h, width=w,
            factor=self.smart_resize_factor,
            min_pixels=self.smart_resize_min_pixels,
            max_pixels=self.smart_resize_max_pixels,
        )
        if (new_w, new_h) != (w, h):
            img = img.resize((new_w, new_h))
        return img

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        result_messages: list[AgentMessage] = []
        if self.system_prompt:
            result_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            })
        for msg in messages:
            result_messages.append(self.convert_message_to_agent(msg))
        return result_messages

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Render an assistant message in the grounding wire format.

        Folds an ``InlineReasoningContent`` (if present) into
        ``<grounding_think>...</grounding_think>`` and the first
        ``LitePointActionSpace.point(coord)`` tool call into
        ``<answer>\n{"coordinate":[x,y]}\n</answer>``. Used for SFT-style
        replay — at env-eval time the model emits this form natively and
        we parse it back out via :meth:`parse_raw_assistant_response`.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        thinking = get_inline_reasoning(message) or ""
        # Run tool_calls through the grounding action space
        # (cua-lite point → synthetic ``answer(coordinate=...)``).
        coord_payload = None
        if "tool_calls" in result:
            for tc in result["tool_calls"]:
                args = tool_call_arguments(tc)
                if tool_call_name(tc) == "point" and "coordinate" in args:
                    coord_payload = args["coordinate"]
                    break

        parts = []
        if thinking:
            parts.append(f"<grounding_think>\n{thinking}\n</grounding_think>")
        if coord_payload is not None:
            parts.append(
                f'<answer>\n{{"coordinate": {json.dumps(list(coord_payload))}}}\n</answer>'
            )
        text = "\n".join(parts) if parts else ""
        result["content"] = [{"type": "text", "text": text}] if text else []
        result.pop("tool_calls", None)
        result.pop("reasoning_content", None)
        return result

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """Parse ``<grounding_think>``+``<answer>`` text → cua-lite assistant.

        Looks at the first text content part, extracts the inline reasoning
        and coordinate, then synthesizes a ``LitePointActionSpace.point``
        tool call. If no ``<answer>`` block is found, returns content-only
        with just the reasoning surface.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        raw_text = ""
        for part in message.get("content") or []:
            if part.get("type") == "text" and part.get("text"):
                raw_text = part["text"]
                break

        if not raw_text:
            # API-style path (tool_calls already structured)
            if "tool_calls" in result:
                result["tool_calls"] = self._route_agent_tool_calls_to_lite(
                    result["tool_calls"]
                )
            return result

        # Extract <grounding_think>...</grounding_think>
        m_think = re.search(r"<grounding_think>(.*?)</grounding_think>", raw_text, re.DOTALL)
        thinking = m_think.group(1).strip() if m_think else ""

        # Extract <answer>{"coordinate":[x,y]}</answer>
        m_ans = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_text, re.DOTALL)
        tool_calls: list[dict[str, Any]] = []
        if m_ans:
            payload = m_ans.group(1).strip()
            try:
                obj = json.loads(payload)
                if isinstance(obj, dict) and "coordinate" in obj:
                    tool_calls = self._route_agent_tool_calls_to_lite([
                        {"name": "answer", "arguments": {"coordinate": obj["coordinate"]}}
                    ])
            except json.JSONDecodeError:
                logger.warning("Failed to parse MAI-UI grounding <answer>: %s", payload)

        result["content"] = make_assistant_content(
            inline_reasoning=thinking if thinking else "",
        )
        if tool_calls:
            result["tool_calls"] = tool_calls
        else:
            result.pop("tool_calls", None)
        return result

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """Wrap raw text verbatim — parsing happens in
        :meth:`convert_message_from_agent`.
        """
        return {"role": "assistant", "content": [{"type": "text", "text": response}]}


__all__ = [
    "MAI_MOBILE_SYS_PROMPT",
    "MAI_GROUNDING_SYS_PROMPT",
    "MAIUIMobileUseAdapter",
    "MAIUIGroundingPointAdapter",
]
