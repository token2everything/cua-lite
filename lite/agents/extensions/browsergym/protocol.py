"""BrowserGym protocols.

This module provides protocols specific to the browsergym env:

- :class:`BrowserGymGenericProtocol` (``browsergym.generic``) — mimics
  AgentLab `GenericAgent.MainPrompt` structure (rebuild-per-turn). This
  is the **shipped default** for every browsergym text+bid yaml under
  ``scripts/configs/{qwen3_5,qwen3_vl}/default/browsergym.*/``.

Why not the chat-style ``lite.history`` here?

cua-lite's chat-style protocols (``lite.history`` / ``qwen3_vl.history``
/ ``qwen3_5.history``) append each turn's user/assistant pair, so turn
N+1 is a strict byte-prefix-extension of turn N → sglang radix cache hits.
That works great when each turn's obs is small (screenshot bytes, short
text). For browsergym text+bid configs we instead surface the FULL AXTree
in every user_N message (no incremental diff — diff would be more
divergent than the rebuild). WA shopping_admin AXTrees can be ~75K
tokens, so by turn 2 a chat-style prompt already hits ~155K and overflows
Qwen3.5's 128K context window. AgentLab's rebuild-per-turn pattern keeps
historical actions in a short ``# History`` section (just ``click('a47')``
strings) and renders ONE current AXTree per turn, so each prompt stays
bounded ~80K. sglang radix cache still hits on the static system + tools
preamble; only the user-side observation diverges.

Vision-mode browsergym yamls (``default.yaml``) keep the adapter
default (``qwen3_vl.history`` / ``qwen3_5.history`` / ``lite.history``)
— screenshots are bytes, not 75K-token AXTrees, so context budget isn't
the bottleneck.

Usage:

    # Default for browsergym text+bid yamls (this protocol):
    agent_kwargs:
      protocol_key: "browsergym.generic"
      protocol_kwargs:
        use_thinking: true        # match A3 paper default
        use_hints: true
        use_concrete_example: true
        use_abstract_example: true

    # Vision/screenshot-driven browsergym configs use the chat-style default:
    agent_kwargs:
      protocol_key: "lite.history"
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from lite.agents.core.protocol import BaseProtocol
from lite.agents.extensions.browsergym.goal_image import splice_goal_images
from lite.agents.models.qwen3_5.protocol import Qwen3_5HistoryProtocol
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.core.messages import LiteMessage, extract_first_text
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.results import (
    extract_projected_tool_result_error,
    project_tool_result_text,
)

# =============================================================================
# Snapshotted strings from AgentLab's `dynamic_prompting.py`
# =============================================================================
# The literals below are lifted from upstream AgentLab
# `src/agentlab/agents/dynamic_prompting.py` and
# `agentlab/agents/generic_agent/generic_agent_prompt.py`. They are intentionally
# duplicated rather than imported so this protocol is self-contained and
# stable against AgentLab/BrowserGym version drift. Bump them deliberately
# when bumping the upstream commit.
#
# Pinned upstream reference:
#   AgentLab @ HEAD on 2026-04-30 (recent main)
# -----------------------------------------------------------------------------

# NOTE on system prompts: this protocol does NOT set the system message — it
# only rewrites the user-side context (see process_messages). The system
# message comes from two layers that compose: (1) the shipped browsergym yamls
# set ``agent_kwargs.system_prompt`` to AgentLab's `SystemPrompt` text VERBATIM
# (byte-identical to dynamic_prompting.SystemPrompt:566-570), and (2) the Qwen
# adapter's chat template appends its `# Tools` / `<tool_call>` format section
# on top. So the model sees AgentLab's semantic framing PLUS the wire-format
# instructions — system-prompt byte-parity on the AgentLab text is already
# achieved via the config, not pending. The reference text, for reference:
#
#     "You are an agent trying to solve a web task based on the content of the page and"
#     "user instructions. You can interact with the page and explore, and send messages"
#     "to the user. Each time you submit an action it will be sent to the browser and"
#     "you will receive a new page."

# GoalInstructions preamble (dynamic_prompting.py:485-489).
_INSTRUCTIONS_PREAMBLE = (
    "# Instructions\n"
    "Review the current state of the page and all other information to find the best\n"
    "possible next action to accomplish your goal. Your answer will be interpreted\n"
    "and executed by a program, make sure to follow the formatting instructions.\n"
    "\n"
    "## Goal:\n"
)

# AXTree note headers (dynamic_prompting.py:306-338).
_AXTREE_BID_INFO = (
    "Note: [bid] is the unique alpha-numeric identifier at the beginning of "
    "lines for each element in the AXTree. Always use bid to refer to "
    "elements in your actions.\n\n"
)
_AXTREE_VISIBLE_TAG_INFO = (
    "Note: You can only interact with visible elements. If the \"visible\" "
    "tag is not present, the element is not visible on the page.\n\n"
)

# Hints static block (dynamic_prompting.py:551-562).
_HINTS_PROMPT = (
    "\nNote:\n"
    "* Some tasks may be game like and may require to interact with the mouse position\n"
    "in x, y coordinates.\n"
    "* Some text field might have auto completion. To see it, you have to type a few\n"
    "characters and wait until next step.\n"
    "* If you have to cut and paste, don't forget to select the text first.\n"
    "* Coordinate inside an SVG are relative to it's top left corner.\n"
    "* Make sure to use bid to identify elements when using commands.\n"
    "* Interacting with combobox, dropdowns and auto-complete fields can be tricky,\n"
    "sometimes you need to use select_option, while other times you need to use fill\n"
    "or click and wait for the reaction of the page.\n"
)

# ActionPrompt preamble (dynamic_prompting.py:585-590).
_ACTION_SET_GENERIC_INFO = (
    "Note: This action set allows you to interact with your environment. Most of them\n"
    "are python function executing playwright code. The primary way of referring to\n"
    "elements in the page is through bid which are specified in your observations.\n\n"
)

# Think concrete example (dynamic_prompting.py:672-679).
_THINK_CONCRETE_EX = (
    "\n<think>\n"
    "From previous action I tried to set the value of year to \"2022\",\n"
    "using select_option, but it doesn't appear to be in the form. It may be a\n"
    "dynamic dropdown, I will try using click with the bid \"a324\" and look at the\n"
    "response from the page.\n"
    "</think>\n"
)

# Think abstract example (dynamic_prompting.py:666-671).
_THINK_ABSTRACT_EX = (
    "\n<think>\n"
    "Think step by step. If you need to make calculations such as coordinates, "
    "write them here. Describe the effect\n"
    "that your previous action had on the current content of the page.\n"
    "</think>\n"
)

# Unlike AgentLab's `MainPrompt`, this protocol renders no `<action>` example.
# The adapter's chat-template tools section owns the action-format example
# (`Qwen3VLBaseAdapter._build_tools_section` for JSON-tool-call models,
# `Qwen3_5BaseAdapter._build_tools_section` for XML-tool-call models) and is the
# single source of format truth.


# =============================================================================
# Protocol
# =============================================================================

@dataclasses.dataclass
class BrowserGymGenericProtocol(BaseProtocol, key="browsergym.generic"):
    """AgentLab `MainPrompt`-shape protocol — single user message per turn.

    Mirrors `agentlab.agents.generic_agent.generic_agent_prompt.MainPrompt`
    section ordering so prompts can be diffed against the paper artifacts.

    Shipped default for every browsergym text+bid yaml under
    ``scripts/configs/{qwen3_5,qwen3_vl}/default/browsergym.*/``. Vision-only
    browsergym configs use ``lite.history`` instead — see module docstring
    for the context-budget rationale.

    Mirrors `agentlab.agents.generic_agent.generic_agent_prompt.GenericPromptFlags`:

    Attributes:
        use_thinking: A3 paper default = True (emit ``<think>...</think>``).
        use_concrete_example: A3 paper default = True.
        use_abstract_example: A3 paper default = True.
        use_hints: A3 paper default = True (renders the static Hints block).
        action_describe_text: snapshot of `action_set.describe(...)` for the
            chosen subsets — passed in via env_kwargs since the protocol
            shouldn't depend on browsergym at import time. If None, omit
            the Action Space section entirely (degraded mode).
        tool_call_format: wire format for rendering ``# History`` tool_calls —
            ``"json"`` (Qwen3-VL ``<tool_call>{...}</tool_call>``, the default)
            or ``"xml"`` (Qwen3.5 ``<tool_call><function=...>...</tool_call>``).
            MUST match the paired adapter's live ``<tool_call>`` format so the
            history examples reinforce — not contradict — the format the system
            prompt + chat-template tools section demand (a JSON history under an
            XML-strict parser drifts the model off-format → every later turn
            drops to noop). Threaded via ``protocol_kwargs`` per config, like
            ``action_describe_text`` — keeps the format choice in the browsergym
            yamls instead of leaking into the shared, per-env adapter render path.
            Qwen3.5 browsergym configs set ``tool_call_format: "xml"``; Qwen3-VL
            configs keep the JSON default.
    """

    use_thinking: bool = False
    use_concrete_example: bool = True
    use_abstract_example: bool = False
    use_hints: bool = False
    action_describe_text: str | None = None
    tool_call_format: str = "json"

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs: Any,
    ) -> list[LiteMessage]:
        """Mimic AgentLab's "consolidate-history-into-single-user-message" flow.

        Input: cua-lite chat-style messages
        ``[system, user(t0), assistant(t0), user(t1), assistant(t1), ...]``.

        Output: ``[system, single_user_message]`` where:
          - ``system`` is **kept verbatim** from input (cua-lite's adapter
            already injected its standard system prompt + tool schemas;
            we don't replace it — caller wanted "standard tools").
          - ``single_user_message`` consolidates the entire history into
            one block, with the AgentLab `MainPrompt` section ordering:
            ``# Instructions / ## Goal: / # Observation / # History /
            # Action space / # Hints / # Abstract Example / # Concrete Example``.

        This matches AgentLab's `Discussion([system, human])` shape
        (always exactly 2 messages, history rebuilt every turn) — NOT
        byte-parity, but mimics the consolidation pattern. Compare to
        ``lite.history`` (chat-style append) which is KV-cache-friendly
        but doesn't match AgentLab's flow.

        Trade-off: each turn rebuilds the user message → current obs is
        placed BEFORE history → prefix diverges turn-to-turn → sglang
        radix cache misses past the static system message preamble.
        Wall-clock will be slower than ``lite.history``.
        """
        if not messages:
            return []

        render_tool_call = _TOOL_CALL_RENDERERS[self.tool_call_format]

        # Keep cua-lite's adapter-provided system message verbatim.
        # The user has been clear: "standard tools" — don't replace.
        system_msg = next((m for m in messages if m.get("role") == "system"), None)

        # Walk observation/assistant messages, extract goal + actions + final obs.
        # Each ``actions`` entry is one assistant turn's wire-format
        # ``<tool_call>...</tool_call>`` block(s), already rendered by
        # ``render_tool_call`` in the adapter's own format.
        goal: str = ""
        actions: list[str] = []
        latest_observation_group: list[LiteMessage] = []
        current_observation_group: list[LiteMessage] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", [])
            if role in {"user", "tool"}:
                # Extract textual content (first text item in this message).
                txt = extract_first_text({"role": role, "content": content})
                if role == "user" and not goal and txt:
                    # First user text is the goal/instruction. Take everything
                    # up to the first ``## `` header (preserves multi-line
                    # goals like VWA's ``"...\nInput image 1/N below..."``)
                    # but stops before the AXTree / HTML obs body.
                    goal = txt.split("\n## ")[0].strip()
                # Track the latest observation group. Turn-0/reset observations
                # are role user; per-call post-action results are consecutive
                # role tool messages after an assistant turn.
                current_observation_group.append(msg)
                latest_observation_group = list(current_observation_group)
            elif role == "assistant":
                current_observation_group = []
                # Build the history block from the STRUCTURED ``tool_calls`` — the
                # PARSED action the adapter already dispatched to BrowserGym —
                # rendered by ``render_tool_call`` into the SAME wire format the
                # adapter emits live (JSON ``<tool_call>{...}</tool_call>`` for
                # Qwen3-VL, XML ``<tool_call><function=...>...</tool_call>`` for
                # Qwen3.5). This matches the format the system prompt + chat-template
                # tools section demand, so the ``# History`` examples reinforce —
                # rather than contradict — the live format (the format-drift bug:
                # rendering the bare BrowserGym Python call ``click('227')`` inside
                # ``<tool_call>`` tags taught the model a non-JSON format, which the
                # JSON-strict parser then dropped to noop).
                #
                # Reading STRUCTURED tool_calls (never ``raw_response.text``) is also
                # what keeps history bounded:
                #   * a noop turn's raw text is a 25-31KB reasoning blob (model ran
                #     past the token cap before emitting an action) — never read here,
                #     so it can't bloat every later turn's ``# History`` (Bug B);
                #   * empty tool_calls ⇒ no parseable action (a noop turn) ⇒
                #     ``action_str=""`` is dropped by the ``if action_str`` guard
                #     below — exactly as an action-less assistant message always was.
                tcs = msg.get("tool_calls") if isinstance(msg, dict) else None
                blocks = [
                    render_tool_call(tc) for tc in (tcs or []) if isinstance(tc, dict)
                ]
                action_str = "\n".join(b for b in blocks if b)
                # No <think> stripping here. Two cases by design:
                #   (a) `-Thinking` model variants emit <think> blocks → the
                #       Qwen3-VL-Thinking chat_template auto-strips them from
                #       historical messages (only the LATEST turn renders <think>).
                #   (b) `-Instruct` model variants are not configured with
                #       enable_thinking; the system prompt does not request <think>;
                #       so the model doesn't emit it in the first place.
                # Either way, no protocol-level intervention is needed.
                if action_str:
                    actions.append(action_str)

        final_observation_msg = _latest_browsergym_payload_message(latest_observation_group)
        final_observation_text = extract_first_text(final_observation_msg)
        action_err = _projected_tool_result_errors(latest_observation_group)

        # Compose the AgentLab MainPrompt-style single user message.
        sections: list[str] = []
        sections.append(_render_instructions(goal))

        # The "current observation" is what the latest user/tool message carried.
        # We split out the AXTree from the leading instruction text using a
        # naive heuristic — anything after the first "## AXTree:" header is
        # the AXTree. Same for HTML.
        ax_block = _extract_section_after(final_observation_text, "## AXTree:")
        html_block = _extract_section_after(final_observation_text, "## HTML:")
        tabs_block = _extract_section_after(final_observation_text, "## Currently open tabs:")
        # Re-parse focused_bid + last_action_error from the model-projected text.
        # Source of truth: ``BrowserGymEnv._build_obs_text`` (main.py) emits
        # ``## Focused element:``; the shared agent projection owner emits the
        # error section header. BrowserGym only reorders that section into the
        # AgentLab-style text layout.
        focused_bid: str | None = None
        focused_block = _extract_section_after(final_observation_text, "## Focused element:")
        if focused_block:
            m = re.search(r"bid=['\"]([^'\"]+)['\"]", focused_block)
            if m:
                focused_bid = m.group(1)
        # Only render the "## Focused element:" section if the env actually
        # emitted one — i.e. the env-side ``use_focused_element=True``. When
        # the env's flag is False, ``focused_block`` is None and we suppress
        # the section entirely instead of rendering ``bid=None``. ``use_error_logs``
        # is implicitly gated by ``last_action_error`` being non-None, so it
        # already follows env behavior without extra plumbing.
        sections.append(_render_observation(
            axtree_txt=ax_block,
            pruned_html=html_block,
            tabs_txt=tabs_block,
            focused_bid=focused_bid,
            last_action_error=action_err,
            use_focused_element=focused_block is not None,
        ))

        sections.append(_render_history(actions))
        sections.append(_render_action_space(self.action_describe_text))

        if self.use_hints:
            sections.append(_HINTS_PROMPT)
        if self.use_abstract_example:
            sections.append(_render_abstract_example(use_thinking=self.use_thinking))
        if self.use_concrete_example:
            sections.append(_render_concrete_example(use_thinking=self.use_thinking))

        user_text = "".join(sections)
        # Carry forward metadata from the turn-0 query message. BrowserGym's
        # consolidation rewrites all observations into one user message, but
        # reset-side side channels (notably VWA goal-image indices) still belong
        # to the query and must survive for protocol/agent wrappers that inspect
        # content metadata before the model-facing sanitizer strips it.
        turn0_query_msg = next(
            (m for m in messages if m.get("role") == "user"),
            None,
        )
        query_metadata_parts: list[dict] = []
        if turn0_query_msg is not None:
            for c in turn0_query_msg.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "metadata":
                    query_metadata_parts.append(c)

        # Carry forward image content parts from the selected current-payload
        # observation message (turn-0 role:user or later role:tool page
        # screenshot if any). A later same-turn error-only tool result must not
        # erase an earlier current page screenshot.
        image_parts: list[dict] = []
        if final_observation_msg is not None:
            for c in final_observation_msg.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "image":
                    image_parts.append(c)
        user_msg: LiteMessage = {
            "role": "user",
            "content": image_parts
            + [{"type": "text", "text": user_text}]
            + query_metadata_parts,
        }
        # Keep cua-lite system message + consolidated user message.
        result = [system_msg, user_msg] if system_msg else [user_msg]
        return splice_goal_images(messages, result)


class _BrowserGymGoalImageMixin:
    """BrowserGym goal-image splice over an existing history protocol."""

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs: Any,
    ) -> list[LiteMessage]:
        return splice_goal_images(messages, super().process_messages(messages, **kwargs))


@dataclasses.dataclass
class BrowserGymGoalImageQwen3VLHistoryProtocol(
    _BrowserGymGoalImageMixin,
    Qwen3VLHistoryProtocol,
    key="browsergym.goal_image.qwen3_vl.history",
):
    """``qwen3_vl.history`` plus BrowserGym/VWA goal-image rendering."""


@dataclasses.dataclass
class BrowserGymGoalImageQwen3_5HistoryProtocol(
    _BrowserGymGoalImageMixin,
    Qwen3_5HistoryProtocol,
    key="browsergym.goal_image.qwen3_5.history",
):
    """``qwen3_5.history`` plus BrowserGym/VWA goal-image rendering."""


_BROWSERGYM_CURRENT_PAYLOAD_HEADERS = (
    "## Currently open tabs:",
    "## AXTree:",
    "## HTML:",
    "## Focused element:",
)


def _has_image_part(message: LiteMessage) -> bool:
    return any(
        isinstance(c, dict) and c.get("type") == "image"
        for c in (message.get("content") or [])
    )


def _has_browsergym_current_payload(message: LiteMessage) -> bool:
    text = extract_first_text(message)
    return _has_image_part(message) or any(
        header in text for header in _BROWSERGYM_CURRENT_PAYLOAD_HEADERS
    )


def _latest_browsergym_payload_message(group: list[LiteMessage]) -> LiteMessage | None:
    return next(
        (message for message in reversed(group) if _has_browsergym_current_payload(message)),
        None,
    )


def _projected_tool_result_errors(group: list[LiteMessage]) -> list[str]:
    errors: list[str] = []
    for message in group:
        section = _browsergym_projected_tool_result_error(extract_first_text(message))
        if section is not None and section.strip():
            errors.append(section.strip())
    return errors


def _browsergym_projected_tool_result_error(text: str | None) -> str | None:
    section = extract_projected_tool_result_error(text)
    if section is None:
        return None

    cut_points: list[int] = []
    for header in _BROWSERGYM_CURRENT_PAYLOAD_HEADERS:
        if section.startswith(header):
            cut_points.append(0)
        idx = section.find(f"\n{header}")
        if idx >= 0:
            cut_points.append(idx)
    if cut_points:
        section = section[:min(cut_points)]
    return section


def _extract_section_after(text: str | None, header: str) -> str | None:
    """Naive: return the text following ``header`` up to the next ``## `` header,
    or until end of string. Used to pull AXTree / HTML out of cua-lite's
    flat user-message text format. Returns None if header not present."""
    if not text or header not in text:
        return None
    after = text.split(header, 1)[1]
    # Cut at next "## " section header if any (preserve markdown).
    next_section_idx = after.find("\n## ")
    if next_section_idx >= 0:
        after = after[:next_section_idx]
    return after.lstrip("\n")


# =============================================================================
# Section renderers — assembled into the user message by process_messages
# =============================================================================

def _render_instructions(goal: str) -> str:
    """`# Instructions` section."""
    return _INSTRUCTIONS_PREAMBLE + goal + "\n"


def _render_observation(
    *,
    axtree_txt: str | None = None,
    pruned_html: str | None = None,
    tabs_txt: str | None = None,
    focused_bid: str | None = None,
    last_action_error: str | list[str] | None = None,
    use_focused_element: bool = True,
) -> str:
    """`# Observation of current step:` section.

    Section order mirrors AgentLab's `agentlab.agents.dynamic_prompting`
    rendering: ``## Currently open tabs:`` → ``## AXTree:`` → ``## HTML:``
    → ``## Focused element:`` → owner-rendered tool-result error section.
    """
    parts: list[str] = ["\n# Observation of current step:\n"]
    if tabs_txt is not None:
        # AgentLab puts the tabs block before the AXTree so the model can
        # parse "which tab am I on" before reading the page content.
        parts.append("\n## Currently open tabs:\n")
        parts.append(tabs_txt)
        parts.append("\n")
    if axtree_txt is not None:
        parts.append("\n## AXTree:\n")
        parts.append(_AXTREE_BID_INFO)
        parts.append(_AXTREE_VISIBLE_TAG_INFO)
        parts.append(axtree_txt)
        parts.append("\n")
    if pruned_html is not None:
        parts.append("\n## HTML:\n")
        parts.append(pruned_html)
        parts.append("\n")
    if use_focused_element:
        parts.append("\n## Focused element:\n")
        parts.append(f"bid={repr(focused_bid)}\n" if focused_bid else "None\n")
    if last_action_error:
        errors = [last_action_error] if isinstance(last_action_error, str) else last_action_error
        for error in errors:
            projected_error = project_tool_result_text(None, error)
            if projected_error:
                parts.append("\n")
                parts.append(projected_error)
                parts.append("\n")
    return "".join(parts)


def render_tool_call_json(tc: dict[str, Any]) -> str:
    """Render a structured tool_call dict as the JSON ``<tool_call>`` wire block
    Qwen3-VL emits (matches its chat-template tools section + system prompt)::

        <tool_call>
        {"name": "click", "arguments": {"bid": "13"}}
        </tool_call>

    Selected by ``tool_call_format="json"`` (the default).
    """
    name = tool_call_name(tc)
    args = tool_call_arguments(tc)
    payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


def render_tool_call_xml(tc: dict[str, Any]) -> str:
    """Render a structured tool_call dict as the XML ``<tool_call>`` wire block
    Qwen3.5 emits (matches its chat-template tools section)::

        <tool_call>
        <function=click>
        <parameter=bid>
        13
        </parameter>
        </function>
        </tool_call>

    Selected by ``tool_call_format="xml"``. Self-contained (mirrors Qwen3.5's
    ``_render_xml_tool_call`` / ``_render_param_value``) so this protocol stays
    independent of the model adapters — consistent with the deliberately
    duplicated AgentLab literals at the top of this module.
    """
    name = tool_call_name(tc)
    args = tool_call_arguments(tc)
    lines = ["<tool_call>", f"<function={name}>"]
    for k, v in args.items():
        lines.append(f"<parameter={k}>")
        lines.append(_render_param_value(v))
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def _render_param_value(value: Any) -> str:
    """Single-line wire repr of a tool_call arg value (Qwen3.5 XML body)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(list(value) if isinstance(value, tuple) else value)
    return str(value)


# ``tool_call_format`` → renderer. Each returns the FULL wire-format
# ``<tool_call>...</tool_call>`` block for one structured tool_call.
_TOOL_CALL_RENDERERS = {
    "json": render_tool_call_json,
    "xml": render_tool_call_xml,
}


def _render_history(actions: list[str]) -> str:
    """`# History of interaction with the task:` section.

    Renders each past step as ``## step N`` followed by that turn's
    ``<tool_call>...</tool_call>`` block(s). Paper's AgentLab uses
    ``<action>code</action>`` here; we use the adapter's own ``<tool_call>``
    wire format (rendered by ``render_tool_call`` in ``process_messages``) so
    the history examples match — not contradict — the format the system prompt
    + chat-template tools section demand.

    Each ``action_block`` is already the wire-format ``<tool_call>`` block built
    from the message's structured ``tool_calls`` (NOT raw ``raw_response.text``),
    so there are no reasoning blobs / orphan ``</think>`` / ``Action:`` prose to
    strip (the Bug-B history fix).
    """
    if not actions:
        return ""
    parts = ["\n# History of interaction with the task:\n\n"]
    for i, action_block in enumerate(actions):
        parts.append(f"## step {i}\n\n{action_block.strip()}\n\n")
    return "".join(parts)


def _render_action_space(action_describe_text: str | None) -> str:
    """`# Action space:` section."""
    if action_describe_text is None:
        return ""
    return f"\n# Action space:\n{_ACTION_SET_GENERIC_INFO}{action_describe_text}\n"


def _render_concrete_example(use_thinking: bool) -> str:
    """`# Concrete Example` section (only renders when ``use_concrete_example=True``).

    Action body intentionally omitted — the chat-template tools section
    emitted by the adapter (``_build_tools_section``) owns the per-adapter
    format example (Qwen3-VL JSON, Qwen3.5 XML).
    """
    parts = [
        "\n# Concrete Example\n\n"
        "Here is a concrete example of how to format your answer.\n"
        "Make sure to follow the template with proper tags:"
    ]
    if use_thinking:
        parts.append(_THINK_CONCRETE_EX)
    parts.append(
        "\n(Then emit a single <tool_call>...</tool_call> in the format "
        "described in the system message.)\n"
    )
    return "".join(parts)


def _render_abstract_example(use_thinking: bool) -> str:
    """`# Abstract Example` section.

    See ``_render_concrete_example`` for why the action body is delegated
    to the adapter's tools section.
    """
    parts = [
        "\n# Abstract Example\n\n"
        "Here is an abstract version of the answer with description of the content of\n"
        "each tag. Make sure you follow this structure, but replace the content with your\n"
        "answer:"
    ]
    if use_thinking:
        parts.append(_THINK_ABSTRACT_EX)
    parts.append(
        "\n(Then emit a single <tool_call>...</tool_call> in the format "
        "described in the system message.)\n"
    )
    return "".join(parts)
