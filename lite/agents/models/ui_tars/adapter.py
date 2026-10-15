"""
UI-TARS Adapters (Desktop, Browser, Mobile)

Provides adapters for the original UI-TARS-7B-DPO model, matching the
OSWorld uitars_agent.py data flow exactly.

    UITarsBaseAdapter (shared logic)
    ├── UITarsDesktopGroundingPointAdapter   (single-step click; desktop+browser)
    ├── UITarsDesktopUseAdapter       (summarized history; desktop+browser)
    ├── UITarsMobileGroundingPointAdapter    (single-step click; mobile schema)
    └── UITarsMobileUseAdapter        (summarized history, mobile)

The ``Desktop`` adapters register under the
``r"ui_tars@(desktop|browser)@..."`` regex, so ``ui_tars@browser@...`` keys
resolve to the same class.

Key differences from ui_tars_15_v1:
- Desktop: finished() has no content parameter, includes call_user() action
- Mobile: finished(content=''), long_press, press_home, press_back (UITars native), scroll with end_box

Supports both structured tool_calls and text-based prompting:
    # Structured (default)
    agent_sample = adapter.unroll(sample)

    # Text-based (no tools API needed)
    agent_sample = adapter.unroll(sample)

For pass-through (understanding, grounding/bbox), use AsIsAdapter from base.

Usage:
    from lite.agents.models.ui_tars.adapter import (
        UITarsDesktopGroundingPointAdapter,
        UITarsDesktopUseAdapter,
        UITarsMobileGroundingPointAdapter,
        UITarsMobileUseAdapter,
    )

    adapter = UITarsDesktopUseAdapter()
    agent_sample = adapter.unroll(sample)
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import logging
import math
import re
from typing import Any

from lite.agents.core.adapter import (
    AgentAdapterRegistry,
    AsIsAdapter,
    BaseAgentAdapter,
)
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.models.ui_tars.action_space import (
    GROUNDING_DESKTOP_ACTION_SPACE,
    GROUNDING_MOBILE_ACTION_SPACE,
    UITarsBBoxActionSpace,
    UITarsDesktopActionSpace,
    UITarsDesktopGroundingPointActionSpace,
    UITarsMobileActionSpace,
    UITarsMobileGroundingPointActionSpace,
)
from lite.agents.models.ui_tars.protocol import UITarsHistoryProtocol
from lite.agents.types import AgentMessage, AgentStep
from lite.core import (
    LiteCUAMetadata,
    LiteMessage,
    LiteSample,
)
from lite.core.messages import instruction_text, make_assistant_content
from lite.core.messages.final import MODEL_OUTPUT_ERROR_KEY, mark_model_output_error
from lite.core.messages.turns import truncate_sample_to_turn

logger = logging.getLogger(__name__)

# =============================================================================
# Helpers
# =============================================================================
#
# These module-level free functions are the canonical, shared implementations
# for the whole UI-TARS family (ui_tars, ui_tars_15_v1); v1 imports them
# directly. The resize body and the per-action parse loop are exposed as methods on
# :class:`UITarsBaseAdapter` (``_linear_resize_image`` reads the per-subclass
# ``min_pixels`` / ``max_pixels`` class attrs; ``_parse_action_text`` reads the
# per-subclass ``model_name`` for the warning string).

def _strip_box_tokens(text: str) -> str:
    """Remove <|box_start|> and <|box_end|> tokens."""
    return text.replace("<|box_start|>", "").replace("<|box_end|>", "")

def _action_parse_error(action_str: str) -> str:
    """Message for a failed ``Action:`` parse — the whole UI-TARS family's marker.

    ``Action:`` is the SFT-trained grammar marker: seeing it with zero parsed
    calls means the model ATTEMPTED an action and got the syntax wrong, which
    must be marked as a terminal parse-failure final rather than mistaken for a
    clean content-only final (prose with no ``Action:`` at all).
    """
    tail = f": {action_str!r}" if action_str else ""
    return (
        "malformed 'Action:' block: expected a call like "
        "click(start_box='(x,y)') after 'Action:'" + tail
    )


def _looks_like_action_call(action_str: str) -> bool:
    return bool(re.match(r"^[A-Za-z_]\w*\s*\(", action_str.strip()))


def _parse_function_call(action_str: str) -> dict[str, Any] | None:
    """
    Parse a Python function call string into {name, args} dict.

    Example:
        "click(start_box='(500,300)')" -> {"name": "click", "args": {"start_box": "(500,300)"}}

    Falls back to regex when AST parsing fails (e.g. truncated output missing
    closing quotes/parens).
    """
    text = action_str.strip()

    # --- Primary: AST-based parsing ---
    try:
        node = ast.parse(text, mode="eval")
        if isinstance(node, ast.Expression) and isinstance(node.body, ast.Call):
            call = node.body
            if isinstance(call.func, ast.Name):
                kwargs = {}
                for kw in call.keywords:
                    if isinstance(kw.value, ast.Constant):
                        kwargs[kw.arg] = kw.value.value
                    else:
                        kwargs[kw.arg] = None
                return {"name": call.func.id, "args": kwargs}
    except Exception:
        pass

    # --- Fallback: regex for truncated output like click(start_box='(674,381) ---
    m = re.match(r"(\w+)\s*\((.*)$", text, re.DOTALL)
    if not m:
        return None
    func_name = m.group(1)
    rest = m.group(2)
    kwargs = {}
    for km in re.finditer(r"(\w+)\s*=\s*['\"]?\(?(\d+)[,\s]+(\d+)\)?", rest):
        kwargs[km.group(1)] = f"({km.group(2)},{km.group(3)})"
    # Also capture non-coordinate string args like key='ctrl c', content='...'
    for km in re.finditer(r"(\w+)\s*=\s*'([^']*)'?", rest):
        if km.group(1) not in kwargs:
            kwargs[km.group(1)] = km.group(2)
    return {"name": func_name, "args": kwargs} if kwargs or func_name in ("wait", "finished", "call_user") else None

def _parse_coordinate_str(coord_str: str) -> list[int] | None:
    """Parse coordinate string "(x,y)" to [x, y] list."""
    if not coord_str:
        return None
    cleaned = coord_str.replace("(", "").replace(")", "").strip()
    parts = cleaned.split(",")
    try:
        result = [int(p.strip()) for p in parts if p.strip()]
        return result if len(result) >= 2 else None
    except ValueError:
        return None

# =============================================================================
# Prompts (matching OSWorld defaults)
# =============================================================================

# System prompt: matches OSWorld's system message
USE_SYSTEM_PROMPT = "You are a helpful assistant."

# Action space: matches OSWorld's UITARS_CALL_USR_ACTION_SPACE
UITARS_CALL_USR_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
"""

# User prompt template: matches OSWorld's UITARS_USR_PROMPT_THOUGHT
USE_USER_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""


# Grounding-only user prompt: drops Thought (single-step click prediction
# doesn't benefit from CoT and adds tokens that compete with the visual
# attention budget). Keeps the same SFT-aligned action_space syntax so the
# model emits its native ``click(start_box=...)`` form.
GROUNDING_USER_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Action: ...
```

## Action Space
{action_space}

## User Instruction
{instruction}
"""

# =============================================================================
# UITars Base Adapter (shared logic)
# =============================================================================

@dataclasses.dataclass
class UITarsBaseAdapter(BaseAgentAdapter):
    """
    Base adapter for UI-TARS format (original UI-TARS-7B-DPO).

    Matches the OSWorld uitars_agent.py data flow:
    - System: "You are a helpful assistant."
    - User: UITARS_USR_PROMPT_THOUGHT (with UITARS_CALL_USR_ACTION_SPACE)
    - History: alternating user(image) + assistant(text) messages
    - Current: user(image)

    """

    system_prompt: str | None = None
    user_prompt_template: str | None = None
    action_space_text: str = UITARS_CALL_USR_ACTION_SPACE

    # ─── Per-subclass vision / parsing knobs ─────────────────────────────────
    # ``_linear_resize_image`` reads ``min_pixels`` / ``max_pixels`` (UI-TARS &
    # v2 use the larger 100*28*28 floor; v1 overrides ``min_pixels`` to the
    # smaller 4*28*28). ``_parse_action_text`` reads ``model_name`` for the
    # parse-failure warning string.
    min_pixels: int = 100 * 28 * 28
    max_pixels: int = 16384 * 28 * 28
    model_name: str = "UITars"

    def _linear_resize_image(self, img):
        """Resize a PIL Image using sqrt-based linear scaling.

        Preserves aspect ratio so that normalized relative coordinates remain
        valid. The ``[min_pixels, max_pixels]`` band is per-subclass (class
        attributes).
        """
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        width, height = img.size
        if width * height > max_pixels:
            factor = math.sqrt(max_pixels / (width * height))
            width, height = int(width * factor), int(height * factor)
            img = img.resize((width, height))
        if width * height < min_pixels:
            factor = math.sqrt(min_pixels / (width * height))
            width, height = math.ceil(width * factor), math.ceil(height * factor)
            img = img.resize((width, height))
        return img

    def _parse_action_text(self, action_str: str) -> list[dict[str, Any]]:
        """Parse UITars action text into a list of tool_call dicts.

        Input: "click(start_box='(500,300)')"
        Output: [{"name": "click", "arguments": {"start_box": [500, 300]}}]
        """
        action_str = _strip_box_tokens(action_str)

        raw_actions = action_str.split("\n\n")
        tool_calls = []

        for raw in raw_actions:
            raw = raw.strip()
            if not raw:
                continue

            parsed = _parse_function_call(raw.replace("\n", "\\n"))
            if not parsed:
                logger.warning("Failed to parse %s action: %s", self.model_name, raw)
                continue

            func_name = parsed["name"]
            raw_args = parsed["args"]

            # Convert coordinate string args to [x, y] lists
            arguments = {}
            for k, v in raw_args.items():
                if k in ("start_box", "end_box") and isinstance(v, str):
                    coord = _parse_coordinate_str(v)
                    if coord is not None:
                        arguments[k] = coord
                    else:
                        arguments[k] = v
                elif v is not None:
                    arguments[k] = v

            tool_calls.append({"name": func_name, "arguments": arguments})

        return tool_calls

    def _process_image_after_target(self, img):
        """Linear-resize per UI-TARS vision constraints (sqrt-based scaling
        between min_pixels and max_pixels).

        Stage-2 hook called after ``BaseAgentAdapter.process_image``'s
        exact-stretch target-resize (when ``self.resolution`` is set).
        """
        return self._linear_resize_image(img)

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render turn ``k`` in UI-TARS wire format.

        When ``user_prompt_template`` is set (``use`` mode), matches
        OSWorld's message structure:
          system: "You are a helpful assistant."
          user: UITARS_USR_PROMPT_THOUGHT (action space + instruction + notes)
          [history assistant/user messages...]
          user: [current image]
        """
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        result_messages: list[AgentMessage] = []
        if self.system_prompt:
            result_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            })

        if self.user_prompt_template:
            instruction = instruction_text(messages)
            # The blob is SFT text, so it is substituted WHOLE: neither the
            # active extra tools nor ``valid_actions`` may drop a row.
            # Advertising only the finish form that reaches an active
            # ``terminate`` was measured on lite.osworld to remove 16
            # inactive-tool rejections but collapse finish attempts from 19 to
            # 2, leaving 5 of 8 trajectories burning ``max_steps``.
            # ``finished(content='xxx')`` is the form UI-TARS was trained on, so
            # it is the form that makes the model declare completion at all.
            # Reachability is env ingress's answer: a call to an inactive tool
            # comes back as model-visible feedback keyed to its call id.
            user_prompt = self.user_prompt_template.format(
                instruction=instruction,
                action_space=self.action_space_text,
            )
            result_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            })

            # Strip text from first user message (instruction already in
            # template); keep images as separate user messages.
            first_user_seen = False
            for msg in messages:
                if not first_user_seen and msg.get("role") == "user":
                    first_user_seen = True
                    content = msg["content"]
                    # ``LiteUserMessage.content`` is ``str | list``
                    # (content.py:62-65); a bare str carries no image part.
                    image_items = [] if isinstance(content, str) else [
                        item for item in content if item.get("type") == "image"
                    ]
                    if image_items:
                        result_messages.append({
                            "role": "user",
                            "content": image_items,
                        })
                    continue
                if msg.get("role") == "tool":
                    observation = copy.deepcopy(msg)
                    observation["role"] = "user"
                    observation.pop("call_id", None)
                    observation.pop("tool_call_id", None)
                    result_messages.append(observation)
                    continue
                result_messages.append(self.convert_message_to_agent(msg))
        else:
            for msg in messages:
                result_messages.append(self.convert_message_to_agent(msg))

        return result_messages

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert a single CUA-lite message to UI-TARS wire format.

        For assistants: full fold into the SFT-trained ``Thought: ...\\n
        <action description>\\nAction: <pyautogui call>`` text via
        ``action_space.format_message_as_text`` (which selectively pulls
        ``InlineReasoningContent`` parts via ``get_inline_reasoning``
        and the action-description text via ``get_action_description``). The
        rendered text is wrapped as a single ``{"type": "text", ...}`` content
        part; structured ``tool_calls`` and ``reasoning_content`` are dropped
        so the chat_template only sees the byte-exact wire text.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result
        if "tool_calls" in result:
            result["tool_calls"] = self.action_space.convert_tool_calls_to_agent(
                result["tool_calls"]
            )
        had_tool_calls = bool(message.get("tool_calls"))
        text = self.action_space.format_message_as_text(result)
        if text:
            result["content"] = [{"type": "text", "text": text}]
        elif not had_tool_calls:
            # Content-only final turn: keep ONLY the plain ``text`` parts (the
            # canonical "Done."), so the turn is not an empty SFT target while
            # non-``text`` kinds this wire format cannot carry still drop.
            result["content"] = [
                p for p in (message.get("content") or [])
                if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
            ]
        else:
            result["content"] = []
        result.pop("tool_calls", None)
        result.pop("reasoning_content", None)
        return result

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """Convert AgentMessage → LiteMessage. UI-TARS has no chat-template
        tokens (no ``<tool_call>`` / ``<think>``); its entire response is
        system-prompt-defined. All Thought/Action/action-string parsing
        lives here, so :meth:`parse_raw_assistant_response` is a trivial
        text wrapper.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        raw_text = ""
        for part in message.get("content") or []:
            if part.get("type") == "text" and part.get("text"):
                raw_text = part["text"]
                break

        # No text (API path): only normalize tool_calls.
        if not raw_text:
            if "tool_calls" in result:
                result["tool_calls"] = self._route_agent_tool_calls_to_lite(
                    result["tool_calls"]
                )
            return result

        clean = raw_text.strip()
        thought_text = ""
        action_str = ""
        if "Action:" in clean:
            parts = clean.split("Action:", 1)
            pre_action = parts[0].strip()
            action_str = parts[1].strip()
            m = re.search(r"Thought:\s*(.+)", pre_action, re.DOTALL)
            if m:
                thought_text = m.group(1).strip()
            elif pre_action:
                thought_text = pre_action
        else:
            m = re.search(r"Thought:\s*(.+)", clean, re.DOTALL)
            if m:
                thought_text = m.group(1).strip()
            elif clean:
                thought_text = clean

        if action_str:
            tool_calls = self._parse_action_text(action_str)
            if tool_calls:
                result["tool_calls"] = self._route_agent_tool_calls_to_lite(tool_calls)
            else:
                result.pop("tool_calls", None)
                if _looks_like_action_call(action_str):
                    mark_model_output_error(result, _action_parse_error(action_str))
        elif "tool_calls" in result:
            result["tool_calls"] = self._route_agent_tool_calls_to_lite(
                result["tool_calls"]
            )
        elif "Action:" in clean:
            mark_model_output_error(result, _action_parse_error(""))

        if not result.get("tool_calls") and MODEL_OUTPUT_ERROR_KEY not in result:
            result["content"] = [{"type": "text", "text": raw_text}]
            return result

        # UI-TARS's ``Thought:`` line is prompted CoT — all of it lives
        # in ``InlineReasoningContent``. There is no separate
        # action_description slot in the SFT distribution (the structured
        # action lives in ``tool_calls``).
        result["content"] = make_assistant_content(
            inline_reasoning=thought_text.strip() if thought_text else "",
        )
        return result

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """Wrap raw UI-TARS output verbatim as an ``AgentMessage``. UI-TARS
        has no chat-template tokens — its entire response (Thought/Action
        text + action calls) is system-prompt-defined and parsed by
        :meth:`convert_message_from_agent`.
        """
        return {"role": "assistant", "content": [{"type": "text", "text": response}]}

# =============================================================================
# Desktop + Browser Adapters
# =============================================================================
# Desktop and browser share one adapter class per task type; the
# ``(desktop|browser)`` regex on each key registers the same body under both
# platforms. Mobile keeps its own classes (different action space and user
# prompt).

@dataclasses.dataclass
class UITarsDesktopGroundingActionAdapter(
    UITarsBaseAdapter,
    key=r"ui_tars@(desktop|browser)@grounding\.action",
):
    """Desktop+browser grounding/action: full UI-TARS action vocabulary.

    Currently NOT routed by env eval (env declares ``grounding.point``;
    see :class:`UITarsDesktopGroundingPointAdapter`). Kept for SFT replay
    where the historical grounding.action SFT data shape (multi-action,
    full pyautogui-style action_space text) is rendered through the
    family-native UI-TARS wire format.
    """
    action_space: UITarsDesktopActionSpace = dataclasses.field(
        default_factory=UITarsDesktopActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    user_prompt_template: str | None = GROUNDING_USER_PROMPT


@dataclasses.dataclass
class UITarsDesktopGroundingPointAdapter(
    UITarsBaseAdapter,
    key=r"ui_tars@(desktop|browser)@grounding\.point",
):
    """Desktop+browser grounding (single-step click) for UI-TARS.

    Wire format is the SFT-aligned pyautogui-style text
    ``Action: click(start_box='(x,y)')``. The advertised ``action_space_text``
    is trimmed to the single native click form that maps to cua-lite
    ``point``. The model emits ``click(start_box=)``;
    :class:`UITarsDesktopGroundingPointActionSpace` routes that to cua-lite
    ``LitePointActionSpace.point(coord)`` instead of the ``use``
    ``LiteDesktopActionSpace.click(coordinate=)``. Refusal lives on the env's
    ``report_infeasible`` extra tool.
    """
    action_space: UITarsDesktopGroundingPointActionSpace = dataclasses.field(
        default_factory=UITarsDesktopGroundingPointActionSpace
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
    user_prompt_template: str | None = GROUNDING_USER_PROMPT
    action_space_text: str = GROUNDING_DESKTOP_ACTION_SPACE


@dataclasses.dataclass
class UITarsDesktopUseAdapter(
    UITarsBaseAdapter,
    key=r"ui_tars@(desktop|browser)@use",
):
    """Desktop+browser ``use`` (multi-step rollout): with system prompt, UITars-style history.

    Matches OSWorld's message structure:
      system: "You are a helpful assistant."
      user: UITARS_USR_PROMPT_THOUGHT (action space + instruction + notes)
      [history assistant/user messages...]
      user: [current image]
    """
    action_space: UITarsDesktopActionSpace = dataclasses.field(
        default_factory=UITarsDesktopActionSpace
    )
    protocol: UITarsHistoryProtocol = dataclasses.field(
        default_factory=lambda: UITarsHistoryProtocol(full_history_size=5)
    )
    system_prompt: str | None = USE_SYSTEM_PROMPT
    user_prompt_template: str | None = USE_USER_PROMPT


# =============================================================================
# Mobile Prompts
# =============================================================================

UITARS_MOBILE_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
type(content='')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
press_home()
press_back()
finished(content='') # Submit the task regardless of whether it succeeds or fails.
"""

MOBILE_USE_USER_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use English in `Thought` part.

- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

# =============================================================================
# Mobile Adapters
# =============================================================================

@dataclasses.dataclass
class UITarsMobileGroundingActionAdapter(
    UITarsBaseAdapter,
    key="ui_tars@mobile@grounding.action",
):
    """Mobile grounding/action: full mobile action vocabulary. SFT-replay only."""
    action_space: UITarsMobileActionSpace = dataclasses.field(
        default_factory=UITarsMobileActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    user_prompt_template: str | None = GROUNDING_USER_PROMPT
    action_space_text: str = UITARS_MOBILE_ACTION_SPACE


@dataclasses.dataclass
class UITarsMobileGroundingPointAdapter(
    UITarsBaseAdapter,
    key="ui_tars@mobile@grounding.point",
):
    """Mobile grounding (single-step click) for UI-TARS.

    Same conversion-layer trim as desktop: emits the model's native
    ``click(start_box=)``, routes to cua-lite ``point(coord)``. The
    advertised ``action_space_text`` is the single native click form.
    """
    action_space: UITarsMobileGroundingPointActionSpace = dataclasses.field(
        default_factory=UITarsMobileGroundingPointActionSpace
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
    user_prompt_template: str | None = GROUNDING_USER_PROMPT
    action_space_text: str = GROUNDING_MOBILE_ACTION_SPACE

@dataclasses.dataclass
class UITarsMobileUseAdapter(
    UITarsBaseAdapter,
    key="ui_tars@mobile@use",
):
    """Mobile ``use`` (multi-step rollout): with system prompt, UITars-style history."""
    action_space: UITarsMobileActionSpace = dataclasses.field(
        default_factory=UITarsMobileActionSpace
    )
    protocol: UITarsHistoryProtocol = dataclasses.field(
        default_factory=lambda: UITarsHistoryProtocol(full_history_size=5)
    )
    system_prompt: str | None = USE_SYSTEM_PROMPT
    user_prompt_template: str | None = MOBILE_USE_USER_PROMPT
    action_space_text: str = UITARS_MOBILE_ACTION_SPACE

# =============================================================================
# Pass-through Adapters (understanding)
# =============================================================================

AgentAdapterRegistry.register(r"ui_tars@(desktop|browser|mobile)@understanding", AsIsAdapter)
# ``grounding.point`` and ``grounding.action`` are both served by concrete
# per-platform classes above (env eval → point, SFT replay → action).

# =============================================================================
# Grounding Adapters (bbox, point)
# =============================================================================

@dataclasses.dataclass
class UITarsGroundingBBoxAdapter(
    UITarsBaseAdapter,
    key=r"ui_tars@(desktop|browser|mobile)@grounding\.bbox",
):
    """Grounding/bbox adapter for UITars.

    ``action_space_text`` is emptied, not inherited. The base class advertises
    the UI-TARS ``use`` vocabulary (``click``/``drag``/``finished``/…); the only
    row of it this surface speaks is ``click``, and only carrying a FOUR-value
    box — that is the wire spelling
    :class:`~lite.agents.models.ui_tars.action_space.UITarsBBoxActionSpace`
    renders and parses. Inert today only because ``user_prompt_template`` is
    ``None``, so the text is never substituted; wiring a template on must not
    resurrect the nine other names, nor the two-value ``click`` this space
    rejects.
    """
    action_space: UITarsBBoxActionSpace = dataclasses.field(
        default_factory=UITarsBBoxActionSpace
    )
    action_space_text: str = ""
    metadata: LiteCUAMetadata = dataclasses.field(
        default_factory=lambda: LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.DESKTOP.value,
                LiteCUAMetadata.TaskType.GROUNDING_BBOX.value,
            )
        )
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )

# ``grounding.point`` is now served by the concrete desktop/browser and mobile
# GroundingPointAdapter classes above (env-side trimmed UI-TARS-native
# ``click(start_box=)`` harness) — no pattern-based AsIsAdapter needed.
