"""
UI-TARS 1.5 v1 Adapters (Desktop, Browser, Mobile)

Provides UI-TARS 1.5 v1 adapters for desktop, browser, and mobile, matching the open-source UI-TARS-1.5-7B model
(HuggingFace: ByteDance-Seed/UI-TARS-1.5-7B).

    UITars15V1BaseAdapter (shared logic)
    ├── UITars15V1DesktopGroundingPointAdapter   (single-step click; desktop+browser)
    ├── UITars15V1DesktopUseAdapter       (summarized history; desktop+browser)
    ├── UITars15V1MobileGroundingPointAdapter    (single-step click; mobile schema)
    └── UITars15V1MobileUseAdapter        (summarized history, mobile)

The ``Desktop`` adapters register under the
``r"ui_tars_15_v1@(desktop|browser)@..."`` regex, so ``ui_tars_15_v1@browser@...``
keys resolve to the same class.

Key differences from v2:
- No <think> reasoning tags (v2 uses <think>...</think> for Seed1.5-VL thinking mode)
- System prompt uses Thought: / Action: format directly (no thinking wrapper)
- Uses the same action space as v2 ((x,y) tokens)

Supports both structured tool_calls and text-based prompting:
    # Structured (default)
    agent_sample = adapter.unroll(sample)

    # Text-based (no tools API needed)
    agent_sample = adapter.unroll(sample)

For pass-through (understanding, grounding/bbox), use AsIsAdapter from base.

Usage:
    from lite.agents.models.ui_tars_15_v1.adapter import (
        UITars15V1DesktopGroundingPointAdapter,
        UITars15V1DesktopUseAdapter,
        UITars15V1MobileGroundingPointAdapter,
        UITars15V1MobileUseAdapter,
    )

    adapter = UITars15V1DesktopUseAdapter()
    agent_sample = adapter.unroll(sample)
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import re
from typing import Any

from lite.agents.core.adapter import (
    AgentAdapterRegistry,
    AsIsAdapter,
)
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.models.ui_tars.adapter import (
    UITarsBaseAdapter,
    _action_parse_error,
    _looks_like_action_call,
)
from lite.agents.models.ui_tars.protocol import UITarsHistoryProtocol
from lite.agents.models.ui_tars_15_v1.action_space import (
    GROUNDING_DESKTOP_ACTION_SPACE,
    GROUNDING_MOBILE_ACTION_SPACE,
    UITars15V1BBoxActionSpace,
    UITars15V1DesktopActionSpace,
    UITars15V1DesktopGroundingPointActionSpace,
    UITars15V1MobileActionSpace,
    UITars15V1MobileGroundingPointActionSpace,
)
from lite.agents.types import AgentMessage
from lite.core import (
    LiteCUAMetadata,
    LiteMessage,
)
from lite.core.messages import make_assistant_content
from lite.core.messages.final import MODEL_OUTPUT_ERROR_KEY, mark_model_output_error
from lite.core.tools.action_space import MAX_NORM, clamp_norm
from lite.utils.image import smart_resize as _canonical_smart_resize

logger = logging.getLogger(__name__)

# =============================================================================
# Helpers
# =============================================================================
#
# Parsing helpers (``_strip_box_tokens`` / ``_parse_function_call`` /
# ``_parse_coordinate_str``) and the ``_linear_resize_image`` /
# ``_parse_action_text`` methods are inherited from
# :class:`lite.agents.models.ui_tars.adapter.UITarsBaseAdapter`. v1 only
# overrides ``min_pixels`` (smaller 4*28*28 floor) and ``model_name``.

# Image pixel limits for UI-TARS-1.5-7B (Qwen2.5-VL backbone). The resize body
# is the canonical ``lite.utils.image.smart_resize`` (byte-identical output to
# the old local copy on every realistic screenshot size); only the per-family
# constants live here. They match the model's image_processor config and
# determine the smart_resize coordinate space used for absolute-pixel output.
_IMAGE_FACTOR = 28
_MIN_PIXELS = 4 * _IMAGE_FACTOR * _IMAGE_FACTOR      # 3136
_MAX_PIXELS = 16384 * _IMAGE_FACTOR * _IMAGE_FACTOR   # 12845056

def _smart_resize(
    height: int,
    width: int,
    factor: int = _IMAGE_FACTOR,
    min_pixels: int = _MIN_PIXELS,
    max_pixels: int = _MAX_PIXELS,
) -> tuple[int, int]:
    """Rescale so both dims are divisible by *factor* and pixel count is in range.

    Thin wrapper binding UI-TARS-1.5 v1's per-family qwen25 constants to the
    canonical ``lite.utils.image.smart_resize``. Returns (height, width).

    v1-specific in its *use*, not its arithmetic: it caches the absolute-pixel
    coordinate space for coord denormalization; ut / v2 do not denormalize.
    """
    return _canonical_smart_resize(
        height=height, width=width,
        factor=factor, min_pixels=min_pixels, max_pixels=max_pixels,
    )

# =============================================================================
# Prompts (matching OSWorld defaults)
# =============================================================================

# System prompt: matches OSWorld's system message
USE_SYSTEM_PROMPT = "You are a helpful assistant."

# Action space: matches OSWorld's UITARS_NORMAL_ACTION_SPACE verbatim. Every
# byte was part of the SFT distribution, so the block is rendered whole — no
# gate trims a row, neither the active extra tools nor ``valid_actions``.
# ``finished(content=)`` lowers to canonical ``response``, so under a
# ``terminate``-only config the advertised form is the one the env rejects, and
# that is deliberate: advertising only bare ``finished()`` instead was measured
# WORSE on lite.osworld (16 inactive-tool rejections removed, but finish
# attempts collapsed from 19 to 2 and 5 of 8 trajectories burned ``max_steps``).
# ``finished(content='xxx')`` is the form UI-TARS was trained on, so it is the
# form that makes the model declare completion at all.
UITARS_NORMAL_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
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

# Grounding-only user prompt: drops Thought. Keeps UI-TARS-1.5's native
# action_space syntax so the model emits its SFT-aligned
# ``click(start_box=...)`` form.
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
# UITars 1.5 v1 Base Adapter (shared logic)
# =============================================================================

@dataclasses.dataclass
class UITars15V1BaseAdapter(UITarsBaseAdapter):
    """
    Base adapter for UI-TARS 1.5 v1 format (open-source UI-TARS-1.5-7B).

    Subclasses :class:`UITarsBaseAdapter` and reuses its shared wire format
    (render_step, instruction extraction, Thought/Action parsing,
    ``_parse_action_text`` / ``_linear_resize_image``, raw-response
    wrapping). Reuses the same action space as v2 (UITars15V1DesktopActionSpace)
    since the tool format ((x,y)) is identical.

    Key differences overridden here:
    - ``min_pixels`` floor is the smaller 4*28*28 (=3136) — v1's Qwen2.5-VL
      image_processor config (ut / v2 use 100*28*28).
    - coordinate denormalization: model output is absolute pixels on the
      smart_resize'd image, mapped back to [0, 1000] (ut / v2 are already
      normalized). Requires the v1-specific ``_smart_resize`` dim caching.

    Key difference from v2: no <think> reasoning tags. Output is strictly:
        Thought: ...
        Action: ...

    """

    system_prompt: str | None = None
    user_prompt_template: str | None = None
    action_space_text: str = UITARS_NORMAL_ACTION_SPACE

    # v1 floors at the smaller 4*28*28 (Qwen2.5-VL image_processor config);
    # ut / v2 keep the inherited 100*28*28. max_pixels is identical to base.
    min_pixels: int = _MIN_PIXELS
    model_name: str = "UITars15V1"

    # Stored smart_resize dimensions (width, height) of the last processed image.
    # Used to convert absolute pixel coordinates from the model to [0, 1000] normalized.
    _last_smart_resize_wh: tuple = dataclasses.field(default=(1000, 1000), init=False, repr=False)

    def _process_image_after_target(self, img):
        """Linear-resize per UI-TARS-1.5 vision constraints + cache the
        smart_resize dimensions of the last seen image so
        ``convert_message_from_agent`` can convert absolute pixel coords
        to [0, 1000] normalized.

        Stage-2 hook called after ``BaseAgentAdapter.process_image``'s
        exact-stretch target-resize (when ``self.resolution`` is set).
        """
        resized = self._linear_resize_image(img)
        # Cache smart_resize dims of THIS image (last call wins — caller
        # iterates in order so the last image is the most-recent screenshot).
        w, h = resized.size
        sr_h, sr_w = _smart_resize(h, w, min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS)
        self._last_smart_resize_wh = (sr_w, sr_h)
        return resized

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert a single CUA-lite message to UI-TARS 1.5 v1 wire format.

        Same full-fold pattern as ``UITarsBaseAdapter`` (see its docstring),
        plus a v1-specific coordinate denormalization step: [0, 1000] cua-lite
        coords are mapped back to absolute pixel space (smart_resize'd image
        dimensions) so the rendered ``Action:`` line matches the model's
        original SFT distribution.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result
        if "tool_calls" in result:
            agent_tool_calls = self.action_space.convert_tool_calls_to_agent(
                result["tool_calls"]
            )
            sr_w, sr_h = self._last_smart_resize_wh
            for tc in agent_tool_calls:
                args = tc["arguments"]
                for key in ("start_box", "end_box"):
                    coord = args.get(key)
                    # ``==`` not ``>=``: this wire format's boxes are (x, y).
                    # A longer list is malformed, and denormalizing its first
                    # two entries would silently turn it into a valid-looking
                    # point. Leave it untouched so it stays visibly wrong.
                    if isinstance(coord, list) and len(coord) == 2:
                        args[key] = [
                            int(coord[0] / 1000.0 * sr_w),
                            int(coord[1] / 1000.0 * sr_h),
                        ]
            result["tool_calls"] = agent_tool_calls
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
        """Convert AgentMessage → LiteMessage. UI-TARS 1.5 v1 has no
        chat-template tokens — entire response is system-prompt-defined.
        Also converts absolute pixel coordinates (on smart_resize'd
        images) back to ``[0, 1000]`` normalized coordinates on the
        extracted tool_calls.
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

        tool_calls: list[dict[str, Any]] = []
        if action_str:
            tool_calls = self._parse_action_text(action_str)
            if tool_calls:
                # Convert absolute pixel coordinates (on smart_resize'd image)
                # to [0, 1000] normalized coordinates expected by CUA-lite.
                sr_w, sr_h = self._last_smart_resize_wh
                for tc in tool_calls:
                    args = tc["arguments"]
                    for key in ("start_box", "end_box"):
                        coord = args.get(key)
                        # ``==`` not ``>=``, mirroring the render loop above: an
                        # over-long box must reach the action space's
                        # ``required_coord(dimensions=2)`` still over-long, so
                        # the parser owns the failure instead of receiving a
                        # pre-truncated two-value coordinate that validates.
                        if isinstance(coord, list) and len(coord) == 2:
                            args[key] = [
                                clamp_norm(int(coord[0] / sr_w * MAX_NORM)),
                                clamp_norm(int(coord[1] / sr_h * MAX_NORM)),
                            ]
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

        # UI-TARS 1.5 v1's ``Thought:`` is prompted CoT — all of it lives
        # in ``InlineReasoningContent``. The structured action lives in
        # ``tool_calls``; there is no separate action_description slot.
        result["content"] = make_assistant_content(
            inline_reasoning=thought_text.strip() if thought_text else "",
        )
        return result

    # ``parse_raw_assistant_response``, ``render_step``, ``_parse_action_text``
    # and ``_linear_resize_image`` are inherited verbatim from
    # :class:`UITarsBaseAdapter`.

# =============================================================================
# Desktop + Browser Adapters
# =============================================================================
# Desktop and browser share one adapter class per task type via a
# ``(desktop|browser)`` regex key. Mobile is its own platform.

@dataclasses.dataclass
class UITars15V1DesktopGroundingActionAdapter(
    UITars15V1BaseAdapter,
    key=r"ui_tars_15_v1@(desktop|browser)@grounding\.action",
):
    """Desktop+browser grounding/action: full UI-TARS-1.5 action vocabulary.

    Currently NOT routed by env eval (env declares ``grounding.point``;
    see :class:`UITars15V1DesktopGroundingPointAdapter`). Kept for SFT
    replay (multi-action shape, full pyautogui-style action_space text).
    """
    action_space: UITars15V1DesktopActionSpace = dataclasses.field(
        default_factory=UITars15V1DesktopActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    user_prompt_template: str | None = GROUNDING_USER_PROMPT


@dataclasses.dataclass
class UITars15V1DesktopGroundingPointAdapter(
    UITars15V1BaseAdapter,
    key=r"ui_tars_15_v1@(desktop|browser)@grounding\.point",
):
    """Desktop+browser grounding (single-step click) for UI-TARS-1.5.

    Wire format is the SFT-aligned pyautogui-style text
    ``Action: click(start_box='(x,y)')``. Conversion layer is trimmed
    (:class:`UITars15V1DesktopGroundingPointActionSpace` routes
    ``click(start_box=)`` to cua-lite ``point(coord)``), and the advertised
    ``action_space_text`` is trimmed to the single native click form.
    Refusal lives on the env's ``report_infeasible`` extra tool.
    """
    action_space: UITars15V1DesktopGroundingPointActionSpace = dataclasses.field(
        default_factory=UITars15V1DesktopGroundingPointActionSpace
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
class UITars15V1DesktopUseAdapter(
    UITars15V1BaseAdapter,
    key=r"ui_tars_15_v1@(desktop|browser)@use",
):
    """Desktop+browser ``use`` (multi-step rollout): with system prompt, UITars-style history.

    Matches OSWorld's message structure:
      system: "You are a helpful assistant."
      user: UITARS_USR_PROMPT_THOUGHT (action space + instruction + notes)
      [history assistant/user messages...]
      user: [current image]
    """
    action_space: UITars15V1DesktopActionSpace = dataclasses.field(
        default_factory=UITars15V1DesktopActionSpace
    )
    protocol: UITarsHistoryProtocol = dataclasses.field(
        default_factory=lambda: UITarsHistoryProtocol(full_history_size=5)
    )
    system_prompt: str | None = USE_SYSTEM_PROMPT
    user_prompt_template: str | None = USE_USER_PROMPT


# =============================================================================
# Mobile Prompts
# =============================================================================

UITARS15_V1_MOBILE_ACTION_SPACE = """
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
class UITars15V1MobileGroundingActionAdapter(
    UITars15V1BaseAdapter,
    key="ui_tars_15_v1@mobile@grounding.action",
):
    """Mobile grounding/action: full mobile action vocabulary. SFT-replay only."""
    action_space: UITars15V1MobileActionSpace = dataclasses.field(
        default_factory=UITars15V1MobileActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    user_prompt_template: str | None = GROUNDING_USER_PROMPT
    action_space_text: str = UITARS15_V1_MOBILE_ACTION_SPACE


@dataclasses.dataclass
class UITars15V1MobileGroundingPointAdapter(
    UITars15V1BaseAdapter,
    key="ui_tars_15_v1@mobile@grounding.point",
):
    """Mobile grounding (single-step click) for UI-TARS-1.5. Conversion
    layer trim per the desktop variant; advertised ``action_space_text``
    is the single native click form.
    """
    action_space: UITars15V1MobileGroundingPointActionSpace = dataclasses.field(
        default_factory=UITars15V1MobileGroundingPointActionSpace
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
class UITars15V1MobileUseAdapter(
    UITars15V1BaseAdapter,
    key="ui_tars_15_v1@mobile@use",
):
    """Mobile ``use`` (multi-step rollout): with system prompt, UITars-style history. """
    action_space: UITars15V1MobileActionSpace = dataclasses.field(
        default_factory=UITars15V1MobileActionSpace
    )
    protocol: UITarsHistoryProtocol = dataclasses.field(
        default_factory=lambda: UITarsHistoryProtocol(full_history_size=5)
    )
    system_prompt: str | None = USE_SYSTEM_PROMPT
    user_prompt_template: str | None = MOBILE_USE_USER_PROMPT
    action_space_text: str = UITARS15_V1_MOBILE_ACTION_SPACE

# =============================================================================
# Pass-through Adapters (understanding)
# =============================================================================

AgentAdapterRegistry.register(r"ui_tars_15_v1@(desktop|browser|mobile)@understanding", AsIsAdapter)
# ``grounding.point`` and ``grounding.action`` are both served by concrete
# per-platform classes above.

# =============================================================================
# Grounding Adapters (bbox, point)
# =============================================================================

@dataclasses.dataclass
class UITars15V1GroundingBBoxAdapter(
    UITars15V1BaseAdapter,
    key=r"ui_tars_15_v1@(desktop|browser|mobile)@grounding\.bbox",
):
    """Grounding/bbox adapter for UITars15 v1.

    ``action_space_text`` is emptied for the reason on
    :class:`~lite.agents.models.ui_tars.adapter.UITarsGroundingBBoxAdapter`; the
    base advertises the v1 ``use`` vocabulary, whose only row this surface speaks
    is ``click`` carrying a FOUR-value box.
    """
    action_space: UITars15V1BBoxActionSpace = dataclasses.field(
        default_factory=UITars15V1BBoxActionSpace
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
# GroundingPointAdapter classes above (env-side trimmed UI-TARS-1.5 native
# ``click(start_box=)`` harness).
