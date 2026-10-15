"""
UI-TARS Action Space (Desktop/Browser & Mobile)

Point-based action space matching the original UI-TARS-7B-DPO model format.
Uses start_box/end_box coordinate parameters in normalized [0, 1000] range.

Key differences from ui_tars_15_v1:
- Desktop: finished() has no content parameter, includes call_user() action
- Mobile: finished(content=''), long_press, press_home, press_back, scroll with end_box

Usage:
    from lite.agents.models.ui_tars.action_space import UITarsDesktopActionSpace, UITarsMobileActionSpace
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Literal

from lite.agents.core.action_space.base import (
    BaseActionSpace,
    LiteBBoxActionSpace,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.agents.core.action_space.utils.geometry import (
    compact_number,
    optional_coord,
    required_coord,
)
from lite.agents.core.action_space.utils.grounding_point import (
    convert_non_point_call_for_grounding_space,
)
from lite.core.tools.action_space import (
    LiteDesktopActionSet,
    LiteMobileActionSet,
    make_lite_action_batch_call,
    merge_adjacent_lite_action_batches,
)
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
)
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.schemas import tool

logger = logging.getLogger(__name__)

# Canonical Lite GUI action catalogs are owned by the core action spaces.
# Original UI-TARS can express only a subset of those semantics natively:
# desktop has no key hold/down/up, mouse down/up/move, screenshot, or cursor
# query; mobile has no drag, pinch, wait, or screenshot. Keep only the
# convertible canonical outputs for wrapping into ``computer`` / ``mobile``.
_DESKTOP_UNSUPPORTED_ACTIONS = frozenset({
    "cursor_position",
    "hold_key",
    "key_down",
    "key_up",
    "mouse_down",
    "mouse_move",
    "mouse_up",
    "screenshot",
})
_MOBILE_UNSUPPORTED_ACTIONS = frozenset({
    "drag",
    "pinch",
    "screenshot",
    "wait",
})
_DESKTOP_ACTIONS = (
    LiteDesktopActionSet.get_action_names() - _DESKTOP_UNSUPPORTED_ACTIONS
)
_MOBILE_ACTIONS = (
    LiteMobileActionSet.get_action_names() - _MOBILE_UNSUPPORTED_ACTIONS
)


def _wrap_action_call(call: dict[str, Any], wrapper: str, actions: frozenset[str]) -> dict[str, Any]:
    name = tool_call_name(call)
    if name not in actions:
        return call
    return make_lite_action_batch_call(wrapper, call)


# =============================================================================
# Text Format Helpers
# =============================================================================

def format_tool_call_as_text(agent_tool_call: dict[str, Any]) -> str:
    """Render one UI-TARS agent projection as wire text.

    Input shape is part of the contract: this takes the family's BARE
    ``{name, arguments}`` projection — what ``convert_tool_calls_to_agent``
    returns — not a canonical Lite call. A canonical call raises ``KeyError``
    here, the mirror of the failure the base renderer documents for a bare call.

    Returns:
        Text string like ``click(start_box='(500,300)')``
    """
    name = agent_tool_call["name"]
    args = agent_tool_call["arguments"]

    parts = []
    for k, v in args.items():
        if k in ("start_box", "end_box") and isinstance(v, list):
            # Match raw model output: bare `(x,y)` without `<|box_start|>` /
            # `<|box_end|>` tokens. The action-space description prompt still
            # shows the token grammar (the model tolerates but does not emit
            # them) — wrapping here on re-render caused a round-trip drift
            # between the raw response and the text placed back into context.
            #
            # EVERY value is rendered, never just the first two: this wire
            # format's boxes are (x, y), so a longer list is malformed, and
            # printing its first two entries would put a well-formed-looking
            # point back into model context after the parser had rejected it.
            values = ",".join(str(c) for c in v)
            parts.append(f"{k}='({values})'")
        elif isinstance(v, str):
            parts.append(f"{k}='{v}'")
        else:
            parts.append(f"{k}={v!r}")

    return f"{name}({', '.join(parts)})"


# =============================================================================
# UI-TARS Desktop Action Space
# =============================================================================

@dataclasses.dataclass
class UITarsDesktopActionSpace(BaseActionSpace, key=r"ui_tars@(desktop|browser)"):
    """
    UI-TARS action space for desktop interactions (original UI-TARS-7B-DPO).

    Uses normalized [0, 1000] coordinate system.
    Actions are separate functions matching the UI-TARS text format:
        click, left_double, right_single, drag, hotkey, type, scroll, wait, finished, call_user.

    Example:
        action_space = UITarsDesktopActionSpace()
        tools = action_space.get_tool_schemas()
        action = UITarsDesktopActionSpace.click(start_box=[500, 300])
    """

    platform: str = "desktop"

    # -------------------------------------------------------------------------
    # Mouse Click Actions
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(start_box="(x, y) coordinates normalized to [0, 1000].")
    def click(start_box: list[int]) -> dict[str, Any]:
        """Click the left mouse button at the specified coordinates."""
        return make_tool_call("click", {"start_box": start_box})

    @staticmethod
    @tool(start_box="(x, y) coordinates normalized to [0, 1000].")
    def left_double(start_box: list[int]) -> dict[str, Any]:
        """Double-click the left mouse button at the specified coordinates."""
        return make_tool_call("left_double", {"start_box": start_box})

    @staticmethod
    @tool(start_box="(x, y) coordinates normalized to [0, 1000].")
    def right_single(start_box: list[int]) -> dict[str, Any]:
        """Click the right mouse button at the specified coordinates."""
        return make_tool_call("right_single", {"start_box": start_box})

    # -------------------------------------------------------------------------
    # Drag
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(
        start_box="Starting (x, y) coordinates normalized to [0, 1000].",
        end_box="Ending (x, y) coordinates normalized to [0, 1000].",
    )
    def drag(start_box: list[int], end_box: list[int]) -> dict[str, Any]:
        """Drag from start coordinates to end coordinates."""
        return make_tool_call("drag", {"start_box": start_box, "end_box": end_box})

    # -------------------------------------------------------------------------
    # Keyboard Actions
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(key="Space-separated key names to press together (e.g. 'ctrl c').")
    def hotkey(key: str) -> dict[str, Any]:
        """Press a keyboard hotkey combination."""
        return make_tool_call("hotkey", {"key": key})

    @staticmethod
    @tool(content="The text to type. Use '\\n' at the end to submit.")
    def type(content: str) -> dict[str, Any]:
        """type text content into the currently focused input field."""
        return make_tool_call("type", {"content": content})

    # -------------------------------------------------------------------------
    # Scroll
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(
        start_box="(x, y) coordinates to scroll at, normalized to [0, 1000].",
        direction="The direction to scroll.",
    )
    def scroll(
        start_box: list[int] | None = None,
        direction: Literal["up", "down", "left", "right"] = "down",
    ) -> dict[str, Any]:
        """Scroll in the specified direction at the given coordinates."""
        # Deliberately NOT passing _DEFAULTS — the default-filter would drop
        # `direction='down'` from the structured tool_call, which makes the
        # re-rendered text (next-turn context) lose the direction entirely.
        return make_tool_call(
            "scroll",
            {"start_box": start_box, "direction": direction},
        )

    # -------------------------------------------------------------------------
    # Utility Actions
    # -------------------------------------------------------------------------

    @staticmethod
    @tool()
    def wait() -> dict[str, Any]:
        """Wait for 5 seconds and take a screenshot to check for changes."""
        return make_tool_call("wait")

    @staticmethod
    @tool()
    def finished() -> dict[str, Any]:
        """Finish the current task."""
        return make_tool_call("finished")

    @staticmethod
    @tool()
    def call_user() -> dict[str, Any]:
        """Submit the task and call the user when the task is unsolvable or when you need help."""
        return make_tool_call("call_user")

    # -------------------------------------------------------------------------
    # Native action / extra-tool declaration
    # -------------------------------------------------------------------------
    # UI-TARS emits FLAT schemas, so the action layer and the extra-tool layer
    # are indistinguishable from class-visible data ``[measured: the generated
    # schemas for ``wait`` (an ACTION) and ``finished``/``call_user`` (finish
    # TOOLS) are byte-identical modulo name and description]``. Declared here.
    #
    # Canonical names are the ones ``_convert_single_from_agent`` PRODUCES:
    # note ``mouse_move`` is absent — the desktop grammar cannot render it and
    # no native entry parses back to it.

    LITE_ACTION_NAME_TO_UI_TARS_PROVIDER_FLAT_TOOL_NAMES = {
        "click": ["click", "left_double", "right_single"],
        "drag": ["drag"],
        "key": ["hotkey"],
        "type": ["type"],
        "scroll": ["scroll"],
        "wait": ["wait"],
    }
    #: The ORIGINAL desktop ``finished()`` takes NO ``content`` argument, so it
    #: parses to ``terminate`` either way and DISCARDS any answer text — unlike
    #: the 1.5 desktop / mobile grammars, where it also spells ``response``.
    UI_TARS_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES = {
        "finished": frozenset({"terminate"}),
        "call_user": frozenset({"terminate"}),
    }

    # -------------------------------------------------------------------------
    # Tool Call Conversion
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single CUA-lite tool call to UITars format."""

        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        if name == "computer":
            result = []
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                result.extend(self._convert_single_to_agent(make_tool_call(action, action_args)))
            return result

        if name == "click":
            coord = args.get("coordinate")
            button = args.get("button", "left")
            clicks = args.get("clicks", 1)
            if button == "right":
                return [UITarsDesktopActionSpace.right_single(start_box=coord)["function"]]
            elif clicks == 2:
                return [UITarsDesktopActionSpace.left_double(start_box=coord)["function"]]
            else:
                return [UITarsDesktopActionSpace.click(start_box=coord)["function"]]

        elif name == "type":
            return [UITarsDesktopActionSpace.type(content=args.get("text", ""))["function"]]

        elif name == "key":
            keys = args.get("keys", [])
            return [UITarsDesktopActionSpace.hotkey(key=" ".join(keys))["function"]]

        elif name == "scroll":
            direction = args.get("direction", "down")
            coord = args.get("coordinate")
            return [
                UITarsDesktopActionSpace.scroll(start_box=coord, direction=direction)["function"]
            ]

        elif name == "drag":
            start = args.get("start_coordinate")
            end = args.get("coordinate")
            if start and end:
                return [UITarsDesktopActionSpace.drag(start_box=start, end_box=end)["function"]]
            elif end:
                return [UITarsDesktopActionSpace.drag(start_box=end, end_box=end)["function"]]
            return []

        elif name == "mouse_move":
            raise ValueError("UI-TARS desktop cannot render canonical tool 'mouse_move'")

        elif name == "wait":
            return [UITarsDesktopActionSpace.wait()["function"]]

        elif name == "terminate":
            return [UITarsDesktopActionSpace.finished()["function"]]

        elif name == "response":
            # UI-TARS desktop ``finished()`` takes NO ``content`` argument (only
            # the mobile grammar has ``finished(content=...)``), so rendering
            # ``response`` here would DISCARD the answer text irrecoverably —
            # ``finished`` parses back to ``terminate``. Fail loudly instead.
            raise ValueError("UI-TARS desktop cannot render canonical tool 'response'")

        else:
            # Canonical Lite names that this grammar cannot spell must fail
            # while rendering; otherwise the action would disappear silently.
            if name in LiteDesktopActionSet.get_action_names():
                raise ValueError(f"UI-TARS desktop cannot render canonical tool {name!r}")
            logger.warning("Unknown CUA-lite action for UITars: %s(%s)", name, args)
            return [{"name": name, "arguments": args}]

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Convert UITars tool_calls back to CUA-lite format.

        UITars format:
            {"name": "click", "arguments": {"start_box": [500, 300]}}

        CUA-lite format:
            {"name": "click", "arguments": {"coordinate": [500, 300]}}
        """
        result = []
        for tc in agent_tool_calls:
            converted = self._convert_single_from_agent(tc)
            result.extend(
                _wrap_action_call(call, "computer", _DESKTOP_ACTIONS)
                for call in converted
            )
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(self, agent_tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single UITars tool call to CUA-lite format."""

        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]

        if name == "click":
            start_box = required_coord(args.get("start_box"), dimensions=2, name="start_box")
            return [LiteDesktopActionSpace.click(coordinate=start_box)]

        elif name == "left_double":
            start_box = required_coord(args.get("start_box"), dimensions=2, name="start_box")
            return [LiteDesktopActionSpace.click(coordinate=start_box, clicks=2)]

        elif name == "right_single":
            start_box = required_coord(args.get("start_box"), dimensions=2, name="start_box")
            return [LiteDesktopActionSpace.click(coordinate=start_box, button="right")]

        elif name == "drag":
            start_box = required_coord(args.get("start_box"), dimensions=2, name="start_box")
            end_box = required_coord(args.get("end_box"), dimensions=2, name="end_box")
            return [LiteDesktopActionSpace.drag(
                coordinate=end_box,
                start_coordinate=start_box,
            )]

        elif name == "hotkey":
            key_str = args.get("key", "")
            if not isinstance(key_str, str):
                raise ValueError("UI-TARS hotkey requires key string")
            parts = key_str.split()
            keys = parts if len(parts) > 1 else key_str
            return [LiteDesktopActionSpace.key(keys=keys)]

        elif name == "type":
            return [LiteDesktopActionSpace.type(text=args.get("content", ""))]

        elif name == "scroll":
            direction = args.get("direction", "down")
            return [LiteDesktopActionSpace.scroll(
                direction=direction,
                amount=5,
                coordinate=optional_coord(args.get("start_box"), dimensions=2),
            )]

        elif name == "wait":
            return [LiteDesktopActionSpace.wait(duration=5)]

        elif name == "finished":
            return [LiteFinishToolSet.terminate(status="success")]

        elif name == "call_user":
            return [LiteFinishToolSet.terminate(status="failure", reason="call_user")]

        else:
            # Parse-side provider output is kept intact; env feedback owns
            # unknown executable names.
            if name not in LiteDesktopActionSet.get_action_names():
                logger.warning("Unknown UITars action: %s(%s)", name, args)
            return [make_tool_call(name, args)]

    # -------------------------------------------------------------------------
    # Text Format Support
    # -------------------------------------------------------------------------

    def format_tool_call_as_text(self, agent_tool_call: dict[str, Any]) -> str:
        """Render this family's bare ``{name, arguments}`` projection as text.

        Narrows the base contract on purpose: the input is what
        :meth:`convert_tool_calls_to_agent` produced, never a canonical Lite
        call. ``format_tool_calls_as_text`` inherits that choice element-wise.
        """
        return format_tool_call_as_text(agent_tool_call)


# =============================================================================
# UI-TARS Mobile Action Space
# =============================================================================

@dataclasses.dataclass
class UITarsMobileActionSpace(BaseActionSpace, key="ui_tars@mobile"):
    """
    UI-TARS action space for mobile interactions (original UI-TARS-7B-DPO).

    Uses normalized [0, 1000] coordinate system.
    Actions match the mobile prompt format:
        click, long_press, type, scroll, press_home, press_back, finished.
    """

    platform: str = "mobile"

    # -------------------------------------------------------------------------
    # Touch Actions
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(start_box="(x, y) coordinates normalized to [0, 1000].")
    def click(start_box: list[int]) -> dict[str, Any]:
        """Tap at the specified coordinates."""
        return make_tool_call("click", {"start_box": start_box})

    @staticmethod
    @tool(
        start_box="(x, y) coordinates normalized to [0, 1000].",
        time="Duration of the long press.",
    )
    def long_press(start_box: list[int], time: str | None = None) -> dict[str, Any]:
        """Long press at the specified coordinates."""
        return make_tool_call(
            "long_press",
            {"start_box": start_box, "time": time},
        )

    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(content="The text to type.")
    def type(content: str) -> dict[str, Any]:
        """type text content into the currently focused input field."""
        return make_tool_call("type", {"content": content})

    # -------------------------------------------------------------------------
    # Scroll (swipe-style with start/end coordinates)
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(
        start_box="Starting (x, y) coordinates normalized to [0, 1000].",
        end_box="Ending (x, y) coordinates normalized to [0, 1000].",
    )
    def scroll(start_box: list[int], end_box: list[int]) -> dict[str, Any]:
        """Scroll (swipe) from start coordinates to end coordinates."""
        return make_tool_call(
            "scroll",
            {"start_box": start_box, "end_box": end_box},
        )

    # -------------------------------------------------------------------------
    # Device Control
    # -------------------------------------------------------------------------

    @staticmethod
    @tool()
    def press_home() -> dict[str, Any]:
        """Press the Home button."""
        return make_tool_call("press_home")

    @staticmethod
    @tool()
    def press_back() -> dict[str, Any]:
        """Press the Back button."""
        return make_tool_call("press_back")

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(content="Answer or final output text.")
    def finished(content: str | None = None) -> dict[str, Any]:
        """Submit the task regardless of whether it succeeds or fails."""
        return make_tool_call("finished", {"content": content})

    # -------------------------------------------------------------------------
    # Native action / extra-tool declaration
    # -------------------------------------------------------------------------
    # Flat schemas — see :class:`UITarsDesktopActionSpace` for why these are
    # declared rather than derived.

    LITE_ACTION_NAME_TO_UI_TARS_PROVIDER_FLAT_TOOL_NAMES = {
        "tap": ["click"],
        "long_press": ["long_press"],
        "type": ["type"],
        "swipe": ["scroll"],
        "system_button": ["press_home", "press_back"],
    }
    #: ``finished(content="x")`` parses to ``response`` and bare ``finished()``
    #: to ``terminate``, so the entry spells BOTH — it must survive whenever
    #: EITHER is active, or the family loses its only terminal verb.
    UI_TARS_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES = {
        "finished": frozenset({"response", "terminate"}),
    }

    # -------------------------------------------------------------------------
    # Tool Call Conversion
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single CUA-lite mobile tool call to UITars mobile format."""
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        if name == "mobile":
            result = []
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                result.extend(self._convert_single_to_agent(make_tool_call(action, action_args)))
            return result

        if name == "tap":
            # The mobile grammar has ``click`` and no ``left_double`` (the
            # DESKTOP grammar does), so ``clicks`` has no carrier. Rendering
            # ``tap(clicks=2)`` as a plain ``click`` re-parses as
            # ``tap(clicks=1)``: a SINGLE tap where a double tap was asked for,
            # which is a different gesture rather than a coarser one, and one the
            # caller cannot recover -- every retry degrades identically. Raise
            # rather than silently substitute, as ``fara@desktop`` does for the
            # same ``clicks`` loss on ``click``.
            clicks = args.get("clicks", 1)
            if clicks != 1:
                raise ValueError(
                    f"UI-TARS mobile cannot render tap(clicks={clicks}): its "
                    "grammar has only 'click' and the wire carries no repeat count"
                )
            return [UITarsMobileActionSpace.click(start_box=args.get("coordinate"))["function"]]

        elif name == "long_press":
            coord = args.get("coordinate")
            duration = compact_number(args.get("duration"))
            time_str = str(duration) if duration is not None else None
            return [UITarsMobileActionSpace.long_press(start_box=coord, time=time_str)["function"]]

        elif name == "type":
            return [UITarsMobileActionSpace.type(content=args.get("text", ""))["function"]]

        elif name == "swipe":
            start = args.get("start_coordinate")
            end = args.get("coordinate")
            return [UITarsMobileActionSpace.scroll(start_box=start, end_box=end)["function"]]

        elif name == "system_button":
            btn = args.get("button", "")
            if btn == "Back":
                return [UITarsMobileActionSpace.press_back()["function"]]
            elif btn == "Home":
                return [UITarsMobileActionSpace.press_home()["function"]]
            else:
                raise ValueError(
                    f"UI-TARS mobile cannot render system_button {btn!r}; "
                    "expected 'Back' or 'Home'"
                )

        elif name == "terminate":
            return [UITarsMobileActionSpace.finished()["function"]]

        elif name == "response":
            return [UITarsMobileActionSpace.finished(content=args.get("text", ""))["function"]]

        else:
            # Canonical Lite names that this grammar cannot spell must fail
            # while rendering; otherwise the action would disappear silently.
            if name in LiteMobileActionSet.get_action_names():
                raise ValueError(f"UI-TARS mobile cannot render canonical tool {name!r}")
            logger.warning("Unknown CUA-lite mobile action for UITars: %s(%s)", name, args)
            return [{"name": name, "arguments": args}]

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Convert UITars mobile tool_calls back to CUA-lite mobile format."""
        result = []
        for tc in agent_tool_calls:
            converted = self._convert_single_from_agent(tc)
            result.extend(
                _wrap_action_call(call, "mobile", _MOBILE_ACTIONS)
                for call in converted
            )
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(self, agent_tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single UITars mobile tool call to CUA-lite mobile format."""
        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]

        if name == "click":
            start_box = required_coord(args.get("start_box"), dimensions=2, name="start_box")
            return [LiteMobileActionSpace.tap(coordinate=start_box)]

        elif name == "long_press":
            time_str = args.get("time")
            start_box = required_coord(args.get("start_box"), dimensions=2, name="start_box")
            return [LiteMobileActionSpace.long_press(
                coordinate=start_box,
                duration=float(time_str) if time_str else None,
            )]

        elif name == "type":
            return [LiteMobileActionSpace.type(text=args.get("content", ""))]

        elif name == "scroll":
            start_box = required_coord(args.get("start_box"), dimensions=2, name="start_box")
            end_box = required_coord(args.get("end_box"), dimensions=2, name="end_box")
            return [LiteMobileActionSpace.swipe(
                start_coordinate=start_box,
                coordinate=end_box,
            )]

        elif name == "press_home":
            return [LiteMobileActionSpace.system_button(button="Home")]

        elif name == "press_back":
            return [LiteMobileActionSpace.system_button(button="Back")]

        elif name == "finished":
            content = args.get("content")
            if content:
                return [LiteFinishToolSet.response(text=content)]
            return [LiteFinishToolSet.terminate(status="success")]

        else:
            # Parse-side provider output is kept intact; env feedback owns
            # unknown executable names.
            if name not in LiteMobileActionSet.get_action_names():
                logger.warning("Unknown UITars mobile action: %s(%s)", name, args)
            return [make_tool_call(name, args)]

    # -------------------------------------------------------------------------
    # Text Format Support
    # -------------------------------------------------------------------------

    def format_tool_call_as_text(self, agent_tool_call: dict[str, Any]) -> str:
        """Render this family's bare ``{name, arguments}`` projection as text.

        Same narrowed input contract as
        :meth:`UITarsDesktopActionSpace.format_tool_call_as_text`.
        """
        return format_tool_call_as_text(agent_tool_call)


class UITarsBBoxActionSpace(LiteBBoxActionSpace):
    """UI-TARS ``grounding.bbox`` surface, rendered and parsed as the family wire.

    The wire spelling is ``click(start_box='(x1,y1,x2,y2)')`` in BOTH directions:
    it is what ``UITarsBaseAdapter._parse_action_text`` hands back, and it is
    what :meth:`_convert_single_to_agent` projects the canonical
    ``bbox(coordinate=)`` call to, so a rendered call re-parses to the call it
    came from.

    ``bbox`` is still accepted on the parse side because it is the name of the
    tool this space advertises, which is what a structured tools-API response
    carries. Both names go through the coordinate parser — the base
    bare-projection path would forward either with whatever the model put in the
    box.
    """

    def format_tool_call_as_text(self, agent_tool_call: dict[str, Any]) -> str:
        """Render this family's bare ``{name, arguments}`` projection as text."""
        return format_tool_call_as_text(agent_tool_call)

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Canonical ``bbox(coordinate=)`` → the family wire ``click(start_box=)``.

        Names this space does not own — an env extra such as
        ``report_infeasible`` — keep the base bare projection, mirroring the
        pass-through in :meth:`convert_tool_calls_from_agent`.
        """
        if tool_call_name(tool_call) == "bbox":
            coordinate = tool_call_arguments(tool_call)["coordinate"]
            return [{"name": "click", "arguments": {"start_box": list(coordinate)}}]
        return super()._convert_single_to_agent(tool_call)

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Parse UI-TARS bbox wire calls into canonical ``bbox(coordinate=)``.

        Names other than the two box spellings pass through untouched, so an
        env extra such as ``report_infeasible`` still reaches the environment —
        the same split the grounding point spaces make.
        """
        result = []
        for agent_tool_call in agent_tool_calls:
            name = agent_tool_call["name"]
            args = agent_tool_call["arguments"]
            if name == "click":
                coordinate = required_coord(
                    args.get("start_box"), dimensions=4, name="start_box"
                )
            elif name == "bbox":
                coordinate = required_coord(args.get("coordinate"), dimensions=4)
            else:
                result.append(make_tool_call(name, args))
                continue
            result.append(LiteBBoxActionSpace.bbox(coordinate=coordinate))
        return result


# =============================================================================
# UI-TARS Grounding (single-step click) action spaces
# =============================================================================
#
# Trimmed harness for env-side grounding eval (osworld_g, screenspot_pro).
# Wire format is the SFT-aligned UI-TARS pyautogui-style text
# ``click(start_box='(x,y)')`` — same as ``use``, but the action space
# advertises only ``click`` and routes ``click(start_box=)`` ↔ cua-lite
# ``LitePointActionSpace.point(coordinate=)`` (NOT
# ``LiteDesktopActionSpace.click(coordinate=)``). Refusal is handled by the
# env's ``report_infeasible`` extra tool.

# Action-space text trimmed to a single line (UI-TARS reads this verbatim
# from the user prompt; see ``GROUNDING_USER_PROMPT`` in adapter.py).
GROUNDING_DESKTOP_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
"""

GROUNDING_MOBILE_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
"""


@dataclasses.dataclass
class UITarsDesktopGroundingPointActionSpace(UITarsDesktopActionSpace, key=r"ui_tars@(desktop|browser)@point"):
    """UI-TARS desktop grounding (single-step click).

    Wire format inherits the parent's pyautogui-style text
    (``click(start_box='(x,y)')``). The conversion paths are overridden so
    the cua-lite side speaks the canonical grounding shape
    (:class:`LitePointActionSpace.point`) instead of nav-style
    :class:`LiteDesktopActionSpace.click`.
    """

    # Both tables are RESET, not inherited: on this trimmed harness only
    # ``click`` converts — ``finished`` / ``call_user`` / ``hotkey`` / ``drag``
    # / ``type`` / ``scroll`` / ``wait`` all pass through UNCONVERTED, so
    # inheriting the parent's tables would be a FALSE declaration.
    LITE_ACTION_NAME_TO_UI_TARS_PROVIDER_FLAT_TOOL_NAMES = {"point": ["click"]}
    UI_TARS_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES = {}
    _LITE_FORMAT_ACTION_NAME = next(iter(LitePointActionSpace.get_action_names()))

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """cua-lite ``point(coord)`` → UI-TARS ``click(start_box=coord)``."""
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)
        if name == "point":
            return [UITarsDesktopActionSpace.click(start_box=args.get("coordinate"))["function"]]
        return convert_non_point_call_for_grounding_space(
            tool_call, surface="UI-TARS grounding (point) action_space",
        )

    def _convert_single_from_agent(self, agent_tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """UI-TARS ``click(start_box=coord)`` → cua-lite ``point(coord)``.

        Anything other than ``click`` (including extra tools like
        ``report_infeasible``) passes through verbatim — same pattern as
        :class:`Qwen3VLDesktopActionSpace._convert_single_from_agent` for
        non-``computer_use`` names.
        """
        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]
        if name == "click":
            start_box = required_coord(args.get("start_box"), dimensions=2, name="start_box")
            return [LitePointActionSpace.point(coordinate=start_box)]
        return [make_tool_call(name, args)]


@dataclasses.dataclass
class UITarsMobileGroundingPointActionSpace(UITarsMobileActionSpace, key="ui_tars@mobile@point"):
    """UI-TARS mobile grounding (single-step click)."""

    # Reset, not inherited — see
    # :class:`UITarsDesktopGroundingPointActionSpace`.
    LITE_ACTION_NAME_TO_UI_TARS_PROVIDER_FLAT_TOOL_NAMES = {"point": ["click"]}
    UI_TARS_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES = {}
    _LITE_FORMAT_ACTION_NAME = next(iter(LitePointActionSpace.get_action_names()))

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """cua-lite ``point(coord)`` → UI-TARS ``click(start_box=coord)``."""
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)
        if name == "point":
            return [UITarsMobileActionSpace.click(start_box=args.get("coordinate"))["function"]]
        return convert_non_point_call_for_grounding_space(
            tool_call, surface="UI-TARS mobile grounding (point) action_space",
        )

    def _convert_single_from_agent(self, agent_tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """UI-TARS ``click(start_box=coord)`` → cua-lite ``point(coord)``.

        Non-``click`` names (extra tools, off-schema actions) pass through.
        """
        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]
        if name == "click":
            start_box = required_coord(args.get("start_box"), dimensions=2, name="start_box")
            return [LitePointActionSpace.point(coordinate=start_box)]
        return [make_tool_call(name, args)]
