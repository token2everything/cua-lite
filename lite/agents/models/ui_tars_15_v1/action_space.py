"""
UI-TARS 1.5 v1 Action Space (Desktop/Browser & Mobile)

Point-based action space matching the open-source UI-TARS-1.5-7B model format.
Uses separate function names (click, left_double, right_single, ...) with start_box/end_box
coordinate parameters in normalized [0, 1000] range.

Supports both structured tool_calls and text-based prompting via format_tool_call_as_text().

Usage:
    from lite.agents.models.ui_tars_15_v1.action_space import UITars15V1DesktopActionSpace, UITars15V1MobileActionSpace

    # Structured tool calls
    action_space = UITars15V1DesktopActionSpace()
    tools = action_space.get_tool_schemas()
    action = UITars15V1DesktopActionSpace.click(start_box=[500, 300])

    # Text-based prompting (no tools API needed)
    text = action_space.format_tool_call_as_text(action)  # "click(start_box='(500,300)')"
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
from lite.core.tools.action_space import merge_adjacent_lite_action_batches
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
)
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.schemas import tool

logger = logging.getLogger(__name__)

# The 1.5 grammar IS the original UI-TARS grammar (see
# /lite/agents/models/ui_tars/action_space.py) — only the text rendering and
# ``finished``'s signature differ — so its expressiveness gaps are the same ones,
# and they are declared the same way. Desktop has no key hold / down / up, no
# half-press mouse verb, no screenshot and no cursor query; ``mouse_move`` joins
# them because rewriting a hover into ``click`` would replay it as a real,
# side-effecting click. Mobile has no pinch, no screenshot and no wait.
# ``drag`` is NOT declared on mobile because the mobile prompt never advertises
# it — the advertised two-endpoint verb is ``scroll(start_box, end_box)``. A
# ``drag`` wire call is the model borrowing DESKTOP grammar and parses back to
# canonical ``swipe`` (see
# ``UITars15V1MobileActionSpace._convert_single_from_agent``), so the pair is
# asymmetric: canonical mobile ``drag`` has no advertised carrier and leaves the
# renderer through its unknown-name warning as a flat pass-through.
# The renderer and these tables are shared by every subclass — same grammar,
# different text rendering.
_DESKTOP_UNRENDERABLE_ACTIONS = frozenset({
    "cursor_position",
    "hold_key",
    "key_down",
    "key_up",
    "mouse_down",
    "mouse_move",
    "mouse_up",
    "screenshot",
})
_MOBILE_UNRENDERABLE_ACTIONS = frozenset({
    "pinch",
    "screenshot",
    "wait",
})


# =============================================================================
# Text Format Helpers
# =============================================================================

def format_tool_call_as_text(agent_tool_call: dict[str, Any]) -> str:
    """Render one UI-TARS 1.5 agent projection as wire text.

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
            # `<|box_end|>` tokens. Wrapping here on re-render caused a
            # round-trip drift between the raw response and the text placed
            # back into context on the next turn.
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
# UI-TARS 1.5 v1 Desktop Action Space
# =============================================================================

@dataclasses.dataclass
class UITars15V1DesktopActionSpace(BaseActionSpace, key=r"ui_tars_15_v1@(desktop|browser)"):
    """
    UI-TARS 1.5 v1 action space for desktop interactions.

    Uses normalized [0, 1000] coordinate system (same as CUA-lite).
    Actions are separate functions matching the UI-TARS text format:
        click, left_double, right_single, drag, hotkey, type, scroll, wait, finished.

    Example:
        action_space = UITars15V1DesktopActionSpace()
        tools = action_space.get_tool_schemas()
        action = UITars15V1DesktopActionSpace.click(start_box=[500, 300])
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
        # Subclasses inherit this method, so the fix travels with it.
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
    @tool(content="Answer or final output text. Use escape characters as needed.")
    def finished(content: str | None = None) -> dict[str, Any]:
        """Finish the current task, optionally providing a final answer."""
        return make_tool_call("finished", {"content": content})

    @staticmethod
    @tool()
    def call_user() -> dict[str, Any]:
        """Submit the task and call the user when the task is unsolvable or when you need help."""
        return make_tool_call("call_user")

    # -------------------------------------------------------------------------
    # Native action / extra-tool declaration
    # -------------------------------------------------------------------------
    # Flat schemas — the action layer and the extra-tool layer are
    # indistinguishable from class-visible data, so both are declared.
    # Canonical names are the ones ``_convert_single_from_agent`` PRODUCES:
    # ``mouse_move`` is absent because no native entry parses back to it (the
    # to-agent path renders it as ``click``, which parses to ``click``).
    # ``v2`` inherits both tables unchanged — same grammar, different text
    # rendering.

    LITE_ACTION_NAME_TO_UI_TARS_PROVIDER_FLAT_TOOL_NAMES = {
        "click": ["click", "left_double", "right_single"],
        "drag": ["drag"],
        "key": ["hotkey"],
        "type": ["type"],
        "scroll": ["scroll"],
        "wait": ["wait"],
    }
    #: Unlike the ORIGINAL ui_tars desktop grammar, 1.5's ``finished`` takes a
    #: ``content`` argument: ``finished(content="x")`` parses to ``response``,
    #: bare ``finished()`` to ``terminate``. It must survive whenever EITHER is
    #: active. ``call_user`` only ever parses to ``terminate``.
    UI_TARS_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES = {
        "finished": frozenset({"response", "terminate"}),
        "call_user": frozenset({"terminate"}),
    }

    # -------------------------------------------------------------------------
    # Tool Call Conversion
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single CUA-lite tool call to UITars v1 format."""
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
                return [UITars15V1DesktopActionSpace.right_single(start_box=coord)["function"]]
            elif clicks == 2:
                return [UITars15V1DesktopActionSpace.left_double(start_box=coord)["function"]]
            else:
                return [UITars15V1DesktopActionSpace.click(start_box=coord)["function"]]

        elif name == "type":
            return [UITars15V1DesktopActionSpace.type(content=args.get("text", ""))["function"]]

        elif name == "key":
            keys = args.get("keys", [])
            return [UITars15V1DesktopActionSpace.hotkey(key=" ".join(keys))["function"]]

        elif name == "scroll":
            direction = args.get("direction", "down")
            coord = args.get("coordinate")
            return [
                UITars15V1DesktopActionSpace.scroll(
                    start_box=coord,
                    direction=direction,
                )["function"]
            ]

        elif name == "drag":
            start = args.get("start_coordinate")
            end = args.get("coordinate")
            if start and end:
                return [UITars15V1DesktopActionSpace.drag(start_box=start, end_box=end)["function"]]
            elif end:
                return [UITars15V1DesktopActionSpace.drag(start_box=end, end_box=end)["function"]]
            return []

        elif name == "wait":
            return [UITars15V1DesktopActionSpace.wait()["function"]]

        elif name == "terminate":
            return [UITars15V1DesktopActionSpace.finished()["function"]]

        elif name == "response":
            return [UITars15V1DesktopActionSpace.finished(content=args.get("text", ""))["function"]]

        elif name in _DESKTOP_UNRENDERABLE_ACTIONS:
            raise ValueError(
                f"UI-TARS 1.5 desktop cannot render canonical tool {name!r}"
            )

        else:
            logger.warning("Unknown CUA-lite action for UITars15V1: %s(%s)", name, args)
            return [{"name": name, "arguments": args}]

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Convert UITars v1 tool_calls back to CUA-lite format.

        UITars v1 format:
            {"name": "click", "arguments": {"start_box": [500, 300]}}

        CUA-lite format:
            {"name": "click", "arguments": {"coordinate": [500, 300]}}
        """
        result = []
        for tc in agent_tool_calls:
            result.extend(self._convert_single_from_agent(tc))
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(self, agent_tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single UITars v1 tool call to CUA-lite format."""
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
            content = args.get("content")
            if content:
                return [LiteFinishToolSet.response(text=content)]
            return [LiteFinishToolSet.terminate(status="success")]

        elif name == "call_user":
            return [LiteFinishToolSet.terminate(status="failure", reason="call_user")]

        else:
            logger.warning("Unknown UITars15V1 action: %s(%s)", name, args)
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
# UI-TARS 1.5 v1 Mobile Action Space
# =============================================================================

@dataclasses.dataclass
class UITars15V1MobileActionSpace(BaseActionSpace, key="ui_tars_15_v1@mobile"):
    """
    UI-TARS 1.5 v1 action space for mobile interactions.

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
    @tool(content="Answer or final output text. Use escape characters as needed.")
    def finished(content: str | None = None) -> dict[str, Any]:
        """Submit the task regardless of whether it succeeds or fails."""
        return make_tool_call("finished", {"content": content})

    # -------------------------------------------------------------------------
    # Native action / extra-tool declaration
    # -------------------------------------------------------------------------
    # Flat schemas — see :class:`UITars15V1DesktopActionSpace`. ``drag`` is
    # parsed (as a swipe) but never ADVERTISED on mobile, so it is not declared
    # here: the tables describe the advertised native surface.

    LITE_ACTION_NAME_TO_UI_TARS_PROVIDER_FLAT_TOOL_NAMES = {
        "tap": ["click"],
        "long_press": ["long_press"],
        "type": ["type"],
        "swipe": ["scroll"],
        "system_button": ["press_home", "press_back"],
    }
    #: ``finished(content="x")`` -> ``response``; bare ``finished()`` ->
    #: ``terminate``. Survives whenever EITHER is active.
    UI_TARS_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES = {
        "finished": frozenset({"response", "terminate"}),
    }

    # -------------------------------------------------------------------------
    # Tool Call Conversion
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single CUA-lite mobile tool call to UITars 1.5 v1 mobile format."""
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
                    f"UI-TARS 1.5 mobile cannot render tap(clicks={clicks}): its "
                    "grammar has only 'click' and the wire carries no repeat count"
                )
            return [UITars15V1MobileActionSpace.click(start_box=args.get("coordinate"))["function"]]

        elif name == "long_press":
            coord = args.get("coordinate")
            duration = compact_number(args.get("duration"))
            time_str = str(duration) if duration is not None else None
            return [
                UITars15V1MobileActionSpace.long_press(
                    start_box=coord,
                    time=time_str,
                )["function"]
            ]

        elif name == "type":
            return [UITars15V1MobileActionSpace.type(content=args.get("text", ""))["function"]]

        elif name == "swipe":
            start = args.get("start_coordinate")
            end = args.get("coordinate")
            return [UITars15V1MobileActionSpace.scroll(start_box=start, end_box=end)["function"]]

        elif name == "system_button":
            btn = args.get("button", "")
            if btn == "Back":
                return [UITars15V1MobileActionSpace.press_back()["function"]]
            elif btn == "Home":
                return [UITars15V1MobileActionSpace.press_home()["function"]]
            raise ValueError(
                f"UI-TARS 1.5 mobile cannot render system_button(button={btn!r}): "
                "its grammar has only press_home and press_back"
            )

        elif name == "terminate":
            return [UITars15V1MobileActionSpace.finished()["function"]]

        elif name == "response":
            return [UITars15V1MobileActionSpace.finished(content=args.get("text", ""))["function"]]

        elif name in _MOBILE_UNRENDERABLE_ACTIONS:
            raise ValueError(
                f"UI-TARS 1.5 mobile cannot render canonical tool {name!r}"
            )

        else:
            logger.warning("Unknown CUA-lite mobile action for UITars15V1: %s(%s)", name, args)
            return [{"name": name, "arguments": args}]

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Convert UITars 1.5 v1 mobile tool_calls back to CUA-lite mobile format."""
        result = []
        for tc in agent_tool_calls:
            result.extend(self._convert_single_from_agent(tc))
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(self, agent_tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single UITars 1.5 v1 mobile tool call to CUA-lite mobile format."""
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

        elif name == "drag":
            # ``drag`` is NOT in the UI-TARS mobile prompt — the advertised
            # two-endpoint verb is ``scroll(start_box, end_box)`` — so a ``drag``
            # here is the model borrowing DESKTOP grammar, and every real
            # emission states scroll intent in its ``Thought``. Map it to
            # canonical ``swipe``, as upstream does.
            #
            # This is not cosmetic: four of five mobile envs dispatch
            # ``("swipe", "drag")`` through one handler, but mobilegym splits
            # them (``ActionType.SWIPE`` vs ``ActionType.DRAG`` in its
            # docker/server.py) into a fling and a press-move-release. Parsing
            # this as ``drag`` sends that env a gesture the model did not intend.
            start = args.get("start_coordinate", args.get("start_box"))
            end = args.get("coordinate", args.get("end_box"))
            return [LiteMobileActionSpace.swipe(
                start_coordinate=required_coord(
                    start, dimensions=2, name="start_coordinate",
                ),
                coordinate=required_coord(end, dimensions=2),
            )]

        elif name == "finished":
            content = args.get("content")
            if content:
                return [LiteFinishToolSet.response(text=content)]
            return [LiteFinishToolSet.terminate(status="success")]

        else:
            logger.warning("Unknown UITars15V1 mobile action: %s(%s)", name, args)
            return [make_tool_call(name, args)]

    # -------------------------------------------------------------------------
    # Text Format Support
    # -------------------------------------------------------------------------

    def format_tool_call_as_text(self, agent_tool_call: dict[str, Any]) -> str:
        """Render this family's bare ``{name, arguments}`` projection as text.

        Same narrowed input contract as
        :meth:`UITars15V1DesktopActionSpace.format_tool_call_as_text`.
        """
        return format_tool_call_as_text(agent_tool_call)


class UITars15V1BBoxActionSpace(LiteBBoxActionSpace):
    """UI-TARS 1.5 ``grounding.bbox`` surface, rendered and parsed as the family wire.

    The wire spelling is ``click(start_box='(x1,y1,x2,y2)')`` in BOTH directions:
    it is what ``UITars15V1BaseAdapter``'s parser hands back, and it is what
    :meth:`_convert_single_to_agent` projects the canonical ``bbox(coordinate=)``
    call to, so a rendered call re-parses to the call it came from.

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
        """Parse UI-TARS 1.5 bbox wire calls into canonical ``bbox(coordinate=)``.

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
# UI-TARS-1.5 Grounding (single-step click) action spaces
# =============================================================================
#
# Mirror of the ui_tars grounding harness. Same SFT-aligned pyautogui-style
# wire format ``click(start_box='(x,y)')`` but trimmed to a single function;
# round-trips cua-lite ``LitePointActionSpace.point(coord)``.

GROUNDING_DESKTOP_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
"""

GROUNDING_MOBILE_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
"""


@dataclasses.dataclass
class UITars15V1DesktopGroundingPointActionSpace(UITars15V1DesktopActionSpace, key=r"ui_tars_15_v1@(desktop|browser)@point"):
    """UI-TARS-1.5 desktop grounding (single-step click)."""

    # Both tables are RESET, not inherited: on this trimmed harness only
    # ``click`` converts — every other parent entry passes through
    # UNCONVERTED, so inheriting would be a FALSE declaration.
    LITE_ACTION_NAME_TO_UI_TARS_PROVIDER_FLAT_TOOL_NAMES = {"point": ["click"]}
    UI_TARS_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES = {}
    _LITE_FORMAT_ACTION_NAME = next(iter(LitePointActionSpace.get_action_names()))

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)
        if name == "point":
            return [UITars15V1DesktopActionSpace.click(start_box=args.get("coordinate"))["function"]]
        return convert_non_point_call_for_grounding_space(
            tool_call, surface="UI-TARS-1.5 grounding (point) action_space",
        )

    def _convert_single_from_agent(self, agent_tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """``click(start_box=)`` → ``point(coord)``; other names pass through."""
        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]
        if name == "click":
            start_box = required_coord(args.get("start_box"), dimensions=2, name="start_box")
            return [LitePointActionSpace.point(coordinate=start_box)]
        return [make_tool_call(name, args)]


@dataclasses.dataclass
class UITars15V1MobileGroundingPointActionSpace(UITars15V1MobileActionSpace, key="ui_tars_15_v1@mobile@point"):
    """UI-TARS-1.5 mobile grounding (single-step click). Same harness as desktop."""

    # Reset, not inherited — see
    # :class:`UITars15V1DesktopGroundingPointActionSpace`.
    LITE_ACTION_NAME_TO_UI_TARS_PROVIDER_FLAT_TOOL_NAMES = {"point": ["click"]}
    UI_TARS_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES = {}
    _LITE_FORMAT_ACTION_NAME = next(iter(LitePointActionSpace.get_action_names()))

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)
        if name == "point":
            return [UITars15V1MobileActionSpace.click(start_box=args.get("coordinate"))["function"]]
        return convert_non_point_call_for_grounding_space(
            tool_call, surface="UI-TARS-1.5 mobile grounding (point) action_space",
        )

    def _convert_single_from_agent(self, agent_tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        """``click(start_box=)`` → ``point(coord)``; other names pass through."""
        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]
        if name == "click":
            start_box = required_coord(args.get("start_box"), dimensions=2, name="start_box")
            return [LitePointActionSpace.point(coordinate=start_box)]
        return [make_tool_call(name, args)]
