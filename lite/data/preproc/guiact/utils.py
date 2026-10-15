"""Shared helpers for the GUIAct preproc adapters.

Both per-task scripts (``grounding-action.py`` for web-single, ``use.py``
for web-multi + smartphone) share the same wiring against
:mod:`lite.data.staging` plus the GUIAct-specific coordinate parsing and
source→``tool_calls`` action converters.

Wiring is the shared :class:`~lite.data.preproc.common.SourceStaging` default.
GUIAct ships its own train/test split, and its task scripts hand it over as a
transient ``metadata.others.split`` hint: the shared splitter's ``canonical_fn``
reads the hint (so the hash split never runs — ``test`` maps to ``validation``
via the upstream map) and the shared ``stage_entry`` pops it, so it never lands
in the persisted parquet. That used to be a private copy of the four staging
helpers, kept only for the pop.

Action conversion:

* coordinates arrive as ``<box>`` / ``<point>`` strings of [0, 1] floats in
  the source ``related`` fields → converted to [0, 1000] integers after
  rejecting true out-of-range values,
* web actions → :class:`LiteDesktopActionSet` tool_calls,
* smartphone actions → :class:`LiteMobileActionSet` tool_calls.

Run via the task scripts; this module has no entry point.
"""

from __future__ import annotations

import re
from typing import Any

from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    MAX_NORM,
    LiteDesktopActionSet,
    LiteMobileActionSet,
    clamp_norm,
    merge_adjacent_lite_action_batches,
)
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
    validate_lite_tool_call,
)
from lite.data.preproc.common import SourceStaging

DATASET_NAME = "GUIAct"
_COORD_EPS = 1e-6


# =============================================================================
# Staging wiring
# =============================================================================

_STAGING = SourceStaging(DATASET_NAME)

out_dir_for = _STAGING.out_dir_for
make_image_store = _STAGING.make_image_store
make_splitter = _STAGING.make_splitter
stage_entry = _STAGING.stage_entry


# =============================================================================
# Errors
# =============================================================================

class SkipTrajectoryError(Exception):
    """Raised when a record/episode should be skipped (ambiguous/malformed)."""


# =============================================================================
# Coordinate parsing — source ``related`` fields are [0, 1] floats
# =============================================================================

_BOX_RE = re.compile(r"<box>\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*</box>")
_POINT_RE = re.compile(r"<point>\s*([\d.]+)\s*,\s*([\d.]+)\s*</point>")


def _require_unit_interval(*values: float) -> None:
    if any(v < -_COORD_EPS or v > 1 + _COORD_EPS for v in values):
        joined = ", ".join(str(v) for v in values)
        raise ValueError(f"coordinate outside [0, 1]: {joined}")


def parse_related_bbox(s: str) -> tuple[float, float, float, float]:
    """Parse '<box>x1, y1, x2, y2</box>' -> (x1, y1, x2, y2) floats in [0, 1]."""
    m = _BOX_RE.search(s)
    if not m:
        raise ValueError(f"Cannot parse bbox from: {s!r}")
    bbox = (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
    _require_unit_interval(*bbox)
    return bbox


def parse_related_point(s: str) -> tuple[float, float]:
    """Parse '<point>x, y</point>' -> (x, y) floats in [0, 1]."""
    m = _POINT_RE.search(s)
    if not m:
        raise ValueError(f"Cannot parse point from: {s!r}")
    point = (float(m.group(1)), float(m.group(2)))
    _require_unit_interval(*point)
    return point


def rel_to_1000(x: float, y: float) -> list[int]:
    """Convert [0, 1] normalized coordinates to [0, 1000] integer coordinates."""
    _require_unit_interval(x, y)
    return [
        clamp_norm(int(round(x * MAX_NORM))),
        clamp_norm(int(round(y * MAX_NORM))),
    ]


def bbox_center_to_1000(x1: float, y1: float, x2: float, y2: float) -> list[int]:
    """Convert a [0, 1] bbox to its center point in [0, 1000]."""
    return rel_to_1000((x1 + x2) / 2, (y1 + y2) / 2)


# =============================================================================
# Action conversion — Web (LiteDesktopActionSet; web-single + web-multi)
# =============================================================================

def _web_click_coord(action: dict[str, Any]) -> list[int]:
    """Click target as [0, 1000].

    Prefer the precise ``point.related`` when present (web-multi carries it on
    click/input/hover); fall back to the ``element`` bbox center (web-single
    only ever carries the bbox).
    """
    point = action.get("point")
    if point and "related" in point:
        x, y = parse_related_point(point["related"])
        return rel_to_1000(x, y)
    x1, y1, x2, y2 = parse_related_bbox(action["element"]["related"])
    return bbox_center_to_1000(x1, y1, x2, y2)


def _convert_web_click(action: dict[str, Any]) -> list[dict[str, Any]]:
    return [LiteDesktopActionSet.click(coordinate=_web_click_coord(action))]


def _convert_web_hover(action: dict[str, Any]) -> list[dict[str, Any]]:
    return [LiteDesktopActionSet.mouse_move(coordinate=_web_click_coord(action))]


def _convert_web_input(action: dict[str, Any]) -> list[dict[str, Any]]:
    """Focus the field (if a target is given), then type the text."""
    calls: list[dict[str, Any]] = []
    if action.get("point", {}).get("related") or action.get("element", {}).get("related"):
        calls.append(LiteDesktopActionSet.click(coordinate=_web_click_coord(action)))
    calls.append(LiteDesktopActionSet.type(text=action["text"]))
    return calls


def _convert_web_enter(_action: dict[str, Any]) -> list[dict[str, Any]]:
    return [LiteDesktopActionSet.key(keys=["enter"])]


def _convert_web_scroll(action: dict[str, Any]) -> list[dict[str, Any]]:
    """Map source viewport deltas to direction(s) at five units per viewport.

    Source ``scroll.related.{down,right}`` are unbounded fractional strings
    (fraction of viewport scrolled); we keep the sign for direction and scale
    the magnitude into the action space's integer ``amount``.
    """
    scroll = action["scroll"]["related"]
    down = float(scroll["down"])
    right = float(scroll["right"])
    absolute = action["scroll"].get("absolute") or {}
    down_is_absolute = down == 0.0 and float(absolute.get("down") or 0) != 0.0
    right_is_absolute = right == 0.0 and float(absolute.get("right") or 0) != 0.0
    if down == 0.0:
        down = float(absolute.get("down") or 0)
    if right == 0.0:
        right = float(absolute.get("right") or 0)
    calls: list[dict[str, Any]] = []
    if down != 0.0:
        calls.append(LiteDesktopActionSet.scroll(
            direction="down" if down > 0 else "up",
            amount=1 if down_is_absolute else max(1, int(abs(down) * 5)),
        ))
    if right != 0.0:
        calls.append(LiteDesktopActionSet.scroll(
            direction="right" if right > 0 else "left",
            amount=1 if right_is_absolute else max(1, int(abs(right) * 5)),
        ))
    if not calls:
        raise SkipTrajectoryError("zero-distance scroll has no canonical action")
    return calls


def _convert_web_select(action: dict[str, Any]) -> list[dict[str, Any]]:
    raise SkipTrajectoryError(
        f"select({action.get('text')!r}) has no executable canonical action"
    )


def _convert_web_select_text(action: dict[str, Any]) -> list[dict[str, Any]]:
    dual = action["dual_point"]["related"]
    from_x, from_y = parse_related_point(dual["from"])
    to_x, to_y = parse_related_point(dual["to"])
    return [LiteDesktopActionSet.drag(
        start_coordinate=rel_to_1000(from_x, from_y),
        coordinate=rel_to_1000(to_x, to_y),
    )]


def _convert_web_copy(_action: dict[str, Any]) -> list[dict[str, Any]]:
    return [LiteDesktopActionSet.key(keys=["ctrl", "c"])]


def _convert_web_answer(action: dict[str, Any], *, is_terminal: bool) -> list[dict[str, Any]]:
    text = action.get("text", "")
    return [make_tool_call("response", {"text": text})]


_WEB_CONVERTERS = {
    "click": _convert_web_click,
    "hover": _convert_web_hover,
    "input": _convert_web_input,
    "enter": _convert_web_enter,
    "scroll": _convert_web_scroll,
    "select": _convert_web_select,
    "select_text": _convert_web_select_text,
    "copy": _convert_web_copy,
}


_WEB_NON_COMPUTER_TOOL_NAMES = {"response", "terminate"}


def _batch_multi_action_web_calls(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge adjacent desktop action-batch calls without crossing extras."""
    _validate_web_tool_call_contract(tool_calls)
    return merge_adjacent_lite_action_batches(tool_calls)


def _validate_web_tool_call_contract(tool_calls: list[dict[str, Any]]) -> None:
    for i, tc in enumerate(tool_calls):
        shape_error = validate_lite_tool_call(
            tc,
            f"tool_calls[{i}]",
            require_id=False,
        )
        if shape_error is not None:
            raise ValueError(shape_error)
        name = tool_call_name(tc)
        arguments = tool_call_arguments(tc)
        if name in _WEB_NON_COMPUTER_TOOL_NAMES:
            if not isinstance(arguments, dict):
                raise ValueError("canonical tool_call.arguments must be a dict")
            continue
        if name != "computer":
            raise ValueError(
                f"GUIAct web converters must emit computer action-batch tool_calls, got {name!r}"
            )
        if not isinstance(arguments, dict):
            raise ValueError("canonical tool_call.arguments must be a dict")
        actions = arguments.get("actions")
        if not isinstance(actions, list):
            raise ValueError("computer.arguments.actions must be a list")
        for ai, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError(f"computer.arguments.actions[{ai}] must be a dict")
            action_name = action.get("action")
            if not isinstance(action_name, str) or not action_name:
                raise ValueError(
                    f"computer.arguments.actions[{ai}].action must be a non-empty string"
                )
            if action_name in LITE_ACTION_BATCH_TOOL_NAMES:
                raise ValueError(
                    f"nested action-batch tool {action_name!r} must not appear inside "
                    "computer.arguments.actions"
                )


def convert_web_actions(
    actions_label: list[dict[str, Any]],
    *,
    is_terminal: bool,
) -> list[dict[str, Any]]:
    """Convert a list of GUIAct web actions to CUA-Lite desktop tool_calls.

    Desktop actions are represented as canonical ``computer`` action-batch calls
    with ordered actions in ``arguments.actions``. Consecutive action-batch calls
    are merged without crossing standalone extras.

    ``answer`` is always a ``response(text)`` tool here. Multi-step ``use``
    terminal-answer markers are structural episode ends and are converted to a
    content-only final ``Done.`` by ``guiact/use.py`` before reaching this helper.
    """
    tool_calls: list[dict[str, Any]] = []
    for i, action in enumerate(actions_label):
        name = action["name"].lower()
        is_last_action = i == len(actions_label) - 1

        if name == "answer":
            tool_calls.extend(_convert_web_answer(action, is_terminal=is_terminal and is_last_action))
            continue

        converter = _WEB_CONVERTERS.get(name)
        if converter is None:
            raise ValueError(f"Unknown web action: {action['name']!r}")
        try:
            converted = converter(action)
            # ``input`` synthesizes a focus click. Drop only that synthesized
            # click when the source already emitted the same click immediately
            # before it; repeated source click/Enter actions are intentional.
            if name == "input" and tool_calls and converted and tool_calls[-1] == converted[0]:
                converted = converted[1:]
            tool_calls.extend(converted)
        except (KeyError, ValueError) as e:
            raise SkipTrajectoryError(f"Failed to convert web action {name!r}: {e}") from e

    return _batch_multi_action_web_calls(tool_calls)


# =============================================================================
# Action conversion — Mobile (LiteMobileActionSet; smartphone)
# =============================================================================

def _convert_mobile_tap(action: dict[str, Any]) -> list[dict[str, Any]]:
    x, y = parse_related_point(action["point"]["related"])
    return [LiteMobileActionSet.tap(coordinate=rel_to_1000(x, y))]


def _convert_mobile_swipe(action: dict[str, Any]) -> list[dict[str, Any]]:
    dual = action["dual_point"]["related"]
    from_x, from_y = parse_related_point(dual["from"])
    to_x, to_y = parse_related_point(dual["to"])
    return [LiteMobileActionSet.swipe(
        start_coordinate=rel_to_1000(from_x, from_y),
        coordinate=rel_to_1000(to_x, to_y),
    )]


def _convert_mobile_input(action: dict[str, Any]) -> list[dict[str, Any]]:
    return [LiteMobileActionSet.type(text=action["text"])]


def _convert_mobile_enter(_action: dict[str, Any]) -> list[dict[str, Any]]:
    return [LiteMobileActionSet.system_button(button="Enter")]


_MOBILE_CONVERTERS = {
    "tap": _convert_mobile_tap,
    "swipe": _convert_mobile_swipe,
    "input": _convert_mobile_input,
    "enter": _convert_mobile_enter,
}


def convert_mobile_action(
    action_label: dict[str, Any],
    *,
    is_terminal: bool,
) -> list[dict[str, Any]]:
    """Convert a single GUIAct smartphone action to CUA-Lite mobile tool_calls.

    Smartphone ``actions_label`` is a single dict (not a list), so this
    processes one action per call.
    """
    name = action_label["name"].lower()

    if name == "answer":
        text = action_label.get("text", "")
        return [make_tool_call("response", {"text": text})]

    converter = _MOBILE_CONVERTERS.get(name)
    if converter is None:
        raise ValueError(f"Unknown mobile action: {action_label['name']!r}")
    try:
        return converter(action_label)
    except (KeyError, ValueError) as e:
        raise SkipTrajectoryError(f"Failed to convert mobile action {name!r}: {e}") from e
