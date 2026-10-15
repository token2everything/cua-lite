"""Debug-only coordinate annotation helpers for prompt-image overlays.

Used by :mod:`lite.agents.core.agent.logger` when ``debug_artifacts`` is on, to
draw crosshairs on the cached prompt image. This is a rendering concern, not a
protocol one: it reads no env metadata and decides no tool availability, it
only expands canonical action-batch calls and keeps the actions that carry a
drawable coordinate.

Run:
    uv run python - <<'PY'
    from lite.agents.core.agent.utils.annotations import coordinate_annotation_records
    print(coordinate_annotation_records([]))
    PY
"""

from __future__ import annotations

import inspect
import math
from typing import Any

from lite.core.tools.action_space import (
    LITE_COMPUTER_ACTION_BATCH_TOOL_NAME,
    LITE_MOBILE_ACTION_BATCH_TOOL_NAME,
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
    unpack_action_batch_call,
)
from lite.core.tools.action_space.geometry import COORDINATE_ARGUMENT_NAMES
from lite.core.tools.calls import (
    LiteToolCall,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
    validate_lite_tool_call,
)

# An action is drawable when its ``@tool`` signature takes a point-valued
# argument. Deriving this from the signatures rather than hand-listing names
# means a new coordinate action is annotated on arrival instead of silently
# dropping its overlay.
_ANNOTATED_TOOL_SETS = (LiteDesktopActionSet, LiteMobileActionSet, LitePointActionSet)


def _coordinate_bearing_action_names() -> frozenset[str]:
    """Not a facade: it DERIVES a fact no class accessor carries.

    ``get_action_names()`` is the right question here — a batched set emits one
    action-batch tool but carries every action, and it is the actions that have
    the signatures being inspected.
    """
    names: set[str] = set()
    for tools in _ANNOTATED_TOOL_SETS:
        for name in tools.get_action_names():
            parameters = inspect.signature(getattr(tools, name)).parameters
            if set(COORDINATE_ARGUMENT_NAMES) & set(parameters):
                names.add(name)
    return frozenset(names)


_COORDINATE_ANNOTATION_VALID_ACTION_NAMES = _coordinate_bearing_action_names()


def _coerce_real_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError
        number = float(raw)
    else:
        raise ValueError
    if not math.isfinite(number):
        raise ValueError
    return number


def annotation_coordinate(arguments: dict[str, Any]) -> Any | None:
    """The drawable point of one action's arguments, or ``None``.

    Single owner of which argument name carries the point and of what counts as
    drawable, so renderers project a resolved coordinate instead of re-reading
    :data:`COORDINATE_ARGUMENT_NAMES` themselves.
    """
    coord = None
    for key in COORDINATE_ARGUMENT_NAMES:
        coord = arguments.get(key)
        if coord is not None:
            break
    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
        return None
    try:
        _coerce_real_number(coord[0])
        _coerce_real_number(coord[1])
    except ValueError:
        return None
    return coord


def action_inspection_records(tool_calls: list[LiteToolCall]) -> list[dict[str, Any]]:
    """Expand calls into ``{name, arguments, result_call_id}`` debug records.

    Best-effort for trace/logger inspection: a non-canonical call is skipped and
    a structurally malformed action-batch payload contributes no records.
    ``result_call_id`` is the parent call id on every record, including one per
    action-batch child, because children carry no id of their own. These are
    debug/annotation trace records, not canonical Lite tool calls.
    """
    out: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if validate_lite_tool_call(tool_call, "tool_call", require_id=False) is not None:
            continue
        result_call_id = tool_call_id(tool_call)
        name = tool_call_name(tool_call)
        if name not in (LITE_COMPUTER_ACTION_BATCH_TOOL_NAME, LITE_MOBILE_ACTION_BATCH_TOOL_NAME):
            out.append({
                "name": name,
                "arguments": dict(tool_call_arguments(tool_call)),
                "result_call_id": result_call_id,
            })
            continue
        for child_call in unpack_action_batch_call(tool_call):
            out.append({
                "name": child_call["name"],
                "arguments": dict(child_call["arguments"]),
                "result_call_id": result_call_id,
            })
    return out


def coordinate_annotation_records(actions: list[LiteToolCall]) -> list[dict[str, Any]]:
    """Return expanded debug records for coordinate actions that can be drawn.

    The logger consumes the uniform records returned by
    ``action_inspection_records``: ``{"name", "arguments", "result_call_id"}``.
    This keeps action-batch children and standalone calls comparable without
    exposing logger overlays as canonical Lite tool calls. Renderers resolve the
    point of each record with :func:`annotation_coordinate`.
    """
    out: list[dict[str, Any]] = []
    for action in action_inspection_records(actions):
        name = action["name"]
        arguments = action["arguments"]
        if name not in _COORDINATE_ANNOTATION_VALID_ACTION_NAMES:
            continue
        if annotation_coordinate(arguments) is None:
            continue
        out.append(action)
    return out


__all__ = [
    "action_inspection_records",
    "annotation_coordinate",
    "coordinate_annotation_records",
]
