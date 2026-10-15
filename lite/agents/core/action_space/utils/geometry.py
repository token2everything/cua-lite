"""Adapter-local action-space helpers.

Provider-free geometry and batch helpers are imported from
``lite.core.tools.action_space`` directly by callers.
"""

from __future__ import annotations

import math
from typing import Any

from lite.agents.core.action_space.errors import ModelToolCallParseError

# The ONE pixels <-> wheel-click conversion used by every wrapper family whose
# native scroll is spelled in screen pixels while cua-lite's canonical
# ``scroll`` counts wheel clicks (qwen3_vl, qwen2_5_vl, qwen3_5 via qwen3_vl,
# fara, evocua, and the gpt native computer tool). 100 pixels ~ 1 wheel click
# (on a 1000x1000 screen, 500px ~ 5 clicks ~ half a screen).
#
# Before this constant had an owner each family re-declared it with the SAME
# value and a comment citing a sibling family rather than any model spec, so
# "is 100 a per-family tuning knob?" had no answer. It is not: it is one
# convention. Tune it here, for everyone, or not at all.
PIXELS_PER_CLICK = 100

# Below this magnitude a pixel-valued scroll is read as a RAW notch count
# instead of pixels: the model sometimes confuses the two units and emits e.g.
# ``pixels=-3`` meaning "3 notches down". Dividing that by
# :data:`PIXELS_PER_CLICK` would round to a single click and silently discard
# the magnitude.
RAW_NOTCH_THRESHOLD = 10


def _finite_float(val: Any) -> float:
    value = float(val)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def _parse_coord(val: Any) -> list[int]:
    if isinstance(val, str):
        val = val.strip().strip("[]()").split(",")
    try:
        return [int(_finite_float(str(c).strip())) for c in val]
    except (ValueError, TypeError, OverflowError) as exc:
        raise ModelToolCallParseError(
            "coordinate must contain only finite numeric values"
        ) from exc


def _check_coord_dimensions(
    parsed: list[int],
    *,
    dimensions: int | None,
    name: str,
) -> list[int]:
    if dimensions is not None and len(parsed) != dimensions:
        raise ModelToolCallParseError(
            f"{name} must contain exactly {dimensions} numeric values; "
            f"got {len(parsed)}"
        )
    return parsed


def required_coord(
    val: Any,
    *,
    dimensions: int | None = None,
    name: str = "coordinate",
) -> list[int]:
    """Parse a required coordinate or raise a model-output parse error.

    ``dimensions`` lets callers distinguish two-point coordinates from bboxes.
    """
    if val is None:
        raise ModelToolCallParseError(f"{name} is required")
    return _check_coord_dimensions(
        _parse_coord(val),
        dimensions=dimensions,
        name=name,
    )


def optional_coord(val: Any, *, dimensions: int | None = None) -> list[int] | None:
    """Parse an optional coordinate; missing or malformed values return ``None``."""
    if val is None:
        return None
    try:
        return required_coord(val, dimensions=dimensions)
    except ModelToolCallParseError:
        return None


def required_scroll_pixels(args: dict[str, Any], action: str) -> int:
    """``args["pixels"]`` as an int, or raise instead of inventing a default."""
    if "pixels" not in args:
        raise ModelToolCallParseError(
            f"computer_use(action={action!r}) requires the 'pixels' argument "
            "(negative = down/left, positive = up/right)"
        )
    raw = args["pixels"]
    try:
        return int(_finite_float(raw))
    except (TypeError, ValueError, OverflowError):
        raise ModelToolCallParseError(
            f"computer_use(action={action!r}) needs a numeric 'pixels'; "
            "got a non-numeric or non-finite value"
        ) from None


def compact_number(val: Any) -> Any:
    """Format a numeric value without the trailing ``.0`` when integer-valued."""
    if val is None:
        return None
    try:
        f = _finite_float(val)
        i = int(f)
    except (TypeError, ValueError, OverflowError):
        return val
    return i if f == i else f


__all__ = [
    "PIXELS_PER_CLICK",
    "RAW_NOTCH_THRESHOLD",
    "compact_number",
    "optional_coord",
    "required_coord",
    "required_scroll_pixels",
]
