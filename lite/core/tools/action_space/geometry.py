"""Canonical action-space coordinate helpers.

Run: not directly. Imported by anything that converts between the action
space's [0, MAX_NORM]-normalized ``coordinate`` and a pixel surface, or that
has to decide whether a normalized coordinate is in range.

This module owns the normalized coordinate *range* as well as the projection:
``MAX_NORM`` is the single upper bound, :func:`clamp_norm` the single clamp,
and :func:`norm_coord_out_of_range` the single range predicate. Pixel projection
is strict: malformed pairs raise, and callers choose clamp policy explicitly
with :func:`strict_norm_to_pixel`.
"""

from __future__ import annotations

import builtins
import math
from collections.abc import Mapping
from typing import Any

from lite.core.errors import LiteContractError

#: Upper bound of the CUA-lite normalized coordinate range, ``[0, MAX_NORM]``.
#: The range is *inclusive* on both ends: ``MAX_NORM`` is a legal coordinate
#: (it maps to the far edge of the surface, which :func:`strict_norm_to_pixel`
#: can clamp to the last addressable pixel).
MAX_NORM = 1000

#: Every normalized-coordinate argument name in the canonical action space.
COORDINATE_ARGUMENT_NAMES = ("coordinate", "start_coordinate")


def clamp_norm(value: int) -> int:
    """Clamp one normalized coordinate component into ``[0, MAX_NORM]``.

    Scalar, because every clamp site in the tree is scalar: sources compute
    one axis at a time (``clamp_norm(int(round(x * MAX_NORM)))``) and pair the
    results themselves.
    """
    return max(0, min(MAX_NORM, value))


def norm_coord_out_of_range(coord: Any) -> bool:
    """Return True if ``coord`` holds a number outside ``[0, MAX_NORM]``.

    The shape the dataset out-of-bounds predicates need: takes the raw
    argument value straight out of a tool call, so a missing / ``None`` /
    non-sequence coordinate is simply "not out of range" rather than an error.
    Non-numeric elements are ignored (schema validation's job, not the range
    check's). Both ``list`` and ``tuple`` are scanned — the per-source copies
    only tested ``list``, so a tuple coordinate went unchecked.
    """
    if not isinstance(coord, (list, tuple)):
        return False
    return any(isinstance(v, (int, float)) and (v < 0 or v > MAX_NORM) for v in coord)


def action_coordinate_arguments_out_of_range(args: Mapping[str, Any]) -> bool:
    """Return True if any coordinate argument of one action leaves ``[0, MAX_NORM]``.

    :data:`COORDINATE_ARGUMENT_NAMES` is the complete set of normalized-coordinate
    argument names in the canonical action space (``drag``/``swipe`` are the
    only actions carrying a second point, and they spell it
    ``start_coordinate``). Callers pass the argument dict of a *single* action:
    one entry from an action-batch tool's ``actions`` list, or an unbatched
    grounding action's arguments.
    """
    return any(norm_coord_out_of_range(args.get(key)) for key in COORDINATE_ARGUMENT_NAMES)


def pixel_to_norm(pixel_x: int, pixel_y: int, width: int, height: int) -> list[int]:
    """Convert pixel coordinates to CUA-lite normalized [0, MAX_NORM]."""
    return [
        int(round(pixel_x / width * MAX_NORM)),
        int(round(pixel_y / height * MAX_NORM)),
    ]


def _project_norm_values(
    norm_x: Any,
    norm_y: Any,
    width: int,
    height: int,
    *,
    round: bool,
) -> tuple[int, int]:
    try:
        raw_x = float(norm_x)
        raw_y = float(norm_y)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LiteContractError("coordinate values must be finite numbers") from exc
    if not math.isfinite(raw_x) or not math.isfinite(raw_y):
        raise LiteContractError("coordinate values must be finite numbers")

    x = raw_x / MAX_NORM * width
    y = raw_y / MAX_NORM * height
    if round:
        return (builtins.round(x), builtins.round(y))
    return (int(x), int(y))


def strict_norm_to_pixel(
    coord: Any,
    width: int,
    height: int,
    *,
    clamp: bool,
    round: bool = True,
) -> tuple[int, int]:
    """Strictly project a normalized ``(x, y)`` pair onto a pixel surface.

    ``coord`` must be exactly a two-element pair; malformed coordinates
    (wrong length, wrong type, non-finite components) raise
    :class:`LiteContractError`. ``clamp`` is keyword-only and required so each
    call site states whether ``MAX_NORM`` may land at the far edge
    ``(width, height)`` or must be capped to the last addressable pixel.
    """
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        raise LiteContractError(f"malformed normalized coordinate: {coord!r}")

    x, y = _project_norm_values(
        coord[0],
        coord[1],
        width,
        height,
        round=round,
    )
    if clamp:
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
    return (x, y)


__all__ = [
    "COORDINATE_ARGUMENT_NAMES",
    "MAX_NORM",
    "action_coordinate_arguments_out_of_range",
    "clamp_norm",
    "norm_coord_out_of_range",
    "pixel_to_norm",
    "strict_norm_to_pixel",
]
