"""Backend/env normalized-coordinate → pixel projection.

Run: not directly. Imported by env adapters that map the action space's
[0, 1000]-normalized ``coordinate`` onto a concrete surface (screen,
viewport, image).

Before this helper the backend projection existed as ~11 hand-rolled copies with
*divergent* semantics on BOTH axes — rounding (truncate vs round vs float)
and malformed-coordinate policy (center-click vs None vs raise). The
divergence is agent-visible: a model emitting ``1000`` landed at pixel ``w``
(off-canvas) in truncating copies vs ``w-1`` in the clamping one. This
helper makes the env backend projection contract explicit:

- ``clamp``       — clamp into ``[0, w-1] × [0, h-1]`` (the canonical
                    default; ``False`` reproduces legacy no-clamp sites).
- ``as_float``    — return sub-pixel floats (Playwright accepts them;
                    browsergym's contract). Ints (``round``) otherwise.
- ``on_malformed``— ``"center"`` (click dead-center), ``"none"`` (return
                    ``None``; online_mind2web's "page" target), or
                    ``"raise"`` (ValueError).

NOTE: env execution paths should use ``on_malformed="raise"`` so bad model
coordinates become paired env feedback instead of a center click/tap. The
Android/mobile and grounding envs still call with
``clamp=False`` where their upstream/native coordinate contract expects no
clamping. Two env adapters (osworld/main.py, webgym/main.py) intentionally use
``lite.core.tools.action_space.geometry.strict_norm_to_pixel`` for
model-coordinate adaptation, not env geometry; not part of this ledger.
"""

from __future__ import annotations

import math
from typing import Any, Literal


def parse_norm_pair_values(coord: Any) -> tuple[float, float]:
    """Parse a shaped normalized coordinate pair into finite float values.

    Callers own the shape policy. This helper owns only the value contract:
    conversion with ``float()`` and rejection of non-finite values.
    """
    try:
        raw_x = float(coord[0])
        raw_y = float(coord[1])
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError("coordinate values must be finite numbers") from e
    if not math.isfinite(raw_x) or not math.isfinite(raw_y):
        raise ValueError("coordinate values must be finite numbers")
    return raw_x, raw_y


def norm_to_pixel(
    coord: Any,
    width: int,
    height: int,
    *,
    as_float: bool = False,
    clamp: bool = True,
    on_malformed: Literal["center", "none", "raise"] = "center",
) -> tuple[int, int] | tuple[float, float] | None:
    """Map a [0, 1000]-normalized ``(x, y)`` onto a ``width × height`` surface.

    Malformed = not an indexable pair (``None``, wrong length, non-sequence).
    Non-numeric *elements* raise (a contract violation, never silently
    centered).
    """
    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
        if on_malformed == "center":
            return (width / 2.0, height / 2.0) if as_float else (width // 2, height // 2)
        if on_malformed == "none":
            return None
        raise ValueError(f"malformed normalized coordinate: {coord!r}")

    raw_x, raw_y = parse_norm_pair_values(coord)

    fx = raw_x / 1000.0 * width
    fy = raw_y / 1000.0 * height
    if as_float:
        if clamp:
            fx = max(0.0, min(float(width - 1), fx))
            fy = max(0.0, min(float(height - 1), fy))
        return (fx, fy)
    x, y = int(round(fx)), int(round(fy))
    if clamp:
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
    return (x, y)
