"""Shared helpers for the cua envs (cua.sandbox + cua.bench).

Tiny + pure: coordinate denormalization, action unpacking, terminal check —
so sandbox/env.py (→ Cua Sandbox SDK) and bench/translate.py (→ cua-bench Action
dataclasses) don't each re-implement them. Screenshots are passed as raw PNG
``bytes`` (``sb.screenshot()``), unencoded — no base64 at this layer.
"""

from __future__ import annotations

from typing import Any

from lite.core.tools.calls import EnvAction
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.gym.utils.backend.coordinate import norm_to_pixel

#: EnvDepsMissingError hint, shared by every cua env (install script + doc pointer).
INSTALL = "uv run --no-sync bash lite/gym/envs/cua/scripts/install.sh"
SEE = "/docs/examples/cua.md"


def to_px(coord: Any, wh: tuple[int, int]) -> tuple[int, int]:
    """Lite normalized ``[0, 1000]`` coordinate → absolute pixels for ``wh=(W, H)``."""
    w, h = wh
    px, py = norm_to_pixel(coord, w, h, clamp=False, on_malformed="raise")
    return int(px), int(py)


def unpack(a: EnvAction) -> tuple[str, dict[str, Any]]:
    """``(name, arguments)`` from an env-internal bare action."""
    return a["name"], a["arguments"]


def is_terminal(name: str) -> bool:
    """True iff the action ends the episode (agent ``terminate``/``response``)."""
    return name in LiteFinishToolSet.get_tool_names()
