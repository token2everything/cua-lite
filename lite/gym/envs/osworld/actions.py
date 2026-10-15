"""OSWorld desktop actions shared by local Docker and remote Sandbox backends.

This module has no container, asset download, or task registration side effects.
``osworld.main`` continues to expose its documented public action imports.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from lite.core.errors import LiteContractError
from lite.core.tools.action_space.geometry import (
    MAX_NORM,
    norm_coord_out_of_range,
    strict_norm_to_pixel,
)
from lite.core.tools.extra_tools import make_report_infeasible_tool
from lite.core.tools.schemas import BaseTools
from lite.gym.utils.backend.model_inputs import coerce_model_duration, project_model_keys

logger = logging.getLogger(__name__)


class OsworldTools(BaseTools):
    """What osworld declares beyond the GUI surface: ``report_infeasible``.

    Shared by local OSWorld backends and the remote Sandbox environment.
    """

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "report_infeasible": make_report_infeasible_tool(),
    }


def to_pyautogui(name: str, args: dict[str, Any], w: int, h: int) -> str | None:
    """Project canonical desktop actions into VM PyAutoGUI source without I/O."""

    def _px(coord: list[int] | tuple[int, ...] | None) -> tuple[int, int]:
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            raise ValueError("arguments could not be interpreted")
        if norm_coord_out_of_range(coord):
            raise ValueError(
                f"coordinates must be normalized to [0, {MAX_NORM}], got {coord!r}. "
                f"Convert screenshot pixels using x * {MAX_NORM} / {w} and "
                f"y * {MAX_NORM} / {h} before calling the tool."
            )
        try:
            # MAX_NORM is valid, but the desktop's last pixel is (w - 1, h - 1).
            x, y = strict_norm_to_pixel(coord, w, h, clamp=True, round=False)
        except (LiteContractError, TypeError, ValueError, IndexError):
            raise ValueError("arguments could not be interpreted") from None
        return x, y

    def _keys_to_pyautogui() -> list[str]:
        return project_model_keys(
            args.get("keys", []),
            action_name=name,
            backend="pyautogui",
        )

    if name == "click":
        x, y = _px(args.get("coordinate"))
        button = args.get("button", "left")
        clicks = args.get("clicks", 1)
        if clicks >= 3:
            return f"pyautogui.tripleClick({x}, {y}, button={button!r})"
        if clicks == 2:
            return f"pyautogui.doubleClick({x}, {y}, button={button!r})"
        return f"pyautogui.click({x}, {y}, button={button!r})"
    if name == "mouse_move":
        x, y = _px(args.get("coordinate"))
        return f"pyautogui.moveTo({x}, {y})"
    if name == "mouse_down":
        return f"pyautogui.mouseDown(button={args.get('button', 'left')!r})"
    if name == "mouse_up":
        return f"pyautogui.mouseUp(button={args.get('button', 'left')!r})"
    if name == "drag":
        ex, ey = _px(args.get("coordinate"))
        button = args.get("button", "left")
        duration = coerce_model_duration(
            args.get("duration", 0.5),
            action_name="drag",
        )
        # start_coordinate optional → omit the pre-move so dragTo starts from the
        # current pyautogui cursor (matches waa / SandboxBaseEnv "drag from cursor").
        prefix = ""
        if args.get("start_coordinate") is not None:
            sx, sy = _px(args["start_coordinate"])
            prefix = f"pyautogui.moveTo({sx}, {sy}); "
        return f"{prefix}pyautogui.dragTo({ex}, {ey}, duration={duration}, button={button!r})"
    if name == "scroll":
        direction = args.get("direction", "down")
        amount = args.get("amount", 3)
        pos = ""
        if args.get("coordinate") is not None:
            x, y = _px(args["coordinate"])
            pos = f", x={x}, y={y}"
        if direction in ("left", "right"):
            clicks = amount if direction == "right" else -amount
            return f"pyautogui.hscroll({clicks}{pos})"
        clicks = amount if direction == "up" else -amount
        return f"pyautogui.scroll({clicks}{pos})"
    if name == "type":
        # repr() safely escapes quotes/newlines/backslashes; NO interval= kwarg. Split on `<`
        # host-side (emitting hotkey('shift', ',') per `<`): upstream's
        # _fix_pyautogui_less_than_bug corrupts any typewrite whose source text contains `<`
        # plus escapes — it unicode_escape-decodes the literal (turning `\n` into real
        # newlines) and re-embeds the parts unescaped, a SyntaxError that silently drops the
        # whole action. With no `<` left in any typewrite literal, that fixer stays inert.
        text = args.get("text", "")
        if not isinstance(text, str):
            raise TypeError("type.arguments.text must be a string")
        parts = text.split("<")
        calls = [f"pyautogui.typewrite({parts[0]!r})"] if parts[0] else []
        for part in parts[1:]:
            calls.append("pyautogui.hotkey('shift', ',')")
            if part:
                calls.append(f"pyautogui.typewrite({part!r})")
        return "; ".join(calls) or None
    if name == "key":
        pg_keys = _keys_to_pyautogui()
        return f"pyautogui.hotkey({', '.join(repr(k) for k in pg_keys)})"
    if name == "key_down":
        pg_keys = _keys_to_pyautogui()
        return "; ".join(f"pyautogui.keyDown({k!r})" for k in pg_keys)
    if name == "key_up":
        pg_keys = _keys_to_pyautogui()
        return "; ".join(f"pyautogui.keyUp({k!r})" for k in reversed(pg_keys))
    if name == "hold_key":
        pg_keys = _keys_to_pyautogui()
        duration = coerce_model_duration(
            args.get("duration", 1.0),
            action_name="hold_key",
        )
        downs = "; ".join(f"pyautogui.keyDown({k!r})" for k in pg_keys)
        ups = "; ".join(f"pyautogui.keyUp({k!r})" for k in reversed(pg_keys))
        return f"import time; {downs}; time.sleep({duration}); {ups}"
    if name == "wait":
        duration = coerce_model_duration(
            args.get("duration", 1.0),
            action_name="wait",
        )
        return f"import time; time.sleep({duration})"
    if name in ("screenshot", "cursor_position"):
        return None
    logger.warning("unknown action: %s(%s)", name, args)
    return None
