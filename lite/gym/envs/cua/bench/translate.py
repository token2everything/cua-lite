"""Pure translation helpers for the cua.bench wrapper — no cua-bench import at
module load, so they're unit-testable standalone (pass the cua_bench module /
a fake namespace into ``to_cb_action``).
"""

from __future__ import annotations

from typing import Any

from lite.core.tools.calls import EnvAction
from lite.gym.envs.cua.utils import is_terminal, to_px, unpack
from lite.gym.utils.backend.model_inputs import coerce_model_duration, project_model_keys


def normalize_reward(result: Any) -> float | None:
    """cua-bench ``evaluate()`` returns ``Any`` → a float reward, mirroring
    ``cua_bench/runners.py:139-145``: ``int/float`` → ``float`` (``bool`` too,
    since ``bool ⊂ int`` → 1.0/0.0), ``list`` → ``float(result[0])``,
    ``dict`` → ``float(result["reward"])``; anything else → ``None``.
    (Success is separately ``reward >= 0.5``, done by the caller.)"""
    if result is None:
        return None
    if isinstance(result, (int, float)):     # bool is an int subclass → 1.0/0.0
        return float(result)
    if isinstance(result, list) and result:
        return float(result[0])
    if isinstance(result, dict) and "reward" in result:
        return float(result["reward"])
    return None


def to_cb_action(
    a: EnvAction,
    wh: tuple[int, int],
    A: Any,
    key_backend: str = "pynput",
    cursor_px: tuple[int, int] | None = None,
) -> tuple[Any, bool]:
    """Translate one LiteDesktopActionSpace action → ``(cua_bench.Action, done)``.

    ``A`` is the cua_bench module (or any namespace exposing the Action
    dataclasses: ``ClickAction``/``RightClickAction``/``DoubleClickAction``/
    ``MiddleClickAction``/``MoveToAction``/``DragAction``/``ScrollAction``/
    ``TypeAction``/``KeyAction``/``HotkeyAction``/``WaitAction``/``DoneAction``).
    cua-bench Actions take ABSOLUTE PIXELS, so we map Lite ``[0,1000]`` → px
    via ``wh``. ``done=True`` iff the action ends the episode.

    ``key_backend`` projects canonical key tokens to the consuming backend's
    spelling (like every other lite env — browsergym/osworld/webgym): cua-bench
    ``local``/native runs on the computer-server → ``"pynput"`` (default, the
    shipped datasets); ``webtop`` runs on Playwright → ``"playwright"``. Without
    this, tokens like ``pageup``/``meta`` reach pynput as-is and fail
    (``page_up``/``cmd`` expected) — the same convention-miss class as scroll.

    ``cursor_px`` mirrors ``SandboxBaseEnv`` drag semantics: Lite desktop drag's
    ``start_coordinate`` is optional, meaning "drag from the current cursor".
    Missing ``coordinate`` is malformed for cua-bench's single DragAction and is
    degraded to a harmless no-op wait instead of surfacing as an env error."""
    name, args = unpack(a)
    if is_terminal(name):
        return A.DoneAction(), True

    def px(coord: list[int]) -> tuple[int, int]:
        return to_px(coord, wh)

    if name == "click":
        x, y = px(args["coordinate"])
        if args.get("clicks") == 2:
            return A.DoubleClickAction(x=x, y=y), False
        if args.get("button") == "right":
            return A.RightClickAction(x=x, y=y), False
        if args.get("button") == "middle":
            return A.MiddleClickAction(x=x, y=y), False
        return A.ClickAction(x=x, y=y), False          # ClickAction has NO button field
    if name == "type":
        return A.TypeAction(text=args["text"]), False
    if name == "key":
        proj = project_model_keys(
            args["keys"],
            action_name=name,
            backend=key_backend,
        )
        if len(proj) > 1:
            return A.HotkeyAction(keys=proj), False
        return A.KeyAction(key=proj[0]), False
    if name == "mouse_move":
        x, y = px(args["coordinate"])
        return A.MoveToAction(x=x, y=y), False          # note: MoveToAction, not MoveAction
    if name == "drag":
        end = args.get("coordinate")
        if not end:
            raise ValueError("drag.coordinate is required")
        if args.get("start_coordinate"):
            sx, sy = px(args["start_coordinate"])
        else:
            sx, sy = cursor_px or (wh[0] // 2, wh[1] // 2)
        ex, ey = px(end)
        return A.DragAction(from_x=sx, from_y=sy, to_x=ex, to_y=ey), False   # DragAction: no button
    if name == "scroll":
        direction = args.get("direction", "down")
        # cua-bench's ScrollAction is vertical-only (Literal["up","down"]); a left/right
        # direction would silently scroll the WRONG axis (native→up, webtop→down), so
        # drop it to a no-op rather than corrupt state (cua.sandbox does honor hscroll).
        if direction not in ("up", "down"):
            raise ValueError(
                f"scroll.direction {direction!r} is not supported by cua.bench; "
                "expected 'up' or 'down'"
            )
        # Lite `amount` is in scroll CLICKS (~100px/click; see gpt action_space
        # `_PIXELS_PER_CLICK`). cua-bench's ScrollAction is px-scale — native does
        # `amount // 100` → clicks (remote.py), webtop uses `amount` as px — so scale
        # clicks→cua units: a 3-click scroll → 3 clicks native / ~300px webtop,
        # matching cua-bench's own native default (`amount=300`).
        return A.ScrollAction(direction=direction,
                              amount=int(args.get("amount", 3)) * 100), False
    if name == "wait":
        duration = coerce_model_duration(
            args.get("duration", 1),
            action_name="wait",
        )
        return A.WaitAction(seconds=duration), False
    raise ValueError(f"unsupported action: {name}")
