"""Shared helpers for the Aguvis (``xlangai/aguvis-stage1`` + ``stage2``) adapters.

Two responsibilities:

1. **PyAutoGUI → cua-lite tool-call parser** (``pyautogui_to_tool_calls``) — an
   ``ast.parse``-based, platform-aware translator (desktop/browser →
   ``LiteDesktopActionSet``, mobile → ``LiteMobileActionSet``). Ported from
   the original Aguvis adapter; the action vocabulary is unchanged. The one
   place a source value is not carried through verbatim is ``scroll``'s
   magnitude, because the source does not have one — see
   :data:`DIRECTION_ONLY_SWIPE_TRAVEL`. ``scroll``'s *direction* is carried
   through verbatim on all three of its source encodings, but for the bare
   ``scroll()`` form it is read out of the step's own ``Action:`` prose rather
   than out of the code — see :func:`scroll_direction_from_description`.
2. **Canonical staging wiring** — the shared
   :class:`~lite.data.preproc.common.SourceStaging` default, shared by
   ``grounding-action.py`` and ``use.py``.

Both Stage 1 (grounding) and Stage 2 (trajectory) land in one cua-lite dataset
``Aguvis`` with the upstream sub-dataset carried as the cohort *variant*.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from typing import Any

from PIL import Image as PILImage

from lite.core.tools.action_space import (
    MAX_NORM,
    LiteDesktopActionSet,
    LiteMobileActionSet,
    clamp_norm,
    normalize_keys,
)
from lite.core.tools.action_space.keys import is_canonical_key_token
from lite.core.tools.calls import make_tool_call
from lite.data.preproc.common import SourceStaging

DATASET_NAME = "Aguvis"
SOURCE_STAGE1 = "xlangai/aguvis-stage1"
SOURCE_STAGE2 = "xlangai/aguvis-stage2"
_COORD_EPS = 1e-6


# --- exceptions ------------------------------------------------------------
class SkipEpisodeError(Exception):
    """A whole trajectory episode dropped, with its reason tag.

    ``reason`` is a required positional argument, not an optional label: the
    catch site counts drops by it, so a drop nobody can count is
    unconstructible. Adding a ``raise`` therefore also adds its bucket to the
    run's ``Skip reasons:`` line, with no second place to remember to edit.
    Same shape as ``scalecua``'s ``SkipTrajectoryError``.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class AguvisParseError(RuntimeError):
    """Raised when a PyAutoGUI code string cannot be parsed."""


def _canonical_desktop_key(value: Any, code: str) -> str:
    if not isinstance(value, str):
        raise AguvisParseError(f"Key must be a string, got {value!r}.\ncode=\n{code}")
    try:
        normalized = normalize_keys([value])
    except (TypeError, ValueError) as exc:
        raise AguvisParseError(f"Unsupported key token {value!r}.\ncode=\n{code}") from exc
    if len(normalized) != 1:
        raise AguvisParseError(f"Expected one key token, got {value!r}.\ncode=\n{code}")
    key = normalized[0]
    if not is_canonical_key_token(key):
        raise AguvisParseError(f"Unsupported key token {value!r}.\ncode=\n{code}")
    return key


# --- canonical staging wiring ---------------------------------------------
_STAGING = SourceStaging(DATASET_NAME)
out_dir_for = _STAGING.out_dir_for
make_image_store = _STAGING.make_image_store
make_splitter = _STAGING.make_splitter
stage_entry = _STAGING.stage_entry


@lru_cache(maxsize=65_536)
def image_resolution(image_path: str) -> list[int] | None:
    """``[width, height]`` of the screenshot on disk, or ``None`` if unreadable.

    Aguvis records carry no pixel surface -- its coordinates arrive already
    normalized to ``[0, 1]`` -- so the image file is the only record of it.

    Returns ``None`` rather than raising: this runs BEFORE ``stage_entry``, and a
    corrupt or missing image is already the staging loop's skip to make.
    """
    try:
        with PILImage.open(image_path) as im:
            return [im.width, im.height]
    except (OSError, ValueError, SyntaxError):
        return None


# --- AST helpers (same pattern as opencua/trajectory.py) -------------------
def _dotted_name(expr: ast.AST) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _dotted_name(expr.value)
        return f"{base}.{expr.attr}" if base else expr.attr
    return ""


def _literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError) as e:
        raise AguvisParseError(f"Failed literal_eval on node={ast.dump(node)}") from e


def _get_kw(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _extract_xy(call: ast.Call) -> tuple[float, float]:
    x_node = _get_kw(call, "x")
    y_node = _get_kw(call, "y")
    if x_node is None and len(call.args) >= 1:
        x_node = call.args[0]
    if y_node is None and len(call.args) >= 2:
        y_node = call.args[1]
    if x_node is None or y_node is None:
        raise AguvisParseError(
            f"Expected x and y arguments, got args={len(call.args)} "
            f"keywords={[kw.arg for kw in call.keywords]}"
        )
    return float(_literal_eval(x_node)), float(_literal_eval(y_node))


# --- coordinate normalization ----------------------------------------------
def norm01_to_1000(x: float, y: float) -> list[int]:
    """Normalized [0, 1] → int [0, 1000].

    Truly out-of-range source coordinates are rejected before endpoint clamping
    handles floating-point noise around 0/1.
    """
    if (
        x < -_COORD_EPS
        or x > 1 + _COORD_EPS
        or y < -_COORD_EPS
        or y > 1 + _COORD_EPS
    ):
        raise AguvisParseError(f"coordinates outside [0, 1]: x={x}, y={y}")
    xi = clamp_norm(int(round(x * MAX_NORM)))
    yi = clamp_norm(int(round(y * MAX_NORM)))
    return [xi, yi]


def _reject_on_mobile(fname: str, is_mobile: bool, code: str) -> None:
    """Drop a step whose action is desktop-only (pointer-based) on mobile.

    A touchscreen has no secondary button, no triple-click and no hover cursor,
    so these have no mobile rendering. Raising drops the step (the caller skips
    it) rather than silently emitting a ``computer`` wrapper, which is invalid
    for ``platform=mobile``. Same policy as the ``hotkey`` branch below.
    """
    if is_mobile:
        raise AguvisParseError(f"{fname} has no mobile equivalent.\ncode=\n{code}")


#: The only ``|page|`` the aguvis corpus this adapter reads ever spells.
#:
#: ``page`` is genuinely a signed *fraction of one viewport* upstream -- provable
#: from the GUIAct records aguvis re-encodes, where the same scroll is written
#: both ways (``{"absolute": {"down": 1000}, "related": {"down": "0.95"}}`` at
#: ``image_size.height = 1050``: 1000/1050 = 0.95), and aguvis emits that 0.95
#: verbatim. But the subsets THIS adapter reads carry no such measurement:
#: ``page`` takes exactly TWO values over 9,833 calls -- 7,929 x -0.1 and
#: 1,904 x +0.1, every one of them in ``android_control.json`` -- because
#: upstream AndroidControl (arXiv 2406.03679) defines ``scroll`` as a bare
#: direction with no distance field at all, so aguvis' converter hardcoded a
#: constant and used only the SIGN and the function name (``scroll`` /
#: ``hscroll``) to spell the four directions. Magnitude is not in the record:
#: the source expresses "scroll further" by repeating the whole step (76.8% of
#: those scroll steps sit in a run of >= 2 identical steps).
_DIRECTION_ONLY_PAGE = 0.1

#: Travel of a direction-only mobile scroll gesture, in [0, 1000] units.
#:
#: Because ``page`` carries no magnitude (see :data:`_DIRECTION_ONLY_PAGE`), the
#: gesture length is this adapter's choice and has to be stated rather than
#: computed. 400 units -- a swipe from 300 to 700 through the screen centre --
#: for three reasons, in order:
#:
#: 1. it is what the sibling direction-only cohort already publishes:
#:    ``ui_genie_agent``'s ``DIRECTION_SWIPES`` renders its 1,010 magnitude-less
#:    swipes as exactly ``[500, 700] -> [500, 300]``, so the two cohorts teach
#:    ONE "scroll" gesture instead of two;
#: 2. it is inside the interquartile band of 39,186 real sibling gestures whose
#:    sources DO record endpoints (guiodyssey / guiact-smartphone / cagui /
#:    ui-genie-amex / scalecua-android: pooled vertical ``|dy|`` p25 = 352,
#:    median = 482, p75 = 614), i.e. a plausible fling rather than a twitch;
#: 3. it is the value the previous expression already declared as a full scroll
#:    -- ``min(400, amount * 40)`` -- and never reached, because ``|page| = 0.1``
#:    floored ``amount`` to 1 and produced 40 units, a 4%-of-screen no-op, for
#:    all 9,833 calls.
DIRECTION_ONLY_SWIPE_TRAVEL = 400

#: scroll function (leaf name) -> the direction its POSITIVE sign means, then
#: the direction its negative sign means.
_SCROLL_AXES: dict[str, tuple[str, str]] = {
    "scroll": ("up", "down"),
    "hscroll": ("right", "left"),
}

#: The direction words a stage-2 ``Action:`` description spells, in the spellings
#: it actually uses. Matched with word boundaries, so ``download`` (33 of the 392
#: descriptions contain it) cannot masquerade as ``down``.
_SCROLL_DIRECTION_RE = re.compile(r"\b(down|downwards?|up|upwards?)\b", re.I)


def scroll_direction_from_description(description: str) -> str | None:
    """The scroll direction a step's own ``Action:`` description states, or ``None``.

    ``guide.json`` spells 392 of its scrolls as the bare, argument-less
    ``pyautogui.scroll()`` — see :func:`_scroll_tool_call` — so for those steps
    the CODE carries neither a direction nor a magnitude, and the only record of
    the direction is the prose in the same step's first gpt turn, which this
    adapter already parses (:func:`extract_thought_and_desc`) and publishes as an
    ``action_description`` content part. Reading it here is therefore not
    inference from a new source: it is reading the field the row already ships.

    Returns a direction only when the text names exactly ONE, so a sentence
    naming both is refused rather than resolved by position. Over the whole
    corpus of bare calls that partitions cleanly — 386 x ``{"down"}``, 6 x
    ``{}``, and zero mentioning ``up`` or both — and every one of the 386 also
    contains the word ``scroll``, so requiring co-occurrence would change no
    row and is not tested for.
    """
    found = {"down" if m.group(1).lower().startswith("down") else "up"
             for m in _SCROLL_DIRECTION_RE.finditer(description)}
    return found.pop() if len(found) == 1 else None


def _mobile_scroll_swipe(direction: str) -> dict[str, Any]:
    """Mobile rendering of a direction-only scroll: one center-anchored swipe.

    The mobile action space has no ``scroll`` verb, only ``swipe``, and the
    source gives a direction with no anchor and no distance — so all four
    directions render the same documented gesture,
    :data:`DIRECTION_ONLY_SWIPE_TRAVEL` units long through the screen centre.
    *direction* is the direction the CONTENT moves, so the finger travels the
    opposite way (scrolling down = swiping up).

    Shared by the ``scroll`` and ``hscroll`` branches so the two axes cannot
    drift apart.
    """
    center = MAX_NORM // 2
    reach = DIRECTION_ONLY_SWIPE_TRAVEL // 2
    near, far = center - reach, center + reach
    if direction == "down":
        return LiteMobileActionSet.swipe(start_coordinate=[center, far], coordinate=[center, near])
    if direction == "up":
        return LiteMobileActionSet.swipe(start_coordinate=[center, near], coordinate=[center, far])
    if direction == "right":
        return LiteMobileActionSet.swipe(start_coordinate=[far, center], coordinate=[near, center])
    if direction == "left":
        return LiteMobileActionSet.swipe(start_coordinate=[near, center], coordinate=[far, center])
    raise AguvisParseError(f"unknown scroll direction {direction!r}")


def _scroll_tool_call(
    call: ast.Call,
    fname: str,
    *,
    is_mobile: bool,
    code: str,
    scroll_direction: str | None,
) -> dict[str, Any]:
    """One ``pyautogui.scroll`` / ``pyautogui.hscroll`` call -> one tool call.

    THREE source encodings carrying different amounts of information:

    * ``page=+-0.1`` — a direction and nothing else
      (:data:`_DIRECTION_ONLY_PAGE`). Desktop gets one wheel click, which IS a
      real scroll there; mobile gets one :func:`_mobile_scroll_swipe`. Any other
      ``|page|`` would be a real viewport fraction this corpus has never
      contained, and neither rendering can express it, so it is refused rather
      than approximated — the convention every unrenderable action in this
      module already follows.
    * a positional wheel-click count — upstream's own integer (``omniact``:
      3..100 across 168 calls, the only file this adapter reads that uses the
      form, and it is ``platform="desktop"``). A touchscreen has no wheel, so a
      click count has no mobile rendering and is refused like ``moveTo`` is.
    * **no argument at all** — the whole call is ``pyautogui.scroll()``. 392
      records of ``guide.json`` and nowhere else in the 14 raw files this adapter
      reads; the code names neither a direction nor a magnitude. The direction
      comes in already resolved as *scroll_direction*, read by
      :func:`scroll_direction_from_description` from the same step's ``Action:``
      prose — the field the row already publishes. It is refused when the prose
      names no direction (6 of the 392) and when it names one that is not on this
      call's axis, so a hypothetical bare ``hscroll()`` cannot borrow a vertical
      word. Magnitude is the same :data:`DIRECTION_ONLY_SWIPE_TRAVEL` the
      ``page=+-0.1`` cohort gets: both are direction-only, and one gesture is the
      point.
    """
    positive, negative = _SCROLL_AXES[fname.rpartition(".")[2]]
    page_node = _get_kw(call, "page")
    if page_node is not None:
        page = float(_literal_eval(page_node))
        if abs(page) != _DIRECTION_ONLY_PAGE:
            raise AguvisParseError(
                f"{fname}(page={page}) carries a viewport fraction; this corpus "
                f"only ever spells the direction-only +-{_DIRECTION_ONLY_PAGE}, "
                f"and neither rendering can express a real magnitude.\ncode=\n{code}"
            )
        direction = positive if page > 0 else negative
        return (
            _mobile_scroll_swipe(direction) if is_mobile
            else LiteDesktopActionSet.scroll(direction=direction, amount=1)
        )
    if call.args:
        _reject_on_mobile(fname, is_mobile, code)
        clicks = round(float(_literal_eval(call.args[0])))
        return LiteDesktopActionSet.scroll(
            direction=positive if clicks > 0 else negative,
            amount=max(1, abs(clicks)),
        )
    if scroll_direction in (positive, negative):
        return (
            _mobile_scroll_swipe(scroll_direction) if is_mobile
            else LiteDesktopActionSet.scroll(direction=scroll_direction, amount=1)
        )
    raise AguvisParseError(
        f"{fname}() carries no argument, and the step's own description states "
        f"no {positive}/{negative} direction (resolved={scroll_direction!r}), so "
        f"the gesture is unknown.\ncode=\n{code}"
    )


def _map_key_to_system_button(key: str) -> str | None:
    return {
        "home": "Home", "back": "Back", "enter": "Enter", "return": "Enter",
        "menu": "Menu", "recent": "Recent", "app_switch": "Recent",
    }.get(key.lower().strip())


# --- PyAutoGUI → cua-lite tool calls ---------------------------------------
def pyautogui_to_tool_calls(
    code: str,
    platform: str,
    *,
    scroll_direction: str | None = None,
    cursor_coordinate: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Convert a PyAutoGUI/mobile/computer code string into cua-lite tool calls.

    *platform* ∈ {"desktop", "browser", "mobile"}; mobile dispatches to
    ``LiteMobileActionSet``, desktop/browser to ``LiteDesktopActionSet``.
    Raises :class:`AguvisParseError` on unparseable / unsupported calls.

    *scroll_direction* is the direction an argument-less ``scroll()`` means,
    already resolved by :func:`scroll_direction_from_description` from the step's
    own ``Action:`` prose. It is ``None`` for the whole ``grounding.action``
    cohort, which has no prose turn at all — and needs none: an argument-less
    scroll occurs in **0** of the 9 stage-1 files (only ``omniact`` scrolls, and
    always with a wheel-click count), so the absent case is structural rather
    than a default standing in for a value somebody forgot to pass.
    """
    if not isinstance(code, str) or not code.strip():
        raise AguvisParseError(f"Expected non-empty code string. Got: {code!r}")
    try:
        module = ast.parse(code)
    except (TypeError, ValueError, SyntaxError) as e:
        raise AguvisParseError(f"ast.parse failed for code:\n{code}") from e

    is_mobile = platform == "mobile"
    tool_calls: list[dict[str, Any]] = []

    for stmt in module.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            raise AguvisParseError(f"Unsupported statement type: {type(stmt).__name__}.\ncode=\n{code}")
        call: ast.Call = stmt.value
        fname = _dotted_name(call.func)

        if fname in {"pyautogui.click", "click"}:
            coord = norm01_to_1000(*_extract_xy(call))
            tool_calls.append(LiteMobileActionSet.tap(coordinate=coord) if is_mobile
                              else LiteDesktopActionSet.click(coordinate=coord))
            continue
        if fname in {"pyautogui.rightClick", "rightClick"}:
            _reject_on_mobile(fname, is_mobile, code)
            coord = norm01_to_1000(*_extract_xy(call))
            tool_calls.append(LiteDesktopActionSet.click(coordinate=coord, button="right"))
            continue
        if fname in {"pyautogui.middleClick", "middleClick"}:
            _reject_on_mobile(fname, is_mobile, code)
            coord = norm01_to_1000(*_extract_xy(call))
            tool_calls.append(LiteDesktopActionSet.click(coordinate=coord, button="middle"))
            continue
        if fname in {"pyautogui.doubleClick", "doubleClick"}:
            has_coordinate = len(call.args) >= 2 or (
                _get_kw(call, "x") is not None and _get_kw(call, "y") is not None
            )
            if not has_coordinate and cursor_coordinate is None:
                raise AguvisParseError(f"{fname}() has no coordinate or prior cursor position")
            coord = (
                norm01_to_1000(*_extract_xy(call))
                if has_coordinate else list(cursor_coordinate)
            )
            # Mobile has a faithful equivalent: a double tap.
            tool_calls.append(LiteMobileActionSet.tap(coordinate=coord, clicks=2) if is_mobile
                              else LiteDesktopActionSet.click(coordinate=coord, clicks=2))
            continue
        if fname in {"pyautogui.tripleClick", "tripleClick", "computer.tripleClick"}:
            # Mobile ``tap`` accepts clicks in {1, 2} only — no triple-tap.
            _reject_on_mobile(fname, is_mobile, code)
            coord = norm01_to_1000(*_extract_xy(call))
            tool_calls.append(LiteDesktopActionSet.click(coordinate=coord, clicks=3))
            continue
        if fname in {"pyautogui.moveTo", "moveTo"}:
            # Touchscreens have no hover/cursor to move.
            _reject_on_mobile(fname, is_mobile, code)
            coord = norm01_to_1000(*_extract_xy(call))
            tool_calls.append(LiteDesktopActionSet.mouse_move(coordinate=coord))
            continue
        if fname in {"pyautogui.dragTo", "dragTo"}:
            coord = norm01_to_1000(*_extract_xy(call))
            if is_mobile:
                raise AguvisParseError(
                    f"{fname} has no mobile equivalent: no source start coordinate."
                    f"\ncode=\n{code}"
                )
            btn_node = _get_kw(call, "button")
            btn = "left" if btn_node is None else str(_literal_eval(btn_node))
            if btn not in {"left", "right", "middle"}:
                raise AguvisParseError(f"Unsupported dragTo button={btn!r}.\ncode=\n{code}")
            tool_calls.append(LiteDesktopActionSet.drag(coordinate=coord, button=btn))
            continue
        if fname in {"mobile.swipe", "swipe"}:
            from_node, to_node = _get_kw(call, "from_coord"), _get_kw(call, "to_coord")
            if from_node is not None and to_node is not None:
                fv, tv = _literal_eval(from_node), _literal_eval(to_node)
                sx, sy, ex, ey = float(fv[0]), float(fv[1]), float(tv[0]), float(tv[1])
            elif len(call.args) >= 4:
                sx, sy, ex, ey = (float(_literal_eval(call.args[i])) for i in range(4))
            else:
                sx_n = _get_kw(call, "startx") or _get_kw(call, "start_x")
                sy_n = _get_kw(call, "starty") or _get_kw(call, "start_y")
                ex_n = _get_kw(call, "endx") or _get_kw(call, "end_x") or _get_kw(call, "x")
                ey_n = _get_kw(call, "endy") or _get_kw(call, "end_y") or _get_kw(call, "y")
                if any(n is None for n in (sx_n, sy_n, ex_n, ey_n)):
                    raise AguvisParseError(f"Cannot extract swipe coordinates.\ncode=\n{code}")
                sx, sy, ex, ey = (float(_literal_eval(n)) for n in (sx_n, sy_n, ex_n, ey_n))
            tool_calls.append(LiteMobileActionSet.swipe(
                start_coordinate=norm01_to_1000(sx, sy), coordinate=norm01_to_1000(ex, ey)))
            continue
        if fname in {"pyautogui.scroll", "scroll", "pyautogui.hscroll", "hscroll"}:
            tool_calls.append(_scroll_tool_call(
                call, fname, is_mobile=is_mobile, code=code,
                scroll_direction=scroll_direction,
            ))
            continue
        if fname in {"pyautogui.hotkey", "hotkey"}:
            if len(call.args) == 1:
                kv = _literal_eval(call.args[0])
                kv = kv if isinstance(kv, (list, tuple)) else [kv]
            elif len(call.args) > 1:
                kv = [_literal_eval(a) for a in call.args]
            else:
                keys_node = _get_kw(call, "keys")
                if keys_node is None:
                    raise AguvisParseError(f"hotkey requires arguments.\ncode=\n{code}")
                kv = _literal_eval(keys_node)
                kv = kv if isinstance(kv, (list, tuple)) else [kv]
            keys = [str(k).lower() for k in kv]
            if is_mobile:
                # Mobile has no keyboard-chord action. Map keys that have a system-button
                # equivalent (e.g. enter); a genuine desktop-only chord (ctrl+a, …) has no
                # mobile equivalent → drop the step (raise) rather than emit a no-op type.
                btns = [_map_key_to_system_button(k) for k in keys]
                if all(btns):
                    for b in btns:
                        tool_calls.append(LiteMobileActionSet.system_button(button=b))
                else:
                    raise AguvisParseError(
                        f"hotkey {keys} has no mobile equivalent.\ncode=\n{code}")
            else:
                keys = [_canonical_desktop_key(k, code) for k in kv]
                tool_calls.append(LiteDesktopActionSet.key(keys=keys))
            continue
        if fname in {"pyautogui.press", "press"}:
            # Aguvis uses both positional ``press('down')`` and keyword
            # ``press(keys='down')`` / ``press(key='down')`` forms.
            key_node = call.args[0] if call.args else (_get_kw(call, "keys") or _get_kw(call, "key"))
            if key_node is None:
                raise AguvisParseError(f"press requires a key argument.\ncode=\n{code}")
            key_val = _literal_eval(key_node)
            keys = key_val if isinstance(key_val, (list, tuple)) else [key_val]
            for k in keys:
                if is_mobile:
                    btn = _map_key_to_system_button(str(k))
                    tool_calls.append(LiteMobileActionSet.system_button(button=btn) if btn
                                      else LiteMobileActionSet.type(text=str(k)))
                else:
                    tool_calls.append(
                        LiteDesktopActionSet.key(keys=[_canonical_desktop_key(k, code)])
                    )
            continue
        if fname in {"pyautogui.write", "pyautogui.typewrite", "write", "typewrite"}:
            msg_node = _get_kw(call, "message")
            if msg_node is None and len(call.args) >= 1:
                msg_node = call.args[0]
            if msg_node is None:
                raise AguvisParseError(f"write requires message argument.\ncode=\n{code}")
            text = str(_literal_eval(msg_node))
            tool_calls.append(LiteMobileActionSet.type(text=text) if is_mobile
                              else LiteDesktopActionSet.type(text=text))
            continue
        if fname in {"computer.wait", "pyautogui.sleep", "wait", "time.sleep", "mobile.wait"}:
            sec_node = _get_kw(call, "seconds") or _get_kw(call, "duration")
            if sec_node is not None:
                t = float(_literal_eval(sec_node))
            elif len(call.args) >= 1:
                t = float(_literal_eval(call.args[0]))
            else:
                t = 1.0
            tool_calls.append(LiteMobileActionSet.wait(duration=t) if is_mobile
                              else LiteDesktopActionSet.wait(duration=t))
            continue
        if fname in {"pyautogui.longClick", "longClick", "long_press", "mobile.long_press"}:
            coord = norm01_to_1000(*_extract_xy(call))
            dur_node = _get_kw(call, "duration")
            dur = float(_literal_eval(dur_node)) if dur_node is not None else None
            tool_calls.append(LiteMobileActionSet.long_press(coordinate=coord, duration=dur))
            continue
        if fname in {"mobile.home", "home"}:
            tool_calls.append(LiteMobileActionSet.system_button(button="Home")); continue
        if fname in {"mobile.back", "back"}:
            tool_calls.append(LiteMobileActionSet.system_button(button="Back")); continue
        if fname in {"mobile.recent", "recent"}:
            tool_calls.append(LiteMobileActionSet.system_button(button="Recent")); continue
        if fname in {"mobile.menu", "menu"}:
            tool_calls.append(LiteMobileActionSet.system_button(button="Menu")); continue
        if fname in {"mobile.enter", "enter"}:
            tool_calls.append(LiteMobileActionSet.system_button(button="Enter")); continue
        if fname in {"open", "open_app", "mobile.open_app"}:
            if len(call.args) >= 1:
                app = str(_literal_eval(call.args[0]))
            else:
                app_node = _get_kw(call, "app_name") or _get_kw(call, "name") or _get_kw(call, "app")
                if app_node is None:
                    raise AguvisParseError(f"open_app requires app name.\ncode=\n{code}")
                app = str(_literal_eval(app_node))
            tool_calls.append(make_tool_call("open_app", {"app_name": app}))
            continue
        if fname in {"computer.terminate", "mobile.terminate", "terminate"}:
            status_node = _get_kw(call, "status")
            if status_node is None and len(call.args) >= 1:
                status_node = call.args[0]
            if status_node is None:
                raise AguvisParseError(f"terminate requires status argument.\ncode=\n{code}")
            status = str(_literal_eval(status_node))
            if status == "fail":
                status = "failure"
            if status not in {"success", "failure"}:
                raise AguvisParseError(f"Unsupported terminate status={status!r}.\ncode=\n{code}")
            reason_node = _get_kw(call, "reason")
            reason = str(_literal_eval(reason_node)) if reason_node is not None else None
            tool_calls.append(make_tool_call("terminate", {"status": status, "reason": reason}))
            continue
        if fname in {"answer", "computer.answer", "response"}:
            if len(call.args) >= 1:
                text = str(_literal_eval(call.args[0]))
            else:
                text_node = _get_kw(call, "text") or _get_kw(call, "message")
                if text_node is None:
                    raise AguvisParseError(f"answer requires text argument.\ncode=\n{code}")
                text = str(_literal_eval(text_node))
            tool_calls.append(make_tool_call("response", {"text": text}))
            continue
        if fname in {"browser.select_option", "select_option"}:
            raise AguvisParseError(f"select_option not supported in cua-lite action space.\ncode=\n{code}")
        raise AguvisParseError(f"Unsupported function call: {fname!r}.\ncode=\n{code}")

    return tool_calls


# --- text helpers ----------------------------------------------------------
def clean_instruction(text: str) -> str:
    """Strip <image> tags and collapse a single leading newline."""
    text = text.replace("<image>", "").strip()
    return text.replace("\n", " ").strip() if "\n" not in text[1:] else text.strip()


# Stage-2 episode grouping (image filename → (episode_id, step_index)).
_STEP_SUFFIX_RE = re.compile(r"^(.+?)_(\d+)\.\w+(?:\.\w+)?$")
_DIR_STEP_RE = re.compile(r"^(.+?)/(?:screenshot|step)_?(\d+)\.\w+$")
_MINIWOB_RE = re.compile(r"^(.+?_seed\d+)_step(\d+)\..*$")


def episode_key(image_name: str) -> tuple[str, int]:
    """Extract (episode_id, step_index) from a Stage-2 image filename."""
    if not isinstance(image_name, str):
        return str(image_name), 0
    for rx in (_MINIWOB_RE, _DIR_STEP_RE, _STEP_SUFFIX_RE):
        m = rx.match(image_name)
        if m:
            return m.group(1), int(m.group(2))
    return image_name, 0


_INSTRUCTION_RE = re.compile(r"Instruction:\s*(.+?)(?:\n\n|\nPrevious actions:)", re.DOTALL)
# COAT (and other verbose l3-style) first gpt turn: "Observation: ... Thought: ... Action: ..."
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?:\nAction:|\Z)", re.DOTALL)
_ACTION_BLOCK_RE = re.compile(r"Action:\s*(.+)\Z", re.DOTALL)


def extract_goal(human_value: str) -> str:
    """Extract the task instruction from the first human turn's value."""
    m = _INSTRUCTION_RE.search(human_value)
    if m:
        return m.group(1).strip()
    cleaned = human_value.replace("<image>", "").strip()
    lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    return lines[0] if lines else ""


def extract_thought_and_desc(first_gpt_value: str) -> tuple[str, str]:
    """Split a Stage-2 first gpt turn into (inline_reasoning, action_description).

    Handles two shapes:
      - plain ``"Action: <desc>"``  → ("", "<desc>")
      - COAT-style ``"Observation: ... Thought: ... Action: ..."``
        → ("<thought>", "<action desc>")
    """
    text = (first_gpt_value or "").strip()
    thought_m = _THOUGHT_RE.search(text)
    if thought_m:
        thought = thought_m.group(1).strip()
        action_m = _ACTION_BLOCK_RE.search(text)
        desc = action_m.group(1).strip() if action_m else ""
        return thought, desc
    if text.startswith("Action: "):
        text = text[len("Action: "):]
    return "", text.rstrip()
