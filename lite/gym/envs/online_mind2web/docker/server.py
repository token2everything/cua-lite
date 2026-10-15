"""In-container FastAPI RPC server for cua-lite/online_mind2web.

The upstream Online-Mind2Web repo ships evaluator utilities, not a browser
rollout runtime. This server owns the CUA-Lite runtime inside one shared
SINGLETON container:

  GET  /healthz  -> Playwright/browser health
  POST /reset    -> fresh browser context/page, navigate to task website
  POST /step     -> execute Lite browser coordinate actions
  POST /close    -> close the per-trajectory browser context
"""

from __future__ import annotations

import asyncio
import base64
import html
import logging
import math
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from playwright.async_api import Browser, BrowserContext, Page, TimeoutError, async_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("online_mind2web-server")

_CAPTURE_CURSOR_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACgAAAA8CAYAAAAUufjgAAAFNklEQVR4nM3ZX0xTVxzA8e+5LVNIFxK53YNtjS7sYWabMTxhtvCiWdXGBUURcNnjMvcAj0tU/G8EAwjCeHBP4PayPSgSgpl/XgjZonGG7E+WrGStYTNLmtRIKfTPPXvA2/TPRXvb29JfcpLb29/vnE9P77m3txcppZBS2qnQUAAlmUyOSSlt640xDCmlTUopE4nEdxWJ1IEVi0wHViQyG1hxSCNgRSHXAlYMMmuRyOvXr1cWMhu4efNmOTY2VjlII6DL5ZLj4+OVgcwGulwu6XK5pNvtrgxkNtDj8aTali1b5I0bN9YVmfMjQQiR8frEiRMAdHR0AGCz2dqSySRSyk+FEMl1BwKcPHkSIQTt7e1lR+YF1JFA2ZF5AwFOnToFlBdpCgjQ3d2NEIK2trayIHOAiqK8tuj06dMIITh69GjJkaZnMB0JlBxZMBDgzJkzQGmRRQEBzp49ixCC1tbWkiCLBupIoCRIS4AA586dQwjBkSNHLEVmAIUQea3iteL8+fMAliItm0E9Lly4gBCCw4cPW4K0HKgjAUuQJQECXLx4ESEELS0tRSFLBtSRQFHIkgIBLl26BBSOLOhabDYuX76MEIJDhw6ZRpZ8BrORBw8eNIUsG1BHAqaQZQUC9PT0APkjyw4E6O3tRQhBc3Pza5HrAtSRwGuRlq3i6upqbt26hcPhKKh+LWSORghRUFteXmZqaqpgXBZyXP9zwDSwrq6OhoYGw/cmJiaQUlqKNHUMqqrK0NAQ4XCY48eP57wfCAR4/PgxDQ0NANy9e3diz5493+fp+hV4kr0zb6CqqgwODuLxePB4PNTX1+P3+3Pybt68mQI2NTU1ulyu7oWFhURWWhWQPZAAaoHn6Tvz+oqdTmcKp0dzc7Nh7uzsLKFQaFVRVeWcn5/fDvyW1Z4Avxi0DJwhUFGUjGaEA9i9ezcOhyMnX9M0bt++ncqz2WxfZI9hJl45g6qqcvXqVdxuNwCapiUjkchzgJqaGrxer+EsTk1NkUwmdeBHL168eM9yoKqqDAwMZOCGh4eHFxYWevVcn89nCAyFQszMzKT63LBhw5eWAlVVpb+/PwfX2dnZ43Q6RzRNiwBs27aNnTt3GiLTv2a73d4hpSzoDJ4DVFWVvr4+I1wv8O+mTZueLy8vf6vnHzhwwBA4NzdHMBjUP/Sb8Xj8s6KBQgiuXLlihOsD/tHzpJQj+vauXbuoq6vLAQJMTk6m+rbb7Z9bAtRXq6ZpyWvXro10dnb2A0/T8xwOx9zS0tJPLwfG5/PlrGZFUbh37x7RaFTv+/1oNNpUFFAPHdfV1dUPBI1yhBBD+rbX68Vut+fM4tLSEg8ePEjVVFVVmT7l5ADTcINAYK3C6urqH+LxeAhWj9vGxkbDY3F6ejpVY7PZPjG7WDKAabghYP5VhUKIeCwW+0Z/vX//fkOg3+8nEEh9zo0rKysfFgSUUibScLkXWaNiRRmRUiYBduzYgcfjMUTqq/nlOG8VAoxPT09/1dXV9XW+OICampqni4uLP8LqAjM65SiKwtatW1M1Qoj/zAJlLBZr37dv3yjwp5liAEVRBvTtvXv3ZvxWVBSFY8eOpV/Ho5FIZNbsGEVHOBz+WX9UpmmafPTokbxz544MBoMZj9FisVh/2XEADx8+fHtxcTEkXxHPnj2bk1JuXBcgQGtrq8/v9z/JhiUSiZX79+9P1tfXby+kXyvvMd8APm5paXnX6/W+U1tbuzEQCIRGR0d/9/v9fwAzgOkbFqtvgt8APgDcrN5OhIG/gL8L7fB/g30TfzONgi4AAAAASUVORK5CYII="
)
_CAPTURE_CURSOR_SRC = f"data:image/png;base64,{_CAPTURE_CURSOR_B64}"
_CAPTURE_CURSOR_WIDTH = 16
_CAPTURE_CURSOR_HEIGHT = 24

# Keep in sync with host-side model/action error handling. This file runs as an
# in-container script and must not import the host ``lite.gym`` package.
_MODEL_ACTION_ERROR_TYPES = (ValueError, TypeError, IndexError, KeyError)
_DEFAULT_MODEL_DURATION_CAP_SECONDS = 30.0
_MODEL_DURATION_CAPS_SECONDS = {
    "wait": 30.0,
    "hold_key": 5.0,
    "long_press": 5.0,
    "swipe": 5.0,
    "drag": 5.0,
}


def _coerce_bounded_seconds(
    value: Any,
    *,
    label: str,
    minimum: float = 0.0,
    maximum: float = _DEFAULT_MODEL_DURATION_CAP_SECONDS,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number, got bool")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"{label} must be a finite number")
        try:
            number = float(raw)
        except ValueError as e:
            raise ValueError(f"{label} must be a finite number") from e
    else:
        raise ValueError(f"{label} must be a finite number, got {type(value).__name__}")
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if number < minimum:
        raise ValueError(f"{label} must be >= {minimum:g}")
    if number > maximum:
        raise ValueError(f"{label} must be <= {maximum:g}")
    return number


def _coerce_model_duration(
    value: Any,
    *,
    action_name: str,
    field: str = "duration",
    minimum: float = 0.0,
) -> float:
    maximum = _MODEL_DURATION_CAPS_SECONDS.get(
        action_name,
        _DEFAULT_MODEL_DURATION_CAP_SECONDS,
    )
    return _coerce_bounded_seconds(
        value,
        label=f"{action_name}.{field}",
        minimum=minimum,
        maximum=maximum,
    )


def _coerce_cursor_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"cursor must be boolean-like, got {value!r}")
    if isinstance(value, float):
        if value in (0.0, 1.0):
            return bool(value)
        raise ValueError(f"cursor must be boolean-like, got {value!r}")
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"", "0", "false", "no", "off"}:
            return False
    raise ValueError(f"cursor must be boolean-like, got {value!r}")


class _UnsupportedAction(Exception):
    """The action has no Online-Mind2Web backend mapping."""


def _unsupported_action_message(name: str) -> str:
    return f"unsupported action: {name}"


def _raw_action_name(action: Any) -> str:
    if isinstance(action, dict):
        name = action.get("name")
        if isinstance(name, str) and name:
            return name
    return "<invalid>"


def _read_flat_action(action: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(action, dict):
        raise TypeError(f"Lite action must be an object, got {type(action).__name__}")
    name = action.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Lite action.name must be a non-empty string")
    args = action.get("arguments")
    if not isinstance(args, dict):
        raise TypeError(
            f"Lite tool_call.arguments must be a dict, got {type(args).__name__}"
        )
    return name, args


def _append_action_error(
    errors: list[str],
    action_errors: list[dict[str, Any]],
    *,
    index: int,
    kind: str,
    name: str,
    error: Any,
    action: Any = None,
    message: str | None = None,
) -> None:
    detail = str(error)
    message = message or f"{name}: {detail}"
    errors.append(message)
    record = {
        "index": index,
        "name": name,
        "error": detail,
        "message": message,
        "kind": kind,
    }
    if isinstance(action, dict) and action.get("call_id"):
        record["call_id"] = action["call_id"]
    action_errors.append(record)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # _startup_impl/_shutdown_impl are defined later in the module; name lookup
    # happens at call time (startup/shutdown), after the module is fully loaded.
    await _startup_impl()
    yield
    await _shutdown_impl()


app = FastAPI(title="cua-lite online_mind2web in-container server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Config. Host services forward yaml choices via docker run -e ONLINE_MIND2WEB_*.
# ---------------------------------------------------------------------------
_HEADLESS = os.environ.get("ONLINE_MIND2WEB_HEADLESS", "1") not in {"0", "false", "False"}
_WINDOW_W = int(os.environ.get("ONLINE_MIND2WEB_VIEWPORT_W", "1280"))
_WINDOW_H = int(os.environ.get("ONLINE_MIND2WEB_VIEWPORT_H", "720"))
# Cursor origin at session start, in the canonical [0, 1000] normalized space:
# the viewport CENTRE. A real desktop warps the pointer to screen centre at
# session start and the browser inherits it, so the turn-0 frame must show the
# cursor there. Seeding it (rather than leaving last_cursor None) is what keeps
# the FIRST observation shaped like every later one — a None seed made
# _show_capture_cursor return False and shipped a cursor-less turn-0 frame.
_CURSOR_ORIGIN_NORM: tuple[int, int] = (500, 500)
_PAGE_LOAD_TIMEOUT_MS = int(float(os.environ.get("ONLINE_MIND2WEB_PAGE_LOAD_TIMEOUT_S", "20")) * 1000)
_RESET_SETTLE_S = float(os.environ.get("ONLINE_MIND2WEB_RESET_SETTLE_S", "1.0"))
_POST_ACTION_DELAY_S = float(os.environ.get("ONLINE_MIND2WEB_POST_ACTION_DELAY_S", "0.5"))
_INSTANCE_TTL_S = float(os.environ.get("ONLINE_MIND2WEB_INSTANCE_TTL_S", "600"))
_REAPER_INTERVAL_S = float(os.environ.get("ONLINE_MIND2WEB_REAPER_INTERVAL_S", "30"))
_MAX_INSTANCES = int(os.environ.get("ONLINE_MIND2WEB_INSTANCES", "8"))
_AX_NODE_LIMIT = int(os.environ.get("ONLINE_MIND2WEB_AX_NODE_LIMIT", "800"))
_BODY_TEXT_LIMIT = int(os.environ.get("ONLINE_MIND2WEB_BODY_TEXT_LIMIT", "12000"))
_OBS_TIMEOUT_S = float(os.environ.get("ONLINE_MIND2WEB_OBS_TIMEOUT_S", "12"))
_OBS_FIELD_TIMEOUT_S = float(os.environ.get("ONLINE_MIND2WEB_OBS_FIELD_TIMEOUT_S", "4"))
_BASE_DOWNLOAD_DIR = Path(os.environ.get("ONLINE_MIND2WEB_DOWNLOAD_DIR", "/tmp/online_mind2web-downloads"))

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_BLANK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8//8/AwMDEwMDAwMDAwAkBgMB/DXemwAAAABJRU5ErkJggg=="
)


@dataclass
class _Instance:
    context: BrowserContext
    page: Page
    task_id: str
    instruction: str
    start_url: str
    download_dir: Path
    max_steps: int
    step_count: int = 0
    answer: str | None = None
    last_screenshot_png: bytes | None = None
    # Last pointer px; drag(start_coordinate=None) and the capture overlay use it.
    # reset() always seeds it to the viewport centre (_CURSOR_ORIGIN_NORM), so the
    # None default never survives a live instance.
    last_cursor: tuple[int, int] | None = None
    last_active: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_playwright: Any | None = None
_browser: Browser | None = None
_instances: dict[str, _Instance] = {}
_instances_lock = asyncio.Lock()
# Capacity slots reserved by an in-flight reset whose (slow) context build has not
# yet inserted into _instances. Guarded by _instances_lock. reset() reserves a
# slot atomically at the gate (reserve-before-build), so N concurrent resets can
# never overshoot _MAX_INSTANCES the way a bare len() check did (C1 TOCTOU).
_reserved = 0


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------
def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if url and "://" not in url:
        return f"https://{url}"
    return url or "https://www.google.com/"


_COOKIE_ACCEPT_SELECTORS = [
    "#onetrust-accept-btn-handler",
    "button:has-text('Accept All Cookies')",
    "button:has-text('Accept all cookies')",
    "button:has-text('Accept All')",
    "button:has-text('Accept Cookies')",
    "button:has-text('Agree and proceed')",
    "button:has-text('I Agree')",
    "[aria-label='Accept cookies']",
]


async def _auto_accept_cookies(page: Page) -> None:
    """Dismiss cookie consent banners that create full-page click-blocking overlays."""
    for selector in _COOKIE_ACCEPT_SELECTORS:
        try:
            locator = page.locator(selector).first
            if await locator.is_visible(timeout=300):
                await locator.click(timeout=1000)
                await asyncio.sleep(0.3)
                return
        except Exception:
            continue


async def _safe_goto(page: Page, url: str) -> str | None:
    normalized = _normalize_url(url)
    try:
        # Phase 1: wait for DOM so the page is at least structurally usable.
        # This never times out on reachable sites even if they are resource-heavy.
        await page.goto(
            normalized,
            wait_until="domcontentloaded",
            timeout=_PAGE_LOAD_TIMEOUT_MS,
        )
        # Phase 2: let JS-heavy SPAs finish rendering without blocking on ads/
        # analytics scripts that some sites load indefinitely.  A timeout here is
        # not an error — the DOM content is already available.
        try:
            await page.wait_for_load_state("load", timeout=_PAGE_LOAD_TIMEOUT_MS)
        except TimeoutError:
            logger.debug("load event did not fire within timeout for %s; proceeding", normalized)
        return None
    except TimeoutError:
        error = f"navigation timed out after {_PAGE_LOAD_TIMEOUT_MS / 1000:.1f}s"
        logger.warning("%s; stopping load: %s", error, normalized)
        try:
            await page.evaluate("window.stop()")
        except Exception:
            pass
        await _render_navigation_error(page, normalized, error)
        return error
    except Exception as e:
        error = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        logger.warning("navigation failed; continuing with current page: %s: %s", normalized, e)
        try:
            await page.evaluate("window.stop()")
        except Exception:
            pass
        await _render_navigation_error(page, normalized, error)
        return error


async def _render_navigation_error(page: Page, url: str, error: str) -> None:
    safe_url = html.escape(url)
    safe_error = html.escape(error)
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Navigation failed</title>
  <style>
    body {{
      margin: 0;
      background: #fff;
      color: #111827;
      font-family: Arial, sans-serif;
    }}
    main {{
      max-width: 760px;
      padding: 48px;
      line-height: 1.5;
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: 28px;
    }}
    code {{
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Navigation failed</h1>
    <p>The browser could not load <code>{safe_url}</code>.</p>
    <p>Error: <code>{safe_error}</code></p>
    <p>Try another direct URL, use search, or navigate with browser history.</p>
  </main>
</body>
</html>"""
    try:
        await page.set_content(content, wait_until="domcontentloaded", timeout=5000)
    except Exception as e:
        logger.warning("failed to render navigation error page for %s: %s", url, e)


async def _new_context(width: int, height: int, download_dir: Path) -> BrowserContext:
    if _browser is None:
        raise RuntimeError("browser not started")
    download_dir.mkdir(parents=True, exist_ok=True)
    context = await _browser.new_context(
        viewport={"width": width, "height": height},
        device_scale_factor=1,
        accept_downloads=True,
        user_agent=_USER_AGENT,
        locale="en-US",
        timezone_id="America/Chicago",
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
    )
    await context.add_init_script(
        """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });

// Redirect target=_blank anchor clicks to stay in the same tab. Computer-use
// agents operate on a single page; new tabs are invisible to the agent and
// leave the current page unchanged, causing infinite retry loops.
// Scope: only <a> click events — does NOT touch window.open or non-link JS.
document.addEventListener('click', function(e) {
    var el = e.target;
    while (el && el.tagName !== 'A') { el = el.parentElement; }
    if (el && el.target === '_blank') { el.target = '_self'; }
}, true);
"""
    )
    return context


def _norm_coord(
    coordinate: list[int] | tuple[int, int],
    width: int,
    height: int,
) -> tuple[int, int]:
    x_norm = float(coordinate[0])
    y_norm = float(coordinate[1])
    if not math.isfinite(x_norm) or not math.isfinite(y_norm):
        raise ValueError(f"coordinate values must be finite numbers: {coordinate!r}")
    x = max(0, min(width - 1, round(width * x_norm / 1000.0)))
    y = max(0, min(height - 1, round(height * y_norm / 1000.0)))
    return x, y


async def _viewport_size(page: Page) -> tuple[int, int]:
    size = page.viewport_size or {"width": _WINDOW_W, "height": _WINDOW_H}
    return int(size["width"]), int(size["height"])


async def _show_capture_cursor(page: Page, cursor: tuple[int, int] | None) -> bool:
    if cursor is None:
        return False
    x, y = cursor
    try:
        await page.evaluate(
            """([x, y, src, width, height]) => {
                let el = document.getElementById('cua-lite-capture-cursor');
                if (!el) {
                    el = document.createElement('img');
                    el.id = 'cua-lite-capture-cursor';
                    el.style.position = 'fixed';
                    el.style.zIndex = '2147483647';
                    el.style.pointerEvents = 'none';
                    el.style.transform = 'translate(0, 0)';
                    document.documentElement.appendChild(el);
                }
                el.src = src;
                el.style.width = `${width}px`;
                el.style.height = `${height}px`;
                el.style.left = `${x}px`;
                el.style.top = `${y}px`;
            }""",
            [x, y, _CAPTURE_CURSOR_SRC, _CAPTURE_CURSOR_WIDTH, _CAPTURE_CURSOR_HEIGHT],
        )
        return True
    except Exception as e:
        logger.warning("cursor capture overlay failed: %s", e)
        return False


async def _hide_capture_cursor(page: Page) -> None:
    try:
        await page.evaluate(
            """() => {
                const cursor = document.getElementById('cua-lite-capture-cursor');
                if (cursor) cursor.remove();
            }"""
        )
    except Exception:
        pass


def _blank_viewport_png(width: int = _WINDOW_W, height: int = _WINDOW_H) -> bytes:
    import io

    image = Image.new("RGB", (max(1, width), max(1, height)), (255, 255, 255))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _cdp_value(raw: Any) -> Any:
    if isinstance(raw, dict) and "value" in raw:
        return raw.get("value")
    return raw


def _compact_ax_tree(raw: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = raw.get("nodes") or []
    compact: list[dict[str, Any]] = []
    for node in nodes[:_AX_NODE_LIMIT]:
        item: dict[str, Any] = {
            "node_id": node.get("nodeId"),
            "role": _cdp_value(node.get("role")),
            "name": _cdp_value(node.get("name")),
        }
        for key in ("description", "value"):
            value = _cdp_value(node.get(key))
            if value not in (None, ""):
                item[key] = value
        child_ids = node.get("childIds")
        if child_ids:
            item["child_ids"] = child_ids
        backend_id = node.get("backendDOMNodeId")
        if backend_id is not None:
            item["backend_dom_node_id"] = backend_id
        compact.append(item)
    return compact


async def _a11y_tree(page: Page) -> list[dict[str, Any]]:
    session = None
    try:
        session = await asyncio.wait_for(
            page.context.new_cdp_session(page),
            timeout=_OBS_FIELD_TIMEOUT_S,
        )
        try:
            raw = await asyncio.wait_for(
                session.send("Accessibility.getFullAXTree"),
                timeout=_OBS_FIELD_TIMEOUT_S,
            )
        finally:
            try:
                await asyncio.wait_for(session.detach(), timeout=1.0)
            except Exception:
                pass
        return _compact_ax_tree(raw)
    except Exception as e:
        logger.warning("a11y snapshot failed: %s", e)
        return []


async def _body_text(page: Page) -> str:
    try:
        text = await page.locator("body").inner_text(timeout=1000)
    except Exception:
        return ""
    if len(text) > _BODY_TEXT_LIMIT:
        return text[:_BODY_TEXT_LIMIT] + "\n...[truncated]"
    return text


async def _capture_screenshot_png(inst: _Instance, *, cursor: bool = True) -> bytes:
    """Capture the viewport as PNG bytes, cursor overlay included.

    Owns the frame and nothing else, so a batch can take one frame per action
    without paying an a11y snapshot + body-text extraction per action;
    :func:`_observation` layers those fields on top for the turn's observation.
    A failed capture degrades through CDP, then the last good frame, then a
    blank viewport-sized frame; call it through
    :func:`_bounded_screenshot_png` to also bound a hang.
    """
    page = inst.page
    cursor_visible = await _show_capture_cursor(page, inst.last_cursor if cursor else None)
    try:
        try:
            png = await asyncio.wait_for(
                page.screenshot(type="png", full_page=False, timeout=8000),
                timeout=_OBS_FIELD_TIMEOUT_S + 1,
            )
        except Exception as e:
            logger.warning("page.screenshot failed; falling back to CDP capture: %s", e)
            session = None
            try:
                session = await asyncio.wait_for(
                    page.context.new_cdp_session(page),
                    timeout=_OBS_FIELD_TIMEOUT_S,
                )
                captured = await asyncio.wait_for(
                    session.send("Page.captureScreenshot", {"format": "png", "fromSurface": True}),
                    timeout=_OBS_FIELD_TIMEOUT_S,
                )
                png = base64.b64decode(captured["data"])
            except Exception as cdp_e:
                if inst.last_screenshot_png is not None:
                    logger.warning(
                        "CDP screenshot failed; reusing last good screenshot: %s",
                        cdp_e,
                    )
                    png = inst.last_screenshot_png
                else:
                    width, height = await _viewport_size(page)
                    logger.warning(
                        "CDP screenshot failed; returning viewport-sized blank screenshot: %s",
                        cdp_e,
                    )
                    png = _blank_viewport_png(width, height)
            finally:
                if session is not None:
                    try:
                        await asyncio.wait_for(session.detach(), timeout=1.0)
                    except Exception:
                        pass
        else:
            inst.last_screenshot_png = png
    finally:
        if cursor_visible:
            await _hide_capture_cursor(page)
    return png


async def _observation(inst: _Instance, *, cursor: bool = True) -> dict[str, Any]:
    page = inst.page
    png = await _capture_screenshot_png(inst, cursor=cursor)
    title = ""
    try:
        title = await asyncio.wait_for(page.title(), timeout=_OBS_FIELD_TIMEOUT_S)
    except Exception:
        pass
    try:
        a11y_tree = await asyncio.wait_for(_a11y_tree(page), timeout=_OBS_FIELD_TIMEOUT_S + 1)
    except Exception as e:
        logger.warning("a11y snapshot timed out: %s", e)
        a11y_tree = []
    try:
        body_text = await asyncio.wait_for(_body_text(page), timeout=_OBS_FIELD_TIMEOUT_S + 1)
    except Exception as e:
        logger.warning("body text snapshot timed out: %s", e)
        body_text = ""
    return {
        "screenshot_b64": base64.b64encode(png).decode("ascii"),
        "url": page.url,
        "title": title,
        "a11y_tree": a11y_tree,
        "body_text": body_text,
    }


async def _bounded_screenshot_png(inst: _Instance, *, cursor: bool = True) -> bytes:
    """Time-bounded :func:`_capture_screenshot_png`, for the per-action frames.

    Same policy as :func:`_bounded_observation`, which is why a hung page never
    reaches the caller: a frame capture inside the action loop must degrade to
    the last good frame, not turn the whole ``/step`` into a 500.
    """
    try:
        return await asyncio.wait_for(
            _capture_screenshot_png(inst, cursor=cursor), timeout=_OBS_TIMEOUT_S,
        )
    except Exception as e:
        if inst.last_screenshot_png is not None:
            logger.warning("frame capture timed out; reusing last good screenshot: %s", e)
            return inst.last_screenshot_png
        logger.warning("frame capture timed out; returning blank screenshot: %s", e)
        return _blank_viewport_png()


async def _bounded_observation(inst: _Instance, *, cursor: bool = True) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(_observation(inst, cursor=cursor), timeout=_OBS_TIMEOUT_S)
    except Exception as e:
        if inst.last_screenshot_png is not None:
            png = inst.last_screenshot_png
            logger.warning("observation timed out; reusing last good screenshot: %s", e)
        else:
            png = _blank_viewport_png()
            logger.warning("observation timed out; returning minimal observation: %s", e)
        return {
            "screenshot_b64": base64.b64encode(png).decode("ascii"),
            "url": inst.page.url,
            "title": "",
            "a11y_tree": [],
            "body_text": "",
        }


async def _close_instance(iid: str) -> bool:
    async with _instances_lock:
        inst = _instances.pop(iid, None)
    if inst is None:
        return False
    async with inst.lock:
        try:
            await inst.context.close()
        except Exception as e:
            logger.warning("context close failed for %s: %s", iid, e)
        shutil.rmtree(inst.download_dir, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# Action translation
# ---------------------------------------------------------------------------
_KEY_MAP = {
    "alt": "Alt",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "arrowup": "ArrowUp",
    "backspace": "Backspace",
    "cmd": "Meta",
    "ctrl": "Control",
    "control": "Control",
    "delete": "Delete",
    "end": "End",
    "enter": "Enter",
    "esc": "Escape",
    "escape": "Escape",
    "home": "Home",
    "option": "Alt",
    "pagedown": "PageDown",
    "pageup": "PageUp",
    "shift": "Shift",
    "space": "Space",
    "tab": "Tab",
    "win": "Meta",
}


def _pw_key(key: str) -> str:
    return _KEY_MAP.get(str(key).lower(), str(key))


def _model_key_list(value: Any, *, action_name: str, allow_empty: bool = True) -> list[str]:
    label = f"{action_name}.keys"
    if isinstance(value, str):
        raise ValueError(f"{label} must be a list of strings, not a string")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of strings")
    keys = list(value)
    if not keys and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    for index, key in enumerate(keys):
        if not isinstance(key, str):
            raise ValueError(f"{label}[{index}] must be a string")
    return keys


async def _execute_action(inst: _Instance, name: str, args: dict[str, Any]) -> dict[str, Any]:
    page = inst.page
    out: dict[str, Any] = {"call": name, "args": args}
    width, height = await _viewport_size(page)

    if name == "click":
        coordinate = args.get("coordinate")
        if coordinate is None:
            raise ValueError("click requires coordinate")
        x, y = _norm_coord(coordinate, width, height)
        button = str(args.get("button", "left")).lower()
        if button not in {"left", "right", "middle"}:
            button = "left"
        # Ensure the page has window focus before clicking. In headless Chromium
        # the window may not hold OS-level focus, causing dialog.showModal() and
        # other focus-gated interactions to silently fail after the click.
        await page.bring_to_front()
        await page.mouse.click(
            x,
            y,
            button=button,  # type: ignore[arg-type]
            click_count=int(args.get("clicks", 1) or 1),
        )
        inst.last_cursor = (x, y)

    elif name == "type":
        text = str(args.get("text", ""))
        typed = False
        try:
            active_tag = await page.evaluate(
                "document.activeElement?.tagName?.toLowerCase() ?? 'body'"
            )
            if active_tag in ("input", "textarea"):
                # Focused element is directly an input — use fill() which is
                # React-safe (sets native value, dispatches input/change events).
                await page.locator("input:focus, textarea:focus").first.fill(text)
                typed = True
            elif active_tag not in ("body", "html"):
                # Active element may be a custom autocomplete wrapper div.
                # Look for a visible, non-hidden input descendant and focus it.
                found = await page.evaluate("""() => {
                    const el = document.activeElement;
                    if (!el) return false;
                    const inner = el.querySelector(
                        'input:not([type=hidden]), textarea'
                    );
                    if (inner) { inner.focus(); return true; }
                    return false;
                }""")
                if found:
                    await page.locator("input:focus, textarea:focus").first.fill(text)
                    typed = True
        except Exception:
            pass
        if not typed:
            await page.keyboard.type(text, delay=50)
        if bool(args.get("press_enter", False)):
            await page.keyboard.press("Enter")

    elif name == "scroll":
        direction = str(args.get("direction", "down")).lower()
        amount = int(args.get("amount", 3) or 3)
        coordinate = args.get("coordinate")
        if coordinate is not None:
            x, y = _norm_coord(coordinate, width, height)
            await page.mouse.move(x, y)
            inst.last_cursor = (x, y)
        sign = -1 if direction in {"up", "left"} else 1
        # 100 px per scroll unit — the cua-lite convention (matches webharbor.webvoyager's
        # `amount*100` and webgym's `page_down` amount*100; the agent's wheel-click
        # `amount` maps to the same physical distance across all browser envs).
        pixels = max(1, amount) * 100
        if direction in {"left", "right"}:
            await page.mouse.wheel(sign * pixels, 0)
        else:
            await page.mouse.wheel(0, sign * pixels)

    elif name == "key":
        keys = args.get("keys")
        if keys is None and args.get("key") is not None:
            keys = [args["key"]]
        keys = [
            _pw_key(k)
            for k in _model_key_list(
                [] if keys is None else keys,
                action_name=name,
            )
        ]
        if keys:
            await page.keyboard.press("+".join(keys))

    elif name == "key_down":
        for key in _model_key_list(args.get("keys", []), action_name=name):
            await page.keyboard.down(_pw_key(key))

    elif name == "key_up":
        for key in _model_key_list(args.get("keys", []), action_name=name):
            await page.keyboard.up(_pw_key(key))

    elif name == "wait":
        await asyncio.sleep(
            _coerce_model_duration(
                args.get("duration", 1.0),
                action_name="wait",
            )
        )

    elif name in {"noop", "screenshot"}:
        pass

    elif name in {"back", "goback"}:
        try:
            await page.go_back(wait_until="domcontentloaded", timeout=_PAGE_LOAD_TIMEOUT_MS)
            try:
                await page.wait_for_load_state("load", timeout=_PAGE_LOAD_TIMEOUT_MS)
            except TimeoutError:
                pass
            await _auto_accept_cookies(page)
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"

    elif name in {"forward", "goforward"}:
        try:
            await page.go_forward(wait_until="domcontentloaded", timeout=_PAGE_LOAD_TIMEOUT_MS)
            try:
                await page.wait_for_load_state("load", timeout=_PAGE_LOAD_TIMEOUT_MS)
            except TimeoutError:
                pass
            await _auto_accept_cookies(page)
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"

    elif name == "refresh":
        try:
            await page.reload(wait_until="domcontentloaded", timeout=_PAGE_LOAD_TIMEOUT_MS)
            try:
                await page.wait_for_load_state("load", timeout=_PAGE_LOAD_TIMEOUT_MS)
            except TimeoutError:
                pass
            await _auto_accept_cookies(page)
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"

    elif name == "goto":
        url = args.get("url")
        if not url:
            raise ValueError("goto requires url")
        error = await _safe_goto(page, str(url))
        if error:
            out["error"] = error
        else:
            await _auto_accept_cookies(page)

    elif name == "drag":
        start = args.get("start_coordinate") or args.get("from") or args.get("source")
        end = args.get("coordinate") or args.get("end_coordinate") or args.get("to")
        if end is None:
            raise ValueError("drag requires coordinate")
        ex, ey = _norm_coord(end, width, height)
        # start_coordinate optional → drag from the current cursor (e.g. after a
        # preceding mouse_move), matching SandboxBaseEnv "drag from current cursor".
        if start is not None:
            sx, sy = _norm_coord(start, width, height)
        else:
            sx, sy = inst.last_cursor or (ex, ey)
        await page.mouse.move(sx, sy)
        button = str(args.get("button", "left")).lower()
        if button not in {"left", "right", "middle"}:
            button = "left"
        await page.mouse.down(button=button)  # type: ignore[arg-type]
        await page.mouse.move(ex, ey, steps=10)
        await page.mouse.up(button=button)  # type: ignore[arg-type]
        inst.last_cursor = (ex, ey)

    elif name in {"mouse_move", "hover"}:
        coordinate = args.get("coordinate")
        if coordinate is None:
            raise ValueError("mouse_move requires coordinate")
        x, y = _norm_coord(coordinate, width, height)
        await page.mouse.move(x, y)
        inst.last_cursor = (x, y)

    elif name == "mouse_down":
        button = str(args.get("button", "left")).lower()
        if button not in {"left", "right", "middle"}:
            button = "left"
        await page.mouse.down(button=button)  # type: ignore[arg-type]

    elif name == "mouse_up":
        button = str(args.get("button", "left")).lower()
        if button not in {"left", "right", "middle"}:
            button = "left"
        await page.mouse.up(button=button)  # type: ignore[arg-type]

    # NOTE: no `select` action. It would need a CSS `selector` the coordinate-
    # based agent cannot emit (online_mind2web valid_actions has no `select`), and the
    # upstream Online-Mind2Web benchmark drops native <select> steps. Direct
    # container callers get an indexed typed unsupported-action record.

    else:
        logger.warning("unknown online_mind2web action: %s(%s)", name, args)
        raise _UnsupportedAction(_unsupported_action_message(name))

    return out


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
async def _idle_reaper_loop() -> None:
    while True:
        await asyncio.sleep(_REAPER_INTERVAL_S)
        now = time.monotonic()
        stale: list[str] = []
        async with _instances_lock:
            for iid, inst in _instances.items():
                # Skip instances with an in-flight step (lock held): reaping one
                # mid-step would close the context out from under it (C2).
                if inst.lock.locked():
                    continue
                if now - inst.last_active > _INSTANCE_TTL_S:
                    stale.append(iid)
        for iid in stale:
            logger.warning("reaping idle online_mind2web instance %s", iid)
            await _close_instance(iid)


async def _startup_impl() -> None:
    global _playwright, _browser
    _BASE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(
        headless=_HEADLESS,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--disable-features=Translate",
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
        ],
    )
    asyncio.create_task(_idle_reaper_loop())
    logger.info(
        "online_mind2web server up: max_instances=%d viewport=%dx%d headless=%s",
        _MAX_INSTANCES,
        _WINDOW_W,
        _WINDOW_H,
        _HEADLESS,
    )


async def _shutdown_impl() -> None:
    global _playwright, _browser
    async with _instances_lock:
        ids = list(_instances)
    for iid in ids:
        await _close_instance(iid)
    if _browser is not None:
        await _browser.close()
        _browser = None
    if _playwright is not None:
        await _playwright.stop()
        _playwright = None


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    browser_ok = bool(_browser is not None and _browser.is_connected())
    return {
        "ok": browser_ok,
        "browser": browser_ok,
        "instances": len(_instances),
        "max_instances": _MAX_INSTANCES,
        "viewport": [_WINDOW_W, _WINDOW_H],
    }


@app.post("/reset")
async def reset(request: Request) -> dict[str, Any]:
    global _reserved
    body = await request.json()
    # Parse the body BEFORE reserving a slot, so a malformed body raises without
    # leaking a reservation.
    task_id = str(body.get("task_id") or uuid.uuid4().hex)
    instruction = str(body.get("instruction") or body.get("confirmed_task") or "")
    start_url = str(body.get("start_url") or body.get("website") or "https://www.google.com/")
    max_steps = int(body.get("max_steps") or 15)
    cursor = _coerce_cursor_bool(body.get("cursor"), default=True)
    viewport = body.get("viewport") or [_WINDOW_W, _WINDOW_H]
    width = int(body.get("window_width") or viewport[0])
    height = int(body.get("window_height") or viewport[1])

    # Reserve a capacity slot atomically BEFORE the slow context build. The
    # check + reserve runs with no await in between, so concurrent resets are
    # serialized by _instances_lock and can never exceed _MAX_INSTANCES (C1).
    async with _instances_lock:
        if len(_instances) + _reserved >= _MAX_INSTANCES:
            raise HTTPException(status_code=503, detail="online_mind2web pool full")
        _reserved += 1

    iid = uuid.uuid4().hex
    download_dir = _BASE_DOWNLOAD_DIR / iid
    context: BrowserContext | None = None
    try:
        context = await _new_context(width, height, download_dir)
        page = await context.new_page()
        page.set_default_timeout(_PAGE_LOAD_TIMEOUT_MS)
        page.set_default_navigation_timeout(_PAGE_LOAD_TIMEOUT_MS)
        navigation_error = await _safe_goto(page, start_url)
        if _RESET_SETTLE_S > 0:
            await asyncio.sleep(_RESET_SETTLE_S)
        await _auto_accept_cookies(page)
        inst = _Instance(
            context=context,
            page=page,
            task_id=task_id,
            instruction=instruction,
            start_url=_normalize_url(start_url),
            download_dir=download_dir,
            max_steps=max_steps,
            last_cursor=_norm_coord(_CURSOR_ORIGIN_NORM, width, height),
        )
        obs = await _bounded_observation(inst, cursor=cursor)
        if navigation_error:
            obs["navigation_error"] = navigation_error
    except Exception:
        if context is not None:
            await context.close()
        shutil.rmtree(download_dir, ignore_errors=True)
        async with _instances_lock:
            _reserved -= 1  # build failed -> release the reserved slot
        logger.exception("reset failed")
        raise

    async with _instances_lock:
        _instances[iid] = inst
        _reserved -= 1  # reservation became a live instance (count invariant held)

    return {
        "instance_id": iid,
        "instruction": instruction,
        "max_steps": max_steps,
        **obs,
    }


@app.post("/step")
async def step(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}") from e
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail=f"step body must be an object, got {type(body).__name__}",
        )
    iid = body.get("instance_id")
    if not iid:
        raise HTTPException(status_code=400, detail="missing instance_id")
    iid = str(iid)
    async with _instances_lock:
        inst = _instances.get(iid)
    if inst is None:
        raise HTTPException(status_code=404, detail=f"unknown instance {iid}")

    try:
        raw_post_action_delay = body.get("post_action_delay", _POST_ACTION_DELAY_S)
        if raw_post_action_delay is None:
            raw_post_action_delay = 0.0
        post_action_delay = _coerce_bounded_seconds(
            raw_post_action_delay,
            label="post_action_delay",
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"invalid post_action_delay: {e}") from e
    terminated = False
    cursor = _coerce_cursor_bool(body.get("cursor"), default=True)
    executed: list[dict[str, Any]] = []
    # One result frame per EXECUTED action, in action order. The host ships the
    # whole batch here, so this loop -- not the host -- is the only place that
    # can see the page between two actions of one batch. These are frames only:
    # the turn's a11y tree / body text / url still come from the single
    # observation taken after the loop, which also replaces the last frame.
    frames_b64: list[str] = []
    # ``terminate``/``response`` execute but never drive the page, so they break
    # out of the loop owing a frame. The post-loop observation becomes theirs by
    # EXTENDING rather than superseding -- otherwise it would overwrite the
    # previous action's frame and a terminating batch would report one frame for
    # two executed actions.
    finish_owes_frame = False
    errors: list[str] = []
    action_errors: list[dict[str, Any]] = []
    raw_actions = body.get("actions", [])
    if isinstance(raw_actions, list):
        actions = raw_actions
    else:
        _append_action_error(
            errors,
            action_errors,
            index=0,
            kind="model_action",
            name="<invalid>",
            error=f"body.actions must be a list, got {type(raw_actions).__name__}",
            action=raw_actions,
        )
        actions = []

    async with inst.lock:
        # Re-check membership under the lock (defense-in-depth, symmetric with
        # webharbor.webvoyager): if the reaper popped this iid, fail 404 instead of acting
        # on a closed context. Safe today (no await between the get above and
        # this acquire on the event loop), but robust to future changes (C2).
        async with _instances_lock:
            if _instances.get(iid) is not inst:
                raise HTTPException(status_code=404, detail=f"instance {iid} was reaped")
        inst.last_active = time.monotonic()
        for index, action in enumerate(actions):
            try:
                name, args = _read_flat_action(action)
            except _MODEL_ACTION_ERROR_TYPES as e:
                logger.warning("invalid action envelope for %s: %s", iid, e)
                _append_action_error(
                    errors,
                    action_errors,
                    index=index,
                    kind="model_action",
                    name=_raw_action_name(action),
                    error=e,
                    action=action,
                )
                continue

            if name == "terminate":
                terminated = True
                executed.append({"call": "terminate", "args": args})
                finish_owes_frame = True
                break

            if name == "response":
                inst.answer = str(args.get("text", ""))
                terminated = True
                executed.append({"call": "response", "args": args})
                finish_owes_frame = True
                break

            try:
                result = await _execute_action(inst, name, args)
                executed.append(result)
                frames_b64.append(
                    base64.b64encode(
                        await _bounded_screenshot_png(inst, cursor=cursor)
                    ).decode("ascii")
                )
                if result.get("error"):
                    message = f"{name}: {result['error']}"
                    _append_action_error(
                        errors,
                        action_errors,
                        index=index,
                        kind="tool_execution",
                        name=name,
                        error=result["error"],
                        action=action,
                        message=message,
                    )
            except _UnsupportedAction as e:
                logger.warning("unsupported action for %s: %s(%s): %s", iid, name, args, e)
                _append_action_error(
                    errors,
                    action_errors,
                    index=index,
                    kind="unsupported_action",
                    name=name,
                    error=e,
                    action=action,
                    message=str(e),
                )
            except _MODEL_ACTION_ERROR_TYPES as e:
                logger.warning("model action failed for %s: %s(%s): %s", iid, name, args, e)
                _append_action_error(
                    errors,
                    action_errors,
                    index=index,
                    kind="model_action",
                    name=name,
                    error=e,
                    action=action,
                )
            except Exception as e:
                logger.warning("action failed for %s: %s(%s): %s", iid, name, args, e)
                message = f"{name}: {e}"
                _append_action_error(
                    errors,
                    action_errors,
                    index=index,
                    kind="tool_execution",
                    name=name,
                    error=e,
                    action=action,
                    message=message,
                )

        if not terminated and post_action_delay > 0:
            await asyncio.sleep(post_action_delay)

        # The host owns the count and ships it (main.py step); it is the side
        # that also sees the turns whose calls were all rejected client-side
        # and never reach us. A private ``+= 1`` here could only ever fall
        # behind it, under-reporting truncation by one turn per rejected turn.
        inst.step_count = int(body["step_count"])
        truncated = not terminated and inst.step_count >= inst.max_steps
        obs = await _bounded_observation(inst, cursor=cursor)
        # This observation IS the last executed action's frame. Normally that
        # action already appended its own, so re-taking it after the settle
        # delay supersedes; when the batch broke on a finish action, or
        # when nothing executed at all, no frame is owed to anyone yet and this
        # one extends instead. Either way the count equals executed actions.
        if frames_b64 and not finish_owes_frame:
            frames_b64[-1] = obs["screenshot_b64"]
        else:
            frames_b64.append(obs["screenshot_b64"])
        # Heartbeat at step END too: a single step slower than the TTL must not
        # look idle to the reaper (C2). The reaper additionally skips any
        # instance whose lock is held, so an in-flight step is never reaped.
        inst.last_active = time.monotonic()

    return {
        "reward": None,
        # ONE frame per executed action, in action order. The model-visible
        # frame is the last entry; the host owns that selection.
        "screenshots_b64": frames_b64,
        "terminated": terminated,
        "truncated": truncated,
        "executed": executed,
        "errors": errors,
        "action_errors": action_errors,
        "answer": inst.answer,
        # ``obs`` carries url/title/a11y_tree/body_text AND its own singular
        # ``screenshot_b64``. Drop that one: ``screenshots_b64`` above is the
        # single owner of this step's frames, and returning both would be two
        # spellings of one fact. ``/reset`` legitimately keeps the singular key --
        # one initial observation is a different contract.
        **{k: v for k, v in obs.items() if k != "screenshot_b64"},
    }


@app.post("/close")
async def close(request: Request) -> dict[str, Any]:
    body = await request.json()
    closed = await _close_instance(str(body.get("instance_id", "")))
    return {"ok": True, "closed": closed}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("ONLINE_MIND2WEB_RPC_PORT", "8000")))
