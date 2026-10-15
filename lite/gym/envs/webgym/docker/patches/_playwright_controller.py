"""VENDORED + PATCHED OmniBoxes Playwright controller (cua-lite).

Base: omniboxes/node/instances/_playwright_controller.py @ pinned commit
574a205e (see docker/Dockerfile). The Dockerfile clones upstream @ that SHA and
``cp``s this file over
``/opt/omniboxes/omniboxes/node/instances/_playwright_controller.py`` before
``pip install`` — so there is no dependency on any cua-lite fork and the delta
is reviewable in one place.

PATCH — navigation no longer blocks on full page ``load``:
  Upstream ``visit_page`` did ``await page.goto(url)`` (Playwright default
  ``wait_until="load"`` — waits for the ``load`` event: DOM **plus every**
  sub-resource: images, fonts, ads, trackers) and THEN ``wait_for_load_state()``
  (waits for ``load`` AGAIN), with no navigation timeout (Playwright default
  30 s). So a single navigation to a heavy/slow site blocked for seconds — up to
  30 s+ — which is the dominant per-step latency on real websites (NOT the
  master/node proxy, which stays sub-second at K=64). The agent only needs the
  DOM ready, not every tracker, so this returns at ``domcontentloaded`` (capped
  10 s) then waits for full ``load`` only best-effort (capped 3 s) so fast sites
  still yield fully-rendered screenshots while slow sites stop burning 30 s.
  Safety net for a screenshot that races late-rendering content: in the shipped
  default the post-action delay is OFF (``post_action_delay: 0.0`` in
  configs/default.yaml), so the ONLY net is the blank-screenshot retry — and
  that retry fires only on a STRICT 100%-white frame (``is_white_image``,
  white_percentage == 1.0). A page that has painted a non-white spinner /
  partial header by the time the best-effort 3 s ``load`` wait expires is NOT
  re-shot. The A/B showing identical eval scores is the empirical cover; raise
  ``post_action_delay`` if a future task set needs a real paint-settle buffer
  on the coordinate-based action paths (which have no per-action sleep).

  The SAME treatment is applied to ``on_new_page`` (which ``_ensure_page_ready``
  calls before EVERY action + screenshot, not just navigation): it previously did
  ``wait_for_load_state(timeout=30000)`` on the ``load`` event, so on ad/tracker-
  heavy pages every single step burned the full 30 s. Capping it the same way
  (domcontentloaded 10 s + best-effort load 3 s) is what actually removes the
  per-step latency, since clicks/types/back/screenshots all route through it.

PATCH — block ad/tracker/analytics/media requests at the network layer:
  The capped load-wait above stops a heavy page from burning the full 30 s, but
  the DOMINANT remaining per-step tail (A/B: act p90 ~77 s, ~21 % of steps >30 s)
  is real third-party requests — analytics beacons, ad exchanges, trackers,
  autoplay video/audio — that keep the ``load`` event pending for tens of
  seconds on ad-heavy sites. ``on_new_page`` registers a ``page.route`` that
  ABORTS those requests (matched by a curated third-party domain list, plus the
  ``media``/``font`` resource types) while KEEPING ``document``/``image``/
  ``stylesheet``/``script``/``xhr``/``fetch`` — so the vision agent's screenshots
  stay faithful (images + CSS intact; text falls back to a system font) while the
  page settles far sooner. The route is registered ONCE per page under a
  ``_cua_setup_done`` guard (``page.route`` / ``add_init_script`` ACCUMULATE — an
  unguarded re-register on every ``_ensure_page_ready`` would stack handlers and
  re-fire the init script on every future navigation).
"""
import asyncio
import base64
import io
import os
import random
import time
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union, TypeVar, Awaitable, cast
from pathlib import Path

from playwright._impl._errors import Error as PlaywrightError
from playwright._impl._errors import TimeoutError, TargetClosedError
from playwright.async_api import Download, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from PIL import Image

from omniboxes.node.instances._types import (
    InteractiveRegion,
    VisualViewport,
    interactiveregion_from_dict,
    visualviewport_from_dict,
)

_CAPTURE_CURSOR_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACgAAAA8CAYAAAAUufjgAAAFNklEQVR4nM3ZX0xTVxzA8e+5LVNIFxK53YNtjS7sYWabMTxhtvCiWdXGBUURcNnjMvcAj0tU/G8EAwjCeHBP4PayPSgSgpl/XgjZonGG7E+WrGStYTNLmtRIKfTPPXvA2/TPRXvb29JfcpLb29/vnE9P77m3txcppZBS2qnQUAAlmUyOSSlt640xDCmlTUopE4nEdxWJ1IEVi0wHViQyG1hxSCNgRSHXAlYMMmuRyOvXr1cWMhu4efNmOTY2VjlII6DL5ZLj4+OVgcwGulwu6XK5pNvtrgxkNtDj8aTali1b5I0bN9YVmfMjQQiR8frEiRMAdHR0AGCz2dqSySRSyk+FEMl1BwKcPHkSIQTt7e1lR+YF1JFA2ZF5AwFOnToFlBdpCgjQ3d2NEIK2trayIHOAiqK8tuj06dMIITh69GjJkaZnMB0JlBxZMBDgzJkzQGmRRQEBzp49ixCC1tbWkiCLBupIoCRIS4AA586dQwjBkSNHLEVmAIUQea3iteL8+fMAliItm0E9Lly4gBCCw4cPW4K0HKgjAUuQJQECXLx4ESEELS0tRSFLBtSRQFHIkgIBLl26BBSOLOhabDYuX76MEIJDhw6ZRpZ8BrORBw8eNIUsG1BHAqaQZQUC9PT0APkjyw4E6O3tRQhBc3Pza5HrAtSRwGuRlq3i6upqbt26hcPhKKh+LWSORghRUFteXmZqaqpgXBZyXP9zwDSwrq6OhoYGw/cmJiaQUlqKNHUMqqrK0NAQ4XCY48eP57wfCAR4/PgxDQ0NANy9e3diz5493+fp+hV4kr0zb6CqqgwODuLxePB4PNTX1+P3+3Pybt68mQI2NTU1ulyu7oWFhURWWhWQPZAAaoHn6Tvz+oqdTmcKp0dzc7Nh7uzsLKFQaFVRVeWcn5/fDvyW1Z4Avxi0DJwhUFGUjGaEA9i9ezcOhyMnX9M0bt++ncqz2WxfZI9hJl45g6qqcvXqVdxuNwCapiUjkchzgJqaGrxer+EsTk1NkUwmdeBHL168eM9yoKqqDAwMZOCGh4eHFxYWevVcn89nCAyFQszMzKT63LBhw5eWAlVVpb+/PwfX2dnZ43Q6RzRNiwBs27aNnTt3GiLTv2a73d4hpSzoDJ4DVFWVvr4+I1wv8O+mTZueLy8vf6vnHzhwwBA4NzdHMBjUP/Sb8Xj8s6KBQgiuXLlihOsD/tHzpJQj+vauXbuoq6vLAQJMTk6m+rbb7Z9bAtRXq6ZpyWvXro10dnb2A0/T8xwOx9zS0tJPLwfG5/PlrGZFUbh37x7RaFTv+/1oNNpUFFAPHdfV1dUPBI1yhBBD+rbX68Vut+fM4tLSEg8ePEjVVFVVmT7l5ADTcINAYK3C6urqH+LxeAhWj9vGxkbDY3F6ejpVY7PZPjG7WDKAabghYP5VhUKIeCwW+0Z/vX//fkOg3+8nEEh9zo0rKysfFgSUUibScLkXWaNiRRmRUiYBduzYgcfjMUTqq/nlOG8VAoxPT09/1dXV9XW+OICampqni4uLP8LqAjM65SiKwtatW1M1Qoj/zAJlLBZr37dv3yjwp5liAEVRBvTtvXv3ZvxWVBSFY8eOpV/Ho5FIZNbsGEVHOBz+WX9UpmmafPTokbxz544MBoMZj9FisVh/2XEADx8+fHtxcTEkXxHPnj2bk1JuXBcgQGtrq8/v9z/JhiUSiZX79+9P1tfXby+kXyvvMd8APm5paXnX6/W+U1tbuzEQCIRGR0d/9/v9fwAzgOkbFqtvgt8APgDcrN5OhIG/gL8L7fB/g30TfzONgi4AAAAASUVORK5CYII="
)
_CAPTURE_CURSOR_SRC = f"data:image/png;base64,{_CAPTURE_CURSOR_B64}"
_CAPTURE_CURSOR_WIDTH = 16
_CAPTURE_CURSOR_HEIGHT = 24
_CAPTURE_CURSOR_CACHE: dict[int, Image.Image] = {}


def _capture_cursor_sprite(height: int = _CAPTURE_CURSOR_HEIGHT) -> Image.Image:
    sprite = _CAPTURE_CURSOR_CACHE.get(height)
    if sprite is None:
        raw = Image.open(io.BytesIO(base64.b64decode(_CAPTURE_CURSOR_B64))).convert("RGBA")
        aspect = raw.width / raw.height
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        sprite = raw.resize((max(1, int(height * aspect)), height), resample)
        _CAPTURE_CURSOR_CACHE[height] = sprite
    return sprite


def _overlay_capture_cursor(
    png: bytes,
    x: float,
    y: float,
    *,
    viewport_width: int,
    viewport_height: int,
) -> bytes:
    raw = Image.open(io.BytesIO(png))
    img = raw.convert("RGBA")
    scale_x = img.width / viewport_width if viewport_width > 0 else 1.0
    scale_y = img.height / viewport_height if viewport_height > 0 else scale_x
    x_px = max(0, min(int(round(x * scale_x)), img.width - 1))
    y_px = max(0, min(int(round(y * scale_y)), img.height - 1))
    cursor_height = max(1, int(round(_CAPTURE_CURSOR_HEIGHT * scale_y)))
    sprite = _capture_cursor_sprite(cursor_height)
    img.paste(sprite, (x_px, y_px), sprite)
    img = img.convert(raw.mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# Some of the Code for clicking coordinates and keypresses adapted from https://github.com/openai/openai-cua-sample-app/blob/main/computers/base_playwright.py
# Copyright 2025 OpenAI - MIT License
# Key-name resolution now happens HOST-side (lite/gym/utils/backend/keys.to_playwright):
# the host sends FINAL Playwright key names, so this controller just runs them.
# (The old map mis-mapped "/"→"Divide" / "\\"→"Backslash" — numpad keys that
# don't match the literal slash/backslash.)

# cua-lite perf patch: abort third-party ad/tracker/analytics requests that keep
# the page ``load`` event pending for tens of seconds on heavy sites (the real
# env.step tail). Matched as a substring of the request URL. KEEP first-party
# content + images + CSS so the vision agent's screenshots stay faithful. Env var
# ``WEBGYM_NO_BLOCK=1`` disables the route (escape hatch / A-side of an A/B).
_BLOCK_DOMAINS: tuple[str, ...] = (
    # Google analytics / tag manager / ads
    "google-analytics.com", "googletagmanager.com", "googlesyndication.com",
    "googleadservices.com", "doubleclick.net", "google.com/ads", "adservice.google.",
    "2mdn.net", "googletagservices.com",
    # Meta / Facebook pixel + ads
    "connect.facebook.net", "facebook.com/tr", "fbcdn.net/ads",
    # Microsoft / Bing / LinkedIn ads + analytics
    "bat.bing.com", "clarity.ms", "ads.linkedin.com", "px.ads.linkedin.com",
    # Twitter/X, TikTok, Reddit, Pinterest, Snap ads/analytics
    "ads-twitter.com", "analytics.twitter.com", "static.ads-twitter.com",
    "analytics.tiktok.com", "ads.tiktok.com", "redditstatic.com/ads",
    "ct.pinterest.com", "sc-static.net", "tr.snapchat.com",
    # Ad exchanges / SSPs / DSPs
    "adnxs.com", "rubiconproject.com", "pubmatic.com", "openx.net",
    "casalemedia.com", "criteo.com", "criteo.net", "taboola.com",
    "outbrain.com", "amazon-adsystem.com", "adsystem.com", "moatads.com",
    "adsrvr.org", "3lift.com", "bidswitch.net", "smartadserver.com",
    "yieldmo.com", "sharethrough.com",
    # Product analytics / session replay / RUM
    "segment.io", "segment.com", "cdn.segment.com", "amplitude.com",
    "mixpanel.com", "hotjar.com", "mouseflow.com", "fullstory.com",
    "logrocket.com", "loggly.com", "newrelic.com", "nr-data.net",
    "bugsnag.com", "browser-intake-datadoghq.com", "sentry.io",
    # Audience measurement
    "scorecardresearch.com", "quantserve.com", "quantcount.com",
    "comscore.com", "chartbeat.com", "parsely.com", "mc.yandex.ru",
    "cdn.cookielaw.org", "onetrust.com",
)
_BLOCK_RESOURCE_TYPES: frozenset[str] = frozenset({"media", "font"})
_NO_BLOCK = os.environ.get("WEBGYM_NO_BLOCK") == "1"

async def _route_blocker(route: Any) -> None:
    """Abort ad/tracker/analytics/media/font requests; continue everything else."""
    try:
        req = route.request
        if req.resource_type in _BLOCK_RESOURCE_TYPES or any(d in req.url for d in _BLOCK_DOMAINS):
            await route.abort()
            return
        await route.continue_()
    except Exception:
        # Routing best-effort: never let a route handler failure break a step.
        try:
            await route.continue_()
        except Exception:
            pass

F = TypeVar('F', bound=Callable[..., Awaitable[Any]])

def handle_target_closed(max_retries: int = 2, timeout_secs: int = 30):
    """
    Decorator to handle TargetClosedError by attempting to recover the page.
    
    Args:
        max_retries: Maximum number of retry attempts
        timeout_secs: Timeout for page operations during recovery
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract the page object - assume it's the first argument after self
            page = None
            if len(args) >= 2 and hasattr(args[1], 'url'):  # Check if second arg looks like a Page
                page = args[1]
            
            retries = 0
            last_error = None
            
            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except TargetClosedError as e:
                    last_error = e
                    retries += 1
                    
                    if retries > max_retries:
                        raise e
                    
                    if page is None:
                        # Can't recover without page reference
                        raise e
                    
                    print(f"TargetClosedError in {func.__name__}, attempting recovery (retry {retries}/{max_retries})")
                    
                    try:
                        # Attempt to recover the page
                        await _recover_page(page, timeout_secs)
                        # Small delay before retry
                        await asyncio.sleep(0.5)
                    except Exception as recovery_error:
                        print(f"Page recovery failed: {recovery_error}")
                        # If recovery fails, raise the original error
                        raise e from recovery_error
            
            # This shouldn't be reached, but just in case
            raise last_error
        
        return wrapper
    return decorator

async def _recover_page(page: Page, timeout_secs: int = 30) -> None:
    """
    Attempt to recover a closed page by reloading it.
    
    Args:
        page: The Playwright page object to recover
        timeout_secs: Timeout for recovery operations
    """
    try:
        # First, try to check if the page is still responsive
        await page.evaluate("1", timeout=1000)
        # If we get here, the page is actually fine
        return
    except Exception:
        # Page is indeed problematic, attempt recovery
        pass
    
    try:
        # Stop any ongoing navigation
        await page.evaluate("window.stop()", timeout=2000)
    except Exception:
        # Ignore errors from window.stop()
        pass
    
    try:
        # Try to reload the page
        await page.reload(timeout=timeout_secs * 1000)
        await page.wait_for_load_state("load", timeout=timeout_secs * 1000)
        print("playwright_controller._recover_page(): Page recovery successful")
    except Exception as e:
        print(f"playwright_controller._recover_page(): Page reload failed: {e}")
        
        # Try alternative recovery: navigate to current URL
        try:
            current_url = page.url
            if current_url and current_url != "about:blank":
                await page.goto(current_url, timeout=timeout_secs * 1000)
                await page.wait_for_load_state("load", timeout=timeout_secs * 1000)
                print("playwright_controller._recover_page(): Page recovery via goto successful")
            else:
                raise Exception("playwright_controller._recover_page(): No valid URL to navigate to")
        except Exception as goto_error:
            raise Exception(f"playwright_controller._recover_page(): All recovery methods failed. Reload error: {e}, Goto error: {goto_error}")

class PlaywrightController:
    def __init__(
        self,
        animate_actions: bool = False,
        downloads_folder: Optional[str] = None,
        viewport_width: int = 1440,
        viewport_height: int = 900,
        _download_handler: Optional[Callable[[Download], None]] = None,
        to_resize_viewport: bool = True,
        single_tab_mode: bool = False,
        sleep_after_action: int = 10,
        timeout_load: int = 1,
        timeout_action: int = 10,
        timeout_download: int = 60,
    ) -> None:
        """
        A controller for Playwright to interact with web pages.
        """
        self.animate_actions = animate_actions
        self.downloads_folder = downloads_folder
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._download_handler = _download_handler
        self.to_resize_viewport = to_resize_viewport
        self.single_tab_mode = single_tab_mode
        self._sleep_after_action = sleep_after_action
        self._timeout_load = timeout_load

        self._page_script: str = ""
        # Cursor origin at session start = viewport CENTRE, not top-left. A real
        # desktop warps the pointer to screen centre at session start and the
        # browser inherits it; (0, 0) is the "never moved" sentinel, not a real
        # pointer state. Keeps the turn-0 frame consistent with every later frame
        # and with the other cursor envs (captcha / online_mind2web /
        # webharbor.webvoyager / waa / cua.*), which all seed the centre. Also
        # feeds drag(start_coordinate=None), so a first drag starts from centre.
        self.last_cursor_position: Tuple[float, float] = (
            viewport_width / 2.0,
            viewport_height / 2.0,
        )

        # Load page script
        script_path = Path(__file__).parent / "_page_script.js"
        with open(script_path, "rt") as fh:
            self._page_script = fh.read()

    async def sleep(self, page: Page, duration: Union[int, float]) -> None:
        await page.wait_for_timeout(duration * 1000)

    @handle_target_closed()
    async def get_interactive_rects(self, page: Page) -> Dict[str, InteractiveRegion]:
        await self._ensure_page_ready(page)
        # Read the regions from the DOM
        try:
            await page.evaluate(self._page_script)
        except Exception:
            pass
        result = cast(Dict[str, Dict[str, Any]], await page.evaluate("MultimodalWebSurfer.getInteractiveRects();"))

        # Convert the results into appropriate types
        assert isinstance(result, dict)
        typed_results: Dict[str, InteractiveRegion] = {}
        for k in result:
            assert isinstance(k, str)
            typed_results[k] = interactiveregion_from_dict(result[k])

        return typed_results

    @handle_target_closed()
    async def get_visual_viewport(self, page: Page) -> VisualViewport:
        await self._ensure_page_ready(page)
        try:
            await page.evaluate(self._page_script)
        except Exception:
            pass
        return visualviewport_from_dict(await page.evaluate("MultimodalWebSurfer.getVisualViewport();"))

    @handle_target_closed()
    async def get_focused_rect_id(self, page: Page) -> str:
        await self._ensure_page_ready(page)
        try:
            await page.evaluate(self._page_script)
        except Exception:
            pass
        result = await page.evaluate("MultimodalWebSurfer.getFocusedElementId();")
        return str(result)

    @handle_target_closed()
    async def get_page_metadata(self, page: Page) -> Dict[str, Any]:
        assert page is not None
        
        # Initialize result with guaranteed fields
        result = {
            "title": "Unknown Page",
            "url": "about:blank"
        }
        
        try:
            # Get basic page information - these should always work
            try:
                title = await page.title()
                if title and title.strip():
                    result["title"] = title.strip()
            except Exception:
                pass
                
            try:
                url = page.url
                if url and url.strip():
                    result["url"] = url.strip()
            except Exception:
                pass
            
            # Try to get additional structured metadata (optional)
            attempts = 3
            while attempts > 0:
                try:
                    await self._ensure_page_ready(page)
                    await page.evaluate(self._page_script)
                    
                    # Get structured metadata from the page script
                    structured_data = await page.evaluate("MultimodalWebSurfer.getPageMetadata();")
                    if isinstance(structured_data, dict):
                        # Merge structured data with basic metadata, keeping title and url as primary
                        for key, value in structured_data.items():
                            if key not in result:  # Don't override title and url
                                result[key] = value
                    break
                except Exception as e:
                    print(f"Error getting structured metadata: {str(e)}, attempting again...")
                    attempts -= 1
                    if attempts > 0:
                        time.sleep(0.5)
                    
        except Exception as e:
            print(f"Error in get_page_metadata: {str(e)}")
            # result already has default values, so we can continue
            
        return result

    @handle_target_closed()
    async def on_new_page(self, page: Page) -> None:
        assert page is not None
        # cua-lite: per-page setup runs ONCE. ``_ensure_page_ready`` calls this
        # before every action+screenshot, and ``add_init_script`` / ``page.route``
        # / the download listener all ACCUMULATE — re-registering on every call
        # would stack handlers and re-fire the init script on each future
        # navigation. The ``_cua_setup_done`` guard makes it idempotent.
        if not getattr(page, "_cua_setup_done", False):
            if self._download_handler:
                page.on("download", self._download_handler) # type: ignore
            if self.to_resize_viewport and self.viewport_width and self.viewport_height:
                await page.set_viewport_size({"width": self.viewport_width, "height": self.viewport_height})
            script_path = Path(__file__).parent / "_page_script.js"
            await page.add_init_script(path=str(script_path))
            if not _NO_BLOCK:
                # Abort ad/tracker/media requests so the ``load`` event settles
                # fast on heavy pages (the dominant env.step tail). See docstring.
                await page.route("**/*", _route_blocker)
            try:
                page._cua_setup_done = True  # type: ignore[attr-defined]
            except Exception:
                pass
        await self.sleep(page, 0.2)
        # Hung-page fast-path: a page that already proved it won't reach
        # domcontentloaded (a prior wait here TIMED OUT) is "load-dead" — re-waiting
        # another 10s before each of the 6-8 getters is futile and is exactly what
        # stacks a hung step toward the 90s step_timeout cap. Skip the wait until a
        # new MAIN-FRAME navigation commits (the framenavigated hook in
        # _ensure_page_ready clears the flag). The flag is set ONLY after a real
        # timeout, so legit pages (DOM-ready in <3s) never enter this path → zero
        # effect on normal steps; only genuine hangs are short-circuited.
        if getattr(page, "_cua_load_dead", False):
            return
        try:
            # cua-lite patch (mirrors visit_page): DOM-ready is enough for the
            # agent, so return at ``domcontentloaded`` (capped 10s) then wait for
            # full ``load`` only best-effort (capped 3s). ``on_new_page`` gates
            # EVERY action + screenshot via ``_ensure_page_ready``, and the old
            # ``wait_for_load_state(timeout=30000)`` blocked on the ``load`` event
            # — which on ad/tracker-heavy pages never fires — so it burned the
            # full 30s on every step even though the visual content settled in the
            # first seconds (this, not the master/node proxy, was the dominant
            # per-step latency). With the default post_action_delay=0 the
            # blank-screenshot retry (strict 100%-white only) is the sole net for
            # late-rendering content. See module docstring.
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            try:
                page._cua_load_dead = False  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                await page.wait_for_load_state("load", timeout=3000)
            except PlaywrightTimeoutError:
                pass
        except PlaywrightTimeoutError:
            try:
                page._cua_load_dead = True  # type: ignore[attr-defined]
            except Exception:
                pass
            print("WARNING: Page load timeout, page might not be loaded")
            # stop page loading
            await page.evaluate("window.stop()")

    def _mark_action(self, page: Page) -> None:
        # A state-changing action just ran: force the NEXT _ensure_page_ready to do
        # a FULL on_new_page wait. Its domcontentloaded wait blocks for any
        # navigation the action triggered, so the screenshot can never return a
        # stale/unchanged frame after a real action (no noop / missed-transition).
        try:
            page._cua_action_pending = True  # type: ignore[attr-defined]
        except Exception:
            pass

    def _set_last_cursor_position(self, x: float, y: float) -> None:
        self.last_cursor_position = (x, y)

    @handle_target_closed()
    async def _ensure_page_ready(self, page: Page) -> None:
        assert page is not None
        # cua-lite memo. PROBLEM: on_new_page's load-wait (domcontentloaded 10s +
        # best-effort load 3s) was paid in EVERY getter — get_screenshot,
        # get_page_metadata (x3), get_interactive_rects, get_visual_viewport, etc.
        # On a SETTLED page each redundant wait is ~instant, so this is a no-op for
        # normal steps. But on a HUNG page (dead host / never-fires-load) each wait
        # times out at 10s, so 6-8 STACK to ~60-80s (the step_timeout cap) — the real
        # env.step tail. FIX: wait at most ONCE per (main-frame nav OR state-changing
        # action); redundant reads on an already-settled page skip the re-wait.
        # SAFETY (no noop / missed transitions): every committed MAIN-FRAME
        # navigation bumps _cua_nav_epoch (framenavigated, main-frame only so
        # ad-iframes don't churn it) and every action sets _cua_action_pending — so
        # the FIRST ensure after either still does the FULL wait the screenshot
        # relies on. Only the *redundant* re-waits by read-only getters later in the
        # same step are skipped; they cannot change the captured frame.
        if not getattr(page, "_cua_hooked", False):
            page._cua_nav_epoch = 0  # type: ignore[attr-defined]
            page._cua_ready_epoch = -1  # type: ignore[attr-defined]
            if not hasattr(page, "_cua_action_pending"):
                page._cua_action_pending = False  # type: ignore[attr-defined]
            def _bump(frame: Any) -> None:
                try:
                    if frame == page.main_frame:
                        page._cua_nav_epoch += 1  # type: ignore[attr-defined]
                        # A new main-frame nav committed — it may load fine, so clear
                        # the hung-page fast-path flag and give this page a fresh wait.
                        page._cua_load_dead = False  # type: ignore[attr-defined]
                except Exception:
                    pass
            page.on("framenavigated", _bump)
            page._cua_hooked = True  # type: ignore[attr-defined]
        # Settled (no main-frame nav since last wait) AND no action pending ->
        # this is a redundant read by a getter; skip the (re-)wait.
        if (not page._cua_action_pending) and page._cua_ready_epoch == page._cua_nav_epoch:
            return
        page._cua_action_pending = False  # type: ignore[attr-defined]
        await self.on_new_page(page)
        page._cua_ready_epoch = page._cua_nav_epoch  # type: ignore[attr-defined]

    @handle_target_closed()
    async def get_screenshot(
        self,
        page: Page,
        path: str | None = None,
        cursor: bool = True,
    ) -> bytes:
        """
        Capture a screenshot of the current page.

        Args:
            page (Page): The Playwright page object.
            path (str, optional): The file path to save the screenshot. If None, the screenshot will be returned as bytes. Default: None
        """
        await self._ensure_page_ready(page)
        x, y = self.last_cursor_position
        try:
            screenshot = await page.screenshot(path=path, timeout=15000)
        except Exception:
            await page.evaluate("window.stop()")
            # try again
            screenshot = await page.screenshot(path=path, timeout=15000)
        if not cursor:
            return screenshot
        try:
            screenshot = _overlay_capture_cursor(
                screenshot,
                x,
                y,
                viewport_width=self.viewport_width,
                viewport_height=self.viewport_height,
            )
            if path is not None:
                Path(path).write_bytes(screenshot)
            return screenshot
        except Exception:
            return screenshot

    @handle_target_closed()
    async def back(self, page: Page) -> None:
        await self._ensure_page_ready(page)
        await page.go_back()
        self._mark_action(page)

    @handle_target_closed()
    async def visit_page(self, page: Page, url: str) -> Tuple[bool, bool]:
        await self._ensure_page_ready(page)
        reset_prior_metadata_hash = False
        reset_last_download = False
        try:
            # Regular webpage. cua-lite patch: return at domcontentloaded (DOM
            # ready) capped 10s instead of blocking on the full load event (all
            # images/ads/trackers, default 30s). Then wait for full load only
            # best-effort (capped 3s) so fast sites still render fully; slow
            # sites no longer burn 30s. See module docstring.
            await page.goto(url, wait_until="domcontentloaded", timeout=10000)
            try:
                await page.wait_for_load_state("load", timeout=3000)
            except PlaywrightTimeoutError:
                pass
            reset_prior_metadata_hash = True
        except Exception as e_outer:
            # Downloaded file
            if self.downloads_folder and "net::ERR_ABORTED" in str(e_outer):
                async with page.expect_download() as download_info:
                    try:
                        await page.goto(url)
                    except Exception as e_inner:
                        if "net::ERR_ABORTED" in str(e_inner):
                            pass
                        else:
                            raise e_inner
                    download = await download_info.value
                    fname = os.path.join(self.downloads_folder, download.suggested_filename)
                    await download.save_as(fname)
                    message = f"<body style=\"margin: 20px;\"><h1>Successfully downloaded '{download.suggested_filename}' to local path:<br><br>{fname}</h1></body>"
                    await page.goto(
                        "data:text/html;base64," + base64.b64encode(message.encode("utf-8")).decode("utf-8")
                    )
                    reset_last_download = True
            else:
                raise e_outer
        self._mark_action(page)
        return reset_prior_metadata_hash, reset_last_download

    @handle_target_closed()
    async def page_down(self, page: Page, amount: int = 400, full_page: bool = False) -> None:
        await self._ensure_page_ready(page)
        # Move mouse to top-left to avoid scrollable elements
        await page.mouse.move(10, 10)
        self._set_last_cursor_position(10, 10)
        if full_page:
            await page.mouse.wheel(0, self.viewport_height - 50)
        else:
            await page.mouse.wheel(0, amount)

    @handle_target_closed()
    async def page_up(self, page: Page, amount: int = 400, full_page: bool = False) -> None:
        await self._ensure_page_ready(page)
        # Move mouse to top-left to avoid scrollable elements
        await page.mouse.move(10, 10)
        self._set_last_cursor_position(10, 10)
        if full_page:
            await page.mouse.wheel(0, -self.viewport_height + 50)
        else:
            await page.mouse.wheel(0, -amount)

    async def gradual_cursor_animation(
        self, page: Page, start_x: float, start_y: float, end_x: float, end_y: float
    ) -> None:
        # animation helper
        steps = 20
        for step in range(steps):
            x = start_x + (end_x - start_x) * (step / steps)
            y = start_y + (end_y - start_y) * (step / steps)
            await page.evaluate(f"""
                (function() {{
                    let cursor = document.getElementById('cua-lite-action-cursor');
                    if (cursor) {{
                        cursor.style.left = '{x}px';
                        cursor.style.top = '{y}px';
                    }}
                }})();
            """)
            await asyncio.sleep(0.05)

        self.last_cursor_position = (end_x, end_y)

    async def add_cursor_box(self, page: Page, identifier: str) -> None:
        # animation helper
        await page.evaluate(f"""
            (function() {{
                let elm = document.querySelector("[__elementId='{identifier}']");
                if (elm) {{
                    elm.style.transition = 'border 0.3s ease-in-out';
                    elm.style.border = '2px solid red';
                }}
            }})();
        """)
        await asyncio.sleep(0.3)

        # Create the shared cursor sprite used by screenshots.
        await page.evaluate(f"""
            (function() {{
                let cursor = document.getElementById('cua-lite-action-cursor');
                if (!cursor) {{
                    cursor = document.createElement('img');
                    cursor.id = 'cua-lite-action-cursor';
                    document.body.appendChild(cursor);
                }}
                cursor.src = '{_CAPTURE_CURSOR_SRC}';
                cursor.style.width = '{_CAPTURE_CURSOR_WIDTH}px';
                cursor.style.height = '{_CAPTURE_CURSOR_HEIGHT}px';
                cursor.style.position = 'absolute';
                cursor.style.zIndex = '10000';
                cursor.style.pointerEvents = 'none';
            }})();
        """)

    async def remove_cursor_box(self, page: Page, identifier: str) -> None:
        # Remove the highlight and cursor
        await page.evaluate(f"""
            (function() {{
                let elm = document.querySelector("[__elementId='{identifier}']");
                if (elm) {{
                    elm.style.border = '';
                }}
                let cursor = document.getElementById('cua-lite-action-cursor');
                if (cursor) {{
                    cursor.remove();
                }}
            }})();
        """)

    async def _safe_remove_cursor_box(self, page: Page, identifier: str) -> None:
        try:
            await self.remove_cursor_box(page, identifier)
        except Exception:
            pass

    @handle_target_closed()
    async def click_coords(self, page: Page, x: float, y: float) -> Page | None:
        new_page: Page | None = None
        await self._ensure_page_ready(page)

        # In single tab mode, remove target attributes to avoid opening new tabs
        if self.single_tab_mode:
            await page.evaluate(f"""
                (x, y) => {{
                    const element = document.elementFromPoint({x}, {y});
                    if (element) {{
                        // Remove target attribute from clicked element and all ancestors
                        let el = element;
                        while (el) {{
                            if (el.removeAttribute) {{
                                el.removeAttribute('target');
                            }}
                            el = el.parentElement;
                        }}
                    }}
                    // Remove target from all _blank links/forms
                    document.querySelectorAll('a[target=_blank], form[target=_blank]')
                        .forEach(e => e.removeAttribute('target'));
                }}
            """)

        if self.animate_actions:
            # Move cursor to the box slowly
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(page, start_x, start_y, x, y)
            await asyncio.sleep(0.1)

            if self.single_tab_mode:
                await page.mouse.click(x, y, delay=10)
            else:
                try:
                    # Give it a chance to open a new page
                    async with page.expect_event("popup", timeout=1000) as page_info:  # type: ignore
                        await page.mouse.click(x, y, delay=10)
                        new_page = await page_info.value  # type: ignore
                        assert isinstance(new_page, Page)
                        await self.on_new_page(new_page)
                except TimeoutError:
                    pass
        else:
            if self.single_tab_mode:
                await page.mouse.click(x, y, delay=10)
            else:
                try:
                    # Give it a chance to open a new page
                    async with page.expect_event("popup", timeout=1000) as page_info:  # type: ignore
                        await page.mouse.click(x, y, delay=10)
                        new_page = await page_info.value  # type: ignore
                        assert isinstance(new_page, Page)
                        await self.on_new_page(new_page)
                except TimeoutError:
                    pass
        self._set_last_cursor_position(x, y)
        self._mark_action(page)
        return new_page

    @handle_target_closed()
    async def click_id(self, page: Page, identifier: str) -> Page | None:
        """
        Returns new page if a new page is opened, otherwise None.
        """
        new_page: Page | None = None
        await self._ensure_page_ready(page)
        selector = f"[__elementId='{identifier}']"
        try:
            # Wait for the element to be visible and scroll it into view
            await page.wait_for_selector(
                selector, state="visible", timeout=self._timeout_load * 1000
            )
            target = page.locator(selector)
            await target.scroll_into_view_if_needed()
        except TimeoutError:
            raise ValueError(
                f"Element with identifier {identifier} not found or not visible"
            )

        # Retrieve bounding box to determine the center for clicking
        box = await target.bounding_box()
        if not box:
            raise ValueError(
                f"Element with identifier {identifier} is not visible on the page."
            )
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        # In single tab mode, override target attributes to avoid opening a new tab
        if self.single_tab_mode:
            await target.evaluate("""
                el => {
                    // Remove target attribute from clicked element and all _blank links/forms
                    el.removeAttribute('target');
                    document.querySelectorAll('a[target=_blank], form[target=_blank]')
                        .forEach(e => e.removeAttribute('target'));
                }
            """)

        download = None
        download_future: asyncio.Task[Download] | None = None

        # Start listening for a download event if downloads are enabled
        if self.downloads_folder:
            try:
                download_future = asyncio.create_task(
                    page.wait_for_event(  # type: ignore
                        "download", timeout=500
                    )
                )
            except Exception as e:
                print(f"Failed to set up download listener: {e}")
                download_future = None

        async def perform_click() -> Optional[Page]:
            nonlocal download
            try:
                if self.single_tab_mode:
                    await page.mouse.move(center_x, center_y, steps=1)
                    self._set_last_cursor_position(center_x, center_y)
                    await page.mouse.click(center_x, center_y)
                    return None
                else:
                    # Create a task to wait for a new page event
                    context = page.context
                    new_page_promise: asyncio.Task[Page] = asyncio.create_task(
                        context.wait_for_event(  # type: ignore
                            "page", timeout=self._timeout_load * 1000
                        )
                    )

                    # Perform the click
                    await page.mouse.move(center_x, center_y, steps=1)
                    self._set_last_cursor_position(center_x, center_y)
                    await page.mouse.click(center_x, center_y, delay=10)

                    try:
                        # Wait for the new page to open
                        new_page = await new_page_promise
                        await self.on_new_page(new_page)
                        return new_page
                    except TimeoutError:
                        # No new page opened within timeout
                        return None
            except Exception as e:
                raise e

        cursor_box_added = False
        try:
            # Optionally animate the click
            if self.animate_actions:
                cursor_box_added = True
                await self.add_cursor_box(page, identifier)
                start_x, start_y = self.last_cursor_position
                await self.gradual_cursor_animation(
                    page, start_x, start_y, center_x, center_y
                )

            new_page = await perform_click()

            # Handle any download that occurred
            if download_future:
                try:
                    if not download:
                        # Use asyncio.wait_for with a reasonable timeout
                        try:
                            download = await asyncio.wait_for(
                                download_future, timeout=self._timeout_load * 1000
                            )
                        except asyncio.TimeoutError:
                            # No download occurred within the timeout period
                            pass

                    if download:
                        print(
                            f"Downloading {download.suggested_filename} to {self.downloads_folder}"
                        )
                        assert self.downloads_folder is not None
                        fname = os.path.join(
                            self.downloads_folder, download.suggested_filename
                        )
                        await download.save_as(fname)
                except Exception as e:
                    pass
                finally:
                    if not download_future.done():
                        download_future.cancel()

            if new_page:
                await new_page.wait_for_load_state()
                if self._sleep_after_action > 0:
                    await new_page.wait_for_timeout(self._sleep_after_action * 1000)
            else:
                await page.wait_for_load_state()
                if self._sleep_after_action > 0:
                    await page.wait_for_timeout(self._sleep_after_action * 1000)
        finally:
            if cursor_box_added:
                await self._safe_remove_cursor_box(page, identifier)

        return new_page

    @handle_target_closed()
    async def select_option(
        self, page: Page, identifier: str
    ) -> Optional[Page]:
        """
        Select an option element with the given identifier.
        """
        await self._ensure_page_ready(page)
        # State-changing action with several return paths (delegates to click_id, or
        # programmatic select). Mark up-front so the next getter re-waits regardless of
        # which path returns; the click_id path re-marks at its own end (see _ensure_page_ready).
        self._mark_action(page)
        new_page: Optional[Page] = None
        try:
            # Wait for element to be present
            await page.wait_for_selector(
                f"[__elementId='{identifier}']", state="attached"
            )

            try:
                # First try normal click if element is visible
                target = page.locator(f"[__elementId='{identifier}']").first
                # Get the bounding box to check element size
                box = await target.bounding_box()

                if box and box["width"] > 0 and box["height"] > 0:
                    # Element has visible size - use normal click
                    return await self.click_id(page, identifier)

            except PlaywrightError as e:
                if "strict mode violation" in str(e):
                    # If multiple elements found, try clicking the first visible one
                    elements = await page.locator(
                        f"[__elementId='{identifier}']"
                    ).all()
                    for element in elements:
                        try:
                            if await element.is_visible():
                                await element.click()
                                return new_page
                        except PlaywrightError:
                            continue

            # If click didn't work, try programmatic selection
            # First check if it's a standard <option> element
            option_element = await page.evaluate(
            """
                (identifier) => {
                    const elements = document.querySelectorAll(`[__elementId='${identifier}']`);
                    for (const el of elements) {
                        if (el.tagName.toLowerCase() === 'option') {
                            return true;
                        }
                    }
                    return false;
                }
                """,
                identifier,
            )

            if option_element:
                # Handle standard <select> dropdown
                await page.evaluate(
                    """
                    (identifier) => {
                        const option = Array.from(document.querySelectorAll(`[__elementId='${identifier}']`))
                            .find(el => el.tagName.toLowerCase() === 'option');
                        if (!option) throw new Error('Option not found');
                        const select = option.closest('select');
                        if (select) {
                            option.selected = true;
                            select.dispatchEvent(new Event('change', { bubbles: true }));
                            select.blur();
                        }
                    }
                    """,
                    identifier,
                )
            else:
                # Handle custom dropdown/combobox options
                await page.evaluate(
                    """
                    (identifier) => {
                        const element = document.querySelector(`[__elementId='${identifier}']`);
                        if (!element) throw new Error('Element not found');

                        // Dispatch multiple events to ensure the selection is registered
                        const events = ['mousedown', 'mouseup', 'click', 'change'];
                        events.forEach(eventType => {
                            element.dispatchEvent(new Event(eventType, { bubbles: true }));
                        });

                        // If element has aria-selected, set it
                        if (element.hasAttribute('aria-selected')) {
                            element.setAttribute('aria-selected', 'true');
                        }

                        // If element has a data-value, try to set it on the parent
                        const value = element.getAttribute('data-value');
                        if (value) {
                            const parent = element.closest('[role="listbox"], [role="combobox"]');
                            if (parent) {
                                parent.setAttribute('data-value', value);
                            }
                        }
                    }
                    """,
                    identifier,
                )

            # Optional sleep/pause after the action
            if self._sleep_after_action > 0:
                await page.wait_for_timeout(self._sleep_after_action * 1000)

        except PlaywrightTimeoutError:
            raise ValueError(
                f"No option found with identifier '{identifier}' within "
                f"{self._timeout_load} seconds."
            ) from None
        return new_page

    @handle_target_closed()
    async def hover_id(self, page: Page, identifier: str) -> None:
        """
        Hovers the mouse over the target with the given id.
        """
        await self._ensure_page_ready(page)
        target = page.locator(f"[__elementId='{identifier}']")

        # See if it exists
        try:
            await target.wait_for(timeout=5000)
        except TimeoutError:
            raise RuntimeError(f"Tool use response is invalid: no such element to hover: {identifier}")

        # Hover over it
        await target.scroll_into_view_if_needed()
        await asyncio.sleep(0.3)

        box = cast(Dict[str, Union[int, float]], await target.bounding_box())

        if self.animate_actions:
            try:
                await self.add_cursor_box(page, identifier)
                # Move cursor to the box slowly
                start_x, start_y = self.last_cursor_position
                end_x, end_y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                await self.gradual_cursor_animation(page, start_x, start_y, end_x, end_y)
                await asyncio.sleep(0.1)
                await page.mouse.move(end_x, end_y)
                self._set_last_cursor_position(end_x, end_y)
            finally:
                await self._safe_remove_cursor_box(page, identifier)
        else:
            end_x, end_y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            await page.mouse.move(end_x, end_y)
            self._set_last_cursor_position(end_x, end_y)

    @handle_target_closed()
    async def hover_coords(self, page: Page, x: float, y: float) -> None:
        """
        Hovers the mouse at the specified coordinates without clicking.

        Args:
            page: The page to interact with
            x: X coordinate
            y: Y coordinate
        """
        await self._ensure_page_ready(page)

        if self.animate_actions:
            # Move cursor to the coordinates slowly
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(page, start_x, start_y, x, y)
            await asyncio.sleep(0.1)

        await page.mouse.move(x, y)
        self._set_last_cursor_position(x, y)

    @handle_target_closed()
    async def hover_and_scroll_coords(self, page: Page, x: float, y: float, direction: str = "down", amount: int = 1) -> None:
        """
        Hovers the mouse at the specified coordinates and then scrolls the element at that position.

        This method implements best practices for coordinate-based scrolling:
        1. Finds the closest scrollable element (e.g., dropdown menu) at the coordinates
        2. Verifies the element is actually scrollable (checks overflow CSS and pointer-events)
        3. Scrolls by 80% of element height for smooth incremental scrolling with content overlap
        4. Handles nested scrollable containers by walking up the DOM tree

        Based on research from Playwright, Selenium, and Puppeteer best practices for
        dropdown menus and virtual scrolling containers.

        Args:
            page: The page to interact with
            x: X coordinate to hover over
            y: Y coordinate to hover over
            direction: Scroll direction ('up' or 'down')
        """
        await self._ensure_page_ready(page)

        # First, hover at the coordinates
        if self.animate_actions:
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(page, start_x, start_y, x, y)
            await asyncio.sleep(0.1)

        await page.mouse.move(x, y)
        self._set_last_cursor_position(x, y)

        # Find the closest scrollable element and scroll it directly using JavaScript.
        # Magnitude honors the agent's `amount` (wheel clicks): 100 px per click,
        # matching the page_down/page_up branch (amount=1 → the prior one-notch step).
        clicks = max(1, int(amount))
        scroll_delta = (-100 if direction.lower() == "up" else 100) * clicks

        scroll_result = await page.evaluate(f"""
            (function() {{
                // Identify element at coordinates
                let element = document.elementFromPoint({x}, {y});
                if (!element) {{
                    return {{ success: false }};
                }}

                // Walk up DOM tree to find scrollable parent
                let scrollable = element;
                while (scrollable && scrollable !== document.documentElement) {{
                    const style = getComputedStyle(scrollable);
                    const overflowY = style.overflowY;
                    const hasOverflow = scrollable.scrollHeight > scrollable.clientHeight;
                    const canScroll = overflowY === 'scroll' || overflowY === 'auto';
                    const canInteract = style.pointerEvents !== 'none';

                    if (hasOverflow && canScroll && canInteract) {{
                        break; // Found scrollable parent
                    }}

                    scrollable = scrollable.parentElement;
                }}

                // If we found a scrollable element (not the document), scroll it directly
                if (scrollable && scrollable !== document.documentElement) {{
                    // Calculate scroll amount: 80% of visible height per click, scaled
                    // by the agent's click count so larger `amount` scrolls farther.
                    const clientHeight = scrollable.clientHeight;
                    const scrollAmount = Math.floor(clientHeight * 0.8) * {clicks};
                    const oldScrollTop = scrollable.scrollTop;
                    scrollable.scrollTop += {scroll_delta} > 0 ? scrollAmount : -scrollAmount;
                    return {{ success: true, scrolled: scrollable.scrollTop !== oldScrollTop }};
                }} else {{
                    // No scrollable container found, scroll the element itself by a small amount
                    // This prevents the entire page from scrolling
                    if (element && element !== document.documentElement && element !== document.body) {{
                        const oldScrollTop = element.scrollTop;
                        element.scrollTop += {scroll_delta};
                        return {{ success: true, scrolled: element.scrollTop !== oldScrollTop }};
                    }}
                }}
                return {{ success: false }};
            }})();
        """)

        # If JavaScript scrolling didn't work, try dispatching wheel event to the element
        # This handles cases like custom dropdowns that respond to wheel events
        if not scroll_result.get('scrolled', False):
            delta_y = (-100 if direction.lower() == "up" else 100) * clicks
            # Dispatch wheel event at the specific coordinates
            await page.mouse.wheel(0, delta_y)

        # Brief wait for content stabilization (especially important for virtual scrolling)
        await asyncio.sleep(0.2)

    @handle_target_closed()
    async def fill_coords(
            self, page: Page, x: float, y: float, value: str, press_enter: bool = True, delete_existing_text: bool = False
        ) -> Page | None:
        await self._ensure_page_ready(page)
        new_page: Page | None = None

        if self.animate_actions:
            # Move cursor to the box slowly
            start_x, start_y = self.last_cursor_position
            await self.gradual_cursor_animation(page, start_x, start_y, x, y)
            await asyncio.sleep(0.1)

        await page.mouse.click(x, y)
        self._set_last_cursor_position(x, y)

        if delete_existing_text:
            await page.keyboard.press("ControlOrMeta+A")
            await page.keyboard.press("Backspace")

        # fill char by char to mimic human speed for short text and type fast for long text
        if len(value) < 100:
            delay_typing_speed = 50 + 100 * random.random()
        else:
            delay_typing_speed = 10

        if self.animate_actions:
            # Give it a chance to open a new page
            try:
                async with page.expect_event("popup", timeout=1000) as page_info:  # type: ignore
                    await page.keyboard.type(value, delay=delay_typing_speed)
                    if press_enter:
                        await page.keyboard.press("Enter")
                    new_page = await page_info.value  # type: ignore
                    assert isinstance(new_page, Page)
                    await self.on_new_page(new_page)
            except TimeoutError:
                pass
        else:
            try:
                # Give it a chance to open a new page
                async with page.expect_event("popup", timeout=1000) as page_info:  # type: ignore
                    try:
                        await page.keyboard.type(value)
                    except PlaywrightError:
                        await page.keyboard.type(value, delay=delay_typing_speed)
                    if press_enter:
                        await page.keyboard.press("Enter")
                    new_page = await page_info.value  # type: ignore
                    assert isinstance(new_page, Page)
                    await self.on_new_page(new_page)
            except TimeoutError:
                pass

        self._mark_action(page)
        return new_page

    @handle_target_closed()
    async def fill_id(
        self, page: Page, identifier: str, value: str, press_enter: bool = True, delete_existing_text: bool = False
    ) -> Page | None:
        """
        Fill the element with the given identifier with the specified value.
        """
        await self._ensure_page_ready(page)
        await page.wait_for_selector(f"[__elementId='{identifier}']", state="visible")
        target = page.locator(f"[__elementId='{identifier}']")
        await target.scroll_into_view_if_needed()

        # See if it exists
        try:
            await target.wait_for(timeout=5000)
        except TimeoutError:
            raise RuntimeError(f"Tool use response is invalid: No such element to fill input_text into: {identifier}") from None

        # Fill it
        box = cast(Dict[str, Union[int, float]], await target.bounding_box())

        if self.single_tab_mode:
            # Remove target attributes to prevent new tabs
            await target.evaluate("""
                el => el.removeAttribute('target')
                // Remove 'target' on all <a> tags
                for (const a of document.querySelectorAll('a[target=_blank]')) {
                    a.removeAttribute('target');
                }
                // Remove 'target' on all <form> tags
                for (const frm of document.querySelectorAll('form[target=_blank]')) {
                    frm.removeAttribute('target');
                }
            """)

        page = await self.fill_coords(
            page,
            float(box["x"] + box["width"] / 2),
            float(box["y"] + box["height"] / 2),
            value,
            press_enter,
            delete_existing_text
        )
        return page

    @handle_target_closed()
    async def scroll_id(self, page: Page, identifier: str, direction: str) -> None:
        await self._ensure_page_ready(page)
        await page.evaluate(
            f"""
        (function() {{
            let elm = document.querySelector("[__elementId='{identifier}']");
            if (elm) {{
                if ("{direction}" == "up") {{
                    elm.scrollTop = Math.max(0, elm.scrollTop - elm.clientHeight);
                }} else if ("{direction}" == "down") {{
                    elm.scrollTop = Math.min(elm.scrollHeight - elm.clientHeight, elm.scrollTop + elm.clientHeight);
                }} else if ("{direction}" == "left") {{
                    elm.scrollLeft = Math.max(0, elm.scrollLeft - elm.clientWidth);
                }} else if ("{direction}" == "right") {{
                    elm.scrollLeft = Math.min(elm.scrollWidth - elm.clientWidth, elm.scrollLeft + elm.clientWidth);
                }}
            }}
        }})();
    """
        )

    async def keypress(self, page: Page, keys: list[str]) -> None:
        """
        Press specified keys in sequence.
        """
        await self._ensure_page_ready(page)
        mapped_keys = list(keys)  # host already resolved to final Playwright names
        try:
            if self.animate_actions:
                for key in mapped_keys:
                    await page.keyboard.down(key)
                    await asyncio.sleep(0.05)  # Small delay between key presses
                for key in reversed(mapped_keys):
                    await page.keyboard.up(key)
                    await asyncio.sleep(0.05)  # Small delay between key releases
            else:
                for key in mapped_keys:
                    await page.keyboard.down(key)
                for key in reversed(mapped_keys):
                    await page.keyboard.up(key)
        except Exception as e:
            raise RuntimeError(f"I tried to keypress(keys={keys}), but I got an error: {e}") from None
        self._mark_action(page)

    @handle_target_closed()
    async def get_webpage_text(self, page: Page, n_lines: int = 100) -> str:
        """
        page: playwright page object
        n_lines: number of lines to return from the page innertext
        return: text in the first n_lines of the page
        """
        await self._ensure_page_ready(page)
        try:
            text_in_viewport = await page.evaluate("""() => {
                return document.body.innerText;
            }""")
            text_in_viewport = "\n".join(text_in_viewport.split("\n")[:n_lines])
            # remove empty lines
            text_in_viewport = "\n".join([line for line in text_in_viewport.split("\n") if line.strip()])
            assert isinstance(text_in_viewport, str)
            return text_in_viewport
        except Exception:
            return ""

    async def get_page_markdown(self, page: Page) -> str:
        # TODO: replace with mdconvert
        await self._ensure_page_ready(page)
        return await self.get_webpage_text(page, n_lines=1000)

    # type TAB then hit ENTER
    async def tab_and_enter(self, page: Page) -> None:
        await self._ensure_page_ready(page)
        await page.keyboard.press("Tab")
        await page.keyboard.press("Enter")
        # Enter can submit a form / navigate; mark so the next getter re-waits for the
        # ready state instead of short-circuiting on a stale frame (see _ensure_page_ready).
        self._mark_action(page)
