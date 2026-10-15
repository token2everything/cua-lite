"""VENDORED + PATCHED OmniBoxes Playwright instance (cua-lite).

Base: omniboxes/node/instances/playwright_instance.py @ pinned commit 574a205e
(see docker/Dockerfile). The Dockerfile clones upstream @ that SHA and ``cp``s
this file over ``/opt/omniboxes/omniboxes/node/instances/playwright_instance.py``
before ``pip install`` — so there is no dependency on any cua-lite fork and the
delta is reviewable in one place.

PATCH — launch hardening + resilience (fixes pool bring-up hangs under load):
  Upstream ``_create`` does a bare ``await self.p.chromium.launch()`` with no
  timeout. Under host browser-contention (hundreds of Chromium procs on a
  shared host) the DevTools handshake can STALL FOREVER — the instance_server
  never reports ready (HTTP 000), and since deploy.py waits on every instance,
  one stuck Chromium wedges the WHOLE pool bring-up (observed: M=128/256 pools
  warming only 1-3 instances before the deploy timeout). This mirrors
  mobilegym's launch discipline (lite/gym/envs/mobilegym/main.py:
  ``asyncio.wait_for(..., _LAUNCH_TIMEOUT_S)`` + stagger) which never hangs:
    * cap each ``launch()`` with ``asyncio.wait_for`` and retry 3× so a stuck
      Chromium is abandoned (not fatal) and the slot tries again,
    * hardening flags (``--no-sandbox`` etc.) keep browsers stable under load
      (fewer screenshot/exec stalls),
    * cap ``new_context`` too (it can stall the same way).
  A slot that can't launch after 3 tries raises (its lease is reclaimed by the
  node_server lease-TTL reaper) instead of silently hanging the pool.

PATCH — structured "this instance is dead" flag on the /execute error body:
  A closed browser/page/context is a PERMANENT failure for this instance: the
  host must fail fast and end the trajectory instead of grinding every remaining
  step (and LLM turn) against the corpse. Upstream flattens the typed playwright
  ``TargetClosedError`` into ``HTTPException(500, detail=str(e))``, so the only
  thing reaching the host is prose — which left the host substring-matching
  "has been closed" (webgym ``main.py::_is_instance_dead``). This carries the
  fact STRUCTURALLY instead. The 500 body becomes::

      {"detail": {"message": "Error executing command: ...",
                  "instance_dead": true}}

  and the flag survives both remaining hops: ``node/server.py::/execute``
  forwards the body verbatim (``JSONResponse(content=response.json())``) and
  ``master/server.py::_execute_error_body`` lifts ``instance_dead`` to the top
  level of the master's error body. So the host reads a boolean field and never
  pattern-matches an error string.
"""
import asyncio
import os
from typing import Any, Dict
from io import BytesIO

from playwright.async_api import async_playwright
from playwright._impl._errors import TargetClosedError
from fastapi import HTTPException

from omniboxes.node.instances.base import InstanceBase, Status
from omniboxes.node.instances._set_of_marks import add_set_of_mark
from omniboxes.node.instances._playwright_controller import PlaywrightController


# Viewport is POOL-LEVEL: the cua-lite host forwards the configured browser
# viewport to the container once at boot via ``-e WEBGYM_VIEWPORT=WxH`` (see
# WebGymContainerServices.ensure). Parse it here; fall back to the historical
# 1280x720 default if the var is unset or malformed.
def _viewport_from_env() -> tuple[int, int]:
    raw = os.environ.get("WEBGYM_VIEWPORT", "")
    try:
        w, h = raw.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return 1280, 720


_VIEWPORT_W, _VIEWPORT_H = _viewport_from_env()

# cua-lite PATCH: the one wire name for "this instance's browser is gone".
# ``master_server.py`` repeats the literal (it must not import this module — that
# would drag playwright into the master process); both spellings are cross-
# referenced in each file's docstring, and the host reads THIS key.
INSTANCE_DEAD_KEY = "instance_dead"


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


class PlaywrightInstance(InstanceBase):
    def __init__(self, instance_num=0, logger=None):
        super().__init__(instance_num=instance_num, logger=logger)
        self.page = None
        self.context = None
        self.browser = None
        self.p = None
        self._last_screenshot_info = None
        self.controller = PlaywrightController(
            viewport_width=_VIEWPORT_W,
            viewport_height=_VIEWPORT_H,
            single_tab_mode=True,
            animate_actions=False
        )

    # Launch hardening + resilience (mirrors lite/gym/envs/mobilegym/main.py).
    # A bare ``chromium.launch()`` with no timeout HANGS FOREVER under host
    # browser-contention (the DevTools handshake stalls) — wedging this
    # instance_server (HTTP 000) and, since deploy.py waits on it, the whole
    # pool bring-up. Cap each launch + retry so a stuck Chromium is abandoned,
    # not fatal. Hardening flags keep browsers stable under load (fewer
    # screenshot/exec stalls). cua-lite patch over vendored OmniBoxes.
    _LAUNCH_ARGS = [
        "--no-sandbox", "--disable-setuid-sandbox",
        "--disable-dev-shm-usage", "--disable-gpu",
    ]
    _LAUNCH_TIMEOUT_S = 60.0

    async def _create(self) -> None:
        self.p = await async_playwright().start()
        last_exc = None
        for _ in range(3):
            try:
                self.browser = await asyncio.wait_for(
                    self.p.chromium.launch(args=self._LAUNCH_ARGS),
                    timeout=self._LAUNCH_TIMEOUT_S,
                )
                break
            except Exception as e:  # incl. asyncio.TimeoutError on a stuck launch
                last_exc = e
                if self.browser is not None:
                    try:
                        await self.browser.close()
                    except Exception:
                        pass
                    self.browser = None
        if self.browser is None:
            raise RuntimeError(
                f"chromium.launch hung/failed after 3 attempts: {last_exc}")
        try:
            self.context = await asyncio.wait_for(
                self.browser.new_context(viewport={'width': _VIEWPORT_W, 'height': _VIEWPORT_H}),
                timeout=30.0,
            )
        except Exception:
            # new_context stalled on a launched-but-wedged browser: close it so
            # we don't leak a Chromium proc/fds when _create raises.
            try:
                await self.browser.close()
            except Exception:
                pass
            self.browser = None
            raise
        self.page = await self.context.new_page()
        await self.controller.on_new_page(self.page)

    async def _delete(self) -> None:
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.p:
            await self.p.stop()

    def _translate_displayed_id_to_original(self, displayed_id: str) -> str:
        """
        Translate a displayed ID (what user sees on screenshot) to the original element ID.
        """
        if self._last_screenshot_info and 'id_mapping' in self._last_screenshot_info:
            id_mapping = self._last_screenshot_info['id_mapping']
            return id_mapping.get(displayed_id, displayed_id)
        return displayed_id

    async def _screenshot(
        self,
        interaction_mode: str = "set_of_marks",
        cursor: bool = True,
    ) -> BytesIO:
        """Take a screenshot with optional set-of-marks annotations based on interaction mode."""
        # Determine whether to apply annotations based on interaction_mode
        apply_annotations = interaction_mode == "set_of_marks"
        use_sequential_ids = True  # Default behavior

        # Get the base screenshot
        screenshot_bytes = await self.controller.get_screenshot(self.page, cursor=cursor)

        # If annotations are disabled (coordinates mode), return plain screenshot
        if not apply_annotations:
            return BytesIO(screenshot_bytes)

        # Get interactive regions using the controller
        interactive_regions = await self.controller.get_interactive_rects(self.page)

        if not interactive_regions:
            # If no interactive regions found, return plain screenshot
            return BytesIO(screenshot_bytes)

        try:
            # Apply set-of-marks annotation and capture the ID mapping
            annotated_image, visible_rects, rects_above, rects_below, id_mapping = add_set_of_mark(
                screenshot_bytes,
                interactive_regions,
                use_sequential_ids=use_sequential_ids
            )

            # Convert PIL Image back to bytes
            output_buffer = BytesIO()
            annotated_image.save(output_buffer, format='PNG')
            output_buffer.seek(0)

            # Store the mapping information for potential use
            self._last_screenshot_info = {
                'visible_rects': visible_rects,
                'rects_above': rects_above,
                'rects_below': rects_below,
                'interactive_regions': interactive_regions,
                'id_mapping': id_mapping  # Store the ID mapping for command translation
            }

            return output_buffer

        except Exception as e:
            # If annotation fails, return plain screenshot
            self.logger.error(f"Error applying set-of-marks: {str(e)}")
            return BytesIO(screenshot_bytes)

    async def _probe(self) -> bool:
        return self.status >= Status.STARTED

    # cua-lite PATCH: typed liveness of this instance's browser/page/context —
    # public playwright state only, no message parsing. Measured in-image
    # (playwright 1.62): after ``browser.close()``, ``context.close()`` OR
    # ``page.close()`` the page reports ``is_closed() is True``, and a closed
    # browser additionally reports ``is_connected() is False`` — so this
    # disjunction covers every way the target can go away, including the case
    # where the controller's ``handle_target_closed`` recovery re-wraps the
    # original TargetClosedError into a plain ``Exception``.
    def _browser_is_dead(self) -> bool:
        if self.page is None or self.page.is_closed():
            return True
        return self.browser is None or not self.browser.is_connected()

    async def _metadata(self) -> Dict[str, Any]:
        return {
            "width": _VIEWPORT_W,
            "height": _VIEWPORT_H
        }

    async def _execute(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Get the first (and should be only) command from the dict
            if len(command_data) != 1:
                raise ValueError("Only one command per request is supported")

            command_type, args = next(iter(command_data.items()))

            if command_type == "visit_page":
                # Format: {"url": "https://example.com"}
                await self.controller.visit_page(self.page, args["url"])

            elif command_type == "back":
                await self.controller.back(self.page)

            elif command_type == "click_coords":
                # Format: {"x": x, "y": y}
                x = float(args["x"])
                y = float(args["y"])
                await self.controller.click_coords(self.page, x, y)

            elif command_type == "click_id":
                # Format: {"id": "123"}
                # Translate displayed ID to original element ID
                original_id = self._translate_displayed_id_to_original(args["id"])
                await self.controller.click_id(self.page, original_id)

            elif command_type == "fill_coords":
                # Format: {"x": x, "y": y, "value": "text", "press_enter": true, "delete_existing": false}
                x = float(args["x"])
                y = float(args["y"])
                value = args["value"]
                press_enter = args.get("press_enter", True)
                delete_existing = args.get("delete_existing", False)

                await self.controller.fill_coords(self.page, x, y, value, press_enter, delete_existing)

            elif command_type == "fill_id":
                # Format: {"id": "123", "value": "text", "press_enter": true, "delete_existing": false}
                displayed_id = args["id"]
                value = args["value"]
                press_enter = args.get("press_enter", True)
                delete_existing = args.get("delete_existing", False)

                # Translate displayed ID to original element ID
                original_id = self._translate_displayed_id_to_original(displayed_id)
                await self.controller.fill_id(self.page, original_id, value, press_enter, delete_existing)

            elif command_type == "select_option":
                # Format: {"id": "123"}
                # Translate displayed ID to original element ID
                original_id = self._translate_displayed_id_to_original(args["id"])
                await self.controller.select_option(self.page, original_id)

            elif command_type == "hover_id":
                # Format: {"id": "123"}
                original_id = self._translate_displayed_id_to_original(args["id"])
                await self.controller.hover_id(self.page, original_id)

            elif command_type == "keypress":
                # Format: {"keys": ["ctrl", "a"]} or {"keys": ["Enter"]}
                keys = args["keys"]
                await self.controller.keypress(self.page, keys)

            elif command_type == "page_down":
                # Format: {"amount": 200, "full_page": false}
                amount = args.get("amount", 200)
                full_page = args.get("full_page", False)
                await self.controller.page_down(self.page, amount, full_page)

            elif command_type == "page_up":
                # Format: {"amount": 200, "full_page": false}
                amount = args.get("amount", 200)
                full_page = args.get("full_page", False)
                await self.controller.page_up(self.page, amount, full_page)

            elif command_type == "scroll_id":
                # Format: {"id": "123", "direction": "down"}
                displayed_id = args["id"]
                direction = args["direction"].lower()

                # Translate displayed ID to original element ID
                original_id = self._translate_displayed_id_to_original(displayed_id)
                await self.controller.scroll_id(self.page, original_id, direction)

            elif command_type == "hover_coords":
                # Format: {"x": 100, "y": 200}
                x = float(args["x"])
                y = float(args["y"])
                await self.controller.hover_coords(self.page, x, y)

            elif command_type == "hover_and_scroll_coords":
                # Format: {"x": 100, "y": 200, "direction": "down", "amount": 3}
                x = float(args["x"])
                y = float(args["y"])
                direction = args.get("direction", "down").lower()
                amount = int(args.get("amount", 1))
                await self.controller.hover_and_scroll_coords(self.page, x, y, direction, amount)

            elif command_type == "sleep":
                # Format: {"duration": 2.0}
                duration = float(args["duration"])
                await self.controller.sleep(self.page, duration)

            elif command_type == "tab_and_enter":
                await self.controller.tab_and_enter(self.page)

            elif command_type == "get_webpage_text":
                # Format: {"n_lines": 100}
                n_lines = args.get("n_lines", 100)
                text = await self.controller.get_webpage_text(self.page, n_lines)
                return {"text": text}

            elif command_type == "get_page_metadata":
                # Get page metadata with guaranteed title and url fields
                metadata = await self.controller.get_page_metadata(self.page)
                return metadata

            elif command_type == "get_interactive_rects":
                # Get current interactive regions using the controller
                rects = await self.controller.get_interactive_rects(self.page)
                return rects

            elif command_type == "get_screenshot_info":
                # Return information about the last screenshot's interactive elements
                if self._last_screenshot_info:
                    return self._last_screenshot_info
                else:
                    return {"error": "No screenshot info available"}

            elif command_type == "screenshot":
                # Format: {"interaction_mode": "set_of_marks", "cursor": true}
                interaction_mode = args.get("interaction_mode", "set_of_marks")
                cursor = _coerce_cursor_bool(args.get("cursor"), default=True)
                screenshot_buffer = await self._screenshot(
                    interaction_mode=interaction_mode,
                    cursor=cursor,
                )
                return {"status": "screenshot_taken"}

            else:
                raise ValueError(f"Unsupported command: {command_type}")

            return {"status": "success"}

        except Exception as e:
            # cua-lite PATCH: the error body is a STRUCTURED object, not prose.
            # ``instance_dead`` is the typed fact the host needs (see the module
            # docstring); ``message`` keeps the historical text for logs.
            raise HTTPException(
                status_code=500,
                detail={
                    "message": f"Error executing command: {str(e)}",
                    INSTANCE_DEAD_KEY: (
                        isinstance(e, TargetClosedError) or self._browser_is_dead()
                    ),
                },
            )
