"""Execute generated tasks on an isolated 1920x1080 desktop.

The coordinator supplies a fresh site baseline. This owner creates only its
Xvfb, Playwright browser, and temporary persistent profile. Deploy the frozen
``scripts/webarena_verified_qwen.py`` helper, its pinned OSWorld cursor checkout,
and the desktop runtime (Xvfb/openbox/scrot/xauth/xclip plus PyAutoGUI). Helpers
provide process ownership, cursor capture, and literal GUI-call execution;
generated task setup and scoring remain here. No official task is instantiated.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from time import monotonic, sleep
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from contract import sha256_file, sha256_json
from cross_site_evaluator import capture_gui_frame
from generated_env import (
    audit_final_state,
    build_public_instruction,
    resolve_public_task,
    score_reference,
)
from PIL import Image

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER
from lite.core.tools.calls import runtime_rejected_reason
from lite.core.tools.results import make_tool_result
from lite.gym.base import LiteBaseEnv
from lite.gym.envs.osworld.actions import to_pyautogui
from lite.gym.types import EXECUTED_ACTIONS_INFO_KEY, LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.backend.model_inputs import (
    coerce_model_duration,
    coerce_model_numeric,
    project_model_keys,
)
from lite.gym.utils.feedback.errors import MODEL_ACTION_ERROR_TYPES
from lite.gym.utils.feedback.ingress import prepare_env_tool_calls
from lite.gym.utils.feedback.results import ordered_tool_call_ids

SCREEN_SIZE = (1920, 1080)
GUI_ACTIONS = [
    name for name in LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER if name != "cursor_position"
]


def load_desktop_helpers(path: Path, expected_sha256: str) -> ModuleType:
    """Load the explicitly deployed, hash-bound desktop helper source."""
    if sha256_file(path) != expected_sha256:
        raise ValueError("desktop helper source differs from the frozen SHA256")
    spec = importlib.util.spec_from_file_location("wavgen_desktop_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load desktop helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeneratedDesktopEnv(LiteBaseEnv):
    """Own generated setup, canonical GUI execution, and private terminal scoring.

    Every model screenshot contains the full desktop. Canonical normalized
    coordinates are projected through the existing OSWorld action owner.
    Persistent profiles are per case; no incognito profile-menu workaround or
    model-action filtering is needed. Sync GUI/browser work stays on one thread.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        candidate: dict[str, Any],
        site_urls: dict[str, str],
        output: Path,
        helpers: ModuleType,
    ) -> None:
        self.args, self.candidate, self.site_urls = args, candidate, site_urls
        self.output, self.helpers = output, helpers
        self.public = resolve_public_task(candidate, site_urls)
        self.instruction = build_public_instruction(self.public)
        self._origins = {(urlsplit(url).scheme, urlsplit(url).netloc) for url in site_urls.values()}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wavgen-desktop")
        self._xvfb = self._playwright = self._context = self._profile = None
        self._browser = self._browser_process = None
        self._pyautogui = self._cursor_class = None
        self._opened = self._closed = self._finished = self._artifacts_closed = False
        self._tracing_started = False
        self._episode_recording_context = None
        self._episode_context_id: str | None = None
        self._episode_started_at: str | None = None
        self._episode_audit_context_ids: list[str] = []
        self._step_count = 0
        self._image = None
        self.final_content: str | None = None
        self.truncated = False
        self.evaluation: dict[str, Any] | None = None

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(
            dims=("desktop", "use"),
            valid_actions=GUI_ACTIONS,
            extra_tool_schemas=self.extra_tool_schemas(["response"]),
            others={
                "generated_task_id": self.public["generated_task_id"],
                "sites": self.public["sites"],
                "observation": "full_desktop",
                "screen_size": list(SCREEN_SIZE),
            },
        )

    async def reset(self) -> LiteEnvObservation:
        """Open authenticated public start URLs on the supplied fixed baseline."""
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._reset_sync)

    def _restrict_origin(self, route: Any) -> None:
        requested = urlsplit(route.request.url)
        if (requested.scheme, requested.netloc) in self._origins:
            route.continue_()
        else:
            route.abort()

    def _reset_sync(self) -> LiteEnvObservation:
        from playwright.sync_api import sync_playwright

        from lite.gym.envs.browsergym.webarena_verified import install_playwright_trace_loader

        if self._opened or self._closed:
            raise RuntimeError("A fresh desktop environment is required for every execution")
        self._xvfb = self.helpers.xvfb(self.args.display)
        self._xvfb.__enter__()
        import pyautogui

        self._pyautogui = pyautogui
        self._pyautogui.FAILSAFE = False
        self._pyautogui.PAUSE = 0.1
        if tuple(self._pyautogui.size()) != SCREEN_SIZE:
            raise RuntimeError("The owned desktop must be exactly 1920x1080")
        self._cursor_class = self.helpers.load_cursor_class(self.args.osworld_root)
        self._profile = tempfile.TemporaryDirectory(prefix="wavgen-desktop-profile-")
        self._playwright = sync_playwright().start()
        launch_options = {
            "executable_path": os.environ.get(
                "PLAYWRIGHT_CHROMIUM_EXECUTABLE", self._playwright.chromium.executable_path
            ),
            "headless": False,
            "no_viewport": True,
            "accept_downloads": False,
            "args": ["--window-size=1920,1080", "--start-maximized"],
        }
        # Restore session state before recording task events. Import local
        # storage once; an init script would overwrite it on every navigation.
        state = json.loads(self.args.storage_state.read_text()) if self.args.storage_state else {}
        if state.get("origins"):
            self._context = self._playwright.chromium.launch_persistent_context(
                self._profile.name, **launch_options
            )
            self._context.route("**/*", self._restrict_origin)
            self._context.add_cookies(state.get("cookies", []))
            for origin in state.get("origins", []):
                parsed_origin = urlsplit(origin["origin"])
                if (parsed_origin.scheme, parsed_origin.netloc) not in self._origins:
                    raise ValueError("storage state contains an undeclared site's origin")
                preparation = self._context.new_page()
                preparation.goto(origin["origin"], wait_until="domcontentloaded")
                preparation.evaluate(
                    "entries => entries.forEach(({name, value}) => "
                    "localStorage.setItem(name, value))",
                    origin["localStorage"],
                )
                preparation.close()
            self._context.close()
        # Authentication preparation has no model trace or HAR. The recording
        # context uses the same isolated profile and starts a fresh event scope.
        self._episode_started_at = datetime.now(UTC).isoformat()
        # Attach without Playwright's focus emulation: visible DOM must belong
        # to the tab the desktop actually displays, including native Ctrl+T tabs.
        port_file = Path(self._profile.name) / "DevToolsActivePort"
        port_file.unlink(missing_ok=True)
        self._browser_process = subprocess.Popen(
            [
                launch_options["executable_path"],
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                "--window-size=1920,1080",
                "--start-maximized",
                "--remote-debugging-port=0",
                "--remote-debugging-address=127.0.0.1",
                f"--user-data-dir={self._profile.name}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = monotonic() + 30
        while not port_file.exists():
            if self._browser_process.poll() is not None:
                raise RuntimeError("Owned Chromium exited before opening its debugging endpoint")
            if monotonic() >= deadline:
                raise TimeoutError("Owned Chromium did not open its debugging endpoint")
            sleep(0.05)
        port, endpoint = port_file.read_text().splitlines()
        self._browser = self._playwright.chromium.connect_over_cdp(
            f"ws://127.0.0.1:{port}{endpoint}", no_defaults=True
        )
        self._context = self._browser.contexts[0]
        session = self._browser.new_browser_cdp_session()
        session.send("Browser.setDownloadBehavior", {"behavior": "deny"})
        session.detach()
        self._context.route_from_har(
            self.output / "network.har",
            update=True,
            update_content="embed",
            update_mode="full",
        )
        self._episode_recording_context = self._context
        self._episode_context_id = str(uuid4())
        self._context.route("**/*", self._restrict_origin)
        self._context.add_cookies(state.get("cookies", []))
        install_playwright_trace_loader()
        self._context.tracing.start(screenshots=True, snapshots=True, sources=False)
        self._tracing_started = True
        page = self._context.pages[0] if self._context.pages else self._context.new_page()
        reset_recovery_used = False
        for index, url in enumerate(self.public["start_urls"]):
            if index:
                page = self._context.new_page()
            response = page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            if response is not None and response.status == 502 and not reset_recovery_used:
                reset_recovery_used = True
                recovery_path = self.output / "initial_reset_502_recovery.json"
                recovery = {
                    "policy": "one_initial_start_GET_502_recovery_before_model",
                    "public_start_url": url,
                    "recording_context_id": self._episode_context_id,
                    "original_error": "Generated start page returned HTTP 502",
                    "initial_http_status": 502,
                    "recovery_GET_attempts": 1,
                    "model_started": False,
                    "business_assets_changed": False,
                    "recovery_completed": False,
                }
                with recovery_path.open("x") as stream:
                    json.dump(recovery, stream, indent=2)
                page.wait_for_timeout(2000)
                response = page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                recovery["recovery_http_status"] = response.status if response else None
                recovery["recovery_completed"] = True
                recovery_path.write_text(json.dumps(recovery, indent=2) + "\n")
            if response is not None and response.status >= 400:
                raise RuntimeError(f"Generated start page returned HTTP {response.status}")
        page.bring_to_front()
        self._pump(self.helpers.RESET_SETTLE_SECONDS)
        page = self._active_page()
        self._opened = True
        (self.output / "desktop").mkdir(exist_ok=True)
        self._image = capture_gui_frame(
            self.output,
            self._active_page,
            self._capture,
            "desktop/step_000.png",
            self._episode_context_id,
            "model_context_only",
            Path(__file__),
        )
        (self.output / "initial.png").write_bytes(self._image)
        (self.output / "instruction.txt").write_text(self.instruction + "\n")
        (self.output / "initial_observation.json").write_text(
            json.dumps(
                {
                    "url": page.url,
                    "title": page.title(),
                    "body_text": page.locator("body").inner_text(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return LiteEnvObservation(image=self._image, text=self.instruction)

    def _active_page(self) -> Any:
        from playwright.sync_api import Error as PlaywrightError

        pages = [page for page in self._context.pages if not page.is_closed()]
        visible = []
        for page in pages:
            try:
                if page.evaluate("document.visibilityState === 'visible'"):
                    visible.append(page)
            except PlaywrightError:
                if not page.is_closed():
                    raise
        if len(visible) == 1:
            return visible[0]
        if not pages:
            raise RuntimeError("No live browser tab remains")
        # Several windows can each have a visible tab. Only genuine document
        # focus identifies the foreground one; browser chrome focus is ambiguous.
        focused = [page for page in visible if page.evaluate("document.hasFocus()")]
        return focused[0] if len(focused) == 1 else None

    def _pump(self, duration: float) -> None:
        from playwright.sync_api import Error as PlaywrightError

        deadline = monotonic() + duration
        while True:
            # Any live page pumps the browser protocol. Full-desktop capture
            # does not require its DOM or focus, and navigation may destroy
            # that document's execution context while the GUI action succeeds.
            pages = [page for page in self._context.pages if not page.is_closed()]
            if not pages:
                raise RuntimeError("No live browser tab remains")
            page = pages[0]
            try:
                page.wait_for_timeout(max(0, math.ceil((deadline - monotonic()) * 1000)))
                return
            except PlaywrightError:
                # A tab can close after selection. Reselect from the context;
                # unrelated browser errors retain their original failure.
                if not page.is_closed():
                    raise

    def _capture(self) -> bytes:
        image = self.helpers.screenshot_bytes(self._pyautogui, self._cursor_class)
        with Image.open(BytesIO(image)) as screenshot:
            if screenshot.size != SCREEN_SIZE:
                raise RuntimeError("Full-desktop capture dimensions changed")
        return image

    def _dispatch_gui(self, name: str, arguments: dict[str, Any]) -> None:
        if name in {"wait", "hold_key"}:
            duration = coerce_model_duration(arguments.get("duration", 1), action_name=name)
            keys = (
                project_model_keys(arguments.get("keys", []), action_name=name, backend="pyautogui")
                if name == "hold_key"
                else []
            )
            for key in keys:
                self._pyautogui.keyDown(key)
            try:
                self._pump(duration)
            finally:
                for key in reversed(keys):
                    self._pyautogui.keyUp(key)
            return
        if name == "type":
            # Clipboard insertion preserves Unicode and punctuation without
            # replacing any PyAutoGUI function or its keyboard mapping.
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=arguments.get("text", "").encode("utf-8"),
                check=True,
                timeout=10,
            )
            self._pyautogui.hotkey("ctrl", "v")
            if arguments.get("press_enter"):
                self._pyautogui.press("enter")
            return
        if name == "scroll":
            arguments = {
                **arguments,
                "amount": coerce_model_numeric(
                    arguments.get("amount", 3),
                    field="amount",
                    action_name=name,
                    min_value=0,
                    integer=True,
                ),
            }
        if name == "click" and arguments.get("coordinate") is None:
            command = (
                f"pyautogui.click(button={arguments.get('button', 'left')!r}, "
                f"clicks={arguments.get('clicks', 1)!r}, interval=0.1)"
            )
        else:
            command = to_pyautogui(name, arguments, *SCREEN_SIZE)
        if command is None and name != "screenshot":
            raise ValueError(f"unsupported desktop action: {name}")
        if command:
            for expression in ast.parse(command).body:
                self.helpers.execute_safe_pyautogui(ast.unparse(expression), self._pyautogui)

    async def step(self, actions: list[dict[str, Any]]) -> LiteEnvStepResult:
        """Execute one model decision and return feedback under its real call IDs."""
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._step_sync, actions
        )

    def _step_sync(self, actions: list[dict[str, Any]]) -> LiteEnvStepResult:
        if not self._opened or self._finished or self._closed:
            raise RuntimeError("reset must precede step; terminal episodes cannot continue")
        call_ids = ordered_tool_call_ids(actions)
        projected, ingress_errors = prepare_env_tool_calls(
            actions, self.metadata, validate_top_level_action=True
        )
        errors = {call_id: feedback.message for call_id, feedback in ingress_errors.items()}
        executed = []
        self._step_count += 1
        terminated = False
        # Parse and execute canonical actions; private task expectations never
        # affect model-visible tool feedback or the decision to terminate.
        for action, call_id in projected:
            if terminated:
                if call_id is not None:
                    errors[call_id] = "Not executed: an earlier call ended the episode"
                continue
            if rejection := runtime_rejected_reason(action):
                if call_id is not None:
                    errors[call_id] = rejection
                continue
            name, arguments = action["name"], action["arguments"]
            if name == "response":
                self.final_content = arguments["text"]
                executed.append({"call": name, "args": arguments})
                terminated = True
                continue
            try:
                self._dispatch_gui(name, arguments)
            except MODEL_ACTION_ERROR_TYPES as error:
                if call_id is not None:
                    errors[call_id] = str(error)
                executed.append({"call": "noop", "args": {"name": name, "reason": str(error)}})
                continue
            executed.append({"call": name, "args": arguments})
            self._pump(self.helpers.POST_ACTION_DELAY_SECONDS)
        self.truncated = not terminated and self._step_count >= self.args.max_steps
        self._finished = terminated or self.truncated
        # Record the complete transition before terminal verification or the
        # next provider turn. Scoring never supplies corrective feedback.
        with (self.output / "desktop_actions.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"step": self._step_count, "executed": executed, "errors": errors})
                + "\n"
            )
        if self._finished:
            self._finish_sync()
        else:
            self._image = capture_gui_frame(
                self.output,
                self._active_page,
                self._capture,
                f"desktop/step_{self._step_count:03d}.png",
                self._episode_context_id,
                "model_context_only",
                Path(__file__),
            )
        return LiteEnvStepResult(
            results=[
                make_tool_result(
                    call_id,
                    images=[self._image]
                    if not self._finished and index == len(call_ids) - 1
                    else [],
                    error=errors.get(call_id),
                )
                for index, call_id in enumerate(call_ids)
            ],
            terminated=terminated,
            truncated=self.truncated,
            reward=float(self.evaluation["status"] == "pass") if self.evaluation else None,
            info={EXECUTED_ACTIONS_INFO_KEY: executed, "model_turn": self._step_count},
        )

    def _finish_sync(self) -> None:
        if self._artifacts_closed:
            return
        self._pump(self.helpers.EVALUATION_SETTLE_SECONDS)
        page = self._active_page()
        (self.output / "terminal.png").write_bytes(self._capture())
        terminal = {"url": page.url, "title": page.title()}
        (self.output / "terminal_state.json").write_text(json.dumps(terminal, indent=2))
        (self.output / "agent_response.txt").write_text(self.final_content or "")
        if self.final_content is None:
            final_state = {
                "task_spec_sha256": sha256_json(self.candidate["task_spec"]),
                "confirmed": False,
                "reloaded_for_persistence": False,
                "reason": "The agent did not supply a final response.",
            }
        else:
            # Read-back uses the real terminal URL and session in a separate
            # context. Audit reloads and selector preparation cannot satisfy
            # the model's network evaluator or change HAR/trace event order.
            # Freeze the original model recordings before any independent GET
            # readback; preserve only the authenticated state for the audit.
            audit_storage_state = self._context.storage_state()
            self._close_browser_artifacts()
            audit_browser = self._playwright.chromium.launch(
                executable_path=os.environ.get(
                    "PLAYWRIGHT_CHROMIUM_EXECUTABLE", self._playwright.chromium.executable_path
                ),
                headless=False,
                args=["--window-size=1920,1080", "--start-maximized"],
            )
            audit_context = None
            try:
                audit_context = audit_browser.new_context(
                    no_viewport=True,
                    accept_downloads=False,
                    storage_state=audit_storage_state,
                )
                # Context identity comes from the actual created objects. Audit
                # browser requests must never enter the model recording scope.
                self._episode_audit_context_ids.append(
                    self._episode_context_id
                    if audit_context is self._episode_recording_context
                    else str(uuid4())
                )
                audit_context.route("**/*", self._restrict_origin)
                audit_page = audit_context.new_page()
                audit_started_at = datetime.now(UTC).isoformat()
                audit_page.goto(terminal["url"], wait_until="domcontentloaded")
                final_state = audit_final_state(self.candidate, self.site_urls, audit_page)
                (self.output / "independent_state_audit_lifecycle.json").write_text(
                    json.dumps(
                        {
                            "producer_sha256": sha256_file(Path(__file__)),
                            "audit_started_at": audit_started_at,
                            "audit_completed_at": datetime.now(UTC).isoformat(),
                            "recording_lifecycle_sha256": sha256_file(
                                self.output / "episode_recording_lifecycle.json"
                            ),
                            "recording_context_id": self._episode_context_id,
                            "audit_context_ids": self._episode_audit_context_ids,
                            "recording_artifact_sha256": {
                                "har": sha256_file(self.output / "network.har"),
                                "trace": sha256_file(self.output / "trace.zip"),
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                audit_page.bring_to_front()
                (self.output / "state_audit.png").write_bytes(self._capture())
            finally:
                try:
                    if audit_context is not None:
                        audit_context.close()
                finally:
                    audit_browser.close()
        (self.output / "final_state.json").write_text(
            json.dumps(final_state, ensure_ascii=False, indent=2)
        )
        if not (self.output / "state_audit.png").exists():
            (self.output / "state_audit.png").write_bytes(
                (self.output / "terminal.png").read_bytes()
            )
        self._close_browser_artifacts()
        # Offline scoring starts a new sync runtime after recording and read-back close.
        self._playwright.stop()
        self._playwright = None
        self.evaluation = score_reference(
            self.candidate, self.final_content, self.output, self.site_urls
        )
        (self.output / "reference_receipt.json").write_text(
            json.dumps(self.evaluation, ensure_ascii=False, indent=2)
        )

    def _close_browser_artifacts(self) -> None:
        if self._artifacts_closed or self._context is None:
            return
        self._artifacts_closed = True
        trace_closed_at = None
        try:
            if self._tracing_started:
                self._context.tracing.stop(path=self.output / "trace.zip")
                trace_closed_at = datetime.now(UTC).isoformat()
        finally:
            try:
                self._context.close()
            finally:
                self._close_recording_browser()
        context_closed_at = datetime.now(UTC).isoformat()
        # Receipt creation follows both successful recording flushes. Exceptions
        # from trace stop/context close retain their failure and emit no proof.
        if self._context is not self._episode_recording_context or trace_closed_at is None:
            (self.output / "episode_boundary_hold.json").write_text(
                json.dumps(
                    {
                        "status": "hold",
                        "reason": "original recording context or trace completion is unproven",
                    }
                )
            )
            return
        boundary_path = self.output / "episode_boundary.json"
        if boundary_path.exists():
            raise FileExistsError("episode boundary is produced once per fresh recording")
        boundary = {
            "scope": "mixed_contexts"
            if self._episode_context_id in self._episode_audit_context_ids
            else "model_context_only",
            "recording_context_id": self._episode_context_id,
            "har_context_id": self._episode_context_id,
            "playwright_context_id": self._episode_context_id,
            "audit_context_ids": self._episode_audit_context_ids,
            "started_at": self._episode_started_at,
            "ended_at": context_closed_at,
            "artifact_sha256": {
                "har": sha256_file(self.output / "network.har"),
                "trace": sha256_file(self.output / "trace.zip"),
                "gui_frames": sha256_file(self.output / "cross_site_gui_frames.jsonl"),
            },
        }
        lifecycle = {
            "producer_sha256": sha256_file(Path(__file__)),
            "recording_started_at": self._episode_started_at,
            "trace_stop_completed_at": trace_closed_at,
            "context_close_and_HAR_flush_completed_at": context_closed_at,
            "recording_context_id": self._episode_context_id,
            "audit_context_ids": self._episode_audit_context_ids,
            "boundary_sha256": sha256_json(boundary),
        }
        (self.output / "episode_recording_lifecycle.json").write_text(
            json.dumps(lifecycle, ensure_ascii=False, indent=2)
        )
        boundary_path.write_text(json.dumps(boundary, ensure_ascii=False, indent=2))

    def _close_recording_browser(self) -> None:
        """Release the owned native browser, including incomplete startup."""
        try:
            if self._browser is not None:
                self._browser.close()
                self._browser = None
        finally:
            if self._browser_process is not None:
                if self._browser_process.poll() is None:
                    self._browser_process.terminate()
                self._browser_process.wait(timeout=10)
                self._browser_process = None

    async def close(self) -> None:
        """Flush partial artifacts and release only this case's owned resources."""
        if self._closed:
            return
        try:
            await asyncio.get_running_loop().run_in_executor(self._executor, self._close_sync)
        finally:
            self._executor.shutdown(wait=True)

    def _close_sync(self) -> None:
        error = None
        try:
            if self._opened and not self._artifacts_closed:
                self._finished = True
                self._finish_sync()
        except BaseException as caught:
            error = caught
        finally:
            for resource, method, arguments in (
                (self, "_close_browser_artifacts", ()),
                (self, "_close_recording_browser", ()),
                (self._playwright, "stop", ()),
                (self._profile, "cleanup", ()),
                (self._xvfb, "__exit__", (None, None, None)),
            ):
                if resource is not None:
                    try:
                        getattr(resource, method)(*arguments)
                    except BaseException as caught:
                        error = error or caught
            self._closed = True
        if error is not None:
            raise error
