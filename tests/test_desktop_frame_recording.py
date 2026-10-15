"""A captured model screenshot survives unavailable navigation-time DOM evidence."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import zipfile
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from contract import sha256_file
from playwright.sync_api import Error as PlaywrightError

from lite.core.tools.calls import make_tool_call

pytest_plugins = ["test_episode_applicability"]

NAVIGATION_ERROR = "Execution context was destroyed, most likely because of a navigation"


@pytest.fixture
def native_keys(monkeypatch):
    """Own an Xvfb display and send native browser shortcuts through XTest."""
    if shutil.which("Xvfb") is None:
        pytest.skip("Xvfb is required for the headed browser regression")
    server = subprocess.Popen(
        ["Xvfb", "-displayfd", "1", "-screen", "0", "1920x1080x24", "-nolisten", "tcp", "-ac"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    display_name = ":" + server.stdout.readline().strip()
    monkeypatch.setenv("DISPLAY", display_name)
    x11 = ctypes.CDLL("libX11.so.6")
    xtest = ctypes.CDLL("libXtst.so.6")
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
    x11.XStringToKeysym.restype = ctypes.c_ulong
    x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    x11.XKeysymToKeycode.restype = ctypes.c_uint
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    xtest.XTestFakeKeyEvent.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_ulong,
    ]
    display = x11.XOpenDisplay(display_name.encode())

    def keys(*names):
        codes = [
            x11.XKeysymToKeycode(display, x11.XStringToKeysym(name.encode())) for name in names
        ]
        for code in codes:
            xtest.XTestFakeKeyEvent(display, code, True, 0)
        for code in reversed(codes):
            xtest.XTestFakeKeyEvent(display, code, False, 0)
        x11.XFlush(display)

    try:
        yield keys
    finally:
        x11.XCloseDisplay(display)
        server.terminate()
        server.wait(timeout=10)


@pytest.mark.parametrize("restore_storage", [False, True])
def test_public_cdp_no_defaults_preserves_native_tabs_and_full_recording(
    runtime, tmp_path, native_keys, monkeypatch, restore_storage
):
    """Native tab changes bind recorded GUI frames to the actual displayed DOM."""
    import desktop_env
    from PIL import ImageGrab
    from playwright.sync_api import sync_playwright
    from webarena_verified.types.tracing import NetworkTrace

    with sync_playwright() as playwright:
        executable = Path(
            os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE", playwright.chromium.executable_path)
        )
    if not executable.is_file():
        pytest.skip("An installed Chromium is required for the headed regression")
    monkeypatch.setenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE", str(executable))
    public = {
        "generated_task_id": "wavgen-native-tabs",
        "sites": ["shopping", "gitlab"],
        "start_urls": ["http://shopping.test/original", "http://gitlab.test/target"],
    }
    monkeypatch.setattr(desktop_env, "resolve_public_task", lambda *_: public)
    monkeypatch.setattr(desktop_env, "build_public_instruction", lambda _: "Use the desktop.")
    monkeypatch.setitem(sys.modules, "pyautogui", SimpleNamespace(size=lambda: (1920, 1080)))
    captures = []
    during_capture = None

    def capture(*_):
        stream = BytesIO()
        ImageGrab.grab(xdisplay=os.environ["DISPLAY"]).save(stream, format="PNG")
        captures.append(stream.getvalue())
        if during_capture is not None:
            during_capture()
        return captures[-1]

    storage = tmp_path / "storage.json"
    storage.write_text(
        json.dumps(
            {
                "cookies": [],
                "origins": [
                    {
                        "origin": "http://shopping.test",
                        "localStorage": [{"name": "fixture", "value": "restored"}],
                    }
                ],
            }
        )
    )
    env = desktop_env.GeneratedDesktopEnv(
        SimpleNamespace(
            display=os.environ["DISPLAY"],
            osworld_root=tmp_path,
            storage_state=storage if restore_storage else None,
            max_steps=150,
        ),
        {},
        {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"},
        tmp_path,
        SimpleNamespace(
            xvfb=lambda _: nullcontext(),
            load_cursor_class=lambda _: None,
            screenshot_bytes=capture,
            RESET_SETTLE_SECONDS=0.1,
            POST_ACTION_DELAY_SECONDS=0.1,
        ),
    )
    env._restrict_origin = lambda route: route.fulfill(
        status=200,
        content_type="text/html",
        body="<title>Same shopping title</title><body>" + route.request.url + "</body>",
    )

    def record_step(expected_url, expected_body=None):
        before = len(captures)
        result = env._step_sync(
            [
                make_tool_call(
                    "computer", {"actions": [{"action": "wait", "duration": 0}]}, call_id="call-1"
                )
            ]
        )
        frame = json.loads((tmp_path / "cross_site_gui_frames.jsonl").read_text().splitlines()[-1])
        assert len(captures) == before + 1
        assert result.results[0].images == [captures[-1]]
        assert (tmp_path / frame["screenshot"]).read_bytes() == captures[-1]
        assert frame["page_url"] == expected_url
        assert frame["body_text"] == (expected_url if expected_body is None else expected_body)
        assert frame["screenshot_sha256"] == sha256_file(tmp_path / frame["screenshot"])

    try:
        env._reset_sync()
        context = env._context
        first, second = context.pages
        if restore_storage:
            assert first.evaluate("localStorage.getItem('fixture')") == "restored"
        native_keys("Control_L", "1")
        record_step("http://shopping.test/original")
        native_keys("Control_L", "2")
        record_step("http://gitlab.test/target")
        with context.expect_page() as opened:
            native_keys("Control_L", "t")
        third = opened.value
        native_keys("Control_L", "l")
        for char in "http://shopping.test/product":
            if char == ":":
                native_keys("Shift_L", "semicolon")
            else:
                native_keys({"/": "slash", ".": "period"}.get(char, char))
        native_keys("Return")
        third.wait_for_url("http://shopping.test/product")
        assert first.title() == third.title()
        record_step("http://shopping.test/product")
        assert [page.evaluate("document.visibilityState") for page in context.pages] == [
            "hidden",
            "hidden",
            "visible",
        ]
        native_keys("Control_L", "2")
        record_step("http://gitlab.test/target")
        native_keys("Control_L", "l")
        record_step("http://gitlab.test/target")
        native_keys("Escape")
        # The screenshot is already captured when each transition occurs. The
        # producer must keep those bytes but discard the mismatched DOM facts.
        for change in ("navigation", "reload", "text", "input", "scroll", "tab", "focus"):
            if change in {"navigation", "reload"}:
                during_capture = (
                    (lambda: second.goto("http://gitlab.test/cart"))
                    if change == "navigation"
                    else second.reload
                )
            elif change == "text":
                during_capture = lambda: second.evaluate("document.body.textContent += ' revised'")
            elif change == "input":
                second.evaluate(
                    "document.body.insertAdjacentHTML('beforeend', '<input value=12.99>')"
                )
                during_capture = lambda: second.evaluate(
                    "document.querySelector('input').value = '58.63'"
                )
            elif change == "scroll":
                second.evaluate("document.body.style.minHeight = '3000px'")
                during_capture = lambda: second.evaluate("window.scrollTo(0, 100)")
            else:

                def change_focus(key="1" if change == "tab" else "l"):
                    native_keys("Control_L", key)
                    second.wait_for_timeout(100)

                during_capture = change_focus
            record_step("")
            during_capture = None
            if change == "tab":
                record_step("http://shopping.test/original")
                native_keys("Control_L", "2")
                second.wait_for_timeout(100)
            elif change == "focus":
                record_step("http://gitlab.test/cart", "")
                native_keys("Escape")
            else:
                record_step(
                    "http://gitlab.test/cart",
                    ""
                    if change == "scroll"
                    else "http://gitlab.test/cart revised"
                    if change in {"text", "input"}
                    else "http://gitlab.test/cart",
                )
                if change == "input":
                    frame = json.loads(
                        (tmp_path / "cross_site_gui_frames.jsonl").read_text().splitlines()[-1]
                    )
                    assert 'value="58.63"' in frame["visible_html"]
        second.evaluate("window.scrollTo(0, 0)")
        second.evaluate(
            """() => fetch('/notes', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({note: 'Briess $12.99'})
            }).then(response => response.text())"""
        )
        process = env._browser_process
        env._close_browser_artifacts()
        assert process.poll() is not None
        entries = json.loads((tmp_path / "network.har").read_text())["log"]["entries"]
        urls = [
            "http://shopping.test/original",
            "http://gitlab.test/target",
            "http://shopping.test/product",
        ]
        for url in urls:
            [entry] = [entry for entry in entries if entry["request"]["url"] == url]
            assert url in entry["response"]["content"]["text"]
            assert "startedDateTime" in entry and "timings" in entry
        with zipfile.ZipFile(tmp_path / "trace.zip") as trace:
            assert "trace.network" in trace.namelist()
        har = NetworkTrace.from_har(tmp_path / "network.har")
        trace = NetworkTrace.from_playwright_trace(tmp_path / "trace.zip")
        assert len(har.events) >= 3 and len(trace.events) >= 3
        for recording in (har, trace):
            [post] = [event for event in recording.events if event.http_method == "POST"]
            assert post.post_data == {"note": "Briess $12.99"}
            for url in urls:
                [event] = [
                    event for event in recording.events if event.data["request"]["url"] == url
                ]
                assert url in event.data["response"]["content"]["text"]
        boundary = json.loads((tmp_path / "episode_boundary.json").read_text())
        for name, path in [
            ("har", "network.har"),
            ("trace", "trace.zip"),
            ("gui_frames", "cross_site_gui_frames.jsonl"),
        ]:
            assert boundary["artifact_sha256"][name] == sha256_file(tmp_path / path)
        assert boundary["scope"] == "model_context_only"
    finally:
        env._opened = False
        env._close_sync()
        env._executor.shutdown(wait=True)


@pytest.fixture
def desktop(runtime, tmp_path):
    from desktop_env import GeneratedDesktopEnv

    env = GeneratedDesktopEnv.__new__(GeneratedDesktopEnv)
    env.args = SimpleNamespace(max_steps=150)
    env.public = {"generated_task_id": "wavgen-frame-test", "sites": ["shopping", "gitlab"]}
    env._opened = True
    env._closed = env._finished = False
    env._step_count = 0
    env.evaluation = None
    env._episode_context_id = "original"
    env.output = tmp_path
    (tmp_path / "desktop").mkdir()
    env.helpers = SimpleNamespace(POST_ACTION_DELAY_SECONDS=0)
    env._dispatch_gui = MagicMock()
    env._pump = MagicMock()
    env._capture = MagicMock(return_value=b"already captured screenshot")
    return env


@pytest.mark.parametrize("phase", ["active_page", "frame"])
def test_navigation_context_destruction_keeps_screenshot_without_guessing_tab(desktop, phase):
    page = MagicMock()
    page.is_closed.return_value = False
    page.evaluate.side_effect = (
        [PlaywrightError(NAVIGATION_ERROR)] if phase == "active_page" else [True]
    )
    page.evaluate_handle.return_value.evaluate.side_effect = PlaywrightError(NAVIGATION_ERROR)
    background = MagicMock()
    background.is_closed.return_value = False
    background.evaluate.return_value = False
    desktop._context = SimpleNamespace(pages=[page, background])
    result = desktop._step_sync(
        [
            make_tool_call(
                "computer", {"actions": [{"action": "wait", "duration": 0}]}, call_id="call-1"
            )
        ]
    )
    assert not result.terminated and not result.truncated
    assert result.results[0].tool_call_id == "call-1"
    assert result.results[0].images == [b"already captured screenshot"]
    assert result.results[0].error is None
    desktop._capture.assert_called_once()
    desktop._pump.assert_called_once_with(0)
    assert page.evaluate.call_count == 1
    assert background.evaluate.call_count == (0 if phase == "active_page" else 1)
    [frame] = [
        json.loads(line)
        for line in (desktop.output / "cross_site_gui_frames.jsonl").read_text().splitlines()
    ]
    assert frame["page_url"] == frame["body_text"] == frame["visible_html"] == ""
    assert frame["links"] == []
    assert frame["screenshot_sha256"] == sha256_file(desktop.output / "desktop/step_001.png")


@pytest.mark.parametrize("phase", ["active_page", "frame"])
@pytest.mark.parametrize(
    "error",
    [
        PlaywrightError("Target page, context or browser has been closed"),
        PlaywrightError("unrelated DOM error"),
        ValueError(NAVIGATION_ERROR),
    ],
)
def test_other_frame_errors_remain_failures(desktop, phase, error):
    page = MagicMock()
    page.is_closed.return_value = False
    page.evaluate.side_effect = [error] if phase == "active_page" else [True]
    page.evaluate_handle.return_value.evaluate.side_effect = error
    desktop._context = SimpleNamespace(pages=[page])
    with pytest.raises(type(error), match=str(error)):
        desktop._step_sync(
            [
                make_tool_call(
                    "computer", {"actions": [{"action": "wait", "duration": 0}]}, call_id="call-1"
                )
            ]
        )
    desktop._capture.assert_called_once()
    assert (desktop.output / "desktop/step_001.png").read_bytes() == b"already captured screenshot"
    assert not (desktop.output / "cross_site_gui_frames.jsonl").exists()


@pytest.mark.parametrize("phase", ["active_page", "frame"])
def test_real_navigation_destroys_context_and_next_gui_turn_recovers(desktop, monkeypatch, phase):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        executable = Path(
            os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE", playwright.chromium.executable_path)
        )
        if not executable.is_file():
            pytest.skip("An installed Chromium is required for the real navigation regression")
        browser = playwright.chromium.launch(executable_path=str(executable), headless=True)
        try:
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            context.route(
                "**/*",
                lambda route: route.fulfill(
                    status=200, content_type="text/html", body="<body>Recovered document</body>"
                ),
            )
            page = context.new_page()
            page.goto("http://shopping.test/cart")
            desktop._context = context
            captured = []

            def capture():
                screenshot = page.screenshot()
                captured.append(screenshot)
                return screenshot

            desktop._capture = MagicMock(side_effect=capture)
            evaluate = page.evaluate
            errors = []

            def navigate_during_evaluation(expression, *args, **kwargs):
                active_probe = expression == "document.visibilityState === 'visible'"
                if not errors and active_probe and phase == "active_page":
                    try:
                        return evaluate("""() => new Promise(() => {
                            location.href = '/after-navigation';
                        })""")
                    except PlaywrightError as error:
                        errors.append(str(error))
                        raise
                return evaluate(expression, *args, **kwargs)

            monkeypatch.setattr(page, "evaluate", navigate_during_evaluation)
            evaluate_handle = page.evaluate_handle

            def document_handle(expression):
                handle = evaluate_handle(expression)
                read = handle.evaluate

                def navigate_during_frame(expression):
                    if phase == "frame" and not errors:
                        try:
                            return evaluate("""() => new Promise(() => {
                                location.href = '/after-navigation';
                            })""")
                        except PlaywrightError as error:
                            errors.append(str(error))
                            raise
                    return read(expression)

                monkeypatch.setattr(handle, "evaluate", navigate_during_frame)
                return handle

            monkeypatch.setattr(page, "evaluate_handle", document_handle)
            first = desktop._step_sync(
                [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "wait", "duration": 0}]},
                        call_id="call-1",
                    )
                ]
            )
            assert len(errors) == 1 and NAVIGATION_ERROR in errors[0]
            assert len(captured) == 1 and first.results[0].images == [captured[0]]
            assert first.results[0].error is None
            [empty] = [
                json.loads(line)
                for line in (desktop.output / "cross_site_gui_frames.jsonl")
                .read_text()
                .splitlines()
            ]
            assert empty["page_url"] == empty["body_text"] == empty["visible_html"] == ""
            assert empty["links"] == []
            second = desktop._step_sync(
                [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "wait", "duration": 0}]},
                        call_id="call-2",
                    )
                ]
            )
            assert len(captured) == 2 and second.results[0].images == [captured[1]]
            empty, recovered = [
                json.loads(line)
                for line in (desktop.output / "cross_site_gui_frames.jsonl")
                .read_text()
                .splitlines()
            ]
            assert recovered["page_url"] == "http://shopping.test/after-navigation"
            assert recovered["body_text"] == "Recovered document"
            assert empty["screenshot_sha256"] == sha256_file(
                desktop.output / "desktop/step_001.png"
            )
            assert recovered["screenshot_sha256"] == sha256_file(
                desktop.output / "desktop/step_002.png"
            )
        finally:
            browser.close()


@pytest.mark.parametrize(
    "visibility,focus,expected",
    [
        ([False, False], [], None),
        ([True, True], [False, False], None),
        ([True, True], [True, False], 0),
        ([True, True], [False, True], 1),
        ([False, True], [], 1),
    ],
)
def test_active_page_never_guesses_background_or_ambiguous_window(
    desktop, visibility, focus, expected
):
    pages = [MagicMock(), MagicMock()]
    for index, page in enumerate(pages):
        page.is_closed.return_value = False
        page.evaluate.side_effect = [visibility[index], *focus[index : index + 1]]
    desktop._context = SimpleNamespace(pages=pages)
    assert desktop._active_page() is (None if expected is None else pages[expected])


def test_no_visible_page_records_empty_evidence_with_original_screenshot(desktop):
    hidden = MagicMock()
    hidden.is_closed.return_value = False
    hidden.evaluate.return_value = False
    desktop._context = SimpleNamespace(pages=[hidden])
    desktop._step_sync(
        [
            make_tool_call(
                "computer", {"actions": [{"action": "wait", "duration": 0}]}, call_id="call-1"
            )
        ]
    )
    [frame] = [
        json.loads(line)
        for line in (desktop.output / "cross_site_gui_frames.jsonl").read_text().splitlines()
    ]
    assert frame["page_url"] == frame["body_text"] == frame["visible_html"] == ""
    assert frame["links"] == []
    hidden.evaluate.assert_called_once_with("document.visibilityState === 'visible'")
    desktop._capture.assert_called_once()
    assert frame["screenshot_sha256"] == sha256_file(desktop.output / "desktop/step_001.png")
