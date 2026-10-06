from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
CAMPAIGN = REPO / "devs/exps/eval/browsergym.webarena_verified/campaign"
HARNESS = Path(__file__).with_name("webarena_verified_capture.cjs")


@pytest.fixture(scope="module")
def capture_runtime(tmp_path_factory):
    spec = importlib.util.find_spec("playwright")
    if spec is None:
        pytest.skip("pinned Playwright is not installed")
    driver = Path(spec.origin).parent / "driver"
    package = driver / "package"
    overlay = tmp_path_factory.mktemp("capture-overlay") / "driver"
    subprocess.run(
        [
            "bash",
            str(CAMPAIGN / "build-playwright-capture.sh"),
            str(package / "lib/server"),
            str(overlay),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return driver / "node", package, overlay


def test_original_redirect_snapshot_loses_late_cookie(capture_runtime):
    node, package, _ = capture_runtime
    result = subprocess.run(
        [
            str(node),
            "--unhandled-rejections=strict",
            str(HARNESS),
            str(package),
            str(package / "lib/server"),
            "redirect",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "capture_probe" in result.stderr


@pytest.mark.parametrize(
    "scenario", ["redirect", "concurrent", "normal200", "failed", "stream", "closing", "deadline"]
)
def test_patched_actual_capture_producer(capture_runtime, scenario):
    node, package, overlay = capture_runtime
    result = subprocess.run(
        [
            str(node),
            "--unhandled-rejections=strict",
            str(HARNESS),
            str(package),
            str(overlay),
            scenario,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert '"passed":true' in result.stdout


@pytest.mark.parametrize("scenario", ["orderPair", "order451"])
def test_original_same_millisecond_trace_uses_completion_order(capture_runtime, scenario):
    node, package, _ = capture_runtime
    result = subprocess.run(
        [
            str(node),
            "--unhandled-rejections=strict",
            str(HARNESS),
            str(package),
            str(package / "lib/server"),
            scenario,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "ERR_ASSERTION" in result.stderr
    expected_ids = ("fast302", "slow200") if scenario == "orderPair" else ("har77", "har76")
    assert all(identifier in result.stderr for identifier in expected_ids)


@pytest.mark.parametrize(
    "scenario",
    [
        "orderPair",
        "order451",
        "orderTriple",
        "orderStream",
        "orderSameStream",
        "orderFailed",
        "orderLateFull",
        "orderAcrossChunk",
    ],
)
def test_patched_actual_request_start_order(capture_runtime, scenario):
    node, package, overlay = capture_runtime
    result = subprocess.run(
        [
            str(node),
            "--unhandled-rejections=strict",
            str(HARNESS),
            str(package),
            str(overlay),
            scenario,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    proof = json.loads(result.stdout)
    assert proof["passed"] and proof["realClasses"]
    if scenario == "order451":
        assert proof["ids"] == ["har73", "har74", "har76", "har77"]
        assert proof["statuses"] == [200, 302, 200, 302]


@pytest.mark.parametrize("scenario", ["orderLateFull", "orderAcrossChunk"])
def test_original_and_patched_inherit_cross_chunk_partial_full_entries(capture_runtime, scenario):
    node, package, overlay = capture_runtime
    counts = []
    for source in (package / "lib/server", overlay):
        result = subprocess.run(
            [
                str(node),
                "--unhandled-rejections=strict",
                str(HARNESS),
                str(package),
                str(source),
                scenario,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        proof = json.loads(result.stdout)
        counts.append(len(proof["ids"]))
        assert proof["statuses"][-1] == 200
    assert counts == ([2, 2] if scenario == "orderLateFull" else [3, 3])


def test_capture_overlay_repeated_build_preserves_source(capture_runtime):
    _, package, overlay = capture_runtime
    source = package / "lib/server/har/harTracer.js"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    patched_before = (overlay / "har/harTracer.js").read_bytes()
    subprocess.run(
        [
            "bash",
            str(CAMPAIGN / "build-playwright-capture.sh"),
            str(package / "lib/server"),
            str(overlay),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert (overlay / "har/harTracer.js").read_bytes() == patched_before


def test_capture_overlay_rejects_unknown_source_before_output(tmp_path):
    source = tmp_path / "source"
    (source / "har").mkdir(parents=True)
    (source / "trace/recorder").mkdir(parents=True)
    (source / "har/harTracer.js").write_text("wrong driver")
    (source / "trace/recorder/tracing.js").write_text("wrong driver")
    output = tmp_path / "overlay"
    result = subprocess.run(
        ["bash", str(CAMPAIGN / "build-playwright-capture.sh"), str(source), str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not output.exists()


def test_capture_builder_uses_short_checksum_and_git_without_patch(capture_runtime, tmp_path):
    _, package, expected_overlay = capture_runtime
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in (
        "bash",
        "git",
        "dirname",
        "basename",
        "mktemp",
        "rm",
        "rmdir",
        "mkdir",
        "cp",
        "mv",
        "cmp",
    ):
        (tools / name).symlink_to(shutil.which(name))
    checksum = tools / "sha256sum"
    checksum.write_text(
        f"#!{shutil.which('bash')}\n"
        'for argument in "$@"; do\n'
        '    case "$argument" in --*|-s) exit 2 ;; esac\n'
        "done\n"
        f'exec {shlex.quote(shutil.which("sha256sum"))} "$@"\n'
    )
    checksum.chmod(0o700)
    environment = {**os.environ, "PATH": str(tools)}
    assert (
        subprocess.run(
            [str(tools / "bash"), "-c", "command -v patch"],
            capture_output=True,
            env=environment,
        ).returncode
        != 0
    )
    source = package / "lib/server"
    before = {
        relative: (source / relative).read_bytes()
        for relative in ("har/harTracer.js", "trace/recorder/tracing.js")
    }
    output = tmp_path / "overlay"
    for _ in range(2):
        result = subprocess.run(
            [
                str(tools / "bash"),
                str(CAMPAIGN / "build-playwright-capture.sh"),
                str(source),
                str(output),
            ],
            capture_output=True,
            env=environment,
        )
        assert result.returncode == 0, result.stderr.decode()
    for relative, data in before.items():
        assert (source / relative).read_bytes() == data
        assert (output / relative).read_bytes() == (expected_overlay / relative).read_bytes()
    assert not list(tmp_path.glob(".playwright-capture.*"))


def test_capture_builder_rejects_wrong_output_hash_before_publish(capture_runtime, tmp_path):
    _, package, _ = capture_runtime
    builder = tmp_path / "builder"
    builder.mkdir()
    for name in (
        "build-playwright-capture.sh",
        "playwright-capture.patch",
        "playwright-capture.sha256",
    ):
        shutil.copyfile(CAMPAIGN / name, builder / name)
    lock = builder / "playwright-capture.sha256"
    lock.write_text("0" * 64 + "  har/harTracer.js\n" + lock.read_text().splitlines()[1] + "\n")
    output = tmp_path / "overlay"
    source = package / "lib/server"
    before = (source / "har/harTracer.js").read_bytes()
    result = subprocess.run(
        ["bash", str(builder / "build-playwright-capture.sh"), str(source), str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "computed checksum" in result.stderr
    assert not output.exists()
    assert (source / "har/harTracer.js").read_bytes() == before
    assert not list(tmp_path.glob(".playwright-capture.*"))


def test_capture_builder_rejects_corrupt_existing_output(capture_runtime, tmp_path):
    _, package, expected_overlay = capture_runtime
    source = package / "lib/server"
    before = (source / "har/harTracer.js").read_bytes()
    output = tmp_path / "overlay"
    shutil.copytree(expected_overlay, output)
    corrupt = output / "trace/recorder/tracing.js"
    corrupt.write_text("corrupt immutable output\n")
    result = subprocess.run(
        ["bash", str(CAMPAIGN / "build-playwright-capture.sh"), str(source), str(output)],
        capture_output=True,
    )
    assert result.returncode != 0
    assert corrupt.read_text() == "corrupt immutable output\n"
    assert (source / "har/harTracer.js").read_bytes() == before
    assert not list(tmp_path.glob(".playwright-capture.*"))


@pytest.mark.live
@pytest.mark.parametrize(
    "patched,mode", [(False, "delayed"), (True, "delayed"), (True, "closing"), (True, "never")]
)
def test_real_chromium_capture_and_owned_browser_cleanup(
    capture_runtime, tmp_path, monkeypatch, patched, mode
):
    """Drive a fresh local HTTP form through the actual Python/Node/Chromium stack."""
    from playwright._impl import _transport
    from playwright.sync_api import sync_playwright

    node, package, overlay = capture_runtime
    isolated = tmp_path / "package"
    subprocess.run(["cp", "-a", str(package), str(isolated)], check=True)
    if patched:
        for relative in ("har/harTracer.js", "trace/recorder/tracing.js"):
            subprocess.run(
                ["cp", str(overlay / relative), str(isolated / "lib/server" / relative)], check=True
            )
    bootstrap = Path(__file__).with_name("webarena_verified_capture_driver.cjs")
    monkeypatch.setattr(
        _transport, "compute_driver_executable", lambda: (str(node), str(bootstrap))
    )
    monkeypatch.setenv("TEST_CAPTURE_PACKAGE", str(isolated))
    monkeypatch.setenv("TEST_CAPTURE_MODE", mode)
    extra_events = tmp_path / "extra-events.jsonl"
    monkeypatch.setenv("TEST_CAPTURE_EVENTS", str(extra_events))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = (
                (
                    b'<form method="post" action="/redirect"><input name="product" value="30874">'
                    b'<button id="add">Add</button></form>'
                )
                if self.path == "/form"
                else b"Wishlist: white desk"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(302)
            self.send_header("Location", "/wishlist")
            self.send_header("Set-Cookie", "capture_probe=complete; Path=/")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    runner_spec = importlib.util.spec_from_file_location(
        "capture_live_runner", REPO / "scripts/webarena_verified_qwen.py"
    )
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    trace_path = tmp_path / "trace.zip"
    har_path = tmp_path / "network.har"
    events = []
    try:
        with sync_playwright() as playwright:
            chromium = Path(playwright.chromium.executable_path)
            if not chromium.is_file():
                pytest.skip("pinned Chromium is not available for the live capture test")
            browser = playwright.chromium.launch(executable_path=str(chromium), headless=True)
            context = browser.new_context(record_har_path=str(har_path), record_har_content="embed")
            context.tracing.start(snapshots=True)
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/form")
            page.click("#add")
            page.wait_for_url("**/wishlist")
            assert page.content().find("Wishlist: white desk") >= 0
            assert (
                json.loads(extra_events.read_text().splitlines()[0])["type"]
                == "real_extra_info_intercepted"
            )
            if mode == "closing":
                page.close()

            stop_trace, close_context, close_browser = (
                context.tracing.stop,
                context.close,
                browser.close,
            )

            def stop(**kwargs):
                events.append("trace")
                return stop_trace(**kwargs)

            def finish_context():
                events.append("context")
                return close_context()

            def finish_browser():
                events.append("browser")
                return close_browser()

            monkeypatch.setattr(context.tracing, "stop", stop)
            monkeypatch.setattr(context, "close", finish_context)
            monkeypatch.setattr(browser, "close", finish_browser)
            started = time.monotonic()
            error = runner.finalize_browser_artifacts(
                context, browser, trace_path, tracing_started=True, sleep_fn=lambda _: None
            )
            elapsed = time.monotonic() - started
            if mode != "delayed":
                assert error is not None and "metadata capture" in str(error)
                assert events == ["trace", "browser", "context"]
                assert not trace_path.exists()
                assert not browser.is_connected()
                if mode == "never":
                    assert 29 <= elapsed < 40, (
                        "real 30000ms capture deadline and owned browser cleanup"
                    )
                else:
                    assert elapsed < 5
            else:
                assert error is None and events == ["trace", "context", "browser"]
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=5)

    if mode == "delayed":
        har = json.loads(har_path.read_text())["log"]["entries"]
        with zipfile.ZipFile(trace_path) as archive:
            rows = [
                json.loads(line)["snapshot"] for line in archive.read("trace.network").splitlines()
            ]
        for entries, cookies in (
            (har, ["capture_probe"]),
            (rows, ["capture_probe"] if patched else []),
        ):
            (redirect,) = [
                entry
                for entry in entries
                if entry["request"]["method"] == "POST"
                and entry["request"]["url"].endswith("/redirect")
            ]
            assert redirect["response"]["status"] == 302
            assert [cookie["name"] for cookie in redirect["response"]["cookies"]] == cookies
            assert [
                header["value"]
                for header in redirect["response"]["headers"]
                if header["name"].lower() == "location"
            ] == ["/wishlist"]
            (wishlist,) = [
                entry for entry in entries if entry["request"]["url"].endswith("/wishlist")
            ]
            assert wishlist["response"]["status"] == 200
            assert wishlist["response"]["cookies"] == []
            assert not any(
                header["name"].lower() == "location" for header in wishlist["response"]["headers"]
            )
        injected = [json.loads(line) for line in extra_events.read_text().splitlines()]
        released = [
            event for event in injected if event["type"] == "same_chain_extra_info_released"
        ]
        assert [event["statusCode"] for event in released] == [302, 200]
        assert len({event["requestId"] for event in released}) == 1


def test_runtime_metadata_records_actual_installed_producer(capture_runtime):
    _, package, _ = capture_runtime
    spec = importlib.util.spec_from_file_location(
        "capture_metadata_runner", REPO / "scripts/webarena_verified_qwen.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    metadata = runner.runtime_metadata("dataset-checksum")
    assert metadata["playwright_capture"] == {
        relative: hashlib.sha256((package / "lib/server" / relative).read_bytes()).hexdigest()
        for relative in ("har/harTracer.js", "trace/recorder/tracing.js")
    }
