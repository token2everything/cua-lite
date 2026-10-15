"""HTTP RPC for androidlab — drives BOTH the per-step adb path
(tap/swipe/text/screencap/uiautomator-dump) and the judge / XML
compression / find_package helpers from inside the container.

Why everything runs in-container
--------------------------------
The original androidlab integration ran adb via host-side
``docker exec container adb …`` (``DockerExecAndroidController``).
That works correctly but every step pays the docker-daemon API
serialization cost (3-4 docker-exec round-trips per step). Under
concurrent workloads (~16+ envs sharing one docker daemon) per-step
latency drifts up to 30+ s, which is why the legacy host-side path
needed ``step_timeout=120`` while androidworld (pure in-container
HTTP RPC) is fine at the default 30 s.

This server unifies androidlab with androidworld's style: every
per-step action is a single localhost HTTP POST into THIS process,
which then runs ``adb -s emulator-5554 …`` directly on the local
adb daemon — no docker exec on the hot path.

API surface
-----------
  GET  /healthz                       -> {ok, controller_ready, task_ready}
  POST /init  (pickled {device})      -> {ok, width, height}

  # adb passthrough (replaces DockerExecAndroidController on host)
  POST /env/tap                {x, y}
  POST /env/long_press         {x, y, duration_ms}
  POST /env/swipe_precise      {sx, sy, ex, ey, duration_ms}
  POST /env/text               {text}
  POST /env/key                {key: home|back|enter|menu}
  POST /env/launch_app         {package}
  POST /env/get_current_activity -> {activity: str}
  POST /env/execute_adb        {cmd: str}  -> {result: str}
  POST /env/get_screenshot     -> pickled bytes (PNG, or None)
  POST /env/get_xml            {ac: bool}  -> pickled (xml_str or None)

  # static helpers + judge (unchanged)
  POST /env/resolve_package    {app_name}   -> {package}
  POST /env/compressed_xml     {xml_str}    -> pickled dict
  POST /task/load              {judge_class_name, app_module_name}
  POST /task/judge             {compressed_xml, line}
  POST /task/release
  POST /close

Run (inside container):
    python /usr/local/bin/server.py --port 9554
"""
from __future__ import annotations

import argparse
import base64
import logging
import pickle
import subprocess
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)
app = FastAPI(title="cua-lite androidlab in-container server")

# One container = one task at a time. Module-level state is fine.
_task: Any = None
_controller: "_CmdController | None" = None


def _pickled_response(obj: Any) -> Response:
    return Response(content=pickle.dumps(obj), media_type="application/octet-stream")


async def _pickled_body(request: Request) -> Any:
    raw = await request.body()
    return pickle.loads(raw) if raw else {}


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "controller_ready": _controller is not None,
        "task_ready": _task is not None,
    }


# ── in-container adb controller ──────────────────────────────────────────────
# Behavior parity with the legacy host-side ``DockerExecAndroidController``
# (which inherited from reference's :class:`AndroidController`):
#   - tap / long_press / swipe_precise / text / back / home / enter / menu
#   - launch_app via ``monkey -p <pkg> -c android.intent.category.LAUNCHER 1``
#   - get_current_activity via the same dumpsys-+-awk pipeline
#   - get_screenshot returns raw PNG bytes (streamed via ``adb exec-out
#     screencap -p`` — no /sdcard temp file, no docker cp)
#   - get_xml / get_ac_xml return the raw XML string (read via
#     ``adb exec-out cat`` from /sdcard — no docker cp)
#
# Where reference's controller has a bug (``swipe_precise`` uses start_x
# twice — see android-lab's upstream ``utils_mobile/and_controller.py``
# ``swipe_precise`` definition), we faithfully reproduce it so replay
# equality vs the legacy path holds.


class _CmdController:
    """Slim subprocess-based adb controller, runs LOCAL to the container.

    Replaces host-side :class:`DockerExecAndroidController` to drop the
    docker-exec round-trip from every per-step adb call. Method surface
    matches the call sites in :mod:`lite.gym.envs.androidlab.main`.
    """

    SCREENSHOT_REMOTE = "/sdcard/step.png"
    XML_REMOTE = "/sdcard/step.xml"
    AC_XML_REMOTE = "/sdcard/Android/data/com.example.android.xml_parser/files/ui.xml"

    def __init__(self, device: str):
        self.device = device
        self.width, self.height = self._get_device_size()

    def _adb(self, *args: str, timeout: float = 15.0) -> str:
        try:
            r = subprocess.run(
                ["adb", "-s", self.device, *args],
                capture_output=True, text=True, timeout=timeout,
            )
            if r.returncode == 0:
                return r.stdout.strip()
            return "ERROR"
        except subprocess.TimeoutExpired:
            return "ERROR"
        except Exception as e:
            logger.warning("adb call failed: %s", e)
            return "ERROR"

    def _shell(self, *args: str, timeout: float = 15.0) -> str:
        return self._adb("shell", *args, timeout=timeout)

    def _get_device_size(self) -> tuple[int, int]:
        # Reference retries up to 10× with 2s sleeps if the device hasn't
        # reported ``wm size`` yet (snapshot-resume can race the boot UI).
        for _ in range(10):
            try:
                out = self._shell("wm", "size")
                if out and ":" in out:
                    res = out.split(":")[1].strip()
                    w, h = res.split("x")
                    return int(w), int(h)
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("could not get device size after 10 retries")

    # --- input -------------------------------------------------------------

    def tap(self, x: int, y: int) -> str:
        return self._shell("input", "tap", str(x), str(y))

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> str:
        return self._shell(
            "input", "swipe", str(x), str(y), str(x), str(y), str(duration_ms),
        )

    def swipe_precise(
        self, sx: int, sy: int, ex: int, ey: int, duration_ms: int = 400,
    ) -> str:
        # Bug-for-bug parity with android-lab's reference controller
        # ``swipe_precise``: it builds the adb shell command as
        # ``swipe sx sx ex ey duration`` (passes start_x in both x
        # positions, never start_y). The legacy host-side path
        # inherited from reference and therefore propagated this bug,
        # so the captured baseline embeds it. Reproducing the bug here
        # keeps replay reward equality with the pre-refactor baseline.
        # To fix the bug for new rollouts, change ``str(sx)`` (2nd
        # arg) → ``str(sy)`` AND re-capture the baseline.
        _ = sy  # intentionally unused — see comment above
        return self._shell(
            "input", "swipe", str(sx), str(sx), str(ex), str(ey), str(duration_ms),
        )

    def text(self, s: str) -> str:
        # Reference's ``text`` clears the field via 100× KEYCODE_DEL
        # (=67) then broadcasts the b64-encoded payload via ADBKeyboard.
        # We expand the 100× repetition inline because the shell syntax
        # reference used (``$(for i in {1..100}; ...)``) needs zsh; we
        # spell it out to stay portable across the bullseye container.
        keycodes = ["67"] * 100
        self._shell("input", "keyevent", "--press", *keycodes)
        b64 = base64.b64encode(s.encode("utf-8")).decode("ascii")
        return self._shell(
            "am", "broadcast", "-a", "ADB_INPUT_B64", "--es", "msg", b64,
        )

    def back(self) -> str:
        return self._shell("input", "keyevent", "KEYCODE_BACK")

    def home(self) -> str:
        return self._shell("input", "keyevent", "KEYCODE_HOME")

    def enter(self) -> str:
        return self._shell("input", "keyevent", "KEYCODE_ENTER")

    def menu(self) -> str:
        # KEYCODE_MENU == 82
        return self._shell("input", "keyevent", "82")

    def recent(self) -> str:
        # KEYCODE_APP_SWITCH == 187
        return self._shell("input", "keyevent", "187")

    def launch_app(self, package: str) -> str:
        return self._shell(
            "monkey", "-p", package, "-c",
            "android.intent.category.LAUNCHER", "1",
        )

    def get_current_activity(self) -> str:
        # Reference's dumpsys-+-awk pipeline. We run it as a single shell
        # command (``adb shell '<pipeline>'``) so the pipes resolve
        # device-side, matching the host-side legacy controller exactly.
        out = self._adb(
            "shell",
            "dumpsys window | grep mCurrentFocus | "
            "awk -F '/' '{print $1}' | awk '{print $NF}'",
        )
        return out if out != "ERROR" else "0"

    def execute_adb(self, cmd: str) -> str:
        # Host-side legacy controller accepted ``adb -s <device> ...``
        # strings via shell=True. We do the same here. The host now
        # rarely uses this path (only Menu-key in main.py), but we keep
        # the entrypoint for completeness.
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30.0,
            )
            return r.stdout.strip() if r.returncode == 0 else "ERROR"
        except Exception:
            return "ERROR"

    # --- screenshot / xml --------------------------------------------------

    def get_screenshot_bytes(self) -> bytes | None:
        """Take a screenshot and return the raw PNG bytes.

        Uses ``adb exec-out screencap -p`` which streams the PNG over
        the adb pipe directly — no /sdcard temp file, no docker cp.
        Matches what reference's ``get_screenshot`` produced byte-wise
        (same encoder), just without the round-trip overhead.
        """
        try:
            r = subprocess.run(
                ["adb", "-s", self.device, "exec-out", "screencap", "-p"],
                capture_output=True, timeout=15.0,
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout
        except Exception as e:
            logger.warning("screencap failed: %s", e)
        return None

    def _read_remote(self, remote: str, timeout: float = 15.0) -> bytes | None:
        try:
            r = subprocess.run(
                ["adb", "-s", self.device, "exec-out", "cat", remote],
                capture_output=True, timeout=timeout,
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout
        except Exception as e:
            logger.warning("exec-out cat %s failed: %s", remote, e)
        return None

    def get_xml(self) -> str | None:
        """uiautomator dump → return raw XML string.

        Reference's ``get_xml`` does ``uiautomator dump <remote>`` then
        ``adb pull`` to host. We replace the pull with ``adb exec-out cat
        <remote>`` so the XML stays in this process — no /tmp file.
        Retries 5× with a 1 s sleep to match reference's robustness.
        """
        for _ in range(5):
            r = self._shell("uiautomator", "dump", self.XML_REMOTE, timeout=15.0)
            if r == "ERROR":
                time.sleep(1)
                continue
            content = self._read_remote(self.XML_REMOTE)
            if content and len(content) > 200:
                return content.decode("utf-8", errors="replace")
            time.sleep(1)
        return None

    def get_ac_xml(self) -> str | None:
        """Read the XMLParser a11y service's ``ui.xml`` (map.me/pimusic).

        Mirrors reference's ``get_ac_xml`` which only PULLs (no dump —
        the XMLParser app writes the file continuously). Same 5× retry.
        """
        for _ in range(5):
            content = self._read_remote(self.AC_XML_REMOTE)
            if content and len(content) > 200:
                return content.decode("utf-8", errors="replace")
            time.sleep(2)
        return None


def _ctrl() -> _CmdController:
    if _controller is None:
        raise HTTPException(409, "controller not initialized (call /init first)")
    return _controller


# ── controller endpoints (hot path) ──────────────────────────────────────────


@app.post("/init")
async def init(request: Request) -> dict[str, Any]:
    """Body: pickled ``{device}``. Idempotent — second call returns the
    cached width/height without reinstantiating."""
    global _controller
    body = await _pickled_body(request)
    device = body.get("device", "emulator-5554")
    if _controller is not None and _controller.device == device:
        return {
            "ok": True, "already_initialized": True,
            "width": _controller.width, "height": _controller.height,
        }
    _controller = _CmdController(device)
    return {
        "ok": True, "already_initialized": False,
        "width": _controller.width, "height": _controller.height,
    }


@app.post("/env/tap")
async def env_tap(request: Request) -> dict[str, Any]:
    body = await _pickled_body(request)
    _ctrl().tap(int(body["x"]), int(body["y"]))
    return {"ok": True}


@app.post("/env/long_press")
async def env_long_press(request: Request) -> dict[str, Any]:
    body = await _pickled_body(request)
    _ctrl().long_press(
        int(body["x"]), int(body["y"]),
        duration_ms=int(body.get("duration_ms", 1000)),
    )
    return {"ok": True}


@app.post("/env/swipe_precise")
async def env_swipe_precise(request: Request) -> dict[str, Any]:
    body = await _pickled_body(request)
    _ctrl().swipe_precise(
        int(body["sx"]), int(body["sy"]),
        int(body["ex"]), int(body["ey"]),
        duration_ms=int(body.get("duration_ms", 400)),
    )
    return {"ok": True}


@app.post("/env/text")
async def env_text(request: Request) -> dict[str, Any]:
    body = await _pickled_body(request)
    _ctrl().text(str(body.get("text", "")))
    return {"ok": True}


@app.post("/env/key")
async def env_key(request: Request) -> dict[str, Any]:
    body = await _pickled_body(request)
    key = body.get("key", "")
    method = {
        "home": "home",
        "back": "back",
        "enter": "enter",
        "menu": "menu",
        "recent": "recent",
    }.get(key)
    if method is None:
        raise HTTPException(
            400,
            f"unknown key {key!r} (expected home/back/enter/menu/recent)",
        )
    getattr(_ctrl(), method)()
    return {"ok": True}


@app.post("/env/launch_app")
async def env_launch_app(request: Request) -> dict[str, Any]:
    body = await _pickled_body(request)
    _ctrl().launch_app(str(body["package"]))
    return {"ok": True}


@app.post("/env/get_current_activity")
def env_get_current_activity() -> dict[str, Any]:
    return {"activity": _ctrl().get_current_activity()}


@app.post("/env/execute_adb")
async def env_execute_adb(request: Request) -> dict[str, Any]:
    body = await _pickled_body(request)
    return {"result": _ctrl().execute_adb(str(body["cmd"]))}


@app.post("/env/get_screenshot")
def env_get_screenshot() -> Response:
    """Returns pickled PNG bytes (or pickled None on failure)."""
    return _pickled_response(_ctrl().get_screenshot_bytes())


@app.post("/env/get_xml")
async def env_get_xml(request: Request) -> Response:
    """Body: ``{ac: bool}``. ac=True uses the XMLParser a11y dump
    (map.me/pimusic), else uiautomator dump. Returns pickled str or
    pickled None on failure."""
    body = await _pickled_body(request)
    if body.get("ac", False):
        return _pickled_response(_ctrl().get_ac_xml())
    return _pickled_response(_ctrl().get_xml())


@app.post("/env/observe")
async def env_observe(request: Request) -> Response:
    """Batched observation: screenshot + xml + compressed_xml +
    current_activity in ONE round trip. Body fields (all bool, all
    default True): ``ac``, ``want_screenshot``, ``want_xml``,
    ``want_compressed_xml``, ``want_activity``.

    Per-step in :meth:`AndroidLabEnv.step` this collapses 4 RPCs
    (screenshot + xml + compressed_xml + activity) into 1 — saves
    ~10-20 ms cumulative per step. Under 96-conc rollouts × tens of
    steps × hundreds of trajectories, that's hours of wall-clock.

    The compressed_xml result mirrors the standalone
    ``/env/compressed_xml`` endpoint's shape: parsed dict on success,
    ``None`` when the raw XML was missing or unparseable.
    """
    body = await _pickled_body(request)
    ctrl = _ctrl()
    result: dict[str, Any] = {}
    if body.get("want_screenshot", True):
        result["screenshot"] = ctrl.get_screenshot_bytes()

    xml_str: str | None = None
    if body.get("want_xml", True):
        xml_str = (
            ctrl.get_ac_xml() if body.get("ac", False) else ctrl.get_xml()
        )
        result["xml"] = xml_str
    if body.get("want_compressed_xml", True):
        if xml_str is None:
            result["compressed_xml"] = None
        else:
            from android_lab.utils_mobile.utils import get_compressed_xml_from_str
            import json
            compressed = get_compressed_xml_from_str(xml_str, type="json")
            result["compressed_xml"] = json.loads(compressed) if compressed else None
    if body.get("want_activity", True):
        result["current_activity"] = ctrl.get_current_activity()
    return _pickled_response(result)


# ── pure helpers (no env state needed) ───────────────────────────────────────

@app.post("/env/resolve_package")
async def env_resolve_package(request: Request) -> dict[str, Any]:
    """app_name → package_name via androidlab's fuzzy lookup."""
    body = await _pickled_body(request)
    from android_lab.templates.packages import find_package
    return {"package": find_package(body["app_name"])}


@app.post("/env/compressed_xml")
async def env_compressed_xml(request: Request) -> Response:
    """Compress a raw uiautomator XML string into androidlab's
    canonical dict form (the format every judge consumes).

    Despite the upstream docstring claiming "Returns a nested-dict",
    ``get_compressed_xml_from_str`` actually returns a JSON STRING.
    The judges all walk a Python dict (``find_subtrees_of_parents_with_key``
    bails on the first ``isinstance(current, dict)`` check), so we MUST
    parse it before handing off — see reference's ``evaluation/task.py:24``
    which does the same ``json.loads`` step. Without this, every judge
    using the tree silently returns ``judge_page=False`` and reward=0.
    """
    body = await _pickled_body(request)
    xml_str = body["xml_str"]
    from android_lab.utils_mobile.utils import get_compressed_xml_from_str
    import json
    compressed = get_compressed_xml_from_str(xml_str, type="json")
    if not compressed:
        return _pickled_response(None)
    return _pickled_response(json.loads(compressed))


# ── task lifecycle ───────────────────────────────────────────────────────────

@app.post("/task/load")
async def task_load(request: Request) -> dict[str, Any]:
    """Body: ``{judge_class_name, app_module_name}``. Imports the judge
    module + class, instantiates a fresh judge, stashes it for
    /task/judge calls.

    androidlab judges are stateful (each judge instance tracks "page
    states seen this episode"), so we re-instantiate on every reset.
    Pass the module path explicitly because the registry path in
    tasks.json carries it (it's derived from the YAML's metric_func).
    """
    global _task
    body = await _pickled_body(request)
    judge_class_name = body["judge_class_name"]
    app_module_name = body["app_module_name"]

    import importlib
    module = importlib.import_module(
        f"android_lab.evaluation.tasks.{app_module_name}"
    )
    function_map = getattr(module, "function_map", None)
    if function_map is None or judge_class_name not in function_map:
        raise HTTPException(
            404,
            f"judge {judge_class_name!r} not found in "
            f"android_lab.evaluation.tasks.{app_module_name}",
        )
    judge_cls = function_map[judge_class_name]
    _task = judge_cls()
    return {"ok": True}


@app.post("/task/judge")
async def task_judge(request: Request) -> Response:
    """Body: ``{compressed_xml, line}``. Runs ``task.judge(...)`` and
    returns the result. Result shape is judge-specific (typically a
    list / tuple / dict produced by the rewarder)."""
    if _task is None:
        raise HTTPException(409, "task not initialized (call /task/load first)")
    body = await _pickled_body(request)
    result = _task.judge(body["compressed_xml"], body["line"])
    return _pickled_response(result)


@app.post("/task/release")
def task_release() -> dict[str, Any]:
    """Drop the in-container judge instance. Called by AndroidLabEnv on
    close() (and before /task/load on reset to free the previous task's
    state)."""
    global _task
    _task = None
    return {"ok": True}


@app.post("/close")
def close() -> dict[str, Any]:
    """Tear-down. androidlab has no env_launcher to close — the
    container's emulator is reaped by AndroidLabContainer's docker rm -f
    when the host releases its handle. We just drop the judge +
    controller instance so the next ``/init`` rebinds cleanly.
    """
    global _task, _controller
    _task = None
    _controller = None
    return {"ok": True}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
