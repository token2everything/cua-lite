"""HTTP boundary between cua-lite (Python 3.12) and WAA (Python 3.9)."""

from __future__ import annotations

import base64
import copy
import json
import logging
import os
import socket
import threading
import traceback
from pathlib import Path
from typing import Any

import requests
from desktop_env.envs.desktop_env import DesktopEnv  # pyright: ignore[reportMissingImports]
from flask import Flask, jsonify, request  # pyright: ignore[reportMissingImports]

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("waa-bridge")

app = Flask(__name__)
_lock = threading.Lock()
_env: DesktopEnv | None = None
_guest_ip = os.environ.get("WAA_GUEST_IP", "20.20.20.21")
_asset_base_url = os.environ.get("WAA_ASSET_BASE_URL", "http://127.0.0.1:5051")
_asset_manifest = json.loads(Path("/opt/waa-bridge/assets.json").read_text(encoding="utf-8"))
_asset_urls = {
    entry["source_url"]: f"{_asset_base_url}/{entry['sha256']}"
    for entry in _asset_manifest["assets"]
}


def _localize_assets(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _localize_assets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_localize_assets(item) for item in value]
    if isinstance(value, str):
        return _asset_urls.get(value, value)
    return value


def _screenshot_b64(observation: dict[str, Any] | None = None) -> str | None:
    screenshot = observation.get("screenshot") if observation else None
    if screenshot is None and _env is not None:
        screenshot = _env.render()
    if screenshot is None:
        return None
    if isinstance(screenshot, str):
        return screenshot.split(",", 1)[-1]
    return base64.b64encode(bytes(screenshot)).decode("ascii")


def _screen_size() -> dict[str, int]:
    if _env is None:
        return {"width": 1440, "height": 900}
    value = _env.vm_screen_size or {}
    return {
        "width": int(value.get("width") or value.get("x") or 1440),
        "height": int(value.get("height") or value.get("y") or 900),
    }


def _ready() -> tuple[bool, str]:
    try:
        response = requests.get(f"http://{_guest_ip}:5000/probe", timeout=2)
        response.raise_for_status()
        with socket.create_connection(("127.0.0.1", 7200), timeout=2):
            pass
        return True, ""
    except Exception as exc:
        return False, str(exc)


@app.errorhandler(Exception)
def _handle_error(exc: Exception):
    logger.error("bridge request failed: %s\n%s", exc, traceback.format_exc())
    return jsonify(ok=False, error=f"{type(exc).__name__}: {exc}"), 500


@app.get("/health")
def health():
    ready, detail = _ready()
    return jsonify(ready=ready, detail=detail), (200 if ready else 503)


@app.post("/reset")
def reset():
    global _env
    task_config = _localize_assets(copy.deepcopy(request.get_json(force=True)["task_config"]))
    with _lock:
        env = _env
        if env is None:
            env = DesktopEnv(
                action_space="pyautogui",
                cache_dir="/tmp/waa-cache",
                screen_size=(1440, 900),
                require_a11y_tree=False,
                require_terminal=False,
                emulator_ip=_guest_ip,
            )
            # QMP writes screendumps inside the QEMU container, while this
            # bridge only shares its network namespace. Fetch pixels from the
            # guest server so observations do not depend on a shared /tmp.
            env.vm_controller.take_screenshot = env.controller.get_screenshot
            _env = env
        observation = env.reset(task_config=task_config)
        return jsonify(
            ok=True,
            screenshot_b64=_screenshot_b64(observation),
            screen_size=_screen_size(),
        )


@app.post("/step")
def step():
    payload = request.get_json(force=True)
    with _lock:
        env = _env
        if env is None:
            raise RuntimeError("reset must be called before step")
        observation, _, done, info = env.step(
            payload["action"],
            pause=float(payload.get("pause", 0.5)),
        )
        return jsonify(
            ok=True,
            screenshot_b64=_screenshot_b64(observation),
            done=done,
            info=info,
        )


@app.post("/evaluate")
def evaluate():
    with _lock:
        env = _env
        if env is None:
            raise RuntimeError("reset must be called before evaluate")
        return jsonify(ok=True, reward=float(env.evaluate()))


@app.post("/screenshot")
def screenshot():
    with _lock:
        if _env is None:
            raise RuntimeError("reset must be called before screenshot")
        return jsonify(ok=True, screenshot_b64=_screenshot_b64())


@app.post("/close")
def close():
    global _env
    with _lock:
        if _env is not None:
            _env.close()
            _env = None
    return jsonify(ok=True)
