#!/usr/bin/env python3
"""osworld (v1) in-container eval server — runs INSIDE `cua-lite/osworld` (the container's
Linux userspace, OUTSIDE the guest VM).

Hosts OSWorld v1's `desktop_env` + `osworld_evaluation_examples` here so the cua-lite HOST
needs ZERO `desktop_env` (eval-in-container, the androidworld pattern). Attaches to the guest
VM's Flask server at the qemu-docker guest IP (20.20.20.21):5000 and exposes a thin JSON HTTP-RPC the host drives:

  GET  /healthz                 -> {ok, vm_ready}
  POST /reset  {domain,task_id} -> load the task JSON + dte.reset(task) -> {instruction, screenshot_b64}
  POST /step   {cmd, pause}     -> dte.step(cmd, pause) -> {ok}
  POST /screenshot              -> {screenshot_b64}
  POST /evaluate                -> {reward}
  POST /close                   -> {ok}

Baked into the image at /usr/local/bin/server.py; the host bind-mounts the checked-in copy
over it and `docker exec -d`'s it once the VM's :5000 answers.

Config via env (set at `docker run`): SCREEN_W/SCREEN_H (1920/1080), CLIENT_PASSWORD
(password), CACHE_DIR (/root/.cache), SERVER_PORT (6000). No cua-lite imports.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
from typing import Any

import traceback

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_SCREEN_W = int(os.environ.get("SCREEN_W", "1920"))
_SCREEN_H = int(os.environ.get("SCREEN_H", "1080"))
_CLIENT_PASSWORD = os.environ.get("CLIENT_PASSWORD", "password")
_CACHE_DIR = os.environ.get("CACHE_DIR", "/root/.cache")
_SERVER_PORT = int(os.environ.get("SERVER_PORT", "6000"))
_VM_HOST = os.environ.get("VM_HOST", "20.20.20.21")  # qemu-docker guest IP (DNAT target); NOT localhost
_VM_SERVER_PORT = 5000
_VM_CHROMIUM_PORT = 9222
_VM_VLC_PORT = 8080

app = FastAPI(title="cua-lite osworld (v1) in-container eval server")
_env: Any = None
_lock = threading.Lock()


@app.exception_handler(Exception)
async def _all_exc(request: Request, exc: Exception) -> JSONResponse:
    # Surface the in-container traceback in the RPC response (else the host sees a bare 500).
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": traceback.format_exc()[-2500:]})


def _task_json(domain: str, task_id: str) -> dict:
    import osworld_evaluation_examples
    d = os.path.dirname(osworld_evaluation_examples.__file__)
    path = os.path.join(d, "examples", domain, f"{task_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"task JSON not found: {path}")
    return json.load(open(path))


def _attached_desktop_env():
    from desktop_env.desktop_env import DesktopEnv
    from desktop_env.controllers.python import PythonController
    from desktop_env.controllers.setup import SetupController

    class _Attached(DesktopEnv):
        def __init__(self):
            # Replicate DesktopEnv.__init__ MINUS create_vm_manager_and_provider — the docker
            # provider connects to docker.sock eagerly, which doesn't exist inside this container
            # (and we never use the provider: _start_emulator attaches, close no-ops). Pinned
            # dist, so this mirror is stable. NOT a monkeypatch — a clean subclass override.
            self.region = None
            self.provider_name = "docker"
            self.enable_proxy = False
            self.client_password = _CLIENT_PASSWORD or "password"
            self.screen_width, self.screen_height = _SCREEN_W, _SCREEN_H
            self.server_port, self.chromium_port, self.vnc_port, self.vlc_port = 5000, 9222, 8006, 8080
            self.current_use_proxy = False
            self.manager, self.provider = None, None
            self.os_type = "Ubuntu"
            self.is_environment_used = False
            self.path_to_vm = "__attached__"
            self.snapshot_name = "init_state"
            self.cache_dir_base = _CACHE_DIR
            self.headless = False
            self.require_a11y_tree = False
            self.require_terminal = False
            self._start_emulator()
            self.instruction = None
            self.action_space = "pyautogui"
            self._traj_no, self._step_no, self.action_history = -1, 0, []

        def _start_emulator(self):
            self.vm_ip = _VM_HOST
            self.server_port = _VM_SERVER_PORT
            self.chromium_port = _VM_CHROMIUM_PORT
            self.vlc_port = _VM_VLC_PORT
            self.vnc_port = 0
            self.controller = PythonController(vm_ip=self.vm_ip, server_port=self.server_port)
            self.setup_controller = SetupController(
                vm_ip=self.vm_ip, server_port=self.server_port,
                chromium_port=self.chromium_port, vlc_port=self.vlc_port,
                cache_dir=self.cache_dir_base, client_password=self.client_password,
                screen_width=self.screen_width, screen_height=self.screen_height)

        def _revert_to_snapshot(self):
            # Attach model has no snapshot + no provider. reset() marks the env dirty before setup,
            # so a setup-retry re-enters reset()'s revert branch → provider.revert_to_snapshot(None)
            # → crash (masking the real setup error). No-op: reset() re-attaches via _start_emulator()
            # right after this; one trajectory per freshly-booted, reaped container — nothing to revert.
            pass

        def close(self):
            pass

    return _Attached()


def _encode_screenshot(obs: Any) -> str | None:
    if obs is None:
        return None
    from PIL import Image
    import numpy as np
    if isinstance(obs, (bytes, bytearray)):
        return base64.b64encode(bytes(obs)).decode()
    buf = io.BytesIO()
    if isinstance(obs, Image.Image):
        obs.save(buf, format="PNG")
    elif isinstance(obs, np.ndarray):
        Image.fromarray(obs).save(buf, format="PNG")
    else:
        return None
    return base64.b64encode(buf.getvalue()).decode()


class ResetBody(BaseModel):
    domain: str
    task_id: str


class StepBody(BaseModel):
    cmd: str
    pause: float = 0


@app.get("/healthz")
def healthz() -> dict:
    import requests
    vm_ready = False
    try:
        vm_ready = requests.get(f"http://{_VM_HOST}:{_VM_SERVER_PORT}/screenshot", timeout=(3, 3)).status_code == 200
    except Exception:
        pass
    return {"ok": True, "vm_ready": vm_ready, "env_ready": _env is not None}


@app.post("/reset")
def reset(body: ResetBody) -> dict:
    global _env
    with _lock:
        task = _task_json(body.domain, body.task_id)
        if _env is not None:
            try:
                _env.close()
            except Exception:
                pass
        _env = _attached_desktop_env()
        _env.reset(task_config=task)
        return {"instruction": task.get("instruction", ""),
                "screenshot_b64": _encode_screenshot(_env.controller.get_screenshot())}


@app.post("/step")
def step(body: StepBody) -> dict:
    if _env is None:
        raise HTTPException(status_code=409, detail="no active env; call /reset first")
    with _lock:
        _env.step(body.cmd, body.pause)
    return {"ok": True}


@app.post("/screenshot")
def screenshot() -> dict:
    if _env is None:
        raise HTTPException(status_code=409, detail="no active env")
    return {"screenshot_b64": _encode_screenshot(_env.controller.get_screenshot())}


@app.post("/evaluate")
def evaluate() -> dict:
    if _env is None:
        raise HTTPException(status_code=409, detail="no active env")
    with _lock:
        raw = _env.evaluate()
    # v1 evaluate() returns a float; mirror v2's tolerant extraction so both servers agree on the
    # contract (a dict result scores its ["score"], never a TypeError).
    reward = float(raw.get("score", 0.0)) if isinstance(raw, dict) else float(raw)
    return {"reward": reward}


@app.post("/close")
def close() -> dict:
    global _env
    with _lock:
        if _env is not None:
            try:
                _env.close()
            except Exception:
                pass
            _env = None
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=_SERVER_PORT, log_level="warning")
