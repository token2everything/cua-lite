"""Strict setup transport for ScaleCUA OSWorld tasks."""

from __future__ import annotations

import copy
import re
from typing import Any

from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_action
from lite.gym.envs.lite.osworld.src.utils.setup import setup_fn as base_setup_fn
from lite.gym.errors import ScaleCuaTaskError
from lite.gym.sandbox.types import SandboxTaskConfig

SUPPORTED_ACTIONS = {
    "execute",
    "command",
    "launch",
    "open",
    "activate_window",
    "close_window",
    "download",
    "sleep",
    "chrome_open_tabs",
    "chrome_close_tabs",
    "update_browse_history",
    "left_click",
    "right_click",
    "double_click",
    "type_text",
    "key",
    "scroll",
    "mouse_move",
    "screenshot",
    "wait",
    "host_push",
}


async def setup_fn(task: SandboxTaskConfig, computer) -> None:
    """Run ScaleCUA setup.

    The base lite.osworld setup is called with an empty config to perform common
    readiness and desktop preamble. ScaleCUA actions are then dispatched one by
    one with a strict action allowlist.
    """
    reason = (task.metadata.get("others") or {}).get("exclude_reason")
    if reason:
        raise ScaleCuaTaskError(
            f"lite.scalecua task {task.task_id} is excluded: {reason}",
            phase="setup",
            kind="excluded_task",
        )
    base_task = copy.copy(task)
    base_task.metadata = {**task.metadata, "config": []}
    await base_setup_fn(base_task, computer)
    for index, action in enumerate(task.metadata.get("config", [])):
        await dispatch_strict(computer, action, phase="config", index=index)
    # No ownership handover — the server runs as the desktop user, so dispatch_strict's
    # files are user-owned by construction.


async def dispatch_strict(
    computer,
    action: dict[str, Any],
    *,
    phase: str,
    index: int,
    cache_dir: str = "/tmp",
) -> None:
    if not isinstance(action, dict):
        raise ScaleCuaTaskError(
            f"{phase}[{index}] must be an action object",
            phase=phase,
            kind="invalid_action",
        )
    action_type = action.get("type")
    if action_type not in SUPPORTED_ACTIONS:
        raise ScaleCuaTaskError(
            f"{phase}[{index}] unsupported action type: {action_type!r}",
            phase=phase,
            kind="unsupported_action",
        )
    action = _repair_generated_pyautogui_hotkeys(action)
    result = await dispatch_action(computer, action, cache_dir=cache_dir)
    _raise_on_dispatch_failure(result, phase=phase, index=index, action_type=str(action_type))
    await _verify_file_outputs(computer, action, phase=phase, index=index)


def _repair_generated_pyautogui_hotkeys(action: dict[str, Any]) -> dict[str, Any]:
    if action.get("type") not in {"execute", "command"}:
        return action
    params = action.get("parameters")
    if not isinstance(params, dict):
        return action
    command = params.get("command")
    if isinstance(command, str):
        fixed = _repair_pyautogui_hotkey_text(command)
    elif isinstance(command, list):
        fixed = [
            _repair_pyautogui_hotkey_text(part) if isinstance(part, str) else part
            for part in command
        ]
    else:
        return action
    if fixed == command:
        return action
    repaired = copy.deepcopy(action)
    repaired["parameters"]["command"] = fixed
    return repaired


_PYAUTOGUI_LIST_HOTKEY_RE = re.compile(
    r"pyautogui\.hotkey\(\s*(\[[^\]\n]*\])\s*\)"
)


def _repair_pyautogui_hotkey_text(command: str) -> str:
    return _PYAUTOGUI_LIST_HOTKEY_RE.sub(r"pyautogui.hotkey(*\1)", command)


def _raise_on_dispatch_failure(
    result: Any,
    *,
    phase: str,
    index: int,
    action_type: str,
) -> None:
    if not isinstance(result, dict):
        return
    status = str(result.get("status") or "").lower()
    returncode = result.get("returncode")
    if status == "error" or (returncode not in (None, 0)):
        stderr = str(result.get("stderr") or result.get("error") or "")[:500]
        raise ScaleCuaTaskError(
            f"{phase}[{index}] {action_type} failed"
            f" (status={status or 'unknown'}, returncode={returncode}): {stderr}",
            phase=phase,
            kind=f"{action_type}_failed",
            returncode=returncode if isinstance(returncode, int) else None,
        )


async def _verify_file_outputs(
    computer,
    action: dict[str, Any],
    *,
    phase: str,
    index: int,
) -> None:
    action_type = action.get("type")
    params = action.get("parameters") or {}
    paths: list[str] = []
    if action_type == "host_push":
        for item in params.get("files", []):
            dst = item.get("dst")
            if isinstance(dst, str):
                paths.append(dst)
    elif action_type == "download":
        for item in params.get("files", []):
            dst = item.get("path")
            if isinstance(dst, str):
                paths.append(dst if dst.startswith("/") else f"/home/user/{dst}")
    for path in paths:
        quoted = "'" + path.replace("'", "'\"'\"'") + "'"
        result = await computer.interface.run_command(f"test -e {quoted} && echo OK || echo MISSING")
        if "OK" not in (getattr(result, "stdout", "") or ""):
            raise ScaleCuaTaskError(
                f"{phase}[{index}] {action_type} did not produce expected file {path!r}",
                phase=phase,
                kind="missing_file_output",
            )
