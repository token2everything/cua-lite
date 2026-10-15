"""Import-safe OSWorld postconfig normalization helpers."""

from __future__ import annotations

import copy

from lite.core.tools.action_space.keys import normalize_keys
from lite.gym.envs.lite.osworld.src.gen.common import (
    LO_SAVE_POSTCONFIG,
    VS_CODE_SAVE_POSTCONFIG,
)

_LO_DOMAINS = frozenset({
    "libreoffice_calc", "libreoffice_writer", "libreoffice_impress",
})


def _step_command_text(step: dict) -> str:
    cmd = step.get("parameters", {}).get("command")
    if cmd is None:
        return ""
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    return str(cmd).lower().replace("'", "").replace('"', "")


def _step_key_tokens(step: dict) -> list[str]:
    if step.get("type") != "key":
        return []
    p = step.get("parameters", {})
    for raw in (p.get("keys"), p.get("key")):
        if raw is None:
            continue
        try:
            return normalize_keys(raw)
        except (TypeError, ValueError):
            continue
    return []


def _step_has_ctrl_s(step: dict) -> bool:
    if _step_key_tokens(step) == ["ctrl", "s"]:
        return True
    cmd = _step_command_text(step)
    return "ctrl+s" in cmd or (
        "hotkey" in cmd and " s" in cmd and "ctrl" in cmd
    ) or ("hotkey" in cmd and ",s" in cmd and "ctrl" in cmd)


def _step_has_shift_ctrl_e(step: dict) -> bool:
    if _step_key_tokens(step) in (["shift", "ctrl", "e"], ["ctrl", "shift", "e"]):
        return True
    cmd = _step_command_text(step)
    return "shift" in cmd and "ctrl" in cmd and "hotkey" in cmd and (
        " e" in cmd or ",e" in cmd or "(e" in cmd
    )


def _step_activate_window_target(step: dict) -> str:
    if step.get("type") != "activate_window":
        return ""
    return str(step.get("parameters", {}).get("window_name", ""))


# --- VLC graceful-quit postconfig upgrade --------------------------------------
# A bare `pkill vlc` / `killall -9 vlc` postconfig discards the agent's un-flushed
# GUI edit (VLC persists `vlcrc` only on a clean quit). Upgrade such a step to a
# graceful ctrl+q quit that runs the config-save path, with SIGTERM (not -9) as
# the fallback. Safe/idempotent: if no VLC window exists the sequence no-ops.
_VLC_GRACEFUL_QUIT_COMMAND = (
    "for wid in $(xdotool search --class vlc 2>/dev/null); do "
    'xdotool windowactivate --sync "$wid" 2>/dev/null || true; '
    'xdotool key --window "$wid" ctrl+q 2>/dev/null || true; '
    "done; "
    "for _ in $(seq 1 20); do pgrep -x vlc >/dev/null 2>&1 || break; sleep 0.5; done; "
    "pkill -TERM vlc 2>/dev/null || true; sleep 2; sync"
)


def _step_is_vlc_kill(step: dict) -> bool:
    cmd = _step_command_text(step)
    if "vlc" not in cmd:
        return False
    return "pkill" in cmd or "killall" in cmd or ("kill " in cmd and "-9" in cmd)


def _find_save_segment(postconfig: list[dict], window_substr: str) -> tuple[int, int] | None:
    n = len(postconfig)
    for start in range(n):
        step = postconfig[start]
        if step.get("type") != "activate_window":
            continue
        if window_substr not in _step_activate_window_target(step):
            continue
        index = start + 1
        while index < n and postconfig[index].get("type") == "sleep":
            index += 1
        if index >= n or not _step_has_ctrl_s(postconfig[index]):
            continue
        end = index + 1
        while end < n and postconfig[end].get("type") == "sleep":
            end += 1
        return (start, end)
    return None


def normalize_postconfig(postconfig: list[dict], domain: str) -> list[dict]:
    """Harden upstream OSWorld postconfig with eval/train safety nets."""

    if not postconfig:
        return postconfig

    has_ctrl_s = any(_step_has_ctrl_s(step) for step in postconfig)
    has_shift_ctrl_e = any(_step_has_shift_ctrl_e(step) for step in postconfig)
    activate_targets = " ".join(_step_activate_window_target(step) for step in postconfig)

    is_lo_save = has_ctrl_s and (
        domain in _LO_DOMAINS or "LibreOffice" in activate_targets
    )
    if is_lo_save:
        segment = _find_save_segment(postconfig, "LibreOffice")
        if segment is not None:
            start, end = segment
            return (
                [copy.deepcopy(step) for step in postconfig[:start]]
                + copy.deepcopy(LO_SAVE_POSTCONFIG)
                + [copy.deepcopy(step) for step in postconfig[end:]]
            )
        return copy.deepcopy(LO_SAVE_POSTCONFIG)

    if has_ctrl_s and "Visual Studio Code" in activate_targets:
        segment = _find_save_segment(postconfig, "Visual Studio Code")
        if segment is not None:
            start, end = segment
            return (
                [copy.deepcopy(step) for step in postconfig[:start]]
                + copy.deepcopy(VS_CODE_SAVE_POSTCONFIG)
                + [copy.deepcopy(step) for step in postconfig[end:]]
            )
        return copy.deepcopy(VS_CODE_SAVE_POSTCONFIG)

    if has_shift_ctrl_e and domain == "gimp":
        # GIMP File→Export As hardening (9 eval tasks + the synth `gimp_export_as_postconfig`
        # helper). Upstream has NO overwrite-confirm handler: it exports via `pyautogui.write`
        # → Enter and, on tasks that export OVER an existing image, the GTK "replace?" confirm
        # would be left un-accepted → the agent's export never lands on disk → false FAIL.
        # This inserts a KEPT extra op (Return-dismiss of that confirm) right after the write.
        #
        # KEEP verdict (keep-rule: rescue a real case AND zero side-effect):
        #   Rescues: export-over-existing-file left blocked on the "replace?" confirm.
        #   Zero side-effect — VERIFIED (probe_gimp_export.py, on lite's baked GIMP): the two
        #     OTHER dialogs in the export flow do NOT match `already exists`, so this matcher
        #     can never grab the wrong window (this is exactly the picker-title collision that
        #     forced us to DROP LO's `^Save$` handler — GIMP does NOT have it):
        #       - Export file-picker  WM_NAME = "Export Image"        (no match)
        #       - format-options       WM_NAME = "Export Image as PNG" (no match)
        #     It is conditional (`if [ -n "$WID" ]`), so absent the confirm it is a pure no-op.
        #   Live/dead of GIMP's own overwrite-confirm title is UNCONFIRMED — 3 probes could not
        #     reliably force the file-collision to catch its WM_NAME. Harmless either way: if the
        #     confirm isn't titled `…already exists` this is a dead no-op; if it is, it rescues.
        #     Kept (maintainer decision) as a zero-risk safety net; different app from LO, so the
        #     LO "confirm titled 'Save'" finding does not transfer.
        out: list[dict] = []
        inserted = False
        for index, step in enumerate(postconfig):
            out.append(copy.deepcopy(step))
            if inserted:
                continue
            cmd = step.get("parameters", {}).get("command")
            cmd_str = (
                " ".join(map(str, cmd)) if isinstance(cmd, list) else str(cmd or "")
            ).lower()
            if "pyautogui.write" in cmd_str and "enter" in cmd_str:
                if index + 1 < len(postconfig) and postconfig[index + 1].get("type") == "sleep":
                    out.append(copy.deepcopy(postconfig[index + 1]))
                    postconfig = list(postconfig)
                    postconfig[index + 1] = {"_consumed": True}
                out.append({
                    "type": "execute",
                    "parameters": {
                        "command": (
                            "WID=$(xdotool search --name 'already exists' 2>/dev/null | head -1); "
                            "if [ -n \"$WID\" ]; then xdotool windowactivate \"$WID\" key Return; fi; true"
                        ),
                        "shell": True,
                    },
                })
                out.append({"type": "sleep", "parameters": {"seconds": 0.5}})
                inserted = True
        return [step for step in out if not step.get("_consumed")]

    if any(_step_is_vlc_kill(step) for step in postconfig):
        out = []
        for step in postconfig:
            if _step_is_vlc_kill(step):
                out.append({
                    "type": "execute",
                    "parameters": {
                        "command": _VLC_GRACEFUL_QUIT_COMMAND,
                        "shell": True,
                    },
                })
            else:
                out.append(copy.deepcopy(step))
        return out

    return postconfig
