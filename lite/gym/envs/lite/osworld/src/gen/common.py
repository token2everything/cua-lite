"""Shared constants used across all generation tracks (synth, perturb, eval).

Holds data that train.synth, train.perturb, and eval all read: the per-domain
noise candidates and the per-app save postconfig (LO / VS Code / GIMP) used
to harden the agent's save-on-exit flow against focus-capture and dialog
quirks. Track-specific helpers live in `train/synth/_utils.py` and
`train/perturb/_utils.py`.

Run-as-script: this module is import-only; it has no `__main__` entry.
"""

from __future__ import annotations

from typing import Any


# Per-domain noise candidates. Declarative — the env reads this at reset
# time and decides whether to apply based on env_kwargs["noise"].
NOISE_CANDIDATES: dict[str, dict[str, Any]] = {
    # NOTE: only use noise types whose commands return immediately.  GUI-launching
    # types (open_terminal, background_app, extra_tab) block run_command and
    # cause reset() timeouts — excluded from all domains.
    "libreoffice_calc": {"candidates": ["mouse_jitter", "window_resize", "window_move", "desktop_files", "notification"], "max_apply": 3},
    "libreoffice_writer": {"candidates": ["mouse_jitter", "window_resize", "window_move", "desktop_files", "notification"], "max_apply": 3},
    "libreoffice_impress": {"candidates": ["mouse_jitter", "window_resize", "window_move", "desktop_files", "notification"], "max_apply": 3},
    "chrome": {"candidates": ["mouse_jitter", "window_resize", "window_move", "desktop_files", "notification"], "max_apply": 3},
    "gimp": {"candidates": ["mouse_jitter", "window_resize", "window_move", "desktop_files", "notification"], "max_apply": 2},
    "vlc": {"candidates": ["mouse_jitter", "window_resize", "desktop_files", "notification"], "max_apply": 2},
    "vs_code": {"candidates": ["mouse_jitter", "window_resize", "window_move", "desktop_files", "notification"], "max_apply": 3},
    "thunderbird": {"candidates": ["mouse_jitter", "window_resize", "window_move", "desktop_files", "notification"], "max_apply": 2},
    "os": {"candidates": ["mouse_jitter", "desktop_files", "notification"], "max_apply": 2},
    "multi_apps": {"candidates": ["mouse_jitter", "window_resize", "desktop_files", "notification"], "max_apply": 2},
}


# LibreOffice save postconfig. Used by libreoffice_{calc,impress,writer} and
# the LO-sink subset of multi_apps across synth, perturb, AND eval (eval swaps
# the upstream save-segment for this via gen/eval/postconfig.py).
#
# Divergence-from-upstream policy (keep an extra op ONLY IF it rescues a real
# failure case AND has zero side-effect in every scenario — GUI or terminal;
# otherwise drop it and match upstream). Upstream OSWorld's LO postconfig is
# bare `activate → sleep → Ctrl+S → sleep`, with ZERO dialog handling. The one
# addition that clears the bar is `alt+f,s`; the four dialog handlers that used
# to live here were all removed because they were redundant or state-damaging.
LO_SAVE_POSTCONFIG: list[dict[str, Any]] = [
    {"type": "activate_window", "parameters": {"window_name": "LibreOffice"}},
    {"type": "sleep", "parameters": {"seconds": 0.5}},
    # Ctrl+S — the upstream/VM save (matches the OSWorld original exactly).
    {"type": "key", "parameters": {"key": "ctrl+s"}},
    {"type": "sleep", "parameters": {"seconds": 1.0}},
    # KEPT extra op — alt+f,s (File → Save via the top-level menubar accelerator).
    #   Rescues: when the agent's last action left focus on an LO sidebar widget
    #     (e.g. Properties → Orientation combobox), plain Ctrl+S is silently
    #     swallowed by the focused inner widget — no dialog, no mtime change
    #     (live-reproduced: probe_swallow.py at the Orientation dropdown; even
    #     `windowactivate --sync` on the exact document window did not help).
    #     Alt+F is a top-level menubar accelerator that bypasses inner-widget
    #     focus capture, so File → Save fires and the edit flushes.
    #   Why zero side-effect:
    #     - Clean / already-saved doc (Ctrl+S above worked): Alt+F opens the
    #       File menu and `s` = Save is a no-op on an unmodified document.
    #       Harmless / idempotent.
    #     - Terminal / out-of-band edit (agent wrote the file via a script, GUI
    #       buffer is stale): forcing a save raises the "Changed by Others"
    #       modal, which we do NOT auto-confirm (see removed handlers below), so
    #       the modal BLOCKS the write and the on-disk edit survives — no clobber
    #       (verified: probe_clobber.py, marker survived).
    {"type": "key", "parameters": {"key": "alt+f"}},
    {"type": "sleep", "parameters": {"seconds": 0.3}},
    {"type": "key", "parameters": {"key": "s"}},
    {"type": "sleep", "parameters": {"seconds": 1.5}},
    # ---- REMOVED extra ops (all lite-only additions; upstream has none) ----
    # Each failed the keep-rule (rescue a real case AND zero side-effect):
    #   • "Changed by Others" → Return  — HAS a side-effect. Auto-confirming
    #     "Save Anyway" writes LO's stale in-memory buffer over an agent's
    #     on-disk / terminal edit = clobber (reproduced: probe_clobber.py).
    #   • "Keep Current Format" → Return — rescues NOTHING. lite's baked LO
    #     profile sets WarnAlienFormat=false, so this dialog never appears
    #     (verified: probe_keepformat.py → 0 dialogs). Dead handler.
    #   • "already exists" → Return      — rescues NOTHING on this substrate.
    #     lite's LO titles the Save-As replace-confirm "Save", not
    #     "…already exists" (verified: probe_saveas.py), so it never matches.
    #   • "^Save$" → Return              — HAS a side-effect. The Save-As FILE
    #     PICKER is itself titled exactly "Save" (verified: probe_saveas.py),
    #     so this matcher grabs the picker and presses Return — prematurely
    #     accepting a Save-As the agent left open (wrong / spurious filename).
    # Consequence: the "agent left a Save-As confirm open" case is intentionally
    # NOT rescued — the confirm and the picker share the WM_NAME "Save", so no
    # title-based matcher can hit one without the other. Upstream/VM doesn't
    # rescue it either, so leaving it un-handled matches parity.
]


# VS Code save postconfig. Used by code-edit synth templates whose eval reads
# the file from disk (`compare_text_file` / `check_gitignore_has_entries`):
# `_make_file_create_template`, `_make_file_edit_template`,
# `_make_asset_file_edit_template`, `_make_gitignore_create_template`.
#
# Mirrors OSWorld upstream's VS Code save postconfig (see reference task
# `0ed39f63-6049-43d4-ba4d-5fa2fe04a951`) plus the SAME single kept extra op as
# LO_SAVE_POSTCONFIG — `alt+f,s` (File → Save via menubar accelerator):
#   Rescues: Ctrl+S silently absorbed by a focused inner widget (Explorer /
#     Search panel / integrated Terminal); Alt+F is a top-level accelerator
#     that bypasses inner-widget focus capture, so File → Save still fires.
#   Why zero side-effect: on a clean/saved buffer it is a no-op; and unlike LO
#     there is no CBO-class auto-confirm — if the on-disk file is newer (agent
#     edited via terminal) VS Code's save FAILS with a passive notification
#     (no modal, no Return), so it cannot clobber the on-disk edit.
# No dialog handlers here (upstream has none, and none clear the keep-rule).
VS_CODE_SAVE_POSTCONFIG: list[dict[str, Any]] = [
    {"type": "activate_window", "parameters": {"window_name": "Visual Studio Code"}},
    {"type": "sleep", "parameters": {"seconds": 0.5}},
    {"type": "key", "parameters": {"key": "ctrl+s"}},
    {"type": "sleep", "parameters": {"seconds": 1.0}},
    {"type": "key", "parameters": {"key": "alt+f"}},
    {"type": "sleep", "parameters": {"seconds": 0.3}},
    {"type": "key", "parameters": {"key": "s"}},
    {"type": "sleep", "parameters": {"seconds": 0.5}},
]


def gimp_export_as_postconfig(out_path: str) -> list[dict[str, Any]]:
    """Per-task GIMP File → Export As postconfig.

    GIMP's native Ctrl+S only writes its own `.xcf`; user-visible image saves
    (`.png` / `.jpg` / etc.) require File → Export As, so this can't be a flat
    constant like `LO_SAVE_POSTCONFIG` — the dialog needs a target filename.

    Factory takes the absolute target path (e.g.
    `/home/user/Desktop/horse-blurred.jpg`) and returns the postconfig list.
    Pattern mirrors OSWorld upstream eval GIMP save (Shift+Ctrl+E →
    `pyautogui.write(out_path)` → Enter → handle Replace? → Enter for the
    format-options dialog). Top-level menubar accelerator (Shift+Ctrl+E)
    bypasses inner-widget focus capture per the LO save-path lesson.
    """
    return [
        {"type": "activate_window", "parameters": {"window_name": "Gimp", "by_class": True}},
        {"type": "sleep", "parameters": {"seconds": 0.5}},
        # File → Export As
        {"type": "key", "parameters": {"key": "shift+ctrl+e"}},
        {"type": "sleep", "parameters": {"seconds": 1.0}},
        # Fill the filename. GIMP's dialog auto-selects the existing entry on
        # open, so `pyautogui.write` replaces it cleanly (matches upstream).
        {"type": "execute", "parameters": {
            "command": [
                "python3", "-c",
                f"import pyautogui; pyautogui.write({out_path!r}); pyautogui.press('enter')",
            ],
        }},
        {"type": "sleep", "parameters": {"seconds": 2.0}},
        # If the target filename already exists on disk, GIMP pops a Replace?
        # confirm. Default focus is Replace — bare Return accepts.
        # KEEP verdict (keep-rule): rescues export-over-existing-file, and VERIFIED
        # zero side-effect — probe_gimp_export.py showed the other export dialogs
        # ("Export Image" picker, "Export Image as PNG" format-options) do NOT
        # match `already exists`, so this matcher can't grab the wrong window (the
        # picker-collision that forced dropping LO's `^Save$`; GIMP lacks it).
        # Conditional, so a no-op absent the confirm. See the matching eval-path
        # insertion + full rationale in gen/eval/postconfig.py (gimp branch).
        {"type": "execute", "parameters": {
            "command": (
                "WID=$(xdotool search --name 'already exists' 2>/dev/null | head -1); "
                "if [ -n \"$WID\" ]; then xdotool windowactivate \"$WID\" key Return; fi; true"
            ),
            "shell": True,
        }},
        {"type": "sleep", "parameters": {"seconds": 0.5}},
        # Accept the format-options dialog (JPEG quality / PNG compression /
        # GIF interlace / etc.) — Return triggers the default Export button.
        {"type": "key", "parameters": {"key": "Return"}},
        {"type": "sleep", "parameters": {"seconds": 3.0}},
    ]
