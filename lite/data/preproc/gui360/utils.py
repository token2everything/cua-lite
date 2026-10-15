"""Shared helpers for the GUI-360 preproc adapters.

Every per-task script (``grounding-point.py``, ``understanding.py``,
``use.py``) shares the same wiring against :mod:`lite.data.staging`:

* image-store with the canonical ``cua-lite/GUI-360/images`` prefix,
* split assigner keyed off ``metadata.others.id`` (per-row unique for the
  processed subsets; per-trajectory unique ``execution_id`` for ``use`` (multi-step rollout)),
* per-row staging helper that hashes images, fills the canonical metadata
  defaults, assigns a split, and produces a partition key.

GUI-360 is desktop / Windows / Microsoft-Office only, so the dataset-wide
constants (OS, source, the Windows virtual-key map used to translate the
upstream ``{VK_*}`` hotkey tokens) live here too.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lite.data.preproc.common import SourceStaging

DATASET_NAME = "GUI-360"

SOURCE = "vyokky/GUI-360"
# Every GUI-360 trajectory is captured on Windows Microsoft-Office apps.
OS = "windows"
PLATFORM = "desktop"

_STAGING = SourceStaging(DATASET_NAME)
out_dir_for = _STAGING.out_dir_for
make_image_store = _STAGING.make_image_store
make_splitter = _STAGING.make_splitter
stage_entry = _STAGING.stage_entry


def iter_json_array(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream top-level objects from a (possibly multi-GB) JSON-array file.

    GUI-360's ``processed_data/*/training_data.json`` files are single JSON
    lists; ``screen_parsing`` is ~2.5 GB, so we never ``json.load`` the whole
    file. This walks the byte stream tracking brace depth (string-aware) and
    yields one parsed object per top-level ``{...}`` element.
    """
    with open(path, "r") as f:
        depth = 0
        in_str = False
        esc = False
        started = False
        closed = False
        chars: list[str] = []
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            for ch in chunk:
                if not started:
                    if ch in "\ufeff \t\r\n":
                        continue
                    if ch == "[":
                        started = True
                        continue
                    raise ValueError(f"Expected a JSON array in {path}")
                if closed:
                    if not ch.isspace():
                        raise ValueError(f"Trailing data after JSON array in {path}")
                    continue
                if in_str:
                    chars.append(ch)
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    chars.append(ch)
                elif ch == "{":
                    depth += 1
                    chars.append(ch)
                elif ch == "}":
                    depth -= 1
                    if depth < 0:
                        raise ValueError(f"Unbalanced JSON object in {path}")
                    chars.append(ch)
                    if depth == 0:
                        yield json.loads("".join(chars))
                        chars = []
                elif ch == "]" and depth == 0:
                    closed = True
                elif depth == 0 and ch not in ", \t\r\n":
                    raise ValueError(f"Expected a JSON object in {path}")
                elif depth > 0:
                    chars.append(ch)
        if not started or not closed or depth or in_str or chars:
            raise ValueError(f"Truncated JSON array in {path}")


def normalize_coordinate(x: float, y: float, width: int, height: int) -> list[int]:
    """Pixel ``(x, y)`` on a ``width × height`` image → ``[0, 1000]`` integers."""
    return [round(x / width * 1000), round(y / height * 1000)]


# Windows virtual-key tokens that appear in GUI-360 ``type(keys=...)`` strings
# (e.g. ``"{VK_CONTROL}c"``, ``"{ENTER}"``). Mapped directly to Lite canonical
# key tokens. Tokens are matched both with and without the ``VK_`` prefix,
# case-insensitively.
VK_KEY_MAP = {
    "control": "ctrl",
    "lcontrol": "ctrl",
    "rcontrol": "ctrl",
    "shift": "shift",
    "lshift": "shift",
    "rshift": "shift",
    "menu": "alt",
    "lmenu": "alt",
    "rmenu": "alt",
    "lwin": "meta",
    "rwin": "meta",
    "return": "enter",
    "enter": "enter",
    "tab": "tab",
    "escape": "esc",
    "esc": "esc",
    "back": "backspace",
    "delete": "delete",
    "insert": "insert",
    "home": "home",
    "end": "end",
    "prior": "pageup",
    "page_up": "pageup",
    "next": "pagedown",
    "page_down": "pagedown",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "space": "space",
    "capital": "capslock",
    "backspace": "backspace",
    "del": "delete",
    "ctrl": "ctrl",
    "alt": "alt",
    "pgdn": "pagedown",
    "add": "+",
    "oem_plus": "+",
    "oem_minus": "-",
    "oem_period": ".",
    "oem_comma": ",",
    "oem_7": "'",
}
VK_KEY_MAP.update({f"f{i}": f"f{i}" for i in range(1, 13)})
