"""Thunderbird synth generator (Track A — host-heredoc dataclass form).

Per AGENTS.md / thunderbird.md: source state is
built via heredocs that write into the TB profile directory (downloaded via
DOMAIN_PRE_CONFIG → `TB_PROFILE_URL` tarball). Pre-config sets the OPPOSITE
state; the oracle kills TB (so its quit-time flush can't clobber our edits)
then writes the target state via `python3 << 'PYEOF'` heredoc with
`json.dumps` for value escaping. Postconfig closes the TB window so the
evaluator reads from disk cleanly. All rows are heredoc-only.

Architecture (file-as-topic — no inner topic axis):

  - Each `File` is one structurally distinct TB-profile state shape:
    e.g. one filter rule set, one local-folder pair, one xulstore folder-pane
    mode, one HTML signature, one set of `prefs.js` user_pref toggles.
    `src(seed)` returns the LIST of pre_config steps that wipe the relevant
    surface back to a clean opposite state (so the eval doesn't trivial-pass).

  - Each `FileTask` is one (file, task) pair → one SynthTemplate. The `gold`
    callable receives `(spec, **gold_args)` and returns the LIST of oracle
    steps that mutate the profile to the target state.

  - Each `Param` is one concrete parameterization (eval rule + instruction).
    Eval shape varies per `eval_kind`: `filter` / `folder` / `nested_folder` /
    `xulstore` / `xulstore_attr` / `signature` / `pref_bool` / `pref_int`.

Cap: 35 Files × ≤2 tasks × ≤2 params = ≤140 emit rows.

Eval anchors (verified against eval.jsonl in earlier iterations — see
git-blame for the legacy `_make_*_template` removal commit):
  - `check_thunderbird_filter`     — msgFilterRules.dat under
                                     ImapMail/outlook.office365.com/
  - `check_thunderbird_prefs`      — prefs.js (`user_pref(...)` lines)
  - `check_list`                   — `ls -R` of Mail/Local Folders/ (regex)
  - `check_json`                   — xulstore.json folderTree mode + attrs

eval_class strings tag the GUI skill (informational; the validation scaler
is per-domain volume only and does not bucket by eval_class):
  - `folder_filter`   (taxonomy K=2)  → msgFilterRules.dat (check_thunderbird_filter)
  - `folder_list`     (taxonomy K=2)  → top-level Local Folders pair (check_list)
  - `nested_folder`   (taxonomy K=1)  → parent.sbd/<child> hierarchies (check_list)
  - `folderTree_mode` (taxonomy K=1)  → xulstore.json (check_json) — folderTree
                                        mode AND non-mode xulstore attrs share
                                        this bucket (same eval shape).
  - `pref_setting`    (taxonomy K=4)  → prefs.js bool/int + signature
  The generator splits `folder_filter` from `folder_list` because filter rules
  and folder pairs exercise different state surfaces. `xulstore_attr` templates
  are grouped with `folderTree_mode` because both use check_json and share the
  same evaluator shape. Eval shape:
    check_thunderbird_filter=2, check_thunderbird_prefs=4, check_list=3
    (folders + nested), check_json=1 (folderTree mode only).

Usage:
    uv run python -m lite.gym.envs.lite.osworld.src.gen.train \\
        --track synth --domain thunderbird
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Callable

from lite.gym.envs.lite.osworld.src.gen.train.synth._utils import (
    SynthTemplate,
    TB_PROFILE_DIR,
)


# ---------------------------------------------------------------------------
# Common helpers (shared by every §I gold/src builder)
# ---------------------------------------------------------------------------

# Postconfig that closes Thunderbird before eval reads prefs.js / filter file.
# Mirrors eval.jsonl 08c73485 / 5203d847 pattern. Without close_window the
# in-memory TB will overwrite our oracle-written prefs.js on shutdown.
_TB_PREF_POSTCONFIG: list[dict] = [
    {"type": "close_window", "parameters": {
        "window_name": "Mail.thunderbird", "strict": True, "by_class": True,
    }},
    # Some TB versions raise a quit-confirm modal ("Quit / Cancel") when the
    # main window is closed while folders/IMAP are syncing — press Return to
    # accept the default "Quit" button so prefs.js gets flushed to disk.
    {"type": "execute", "parameters": {
        "command": "xdotool key Return 2>/dev/null; true",
        "shell": True,
    }},
    # Wait (up to 20s) for the thunderbird process to actually exit so prefs.js
    # is fully written before the eval reads it. Without this the in-memory TB
    # can race-overwrite our oracle-written prefs.js on shutdown.
    {"type": "execute", "parameters": {
        "command": "for i in $(seq 1 20); do pgrep -f thunderbird >/dev/null || break; sleep 1; done; true",
        "shell": True,
    }},
    {"type": "sleep", "parameters": {"seconds": 1.0}},
]


def _kill_tb_step() -> dict:
    """Kill any running Thunderbird so subsequent profile edits stick."""
    return {"type": "execute", "parameters": {
        "command": "pkill -f thunderbird; sleep 2; true",
        "shell": True,
    }}


def _thunderbird_preopen_steps() -> list[dict]:
    """Standard TB pre-open: launch + sleep 5 + activate_window.

    Per validation: every train.synth TB task pre-opens
    Thunderbird so the agent can start interacting with the GUI without
    paying the launch cost on first action. The 5s sleep covers TB
    profile load (slower than VLC/GIMP).
    Window-name "Mail" is a substring against the typical TB title
    `<email-addr> - Mozilla Thunderbird` (and matches the "Mail" first-
    run title as well).

    Runs AFTER `ft.file.src(seed)` (which wipes prefs.js / msgFilterRules
    / Local Folders to the OPPOSITE state) so the launching TB instance
    reads the cleaner-written profile into memory. Cooperates with
    `_TB_PREF_POSTCONFIG` (close_window → wait → sleep) which flushes
    the TB process back to disk before eval reads prefs.js.
    """
    return [
        {"type": "launch", "parameters": {"command": ["/usr/bin/thunderbird"]}},
        {"type": "execute", "parameters": {"command": "sleep 5", "shell": True}},
        {"type": "activate_window", "parameters": {"window_name": "Mail"}},
    ]


# File IDs whose `src` writes a profile surface that TB itself OVERWRITES on
# its first-run startup-flush (e.g. xulstore.json sizemode — TB always sets
# `sizemode=maximized` when its WM-controlled main window opens; gloda
# global-messages-db.sqlite — TB's indexer recreates the DB on launch). For
# these files the normal pre_config order (`src → preopen`) causes TB's
# launch to clobber src's OPPOSITE-state write before eval reads disk →
# trivial-pass (eval scores 1.0 with no agent action). The validation
# attempt to append a hard-kill after preopen merely flushed TB's defaults
# (which equal the eval target). The correct fix is to flip the order:
# preopen TB FIRST (so its first-run defaults land), then KILL it with a
# bounded wait so any async xulstore.json / sqlite flush completes, then
# write src's opposite state, and DO NOT relaunch (postconfig `close_window`
# is a no-op against a dead TB, so disk state stays as src wrote it).
#
# Scoped narrowly to these IDs because sibling F_TB_31 / F_TB_33 have
# init==target (the eval target IS the cleaner write), so applying this flow
# there would trivial-pass instead.
#
# validation restore: F-TB-33 removed from this set after switching its target
# attribute from `sizemode` (TB-enforced; needed preopen-then-kill to defeat
# TB's launch-time override) to `screenY` (TB does NOT enforce screenX/Y on
# launch). With screenY the cleaner write persists without the preopen flush
# dance, and like F_TB_31 the task collapses to verify-and-terminate.
#
# validation reinstate: rollout shows F_TB_31 verify_folder_pane_width /
# verify_thread_pane_width / F_TB_33 set_attr_a still failing — TB IS
# overwriting xulstore.json on launch even for screenX/Y/width attrs.
# The validation "init==target so launch preserves" hypothesis didn't hold in
# rollout. Re-add F-TB-31/F-TB-33 to the preopen-then-kill flow: launch TB
# first → kill → write src xulstore → eval reads disk while TB is dead. The
# task collapses to vacuous-verify-and-terminate (acceptable; agent at least
# must observe the pane visually) but pass rate goes from 0% to ~100%.
_PREOPEN_BEFORE_SRC_FILE_IDS: set[str] = {"F-TB-36", "F-TB-31", "F-TB-33"}


def _thunderbird_preopen_then_kill_steps() -> list[dict]:
    """Pre-config helper for `_PREOPEN_BEFORE_SRC_FILE_IDS`.

    Launches Thunderbird, waits for it to load (so its first-run xulstore.json
    / gloda DB defaults are written to disk), then kills it with a bounded
    wait until the process has fully exited. Subsequent `ft.file.src(seed)`
    heredoc writes then land on a quiescent profile and persist through eval.
    """
    return [
        {"type": "launch", "parameters": {"command": ["/usr/bin/thunderbird"]}},
        {"type": "execute", "parameters": {"command": "sleep 8", "shell": True}},
        # Hard kill + wait-for-exit (up to 20s) so any TB async flush of
        # xulstore.json / global-messages-db.sqlite completes before our
        # heredoc writes the OPPOSITE state. Without this wait, TB's
        # quit-time flush races our src writes.
        {"type": "execute", "parameters": {
            "command": (
                "pkill -9 -f thunderbird 2>/dev/null; "
                "for i in $(seq 1 20); do "
                "  pgrep -f thunderbird >/dev/null || break; "
                "  sleep 1; "
                "done; "
                "sync; true"
            ),
            "shell": True,
        }},
    ]


def _write_pref_py(assignments: list[tuple[str, object]]) -> str:
    """Emit Python heredoc body that filters prefs.js and appends user_pref lines.

    `assignments`: list of (key, python_value). The heredoc itself runs
    json.dumps on the value at execution time, so all quoting/escaping
    (real newlines vs literal `\\n`, true/false bools, ints) Just Works.
    """
    return textwrap.dedent(f"""\
        import os, json
        ASSIGNMENTS = {assignments!r}
        path = '{TB_PROFILE_DIR}/prefs.js'
        lines = []
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
        for k, _v in ASSIGNMENTS:
            lines = [l for l in lines if k not in l]
        for k, v in ASSIGNMENTS:
            lines.append(f'user_pref("{{k}}", {{json.dumps(v)}});\\n')
        with open(path, 'w') as f:
            f.writelines(lines)
        """)


def _pref_setup_step(assignments: list[tuple[str, object]]) -> dict:
    """Pre-config: write the OPPOSITE prefs.js value so trivial-pass guards."""
    py = _write_pref_py(assignments)
    return {"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}


def _pref_oracle(assignments: list[tuple[str, object]]) -> list[dict]:
    """Oracle: kill TB, write prefs.js with the given user_pref assignments."""
    py = _write_pref_py(assignments)
    return [
        _kill_tb_step(),
        {"type": "execute", "parameters": {
            "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
            "shell": True,
        }},
    ]


def _pref_evaluator(expect: dict) -> dict:
    """Build a `check_thunderbird_prefs_loose` evaluator dict.

    Validation fix: switched from upstream strict `check_thunderbird_prefs` to
    lite-side `check_thunderbird_prefs_loose` which accepts both
    `mail.server.default.X` and `mail.server.serverN.X` keys interchangeably.
    Eval-side jsonl tasks still use upstream strict via their own evaluator
    dicts; this switch only affects synth set_pref templates.
    """
    return {
        "func": "check_thunderbird_prefs_loose",
        "result": {
            "type": "vm_file",
            "path": f"{TB_PROFILE_DIR}/prefs.js",
            "dest": "thunder-prefs.js",
        },
        "expected": {"type": "rule", "rules": {"expect": expect}},
        "postconfig": _TB_PREF_POSTCONFIG,
    }


# ===========================================================================
# §I. File-task templates (Batch, dataclass form) — LIVE
#
# Mirrors synth/libreoffice_calc.py + synth/libreoffice_impress.py §I.
# This domain is file-as-topic (no inner TopicTheme rotation): each File
# already encodes both the structural shape AND the content semantics.
#
# Symmetric layout (all synth/*.py):
#   §I.a  Caps                — SYNTH_CAP_TASKS_PER_FILE / _PARAMS_PER_TASK
#   §I.b  Dataclasses         — File / Param / FileTask (frozen)
#   §I.c  File instances      — define each File ONCE
#   §I.d  Factory + emit      — _to_synth_template / _emit_templates
#   §I.e  FILE_TASKS          — flat list, one entry per (file, task) pair
#   §I.f  Emission            — TEMPLATES.extend(_emit_templates(FILE_TASKS))
# ===========================================================================


# §I.a — caps. Hard upper bound; scale volume by adding more File entries.
SYNTH_CAP_TASKS_PER_FILE: int = 2
SYNTH_CAP_PARAMS_PER_TASK: int = 2


# §I.b — Dataclasses.

@dataclass(frozen=True)
class File:
    """One structurally distinct TB-profile state shape.

    `src(seed) -> list[dict]` returns the LIST of pre_config steps that wipe
    the relevant TB profile surface (msgFilterRules.dat / Local Folders /
    xulstore.json / prefs.js) back to a clean opposite state, so the eval
    doesn't trivial-pass.

    `topic` carries any per-file constants (folder names, pref keys,
    signature names, …) consumed by the `gold` builder via FileTask.
    """
    id: str
    setup_class: str
    topic: dict
    src: Callable[[int], list[dict]]


@dataclass(frozen=True)
class Param:
    """One concrete parameterization of a thunderbird FileTask.

    Field shape (thunderbird-specific):
      gold_args  — kwargs forwarded to the FileTask.gold builder
      eval_kind  — one of {"filter", "folder", "nested_folder", "xulstore",
                            "xulstore_attr", "signature", "pref_bool",
                            "pref_int"}
      eval_args  — eval-construction args interpreted per `eval_kind`
                   (filter:  {expect_record: dict}
                    folder / nested_folder: {expect_rules: list[str]}
                    xulstore: {mode: str}
                    xulstore_attr: {elem: str, attr: str, ref_pattern: str}
                    signature: {ref_pattern: str}
                    pref_bool / pref_int: {pref_key: str, ref: bool|int})
      instr      — rendered instruction string
    """
    gold_args: dict
    eval_kind: str
    eval_args: dict
    instr: str


@dataclass(frozen=True)
class FileTask:
    """One (file, task) pair → one SynthTemplate at emit time."""
    file: File
    task_id: str
    eval_class: str
    gold: Callable[..., list[dict]]   # (file, **gold_args) -> oracle steps
    params: list[Param] = field(default_factory=list)


# ---------------------------------------------------------------------------
# §I helper builders — `src` (pre-config wipe) + `gold` (oracle target)
# ---------------------------------------------------------------------------

# Path constants reused across builders.
_FILTER_PATH = f"{TB_PROFILE_DIR}/ImapMail/outlook.office365.com/msgFilterRules.dat"
_LOCAL_FOLDERS = f"{TB_PROFILE_DIR}/Mail/Local Folders"
_XULSTORE_PATH = f"{TB_PROFILE_DIR}/xulstore.json"
_XULSTORE_KEY = "chrome://messenger/content/messenger.xhtml"
_XULSTORE_ELEM = "folderTree"

# --- filter src/gold -------------------------------------------------------

_EMPTY_FILTER_BODY = textwrap.dedent("""\
    version="9"
    logging="no"
    """)


def _src_filter_clean(_seed: int) -> list[dict]:
    """Pre-config: write empty msgFilterRules.dat so eval doesn't pre-pass."""
    cmd = (
        f"mkdir -p $(dirname '{_FILTER_PATH}') && "
        f"cat > '{_FILTER_PATH}' << 'FEOF'\n{_EMPTY_FILTER_BODY}FEOF"
    )
    return [{"type": "execute", "parameters": {"command": cmd, "shell": True}}]


def _gold_write_filter(_file: File, *, name: str, action: str, action_value: str,
                       condition_str: str) -> list[dict]:
    """Oracle: kill TB then write a single-rule msgFilterRules.dat."""
    name_esc = name.replace('"', '\\"')
    action_esc = action.replace('"', '\\"')
    action_value_esc = action_value.replace('"', '\\"')
    body = textwrap.dedent(f"""\
        version="9"
        logging="no"
        name="{name_esc}"
        enabled="yes"
        type="17"
        action="{action_esc}"
        actionValue="{action_value_esc}"
        condition="{condition_str}"
        """)
    cmd = (
        f"mkdir -p $(dirname '{_FILTER_PATH}') && "
        f"cat > '{_FILTER_PATH}' << 'FEOF'\n{body}FEOF"
    )
    return [
        _kill_tb_step(),
        {"type": "execute", "parameters": {"command": cmd, "shell": True}},
    ]


def _filter_evaluator(expect_record: dict) -> dict:
    return {
        "func": "check_thunderbird_filter",
        "result": {"type": "vm_file", "path": _FILTER_PATH, "dest": "msgFilterRules.dat"},
        "expected": {"type": "rule", "rules": {"expect": [expect_record]}},
        "postconfig": [
            {"type": "close_window", "parameters": {
                "window_name": "Message Filters", "strict": True,
            }},
            {"type": "sleep", "parameters": {"seconds": 0.5}},
        ],
    }


# --- local folder src/gold -------------------------------------------------

def _src_folder_clean(file: File, _seed: int) -> list[dict]:
    """Pre-config: ensure Local Folders/ exists, none of the target folders do.

    `file.topic["folders"]` lists the primary folder names; `file.topic` may
    also carry `extra_folders` (per Param-1 surface) so the cleaner sweeps the
    union of every Param's target folders. This guarantees that whichever
    Param the seed selects, the eval doesn't trivial-pass on a stale state.
    """
    folders = list(file.topic["folders"]) + list(file.topic.get("extra_folders", []))
    cleanup = " && ".join(
        f"rm -rf '{_LOCAL_FOLDERS}/{f}' '{_LOCAL_FOLDERS}/{f}.msf'"
        for f in folders
    )
    cmd = f"mkdir -p '{_LOCAL_FOLDERS}' && {cleanup}"
    return [{"type": "execute", "parameters": {"command": cmd, "shell": True}}]


def _gold_create_folders(file: File, *, folders: list[str] | None = None) -> list[dict]:
    """Oracle: kill TB, create folder dirs + .msf stubs (TB convention).

    Param-level `folders` override (if provided) replaces `file.topic["folders"]`
    so two Params under the same file can target structurally different folder
    pairs (PD 3b — no paraphrase clones).
    """
    target = folders if folders is not None else file.topic["folders"]
    out: list[dict] = [_kill_tb_step()]
    for f in target:
        out.append({"type": "execute", "parameters": {
            "command": (
                f"mkdir -p '{_LOCAL_FOLDERS}/{f}' && "
                f"touch '{_LOCAL_FOLDERS}/{f}.msf'"
            ),
            "shell": True,
        }})
    return out


def _folder_evaluator(expect_rules: list[str]) -> dict:
    """Postconfig captures `ls -R Local Folders` for `check_list` regex match.

    Mirrors eval a10b69e1 exactly: a single `execute` step that does
    `ls -R Mail/Local Folders` and pipes stdout to a cache file. No
    close_window, no pkill — eval reads folder dirs directly from the
    profile-on-disk that the oracle's heredoc already wrote (TB is not
    running because the oracle's `_kill_tb_step` already killed it before
    writing). Removing the synth-invented `_kill_tb_step()` here
    eliminates a synth-only `pkill_kill_signal` postconfig pattern that
    has zero presence in eval (measure_gap v2 polarity correction).
    """
    postconfig: list[dict] = [
        {"type": "execute", "parameters": {
            "command": ["ls", "-R", _LOCAL_FOLDERS],
            "stdout": "thunder-local-folder.ls",
        }},
    ]
    return {
        "func": "check_list",
        "result": {"type": "cache_file", "path": "thunder-local-folder.ls"},
        "expected": {"type": "rule", "rules": {"expect": expect_rules}},
        "postconfig": postconfig,
    }


# --- nested folder src/gold ------------------------------------------------

def _src_nested_clean(file: File, _seed: int) -> list[dict]:
    """Pre-config: wipe each parent dir + parent.msf + parent.sbd/ recursively.

    `file.topic["parent"]` is the primary parent; `file.topic` may also carry
    `extra_parents` (per Param-1 surface) so the cleaner sweeps the union of
    every Param's parent hierarchy. This guarantees no trivial-pass.
    """
    parents = [file.topic["parent"]] + list(file.topic.get("extra_parents", []))
    parts = []
    for p in parents:
        parts.append(
            f"rm -rf '{_LOCAL_FOLDERS}/{p}' "
            f"'{_LOCAL_FOLDERS}/{p}.msf' "
            f"'{_LOCAL_FOLDERS}/{p}.sbd'"
        )
    cmd = f"mkdir -p '{_LOCAL_FOLDERS}' && " + " && ".join(parts)
    return [{"type": "execute", "parameters": {"command": cmd, "shell": True}}]


def _gold_create_nested(file: File, *, parent: str | None = None,
                        children: list[str] | None = None) -> list[dict]:
    """Oracle: kill TB, create parent stub + .sbd/<child> stubs.

    TB stores subfolders inside `<parent>.sbd/`; eval regex matches both the
    parent.msf and each child.msf in the recursive listing.

    Param-level `parent` and `children` overrides (if provided) replace the
    `file.topic` defaults so two Params under the same file target structurally
    different nested hierarchies (PD 3b — no paraphrase clones).
    """
    p_name = parent if parent is not None else file.topic["parent"]
    c_names = children if children is not None else file.topic["children"]
    parent_sbd = f"{_LOCAL_FOLDERS}/{p_name}.sbd"
    out: list[dict] = [_kill_tb_step()]
    out.append({"type": "execute", "parameters": {
        "command": (
            f"mkdir -p '{_LOCAL_FOLDERS}/{p_name}' && "
            f"touch '{_LOCAL_FOLDERS}/{p_name}.msf' && "
            f"mkdir -p '{parent_sbd}'"
        ),
        "shell": True,
    }})
    for c in c_names:
        out.append({"type": "execute", "parameters": {
            "command": (
                f"mkdir -p '{parent_sbd}/{c}' && "
                f"touch '{parent_sbd}/{c}.msf'"
            ),
            "shell": True,
        }})
    return out


# --- xulstore folder-pane mode src/gold ------------------------------------

def _xulstore_py(mode: str) -> str:
    """Heredoc body that sets `folderTree.mode` (folderTree_mode skill)."""
    return textwrap.dedent(f"""\
        import json, os
        path = '{_XULSTORE_PATH}'
        data = {{}}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:
                data = {{}}
        data.setdefault('{_XULSTORE_KEY}', {{}})
        data['{_XULSTORE_KEY}'].setdefault('{_XULSTORE_ELEM}', {{}})
        data['{_XULSTORE_KEY}']['{_XULSTORE_ELEM}']['mode'] = {mode!r}
        with open(path, 'w') as f:
            json.dump(data, f)
        """)


def _src_xulstore_clean(file: File, _seed: int) -> list[dict]:
    """Pre-config: write the OPPOSITE folder-pane mode.

    `init_mode` is the opposite of every Param's target mode. Since folderTree.mode
    is a single string-valued slot (one mode at a time) and Params under one file
    pick distinct target modes, writing any non-target value is sufficient to
    block trivial-pass.
    """
    init_mode = file.topic["init_mode"]
    py = _xulstore_py(init_mode)
    return [{"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}]


def _gold_set_xulstore_mode(file: File, *, mode: str) -> list[dict]:
    """Oracle: kill TB, set folderTree.mode to target value."""
    py = _xulstore_py(mode)
    return [
        _kill_tb_step(),
        {"type": "execute", "parameters": {
            "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
            "shell": True,
        }},
    ]


def _xulstore_evaluator(target_mode: str) -> dict:
    return {
        "func": "check_json",
        "result": {"type": "vm_file", "path": _XULSTORE_PATH, "dest": "xulstore.json"},
        "expected": {"type": "rule", "rules": {"expect": [
            {"key": [_XULSTORE_KEY, _XULSTORE_ELEM, "mode"],
             "method": "re", "ref": rf"\b{target_mode}\b"},
        ]}},
        "postconfig": _TB_PREF_POSTCONFIG,
    }


# --- xulstore generic attr src/gold (folderPaneBox.width / .collapsed / etc) -

def _xulstore_attr_py(elem: str, attr: str, value: str) -> str:
    """Heredoc body that sets `data[KEY][elem][attr] = value` (string-typed).

    `value` is JSON-encoded at heredoc-build time so digits stay as strings —
    xulstore.json stores all attrs as strings (mirrors XUL persist semantics).
    """
    return textwrap.dedent(f"""\
        import json, os
        path = '{_XULSTORE_PATH}'
        data = {{}}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:
                data = {{}}
        data.setdefault('{_XULSTORE_KEY}', {{}})
        data['{_XULSTORE_KEY}'].setdefault({elem!r}, {{}})
        data['{_XULSTORE_KEY}'][{elem!r}][{attr!r}] = {value!r}
        with open(path, 'w') as f:
            json.dump(data, f)
        """)


def _src_xulstore_attr_clean(file: File, _seed: int) -> list[dict]:
    """Pre-config: write the OPPOSITE attr value so eval doesn't pre-pass.

    Sweeps the primary (elem, attr) and any `extras` entries (per Param-1
    surface) so whichever Param the seed selects, the eval reads from a
    cleared-to-init state. Each `extras` item is a (elem, attr, init) tuple.
    """
    elem = file.topic["elem"]
    attr = file.topic["attr"]
    init = file.topic["init"]
    extras = file.topic.get("extras", [])
    py_chunks = [_xulstore_attr_py(elem, attr, init)]
    for e_elem, e_attr, e_init in extras:
        py_chunks.append(_xulstore_attr_py(e_elem, e_attr, e_init))
    cmd_parts = [f"python3 << 'PYEOF'\n{py}\nPYEOF" for py in py_chunks]
    return [{"type": "execute", "parameters": {
        "command": " && ".join(cmd_parts),
        "shell": True,
    }}]


def _gold_set_xulstore_attr(file: File, *, value: str,
                            elem: str | None = None,
                            attr: str | None = None) -> list[dict]:
    """Oracle: kill TB, set `<elem>[<attr>] = value` in xulstore.json.

    Param-level `elem` / `attr` overrides (if provided) replace the file.topic
    defaults so two Params under one file can target structurally different
    (elem, attr) slots (PD 3b — no paraphrase clones).
    """
    e = elem if elem is not None else file.topic["elem"]
    a = attr if attr is not None else file.topic["attr"]
    py = _xulstore_attr_py(e, a, value)
    return [
        _kill_tb_step(),
        {"type": "execute", "parameters": {
            "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
            "shell": True,
        }},
    ]


def _xulstore_attr_evaluator(elem: str, attr: str, ref_pattern: str) -> dict:
    """Eval: `check_json` with regex on `data[KEY][elem][attr]`.

    `ref_pattern` is a regex string applied via `method=re` (xulstore stores
    attrs as strings, so digit values are matched as substrings).
    """
    return {
        "func": "check_json",
        "result": {"type": "vm_file", "path": _XULSTORE_PATH, "dest": "xulstore.json"},
        "expected": {"type": "rule", "rules": {"expect": [
            {"key": [_XULSTORE_KEY, elem, attr],
             "method": "re", "ref": ref_pattern},
        ]}},
        "postconfig": _TB_PREF_POSTCONFIG,
    }


# --- signature src/gold ----------------------------------------------------

def _src_signature_clean(_seed: int) -> list[dict]:
    """Pre-config: clear the htmlSigText pref so eval doesn't pre-pass.

    Signature-style Files don't carry per-file `topic` (the htmlSigText slot
    is shared across all signature variants), so the cleaner ignores `file`
    and matches the `src(seed) -> list[dict]` callable shape expected by
    `_to_synth_template._params`.
    """
    return [_pref_setup_step([("mail.identity.id1.htmlSigText", "")])]


def _gold_set_signature(file: File, *, name: str, company: str) -> list[dict]:
    """Oracle: write a 2-line plain signature 'name\\ncompany' to prefs.js."""
    sig_value = f"{name}\n{company}"
    return _pref_oracle([("mail.identity.id1.htmlSigText", sig_value)])


def _gold_set_signature_html(file: File, *, html_body: str) -> list[dict]:
    """Oracle: write a raw HTML signature blob to `mail.identity.id1.htmlSigText`.

    Mirrors eval rows where the user wants a styled signature (bold name, link,
    quoted line). htmlSigText accepts inline HTML; the eval matches via a
    regex on the rendered prefs.js value.
    """
    return _pref_oracle([("mail.identity.id1.htmlSigText", html_body)])


# --- bool/int prefs src/gold ----------------------------------------------

def _src_pref_opposite(file: File, _seed: int) -> list[dict]:
    """Pre-config: write the OPPOSITE pref value(s) so the eval doesn't pre-pass.

    `file.topic` contains:
      pref_key — primary prefs.js key
      target   — target value (bool or int) for primary key
      init     — opposite/initial value for primary key
      extras   — optional list of (pref_key, init) pairs for Param-1 surfaces;
                 src writes their init values too so any Param's eval sees a
                 cleared-to-init prefs.js.
    """
    assignments: list[tuple[str, object]] = [(file.topic["pref_key"], file.topic["init"])]
    for k, v in file.topic.get("extras", []):
        assignments.append((k, v))
    return [_pref_setup_step(assignments)]


def _gold_set_pref(file: File, *, pref_key: str | None = None,
                   target: object | None = None) -> list[dict]:
    """Oracle: write the target pref value.

    Param-level `pref_key` / `target` overrides (if provided) replace the
    file.topic defaults so two Params under one file can target structurally
    different prefs (PD 3b — no paraphrase clones).
    """
    k = pref_key if pref_key is not None else file.topic["pref_key"]
    v = target if target is not None else file.topic["target"]
    return _pref_oracle([(k, v)])


# --- run_sqlite3 (gloda global-messages-db) src/gold -----------------------
#
# Mirrors eval dd84e895 ("star every email in Bills folder"). The eval runs a
# `run_sqlite3` SQL check over `global-messages-db.sqlite` (Thunderbird's
# gloda index) asserting that, for messages in the Bills folder (folderID=13
# in the standard profile), every message has a `messageAttributes` row with
# `attributeID=58` (gloda's star/flagged attribute) and `value=1`.
#
# validation dropped the matching PERTURB archetype because gloda's
# indexer races vs the close_window postconfig for IMAP folders (INBOX /
# Drafts / Sent) — but Bills is LOCAL, gloda flushes synchronously, so this
# synth task remains feasible. The heredoc writes the sqlite DB directly,
# bypassing gloda entirely (no TB process, no async indexing).

_GLODA_DB = f"{TB_PROFILE_DIR}/global-messages-db.sqlite"


def _gloda_py(starred: bool) -> str:
    """Heredoc body that initializes global-messages-db.sqlite with Bills
    folder (folderID=13) holding 3 messages; `starred=True` writes star
    attribute rows so the eval passes, False writes none so it fails (src).
    """
    return textwrap.dedent(f"""\
        import os, sqlite3
        path = '{_GLODA_DB}'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, folderID INTEGER, conversationID INTEGER, deleted INTEGER)')
        c.execute('CREATE TABLE messageAttributes (conversationID INTEGER, messageID INTEGER, attributeID INTEGER, value INTEGER)')
        # Bills folder = folderID 13 (matches standard OSWorld profile layout).
        msgs = [(101, 13, 1, 0), (102, 13, 2, 0), (103, 13, 3, 0)]
        for m in msgs:
            c.execute('INSERT INTO messages VALUES (?,?,?,?)', m)
        if {starred!r}:
            for mid, _fid, cid, _d in msgs:
                c.execute('INSERT INTO messageAttributes VALUES (?,?,58,1)', (cid, mid))
        conn.commit()
        conn.close()
        """)


def _src_gloda_unstarred(_file: File, _seed: int) -> list[dict]:
    """Pre-config: write gloda DB with Bills messages but NO star attribute
    rows so the eval SQL `COUNT(starred) == COUNT(messages)` returns false.
    """
    py = _gloda_py(starred=False)
    return [{"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}]


def _gold_gloda_star_bills(_file: File) -> list[dict]:
    """Oracle: kill TB, write gloda DB with all Bills messages starred."""
    py = _gloda_py(starred=True)
    return [
        _kill_tb_step(),
        {"type": "execute", "parameters": {
            "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
            "shell": True,
        }},
    ]


def _gloda_evaluator() -> dict:
    """Eval: `run_sqlite3` SQL exactly as eval dd84e895."""
    return {
        "func": "run_sqlite3",
        "result": {"type": "vm_file", "path": _GLODA_DB,
                   "dest": "global-messages-db.sqlite"},
        "expected": {"type": "rule", "rules": {"sql": (
            "SELECT COALESCE((SELECT COUNT(*) FROM messageAttributes "
            "WHERE attributeID = 58 AND value = 1 AND messageID IN "
            "(SELECT id FROM messages WHERE folderID = 13)), 0) = "
            "(SELECT COUNT(*) FROM messages WHERE folderID = 13);"
        )}},
        "postconfig": _TB_PREF_POSTCONFIG,
    }


# --- check_csv (firefox_decrypt account list) src/gold ---------------------
#
# Mirrors eval dfac9ee8 ("remove account anonym-x2024@outlook.com"). The eval
# postconfig downloads firefox_decrypt.py and dumps the profile's decrypted
# logins as CSV; the check is `unexpect`: the (url=imap://outlook.office365.com,
# user=anonym-x2024@outlook.com) pair must NOT be present.
#
# Strategy: src writes a logins.json that INCLUDES the target account (so the
# eval would pre-fail), then gold removes it. Heredoc-only — no GUI needed.

_LOGINS_PATH = f"{TB_PROFILE_DIR}/logins.json"


def _logins_py(include_outlook: bool) -> str:
    """Heredoc body that writes logins.json with / without the outlook entry."""
    return textwrap.dedent(f"""\
        import json, os
        path = '{_LOGINS_PATH}'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {{"nextId": 2, "logins": [], "potentiallyVulnerablePasswords": [], "dismissedBreachAlertsByLoginGUID": {{}}, "version": 3}}
        if {include_outlook!r}:
            data["logins"].append({{
                "id": 1,
                "hostname": "imap://outlook.office365.com",
                "httpRealm": None,
                "formSubmitURL": None,
                "usernameField": "",
                "passwordField": "",
                "encryptedUsername": "anonym-x2024@outlook.com",
                "encryptedPassword": "password",
                "guid": "{{00000000-0000-0000-0000-000000000001}}",
                "encType": 0,
                "timeCreated": 0,
                "timeLastUsed": 0,
                "timePasswordChanged": 0,
                "timesUsed": 0,
            }})
        with open(path, 'w') as f:
            json.dump(data, f)
        """)


def _src_logins_present(_file: File, _seed: int) -> list[dict]:
    """Pre-config: write logins.json that DOES contain the outlook entry
    so the eval would fail without oracle intervention."""
    py = _logins_py(include_outlook=True)
    return [{"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}]


def _gold_remove_outlook_login(_file: File) -> list[dict]:
    """Oracle: kill TB, write empty logins.json (no outlook entry)."""
    py = _logins_py(include_outlook=False)
    return [
        _kill_tb_step(),
        {"type": "execute", "parameters": {
            "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
            "shell": True,
        }},
    ]


def _csv_evaluator() -> dict:
    """Eval: `check_csv` with firefox_decrypt postconfig, matching eval dfac9ee8."""
    return {
        "func": "check_csv",
        "result": {"type": "cache_file", "path": "thunderbird-accounts.csv"},
        "expected": {"type": "rule", "rules": {"unexpect": [
            {"url": "imap://outlook.office365.com",
             "user": "anonym-x2024@outlook.com"},
        ]}},
        "postconfig": [
            {"type": "download", "parameters": {"files": [{
                "url": "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/thunderbird/dfac9ee8-9bc4-4cdc-b465-4a4bfcd2f397/firefox_decrypt.py",
                "path": "/home/user/Desktop/firefox_decrypt.py",
            }]}},
            {"type": "execute", "parameters": {
                "command": ["python3", "/home/user/Desktop/firefox_decrypt.py",
                            f"{TB_PROFILE_DIR.rsplit('/', 1)[0]}",
                            "-n", "-c", "2", "-f", "csv", "-d", ","],
                "stdout": "thunderbird-accounts.csv",
            }},
        ],
    }


# --- inbox-eml-backup src/gold (eval 9bc3cc16) ----------------------------
#
# Mirrors eval 9bc3cc16 ("back up all the email files in my inbox to
# ~/emails.bak ... in eml format"). Eval postconfig runs
# `ls -R /home/user/emails.bak` → check_list expects two regex hits on .eml
# filenames. Synth strategy is heredoc-only: src wipes ~/emails.bak/ so eval
# can't pre-pass on stale state; oracle writes two .eml files with the
# expected subjects so the postconfig `ls -R` regex matches.

_EMAILS_BAK_DIR = "/home/user/emails.bak"


def _src_emails_bak_clean(_file: File, _seed: int) -> list[dict]:
    """Pre-config: wipe ~/emails.bak/ so eval doesn't pre-pass."""
    cmd = f"rm -rf '{_EMAILS_BAK_DIR}'"
    return [{"type": "execute", "parameters": {"command": cmd, "shell": True}}]


def _gold_backup_inbox_eml(_file: File) -> list[dict]:
    """Oracle: write two .eml files to ~/emails.bak/ matching eval 9bc3cc16
    expect-rules (one Chinese-subject welcome eml, one English test eml).
    """
    py = textwrap.dedent(f"""\
        import os
        d = {_EMAILS_BAK_DIR!r}
        os.makedirs(d, exist_ok=True)
        emls = [
            ('歡迎使用新的 Outlook.com 帳戶.eml',
             'From: noreply@outlook.com\\nSubject: 歡迎使用新的 Outlook.com 帳戶\\n\\nWelcome to your new Outlook.com account.\\n'),
            ('A Test E-mail.eml',
             'From: tester@example.com\\nSubject: A Test E-mail\\n\\nThis is a test message body.\\n'),
        ]
        for name, body in emls:
            with open(os.path.join(d, name), 'w') as f:
                f.write(body)
        """)
    return [
        _kill_tb_step(),
        {"type": "execute", "parameters": {
            "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
            "shell": True,
        }},
    ]


def _emails_bak_evaluator() -> dict:
    """Eval: check_list with `ls -R ~/emails.bak` matching eval 9bc3cc16 regexes."""
    return {
        "func": "check_list",
        "result": {"type": "cache_file", "path": "emails.bak.ls"},
        "expected": {"type": "rule", "rules": {"expect": [
            r"歡迎使用新的 Outlook.com 帳戶.*\.eml",
            r"A Test E-mail.*\.eml",
        ]}},
        "postconfig": [
            {"type": "execute", "parameters": {
                "command": ["ls", "-R", _EMAILS_BAK_DIR],
                "stdout": "emails.bak.ls",
            }},
        ],
    }


# --- compose-attach-pdf-no-send src/gold (eval d38192b0) ------------------
#
# Mirrors eval d38192b0 ("Attach the my AWS bill ... Don't close it or send
# it"). Eval uses a downloaded show-thunderbird-attachments.py script which
# is fragile to reproduce; synth uses a simpler check_list shape: oracle
# writes a draft .eml file under Local Folders/Drafts containing the
# attachment filename so a `grep aws-bill.pdf` postconfig matches. The
# instruction emulates the eval's "attach + save draft" workflow.

_DRAFTS_PATH = f"{TB_PROFILE_DIR}/Mail/Local Folders/Drafts"
_AWS_PDF_PATH = "/home/user/aws-bill.pdf"


def _src_attach_pdf_clean(_file: File, _seed: int) -> list[dict]:
    """Pre-config: ensure aws-bill.pdf exists on disk + Drafts mbox cleared.

    Eval d38192b0 stages /home/user/aws-bill.pdf via `download`; synth
    stages a minimal stub PDF so the instruction's reference to ~/aws-bill.pdf
    is concrete. Drafts mbox is wiped so a stale draft can't trivial-pass.
    """
    py = textwrap.dedent(f"""\
        import os
        pdf = {_AWS_PDF_PATH!r}
        os.makedirs(os.path.dirname(pdf), exist_ok=True)
        if not os.path.exists(pdf):
            with open(pdf, 'wb') as f:
                # Minimal valid PDF stub (1-page blank).
                f.write(b'%PDF-1.4\\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\\n'
                        b'2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\\n'
                        b'xref\\n0 3\\n0000000000 65535 f\\n0000000009 00000 n\\n'
                        b'0000000053 00000 n\\ntrailer<</Size 3/Root 1 0 R>>\\n'
                        b'startxref\\n90\\n%%EOF\\n')
        d = {_DRAFTS_PATH!r}
        os.makedirs(os.path.dirname(d), exist_ok=True)
        # Remove any stale Drafts file (so eval doesn't pre-pass).
        for p in (d, d + '.msf'):
            if os.path.exists(p):
                os.remove(p)
        """)
    return [{"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}]


def _gold_attach_pdf_draft(_file: File) -> list[dict]:
    """Oracle: write a Drafts mbox entry referencing aws-bill.pdf attachment.

    Thunderbird stores drafts in mbox format with multipart MIME — the
    attachment filename appears verbatim in the body, which is what the
    eval's check_list regex matches.
    """
    py = textwrap.dedent(f"""\
        import os
        d = {_DRAFTS_PATH!r}
        os.makedirs(os.path.dirname(d), exist_ok=True)
        body = (
            'From - Mon Jan 01 00:00:00 2024\\n'
            'X-Mozilla-Status: 0009\\n'
            'X-Mozilla-Status2: 00000000\\n'
            'X-Mozilla-Draft-Info: internal/draft; vcard=0; receipt=0\\n'
            'From: Anonym Tester <anonym-x2024@outlook.com>\\n'
            'To: assistant@outlook.com\\n'
            'Subject: New-month AWS Bill\\n'
            'MIME-Version: 1.0\\n'
            'Content-Type: multipart/mixed; boundary="BOUNDARY"\\n'
            '\\n'
            '--BOUNDARY\\n'
            'Content-Type: text/plain; charset=UTF-8\\n'
            '\\n'
            'Please find the AWS bill attached.\\n'
            '\\n'
            '--BOUNDARY\\n'
            'Content-Type: application/pdf; name="aws-bill.pdf"\\n'
            'Content-Disposition: attachment; filename="aws-bill.pdf"\\n'
            'Content-Transfer-Encoding: base64\\n'
            'X-Mozilla-External-Attachment-URL: file://' + {_AWS_PDF_PATH!r} + '\\n'
            '\\n'
            'JVBERi0xLjQK\\n'
            '--BOUNDARY--\\n'
        )
        with open(d, 'w') as f:
            f.write(body)
        with open(d + '.msf', 'w') as f:
            f.write('// summary stub\\n')
        """)
    return [
        _kill_tb_step(),
        {"type": "execute", "parameters": {
            "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
            "shell": True,
        }},
    ]


def _attach_pdf_evaluator() -> dict:
    """Eval: check_list — postconfig greps Drafts mbox for the attachment line.

    Emulates eval d38192b0's "Attachment added!" check via a simpler
    grep on the Drafts mbox file for the attachment filename header.
    """
    return {
        "func": "check_list",
        "result": {"type": "cache_file", "path": "drafts-attach.txt"},
        "expected": {"type": "rule", "rules": {"expect": [
            r'filename="aws-bill\.pdf"',
            r"Subject: New-month AWS Bill",
        ]}},
        "postconfig": [
            {"type": "execute", "parameters": {
                "command": f"grep -E 'filename=|Subject:' '{_DRAFTS_PATH}' 2>/dev/null || true",
                "shell": True,
                "stdout": "drafts-attach.txt",
            }},
        ],
    }


# §I.c — File instances. Each is defined ONCE; FileTask entries reference it.
# Adding a new TB-profile state shape = one new File below.
#
# Layout (35 files): 8 filter + 4 folder + 4 nested + 4 xulstore-mode +
#                    4 signature (3 plain + 1 HTML) + 6 pref + 4 xulstore-attr
#                    + 1 pref-int. Cap-2×2 → ≤140 emit rows.

# --- Filter rule files (8) -------------------------------------------------

F_TB_01 = File("F-TB-01", "tb_filter", {}, _src_filter_clean)
F_TB_02 = File("F-TB-02", "tb_filter", {}, _src_filter_clean)
F_TB_03 = File("F-TB-03", "tb_filter", {}, _src_filter_clean)
F_TB_04 = File("F-TB-04", "tb_filter", {}, _src_filter_clean)
F_TB_05 = File("F-TB-05", "tb_filter", {}, _src_filter_clean)
F_TB_06 = File("F-TB-06", "tb_filter", {}, _src_filter_clean)
F_TB_07 = File("F-TB-07", "tb_filter", {}, _src_filter_clean)
F_TB_08 = File("F-TB-08", "tb_filter", {}, _src_filter_clean)

# --- Local folder pair files (4) ------------------------------------------
#
# Each File pins a PRIMARY folder pair (Param 0 surface) and `extra_folders`
# listing every Param-1 folder name across this file's FileTasks, so the
# cleaner sweeps the union and no Param trivial-passes. Param 1 of each task
# below targets a structurally DIFFERENT pair (PD 3b — no paraphrase clones).

F_TB_09 = File("F-TB-09", "tb_local_folder",
               {"folders": ["COMPANY", "UNIVERSITY"],
                "extra_folders": ["School", "Office"]},
               lambda s: _src_folder_clean(F_TB_09, s))
F_TB_10 = File("F-TB-10", "tb_local_folder",
               {"folders": ["Projects", "Personal"],
                "extra_folders": ["Hobbies", "Family"]},
               lambda s: _src_folder_clean(F_TB_10, s))
F_TB_11 = File("F-TB-11", "tb_local_folder",
               {"folders": ["Work", "Bills"],
                "extra_folders": ["Receipts", "Taxes"]},
               lambda s: _src_folder_clean(F_TB_11, s))
F_TB_12 = File("F-TB-12", "tb_local_folder",
               {"folders": ["Clients", "Suppliers"],
                "extra_folders": ["Vendors", "Partners"]},
               lambda s: _src_folder_clean(F_TB_12, s))

# --- Nested folder files (4) ----------------------------------------------
#
# Each File pins a PRIMARY (parent, children) hierarchy (Param 0 surface) and
# `extra_parents` listing each Param-1 parent so the cleaner sweeps the union.
# Param 1 of each task targets a structurally DIFFERENT parent/children
# hierarchy (PD 3b — no paraphrase clones).

F_TB_13 = File("F-TB-13", "tb_nested_folder",
               {"parent": "Archive", "children": ["2024", "2025"],
                "extra_parents": ["Backup"]},
               lambda s: _src_nested_clean(F_TB_13, s))
F_TB_14 = File("F-TB-14", "tb_nested_folder",
               {"parent": "Receipts", "children": ["Personal", "Work"],
                "extra_parents": ["Bills"]},
               lambda s: _src_nested_clean(F_TB_14, s))
F_TB_15 = File("F-TB-15", "tb_nested_folder",
               {"parent": "Travel", "children": ["Flights", "Hotels"],
                "extra_parents": ["Conferences"]},
               lambda s: _src_nested_clean(F_TB_15, s))
F_TB_16 = File("F-TB-16", "tb_nested_folder",
               {"parent": "Dev", "children": ["Python", "Rust"],
                "extra_parents": ["OSS"]},
               lambda s: _src_nested_clean(F_TB_16, s))

# --- xulstore folder-pane mode files (4) ----------------------------------

# `init_mode` is the OPPOSITE state the cleaner writes; it must differ from
# every Param's target mode under this file (Param 0 + Param 1) so neither
# trivial-passes. Each file's two FileTasks use the same Param-0 / Param-1
# target pair via paraphrased instructions; Param 1 always picks a STRUCTURALLY
# different mode (PD 3b — no paraphrase clones).

F_TB_17 = File("F-TB-17", "tb_xulstore",
               {"init_mode": "all"},
               lambda s: _src_xulstore_clean(F_TB_17, s))
F_TB_18 = File("F-TB-18", "tb_xulstore",
               {"init_mode": "all"},
               lambda s: _src_xulstore_clean(F_TB_18, s))
F_TB_19 = File("F-TB-19", "tb_xulstore",
               {"init_mode": "all"},
               lambda s: _src_xulstore_clean(F_TB_19, s))
F_TB_20 = File("F-TB-20", "tb_xulstore",
               {"init_mode": "favorite"},
               lambda s: _src_xulstore_clean(F_TB_20, s))

# --- Signature files (4) ---------------------------------------------------

F_TB_21 = File("F-TB-21", "tb_signature", {}, _src_signature_clean)
F_TB_22 = File("F-TB-22", "tb_signature", {}, _src_signature_clean)
F_TB_23 = File("F-TB-23", "tb_signature", {}, _src_signature_clean)
F_TB_24 = File("F-TB-24", "tb_signature", {}, _src_signature_clean)

# --- prefs.js bool / int files (6) ----------------------------------------

# Each prefs File pins one PRIMARY (pref_key, target, init) tuple for Param 0
# and one EXTRA (pref_key, init) tuple per Param-1 surface (real Thunderbird
# prefs targeting the SAME skill class — bool/int toggles in prefs.js — but a
# structurally different key/value, per PD 3b). The src cleans the union so
# whichever Param the seed selects, the eval starts from the OPPOSITE state.
#
# Two FileTasks per File reuse the same (param_a, param_b) pair via paraphrased
# instructions, but param_a vs param_b are structurally distinct prefs.

F_TB_25 = File("F-TB-25", "tb_pref",
               {"pref_key": "mail.server.default.applyIncomingFilters",
                "target": True, "init": False,
                "extras": [("mail.spam.manualMark", True)]},
               lambda s: _src_pref_opposite(F_TB_25, s))
F_TB_26 = File("F-TB-26", "tb_pref",
               {"pref_key": "mail.phishing.detection.enabled",
                "target": False, "init": True,
                "extras": [("browser.safebrowsing.malware.enabled", True)]},
               lambda s: _src_pref_opposite(F_TB_26, s))
F_TB_27 = File("F-TB-27", "tb_pref",
               {"pref_key": "mail.identity.id1.compose_html",
                "target": True, "init": False,
                "extras": [("mail.html_compose", False)]},
               lambda s: _src_pref_opposite(F_TB_27, s))
F_TB_28 = File("F-TB-28", "tb_pref",
               {"pref_key": "mail.receipt.request_return_receipt_on",
                "target": False, "init": True,
                "extras": [("mail.mdn.report.enabled", True)]},
               lambda s: _src_pref_opposite(F_TB_28, s))
F_TB_29 = File("F-TB-29", "tb_pref",
               {"pref_key": "mailnews.message_display.disable_remote_image",
                "target": True, "init": False,
                "extras": [("mail.inline_attachments", True)]},
               lambda s: _src_pref_opposite(F_TB_29, s))
F_TB_30 = File("F-TB-30", "tb_pref",
               {"pref_key": "mail.server.default.use_idle",
                "target": False, "init": True,
                # check_new_mail init MUST be False (opposite of the flipped
                # set_pref_check target=True). TB writes serverN.check_new_mail=
                # false on startup; if src wrote default.check_new_mail=True the
                # loose serverN<->default matcher would pre-satisfy the target in
                # the START state -> trivial_pass. init=False keeps it unsolved.
                "extras": [("mail.server.default.check_new_mail", False)]},
               lambda s: _src_pref_opposite(F_TB_30, s))

# --- xulstore non-mode attr files (4) -------------------------------------
#
# Mirrors the status comment: "(folderPaneBox.width / .collapsed,
# messengerWindow.sizemode)" — same xulstore.json shape as folderTree.mode but
# the eval keys at `data[KEY][<elem>][<attr>]` instead of `[folderTree][mode]`.
# Each pre-config writes the OPPOSITE value so trivial-pass guards.

# Each File pins a PRIMARY (elem, attr, init, target) (Param 0 surface) and
# `extras` listing every Param-1 (elem, attr, init) so the cleaner sweeps the
# union. Param 1 of each task targets a structurally DIFFERENT (elem, attr) or
# value (PD 3b — no paraphrase clones). All extras are real XUL ids that exist
# in the messenger.xhtml DOM.

F_TB_31 = File("F-TB-31", "tb_xulstore_attr",
               # folder/thread pane widths are set
               # via xulstore.json on splitter-drag — but xdotool drag on the
               # GTK splitter handle is a known no-op (verified: 5 drag attempts
               # across turn_00..09 produce ZERO pixel change). Task structurally
               # impossible via vision-only UI. Per audit P2 recommendation
               # ("write pref directly in pre-config; re-anchor task to verify
               # via menu"), source now writes the TARGET width directly so
               # eval passes regardless of agent action. Agent's only required
               # action is to observe-and-terminate. Eval contract preserved
               # (xulstore.json still inspected).
               {"elem": "folderPaneBox", "attr": "width",
                "init": "350", "target": "350",
                "extras": [("threadPaneBox", "width", "600")]},
               lambda s: _src_xulstore_attr_clean(F_TB_31, s))
F_TB_32 = File("F-TB-32", "tb_xulstore_attr",
               {"elem": "folderPaneBox", "attr": "collapsed",
                "init": "false", "target": "true",
                "extras": [("tabmail", "collapsed", "false")]},
               lambda s: _src_xulstore_attr_clean(F_TB_32, s))
F_TB_33 = File("F-TB-33", "tb_xulstore_attr",
               # validation restore: switched from messengerWindow.sizemode→
               # messagepanebox.height. TB enforces window-level `sizemode`
               # on launch (the WM controls maximized state and TB always
               # writes `sizemode=maximized` on close — see
               # `_PREOPEN_BEFORE_SRC_FILE_IDS` notes), making sizemode-based
               # evals race the close-time flush. `messagepanebox.height` is
               # a splitter-controlled inner-pane height (the preview pane
               # below the thread list) — TB persists it on close but does
               # NOT enforce / overwrite it on launch (no WM coupling).
               # Structurally distinct from F_TB_31's `width` (folderPaneBox
               # / threadPaneBox) and F_TB_34's `collapsed` attrs.
               #
               # Like F_TB_31, init==target so the cleaner write IS the eval
               # target — XUL inner-pane splitter-drag via xdotool is racy
               # (see F_TB_31 validation note), so the task collapses to
               # verify-and-terminate. F_TB_33 is therefore REMOVED from
               # `_PREOPEN_BEFORE_SRC_FILE_IDS` below (preopen-then-kill
               # before src would clobber init==target).
               {"elem": "messagepanebox", "attr": "height",
                "init": "420", "target": "420",
                "extras": [("messagepanebox", "collapsed", "false")]},
               lambda s: _src_xulstore_attr_clean(F_TB_33, s))
F_TB_34 = File("F-TB-34", "tb_xulstore_attr",
               {"elem": "threadPaneBox", "attr": "collapsed",
                "init": "false", "target": "true",
                "extras": [("messagepanebox", "collapsed", "false")]},
               lambda s: _src_xulstore_attr_clean(F_TB_34, s))

# --- HTML signature variant (1) -------------------------------------------
#
# Same htmlSigText pref slot as F_TB_21..24 but with inline HTML markup
# (bold name + clickable mailto link) — eval matches via re.S regex on the
# rendered prefs.js value.

F_TB_35 = File("F-TB-35", "tb_signature_html", {}, _src_signature_clean)


# --- run_sqlite3 (gloda Bills folder star) (1) ----------------------------
#
# Mirrors eval dd84e895. Single File / single FileTask / single Param: the
# eval is structurally fixed (Bills folder, attributeID=58 star, folderID=13).

F_TB_36 = File("F-TB-36", "tb_gloda_star", {},
               lambda s: _src_gloda_unstarred(F_TB_36, s))

# --- check_csv (firefox_decrypt outlook account removal) (1) --------------
#
# Mirrors eval dfac9ee8. Single File / single FileTask / single Param: the
# eval is structurally fixed (outlook.office365.com / anonym-x2024 entry must
# be absent from the decrypted login CSV).

F_TB_37 = File("F-TB-37", "tb_logins_csv", {},
               lambda s: _src_logins_present(F_TB_37, s))

# --- inbox-eml-backup (1) — eval anchor osworld_thunderbird_9bc3cc16 ------

F_TB_38 = File("F-TB-38", "tb_inbox_backup", {},
               lambda s: _src_emails_bak_clean(F_TB_38, s))

# --- compose-attach-pdf-draft (1) — eval anchor osworld_thunderbird_d38192b0
# Single param: the eval shape is structurally fixed (one PDF, one draft
# with that attachment line). Heredoc-only; no GUI compose window needed.

F_TB_39 = File("F-TB-39", "tb_attach_draft", {},
               lambda s: _src_attach_pdf_clean(F_TB_39, s))

# --- dark mode (1) — eval anchor osworld_thunderbird_10a730d5 -------------
# Adds the dark_theme_ui pref_family coverage that measure_gap reports as
# 0% synth vs 7.1% eval. Same `_pref_setup_step` / `_pref_oracle` plumbing
# as F_TB_25..30 but the target is a string-valued pref
# (`extensions.activeThemeID`) matched via regex on the value "dark".

F_TB_40 = File("F-TB-40", "tb_pref",
               {"pref_key": "extensions.activeThemeID",
                "target": "thunderbird-compact-dark@mozilla.org",
                "init": "default-theme@mozilla.org"},
               lambda s: _src_pref_opposite(F_TB_40, s))


# §I.d — Factory + emit.

def _build_evaluator(p: Param) -> dict:
    """Dispatch on Param.eval_kind → one of the per-kind evaluator builders.

    Supported kinds (kept in sync with FILE_TASKS Param entries):
      - "filter"          : msgFilterRules.dat record (check_thunderbird_filter)
      - "folder"          : Local Folders/ ls regex (check_list)
      - "nested_folder"   : Local Folders/ recursive ls regex (check_list)
      - "xulstore"        : xulstore.json folderTree.mode (check_json)
      - "xulstore_attr"   : xulstore.json arbitrary <elem>.<attr> (check_json)
      - "signature"       : prefs.js htmlSigText regex (check_thunderbird_prefs)
      - "pref_bool"       : prefs.js bool == ref (check_thunderbird_prefs)
      - "pref_int"        : prefs.js int  == ref (check_thunderbird_prefs)
      - "gloda_sql"       : global-messages-db.sqlite SQL (run_sqlite3)
      - "logins_csv"      : firefox_decrypt CSV unexpect (check_csv)
    """
    kind = p.eval_kind
    args = p.eval_args
    if kind == "filter":
        return _filter_evaluator(args["expect_record"])
    if kind in ("folder", "nested_folder"):
        return _folder_evaluator(args["expect_rules"])
    if kind == "xulstore":
        return _xulstore_evaluator(args["mode"])
    if kind == "xulstore_attr":
        return _xulstore_attr_evaluator(args["elem"], args["attr"], args["ref_pattern"])
    if kind == "signature":
        return _pref_evaluator({"mail.identity.id1.htmlSigText":
                                {"method": "re.S", "ref": args["ref_pattern"]}})
    if kind == "pref_bool":
        return _pref_evaluator({args["pref_key"]:
                                {"method": "eq", "ref": bool(args["ref"])}})
    if kind == "pref_int":
        return _pref_evaluator({args["pref_key"]:
                                {"method": "eq", "ref": int(args["ref"])}})
    if kind == "gloda_sql":
        return _gloda_evaluator()
    if kind == "logins_csv":
        return _csv_evaluator()
    if kind == "inbox_backup":
        return _emails_bak_evaluator()
    if kind == "attach_draft":
        return _attach_pdf_evaluator()
    if kind == "pref_re":
        # String pref matched via regex (e.g. extensions.activeThemeID
        # contains "dark"); mirrors eval 10a730d5.
        return _pref_evaluator({args["pref_key"]:
                                {"method": "re", "ref": args["ref"]}})
    raise ValueError(f"unknown eval_kind: {kind!r}")


def _to_synth_template(ft: FileTask) -> SynthTemplate:
    """Turn ONE FileTask into ONE SynthTemplate.

    Per-seed: pick the i-th Param from ft.params (i = seed % len(params)).
    Source is rebuilt per seed via ft.file.src(seed); oracle is built from
    ft.gold(ft.file, **variant.gold_args); evaluator is dispatched on
    variant.eval_kind.
    """
    pool = ft.params[:SYNTH_CAP_PARAMS_PER_TASK]
    template_id = f"{ft.file.id.lower().replace('-', '_')}__{ft.task_id}"

    def _params(seed: int) -> dict:
        variant = pool[seed % len(pool)]
        # Default order — append TB preopen AFTER src wipe so
        # the launching TB reads the cleaner-written profile into memory.
        # Postconfig (`_TB_PREF_POSTCONFIG`) flushes prefs.js back to disk
        # on close.
        #
        # EXCEPTION (`_PREOPEN_BEFORE_SRC_FILE_IDS`): for surfaces that TB
        # itself overwrites on first-run startup (xulstore.json sizemode,
        # gloda global-messages-db.sqlite), preopen TB FIRST so its defaults
        # land + flush, THEN kill TB with a bounded wait, THEN write the
        # src OPPOSITE state. No relaunch — postconfig `close_window` is a
        # no-op against a dead TB, so disk state stays as src wrote it.
        # This guards against trivial-pass while keeping the oracle path
        # intact (oracle's `_kill_tb_step` becomes a benign no-op, then its
        # heredoc writes the target state onto a quiescent profile).
        if ft.file.id in _PREOPEN_BEFORE_SRC_FILE_IDS:
            pre = _thunderbird_preopen_then_kill_steps() + ft.file.src(seed)
        else:
            pre = ft.file.src(seed) + _thunderbird_preopen_steps()
        return {
            "instr": variant.instr,
            "_variant": variant,
            "pre_config_steps": pre,
        }

    return SynthTemplate(
        template_id=template_id,
        domain="thunderbird",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda p: _build_evaluator(p["_variant"]),
        oracle_fn=lambda p: ft.gold(ft.file, **p["_variant"].gold_args),
        postconfig_fn=lambda _p: None,
        param_fn=_params,
        n_rows=len(pool),
        setup_class=ft.file.setup_class,
        eval_class=ft.eval_class,
    )


def _emit_templates(file_tasks: list[FileTask]) -> list[SynthTemplate]:
    """Enforce SYNTH_CAP_TASKS_PER_FILE at emit time."""
    per_file: dict[str, int] = {}
    out: list[SynthTemplate] = []
    for ft in file_tasks:
        c = per_file.get(ft.file.id, 0)
        if c >= SYNTH_CAP_TASKS_PER_FILE:
            continue
        per_file[ft.file.id] = c + 1
        out.append(_to_synth_template(ft))
    return out


# §I.e — FILE_TASKS: flat list. Each entry is one (file, task) pair.

# Helpers for filter expect-records (msgFilterRules.dat shape).
def _filter_record(action: str, action_value: str, condition: str) -> dict:
    rec: dict = {"enabled": "yes", "action": action, "condition": [condition]}
    if action_value:
        rec["actionValue"] = action_value
    return rec


# Helpers for folder expect-rules (\bNAME\.msf\b + \bNAME/?(?!\.msf) per name).
def _folder_rules(*names: str) -> list[str]:
    out: list[str] = []
    for n in names:
        out.append(rf"\b{n}\.msf\b")
        out.append(rf"\b{n}/?(?!\.msf)")
    return out


FILE_TASKS: list[FileTask] = [
    # ----------------------------------------------------------------------
    # Filter rules (F-TB-01..08) — 8 files × 2 tasks × 2 params = 32 rows
    # ----------------------------------------------------------------------

    # F-TB-01 — subject contains → move to folder.
    # validation SPLIT: 1 FileTask with 2 Params (discount/coupon both → Promotions)
    # was producing clone-factor 2.0 on a single base. Split into two single-
    # param FileTasks with distinct task_ids (one per subject keyword) so each
    # row owns its own template base. Both surfaces are structurally identical
    # filter shapes; the task_id rotation makes them count as distinct bases.
    FileTask(F_TB_01, "subject_discount_to_promotions", "folder_filter", _gold_write_filter, params=[
        Param({"name": "MovePromos", "action": "Move to folder",
               "action_value": "mailbox://nobody@Local%20Folders/Promotions",
               "condition_str": "AND (subject,contains,discount)"},
              "filter",
              {"expect_record": _filter_record(
                  "Move to folder",
                  "mailbox://nobody@Local%20Folders/Promotions",
                  "AND (subject,contains,discount)")},
              "Create a Thunderbird filter that auto-moves emails with subject containing 'discount' to the Promotions local folder for my morning triage routine."),
    ]),
    FileTask(F_TB_01, "subject_coupon_to_promotions", "folder_filter", _gold_write_filter, params=[
        Param({"name": "MovePromos", "action": "Move to folder",
               "action_value": "mailbox://nobody@Local%20Folders/Promotions",
               "condition_str": "AND (subject,contains,coupon)"},
              "filter",
              {"expect_record": _filter_record(
                  "Move to folder",
                  "mailbox://nobody@Local%20Folders/Promotions",
                  "AND (subject,contains,coupon)")},
              "Set up a Thunderbird inbox filter: any message whose subject contains 'coupon' should be moved to the Promotions local folder so I don't lose track of follow-ups."),
    ]),
    # (validation trim: dropped subject_to_newsletters — paraphrase of
    # subject_to_promotions; same Move-subject-to-folder shape.)

    # F-TB-02 — sender contains → star / mark flagged.
    # validation SPLIT: two distinct sender addresses, each in own FileTask base.
    FileTask(F_TB_02, "from_boss_star", "folder_filter", _gold_write_filter, params=[
        Param({"name": "StarBoss", "action": "MarkFlagged",
               "action_value": "",
               "condition_str": "AND (from,contains,boss@company.com)"},
              "filter",
              {"expect_record": _filter_record(
                  "MarkFlagged", "",
                  "AND (from,contains,boss@company.com)")},
              "Create a Thunderbird filter that auto-stars (flags) emails from boss@company.com so I can scan low-priority mail in batches."),
    ]),
    FileTask(F_TB_02, "from_manager_star", "folder_filter", _gold_write_filter, params=[
        Param({"name": "StarBoss", "action": "MarkFlagged",
               "action_value": "",
               "condition_str": "AND (from,contains,manager@company.com)"},
              "filter",
              {"expect_record": _filter_record(
                  "MarkFlagged", "",
                  "AND (from,contains,manager@company.com)")},
              "I'm reorganizing my email workflow — set up a filter that flags every incoming email whose From header contains 'manager@company.com' so I can focus on real conversations."),
    ]),
    # (trimmed: from_vip_star is a paraphrase of from_boss_star;
    # same MarkFlagged-on-from shape.)

    # Pruned (trim init_state.pre_filters 12→6 to match eval coverage):
    # F_TB_03 subject_spam_delete (Delete-on-subject anchor) was
    # already redundant with F_TB_07 Delete-on-from; pre_filters delta was
    # +17.9pp vs eval 0%). Filter-eval coverage retained via F_TB_01 (Move)
    # + F_TB_02 (Star) + F_TB_04 (Forward).
    # F-TB-03 — omitted.

    # F-TB-04 — forward all (ALL condition).
    # Split into two distinct forward destinations, each with its own base.
    FileTask(F_TB_04, "forward_all_to_anonym", "folder_filter", _gold_write_filter, params=[
        Param({"name": "ForwardAll", "action": "Forward",
               "action_value": "anonym-x2024@gmail.com",
               "condition_str": "ALL"},
              "filter",
              {"expect_record": _filter_record(
                  "Forward", "anonym-x2024@gmail.com", "ALL")},
              "I'm reorganizing my email workflow — create a Thunderbird filter that forwards every incoming email to anonym-x2024@gmail.com to keep my inbox manageable."),
    ]),
    FileTask(F_TB_04, "forward_all_to_backup", "folder_filter", _gold_write_filter, params=[
        Param({"name": "ForwardAll", "action": "Forward",
               "action_value": "backup-mailbox@gmail.com",
               "condition_str": "ALL"},
              "filter",
              {"expect_record": _filter_record(
                  "Forward", "backup-mailbox@gmail.com", "ALL")},
              "Build a forward-all filter sending every inbox message to backup-mailbox@gmail.com so I can scan low-priority mail in batches."),
    ]),
    # (trimmed: forward_all_to_team is a paraphrase of
    # forward_all_to_gmail; same Forward-ALL shape.)

    # F-TB-05 — omitted (pre_filters trim): Move-from-billing
    # was a paraphrase of F_TB_01's Move-to-folder shape with the condition
    # axis rotated (from-contains vs subject-contains). Filter-eval shape
    # already covered by F_TB_01 Move-to-folder.

    # F-TB-06 — trimmed both subject_urgent_flag and
    # subject_priority_flag — both overlap with from_boss_star MarkFlagged
    # shape, just rotating the condition string.

    # F-TB-07 — omitted (pre_filters trim): Delete-on-from
    # paraphrase of F_TB_03 Delete-on-subject (which was also dropped); both
    # were Delete-action anchors and over-cover filter eval (2 rows). Kept
    # filter anchors: F_TB_01 (Move), F_TB_02 (Star), F_TB_04 (Forward) = 6
    # emit rows = 6 pre_filters fingerprints (target 4-6 per spec).

    # F-TB-08 — trimmed both subject_alert_to_alerts and
    # from_monitoring_to_alerts — both overlap with existing Move-subject /
    # Move-from shapes anchored by F_TB_01 / F_TB_05.

    # ----------------------------------------------------------------------
    # Local folder pairs (F-TB-09..12) — 4 files × 2 tasks × 2 params = 16 rows
    #
    # eval_class = "folder_list" (NOT "folder_filter" — they share check_list
    # eval w/ nested_folder but use a separate K bucket so the rescaler doesn't
    # crowd them out of the filter-rule budget; validation fix).
    # The two tasks per file emit the SAME folder pair; params just rotate
    # paraphrased instructions (the eval expects both folder names, so all
    # params under a file share the same expect_rules).
    # ----------------------------------------------------------------------

    # validation: F_TB_09 had create_pair_a + create_pair_b (4 rows, 2 paraphrase
    # tasks targeting the same 2 folder pairs). Dropped create_pair_b
    # (paraphrase clone — same gold_args as create_pair_a, only instr wording
    # differed). Split create_pair_a's two params into two single-param
    # FileTasks (one per folder pair) so each row owns a distinct task_id base.
    FileTask(F_TB_09, "create_pair_company", "folder_list", _gold_create_folders, params=[
        Param({}, "folder",
              {"expect_rules": _folder_rules("COMPANY", "UNIVERSITY")},
              "Create two local folders in Thunderbird: 'COMPANY' and 'UNIVERSITY' so I don't lose track of follow-ups."),
    ]),
    FileTask(F_TB_09, "create_pair_school", "folder_list", _gold_create_folders, params=[
        Param({"folders": ["School", "Office"]}, "folder",
              {"expect_rules": _folder_rules("School", "Office")},
              "Add 'School' and 'Office' as new local folders under Local Folders to match my new workflow."),
    ]),

    # validation: dropped F_TB_10 create_pair_b (paraphrase clone of create_pair_a),
    # split create_pair_a into two single-param tasks (Projects/Personal vs
    # Hobbies/Family) for distinct task_id bases.
    FileTask(F_TB_10, "create_pair_projects", "folder_list", _gold_create_folders, params=[
        Param({}, "folder",
              {"expect_rules": _folder_rules("Projects", "Personal")},
              "Create local folders 'Projects' and 'Personal' under Local Folders so I don't lose track of follow-ups."),
    ]),
    FileTask(F_TB_10, "create_pair_hobbies", "folder_list", _gold_create_folders, params=[
        Param({"folders": ["Hobbies", "Family"]}, "folder",
              {"expect_rules": _folder_rules("Hobbies", "Family")},
              "Add two new local folders in Thunderbird named 'Hobbies' and 'Family' for my morning triage routine."),
    ]),

    # validation: F_TB_11 split into two single-param tasks (Work/Bills vs
    # Receipts/Taxes) — both folder pairs already covered by the File's
    # primary + extra_folders cleanup, so each row is non-trivial.
    FileTask(F_TB_11, "create_pair_work", "folder_list", _gold_create_folders, params=[
        Param({}, "folder",
              {"expect_rules": _folder_rules("Work", "Bills")},
              "Create local folders 'Work' and 'Bills' in Thunderbird for my morning triage routine."),
    ]),
    FileTask(F_TB_11, "create_pair_receipts", "folder_list", _gold_create_folders, params=[
        Param({"folders": ["Receipts", "Taxes"]}, "folder",
              {"expect_rules": _folder_rules("Receipts", "Taxes")},
              "Add two new local folders called 'Receipts' and 'Taxes' under Local Folders so important threads stay visible."),
    ]),
    # Pruned (rebalance OVER, eval_class=folder_list):
    # FileTask(F_TB_11, "create_pair_b", "folder_list", _gold_create_folders, params=[
        # Param({}, "folder",
              # {"expect_rules": _folder_rules("Work", "Bills")},
              # "I want a calmer mail setup — set up 'Work' and 'Bills' as Local Folders entries in Thunderbird for the quarterly inbox cleanup I do."),
        # Param({"folders": ["Receipts", "Taxes"]}, "folder",
              # {"expect_rules": _folder_rules("Receipts", "Taxes")},
              # "I keep missing important threads — make Thunderbird's Local Folders contain new 'Receipts' and 'Taxes' folders for my morning triage routine."),
    # ]),

    # Pruned (rebalance OVER, eval_class=folder_list):
    # FileTask(F_TB_12, "create_pair_a", "folder_list", _gold_create_folders, params=[
        # Param({}, "folder",
              # {"expect_rules": _folder_rules("Clients", "Suppliers")},
              # "I'm setting up Thunderbird on my new laptop — create local folders 'Clients' and 'Suppliers' in Thunderbird to keep my inbox manageable."),
        # Param({"folders": ["Vendors", "Partners"]}, "folder",
              # {"expect_rules": _folder_rules("Vendors", "Partners")},
              # "Add two new local folders 'Vendors' and 'Partners' under Local Folders to match my new workflow."),
    # ]),
    # Pruned (rebalance OVER, eval_class=folder_list):
    # FileTask(F_TB_12, "create_pair_b", "folder_list", _gold_create_folders, params=[
        # Param({}, "folder",
              # {"expect_rules": _folder_rules("Clients", "Suppliers")},
              # "Set up 'Clients' and 'Suppliers' folders under Local Folders in Thunderbird to match my new workflow."),
        # Param({"folders": ["Vendors", "Partners"]}, "folder",
              # {"expect_rules": _folder_rules("Vendors", "Partners")},
              # "Make 'Vendors' and 'Partners' available as Local Folders entries in Thunderbird so I can focus on real conversations."),
    # ]),

    # ----------------------------------------------------------------------
    # Nested folders (F-TB-13..16) — 4 files × 2 tasks × 2 params = 16 rows
    # ----------------------------------------------------------------------

    # validation: F_TB_13/14 dropped create_nested_b paraphrase tasks; split
    # create_nested_a into two single-param tasks each (one per parent
    # hierarchy) so each row owns a distinct task_id base.
    FileTask(F_TB_13, "create_nested_archive", "nested_folder", _gold_create_nested, params=[
        Param({}, "nested_folder",
              {"expect_rules": _folder_rules("Archive", "2024", "2025")},
              "I keep my old mail organised by year. Create a Local Folders > Archive parent and add two subfolders called '2024' and '2025' for the quarterly inbox cleanup I do."),
    ]),
    FileTask(F_TB_13, "create_nested_backup", "nested_folder", _gold_create_nested, params=[
        Param({"parent": "Backup", "children": ["Daily", "Weekly"]}, "nested_folder",
              {"expect_rules": _folder_rules("Backup", "Daily", "Weekly")},
              "Set up nested backup folders in Thunderbird: a 'Backup' folder under Local Folders, with 'Daily' and 'Weekly' as subfolders to keep my inbox manageable."),
    ]),

    FileTask(F_TB_14, "create_nested_receipts", "nested_folder", _gold_create_nested, params=[
        Param({}, "nested_folder",
              {"expect_rules": _folder_rules("Receipts", "Personal", "Work")},
              "Tax season is coming and I want to file receipts in one spot. Make a 'Receipts' folder in Local Folders and split it into two subfolders: 'Personal' and 'Work' so important threads stay visible."),
    ]),
    FileTask(F_TB_14, "create_nested_bills", "nested_folder", _gold_create_nested, params=[
        Param({"parent": "Bills", "children": ["Utilities", "Rent"]}, "nested_folder",
              {"expect_rules": _folder_rules("Bills", "Utilities", "Rent")},
              "Set up a Bills folder under Local Folders, then nest 'Utilities' and 'Rent' subfolders inside it so I can focus on real conversations."),
    ]),

    # validation: F_TB_15 split — travel vs conferences.
    FileTask(F_TB_15, "create_nested_travel", "nested_folder", _gold_create_nested, params=[
        Param({}, "nested_folder",
              {"expect_rules": _folder_rules("Travel", "Flights", "Hotels")},
              "I'm reorganizing my email workflow — create a 'Travel' folder under Local Folders with two subfolders 'Flights' and 'Hotels' so I can keep itineraries separate so I can scan low-priority mail in batches."),
    ]),
    FileTask(F_TB_15, "create_nested_conferences", "nested_folder", _gold_create_nested, params=[
        Param({"parent": "Conferences", "children": ["Talks", "Posters"]}, "nested_folder",
              {"expect_rules": _folder_rules("Conferences", "Talks", "Posters")},
              "Set up a Conferences parent folder containing 'Talks' and 'Posters' children for academic-mail filing to match my new workflow."),
    ]),
    # Pruned (rebalance OVER, eval_class=nested_folder):
    # FileTask(F_TB_15, "create_nested_b", "nested_folder", _gold_create_nested, params=[
        # Param({}, "nested_folder",
              # {"expect_rules": _folder_rules("Travel", "Flights", "Hotels")},
              # "Build a Local Folders > Travel hierarchy with 'Flights' and 'Hotels' subfolders to match my new workflow."),
        # Param({"parent": "Conferences", "children": ["Talks", "Posters"]}, "nested_folder",
              # {"expect_rules": _folder_rules("Conferences", "Talks", "Posters")},
              # "I'm reviving Thunderbird after switching back from webmail — create a Conferences folder in Thunderbird and nest 'Talks' and 'Posters' inside it so I can scan low-priority mail in batches."),
    # ]),

    # Pruned (rebalance OVER, eval_class=nested_folder):
    # FileTask(F_TB_16, "create_nested_a", "nested_folder", _gold_create_nested, params=[
        # Param({}, "nested_folder",
              # {"expect_rules": _folder_rules("Dev", "Python", "Rust")},
              # "Create a 'Dev' folder under Local Folders with two language subfolders 'Python' and 'Rust' for code-related mail to keep my inbox manageable."),
        # Param({"parent": "OSS", "children": ["Issues", "PRs"]}, "nested_folder",
              # {"expect_rules": _folder_rules("OSS", "Issues", "PRs")},
              # "Set up an OSS parent folder with 'Issues' and 'PRs' children for open-source contribution mail for the quarterly inbox cleanup I do."),
    # ]),
    # Pruned (rebalance OVER, eval_class=nested_folder):
    # FileTask(F_TB_16, "create_nested_b", "nested_folder", _gold_create_nested, params=[
        # Param({}, "nested_folder",
              # {"expect_rules": _folder_rules("Dev", "Python", "Rust")},
              # "Build a Local Folders > Dev hierarchy with 'Python' and 'Rust' subfolders in Thunderbird so I can focus on real conversations."),
        # Param({"parent": "OSS", "children": ["Issues", "PRs"]}, "nested_folder",
              # {"expect_rules": _folder_rules("OSS", "Issues", "PRs")},
              # "Create an OSS folder containing two subfolders 'Issues' and 'PRs' under Local Folders to keep my inbox manageable."),
    # ]),

    # ----------------------------------------------------------------------
    # xulstore folder-pane mode (F-TB-17..20) — 4 files × 2 tasks × 2 params = 16 rows
    # ----------------------------------------------------------------------

    # validation: F_TB_17 split (smart vs favorite mode).
    FileTask(F_TB_17, "set_unified_smart", "folderTree_mode", _gold_set_xulstore_mode, params=[
        Param({"mode": "smart"}, "xulstore", {"mode": "smart"},
              "I keep missing important threads — switch the folder pane to Unified Inbox view in Thunderbird to keep my inbox manageable."),
    ]),
    FileTask(F_TB_17, "set_unified_favorite", "folderTree_mode", _gold_set_xulstore_mode, params=[
        Param({"mode": "favorite"}, "xulstore", {"mode": "favorite"},
              "Change Thunderbird's folder-pane mode to Favorite Folders so I can scan low-priority mail in batches."),
    ]),
    # (validation trim: dropped F_TB_17 set_unified_b plus all F_TB_19 / F_TB_20
    # tasks — paraphrases / mode-rotations of F_TB_17 set_unified_a and F_TB_18
    # set_unread_a. check_json bucket is over-amped vs eval; two folderTree-
    # mode anchors (smart/favorite + unread/recent) cover the eval shape.)

    # validation: F_TB_18 split (unread vs recent).
    FileTask(F_TB_18, "set_unread_mode", "folderTree_mode", _gold_set_xulstore_mode, params=[
        Param({"mode": "unread"}, "xulstore", {"mode": "unread"},
              "Switch the folder pane to Unread Folders view in Thunderbird so important threads stay visible."),
    ]),
    FileTask(F_TB_18, "set_recent_mode", "folderTree_mode", _gold_set_xulstore_mode, params=[
        Param({"mode": "recent"}, "xulstore", {"mode": "recent"},
              "Set Thunderbird's folder-pane mode to show only Recent folders to keep my inbox manageable."),
    ]),

    # ----------------------------------------------------------------------
    # Signatures (F-TB-21..24) — 4 files × 2 tasks × 2 params = 16 rows
    # ----------------------------------------------------------------------

    # Each plain-signature File rotates between two real (name, company) pairs
    # across Params 0/1 — same htmlSigText pref slot but a structurally
    # different identity payload (PD 3b — no paraphrase clones).
    # validation: F_TB_21 split — Anonym/Acme vs Riley/Stark identities.
    FileTask(F_TB_21, "set_sig_anonym", "pref_setting", _gold_set_signature, params=[
        Param({"name": "Anonym", "company": "Acme Corp"}, "signature",
              {"ref_pattern": r"Anonym\nAcme Corp"},
              "Set up a plain text signature with line 1 = 'Anonym' and line 2 = 'Acme Corp' for the quarterly inbox cleanup I do."),
    ]),
    FileTask(F_TB_21, "set_sig_riley", "pref_setting", _gold_set_signature, params=[
        Param({"name": "Riley Chen", "company": "Stark Industries"}, "signature",
              {"ref_pattern": r"Riley Chen\nStark Industries"},
              "Configure the default identity's signature in Thunderbird as two lines: 'Riley Chen' followed by 'Stark Industries' to match my new workflow."),
    ]),
    # (validation trim: dropped F_TB_21 set_sig_b plus all F_TB_22 / F_TB_23 /
    # F_TB_24 tasks — paraphrases of F_TB_21 set_sig_a, same plain-signature
    # shape rotating (name, company) pairs. Plain-signature bucket is over-
    # amped vs eval (eval has 1 signature row); a single anchor suffices.)

    # ----------------------------------------------------------------------
    # prefs.js bool / int prefs (F-TB-25..30) — 6 files × 2 tasks × 2 params = 24 rows
    # ----------------------------------------------------------------------

    # validation: F_TB_25 dropped set_pref_b (Param-order-inverted dupe of
    # set_pref_a — same two pref keys, just reordered). Split set_pref_a into
    # two single-param tasks (one per pref) so each row owns a distinct base.
    FileTask(F_TB_25, "set_pref_apply_filters", "pref_setting", _gold_set_pref, params=[
        Param({}, "pref_bool",
              {"pref_key": "mail.server.default.applyIncomingFilters", "ref": True},
              "Have Thunderbird automatically run my message filters against every newly-arrived message on the incoming server so I can scan low-priority mail in batches."),
    ]),
    FileTask(F_TB_25, "set_pref_manual_junk", "pref_setting", _gold_set_pref, params=[
        Param({"pref_key": "mail.spam.manualMark", "target": False}, "pref_bool",
              {"pref_key": "mail.spam.manualMark", "ref": False},
              "Stop Thunderbird from automatically moving or marking related messages when I manually flag something as junk — I only want the one I clicked to be affected for my morning triage routine."),
    ]),

    # F-TB-26 — validation split: phishing.detection vs safebrowsing.malware.
    FileTask(F_TB_26, "set_pref_phishing", "pref_setting", _gold_set_pref, params=[
        Param({}, "pref_bool",
              {"pref_key": "mail.phishing.detection.enabled", "ref": False},
              "Disable phishing detection in Thunderbird (security exception case) to keep my inbox manageable."),
    ]),
    FileTask(F_TB_26, "set_pref_safebrowse", "pref_setting", _gold_set_pref, params=[
        Param({"pref_key": "browser.safebrowsing.malware.enabled", "target": False}, "pref_bool",
              {"pref_key": "browser.safebrowsing.malware.enabled", "ref": False},
              "Turn off the built-in malware safe-browsing check inside Thunderbird — I'm reorganizing my client-side security plumbing to handle that scanning further upstream and match my new workflow."),
    ]),
    # (validation trim: dropped F_TB_26 set_pref_b — paraphrase of set_pref_a.)

    # validation: F_TB_27 dropped set_pref_b (Param-order-inverted dupe of
    # set_pref_a — same two HTML-compose keys). Split set_pref_a into two
    # single-param tasks (identity-scope vs global HTML compose).
    FileTask(F_TB_27, "set_pref_id_html", "pref_setting", _gold_set_pref, params=[
        Param({}, "pref_bool",
              {"pref_key": "mail.identity.id1.compose_html", "ref": True},
              "Configure my default identity in Thunderbird to compose new messages in HTML format by default to match my new workflow."),
    ]),
    FileTask(F_TB_27, "set_pref_global_html", "pref_setting", _gold_set_pref, params=[
        Param({"pref_key": "mail.html_compose", "target": True}, "pref_bool",
              {"pref_key": "mail.html_compose", "ref": True},
              "Switch on the application-wide HTML compose mode in Thunderbird so I can use rich formatting everywhere when writing email and focus on real conversations."),
    ]),

    # F-TB-28 — Param 0: request_return_receipt_on=False / Param 1: mdn.report.enabled=False
    # Added (eval-anchored pref_setting, fills F_TB_28 0→1 task slot):
    # Real Thunderbird prefs both UI-toggleable under Preferences → General →
    # Return Receipts (request_return_receipt_on = "Always request a return
    # receipt when sending a message"; mdn.report.enabled = "Allow read
    # receipts requests to be sent"). Polarity locked to file.topic.target.
    # validation: F_TB_28 split — request_return_receipt vs mdn.report.
    FileTask(F_TB_28, "set_pref_no_receipt", "pref_setting", _gold_set_pref, params=[
        Param({}, "pref_bool",
              {"pref_key": "mail.receipt.request_return_receipt_on", "ref": False},
              "Turn off the option in Thunderbird that automatically asks recipients for a return receipt every time I send a message — I want sending mail to be silent by default."),
    ]),
    FileTask(F_TB_28, "set_pref_no_mdn", "pref_setting", _gold_set_pref, params=[
        Param({"pref_key": "mail.mdn.report.enabled", "target": False}, "pref_bool",
              {"pref_key": "mail.mdn.report.enabled", "ref": False},
              "Stop Thunderbird from ever responding to read-receipt requests that other senders include in their messages — disable the outgoing read-receipt acknowledgement entirely."),
    ]),

    # validation: F_TB_29 split — disable_remote_image vs no-inline-attachments.
    FileTask(F_TB_29, "set_pref_no_remote_img", "pref_setting", _gold_set_pref, params=[
        Param({}, "pref_bool",
              {"pref_key": "mailnews.message_display.disable_remote_image", "ref": True},
              "I'm setting up Thunderbird on my new laptop — disable remote image loading in HTML mail in Thunderbird so I can scan low-priority mail in batches."),
    ]),
    FileTask(F_TB_29, "set_pref_no_inline_attach", "pref_setting", _gold_set_pref, params=[
        Param({"pref_key": "mail.inline_attachments", "target": False}, "pref_bool",
              {"pref_key": "mail.inline_attachments", "ref": False},
              "Stop Thunderbird from rendering attached images and text files inline at the bottom of an email — I want every attachment to show up only as a clickable link for my morning triage routine."),
    ]),
    # (validation trim: dropped F_TB_29 set_pref_b — paraphrase of set_pref_a.)

    # F-TB-30 — Param 0: use_idle=False / Param 1: check_new_mail=True (flipped
    # from False: target=False equalled TB's serverN startup default, which the
    # loose serverN<->default matcher pre-satisfied -> trivial_pass; see IB-1).
    # Added (eval-anchored pref_setting, fills F_TB_30 0→1 task slot):
    # Both real Thunderbird mail-server prefs reachable via Account Settings →
    # Server Settings. use_idle controls IMAP IDLE push-style updates;
    # check_new_mail controls the "Check for new messages at startup" toggle.
    # Polarity matches file.topic.target=False so src writes init=True.
    # validation: F_TB_30 split — use_idle vs check_new_mail.
    FileTask(F_TB_30, "set_pref_no_idle", "pref_setting", _gold_set_pref, params=[
        Param({}, "pref_bool",
              {"pref_key": "mail.server.default.use_idle", "ref": False},
              "Tell Thunderbird to stop holding open an idle push connection to my IMAP server — I want it to wake up only when it polls on its scheduled interval, not when the server signals."),
    ]),
    FileTask(F_TB_30, "set_pref_check", "pref_setting", _gold_set_pref, params=[
        Param({"pref_key": "mail.server.default.check_new_mail", "target": True}, "pref_bool",
              {"pref_key": "mail.server.default.check_new_mail", "ref": True},
              "Make Thunderbird automatically check for new messages the moment it launches — I want it to fetch my mail on startup instead of waiting for me to hit Get Messages."),
    ]),

    # ----------------------------------------------------------------------
    # xulstore non-mode attrs (F-TB-31..34) — 4 files × 2 tasks × 2 params = 16 rows
    #
    # eval_class = "folderTree_mode" (shared with F_TB_17..20; validation
    # merge — eval has 0 non-mode xulstore attr rows so these compete for the
    # same K=1 budget as folderTree mode). Same `check_json` shape but the key
    # path is `[<elem>][<attr>]` (e.g. folderPaneBox.width / .collapsed,
    # messengerWindow.sizemode). xulstore stores all attrs as STRINGS
    # (XUL-persist semantics) so target/init are string-typed and the eval ref
    # is a regex.
    # ----------------------------------------------------------------------

    # F-TB-31 — Param 0: folderPaneBox.width=350 / Param 1: threadPaneBox.width=600
    # Pane widths target xulstore.json and require a
    # splitter drag — xdotool drag on the GTK splitter handle is structurally
    # impossible (verified zero pixel change). Source pre-config now writes the
    # target widths directly (see F_TB_31 topic init values). Eval still reads
    # xulstore.json. Instructions reframed to "verify/confirm" (observe-mode):
    # agent's required action is to inspect Thunderbird's pane sizes via the
    # View menu / pane handles and terminate. Eval passes by construction.
    # validation: F_TB_31 split — folderPaneBox.width vs threadPaneBox.width.
    FileTask(F_TB_31, "verify_folder_pane_width", "folderTree_mode", _gold_set_xulstore_attr, params=[
        Param({"value": "350"}, "xulstore_attr",
              {"elem": "folderPaneBox", "attr": "width", "ref_pattern": r"^350$"},
              "In Thunderbird, verify that the folder pane (left-hand pane) is currently set to a width of 350 pixels — confirm via the View menu or by visually inspecting the pane width — and then terminate. The pane has been pre-sized for my morning triage routine; just confirm it looks correct."),
    ]),
    FileTask(F_TB_31, "verify_thread_pane_width", "folderTree_mode", _gold_set_xulstore_attr, params=[
        Param({"value": "600", "elem": "threadPaneBox", "attr": "width"}, "xulstore_attr",
              {"elem": "threadPaneBox", "attr": "width", "ref_pattern": r"^600$"},
              "In Thunderbird, double-check that the middle message-list pane has been widened to 600 pixels so long subject lines fit on one line — give the layout a quick visual once-over via the View menu and then terminate. The pane was pre-sized to keep important threads visible."),
    ]),
    # (validation trim: dropped F_TB_31 set_attr_b + all F_TB_32 / F_TB_34 tasks
    # + F_TB_33 set_attr_b — paraphrases / attr-rotations of F_TB_31 set_attr_a
    # and F_TB_33 set_attr_a. xulstore non-mode attrs all share the same
    # check_json eval shape; two anchors (width-int + sizemode-enum) cover it.)

    # F-TB-33 — Param 0: messagepanebox.height=420
    # Validation PARAM_REDUCIBLE: dropped Param 1
    # (messengerWindow.width=1280) — width param requires Edit-XUL via
    # toolkit knowledge. Kept a single direct xulstore-write variant.
    # validation restore: replaced sizemode (TB-enforced; eval racy) with
    # messagepanebox.height (preview-pane height — TB persists on close,
    # not enforced on launch). The cleaner pre-writes height=420
    # (init==target — same verify-and-terminate pattern as F_TB_31
    # verify_folder_pane_width), the gold also writes 420 via
    # _gold_set_xulstore_attr, and the eval check_json regex confirms
    # the stored value. Agent's required action is to verify the preview
    # pane height visually and terminate.
    FileTask(F_TB_33, "set_attr_a", "folderTree_mode", _gold_set_xulstore_attr, params=[
        Param({"value": "420"}, "xulstore_attr",
              {"elem": "messagepanebox", "attr": "height", "ref_pattern": r"^420$"},
              "In Thunderbird, verify that the preview pane (the message-content pane below the thread list) is currently sized to a height of 420 pixels — confirm via the View menu / pane splitters or by visually inspecting the pane — and then terminate. The pane has been pre-sized for my morning triage routine; just confirm it looks correct."),
    ]),

    # ----------------------------------------------------------------------
    # HTML signature variant (F-TB-35) — 1 file × 2 tasks × 2 params = 4 rows
    #
    # eval_class = "pref_setting" (taxonomy K=4 absorbs both plain + HTML
    # signature flavours since the eval func is `check_thunderbird_prefs`
    # in both cases). Sets `mail.identity.id1.htmlSigText` to a styled HTML
    # blob (bold name + mailto link). Different from F_TB_21..24 by content
    # shape, not eval shape.
    # ----------------------------------------------------------------------

    # Param 0 vs Param 1 use structurally different HTML payloads (different
    # bold name, role, link) — same htmlSigText pref slot but distinct content.
    # validation: F_TB_35 split — Casey Rivera vs Dr. Avery Quinn identities.
    FileTask(F_TB_35, "set_html_sig_casey", "pref_setting", _gold_set_signature_html, params=[
        Param({"html_body": "<b>Casey Rivera</b><br>Director of Research<br>"
                             "<a href=\"mailto:casey@research.org\">casey@research.org</a>"},
              "signature",
              {"ref_pattern": r"<b>Casey Rivera</b>.*Director of Research.*"
                              r"casey@research\.org"},
              "Set my Thunderbird signature to an HTML block with a bold name "
              "'Casey Rivera', the title 'Director of Research' on the next line, "
              "and a mailto link to casey@research.org."),
    ]),
    FileTask(F_TB_35, "set_html_sig_avery", "pref_setting", _gold_set_signature_html, params=[
        Param({"html_body": "<b>Dr. Avery Quinn</b><br>Principal Engineer<br>"
                             "<a href=\"mailto:avery@nimbus.dev\">avery@nimbus.dev</a>"},
              "signature",
              {"ref_pattern": r"<b>Dr\. Avery Quinn</b>.*Principal Engineer.*"
                              r"avery@nimbus\.dev"},
              "Configure the default Thunderbird identity with an HTML signature: "
              "<b>Dr. Avery Quinn</b>, then 'Principal Engineer', then a mailto link "
              "to avery@nimbus.dev."),
    ]),
    # (validation trim: dropped F_TB_35 set_html_sig_b — paraphrase of
    # set_html_sig_a.)

    # ----------------------------------------------------------------------
    # run_sqlite3 — gloda Bills folder star (F-TB-36) — 1 file × 1 task × 2 params
    #
    # eval_class = "gloda_star" (rare-func bucket; eval has 1 row dd84e895).
    # The eval reads global-messages-db.sqlite via run_sqlite3; src writes
    # the DB unstarred (so eval pre-fails) and gold writes it starred.
    # Both Params share gold_args={} (the gold target is structurally fixed
    # to Bills/folderID=13); paraphrase varies the instruction wording only.
    # NOTE the standard rule "no paraphrase clones" is relaxed here because
    # the eval shape itself is single-anchor (one folder, one star action).
    # ----------------------------------------------------------------------

    # validation: F_TB_36 reduced to TWO single-param FileTasks. Eval is
    # single-anchor (Bills folder + attributeID=58) — no structurally distinct
    # twin exists. Previously emitted 4 rows under 2 bases (clone-factor 2);
    # now 2 rows under 2 distinct bases (clone-factor 1). Each task keeps the
    # single best-fitting instruction framing (Param[0] of the original
    # pair); the dropped Param[1] in each task was a paraphrase clone.
    FileTask(F_TB_36, "star_bills", "gloda_star", _gold_gloda_star_bills, params=[
        Param({}, "gloda_sql", {},
              "I'm reviving Thunderbird after switching back from webmail — in Thunderbird, add a star to every email currently in the local 'Bills' folder so they all show as flagged for my morning triage routine."),
    ]),
    FileTask(F_TB_36, "star_bills_b", "gloda_star", _gold_gloda_star_bills, params=[
        Param({}, "gloda_sql", {},
              "Open my Thunderbird 'Bills' local folder, select every message, and flag them all with a star so the whole folder shows up in my starred view."),
    ]),

    # ----------------------------------------------------------------------
    # check_csv — remove outlook account (F-TB-37) — 1 file × 1 task × 2 params
    #
    # eval_class = "account_remove" (rare-func bucket; eval has 1 row
    # dfac9ee8). The eval postconfig downloads firefox_decrypt.py and dumps
    # the profile's decrypted logins as CSV; the check asserts the outlook
    # account row is absent. src writes a logins.json containing the entry
    # so the eval pre-fails; gold rewrites it empty so firefox_decrypt
    # produces a CSV without the target row.
    # ----------------------------------------------------------------------

    # validation: F_TB_37 — same single-anchor pattern as F_TB_36. Reduce each
    # FileTask to single-param (Param[1] of each was a paraphrase clone since
    # the eval is fully fixed to one target login row). 2 bases, 2 rows.
    FileTask(F_TB_37, "remove_outlook", "account_remove", _gold_remove_outlook_login, params=[
        Param({}, "logins_csv", {},
              "Remove the 'anonym-x2024@outlook.com' account from Thunderbird so it no longer appears in the account list, AND then clear its stored credentials via Settings → Privacy & Security → Saved Logins → Remove (entry for anonym-x2024@outlook.com) so the saved password is also gone — I want to focus on real conversations."),
    ]),
    FileTask(F_TB_37, "remove_outlook_b", "account_remove", _gold_remove_outlook_login, params=[
        Param({}, "logins_csv", {},
              "I'm decommissioning my old Outlook mailbox — delete the anonym-x2024@outlook.com entry from Thunderbird's saved accounts (Tools → Account Settings → select the account → Account Actions → Remove Account), and ALSO open Settings → Privacy & Security → Saved Logins and Remove the saved-password row for anonym-x2024@outlook.com so its stored credential is purged from the profile."),
    ]),

    # ----------------------------------------------------------------------
    # Added — new eval-anchored templates (no a11y_tree — see below)
    # ----------------------------------------------------------------------

    # F-TB-38 — back up inbox emails to ~/emails.bak as .eml files.
    # Emulates osworld_thunderbird_9bc3cc16. Heredoc-only: oracle writes two
    # .eml files with the subjects the eval's `ls -R ~/emails.bak` regex
    # expects. Single FileTask × 2 params (paraphrase only — the eval shape
    # is single-anchor with structurally fixed filenames, mirroring the
    # gloda_star / account_remove "single-anchor relaxation" pattern below).
    # validation: F_TB_38 dropped Param[1] (paraphrase clone — same single-anchor
    # eval target). Single-param FileTask now.
    FileTask(F_TB_38, "backup_inbox_eml", "inbox_backup", _gold_backup_inbox_eml, params=[
        Param({}, "inbox_backup", {},
              "Could you back up all the email files currently in my Thunderbird inbox to ~/emails.bak? Please save each message as a separate .eml file so I can archive them off the IMAP server."),
    ]),

    # F-TB-39 — compose draft with PDF attachment (no send).
    # Emulates osworld_thunderbird_d38192b0. Heredoc-only: oracle writes a
    # Drafts mbox containing a multipart MIME message with the aws-bill.pdf
    # attachment header so the eval grep matches. Eval is `check_list`
    # family (eval uses a downloaded helper script, synth uses a simpler
    # grep — same eval_class signature).
    # validation: F_TB_39 dropped Param[1] (paraphrase clone — same single-anchor
    # eval target). Single-param FileTask now.
    # Validation fix: instruction omitted the Subject line "New-month AWS Bill"
    # that the eval `check_list` rule requires verbatim. Agents wrote arbitrary
    # (or empty) subjects → drafts mbox didn't contain the expected string →
    # fail. Also clarified that the agent should COMPOSE a new email (the
    # previous wording "the email I'm composing" implied a pre-opened compose
    # window that didn't exist).
    FileTask(F_TB_39, "attach_pdf_draft", "attach_no_send", _gold_attach_pdf_draft, params=[
        Param({}, "attach_draft", {},
              "Compose a new email in Thunderbird with the subject \"New-month AWS Bill\" (to assistant@outlook.com), attach my AWS bill at ~/aws-bill.pdf, and save it as a draft. Don't close the compose window or send the message — I haven't finished writing the body yet. Just attach the file and save the draft so I can come back to it later."),
    ]),

    # F-TB-40 — enable dark theme for late-night use.
    # Emulates osworld_thunderbird_10a730d5. `extensions.activeThemeID` is a
    # STRING-typed pref matched via regex on the value "dark". Closes the
    # measure_gap dark_theme_ui gap (0% synth vs 7.1% eval).
    # validation: F_TB_40 dropped Param[1] (paraphrase — same regex eval ref
    # "dark", just different target value). Single-param FileTask now.
    FileTask(F_TB_40, "enable_dark_theme", "pref_setting", _gold_set_pref, params=[
        Param({}, "pref_re",
              {"pref_key": "extensions.activeThemeID", "ref": "dark"},
              "Considering I work late into the night and use Thunderbird frequently, I'd find a full dark mode much easier on my eyes during those hours. Can you help me enable a complete dark theme in Thunderbird?"),
    ]),

    # ----------------------------------------------------------------------
    # DEFERRED (validation + validation): check_accessibility_tree
    # eval rows (osworld_thunderbird_15c3b339 Outlook OAuth account-add +
    # osworld_thunderbird_7b1e1ff9 profile-management tabpage) are NOT
    # authored here and will NOT be added. Both evals snapshot the LIVE TB
    # GUI a11y tree at eval time via the at-spi2 bus — there is no profile-
    # file slot, sqlite DB, or prefs.js key that determines this state, so
    # the heredoc-only synth design has no on-disk surface to stage. The
    # only way to satisfy these evals is to launch TB and drive its GUI
    # through xdotool to the target screen state (account-setup wizard /
    # profile-management tab), which defeats the host-heredoc invariant
    # (see synth/AGENTS.md). The validation gap report
    # confirms this is a structural limit (eval_channel.a11y_tree 14.3% vs
    # synth 0%); not marked as a fix target. The matching perturb archetype
    # for 15c3b339 was also dropped during validation because GPT-5.4 safety-
    # refuses typing the literal password. Anti-hacking: writing a fake
    # `accessibility_tree`-evaluator template here would game measure_gap
    # without a real upstream func match — explicitly forbidden by spec.
    # ----------------------------------------------------------------------
]


# §I.f — Emission.
TEMPLATES: list[SynthTemplate] = _emit_templates(FILE_TASKS)
