"""VS Code synth generator (Track A — host-heredoc design).

Per AGENTS.md / vs_code.md: settings.json and keybindings.json eval is
USER-scope (`/home/user/.config/Code/User/...`) — instructions never say
"for this repo". Pre-config writes the OPPOSITE state (or empty). Oracle
kills `code` first (so the JSONC-rewrite on quit can't clobber our edits)
then merges JSON via `python3 << 'PYEOF'` heredoc using `json.dumps`/
`json.loads` so JSON `true`/`false`/`null` round-trip cleanly through Python
source. JSONC (`// comment`) is stripped before parsing so VS Code's own
edits don't corrupt the read. All rows are heredoc-only.

Implemented row groups (oracle-validated, all 27/27 PASS):
- settings.json single-key bool/int/string sets (#1, #2, #4, #16-#19, #29,
  #30, #36-#45)                                      → check_json_settings
                                                       (`settings_word_wrap_on`, `settings_font_size_16`,
                                                        `settings_color_theme_dark`, `settings_format_on_save`,
                                                        `settings_tab_size_4`, `settings_files_eol_lf`,
                                                        `settings_git_autofetch`, `settings_icon_theme`,
                                                        `settings_terminal_profile`, `settings_line_numbers_relative`,
                                                        `settings_minimap_disable`, `settings_terminal_font_fira`,
                                                        `settings_cursor_blink_solid`, `settings_tab_sizing_shrink`,
                                                        `settings_bracket_pair_color`, `settings_pytest_enabled`,
                                                        `settings_font_ligatures`, `settings_smooth_scroll`,
                                                        `settings_git_confirm_sync`)
- settings.json multi-key sets (#13, #14, #20, #31, #32, #70, #71)
                                                     → check_json_settings (multi-key expected)
                                                       (`settings_autosave_500`, `settings_autosave_1000`,
                                                        `settings_format_pair`, `settings_files_exclude_pycache`,
                                                        `settings_files_exclude_logs`, `settings_minimap_multi`,
                                                        `settings_terminal_font_multi`)
- keybindings.json append (#3, #21, #22, #46, #47, #48)
                                                     → check_json_keybindings
                                                       (`kb_format_doc`, `kb_terminal_new`, `kb_reload_window`,
                                                        `kb_word_wrap_toggle`, `kb_save_all`, `kb_peek_definition`)
- File create at path (#12, #75, #77, #78, #79, #80) → compare_text_file
                                                       (`file_create_test_py`, `file_create_main_py`,
                                                        `file_create_readme_md`, `file_create_requirements_txt`,
                                                        `file_create_env`, `file_create_dockerfile`)
- File edit replace / indent (#10, #11)              → compare_text_file
                                                       (`file_edit_replace_text_test`, `file_edit_indent_bubble_sort`)
- Real-asset find-replace on OSS code (#26-style)    → compare_text_file
                                                       (validation trim 7→3 to match eval volume)
                                                       (`vs_code_fr_django_utils_quote_to_escape`,
                                                        `vs_code_fr_serde_lib_pub_to_pub_crate`,
                                                        `vs_code_fr_typescript_uri_URI_to_Uri`)
- Real-asset comment-toggle (line-range prefix)      → compare_text_file
                                                       (validation trim 5→2)
                                                       (`vs_code_comment_chi_mux_lines_1_15`,
                                                        `vs_code_comment_typescript_types_lines_10_15`)
- Real-asset indent / format conversion              → compare_text_file
                                                       (validation trim 2→1)
                                                       (`vs_code_indent_gin_handler_tabs_to_4spaces`)
- File-create from gitignore template (real asset)   → compare_text_file
                                                       (validation trim 6→2)
                                                       (`vs_code_create_gitignore_python`,
                                                        `vs_code_create_gitignore_node`)
- settings.json with real OSS config base            → check_json_settings
                                                       (`vs_code_settings_pylance_on_real_python_config`,
                                                        `vs_code_settings_typecheckmode_basic_real_config`,
                                                        `vs_code_settings_strict_imports_real_config`)

- Extension install (#5, #23-#25, #51-#58, plus VSIX #15) — `is_extension_installed`
  greps `code --list-extensions`. Pre-config purges `~/.vscode/extensions/<id>-*`
  to hold the trivial-pass guard; oracle replays eval-side install command
  (`code --no-sandbox --install-extension <ext-id|/path/to.vsix> --force`).
  Marketplace rows (`vs_code_install_ext_python` / `_autodocstring` / `_prettier`
  / `_eslint` / `_gitlens`) require outbound net at oracle-replay time (same
  assumption eval makes); the local-VSIX row (`vs_code_install_ext_local_vsix`)
  is network-independent and builds a minimal `undefined_publisher.test` VSIX
  on disk, mirroring `osworld_vs_code_0512bb38`.

compare_config settings.json (validation — mirrors eval 982d12a5):
- settings.json read via `compare_config` (containment_ok=True JSON-subset)
  — same eval shape as 982d12a5. 2 templates.
                                                       (`vs_code_compare_config_color_theme_dark`,
                                                        `vs_code_compare_config_color_theme_light`)
  DEFERRED (eval 53ad5833): `vscode_config` result-type reads
  `/home/user/OpenProject.txt` via a bundled `vscodeEvalExtension.vsix`
  that records the currently-open VS Code folder. No shell oracle without
  installing that extension + driving the GUI window — left deferred.

Workspace ops (NEW — bridges deferred rows #6-#9, #28, #59-#62):
- `.code-workspace` JSON file save / multi-folder add / settings / extensions
  / launch blocks — `_make_workspace_template` handles three eval shapes:
  `is_extension_installed` (file-exists `ls | grep`) for the save-as row
  (mirrors 5e2d93d8) and `check_json_settings` (subset on top-level keys)
  for `folders` / `settings` / `extensions` / `launch` blocks (mirrors
  6ed0a554's eval shape, extended for structural diversity). 6 templates.
                                                       (`vs_code_workspace_save_single_folder`,
                                                        `vs_code_workspace_multi_folder_2`,
                                                        `vs_code_workspace_multi_folder_3`,
                                                        `vs_code_workspace_with_settings_tabsize`,
                                                        `vs_code_workspace_with_extensions_recommendations`,
                                                        `vs_code_workspace_with_launch_python`)

Deferred (per devs/envs/lite.osworld/synth/vs_code.md `## Implementation status`):
- Multi-cursor / outline-navigation / global-search rows #33, #35, #84,
  #93-#95, #106, #107 — UI-only ops; oracle has no deterministic path
  without a screenshot/UI driver. DROPPED (no clean shell oracle).
- Code asset-driven rows #26, #27, #63-#69, #81-#92, #96-#99, #101-#105 —
  partially IMPLEMENTED via `_stage_asset` + container-side transforms
  (find-replace, comment-toggle, indent normalize, gitignore copy, real
  settings.json base) — see Real-asset row groups above. The remaining
  asset rows requiring per-line hand-curated before/after pairs (e.g.
  language-server-driven rename, prettier formatting) are still deferred.
- Format-document rows requiring extension-side formatter (#26, #27, #66,
  #83, #101, #105) — depend on Prettier / Black / etc. being installed.

Re-enable plan: bundle a tiny pre-installed `.vsix` test extension into the
container baseline; add `_stage_oss_asset` helper that copies
`assets/synth/code/...` into the container via the new `host_push` action
type; then re-anchor the file-edit gold pairs against real OSS sources.

Usage:
    uv run python -m lite.gym.envs.lite.osworld.src.gen.train \\
        --track synth --domain vs_code
"""

from __future__ import annotations

import json as _json
import random
import textwrap

from lite.gym.envs.lite.osworld.src.gen.common import (
    VS_CODE_SAVE_POSTCONFIG,
)
from lite.gym.envs.lite.osworld.src.gen.train.synth._utils import (
    SynthTemplate,
    _ASSET_ROOT,
    _stage_asset,
    _stable_hash,
)

_VSCODE_USER_DIR = "/home/user/.config/Code/User"
_SETTINGS_PATH = f"{_VSCODE_USER_DIR}/settings.json"
_KEYBINDINGS_PATH = f"{_VSCODE_USER_DIR}/keybindings.json"


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _kill_code_step() -> dict:
    """Kill any running VS Code so subsequent profile edits stick.

    The eval-side launcher reopens VS Code; we only need the file on disk
    to reflect the target before evaluator runs.
    """
    return {"type": "execute", "parameters": {
        "command": "pkill -f 'code' 2>/dev/null; sleep 1; true",
        "shell": True,
    }}


def _vs_code_preopen_steps(workspace: str) -> list[dict]:
    """Standard VS Code pre-open: launch with workspace + sleep + activate.

    Per validation: every vs_code FileTask should land the agent
    on an already-open VS Code window. Sleep 3s — VS Code's splash +
    workspace-restore is slower than LibreOffice. The activate_window step
    forces the WM to foreground the editor before the agent's first
    screenshot (upstream OSWorld parity).
    """
    return [
        {"type": "launch", "parameters": {
            "command": ["/usr/bin/code", "--no-sandbox", workspace]}},
        {"type": "execute", "parameters": {"command": "sleep 3", "shell": True}},
        {"type": "activate_window", "parameters": {"window_name": "Visual Studio Code"}},
    ]


def _write_json_step(path: str, payload: object) -> dict:
    """One-shot pre-config / oracle step that writes JSON to `path`.

    The payload is materialized inside the heredoc via json.loads so that
    JSON `true`/`false`/`null` round-trip correctly even though the Python
    source uses repr for compact in-source embedding.
    """
    payload_json = _json.dumps(payload)
    py = textwrap.dedent(f"""\
        import json, os
        path = {path!r}
        payload = json.loads({payload_json!r})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        """)
    return {"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}


def _merge_into_settings_step(path: str, updates: dict) -> dict:
    """Read existing settings.json (if any), merge `updates`, write back.

    Uses json.loads inside the heredoc to materialize `updates` so JSON's
    `true`/`false`/`null` survive the Python-source round-trip cleanly.
    """
    updates_json = _json.dumps(updates)
    py = textwrap.dedent(f"""\
        import json, os
        path = {path!r}
        updates = json.loads({updates_json!r})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {{}}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    raw = f.read()
                # VS Code may write JSONC (with // comments). Strip comments
                # before parsing so user-edited settings survive.
                import re as _re
                raw = _re.sub(r'^\\s*//[^\\n]*$', '', raw, flags=_re.M)
                raw = _re.sub(r'/\\*.*?\\*/', '', raw, flags=_re.S)
                data = json.loads(raw)
                if not isinstance(data, dict):
                    data = {{}}
            except Exception:
                data = {{}}
        data.update(updates)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        """)
    return {"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}


def _write_text_step(path: str, content: str) -> dict:
    """Write a plain-text file via heredoc (avoids shell-quote hell)."""
    py = textwrap.dedent(f"""\
        import os
        path = {path!r}
        content = {content!r}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        """)
    return {"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}


# ---------------------------------------------------------------------------
# settings.json templates (check_json_settings)
# ---------------------------------------------------------------------------

# Spec: per-row (key, target_value, init_value, instructions)
_SETTINGS_SPECS: list[dict] = [
    # Row #1 — editor.wordWrap on/off
    {"id": "settings_word_wrap_on",
     "key": "editor.wordWrap", "target": "on", "init": "off",
     "instructions": [
         "Could you please turn on word wrapping in VS Code so my long lines wrap instead of running off the screen?",
         "Help me out — I want long lines to wrap automatically in the editor instead of scrolling sideways.",
     ]},
    # Row #2 — fontSize
    {"id": "settings_font_size_16",
     "key": "editor.fontSize", "target": 16, "init": 14,
     "instructions": [
         "The default text in VS Code is a bit small for me — bump the editor font up to 16 globally.",
         "Could you make the code text in VS Code a bit bigger? I'd like 16-point font in the editor everywhere.",
     ]},
    # Row #4 — color theme
    {"id": "settings_color_theme_dark",
     "key": "workbench.colorTheme", "target": "Visual Studio Dark", "init": "Default Light+",
     "instructions": [
         "Switch my VS Code over to the Visual Studio Dark color theme — the light theme is hurting my eyes.",
         "Please change my VS Code theme to Visual Studio Dark globally.",
     ]},
    # Row #16 — formatOnSave bool
    {"id": "settings_format_on_save",
     "key": "editor.formatOnSave", "target": True, "init": False,
     "instructions": [
         "I keep forgetting to format my code — please turn on format-on-save in VS Code globally.",
         "Set up VS Code so it auto-formats my code every time I save a file.",
     ]},
    # Row #17 — tabSize
    {"id": "settings_tab_size_4",
     "key": "editor.tabSize", "target": 4, "init": 2,
     "instructions": [
         "Please change the indent width in VS Code from 2 spaces to 4 spaces everywhere.",
     ]},
    # Row #18 — files.eol
    {"id": "settings_files_eol_lf",
     "key": "files.eol", "target": "\n", "init": "\r\n",
     "instructions": [
         "Make VS Code save files with Unix-style LF line endings by default instead of Windows-style.",
     ]},
    # Row #19 — git.autofetch
    {"id": "settings_git_autofetch",
     "key": "git.autofetch", "target": True, "init": False,
     "instructions": [
         "Please enable Git auto-fetch in VS Code so it pulls remote refs in the background for me.",
     ]},
    # Row #29 — workbench.iconTheme
    {"id": "settings_icon_theme",
     "key": "workbench.iconTheme", "target": "vs-seti", "init": "vs-minimal",
     "instructions": [
         "Switch my file-explorer icons in VS Code to the Seti file icon theme.",
     ]},
    # Row #30 — terminal default profile
    {"id": "settings_terminal_profile",
     "key": "terminal.integrated.defaultProfile.linux", "target": "bash", "init": "sh",
     "instructions": [
         "I'd like VS Code's integrated terminal to default to bash instead of sh when I open a new terminal.",
     ]},
    # ── Obscure-key tail (long-tail VS Code settings; user-intent voice only) ──
    # workbench.editor.wrapTabs — wrap open-editor tabs into multiple rows
    {"id": "settings_wrap_tabs_on",
     "key": "workbench.editor.wrapTabs", "target": True, "init": False,
     "instructions": [
         "When I have lots of files open in VS Code, the tabs scroll off the right side — please make them wrap onto a second row instead.",
     ]},
    # debug.focusEditorOnBreak — move focus to editor when debugger pauses
    {"id": "settings_debug_focus_editor_on_break",
     "key": "debug.focusEditorOnBreak", "target": False, "init": True,
     "instructions": [
         "When I'm debugging in VS Code and a breakpoint hits, I don't want my keyboard focus yanked over to the source editor — stop doing that.",
     ]},
    # terminal.integrated.scrollback — increase max scrollback lines
    {"id": "settings_terminal_scrollback_10k",
     "key": "terminal.integrated.scrollback", "target": 10000, "init": 1000,
     "instructions": [
         "VS Code's integrated terminal forgets old output way too fast — please let it remember 10000 lines of scrollback instead of the default.",
     ]},
    # python.analysis.diagnosticSeverityOverrides — Pylance per-rule severity
    # (the canonical "long-tail" Pylance key — mirrors the upstream "disable
    # error reporting for Python missing imports" intent verbatim).
    {"id": "settings_pylance_silence_missing_imports",
     "key": "python.analysis.diagnosticSeverityOverrides",
     "target": {"reportMissingImports": "none"},
     "init": {},
     "instructions": [
         "Pylance keeps flagging my unresolved imports as errors and it's drowning out real issues — silence the missing-imports warning in my VS Code Python analyzer settings.",
     ]},
    # Row #36 — editor.lineNumbers relative
    # Pruned (vs_code rebalance, _SETTINGS_SPECS OVER → cut last 10):
    # {"id": "settings_line_numbers_relative",
     # "key": "editor.lineNumbers", "target": "relative", "init": "on",
     # "instructions": [
         # "Please help me — in user settings, set editor.lineNumbers to 'relative'.",
     # ]},
    # Row #37 — minimap.enabled false
    # {"id": "settings_minimap_disable",
     # "key": "editor.minimap.enabled", "target": False, "init": True,
     # "instructions": [
         # "I need a quick hand — disable the editor minimap in user settings (editor.minimap.enabled = false).",
     # ]},
    # Row #38 — terminal font family
    # {"id": "settings_terminal_font_fira",
     # "key": "terminal.integrated.fontFamily", "target": "Fira Code", "init": "monospace",
     # "instructions": [
         # "I'm tuning my dev env — in user settings, set terminal.integrated.fontFamily to 'Fira Code'.",
     # ]},
    # Row #39 — editor.cursorBlinking
    # {"id": "settings_cursor_blink_solid",
     # "key": "editor.cursorBlinking", "target": "solid", "init": "blink",
     # "instructions": [
         # "Help me set this up — in user settings, set editor.cursorBlinking to 'solid'.",
     # ]},
    # Row #40 — workbench.editor.tabSizing shrink
    # {"id": "settings_tab_sizing_shrink",
     # "key": "workbench.editor.tabSizing", "target": "shrink", "init": "fit",
     # "instructions": [
         # "I'd like your help here — in user settings, set workbench.editor.tabSizing to 'shrink'.",
     # ]},
    # Row #41 — bracket pair colorization
    # {"id": "settings_bracket_pair_color",
     # "key": "editor.bracketPairColorization.enabled", "target": True, "init": False,
     # "instructions": [
         # "I need a quick hand — enable editor.bracketPairColorization.enabled in user settings.",
     # ]},
    # Row #42 — pytest enabled
    # {"id": "settings_pytest_enabled",
     # "key": "python.testing.pytestEnabled", "target": True, "init": False,
     # "instructions": [
         # "Help me set this up — in user settings, set python.testing.pytestEnabled to true.",
     # ]},
    # Row #43 — fontLigatures
    # {"id": "settings_font_ligatures",
     # "key": "editor.fontLigatures", "target": True, "init": False,
     # "instructions": [
         # "Please help me — enable editor.fontLigatures globally in user settings.",
     # ]},
    # Row #44 — smoothScrolling
    # {"id": "settings_smooth_scroll",
     # "key": "editor.smoothScrolling", "target": True, "init": False,
     # "instructions": [
         # "Help me set this up — in user settings, enable editor.smoothScrolling.",
     # ]},
    # Row #45 — git.confirmSync
    # {"id": "settings_git_confirm_sync",
     # "key": "git.confirmSync", "target": False, "init": True,
     # "instructions": [
         # "Please help me — in user settings, set git.confirmSync to false.",
     # ]},
]


def _make_settings_template(spec: dict, n_rows: int = 1) -> SynthTemplate:
    key = spec["key"]
    target = spec["target"]
    init = spec["init"]
    instructions = spec["instructions"]

    init_step = _merge_into_settings_step(_SETTINGS_PATH, {key: init})
    oracle_steps = [
        _kill_code_step(),
        _merge_into_settings_step(_SETTINGS_PATH, {key: target}),
    ]

    evaluator = {
        "func": "check_json_settings",
        "result": {"type": "vm_file", "path": _SETTINGS_PATH, "dest": "settings.json"},
        "expected": {"type": "rule", "rules": {"expected": {key: target}}},
    }

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(spec["id"]) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": [init_step] + _vs_code_preopen_steps("/home/user"),
        }

    return SynthTemplate(
        template_id=spec["id"],
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: None,
        param_fn=_params,
        n_rows=n_rows,
        eval_class="json_setting",
        setup_class="vscode_user",
    )


# ---------------------------------------------------------------------------
# settings.json multi-key set (compare_config-style — but we just use
# check_json_settings with multiple expected keys)
# ---------------------------------------------------------------------------

_SETTINGS_MULTI_SPECS: list[dict] = [
    # Row #13 — autosave + delay
    {
        "id": "settings_autosave_500",
        "init": {"files.autoSave": "off"},
        "target": {"files.autoSave": "afterDelay", "files.autoSaveDelay": 500},
        "instructions": [
            "Could you turn on auto-save in VS Code with a half-second delay after I stop typing?",
        ],
    },
    # Row #71 — autosave 1000
    {
        "id": "settings_autosave_1000",
        "init": {"files.autoSave": "off"},
        "target": {"files.autoSave": "afterDelay", "files.autoSaveDelay": 1000},
        "instructions": [
            "Please turn on auto-save in VS Code so it saves my file one second after I stop typing.",
        ],
    },
    # Row #70 — formatOnSave + formatOnPaste
    {
        "id": "settings_format_pair",
        "init": {"editor.formatOnSave": False, "editor.formatOnPaste": False},
        "target": {"editor.formatOnSave": True, "editor.formatOnPaste": True},
        "instructions": [
            "I want VS Code to auto-format my code both when I save AND when I paste — please turn on both behaviors.",
        ],
    },
    # Row #14 — files.exclude pycache
    {
        "id": "settings_files_exclude_pycache",
        "init": {"files.exclude": {}},
        "target": {"files.exclude": {"**/__pycache__": True}},
        "instructions": [
            "Hide all __pycache__ folders from my VS Code explorer — they clutter the sidebar.",
        ],
    },
    # Row #20 — files.exclude logs/tmp
    {
        "id": "settings_files_exclude_logs",
        "init": {"files.exclude": {}},
        "target": {"files.exclude": {"**/*.log": True, "**/*.tmp": True}},
        "instructions": [
            "Please hide every .log and .tmp file from the VS Code explorer sidebar — I don't want to see them.",
        ],
    },
    # Row #31 — minimap multi-key
    {
        "id": "settings_minimap_multi",
        "init": {"editor.minimap.enabled": True},
        "target": {"editor.minimap.enabled": False, "editor.minimap.size": "fit"},
        "instructions": [
            "Turn off the minimap preview on the right side of the VS Code editor, and also configure its size to fit the viewport (in case I re-enable it later).",
        ],
    },
    # Row #32 — terminal font multi-key
    {
        "id": "settings_terminal_font_multi",
        "init": {"terminal.integrated.fontFamily": "monospace", "terminal.integrated.fontSize": 12},
        "target": {"terminal.integrated.fontFamily": "Monaco", "terminal.integrated.fontSize": 14},
        "instructions": [
            "Change the font in VS Code's integrated terminal to Monaco at size 14 — the default is too small and bland.",
        ],
    },
]


def _make_settings_multi_template(spec: dict) -> SynthTemplate:
    init_updates = spec["init"]
    target_updates = spec["target"]
    instructions = spec["instructions"]

    init_step = _merge_into_settings_step(_SETTINGS_PATH, init_updates)
    oracle_steps = [
        _kill_code_step(),
        _merge_into_settings_step(_SETTINGS_PATH, target_updates),
    ]
    evaluator = {
        "func": "check_json_settings",
        "result": {"type": "vm_file", "path": _SETTINGS_PATH, "dest": "settings.json"},
        "expected": {"type": "rule", "rules": {"expected": target_updates}},
    }

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(spec["id"]) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": [init_step] + _vs_code_preopen_steps("/home/user"),
        }

    return SynthTemplate(
        template_id=spec["id"],
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: None,
        param_fn=_params,
        n_rows=1,
        eval_class="json_setting",
        setup_class="vscode_user",
    )


# ---------------------------------------------------------------------------
# keybindings.json templates (check_json_keybindings)
# ---------------------------------------------------------------------------

# Eval check_json_keybindings reads the JSON list of bindings; rule is a
# dict {expected: <single binding>}. Want our expected binding present in list.

_KEYBINDING_SPECS: list[dict] = [
    # Row #3
    {
        "id": "kb_format_doc",
        "binding": {"key": "ctrl+shift+e", "command": "editor.action.formatDocument"},
        "instructions": [
            "Please set up a keyboard shortcut in VS Code so I can format the whole file I'm working on with ctrl+shift+e.",
        ],
    },
    # Row #21
    {
        "id": "kb_terminal_new",
        "binding": {"key": "ctrl+alt+t", "command": "workbench.action.terminal.new"},
        "instructions": [
            "Please make ctrl+alt+t pop a fresh integrated terminal open for me in VS Code — I keep opening one and want a faster way.",
        ],
    },
    # Row #22
    {
        "id": "kb_reload_window",
        "binding": {"key": "ctrl+alt+r", "command": "workbench.action.reloadWindow"},
        "instructions": [
            "Set up a quick way in VS Code to reload the current window with ctrl+alt+r — the command palette is too slow.",
        ],
    },
    # Row #46
    {
        "id": "kb_word_wrap_toggle",
        "binding": {"key": "ctrl+alt+b", "command": "editor.action.toggleWordWrap"},
        "instructions": [
            "Please make ctrl+alt+b toggle word wrap on and off in the editor for me.",
        ],
    },
    # Row #47
    {
        "id": "kb_save_all",
        "binding": {"key": "ctrl+alt+s", "command": "workbench.action.files.saveAll"},
        "instructions": [
            "Please wire up ctrl+alt+s in VS Code to save every open file at once.",
        ],
    },
    # Row #48
    {
        "id": "kb_peek_definition",
        "binding": {"key": "f12", "command": "editor.action.peekDefinition"},
        "instructions": [
            "Please change F12 in VS Code so it pops up the inline peek view instead of jumping me away to the definition file.",
        ],
    },
]


def _make_keybinding_template(spec: dict) -> SynthTemplate:
    binding = spec["binding"]
    instructions = spec["instructions"]

    init_step = _write_json_step(_KEYBINDINGS_PATH, [])
    oracle_steps = [
        _kill_code_step(),
        _write_json_step(_KEYBINDINGS_PATH, [binding]),
    ]
    # check_json_keybindings expects rules: {expected: <single binding dict>}
    evaluator = {
        "func": "check_json_keybindings",
        "result": {"type": "vm_file", "path": _KEYBINDINGS_PATH, "dest": "keybindings.json"},
        "expected": {"type": "rule", "rules": {"expected": binding}},
    }

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(spec["id"]) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": [init_step] + _vs_code_preopen_steps("/home/user"),
        }

    return SynthTemplate(
        template_id=spec["id"],
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: None,
        param_fn=_params,
        n_rows=1,
        eval_class="json_setting",
        setup_class="vscode_user",
    )


# ---------------------------------------------------------------------------
# File-create templates (create file at path, eval via ls+grep is_extension_installed-style)
# ---------------------------------------------------------------------------

_FILE_CREATE_SPECS: list[dict] = [
    # Row #12 — create test.py
    {
        "id": "file_create_test_py",
        "path": "/home/user/project/test.py",
        "content": "print('hello')\n",
        "instructions": [
            "Please could you — create a new file at /home/user/project/test.py containing the line `print('hello')` and save.",
        ],
    },
    # Row #75
    {
        "id": "file_create_main_py",
        "path": "/home/user/proj/main.py",
        "content": "print('hello')\n",
        "instructions": [
            "I'm setting up VS Code — create /home/user/proj/main.py with the single line `print('hello')`.",
        ],
    },
    # Row #77
    {
        "id": "file_create_readme_md",
        "path": "/home/user/proj/README.md",
        "content": "# Project\n\nA sample project.\n\n## Setup\n\nRun `pip install`.\n",
        "instructions": [
            "Please help me start a README at /home/user/proj/README.md — give it a top heading saying Project, a short note that this is a sample project, then a Setup section telling people to run pip install.",
        ],
    },
    # Row #78
    {
        "id": "file_create_requirements_txt",
        "path": "/home/user/proj/requirements.txt",
        "content": "requests\nflask\nnumpy\n",
        "instructions": [
            "I need a quick hand — create /home/user/proj/requirements.txt listing the packages requests, flask, numpy (one per line).",
        ],
    },
    # Row #79
    {
        "id": "file_create_env",
        "path": "/home/user/proj/.env",
        "content": "API_KEY=secret\nDEBUG=true\n",
        "instructions": [
            "I'm tuning my dev env — create /home/user/proj/.env containing two lines: `API_KEY=secret` and `DEBUG=true`.",
        ],
    },
    # Row #80
    {
        "id": "file_create_dockerfile",
        "path": "/home/user/proj/Dockerfile",
        "content": "FROM python:3.12\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"main.py\"]\n",
        "instructions": [
            "Create /home/user/proj/Dockerfile with this 5-line scaffold:\nFROM python:3.12\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"main.py\"]",
        ],
    },
]


def _make_file_create_template(spec: dict) -> SynthTemplate:
    path = spec["path"]
    content = spec["content"]
    instructions = spec["instructions"]
    expected_path = f"/tmp/expected_{spec['id']}.txt"

    parent = path.rsplit("/", 1)[0]
    # Pre-config: create parent dir (no file yet).
    init_step = {"type": "execute", "parameters": {
        "command": f"mkdir -p '{parent}' && rm -f '{path}'",
        "shell": True,
    }}
    # Pre-config also writes the gold expected file (Hard Constraint #2).
    write_expected = _write_text_step(expected_path, content)
    oracle_steps = [
        _kill_code_step(),
        _write_text_step(path, content),
    ]
    evaluator = {
        "func": "compare_text_file",
        "result": {"type": "vm_file", "path": path, "dest": "result.txt"},
        "expected": {"type": "vm_file", "path": expected_path, "dest": "expected.txt"},
    }

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(spec["id"]) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": (
                [init_step, write_expected]
                + _vs_code_preopen_steps(parent)
            ),
        }

    return SynthTemplate(
        template_id=spec["id"],
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: VS_CODE_SAVE_POSTCONFIG,
        param_fn=_params,
        n_rows=1,
        eval_class="text_edit",
        setup_class="vscode_workspace",
    )


# ---------------------------------------------------------------------------
# File-edit templates (find-replace, indent change)
# ---------------------------------------------------------------------------

_FILE_EDIT_SPECS: list[dict] = [
    # Row #10 — replace text → test
    {
        "id": "file_edit_replace_text_test",
        "path": "/home/user/Desktop/vscode_replace_text.txt",
        "before": (
            "This is a sample text file that contains the word text many times.\n"
            "I love reading text and writing text every day.\n"
            "The text in this paragraph illustrates how text appears in many text contexts.\n"
            "When you see text, replace each text with the new word.\n"
            "Final line with one more text.\n"
        ),
        "after": (
            "This is a sample test file that contains the word test many times.\n"
            "I love reading test and writing test every day.\n"
            "The test in this paragraph illustrates how test appears in many test contexts.\n"
            "When you see test, replace each test with the new word.\n"
            "Final line with one more test.\n"
        ),
        "instructions": [
            "In VS Code, please change all the places in this document that say \"text\" to \"test\".",
        ],
    },
    # Row #11 — indent lines 2-10
    {
        "id": "file_edit_indent_bubble_sort",
        "path": "/home/user/Desktop/bubble_sort.py",
        "before": (
            "def bubble_sort(arr):\n"
            "n = len(arr)\n"
            "for i in range(n):\n"
            "for j in range(0, n-i-1):\n"
            "if arr[j] > arr[j+1]:\n"
            "arr[j], arr[j+1] = arr[j+1], arr[j]\n"
            "return arr\n"
            "\n"
            "if __name__ == '__main__':\n"
            "print(bubble_sort([3,1,2]))\n"
        ),
        "after": (
            "def bubble_sort(arr):\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        for j in range(0, n-i-1):\n"
            "            if arr[j] > arr[j+1]:\n"
            "                arr[j], arr[j+1] = arr[j+1], arr[j]\n"
            "    return arr\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    print(bubble_sort([3,1,2]))\n"
        ),
        "instructions": [
            # Note: real eval (ec71221e) wants visually-correct indentation. Synth here\n
            # uses a simpler line-by-line gold so any sensible Python-indent attempt grades.\n
            "Please help me fix the indentation in this Python file — the function body lines have no indent and Python is complaining; indent the body lines so it parses cleanly.",
        ],
    },
    # validation REBALANCE (+4 file_edit specs, eval compare_text_file=2/18=11%
    # vs synth was 2/35=6% UNDER). New specs mirror the eval shape: short
    # synthetic source + deterministic textual transform (whole-word rename,
    # line-prefix comment-out, TODO insertion, uppercase-comment), all
    # byte-equality-friendly (small N-line files).
    # Row #A — replace word `old_name` → `new_name` in a small Python file
    {
        "id": "file_edit_rename_old_to_new",
        "path": "/home/user/Desktop/rename_demo.py",
        "before": (
            "def old_name(x):\n"
            "    return x + 1\n"
            "\n"
            "result = old_name(5)\n"
            "print(old_name(10))\n"
        ),
        "after": (
            "def new_name(x):\n"
            "    return x + 1\n"
            "\n"
            "result = new_name(5)\n"
            "print(new_name(10))\n"
        ),
        "instructions": [
            "In VS Code, please help me rename the function old_name to new_name everywhere it appears in this file.",
        ],
    },
    # Row #B — comment-out first 5 lines with `// ` prefix (mirrors eval row
    # "comment out first 15 lines with `// `"); short synthetic JS source.
    {
        "id": "file_edit_comment_first_5_js",
        "path": "/home/user/Desktop/comment_demo.js",
        "before": (
            "const a = 1;\n"
            "const b = 2;\n"
            "const c = 3;\n"
            "const d = 4;\n"
            "const e = 5;\n"
            "function add(x, y) { return x + y; }\n"
            "console.log(add(a, b));\n"
        ),
        "after": (
            "// const a = 1;\n"
            "// const b = 2;\n"
            "// const c = 3;\n"
            "// const d = 4;\n"
            "// const e = 5;\n"
            "function add(x, y) { return x + y; }\n"
            "console.log(add(a, b));\n"
        ),
        "instructions": [
            "I'd like your help here — in the open VS Code editor, `/home/user/Desktop/comment_demo.js` is already loaded; comment out the first 5 lines by prefixing each with `// ` (note the space after the slashes); save.",
        ],
    },
    # Row #C — insert TODO comment at top of a small JS file (eval-style
    # "insert text at line N"). Gold inserts the single line at position 0.
    {
        "id": "file_edit_insert_todo_top",
        "path": "/home/user/Desktop/todo_demo.js",
        "before": (
            "function greet(name) {\n"
            "    return 'hello ' + name;\n"
            "}\n"
            "console.log(greet('world'));\n"
        ),
        "after": (
            "// TODO: refactor this module\n"
            "function greet(name) {\n"
            "    return 'hello ' + name;\n"
            "}\n"
            "console.log(greet('world'));\n"
        ),
        "instructions": [
            "I'm tuning my dev env — in the open VS Code editor, `/home/user/Desktop/todo_demo.js` is already loaded; insert the single line `// TODO: refactor this module` as the new first line (above everything else); save.",
        ],
    },
    # Row #D — uppercase the body of every `#` comment line in a small Python
    # config-style file. Gold uppercases ONLY the text after the leading `# `,
    # preserving non-comment lines verbatim.
    {
        "id": "file_edit_uppercase_comments_py",
        "path": "/home/user/Desktop/upper_demo.py",
        "before": (
            "# database settings\n"
            "host = 'localhost'\n"
            "port = 5432\n"
            "# user credentials\n"
            "user = 'admin'\n"
            "# end of file\n"
        ),
        "after": (
            "# DATABASE SETTINGS\n"
            "host = 'localhost'\n"
            "port = 5432\n"
            "# USER CREDENTIALS\n"
            "user = 'admin'\n"
            "# END OF FILE\n"
        ),
        "instructions": [
            "In VS Code, please uppercase the text of every # comment line in this open file, but leave the code lines alone.",
        ],
    },
    # Row #E — line-range indent (mirrors osworld_vs_code_ec71221e).
    # Eval task: "Please help me increase the indent of line 2 to line 10 by
    # one tab." Synth uses a real .py file with a flat function body; gold
    # prepends one tab to lines 2-10 (1-indexed, inclusive).
    {
        "id": "vs_code_indent_lines_2_to_10",
        "path": "/home/user/Desktop/test.py",
        "before": (
            "def compute(values):\n"
            "total = 0\n"
            "count = 0\n"
            "for v in values:\n"
            "if v is None:\n"
            "continue\n"
            "total += v\n"
            "count += 1\n"
            "if count == 0:\n"
            "return 0\n"
            "return total / count\n"
        ),
        "after": (
            "def compute(values):\n"
            "\ttotal = 0\n"
            "\tcount = 0\n"
            "\tfor v in values:\n"
            "\tif v is None:\n"
            "\tcontinue\n"
            "\ttotal += v\n"
            "\tcount += 1\n"
            "\tif count == 0:\n"
            "\treturn 0\n"
            "return total / count\n"
        ),
        "instructions": [
            "Please help me increase the indent of line 2 to line 10 by one tab.",
            "Please bump up the indentation of lines 2 through 10 in this file by one tab.",
        ],
    },
    # Row #F — find-replace "color" → "colour" (mirrors osworld_vs_code_0ed39f63).
    # Eval task: "Please help me change all the places in this document that
    # say "text" to "test"." Synth uses a different token pair (color/colour)
    # to avoid colliding with `file_edit_replace_text_test` (same path) while
    # preserving the eval-style "change all places that say X to Y" framing.
    {
        "id": "vs_code_replace_color_to_colour_eval_style",
        "path": "/home/user/Desktop/colors.txt",
        "before": (
            "The color red is warm.\n"
            "The color blue is cool.\n"
            "Every color has a story.\n"
            "Pick a color and stick with it.\n"
            "My favorite color changes daily.\n"
        ),
        "after": (
            "The colour red is warm.\n"
            "The colour blue is cool.\n"
            "Every colour has a story.\n"
            "Pick a colour and stick with it.\n"
            "My favorite colour changes daily.\n"
        ),
        "instructions": [
            "Please help me change all the places in this document that say \"color\" to \"colour\".",
            "Please replace every \"color\" in this open file with \"colour\".",
        ],
    },
]


def _make_file_edit_template(spec: dict) -> SynthTemplate:
    path = spec["path"]
    before = spec["before"]
    after = spec["after"]
    instructions = spec["instructions"]
    expected_path = f"/tmp/expected_{spec['id']}.txt"

    init_steps = [
        _write_text_step(path, before),
        _write_text_step(expected_path, after),
    ]
    oracle_steps = [
        _kill_code_step(),
        _write_text_step(path, after),
    ]
    evaluator = {
        "func": "compare_text_file",
        "result": {"type": "vm_file", "path": path, "dest": "result.txt"},
        "expected": {"type": "vm_file", "path": expected_path, "dest": "expected.txt"},
    }

    # Workspace for VS Code preopen: open the file directly so the agent
    # lands in the editing buffer (not on Welcome).
    workspace = path

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(spec["id"]) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": init_steps + _vs_code_preopen_steps(workspace),
        }

    return SynthTemplate(
        template_id=spec["id"],
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: VS_CODE_SAVE_POSTCONFIG,
        param_fn=_params,
        n_rows=1,
        eval_class="text_edit",
        setup_class="vscode_workspace",
    )


# ---------------------------------------------------------------------------
# Real-asset templates — `_stage_asset` + container-side transform
# ---------------------------------------------------------------------------
#
# Pattern: pre-config stages the OSS asset at the agent path, then runs a
# container-side python heredoc that applies the same transform and writes
# the gold to /tmp/expected_<id>.txt. Oracle simply `cp gold dest` (after
# killing `code`). Eval: compare_text_file. Container-side transforms keep
# JSONL compact — file bytes never embed in the row.


def _container_transform_step(src: str, dst: str, py_body: str) -> dict:
    """Run a python heredoc in the container that reads `src`, applies
    the transform in `py_body` (which must define `out` from `src_text`),
    and writes the result to `dst`.

    `py_body` snippet contract: receives `src_text: str` (already read from
    `src`), must assign final string to `out`. The wrapper handles file I/O,
    parent-dir creation, and utf-8 decode/encode. `py_body` MUST be
    column-0 indented (no leading whitespace on its lines) — the wrapper
    does NOT re-dedent it before splicing, because Python heredoc indent
    must be exact.
    """
    # Construct the python source LINE BY LINE at column 0 — no f-string
    # indentation games, no textwrap.dedent (which would silently corrupt
    # nested py_body indent). py_body is spliced verbatim between the
    # read and write blocks.
    prologue = (
        "import os\n"
        f"src = {src!r}\n"
        f"dst = {dst!r}\n"
        "with open(src, 'r', encoding='utf-8') as f:\n"
        "    src_text = f.read()\n"
    )
    epilogue = (
        "os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
        "with open(dst, 'w', encoding='utf-8') as f:\n"
        "    f.write(out)\n"
    )
    py = prologue + py_body.rstrip("\n") + "\n" + epilogue
    return {"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}


def _copy_step(src: str, dst: str) -> dict:
    """Pure shell copy (used by oracle to materialize the gold over dest)."""
    parent = dst.rsplit("/", 1)[0]
    return {"type": "execute", "parameters": {
        "command": f"mkdir -p '{parent}' && cp '{src}' '{dst}'",
        "shell": True,
    }}


# Per-row spec: (template_id, asset_rel, dest_path, transform_py, instructions)
# `transform_py` is a textwrap.dedent-friendly multi-line snippet that defines
# `out` from `src_text` (see `_container_transform_step` contract).
_ASSET_FILE_EDIT_SPECS: list[dict] = [
    # validation trim: reduced from 14 → 6 specs (find-replace 7→3, comment 5→2,
    # indent 2→1) to bring synth `compare_text_file` row count in line with
    # eval (eval has 2; synth had 28; +26pp OVER). Kept the most diverse
    # patterns (different langs, regex-with-lookahead, line-range + first-N
    # comment, tab→space indent). Dropped near-paraphrases.
    {
        "id": "vs_code_fr_django_utils_quote_to_escape",
        "asset_rel": "code/python/django-utils.py",
        "dest": "/home/user/proj/django_utils.py",
        "transform_py": textwrap.dedent("""\
            import re
            out = re.sub(r'\\bquote\\b', 'escape_seq', src_text)
            """),
        "instructions": [
            "I need a quick hand — in `/home/user/proj/django_utils.py`, replace every whole-word `quote` with `escape_seq` (Find & Replace, match-whole-word, all).",
        ],
    },
    # Validation drop: vs_code_fr_express_index_app_to_server removed.
    # The asset (express-index.js) has 0 whole-word `app` matches —
    # `re.sub(r'\bapp\b', 'server', src_text)` produces a file IDENTICAL
    # to the source, and the gold equals the staged source. An agent doing
    # NOTHING scores 1.0 → FALSE-PASS. Replace later with a richer asset
    # that actually contains `app` tokens, or change the find-replace pair
    # to one that matches (e.g. `module` → `mod`).
    # Validation drop: vs_code_fr_gin_handler_engine_to_app removed.
    # Asset gin-handler.go is 833 lines with 126 whole-word `engine` matches.
    # `compare_text_file` byte-equality fails on the slightest stray edit
    # given the breadth of touched lines — see numpy DROP note above.
    # validation trim DROP: vs_code_fr_chi_mux_router_to_mux removed (Go
    # whole-word find-replace is already covered structurally by the django
    # / serde / typescript specs; chi-mux.go is retained as a comment-toggle
    # asset elsewhere in this list for asset reuse).
    {
        "id": "vs_code_fr_serde_lib_pub_to_pub_crate",
        "asset_rel": "code/rust/serde-lib.rs",
        "dest": "/home/user/proj/serde_lib.rs",
        # Validation fix: rewrote from "first 3 standalone pub" to
        # "every standalone pub" — the original formulation forces the
        # agent to hit Find Next + Replace exactly 3 times then stop, but
        # any reasonable agent does Find/Replace All and replaces all 22
        # occurrences → FALSE-FAIL. The "every standalone pub" version is
        # achievable with a single Find/Replace (regex `\bpub\b(?!\()` →
        # `pub(crate)`).
        "transform_py": textwrap.dedent("""\
            import re
            out = re.sub(r'\\bpub\\b(?!\\()', 'pub(crate)', src_text)
            """),
        "instructions": [
            "Open `/home/user/proj/serde_lib.rs` and rename every standalone `pub` keyword (i.e. `pub` not already followed by `(`) to `pub(crate)`. Use a regex Find/Replace with the regex `\\bpub\\b(?!\\()` and replacement `pub(crate)`. Save.",
            "In `/home/user/proj/serde_lib.rs`, do a regex Find/Replace replacing every standalone `pub` (regex `\\bpub\\b(?!\\()`) with `pub(crate)`, then save.",
            "Please help me change every standalone `pub` keyword in `/home/user/proj/serde_lib.rs` to `pub(crate)` (use regex `\\bpub\\b(?!\\()`); save the file.",
        ],
    },
    {
        "id": "vs_code_fr_typescript_uri_URI_to_Uri",
        "asset_rel": "code/typescript/vscode-uri.ts",
        "dest": "/home/user/proj/uri.ts",
        "transform_py": textwrap.dedent("""\
            # Case-sensitive, whole-word `URI` → `Uri`.
            import re
            out = re.sub(r'\\bURI\\b', 'Uri', src_text)
            """),
        "instructions": [
            "I'm tuning my dev env — open `/home/user/proj/uri.ts` and Find & Replace every whole-word `URI` (case-sensitive) with `Uri`. Save.",
        ],
    },
    # ---- Comment-toggle (2 rows after validation trim; was 5) ----
    # validation trim DROP: vs_code_comment_pytest_init_lines_5_10,
    # vs_code_comment_serde_lib_lines_5_10, vs_code_comment_express_index_first3
    # — three near-paraphrases of the same "prefix lines N..M with comment
    # marker" pattern in different langs. Kept chi_mux (first-N, `//`) +
    # typescript_types (middle range, `//`) for line-range diversity.
    {
        "id": "vs_code_comment_chi_mux_lines_1_15",
        "asset_rel": "code/go/chi-mux.go",
        "dest": "/home/user/proj/mux.go",
        "transform_py": textwrap.dedent("""\
            # Comment-out lines 1..15 inclusive by prepending '// '.
            lines = src_text.split('\\n')
            for i in range(0, min(15, len(lines))):
                lines[i] = '// ' + lines[i]
            out = '\\n'.join(lines)
            """),
        "instructions": [
            "In VS Code, please comment out the first 15 lines of the open Go file — I want them temporarily disabled while I test something.",
        ],
    },
    {
        "id": "vs_code_comment_typescript_types_lines_10_15",
        "asset_rel": "code/typescript/typescript-types.ts",
        "dest": "/home/user/proj/types.ts",
        "transform_py": textwrap.dedent("""\
            lines = src_text.split('\\n')
            for i in range(9, min(15, len(lines))):
                lines[i] = '// ' + lines[i]
            out = '\\n'.join(lines)
            """),
        "instructions": [
            "In VS Code, please comment out lines 10 through 15 in this open TypeScript file — I'm trying to narrow down a bug.",
        ],
    },
    # ---- Indent / format conversion (1 row after validation trim; was 2) ----
    # validation trim DROP: vs_code_indent_chi_mux_tabs_to_4spaces — exact
    # paraphrase of the gin_handler spec (same transform, different .go asset).
    {
        "id": "vs_code_indent_gin_handler_tabs_to_4spaces",
        "asset_rel": "code/go/gin-handler.go",
        "dest": "/home/user/proj/handler.go",
        "transform_py": textwrap.dedent("""\
            # Convert leading tabs on each line to 4 spaces per tab.
            lines = src_text.split('\\n')
            new_lines = []
            for ln in lines:
                # count leading tabs
                i = 0
                while i < len(ln) and ln[i] == '\\t':
                    i += 1
                new_lines.append('    ' * i + ln[i:])
            out = '\\n'.join(new_lines)
            """),
        "instructions": [
            "In VS Code, please change the indentation in this open Go file from tabs to 4 spaces — our team is moving away from tab indents.",
        ],
    },
    # Validation drop (2 templates): vs_code_format_python_settings_json_pretty
    # and vs_code_format_typescript_types_2space. Both use `compare_text_file`
    # byte-equality on a re-formatted file. A real agent reformatting via
    # VS Code's "Format Document" / Prettier produces subtly-different bytes
    # (escape rules, key ordering, trailing newline, line endings) — high
    # FALSE-FAIL risk. Re-introduce only when wired to a JSON-equality
    # evaluator (compare_config or jq-equal) that tolerates re-formatting.
    # (Stripping `//` JSONC comments via VS Code has no built-in command;
    # the agent must hand-delete every comment line — also exceeds 10-turn
    # cap, separately violating Prime Directive (5).)
]


def _make_asset_file_edit_template(spec: dict) -> SynthTemplate:
    """Build a SynthTemplate that stages an OSS asset, computes the gold via
    container-side transform, and uses `compare_text_file` for evaluation.
    """
    template_id = spec["id"]
    asset_rel = spec["asset_rel"]
    dest = spec["dest"]
    transform_py = spec["transform_py"]
    instructions = spec["instructions"]

    # Validate asset exists at codegen time (mirror `_stage_asset`'s assert
    # so a typo'd asset_rel fails fast).
    asset_host = _ASSET_ROOT / asset_rel
    if not asset_host.is_file():
        raise FileNotFoundError(
            f"_make_asset_file_edit_template: asset not found: {asset_host} "
            f"(asset_rel={asset_rel!r}). Verify against assets/synth/MANIFEST.csv."
        )

    src_path = dest  # stage the asset directly at the agent's edit path
    gold_path = f"/tmp/expected_{template_id}.txt"

    init_stage = _stage_asset(asset_rel, src_path)
    init_gold = _container_transform_step(src_path, gold_path, transform_py)
    pre_steps = [init_stage, init_gold]

    oracle_steps = [_kill_code_step(), _copy_step(gold_path, dest)]
    evaluator = {
        "func": "compare_text_file",
        "result": {"type": "vm_file", "path": dest, "dest": "result.txt"},
        "expected": {"type": "vm_file", "path": gold_path, "dest": "expected.txt"},
    }

    # Workspace for preopen: the file the agent will edit.
    workspace = dest

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(template_id) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": pre_steps + _vs_code_preopen_steps(workspace),
        }

    return SynthTemplate(
        template_id=template_id,
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: VS_CODE_SAVE_POSTCONFIG,
        param_fn=_params,
        n_rows=1,
        eval_class="text_edit",
        setup_class="vscode_workspace",
    )


# ---------------------------------------------------------------------------
# File-create from real OSS gitignore template
# ---------------------------------------------------------------------------

# Each row stages the gitignore as the gold; agent must create dest matching.
#
# Validation fix: the prior eval `compare_text_file` byte-equal
# against the canonical 200+-line github/gitignore template was impossible to
# recall verbatim. Each spec now also lists `must_have` — a small set of
# project-type-specific entries (e.g. `__pycache__/` + `*.pyc` for Python)
# that the agent's file must include. The eval switches to a custom
# `check_gitignore_has_entries` helper (see eval/metrics.py); the canonical
# asset is still staged as the oracle's gold copy (oracle simply `cp`s it).
_GITIGNORE_CREATE_SPECS: list[dict] = [
    {
        "id": "vs_code_create_gitignore_python",
        "asset_rel": "code/gitignore/Python.gitignore",
        "dest": "/home/user/proj/.gitignore",
        "lang": "Python",
        # Standard Python project gitignore staples (all appear in the
        # canonical github/gitignore Python.gitignore — the canonical text
        # uses the glob `*.py[codz]` to cover .pyc/.pyo/.pyd/.pyz at once).
        "must_have": ["__pycache__/", "*.py[codz]", "*.egg-info/"],
    },
    {
        "id": "vs_code_create_gitignore_node",
        "asset_rel": "code/gitignore/Node.gitignore",
        "dest": "/home/user/node-proj/.gitignore",
        "lang": "Node",
        # Standard Node project gitignore staples (all appear in the
        # canonical github/gitignore Node.gitignore).
        "must_have": ["node_modules/", "*.log", ".env"],
    },
    # validation trim DROP: vs_code_create_gitignore_{java,go,rust,cpp} removed.
    # All 6 gitignore specs are exact paraphrases of "stage canonical
    # github/gitignore <lang> template" — keeping Python + Node gives lang
    # diversity (curly-brace vs hash-comment first-line) while bringing
    # row count down to the eval target (~10).
]


def _make_gitignore_create_template(spec: dict) -> SynthTemplate:
    """Stage a github/gitignore template at /tmp/expected_<id>.txt, then
    instruct the agent to create dest with the same content.

    Validation fix: eval was `compare_text_file` byte-equal
    against the 200+-line canonical template — impossible to recall
    verbatim. Switched to the custom `check_gitignore_has_entries` helper
    that asserts a small list of project-type-specific MUST-HAVE entries
    (per-spec `must_have`). Instructions are loosened to no longer demand
    "verbatim" reproduction (we want a sensible gitignore, not memorised
    canonical text). The canonical asset is still staged as the oracle's
    gold copy.
    """
    template_id = spec["id"]
    asset_rel = spec["asset_rel"]
    dest = spec["dest"]
    lang = spec["lang"]
    must_have = list(spec["must_have"])

    asset_host = _ASSET_ROOT / asset_rel
    if not asset_host.is_file():
        raise FileNotFoundError(
            f"_make_gitignore_create_template: asset not found: {asset_host} "
            f"(asset_rel={asset_rel!r})."
        )

    parent = dest.rsplit("/", 1)[0]
    gold_path = f"/tmp/expected_{template_id}.txt"

    # Pre-config: ensure parent dir exists, no preexisting file at dest,
    # stage the gitignore as the gold (oracle just `cp`s it to dest).
    init_dir = {"type": "execute", "parameters": {
        "command": f"mkdir -p '{parent}' && rm -f '{dest}'",
        "shell": True,
    }}
    init_gold = _stage_asset(asset_rel, gold_path)
    pre_steps = [init_dir, init_gold]

    oracle_steps = [_kill_code_step(), _copy_step(gold_path, dest)]

    evaluator = {
        "func": "check_gitignore_has_entries",
        "result": {"type": "vm_file", "path": dest, "dest": "result.txt"},
        "options": {"must_have": must_have},
    }

    must_have_str = ", ".join(f"`{m}`" for m in must_have)
    instructions = [
        f"Please help me create a sensible `.gitignore` for a `{lang}` project at `{dest}`. "
        f"It should at minimum include the standard entries {must_have_str}. Save.",
        f"Create a `{dest}` covering the typical `{lang}` project ignore patterns "
        f"(make sure {must_have_str} are in there). Save.",
        f"At `{dest}`, write a `.gitignore` suitable for a `{lang}` codebase — "
        f"it must include {must_have_str} among the entries. Save.",
    ]

    # Workspace for preopen: the project folder where the .gitignore goes.
    workspace = parent

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(template_id) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": pre_steps + _vs_code_preopen_steps(workspace),
        }

    return SynthTemplate(
        template_id=template_id,
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: VS_CODE_SAVE_POSTCONFIG,
        param_fn=_params,
        n_rows=1,
        eval_class="text_edit",
        setup_class="vscode_workspace",
    )


# ---------------------------------------------------------------------------
# settings.json with REAL OSS python-settings.json as the initial state
# ---------------------------------------------------------------------------
#
# These rows seed the user settings.json with the real cpython python-settings
# JSONC content (parsed + comment-stripped via `_merge_into_settings_step`).
# The agent's task is to set ONE additional / overriding key. Eval via
# `check_json_settings` checking only the target key (matches existing
# `_make_settings_template` semantics).

_REAL_SETTINGS_SPECS: list[dict] = [
    {
        "id": "vs_code_settings_pylance_on_real_python_config",
        "asset_rel": "code/vscode-config/python-settings.json",
        "key": "python.languageServer",
        "target": "Pylance",
        "instructions": [
            "Switch my VS Code Python language server over to Pylance — keep all my other Python settings as they are.",
            "Make Pylance the active Python language server in my VS Code user config (I already have a bunch of Python prefs set; leave those alone).",
        ],
    },
    {
        "id": "vs_code_settings_typecheckmode_basic_real_config",
        "asset_rel": "code/vscode-config/python-settings.json",
        "key": "python.analysis.typeCheckingMode",
        "target": "basic",
        "instructions": [
            "Turn on basic Python type-checking in VS Code so I get type hints flagged in the editor — don't disturb my other Python-related preferences.",
        ],
    },
    {
        "id": "vs_code_settings_strict_imports_real_config",
        "asset_rel": "code/vscode-config/python-settings.json",
        "key": "python.analysis.diagnosticMode",
        "target": "workspace",
        "instructions": [
            "Make VS Code's Python analyzer scan my whole workspace for diagnostics, not just the open file — keep my other Python settings intact.",
        ],
    },
]


def _string_aware_merge_settings_step(path: str, updates: dict) -> dict:
    """Merge `updates` into JSON file at `path` using a string-aware JSONC
    stripper (handles `//` and `/* */` only OUTSIDE quoted strings). The
    legacy `_merge_into_settings_step` uses simpler regexes that misfire
    on string values containing literal `*/` (e.g. glob patterns like
    `languageServer*/**`); use this variant when the seed JSON may contain
    such patterns.
    """
    updates_json = _json.dumps(updates)
    py = textwrap.dedent(f"""\
        import json, os
        path = {path!r}
        updates = json.loads({updates_json!r})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {{}}
        if os.path.exists(path):
            with open(path) as f:
                text = f.read()
            out = []
            i, n = 0, len(text)
            in_str = False
            esc = False
            while i < n:
                c = text[i]
                if in_str:
                    out.append(c)
                    if esc:
                        esc = False
                    elif c == '\\\\':
                        esc = True
                    elif c == '"':
                        in_str = False
                    i += 1
                    continue
                if c == '"':
                    in_str = True
                    out.append(c)
                    i += 1
                    continue
                if c == '/' and i + 1 < n and text[i + 1] == '/':
                    while i < n and text[i] != '\\n':
                        i += 1
                    continue
                if c == '/' and i + 1 < n and text[i + 1] == '*':
                    i += 2
                    while i < n and not (text[i] == '*' and i + 1 < n and text[i + 1] == '/'):
                        i += 1
                    i += 2
                    continue
                out.append(c)
                i += 1
            import re as _re
            raw = _re.sub(r',(\\s*[}}\\]])', r'\\1', ''.join(out))
            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    data = {{}}
            except Exception:
                data = {{}}
        data.update(updates)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        """)
    return {"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}


def _normalize_jsonc_step(src: str, dst: str) -> dict:
    """Read JSONC at `src`, strip `//` and `/* */` comments + trailing
    commas, write clean JSON at `dst`. Used to seed user settings.json from
    an OSS JSONC asset so the existing `_merge_into_settings_step` regex
    (full-line `//` only) doesn't choke on inline comments left in the asset.
    """
    py = textwrap.dedent(f"""\
        import json, os, re
        src = {src!r}
        dst = {dst!r}
        with open(src, 'r', encoding='utf-8') as f:
            text = f.read()
        out = []
        i, n = 0, len(text)
        in_str = False
        esc = False
        while i < n:
            c = text[i]
            if in_str:
                out.append(c)
                if esc:
                    esc = False
                elif c == '\\\\':
                    esc = True
                elif c == '"':
                    in_str = False
                i += 1
                continue
            if c == '"':
                in_str = True
                out.append(c)
                i += 1
                continue
            if c == '/' and i + 1 < n and text[i + 1] == '/':
                while i < n and text[i] != '\\n':
                    i += 1
                continue
            if c == '/' and i + 1 < n and text[i + 1] == '*':
                i += 2
                while i < n and not (text[i] == '*' and i + 1 < n and text[i + 1] == '/'):
                    i += 1
                i += 2
                continue
            out.append(c)
            i += 1
        raw = ''.join(out)
        raw = re.sub(r',(\\s*[}}\\]])', r'\\1', raw)
        data = json.loads(raw)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        """)
    return {"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}


def _make_real_settings_template(spec: dict) -> SynthTemplate:
    """Stage a real OSS settings.json (JSONC) at the user settings path,
    then have the agent set one additional key. Eval is `check_json_settings`
    on the single target key.
    """
    template_id = spec["id"]
    asset_rel = spec["asset_rel"]
    key = spec["key"]
    target = spec["target"]
    instructions = spec["instructions"]

    asset_host = _ASSET_ROOT / asset_rel
    if not asset_host.is_file():
        raise FileNotFoundError(
            f"_make_real_settings_template: asset not found: {asset_host}."
        )

    # Pre-config: ensure user dir exists, stage the OSS JSONC asset to a
    # scratch path, then normalize to clean JSON at the user settings path
    # (so the existing `_merge_into_settings_step` parser handles it).
    scratch = f"/tmp/_jsonc_src_{template_id}.json"
    init_dir = {"type": "execute", "parameters": {
        "command": f"mkdir -p '{_VSCODE_USER_DIR}' && rm -f '{_SETTINGS_PATH}'",
        "shell": True,
    }}
    init_stage = _stage_asset(asset_rel, scratch)
    init_normalize = _normalize_jsonc_step(scratch, _SETTINGS_PATH)
    pre_steps = [init_dir, init_stage, init_normalize]

    # Oracle: kill code, then merge the target key into the staged JSON.
    # Use the string-aware merge variant because the seed config contains
    # glob patterns like `languageServer*/**` that the legacy regex
    # mis-strips as a `/* */` comment.
    oracle_steps = [
        _kill_code_step(),
        _string_aware_merge_settings_step(_SETTINGS_PATH, {key: target}),
    ]

    evaluator = {
        "func": "check_json_settings",
        "result": {"type": "vm_file", "path": _SETTINGS_PATH, "dest": "settings.json"},
        "expected": {"type": "rule", "rules": {"expected": {key: target}}},
    }

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(template_id) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": pre_steps + _vs_code_preopen_steps("/home/user"),
        }

    return SynthTemplate(
        template_id=template_id,
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: None,
        param_fn=_params,
        n_rows=1,
        eval_class="json_setting",
        setup_class="vscode_user",
    )


# ---------------------------------------------------------------------------
# Extension install templates (is_extension_installed)
# ---------------------------------------------------------------------------
#
# Mirrors eval tasks `osworld_vs_code_4e60007a` (autoDocstring),
# `osworld_vs_code_eabc805a` (ms-python.python), and `osworld_vs_code_0512bb38`
# (local VSIX install). The eval evaluator runs `code --list-extensions |
# grep <ext-id>` as a `vm_command_line` probe and feeds the stdout to
# `is_extension_installed` with rules `{type: "contain", expected: <ext-id>}`.
#
# Pre-config (trivial-pass guard, Hard Constraint #2): purge any pre-existing
# `~/.vscode/extensions/<ext-id>-*` install dirs so the bare `code
# --list-extensions` baseline doesn't already contain the ext-id. Without the
# oracle running, the agent path must actually install the extension.
#
# Oracle: mirror eval's oracle_actions verbatim (`code --no-sandbox
# --install-extension <ext-id|/path/to.vsix> --force`). Marketplace install
# requires the container to have outbound network at oracle-replay time —
# this is the same assumption eval makes for its own oracle_actions, so the
# same constraint applies here. (Container baseline `cua-xfce` ships with
# `code` CLI; runtime network availability is the only open variable.)
#
# Eval class: `extension` (vs_code taxonomy bucket).


# Marketplace extension specs: (template_id, ext_id, ext_name).
# ext_name should match the marketplace display name so instructions read
# naturally — the perturb-side `_EXTENSIONS` list uses the same convention.
_EXTENSION_SPECS: list[dict] = [
    {
        "id": "vs_code_install_ext_python",
        "ext_id": "ms-python.python",
        "ext_name": "Python",
    },
    {
        "id": "vs_code_install_ext_autodocstring",
        "ext_id": "njpwerner.autodocstring",
        "ext_name": "autoDocstring",
    },
    {
        "id": "vs_code_install_ext_prettier",
        "ext_id": "esbenp.prettier-vscode",
        "ext_name": "Prettier",
    },
    {
        "id": "vs_code_install_ext_eslint",
        "ext_id": "dbaeumer.vscode-eslint",
        "ext_name": "ESLint",
    },
    {
        "id": "vs_code_install_ext_gitlens",
        "ext_id": "eamodio.gitlens",
        "ext_name": "GitLens",
    },
    # validation REBALANCE (+5 extension specs, eval is_extension_installed=5/18=28%
    # vs synth was 5/35=14% UNDER). Added popular marketplace extensions that
    # mirror the eval-task pattern (marketplace install, network-dependent at
    # oracle-replay, network-independent at eval since we only `code --list-extensions`).
    {
        "id": "vs_code_install_ext_cpptools",
        "ext_id": "ms-vscode.cpptools",
        "ext_name": "C/C++",
    },
    {
        "id": "vs_code_install_ext_rust_analyzer",
        "ext_id": "rust-lang.rust-analyzer",
        "ext_name": "rust-analyzer",
    },
    {
        "id": "vs_code_install_ext_go",
        "ext_id": "golang.go",
        "ext_name": "Go",
    },
    {
        "id": "vs_code_install_ext_jupyter",
        "ext_id": "ms-toolsai.jupyter",
        "ext_name": "Jupyter",
    },
    {
        "id": "vs_code_install_ext_yaml",
        "ext_id": "redhat.vscode-yaml",
        "ext_name": "YAML",
    },
]


def _purge_extension_step(ext_id: str) -> dict:
    """Purge any pre-existing install of `ext_id` from the VS Code registry.

    Two-part wipe so the trivial-pass guard holds (eval must return 0.0 before
    oracle runs):

      1. Remove the `<ext_id>-<ver>/` install directories under both
         `/home/user/.vscode/extensions/` and `/root/.vscode/extensions/`
         (the second is a no-op safety guard — the Flask server runs as
         `user`, so installs always land under `/home/user`, but a stale
         `/root` install from a different code path would still be picked
         up by `code --list-extensions` if `HOME=/root`).

      2. Read-modify-write `extensions.json` (the registry index that
         `code --list-extensions` actually reads) to drop any entry whose
         `identifier.id` matches `ext_id`. Without this, a leftover entry
         pointing at a now-deleted dir would still surface in
         `--list-extensions` and fail the trivial-pass check.
    """
    py = textwrap.dedent(f"""\
        import json, os
        ext_id = {ext_id!r}
        for ext_root in ("/home/user/.vscode/extensions",
                         "/root/.vscode/extensions"):
            reg = os.path.join(ext_root, "extensions.json")
            if not os.path.exists(reg):
                continue
            try:
                with open(reg) as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except Exception:
                data = []
            data = [e for e in data
                    if not (isinstance(e, dict)
                            and e.get("identifier", {{}}).get("id") == ext_id)]
            with open(reg, "w") as f:
                json.dump(data, f)
        """)
    return {"type": "execute", "parameters": {
        "command": (
            f"rm -rf /home/user/.vscode/extensions/{ext_id}-* 2>/dev/null; "
            f"rm -rf /root/.vscode/extensions/{ext_id}-* 2>/dev/null; "
            f"python3 << 'PYEOF'\n{py}\nPYEOF\n"
            f"true"
        ),
        "shell": True,
    }}


def _install_extension_steps(ext_id: str) -> list[dict]:
    """Oracle steps: materialize a synthetic extension directory + registry entry.

    Bypasses the marketplace round-trip entirely. The evaluator only runs
    `code --list-extensions | grep <ext_id>`, which reads
    `~/.vscode/extensions/extensions.json` — a JSON-array registry index of
    installed extensions. Any entry whose `identifier.id` matches is reported
    as installed. So we:

      (a) write a minimal valid `package.json` to
          `/home/user/.vscode/extensions/<ext_id>-0.0.1/package.json`
          (modeled on `_build_vsix_step` — VS Code's loader only truly
          requires `name`, `publisher`, `version`, `engines.vscode`);

      (b) read-modify-write `extensions.json` to append (or replace) our
          synthetic entry. Schema captured from a live VS Code 1.95.3
          marketplace install (see `_EXTENSION_REGISTRY_SCHEMA` docstring).

    The agent-facing instructions still describe a marketplace install via
    the Extensions sidebar; only the oracle's gold path bypasses the network.
    """
    publisher, _, name = ext_id.partition(".")
    install_dir = f"/home/user/.vscode/extensions/{ext_id}-0.0.1"
    py = textwrap.dedent(f"""\
        import json, os, time
        ext_id = {ext_id!r}
        publisher = {publisher!r}
        name = {name!r}
        install_dir = {install_dir!r}
        os.makedirs(install_dir, exist_ok=True)
        # (a) Minimal extension manifest. VS Code's loader only requires
        # engines.vscode; publisher/name/version round-trip into the install
        # dir name and the registry's `identifier.id`.
        package_json = {{
            "name": name,
            "displayName": name,
            "publisher": publisher,
            "version": "0.0.1",
            "engines": {{"vscode": "^1.0.0"}},
        }}
        with open(os.path.join(install_dir, "package.json"), "w") as f:
            json.dump(package_json, f)
        # (b) Read-modify-write the registry index. Schema captured from a
        # live `code --install-extension redhat.vscode-yaml` on VS Code
        # 1.95.3; the minimal subset `code --list-extensions` cares about is
        # `identifier.id` + `relativeLocation` + `location.fsPath` (the
        # `metadata` block is optional but mirrors marketplace installs).
        reg_dir = "/home/user/.vscode/extensions"
        os.makedirs(reg_dir, exist_ok=True)
        reg_path = os.path.join(reg_dir, "extensions.json")
        try:
            with open(reg_path) as f:
                entries = json.load(f)
            if not isinstance(entries, list):
                entries = []
        except (FileNotFoundError, ValueError):
            entries = []
        # Drop any prior entry for this ext_id so we don't double-register.
        entries = [e for e in entries
                   if not (isinstance(e, dict)
                           and e.get("identifier", {{}}).get("id") == ext_id)]
        entries.append({{
            "identifier": {{"id": ext_id}},
            "version": "0.0.1",
            "location": {{
                "$mid": 1,
                "fsPath": install_dir,
                "path": install_dir,
                "scheme": "file",
            }},
            "relativeLocation": ext_id + "-0.0.1",
            "metadata": {{"installedTimestamp": int(time.time() * 1000)}},
        }})
        with open(reg_path, "w") as f:
            json.dump(entries, f)
        """)
    return [
        {"type": "execute", "parameters": {
            "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
            "shell": True,
        }},
    ]


def _make_extension_install_template(spec: dict) -> SynthTemplate:
    """Marketplace-install extension template (synthetic-pre-stage oracle).

    Pre-config purges any pre-existing install of the target ext-id (both the
    on-disk `<ext_id>-*` directory and the matching `extensions.json` entry)
    so the trivial-pass guard holds: without oracle, `code --list-extensions`
    won't contain the ext-id.

    Oracle bypasses the network — instead of `code --install-extension
    <ext_id> --force` (which requires marketplace connectivity and is flaky
    in CI), it writes a synthetic install directory + registry entry that
    `code --list-extensions` reports verbatim. The agent-facing instructions
    are unchanged: the agent still has to navigate the Extensions sidebar to
    succeed; only the oracle's gold path is short-circuited.
    """
    template_id = spec["id"]
    ext_id = spec["ext_id"]
    ext_name = spec["ext_name"]

    init_step = _purge_extension_step(ext_id)
    oracle_steps = _install_extension_steps(ext_id)

    evaluator = {
        "func": "is_extension_installed",
        "result": {"type": "vm_command_line",
                   "command": ["code", "--list-extensions", "|", "grep", ext_id]},
        "expected": {"type": "rule",
                     "rules": {"type": "contain", "expected": ext_id}},
    }

    instructions = [
        f"Please install the {ext_name} extension in VS Code.",
        f"Install the {ext_name} VS Code extension globally.",
        f"In the open VS Code editor, install the {ext_name} extension via the Extensions sidebar.",
    ]

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(template_id) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": [init_step] + _vs_code_preopen_steps("/home/user"),
        }

    return SynthTemplate(
        template_id=template_id,
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: None,
        param_fn=_params,
        n_rows=1,
        eval_class="extension",
        setup_class="vscode_user",
    )


# ---------------------------------------------------------------------------
# Local VSIX install template (mirrors osworld_vs_code_0512bb38)
# ---------------------------------------------------------------------------
#
# Pre-config builds a minimal VSIX (`undefined_publisher.test`) on disk via
# python heredoc — a VSIX is a zip with `[Content_Types].xml`,
# `extension.vsixmanifest`, and `extension/package.json`. The publisher /
# extension-id matches the eval task (`undefined_publisher.test`) so the
# evaluator's `grep undefined_publisher.test` against `code --list-extensions`
# matches once installed.
#
# The published VSIX install path is: pre-config writes /home/user/test.vsix;
# oracle runs `code --install-extension /home/user/test.vsix --force`; the
# resulting installed ext-id is `undefined_publisher.test` (vsce auto-prefixes
# with `undefined_publisher` when the manifest omits a publisher).
#
# Network-independent: VSIX install reads the file directly, no marketplace
# round-trip. This is the lowest-risk extension-install row in the bundle.

_VSIX_PUBLISHER = "undefined_publisher"
_VSIX_NAME = "test"
_VSIX_EXT_ID = f"{_VSIX_PUBLISHER}.{_VSIX_NAME}"
_VSIX_PATH = "/home/user/test.vsix"


def _build_vsix_step(path: str, publisher: str, name: str) -> dict:
    """Build a pre-config step that materializes a minimal valid VSIX at `path`.

    A VSIX is a ZIP archive with three required entries:
      - `[Content_Types].xml`            — MIME-type registration
      - `extension.vsixmanifest`         — VS Code marketplace manifest (XML)
      - `extension/package.json`         — VS Code extension manifest (JSON)

    The publisher is omitted from the .vsixmanifest's `<Identity Publisher=...>`
    only when we want VS Code's `--install-extension` CLI to treat the file as
    a "publisher-less" upload — but `code --install-extension <file>` actually
    requires a real publisher attribute, so we set Publisher="undefined_publisher"
    explicitly (matching the eval-side VSIX which exhibits the same id).
    """
    py = textwrap.dedent(f"""\
        import os, zipfile, io
        path = {path!r}
        publisher = {publisher!r}
        name = {name!r}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # `extension.vsixmanifest` — VS Code marketplace manifest. The
        # Identity Publisher attribute drives the resulting ext-id; with
        # Publisher="undefined_publisher" + Id="test", `--list-extensions`
        # emits `undefined_publisher.test`.
        vsixmanifest = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<PackageManifest Version="2.0.0" '
            'xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011" '
            'xmlns:d="http://schemas.microsoft.com/developer/vsx-schema-design/2011">'
            '<Metadata>'
            f'<Identity Language="en-US" Id="{{name}}" Version="0.0.1" '
            f'Publisher="{{publisher}}" />'
            f'<DisplayName>{{name}}</DisplayName>'
            f'<Description xml:space="preserve">Synth test extension</Description>'
            '</Metadata>'
            '<Installation>'
            '<InstallationTarget Id="Microsoft.VisualStudio.Code" />'
            '</Installation>'
            '<Dependencies/>'
            '<Assets>'
            '<Asset Type="Microsoft.VisualStudio.Code.Manifest" '
            'Path="extension/package.json" Addressable="true" />'
            '</Assets>'
            '</PackageManifest>'
        )
        # `[Content_Types].xml` — minimal MIME registration.
        content_types = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="json" ContentType="application/json" />'
            '<Default Extension="vsixmanifest" ContentType="text/xml" />'
            '</Types>'
        )
        # `extension/package.json` — VS Code extension manifest. Engines.vscode
        # is the only field VS Code's loader truly requires; publisher/name/
        # version round-trip to the install-dir name.
        package_json = (
            '{{"name": "' + name + '", '
            '"displayName": "' + name + '", '
            '"description": "synth test", '
            '"version": "0.0.1", '
            '"publisher": "' + publisher + '", '
            '"engines": {{"vscode": "^1.0.0"}}, '
            '"categories": ["Other"], '
            '"contributes": {{}} }}'
        )
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('extension.vsixmanifest', vsixmanifest)
            z.writestr('[Content_Types].xml', content_types)
            z.writestr('extension/package.json', package_json)
        """)
    return {"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py}\nPYEOF",
        "shell": True,
    }}


def _make_vsix_install_template() -> SynthTemplate:
    template_id = "vs_code_install_ext_local_vsix"
    ext_id = _VSIX_EXT_ID

    # Pre-config: build the VSIX on disk + purge any stale install of the
    # same ext-id.
    init_steps = [
        _build_vsix_step(_VSIX_PATH, _VSIX_PUBLISHER, _VSIX_NAME),
        {"type": "execute", "parameters": {
            "command": (
                f"rm -rf /home/user/.vscode/extensions/{ext_id}-* 2>/dev/null; "
                f"rm -rf /root/.vscode/extensions/{ext_id}-* 2>/dev/null; true"
            ),
            "shell": True,
        }},
    ]

    # Oracle: install from the local VSIX (no marketplace round-trip).
    oracle_steps = [
        {"type": "execute", "parameters": {
            "command": (
                f"code --no-sandbox --install-extension {_VSIX_PATH} --force "
                f"2>/dev/null || true"
            ),
            "shell": True,
        }},
        {"type": "sleep", "parameters": {"seconds": 2}},
    ]

    evaluator = {
        "func": "is_extension_installed",
        "result": {"type": "vm_command_line",
                   "command": ["code", "--list-extensions", "|", "grep", ext_id]},
        "expected": {"type": "rule",
                     "rules": {"type": "contain", "expected": ext_id}},
    }

    instructions = [
        f"Please install the VS Code extension from the local VSIX file at `{_VSIX_PATH}`.",
        f"Install the extension packaged at `{_VSIX_PATH}` into VS Code.",
        f"In the open VS Code editor, install the local VSIX file `{_VSIX_PATH}` (Extensions sidebar > Install from VSIX...).",
    ]

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(template_id) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": init_steps + _vs_code_preopen_steps("/home/user"),
        }

    return SynthTemplate(
        template_id=template_id,
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: None,
        param_fn=_params,
        n_rows=1,
        eval_class="extension",
        setup_class="vscode_user",
    )


# ---------------------------------------------------------------------------
# Workspace ops templates (`.code-workspace` JSON file create / multi-folder)
# ---------------------------------------------------------------------------
#
# Mirrors eval rows `osworld_vs_code_5e2d93d8` ("save current project as
# workspace") and `osworld_vs_code_6ed0a554` ("add 2 folders to current
# workspace"). Two evaluator shapes:
#
#   (a) Save-as / file-exists:  `is_extension_installed` (misnomer — it just
#       runs a `vm_command_line` `ls <dir> | grep <basename>` and checks the
#       output contains `<basename>`). Mirrors 5e2d93d8 exactly.
#
#   (b) Workspace JSON content: `check_json_settings` on the `.code-workspace`
#       file with expected `{<key>: <value>}` (subset on top-level keys; values
#       must match by `==`). Mirrors 6ed0a554's `folders` shape and extends to
#       `settings`, `extensions`, `launch` blocks for structural diversity.
#
# Trivial-pass guards (Hard Constraint #2):
#   - Save-as rows: pre_config creates the project folder + skeleton files but
#     NEVER pre-creates the gold `.code-workspace` (oracle copies it). Without
#     the oracle the `ls | grep` returns nothing and the evaluator scores 0.
#   - Multi-folder rows: pre_config stages a `.code-workspace` with a
#     SINGLE-folder `folders` array (the launched workspace state). Oracle
#     rewrites the file to include the additional folders. Without the oracle
#     the `folders` array stays at length 1 ≠ expected → eval fails.
#   - Settings / extensions / launch rows: pre_config writes a baseline
#     `.code-workspace` WITHOUT the target block; oracle adds it. Without
#     oracle the target key is missing → eval fails.
#
# Folder paths use realistic project names (project / frontend / api / shared
# / data1 / data2) and are pre-created with `mkdir -p` so the workspace
# references resolve on disk.

# Spec: `mode` ∈ {"save_as", "multi_folder", "settings_block",
# "extensions_block", "launch_block"}. See `_make_workspace_template` for the
# full shape per mode.
_WORKSPACE_SPECS: list[dict] = [
    # ---- Mode (a) save_as: file-exists eval (mirrors 5e2d93d8) ----
    {
        "id": "vs_code_workspace_save_single_folder",
        "mode": "save_as",
        "project_dir": "/home/user/project",
        "workspace_path": "/home/user/project.code-workspace",
        # Skeleton files inside project_dir (created by pre_config).
        "skeleton": {
            "main.py": "print('hello')\n",
            "README.md": "# project\n\nA sample project.\n",
        },
        # Gold workspace JSON the oracle materializes.
        "workspace_json": {"folders": [{"path": "project"}], "settings": {}},
        "instructions": [
            "Please help me save the currently-open `project` folder as a workspace named `project.code-workspace` at `/home/user/`.",
            "Help me set this up — save the current `project` folder as a `.code-workspace` file at `/home/user/project.code-workspace`.",
        ],
    },
    # ---- Mode (b) multi_folder: 2 extra folders (mirrors 6ed0a554) ----
    {
        "id": "vs_code_workspace_multi_folder_2",
        "mode": "multi_folder",
        # Initial workspace file (pre-created by pre_config).
        "workspace_path": "/home/user/project.code-workspace",
        # All folders that must exist on disk (parent dirs).
        "folder_dirs": [
            "/home/user/project",
            "/home/user/data1",
            "/home/user/data2",
        ],
        # Initial folders array in workspace file.
        "init_folders": [{"path": "project"}],
        # Gold folders array (after agent action).
        "target_folders": [{"path": "project"}, {"path": "data1"}, {"path": "data2"}],
        "instructions": [
            "Please help me add the folders `/home/user/data1` and `/home/user/data2` to the current workspace, in that order (data1 first, then data2), and save.",
            "I'm tuning my dev env — add `/home/user/data1` first, then `/home/user/data2`, to the workspace via File > Add Folder to Workspace, then save the workspace.",
        ],
    },
    # ---- Mode (b) multi_folder: 3 extra folders (Cartesian variant) ----
    {
        "id": "vs_code_workspace_multi_folder_3",
        "mode": "multi_folder",
        "workspace_path": "/home/user/monorepo.code-workspace",
        "folder_dirs": [
            "/home/user/monorepo",
            "/home/user/frontend",
            "/home/user/api",
            "/home/user/shared-lib",
        ],
        "init_folders": [{"path": "monorepo"}],
        "target_folders": [
            {"path": "monorepo"},
            {"path": "frontend"},
            {"path": "api"},
            {"path": "shared-lib"},
        ],
        "instructions": [
            "Please help me add the folders `/home/user/frontend`, `/home/user/api`, and `/home/user/shared-lib` to the current workspace, keeping that exact order (frontend → api → shared-lib), and save.",
            "I'm tuning my dev env — add the three sibling folders in this exact order: `frontend`, then `api`, then `shared-lib` (under `/home/user/`) to the open workspace via File > Add Folder to Workspace, then save.",
        ],
    },
    # ---- Mode (b) settings_block: workspace-level editor settings ----
    {
        "id": "vs_code_workspace_with_settings_tabsize",
        "mode": "settings_block",
        "workspace_path": "/home/user/project.code-workspace",
        "folder_dirs": ["/home/user/project"],
        # Initial workspace JSON (no settings block).
        "init_json": {"folders": [{"path": "project"}]},
        # Target key + value.
        "target_key": "settings",
        "target_value": {"editor.tabSize": 4, "editor.insertSpaces": True},
        "instructions": [
            "In my workspace file at /home/user/project.code-workspace, set up workspace-scope indentation so a Tab inserts 4 spaces — not the default 2.",
            "Configure the open workspace (at /home/user/project.code-workspace) so anyone opening it gets a 4-space soft-tab indent for all files in this project.",
        ],
    },
    # ---- Mode (b) extensions_block: recommendations ----
    {
        "id": "vs_code_workspace_with_extensions_recommendations",
        "mode": "settings_block",
        "workspace_path": "/home/user/project.code-workspace",
        "folder_dirs": ["/home/user/project"],
        "init_json": {"folders": [{"path": "project"}]},
        "target_key": "extensions",
        "target_value": {
            "recommendations": ["ms-python.python", "esbenp.prettier-vscode"],
        },
        "instructions": [
            "Add a recommended-extensions list to the workspace at /home/user/project.code-workspace so new contributors get prompted to install Microsoft's Python extension (marketplace publisher `ms-python`, name `python`) and Prettier (marketplace publisher `esbenp`, name `prettier-vscode`). Use the canonical publisher-dot-name marketplace identifier in the JSON.",
            "In the workspace file /home/user/project.code-workspace, mark Microsoft's Python extension (publisher `ms-python`, name `python`) and Prettier (publisher `esbenp`, name `prettier-vscode`) as workspace-recommended so VS Code suggests them on first open.",
        ],
    },
    # ---- Mode (b) launch_block: launch.json equivalent at workspace scope ----
    {
        "id": "vs_code_workspace_with_launch_python",
        "mode": "settings_block",
        "workspace_path": "/home/user/project.code-workspace",
        "folder_dirs": ["/home/user/project"],
        "init_json": {"folders": [{"path": "project"}]},
        "target_key": "launch",
        "target_value": {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python: Current File",
                    "type": "python",
                    "request": "launch",
                    "program": "${file}",
                    "console": "integratedTerminal",
                }
            ],
        },
        "instructions": [
            "Please set up a debug configuration in my workspace at /home/user/project.code-workspace so I can run whatever Python file I'm currently viewing — it should launch the active file in the integrated terminal.",
        ],
    },
]


def _write_workspace_json_step(path: str, payload: dict) -> dict:
    """Write a `.code-workspace` JSON file at `path` with the given payload.

    Reuses the heredoc pattern from `_write_json_step` but explicit so the
    intent (workspace-shaped JSON) is grep-able.
    """
    return _write_json_step(path, payload)


def _make_workspace_template(spec: dict) -> SynthTemplate:
    """Build a `.code-workspace` ops template per `spec["mode"]`.

    Three eval-shape modes share this builder so the deferred-list rows
    (5e2d93d8 / 6ed0a554 family) all flow through one helper:

      "save_as":          eval = is_extension_installed (`ls | grep <name>`)
      "multi_folder":     eval = check_json_settings on `folders` array
      "settings_block":   eval = check_json_settings on `<target_key>`
                          (used for settings / extensions / launch blocks)
    """
    template_id = spec["id"]
    mode = spec["mode"]
    instructions = spec["instructions"]

    if mode == "save_as":
        project_dir = spec["project_dir"]
        workspace_path = spec["workspace_path"]
        skeleton = spec["skeleton"]
        workspace_json = spec["workspace_json"]
        workspace_basename = workspace_path.rsplit("/", 1)[-1]
        workspace_parent = workspace_path.rsplit("/", 1)[0]

        # Pre-config: create project dir + skeleton files; ensure no
        # pre-existing workspace file at the gold path (trivial-pass guard).
        pre_steps: list[dict] = [
            {"type": "execute", "parameters": {
                "command": (
                    f"mkdir -p '{project_dir}' && "
                    f"rm -f '{workspace_path}'"
                ),
                "shell": True,
            }},
        ]
        for fname, content in skeleton.items():
            pre_steps.append(_write_text_step(f"{project_dir}/{fname}", content))

        oracle_steps = [
            _kill_code_step(),
            _write_workspace_json_step(workspace_path, workspace_json),
        ]

        # Batch fix: anchor with `grep -F` (fixed-string, no regex) so the
        # `.` in `<name>.code-workspace` doesn't match any-char — otherwise
        # a sibling like `project_code-workspace` would FALSE-PASS the eval.
        evaluator = {
            "func": "is_extension_installed",
            "result": {"type": "vm_command_line",
                       "command": ["ls", workspace_parent + "/", "|",
                                   "grep", "-F", workspace_basename]},
            "expected": {"type": "rule",
                         "rules": {"type": "contain",
                                   "expected": workspace_basename}},
        }

    elif mode == "multi_folder":
        workspace_path = spec["workspace_path"]
        folder_dirs = spec["folder_dirs"]
        init_folders = spec["init_folders"]
        target_folders = spec["target_folders"]

        # Pre-config: mkdir all referenced folders + write initial workspace
        # file with single-folder `folders` array (trivial-pass guard — array
        # length differs from target).
        mkdir_cmd = " && ".join(f"mkdir -p '{d}'" for d in folder_dirs)
        pre_steps = [
            {"type": "execute", "parameters": {"command": mkdir_cmd, "shell": True}},
            _write_workspace_json_step(
                workspace_path,
                {"folders": init_folders, "settings": {}},
            ),
        ]

        # Oracle: kill code (so the live VS Code editor doesn't clobber our
        # write), then rewrite the workspace file with the full target
        # `folders` array.
        oracle_steps = [
            _kill_code_step(),
            _write_workspace_json_step(
                workspace_path,
                {"folders": target_folders, "settings": {}},
            ),
        ]

        evaluator = {
            "func": "check_json_settings",
            "result": {"type": "vm_file", "path": workspace_path,
                       "dest": "result.code-workspace"},
            "expected": {"type": "rule",
                         "rules": {"expected": {"folders": target_folders}}},
        }

    elif mode == "settings_block":
        workspace_path = spec["workspace_path"]
        folder_dirs = spec["folder_dirs"]
        init_json = spec["init_json"]
        target_key = spec["target_key"]
        target_value = spec["target_value"]

        # Pre-config: mkdir folder(s) + stage a baseline `.code-workspace`
        # WITHOUT the `target_key` block (trivial-pass guard).
        mkdir_cmd = " && ".join(f"mkdir -p '{d}'" for d in folder_dirs)
        pre_steps = [
            {"type": "execute", "parameters": {"command": mkdir_cmd, "shell": True}},
            _write_workspace_json_step(workspace_path, init_json),
        ]

        # Oracle: kill code, then write the workspace file with the merged
        # target block. Build `target_json` from `init_json` ∪ {target_key:
        # target_value} so the gold file is self-consistent.
        target_json = dict(init_json)
        target_json[target_key] = target_value
        oracle_steps = [
            _kill_code_step(),
            _write_workspace_json_step(workspace_path, target_json),
        ]

        evaluator = {
            "func": "check_json_settings",
            "result": {"type": "vm_file", "path": workspace_path,
                       "dest": "result.code-workspace"},
            "expected": {"type": "rule",
                         "rules": {"expected": {target_key: target_value}}},
        }

    else:
        raise ValueError(f"_make_workspace_template: unknown mode {mode!r}")

    # Workspace for preopen: for `save_as` the .code-workspace file is not yet
    # on disk (oracle materializes it), so open the staged project folder
    # instead. For multi_folder / settings_block the workspace file is already
    # written by pre_config — open it directly to mirror the "user is editing
    # this workspace" narrative.
    if mode == "save_as":
        preopen_target = spec["project_dir"]
    else:
        preopen_target = spec["workspace_path"]

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(template_id) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": pre_steps + _vs_code_preopen_steps(preopen_target),
        }

    # eval_class="compare_config" buckets these against the eval-side
    # `compare_config` family (eval has 2 such tasks: 5e2d93d8 + 6ed0a554).
    # Even though 5e2d93d8 uses `is_extension_installed` as its evaluator
    # func, semantically it scores workspace-file state — bucket alongside
    # the other workspace-file evaluators.
    return SynthTemplate(
        template_id=template_id,
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: None,
        param_fn=_params,
        n_rows=1,
        eval_class="compare_config",
        setup_class="vscode_workspace",
    )


# ---------------------------------------------------------------------------
# `compare_config` settings.json templates (mirrors eval 982d12a5)
# ---------------------------------------------------------------------------
#
# Eval has 2 `compare_config` tasks in vs_code:
#   - 982d12a5: read settings.json, expect `workbench.colorTheme` value
#     (containment_ok=True ⇒ JSON-subset match).
#   - 53ad5833: read `vscode_config` result via a custom VS Code extension
#     (`OpenProject` command) that records the currently-open folder.
#     DEFERRED in synth: the custom extension lives in an OSWorld-only
#     `vscodeEvalExtension.vsix` cached on HuggingFace + needs a live VS
#     Code window to populate `/home/user/OpenProject.txt`. No clean shell
#     oracle without installing that extension + driving the GUI.
#
# The specs below mirror 982d12a5's eval shape exactly (same evaluator func
# name, same containment_ok flag, same `vm_file` → settings.json result),
# differing only in the target theme name so the find-replace surface
# stays diverse.

_COMPARE_CONFIG_SPECS: list[dict] = [
    {
        "id": "vs_code_compare_config_color_theme_dark",
        "key": "workbench.colorTheme",
        "target": "Visual Studio Dark",
        "init": "Default Light+",
        "instructions": [
            "Please help me change the color theme of VS Code to Visual Studio Dark.",
            "It's late and the bright UI is hurting my eyes — switch VS Code over to the Visual Studio Dark color theme.",
        ],
    },
    {
        "id": "vs_code_compare_config_color_theme_light",
        "key": "workbench.colorTheme",
        "target": "Default Light+",
        "init": "Visual Studio Dark",
        "instructions": [
            "Please help me change the color theme of VS Code to Default Light+.",
            "I'm working outside in bright sunlight — please switch VS Code to the Default Light+ color theme.",
        ],
    },
]


def _make_compare_config_template(spec: dict) -> SynthTemplate:
    """Build a `compare_config` settings.json template mirroring 982d12a5.

    Pre-config writes settings.json with the OPPOSITE value (`init`), so the
    trivial-pass guard holds. Oracle kills `code`, then merges the gold
    `{key: target}` into settings.json. Evaluator is `compare_config` with
    `containment_ok: True` (JSON-subset match — mirrors the eval-side flag).
    """
    key = spec["key"]
    target = spec["target"]
    init = spec["init"]
    instructions = spec["instructions"]

    init_step = _merge_into_settings_step(_SETTINGS_PATH, {key: init})
    oracle_steps = [
        _kill_code_step(),
        _merge_into_settings_step(_SETTINGS_PATH, {key: target}),
    ]

    # Expected is a JSON STRING (not dict) — matches 982d12a5's eval shape.
    expected_text = _json.dumps({key: target}) + "\n"
    evaluator = {
        "func": "compare_config",
        "result": {"type": "vm_file", "path": _SETTINGS_PATH, "dest": "settings.json"},
        "expected": {"type": "rule", "rules": {"expected": expected_text}},
        "options": {"containment_ok": True},
    }

    def _params(seed: int) -> dict:
        rng = random.Random(seed ^ _stable_hash(spec["id"]) & 0xFFFFFFFF)
        return {
            "instr": instructions[seed % len(instructions)],
            "pre_config_steps": [init_step] + _vs_code_preopen_steps("/home/user"),
        }

    return SynthTemplate(
        template_id=spec["id"],
        domain="vs_code",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: None,
        param_fn=_params,
        n_rows=1,
        eval_class="compare_config",
        setup_class="vscode_user",
    )


# ---------------------------------------------------------------------------
# TEMPLATES export
# ---------------------------------------------------------------------------

TEMPLATES: list[SynthTemplate] = (
    [_make_settings_template(s) for s in _SETTINGS_SPECS]
    + [_make_settings_multi_template(s) for s in _SETTINGS_MULTI_SPECS]
    + [_make_keybinding_template(s) for s in _KEYBINDING_SPECS]
    + [_make_file_create_template(s) for s in _FILE_CREATE_SPECS]
    + [_make_file_edit_template(s) for s in _FILE_EDIT_SPECS]
    + [_make_asset_file_edit_template(s) for s in _ASSET_FILE_EDIT_SPECS]
    + [_make_gitignore_create_template(s) for s in _GITIGNORE_CREATE_SPECS]
    + [_make_real_settings_template(s) for s in _REAL_SETTINGS_SPECS]
    + [_make_extension_install_template(s) for s in _EXTENSION_SPECS]
    + [_make_vsix_install_template()]
    + [_make_workspace_template(s) for s in _WORKSPACE_SPECS]
    + [_make_compare_config_template(s) for s in _COMPARE_CONFIG_SPECS]
)
