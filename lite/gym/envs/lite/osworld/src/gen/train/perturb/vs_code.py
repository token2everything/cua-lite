"""VS Code domain perturbation functions (Track B).

Structural functions: (eval_row, rng) -> list[dict]

Usage:
    from lite.gym.envs.lite.osworld.src.gen.train.perturb.vs_code import VSCODE_PERTURB_FNS
    rows = VSCODE_PERTURB_FNS["json_setting"](eval_row, rng)
"""

from __future__ import annotations

import copy
import json
import logging
import random

logger = logging.getLogger(__name__)
import re

from lite.gym.envs.lite.osworld.src.gen.train.perturb._utils import make_perturb_row, oracle_actions_of


# ---------------------------------------------------------------------------
# Structural perturbation functions (Track B)
# ---------------------------------------------------------------------------

_SETTINGS_PATH = "/home/user/.config/Code/User/settings.json"
_KEYBINDINGS_PATH = "/home/user/.config/Code/User/keybindings.json"

# Setting specs: maps a settings.json key to perturbation options
_JSON_SETTING_SPECS: dict[str, dict] = {
    "files.autoSaveDelay": {
        "values": [200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000],
        # NOTE: User-scope settings.json — drop workspace
        # wording ("for this repo" / "new project") that misleads agent
        # to look for a workspace folder.
        # #155 #10: the evaluator requires BOTH files.autoSave="afterDelay" AND
        # files.autoSaveDelay=N (a delay is inert while autoSave is off), and setup
        # does not seed files.autoSave. The old templates named only the delay -> a
        # literal agent left autoSave off -> FN. State the coupling explicitly (both
        # are real Settings-UI controls) and keep the honest two-key evaluator.
        "instruction_templates": [
            "Could you enable VS Code's Auto Save in after-delay mode with a {value} millisecond delay in my user settings?",
            "Please turn on Auto Save (after-delay) in VS Code's user settings and set the delay to {value} milliseconds.",
            "I'd like VS Code to auto-save files after a {value} millisecond delay — please set Auto Save to afterDelay with that delay in user settings.",
            "Can you configure VS Code to automatically save after {value}ms of inactivity (Auto Save = after delay) in the global user settings?",
            "In VS Code's user settings, enable Auto Save in 'afterDelay' mode and set the auto-save delay to {value} milliseconds.",
            "Please set VS Code to save files automatically {value} milliseconds after I stop typing (Auto Save after-delay) in user settings.",
        ],
    },
    "editor.wordWrapColumn": {
        "values": [40, 50, 60, 72, 80, 100, 120, 140],
        # NOTE: same — User-scope, no workspace wording.
        "instruction_templates": [
            "Could you help me set VS Code's word wrap column to {value} characters in my user settings?",
            "Please change the line wrapping column to {value} characters in VS Code's user settings.",
            "I'd like VS Code to wrap lines at {value} characters globally — it helps readability everywhere.",
            "Can you help me configure VS Code so code wraps at column {value} in my user settings?",
            "I want the VS Code editor to wrap lines at {value} characters in my user settings.",
            "In VS Code's user settings, set the editor word wrap column to {value} characters.",
        ],
    },
    "workbench.colorTheme": {
        "values": [
            "Visual Studio Dark", "Visual Studio Light", "Monokai",
            "Solarized Dark", "Solarized Light", "Default Dark+",
            "Quiet Light", "Kimbie Dark",
        ],
        "instruction_templates": [
            'Could you help me switch VS Code\'s color theme to "{value}" — my eyes are tired today.',
            'Please change the VS Code color theme to "{value}" for my dev workflow.',
            'I\'d like to use the "{value}" theme in VS Code\'s user settings.',
            'Can you help me set the VS Code editor theme to "{value}" in user settings?',
            'I want to change my VS Code color theme to "{value}" globally in user settings.',
            'Switch the VS Code color theme to "{value}" so the editor is easier on my eyes.',
        ],
    },
}

# Extension specs
_EXTENSIONS = [
    ("ms-python.python", "Python"),
    ("njpwerner.autodocstring", "autoDocstring"),
    ("esbenp.prettier-vscode", "Prettier"),
    ("dbaeumer.vscode-eslint", "ESLint"),
    ("ms-vscode.cpptools", "C/C++"),
    ("golang.go", "Go"),
    ("rust-lang.rust-analyzer", "rust-analyzer"),
]

# create_file pools: filenames + target dirs for "create new file via VS Code".
# Used by perturb_vscode_create_file. Eval target file (e.g. "test.py" at
# "/home/user/Desktop") is excluded from candidate sampling.
_NEW_FILE_NAMES_PY = [
    "hello.py", "main.py", "utils.py", "app.py",
    "script.py", "demo.py", "lib.py", "module.py",
]
_NEW_FILE_NAMES_GENERIC = [
    ("notes.md", "markdown"), ("readme.md", "markdown"),
    ("todo.txt", "text"), ("data.json", "JSON"),
]
_NEW_FILE_PATHS = [
    "/home/user/Desktop", "/home/user/Documents", "/home/user/Downloads",
]
_CREATE_PY_TEMPLATES = [
    'Please help me create a new python file named "{filename}" via VS Code and save it at "{path}".',
    'Could you help me create a python file "{filename}" in VS Code, saved to {path} for this project?',
    'Please i\'d like to make a new Python file "{filename}" using VS Code, saved at {path} for my repo.',
    'Can you help me create a new python file called "{filename}" in VS Code under {path}?',
    'Please create a new python file named "{filename}" via VS Code at {path} for my dev workflow.',
    'Please open VS Code and create a new python file named "{filename}", then save it in {path}.',
]
_CREATE_GENERIC_TEMPLATES = [
    'Please help me create a new {kind} file named "{filename}" via VS Code and save it at "{path}".',
    'Could you help me create a {kind} file "{filename}" in VS Code, saved to {path} for this repo?',
    'Please i\'d like to make a new {kind} file "{filename}" using VS Code, saved at {path} for my project.',
    'Can you help me create a new {kind} file called "{filename}" in VS Code under {path}?',
    'Please create a new {kind} file named "{filename}" via VS Code at {path} for my workflow.',
    'Please open VS Code and create a new {kind} file named "{filename}", then save it in {path}.',
]

_EXTENSION_TEMPLATES = [
    "Please help me install the {name} extension in VS Code so I can use it in this project.",
    "Could you help me install the {name} extension in VS Code for my dev workflow?",
    "Please install the {name} extension in VS Code — I'm setting up a new repo.",
    "Can you help me set up VS Code with the {name} extension for debugging this codebase?",
    "Please install the {name} extension in VS Code before starting work on my project.",
    "Please install the {name} extension in VS Code so it's available for this repo.",
]


def _build_settings_oracle(config_path: str, settings: dict) -> list[dict]:
    """Build oracle that writes settings to a VS Code JSON config file."""
    settings_repr = repr(settings)
    return [
        {"type": "execute", "parameters": {
            "command": f"mkdir -p $(dirname {config_path})", "shell": True,
        }},
        {"type": "execute", "parameters": {
            "command": (
                f"python3 << 'PYEOF'\n"
                f"import json, os\n"
                f"path = '{config_path}'\n"
                f"existing = {{}}\n"
                f"if os.path.exists(path):\n"
                f"    with open(path) as f:\n"
                f"        existing = json.load(f)\n"
                f"existing.update({settings_repr})\n"
                f"with open(path, 'w') as f:\n"
                f"    json.dump(existing, f, indent=2)\n"
                f"PYEOF"
            ),
            "shell": True,
        }},
    ]


def _build_keybinding_oracle(config_path: str, binding: dict) -> list[dict]:
    """Build oracle that appends a keybinding to VS Code keybindings.json."""
    binding_repr = repr(binding)
    return [
        {"type": "execute", "parameters": {
            "command": f"mkdir -p $(dirname {config_path})", "shell": True,
        }},
        {"type": "execute", "parameters": {
            "command": (
                f"python3 << 'PYEOF'\n"
                f"import json, os\n"
                f"path = '{config_path}'\n"
                f"existing = []\n"
                f"if os.path.exists(path):\n"
                f"    with open(path) as f:\n"
                f"        existing = json.load(f)\n"
                f"existing.append({binding_repr})\n"
                f"with open(path, 'w') as f:\n"
                f"    json.dump(existing, f, indent=2)\n"
                f"PYEOF"
            ),
            "shell": True,
        }},
    ]


def perturb_vscode_setting(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate structural perturbations for VS Code settings.json tasks.

    Handles check_json_settings and compare_config evaluators by identifying
    the setting key, then generating variants with different values.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    result = evaluator.get("result", {})
    if not isinstance(result, dict):
        return []
    result_path = result.get("path", "")

    # Only handle settings.json tasks
    if func not in ("check_json_settings", "compare_config"):
        return []
    if "settings.json" not in result_path:
        return []

    expected = evaluator.get("expected", {}).get("rules", {}).get("expected", {})

    # For compare_config, expected is a JSON string
    if isinstance(expected, str):
        try:
            expected = json.loads(expected.strip())
        except (json.JSONDecodeError, ValueError):
            return []

    if not isinstance(expected, dict):
        return []

    # Find a perturbable key
    rows = []
    for setting_key, spec in _JSON_SETTING_SPECS.items():
        if setting_key not in expected:
            continue

        orig_val = expected[setting_key]
        candidates = [v for v in spec["values"] if v != orig_val]
        if not candidates:
            continue

        for new_val in rng.sample(candidates, min(4, len(candidates))):
            instruction = rng.choice(spec["instruction_templates"]).format(value=new_val)

            new_expected = copy.deepcopy(expected)
            new_expected[setting_key] = new_val

            new_evaluator = copy.deepcopy(evaluator)
            if func == "compare_config":
                new_evaluator["expected"]["rules"]["expected"] = json.dumps(
                    {setting_key: new_val},
                ) + "\n"
            else:
                new_evaluator["expected"]["rules"]["expected"] = new_expected

            oracle = _build_settings_oracle(result_path, new_expected)

            rows.append(make_perturb_row(
                eval_row=eval_row,
                knob_assignment={setting_key: new_val},
                new_instruction=instruction,
                new_oracle=oracle,
                new_evaluator=new_evaluator,
            ))

        # Only perturb the first matching key
        break

    return rows


def perturb_vscode_extension(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate structural perturbations for VS Code extension install tasks.

    Handles is_extension_installed evaluator by swapping the target extension.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "is_extension_installed":
        return []

    expected = evaluator.get("expected", {}).get("rules", {})
    orig_ext = expected.get("expected", "")
    if not orig_ext or not isinstance(orig_ext, str):
        return []

    # Only perturb tasks that check extensions via `code --list-extensions`.
    # Tasks that use `ls` / `grep` to check files (e.g. .code-workspace) are NOT
    # extension-install tasks and must not be treated as such.
    result_cmd = evaluator.get("result", {}).get("command", [])
    if not any("list-extensions" in str(c) for c in result_cmd):
        return []

    # Skip non-marketplace extensions (local VSIX installs, file checks)
    # Extension IDs look like "ms-python.python", not "test.py"
    if "." not in orig_ext or orig_ext.endswith((".py", ".js", ".ts", ".txt", ".json", ".md")):
        return []

    candidates = [(eid, name) for eid, name in _EXTENSIONS if eid != orig_ext]
    if not candidates:
        return []

    rows = []
    for ext_id, ext_name in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_EXTENSION_TEMPLATES).format(name=ext_name)

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["expected"]["rules"]["expected"] = ext_id
        # Also update result.command to grep for the correct extension ID
        result_cmd = new_evaluator.get("result", {}).get("command", [])
        new_evaluator["result"]["command"] = [
            c.replace(orig_ext, ext_id) if isinstance(c, str) else c
            for c in result_cmd
        ]

        oracle = [{"type": "execute", "parameters": {
            "command": f"code --install-extension {ext_id} --force",
            "shell": True,
        }}]

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"extension": ext_id},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# keybinding: perturb VS Code keybinding tasks
# ---------------------------------------------------------------------------

_KB_COMMANDS = [
    ("workbench.action.focusActiveEditorGroup", "move cursor focus from terminal to editor"),
    ("workbench.action.terminal.focus", "move cursor focus to terminal"),
    ("workbench.action.toggleSidebarVisibility", "toggle the sidebar"),
    ("workbench.action.togglePanel", "toggle the bottom panel"),
    ("editor.action.formatDocument", "format the document"),
    ("workbench.action.quickOpen", "open Quick Open"),
    ("editor.action.commentLine", "toggle line comment"),
]

_KB_KEYS = ["ctrl+j", "ctrl+k", "ctrl+m", "ctrl+shift+j", "ctrl+shift+e", "ctrl+shift+k", "ctrl+shift+m", "ctrl+alt+n"]

_KB_CREATE_TEMPLATES = [
    # Validation fix: embed `"when": "{when}"` literally because eval rejects
    # entries without the matching context clause.
    'Please help me create a shortcut "{key}" to {action} in VS Code for my dev workflow. Include `"when": "{when}"` so it only fires in the right context.',
    'Could you help me set up the keyboard shortcut "{key}" for {action_short} in VS Code? Bind with `"when": "{when}"`.',
    'Please i\'d like to bind "{key}" to {action_short} in VS Code\'s user keybindings. Use `"when": "{when}"`.',
    'Can you help me add a keybinding "{key}" for {action_short} in VS Code\'s user keybindings.json? It needs `"when": "{when}"`.',
    'Please create a shortcut "{key}" for {action_short} in VS Code with `"when": "{when}"` before I start coding.',
    'Please set "{key}" as the shortcut to {action} in VS Code for my workflow — must include `"when": "{when}"`.',
]

# NOTE: the eval `check_json_keybindings`
# requires an explicit NEGATION entry — `{"key": "...", "command":
# "-list.find", "when": "..."}` — to be present in keybindings.json.
# Just deleting the line doesn't satisfy it. Templates now spell out
# the leading-minus negation explicitly so the agent adds the right
# entry rather than removing the original line.
_KB_REMOVE_TEMPLATES = [
    # Validation fix: the `when` clause matters because eval expects the same
    # context as the original binding being negated. Embed it literally.
    'Please help me disable "{key}" for {action_short} in VS Code by adding a keybindings.json entry `{{"key": "{key}", "command": "-{cmd_bare}", "when": "{when}"}}` (leading minus negates the default).',
    'Could you help me override "{key}" in VS Code so it no longer triggers {action_short}? Add `{{"key": "{key}", "command": "-{cmd_bare}", "when": "{when}"}}` to keybindings.json.',
    'Please i\'d like to suppress the "{key}" shortcut for {action_short} in VS Code — please add a keybindings.json entry `{{"key": "{key}", "command": "-{cmd_bare}", "when": "{when}"}}` (negation dash + matching context).',
    'Can you help me add a keybindings.json entry that disables "{key}" for {action_short} in VS Code: `{{"key": "{key}", "command": "-{cmd_bare}", "when": "{when}"}}`.',
    'Please remove the default "{key}" binding for {action_short} in VS Code — append `{{"key": "{key}", "command": "-{cmd_bare}", "when": "{when}"}}` to keybindings.json for my workflow.',
    'Please in VS Code\'s keybindings.json, add `{{"key": "{key}", "command": "-{cmd_bare}", "when": "{when}"}}` — the leading dash disables the {action_short} shortcut and the `when` matches the default context.',
]


def perturb_vscode_keybinding(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for VS Code keybinding tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "check_json_keybindings":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    orig_binding = rules.get("expected", {})
    if not isinstance(orig_binding, dict):
        return []

    orig_key = orig_binding.get("key", "")
    orig_command = orig_binding.get("command", "")
    is_remove = orig_command.startswith("-")

    key_candidates = [k for k in _KB_KEYS if k != orig_key]
    rows = []
    for new_key in rng.sample(key_candidates, min(4, len(key_candidates))):
        # Find action description
        action_short = "perform action"
        for cmd, desc in _KB_COMMANDS:
            bare_cmd = orig_command.lstrip("-")
            if cmd == bare_cmd:
                action_short = desc
                break

        # Validation fix: eval requires the `when` clause to match exactly;
        # extract it from the original binding so the instruction can embed it.
        orig_when = orig_binding.get("when", "")
        if is_remove:
            instruction = rng.choice(_KB_REMOVE_TEMPLATES).format(
                key=new_key,
                action_short=action_short,
                cmd_bare=orig_command.lstrip("-"),
                when=orig_when,
            )
        else:
            instruction = rng.choice(_KB_CREATE_TEMPLATES).format(
                key=new_key, action=action_short, action_short=action_short,
                when=orig_when,
            )

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["expected"]["key"] = new_key

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action_item in oracle:
            cmd = action_item.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_key in cmd:
                action_item["parameters"]["command"] = cmd.replace(orig_key, new_key)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"key": new_key},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# boolean_setting: perturb VS Code boolean settings
# ---------------------------------------------------------------------------

_BOOLEAN_SETTINGS: dict[str, dict] = {
    "debug.focusEditorOnBreak": {
        "instruction_true": "Keep the cursor focused on the editor when a breakpoint is hit during debugging.",
        "instruction_false": "Keep the cursor focused on the debug console when a breakpoint is hit.",
    },
    "workbench.editor.wrapTabs": {
        "instruction_true": "Enable tab wrapping across multiple lines when there are too many tabs.",
        "instruction_false": "Disable tab wrapping - show tabs in a single scrollable row.",
    },
}

# NOTE: the previous
# templates used "for my project" / "for this project" / "for this repo",
# which made the agent infer workspace-scope settings and refuse at turn_00
# because no folder was open. Eval reads User-scope `~/.config/Code/User/
# settings.json`, so workspace-scope wording contradicts the contract.
# Globalised to user-settings phrasing; kept narrative diversity (workflow
# / dev / coding) without scope ambiguity.
_BOOL_SETTING_TEMPLATES = [
    "Please help me {action_lower} in VS Code's user settings.",
    "Could you help me {action_lower} in VS Code globally?",
    "Please {action_lower} in VS Code's settings — I want this on for all sessions.",
    "Can you help me {action_lower} in VS Code's user settings for my dev workflow?",
    "Please {action_lower} in VS Code before I continue coding.",
    "Please {action} in VS Code's user settings.",
]


def perturb_vscode_bool_setting(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for VS Code boolean setting tasks.

    Boolean keys only have one alternative value (true ↔ false), so each
    base would otherwise emit a single variant. To upgrade single-variant
    bases into multi-variant ones (R1 audit), we emit several paraphrase
    variants per base — same target value, different instruction phrasing.
    Variants differ on a `paraphrase_idx` knob so each gets a unique
    task_id hash.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "check_json_settings":
        return []

    result = evaluator.get("result", {})
    if not isinstance(result, dict):
        return []
    result_path = result.get("path", "")
    if "settings.json" not in result_path:
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    exp_settings = rules.get("expected", {})
    if not isinstance(exp_settings, dict):
        return []

    rows = []
    for setting_key, spec in _BOOLEAN_SETTINGS.items():
        if setting_key not in exp_settings:
            continue
        orig_val = exp_settings[setting_key]
        if not isinstance(orig_val, bool):
            continue

        new_val = not orig_val
        action_desc = spec[f"instruction_{str(new_val).lower()}"]
        # Strip trailing period so the fragment can be embedded mid-sentence
        # (e.g. "Please help me {action_lower} in VS Code settings...").
        action_frag = action_desc.rstrip(".")

        # Sample 4 distinct paraphrase templates → 4 variants per base.
        # paraphrase_idx ensures unique knob hash so task_ids don't collide.
        n_variants = min(4, len(_BOOL_SETTING_TEMPLATES))
        templates = rng.sample(_BOOL_SETTING_TEMPLATES, n_variants)
        for idx, template in enumerate(templates):
            instruction = template.format(
                action=action_frag,
                action_lower=action_frag[0].lower() + action_frag[1:],
            )

            new_evaluator = copy.deepcopy(evaluator)
            new_exp = new_evaluator.get("expected", {})
            if isinstance(new_exp, list):
                new_exp = new_exp[0]
            new_exp["rules"]["expected"][setting_key] = new_val

            oracle = _build_settings_oracle(
                result_path, {setting_key: new_val},
            )
            # Seed the NON-target (starting) value at setup so a no-op agent
            # begins off-target. Without this, the default-aware
            # check_json_settings override (#155 §C #9) trivially passes a no-op
            # whenever new_val == the VS Code shipped default (absent key ==
            # pass): the base image ships settings.json WITHOUT these keys and
            # setup does not write them. Seeding the opposite forces
            # present != target -> 0 for a no-op, while a real UI-set-to-default
            # (key removed) still recovers via the override.
            seed_steps = _build_settings_oracle(
                result_path, {setting_key: not new_val},
            )

            rows.append(make_perturb_row(
                eval_row=eval_row,
                knob_assignment={setting_key: new_val, "paraphrase_idx": idx},
                new_instruction=instruction,
                new_oracle=oracle,
                new_evaluator=new_evaluator,
                pre_config_steps=seed_steps,
            ))

    return rows


# ---------------------------------------------------------------------------
# workspace_folder: perturb VS Code workspace/project folder tasks
# ---------------------------------------------------------------------------

_FOLDER_NAMES = ["data1", "data2", "src1", "src2", "logs1", "logs2", "files1", "files2", "tests1", "docs1"]

# NOTE: the eval expects ALL folders in the
# .code-workspace file to match exactly (e.g. ["project", "src2",
# "data2"]). Single-folder "Add /home/user/X" instructions led the
# agent to either:
#   (1) only add X and leave the original X' untouched (eval expects
#       X but file still has X' → mismatch), or
#   (2) type a NONEXISTENT folder name verbatim into the dialog
#       (VS Code accepts any string) and pass strict-eq.
# Templates now list the FULL desired folder set explicitly so the
# agent knows it must remove the obsolete folder and add the new one.
_WORKSPACE_TEMPLATES = [
    'Please help me update the VS Code workspace so its folder list contains EXACTLY these folders (all under /home/user): {full_list}. Remove any folders not in this list.',
    'Could you help me edit the VS Code workspace file so the "folders" array is precisely: {full_list_paths}? Each must reference an existing /home/user/ subdirectory.',
    'Please i\'d like the VS Code workspace folder list to be only: {full_list}. Add any missing ones (creating /home/user/{folder} if needed) and remove anything else.',
    'Can you help me configure the workspace so it opens exactly these folders: {full_list}? Replace the obsolete folder with /home/user/{folder} if needed.',
    'Please ensure the VS Code workspace file\'s "folders" array equals {full_list_paths} (and ensure /home/user/{folder} exists on disk).',
    'Please make the VS Code workspace file\'s "folders" array equal to {full_list_paths} for my project (and ensure /home/user/{folder} exists on disk).',
]


def perturb_vscode_workspace(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for VS Code workspace folder tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "check_json_settings":
        return []

    result = evaluator.get("result", {})
    if not isinstance(result, dict):
        return []
    result_path = result.get("path", "")
    if "settings.json" in result_path:
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    exp_settings = rules.get("expected", {})
    if not isinstance(exp_settings, dict):
        return []

    folders = exp_settings.get("folders", [])
    if not isinstance(folders, list) or len(folders) < 2:
        return []

    # Find perturbable folder paths
    orig_paths = [f.get("path", "") for f in folders if isinstance(f, dict)]
    perturbable = [p for p in orig_paths if p in _FOLDER_NAMES]
    if not perturbable:
        return []

    orig_folder = perturbable[0]
    candidates = [f for f in _FOLDER_NAMES if f not in orig_paths]

    rows = []
    for new_folder in rng.sample(candidates, min(4, len(candidates))):
        # Build the FULL final folder list so the instruction mentions
        # every entry the eval will check.
        final_paths = [
            new_folder if p == orig_folder else p for p in orig_paths
        ]
        full_list = ", ".join(f'"{p}"' for p in final_paths)
        full_list_paths = "[" + ", ".join(
            f'{{"path": "{p}"}}' for p in final_paths
        ) + "]"
        instruction = rng.choice(_WORKSPACE_TEMPLATES).format(
            folder=new_folder,
            full_list=full_list,
            full_list_paths=full_list_paths,
        )

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_folders = []
        for f in new_exp["rules"]["expected"]["folders"]:
            if isinstance(f, dict) and f.get("path") == orig_folder:
                new_folders.append({"path": new_folder})
            else:
                new_folders.append(f)
        new_exp["rules"]["expected"]["folders"] = new_folders

        # Pre-create the /home/user/<new_folder> directory + an obvious
        # file inside, so VS Code's Add Folder dialog can resolve it.
        # Without this, agent typing "data1" into the dialog adds an
        # invalid path that clutters the workspace and may be rejected.
        pre_config_steps = [{
            "type": "execute",
            "parameters": {
                "command": (
                    f"mkdir -p /home/user/{new_folder} && "
                    f"echo 'placeholder' > /home/user/{new_folder}/README.txt"
                ),
                "shell": True,
            },
        }]

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action_item in oracle:
            cmd = action_item.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_folder in cmd:
                action_item["parameters"]["command"] = cmd.replace(orig_folder, new_folder)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"workspace_folder": new_folder},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            pre_config_steps=pre_config_steps,
        ))

    return rows


# ---------------------------------------------------------------------------
# vscode_project_folder: perturb VS Code open project tasks (compare_config)
# ---------------------------------------------------------------------------

_PROJECT_NAMES = ["project", "workspace", "mycode", "devdir", "source", "codebase", "sandbox", "prototype"]

_PROJECT_TEMPLATES = [
    'Please help me open the "{name}" folder in VS Code so I can start working on it.',
    'Could you help me open VS Code with the "{name}" directory for my dev workflow?',
    'Please i\'d like to open the "{name}" project in VS Code — I\'m setting up a new repo.',
    'Can you help me launch VS Code with the "{name}" folder for this project?',
    'Please ensure VS Code opened with the "{name}" folder so I can begin coding.',
    'Please launch VS Code and open the "{name}" folder for my dev workflow.',
]


def perturb_vscode_project_folder(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for VS Code open project tasks (compare_config)."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "compare_config":
        return []

    result = evaluator.get("result", {})
    if not isinstance(result, dict):
        return []
    result_type = result.get("type", "")
    if result_type != "vscode_config":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    orig_name = str(rules.get("expected", "")).strip()
    if not orig_name:
        return []

    candidates = [n for n in _PROJECT_NAMES if n != orig_name]
    rows = []
    for new_name in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_PROJECT_TEMPLATES).format(name=new_name)

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["expected"] = new_name

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action_item in oracle:
            params = action_item.get("parameters", {})
            cmd = params.get("command", "")
            # Handle string commands
            if isinstance(cmd, str) and orig_name in cmd:
                params["command"] = cmd.replace(orig_name, new_name)
            # Handle list commands (e.g. ["code", "--no-sandbox", "/home/user/project"])
            elif isinstance(cmd, list):
                params["command"] = [
                    c.replace(orig_name, new_name) if isinstance(c, str) and orig_name in c else c
                    for c in cmd
                ]
            # Bump the post-launch settle sleeps. The base eval task uses 5s+5s
            # which is enough when /home/user/project is already on disk and
            # VS Code's folder cache is warm; for the perturb variants the
            # folder is freshly cp -r'd in oracle step 1, so VS Code's first
            # open of the new dir + extension activation can take >10s in the
            # 4GB / 1 CPU container, intermittently leaving Ctrl+Shift+P in
            # postconfig dispatched before the OpenProject command is wired
            # up. Doubling each sleep gives extension activation time to
            # complete before postconfig runs.
            if action_item.get("type") == "sleep":
                secs = params.get("seconds", 0)
                if isinstance(secs, (int, float)) and 0 < secs <= 10:
                    params["seconds"] = secs * 2

        # Prepend an action to copy the original project dir to the new name
        # so that the project files exist under the new folder.
        cp_step = {
            "type": "execute",
            "parameters": {
                "command": (
                    f"if [ -d /home/user/{orig_name} ] && [ ! -d /home/user/{new_name} ]; then "
                    f"cp -r /home/user/{orig_name} /home/user/{new_name}; fi"
                ),
                "shell": True,
            },
        }
        oracle.insert(0, cp_step)
        # Re-activate the new VS Code window before postconfig fires its own
        # activate_window. With multiple `code` launches racing during
        # validate.py's pkill cycle it's possible for the wmctrl-based
        # postconfig to focus a stale window; doing it explicitly inside the
        # oracle (after the final settle sleep) makes focus deterministic.
        oracle.append({
            "type": "activate_window",
            "parameters": {"window_name": "Visual Studio Code"},
        })

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"project_name": new_name},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            # Also create the renamed folder at setup-time, otherwise the agent
            # opens VS Code into an empty/non-existent dir before the oracle
            # would have run. (validation pattern vs_code.perturb_53ad5833_named_folder_not_created)
            perturb_config_step=cp_step,
        ))

    return rows


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Text-edit perturbations (compare_text_file evaluator)
# ---------------------------------------------------------------------------
# Both follow the libreoffice_impress pattern: synthesize source + gold at
# perturb time. perturb_config_step writes (a) a synthetic source file
# overwriting eval's downloaded source so the agent sees a controlled value
# and (b) the corresponding gold file at expected_path. Oracle then `cp`s
# gold→source after validate.py kills VS Code (so VS Code's in-memory copy
# can't undo the change). Evaluator's `expected` switches from cloud_file
# (HF gold URL) to vm_file (locally-generated gold).
#
# /home/user/Desktop is hardcoded — both target eval tasks (0ed39f63, ec71221e)
# put their source there, and it's pre-created in the OSWorld VM image.

_TEXT_REPLACE_PAIRS = [
    ("foo", "bar"), ("apple", "grape"), ("dog", "cat"), ("light", "dark"),
    ("yellow", "green"), ("hello", "goodbye"), ("fast", "slow"), ("up", "down"),
]

_TEXT_REPLACE_TEMPLATES = [
    'Please help me change all the places in this document that say "{find}" to "{replace}".',
    'Could you help me replace every "{find}" in this document with "{replace}" for my project?',
    'Please i\'d like to substitute all occurrences of "{find}" with "{replace}" in this file for the repo.',
    'Can you help me change every "{find}" to "{replace}" in this document for my workflow?',
    'Please find all "{find}" in this file and replace them with "{replace}" for this project.',
    'Please in this document, change every "{find}" to "{replace}" for my dev workflow.',
]


def perturb_vscode_text_replace(eval_row: dict, rng: random.Random) -> list[dict]:
    """Find/replace text-edit perturb for compare_text_file tasks (e.g. 0ed39f63).

    Synthesizes source+gold pair: source contains N copies of `find_w`, gold is
    source with `find_w → repl_w` applied. Source is written via perturb_config_step
    (overwriting eval's download); oracle `cp`s gold over source after VS Code is
    killed by the validate pkill.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    if evaluator.get("func") != "compare_text_file":
        return []

    # Detect by config: must download a .txt file (find/replace, not indent).
    source_path = None
    for s in eval_row["metadata"]["config"]:
        if s.get("type") == "download":
            for f in s["parameters"].get("files", []):
                p = f.get("path", "")
                if p.endswith(".txt"):
                    source_path = p
                    break
            if source_path:
                break
    if not source_path:
        return []

    instr_low = eval_row["instruction"].lower()
    if not (("replace" in instr_low or "change" in instr_low) and "say" in instr_low):
        return []

    expected_path = f"/tmp/perturb_expected_{eval_row['task_id'][-8:]}.txt"
    rows: list[dict] = []
    for find_w, repl_w in rng.sample(_TEXT_REPLACE_PAIRS, min(4, len(_TEXT_REPLACE_PAIRS))):
        source = (
            f"Welcome to the document.\n"
            f"This is a test of the {find_w} replacement system.\n"
            f"We use {find_w} as a sample word here.\n"
            f"Multiple {find_w} occurrences should all be changed.\n"
            f"The {find_w} is important.\n"
            f"End of document.\n"
        )
        gold = source.replace(find_w, repl_w)
        instruction = rng.choice(_TEXT_REPLACE_TEMPLATES).format(find=find_w, replace=repl_w)

        cmd = (
            f"mkdir -p /home/user/Desktop && "
            f"cat > '{source_path}' << 'EOFSRC'\n{source}EOFSRC\n"
            f"cat > '{expected_path}' << 'EOFGOLD'\n{gold}EOFGOLD"
        )
        perturb_config_step = {"type": "execute", "parameters": {"command": cmd, "shell": True}}

        oracle = [{"type": "execute", "parameters": {
            "command": f"cp '{expected_path}' '{source_path}'", "shell": True,
        }}]

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["expected"] = {
            "type": "vm_file", "path": expected_path, "dest": "expected_text.txt",
        }

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"find": find_w, "replace": repl_w},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=perturb_config_step,
        ))
    return rows


# Source template for indent perturb — 12-line Python script with predictable indent.
_INDENT_SOURCE_LINES = [
    "def example_function():",
    '    """A simple function for testing."""',
    "    print('Starting computation')",
    "    x = 1",
    "    y = 2",
    "    z = 3",
    "    result = x + y + z",
    "    print('Almost done')",
    "    print('Computing more')",
    "    print('Finished')",
    "    return result",
    "",
]
# (start_line, end_line) — 1-based inclusive ranges. Avoid line 1 (def line)
# and line 12 (blank) to keep edits semantically meaningful.
_INDENT_RANGES = [(2, 5), (3, 7), (4, 9), (5, 10), (2, 8), (6, 11)]

_INDENT_TEMPLATES = [
    "Please help me increase the indent of line {start} to line {end} by one tab.",
    "Could you help me add one tab of indentation to lines {start} through {end} in this file?",
    "Please indent line {start} to line {end} by one extra tab in this file for my project.",
    "Can you help me add one tab to the start of lines {start}-{end} in this document for my repo?",
    "Please increase indentation by one tab for line {start} to line {end} in this file for my workflow.",
    "Please increase indentation by one tab for line {start} to line {end} in this document.",
]


def perturb_vscode_text_indent(eval_row: dict, rng: random.Random) -> list[dict]:
    """Line-indent text-edit perturb for compare_text_file tasks (e.g. ec71221e).

    Synthesizes a fixed Python source template + gold with the (start..end) line
    range prefixed with one tab. Same source-write/gold-write/oracle-cp pattern
    as perturb_vscode_text_replace.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    if evaluator.get("func") != "compare_text_file":
        return []

    source_path = None
    for s in eval_row["metadata"]["config"]:
        if s.get("type") == "download":
            for f in s["parameters"].get("files", []):
                p = f.get("path", "")
                if p.endswith(".py"):
                    source_path = p
                    break
            if source_path:
                break
    if not source_path:
        return []

    if "indent" not in eval_row["instruction"].lower():
        return []

    expected_path = f"/tmp/perturb_expected_{eval_row['task_id'][-8:]}.py"
    rows: list[dict] = []
    source = "\n".join(_INDENT_SOURCE_LINES)
    for start, end in rng.sample(_INDENT_RANGES, min(4, len(_INDENT_RANGES))):
        gold_lines = [
            ("\t" + line) if (start <= i <= end) else line
            for i, line in enumerate(_INDENT_SOURCE_LINES, 1)
        ]
        gold = "\n".join(gold_lines)
        instruction = rng.choice(_INDENT_TEMPLATES).format(start=start, end=end)

        cmd = (
            f"mkdir -p /home/user/Desktop && "
            f"cat > '{source_path}' << 'EOFSRC'\n{source}\nEOFSRC\n"
            f"cat > '{expected_path}' << 'EOFGOLD'\n{gold}\nEOFGOLD"
        )
        perturb_config_step = {"type": "execute", "parameters": {"command": cmd, "shell": True}}

        # Force VS Code's user-settings to a literal Tab character before VS Code
        # launches (eval_row's config has the `code <file>` launch step). Without
        # this, `editor.insertSpaces=true` (VS Code's default in many languages)
        # would make Tab insert 4 spaces — mismatching gold's '\t' bytes.
        # validation finding: previously `echo > settings.json` clobbered
        # the validation `workbench.startupEditor=none` welcome-modal pre-dismiss
        # seed if it was present, causing the welcome modal to hijack turn_00
        # (trigger H). Switch to a python3 read-merge-write so any existing
        # settings (incl. the welcome-modal disable) survive.
        pre_config_steps = [{"type": "execute", "parameters": {
            "command": (
                "mkdir -p /home/user/.config/Code/User && "
                "python3 << 'PYEOF'\n"
                "import json, os\n"
                "path = '/home/user/.config/Code/User/settings.json'\n"
                "existing = {}\n"
                "if os.path.exists(path):\n"
                "    try:\n"
                "        with open(path) as f:\n"
                "            existing = json.load(f)\n"
                "    except (json.JSONDecodeError, OSError):\n"
                "        existing = {}\n"
                "existing.update({\n"
                "    'editor.insertSpaces': False,\n"
                "    'editor.detectIndentation': False,\n"
                "    'editor.useTabStops': True,\n"
                "    'workbench.startupEditor': 'none',\n"
                "})\n"
                "with open(path, 'w') as f:\n"
                "    json.dump(existing, f)\n"
                "PYEOF"
            ),
            "shell": True,
        }}]

        oracle = [{"type": "execute", "parameters": {
            "command": f"cp '{expected_path}' '{source_path}'", "shell": True,
        }}]

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["expected"] = {
            "type": "vm_file", "path": expected_path, "dest": "expected_text.py",
        }

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"start": start, "end": end},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=perturb_config_step,
            pre_config_steps=pre_config_steps,
        ))
    return rows


# ---------------------------------------------------------------------------
# save_workspace: P3-4 base coverage for 5e2d93d8.
# Eval: ``ls /home/user/<dir>/ | grep <name>.code-workspace`` (is_extension_installed
# / contain). Intent: agent uses File > Save Workspace As… to write a
# .code-workspace next to the project. We perturb the workspace name + parent
# dir while keeping the same evaluator shape.
# ---------------------------------------------------------------------------

_WORKSPACE_NAMES = ["myrepo", "devspace", "codebox", "labwork", "scratch", "playground"]
# Re-use _NEW_FILE_PATHS-style anchors but only those eval would naturally use
# for VS Code project trees.
_WORKSPACE_DIRS = [
    "/home/user/project", "/home/user/Documents",
    "/home/user/Desktop", "/home/user/repo",
]

_SAVE_WORKSPACE_TEMPLATES = [
    'Please help me save the current project as a workspace named "{name}.code-workspace" at "{dir}/" so I can reopen this layout later.',
    'Could you help me save VS Code\'s current workspace as "{name}.code-workspace" inside "{dir}/" for my dev workflow?',
    'Please i\'d like to save this project as a workspace file called "{name}.code-workspace" at "{dir}/" — I\'m setting up a new repo.',
    'Can you help me write the current workspace to "{dir}/{name}.code-workspace" via File > Save Workspace As… for this project?',
    'Please capture the current VS Code workspace as "{name}.code-workspace" under "{dir}/" before I keep coding.',
    'Please save the current VS Code workspace as "{name}.code-workspace" in "{dir}/" for my project.',
]


def perturb_vscode_save_workspace(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for "save current project as workspace" tasks.

    Eval signature (5e2d93d8): ``is_extension_installed`` evaluator with
    ``ls <dir>/ | grep <name>.code-workspace`` and ``expected: <name>.code-workspace``
    (rule type ``contain``). The eval intent is *file presence*, not extension
    install — distinguished from the create_file pattern by the
    ``.code-workspace`` extension which we explicitly excluded from
    ``perturb_vscode_create_file``'s ``_REGULAR_FILE_EXTS``.

    Resamples (workspace_name, parent_dir) pairs and rebuilds the oracle
    so the gold ``project.code-workspace`` stays consistent with the new
    name + path.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    if evaluator.get("func") != "is_extension_installed":
        return []
    result = evaluator.get("result", {})
    if result.get("type") != "vm_command_line":
        return []
    cmd = result.get("command", [])
    if not isinstance(cmd, list):
        return []
    cmd_strs = [str(c) for c in cmd]
    if "ls" not in cmd_strs or "grep" not in cmd_strs:
        return []
    if "list-extensions" in " ".join(cmd_strs):
        return []

    expected_rules = evaluator.get("expected", {}).get("rules", {})
    orig_name_full = expected_rules.get("expected", "")
    if not isinstance(orig_name_full, str) or not orig_name_full.endswith(".code-workspace"):
        return []
    orig_name = orig_name_full[: -len(".code-workspace")]
    orig_dir = next((c.rstrip("/") for c in cmd_strs if c.startswith("/")), None)
    if not orig_dir:
        return []

    candidates = [
        (n, d) for n in _WORKSPACE_NAMES for d in _WORKSPACE_DIRS
        if (n, d) != (orig_name, orig_dir)
    ]
    if not candidates:
        return []

    rows = []
    for new_name, new_dir in rng.sample(candidates, min(4, len(candidates))):
        new_full = f"{new_name}.code-workspace"
        instruction = rng.choice(_SAVE_WORKSPACE_TEMPLATES).format(name=new_name, dir=new_dir)

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["result"]["command"] = ["ls", f"{new_dir}/", "|", "grep", new_full]
        new_evaluator["expected"]["rules"]["expected"] = new_full

        # Oracle: ensure dir exists, then write a minimal .code-workspace JSON
        # pointing at the parent dir (matches eval oracle pattern: a single
        # folder entry + empty settings).
        oracle = [
            {"type": "execute", "parameters": {
                "command": f"mkdir -p {new_dir}", "shell": True,
            }},
            {"type": "execute", "parameters": {
                "command": (
                    "echo '" + json.dumps({"folders": [{"path": "."}], "settings": {}}) + "' "
                    f"> {new_dir}/{new_full}"
                ),
                "shell": True,
            }},
        ]

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"workspace_name": new_name, "workspace_dir": new_dir},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


# ---------------------------------------------------------------------------
# files_exclude: P3-4 base coverage for c6bf789c.
# Eval: ``check_json_settings`` with nested dict expected
# ``{"files.exclude": {"**/__pycache__": true}}``. Pattern + boolean toggle
# both perturbable. settings.json's files.exclude semantics: the *entire*
# nested dict must equal the agent-written value (eval does ``data[key] !=
# value`` strict-eq), so we resample one (pattern, hide-bool) pair per
# variant.
# ---------------------------------------------------------------------------

_FILES_EXCLUDE_PATTERNS = [
    "**/__pycache__",
    "**/.git",
    "**/node_modules",
    "**/.DS_Store",
    "**/*.pyc",
    "**/.idea",
    "**/.vscode",
]

_FILES_EXCLUDE_TEMPLATES = [
    'Please help me modify VS Code\'s user settings to hide all "{pattern}" entries in the explorer view.',
    'Could you help me update VS Code\'s user settings so the explorer globally hides "{pattern}" matches?',
    'Please i\'d like VS Code to exclude "{pattern}" from the file explorer in user settings — please update settings.json.',
    'Can you help me set VS Code\'s files.exclude in my user settings so "{pattern}" is hidden in the explorer globally?',
    'Please ensure VS Code\'s explorer stops showing "{pattern}" files. Please add it to files.exclude in your user settings.json.',
    'Please in VS Code\'s user settings, add "{pattern}" to files.exclude so it disappears from the explorer view.',
]


def perturb_vscode_files_exclude(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for VS Code files.exclude tasks (c6bf789c).

    Eval shape: settings.json with
    ``{"files.exclude": {<pattern>: true}}``. We resample <pattern> from a
    pool of common ignore globs, keeping the value ``true`` (hide).
    Evaluator does strict dict-eq on ``data["files.exclude"]``, so the
    oracle writes exactly the new single-pattern dict.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    if evaluator.get("func") != "check_json_settings":
        return []
    result = evaluator.get("result", {})
    if not isinstance(result, dict):
        return []
    result_path = result.get("path", "")
    if "settings.json" not in result_path:
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    exp_settings = rules.get("expected", {})
    if not isinstance(exp_settings, dict):
        return []
    if "files.exclude" not in exp_settings:
        return []
    orig_excl = exp_settings["files.exclude"]
    if not isinstance(orig_excl, dict) or len(orig_excl) != 1:
        return []
    orig_pattern, orig_val = next(iter(orig_excl.items()))
    if not isinstance(orig_val, bool):
        return []

    candidates = [p for p in _FILES_EXCLUDE_PATTERNS if p != orig_pattern]
    rows = []
    for new_pattern in rng.sample(candidates, min(4, len(candidates))):
        new_excl = {new_pattern: True}
        instruction = rng.choice(_FILES_EXCLUDE_TEMPLATES).format(pattern=new_pattern)

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["expected"] = {"files.exclude": new_excl}

        oracle = _build_settings_oracle(result_path, {"files.exclude": new_excl})

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"files_exclude_pattern": new_pattern},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


# ---------------------------------------------------------------------------
# lint_severity: P3-4 base coverage for e2b5e914.
# Eval: ``check_json_settings`` with nested dict
# ``{"python.analysis.diagnosticSeverityOverrides": {"reportMissingImports": "none"}}``.
# Both the diagnostic key and the severity are perturbable. Like
# files.exclude, evaluator does strict dict-eq so the oracle writes the
# whole single-key nested dict.
# ---------------------------------------------------------------------------

_LINT_DIAGNOSTICS = [
    ("reportMissingImports", "missing import errors"),
    ("reportUndefinedVariable", "undefined variable warnings"),
    ("reportUnusedImport", "unused import warnings"),
    ("reportGeneralTypeIssues", "general type issues"),
    ("reportOptionalMemberAccess", "optional member access warnings"),
    ("reportPrivateUsage", "private member usage warnings"),
]
_LINT_SEVERITIES = ["none", "warning", "information", "error"]

_LINT_TEMPLATES = [
    'Please modify VS Code settings to set Python\'s {desc} severity to "{severity}" for my dev workflow.',
    'Could you help me change VS Code so Pylance reports {desc} as "{severity}" in user settings?',
    'Please i\'d like VS Code\'s python.analysis to report {desc} at severity "{severity}" — please update user settings.json.',
    'Can you help me configure VS Code so {desc} are reported as "{severity}" globally?',
    'Please ensure VS Code\'s Python analysis to treat {desc} as "{severity}" for my workflow.',
    'Please in VS Code settings, set the Python diagnostic severity for {desc} to "{severity}".',
]


def perturb_vscode_lint_severity(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for python.analysis.diagnosticSeverityOverrides (e2b5e914).

    Eval shape:
    ``{"python.analysis.diagnosticSeverityOverrides": {<diag>: <severity>}}``.
    We resample (diagnostic_key, severity) excluding the eval pair.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    if evaluator.get("func") != "check_json_settings":
        return []
    result = evaluator.get("result", {})
    if not isinstance(result, dict):
        return []
    result_path = result.get("path", "")
    if "settings.json" not in result_path:
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    exp_settings = rules.get("expected", {})
    if not isinstance(exp_settings, dict):
        return []
    target_key = "python.analysis.diagnosticSeverityOverrides"
    if target_key not in exp_settings:
        return []
    orig_overrides = exp_settings[target_key]
    if not isinstance(orig_overrides, dict) or len(orig_overrides) != 1:
        return []
    orig_diag, orig_sev = next(iter(orig_overrides.items()))
    if not isinstance(orig_sev, str):
        return []

    candidates = [
        (diag, desc, sev)
        for diag, desc in _LINT_DIAGNOSTICS
        for sev in _LINT_SEVERITIES
        if (diag, sev) != (orig_diag, orig_sev)
    ]
    rows = []
    for new_diag, new_desc, new_sev in rng.sample(candidates, min(4, len(candidates))):
        new_overrides = {new_diag: new_sev}
        instruction = rng.choice(_LINT_TEMPLATES).format(desc=new_desc, severity=new_sev)

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["expected"] = {target_key: new_overrides}

        oracle = _build_settings_oracle(result_path, {target_key: new_overrides})

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"diagnostic": new_diag, "severity": new_sev},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


def perturb_vscode_create_file(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for VS Code create-new-file tasks.

    Targets tasks whose evaluator runs ``ls {dir} | grep {name}`` with
    expected={name}. Resamples both filename (.py / .md / .txt / .json)
    and target dir (Desktop / Documents / Downloads) excluding the eval
    pair. Oracle: ``mkdir -p {dir} && touch {dir}/{name}``.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    if evaluator.get("func") != "is_extension_installed":
        return []
    result = evaluator.get("result", {})
    if result.get("type") != "vm_command_line":
        return []
    cmd = result.get("command", [])
    if not isinstance(cmd, list):
        return []
    cmd_strs = [str(c) for c in cmd]
    # Match `ls {dir} | grep {name}` pattern (file-presence check, not extension list).
    if "ls" not in cmd_strs or "grep" not in cmd_strs or "list-extensions" in " ".join(cmd_strs):
        return []

    expected_rules = evaluator.get("expected", {}).get("rules", {})
    orig_name = expected_rules.get("expected", "")
    if not isinstance(orig_name, str) or "." not in orig_name:
        return []
    # Recover original target dir from the ls argument (first absolute-path token).
    orig_path = next((c for c in cmd_strs if c.startswith("/")), None)
    if not orig_path:
        return []

    orig_ext = "." + orig_name.rsplit(".", 1)[-1]
    # Only handle regular source/text files. Excludes .code-workspace (workspace
    # save-as), .vsix (extension package), etc. — these are different operations
    # that look superficially like file-presence checks but aren't "create new file".
    _REGULAR_FILE_EXTS = {
        ".py", ".md", ".txt", ".json", ".html", ".js", ".ts",
        ".yaml", ".yml", ".toml", ".csv", ".xml",
    }
    if orig_ext not in _REGULAR_FILE_EXTS:
        return []
    if orig_ext == ".py":
        # Stay within Python file pool (matches eval semantics: "python file").
        candidates = [(n, "python", p) for n in _NEW_FILE_NAMES_PY for p in _NEW_FILE_PATHS
                      if (n, p) != (orig_name, orig_path)]
    else:
        candidates = [(n, k, p) for (n, k) in _NEW_FILE_NAMES_GENERIC for p in _NEW_FILE_PATHS
                      if (n, p) != (orig_name, orig_path)]
    if not candidates:
        return []

    rows = []
    for new_name, kind, new_path in rng.sample(candidates, min(4, len(candidates))):
        if kind == "python":
            instruction = rng.choice(_CREATE_PY_TEMPLATES).format(filename=new_name, path=new_path)
        else:
            instruction = rng.choice(_CREATE_GENERIC_TEMPLATES).format(filename=new_name, path=new_path, kind=kind)

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["result"]["command"] = ["ls", new_path, "|", "grep", new_name]
        new_evaluator["expected"]["rules"]["expected"] = new_name

        oracle = [
            {"type": "execute", "parameters": {
                "command": f"mkdir -p '{new_path}'", "shell": True,
            }},
            {"type": "execute", "parameters": {
                "command": f"touch '{new_path}/{new_name}'", "shell": True,
            }},
        ]

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"filename": new_name, "path": new_path},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


_INTERNAL_FNS = [
    perturb_vscode_setting,
    perturb_vscode_extension,
    perturb_vscode_keybinding,
    perturb_vscode_bool_setting,
    perturb_vscode_workspace,
    perturb_vscode_project_folder,
    perturb_vscode_create_file,
    perturb_vscode_text_replace,
    perturb_vscode_text_indent,
    perturb_vscode_save_workspace,
    perturb_vscode_files_exclude,
    perturb_vscode_lint_severity,
]


def perturb_vscode_per_task(
    eval_row: dict,
    rng: random.Random,
    max_type1: int = 4,
) -> list[dict]:
    """Iterate internal op fns; each contributes up to max_type1 unique rows."""
    rows: list[dict] = []
    seen: set[str] = set()
    for fn in _INTERNAL_FNS:
        try:
            sub = fn(eval_row, rng)
        except Exception:
            logger.exception("%s failed for %s", fn.__name__, eval_row["task_id"])
            continue
        for r in sub[:max_type1]:
            if r["task_id"] not in seen:
                rows.append(r)
                seen.add(r["task_id"])
    return rows
