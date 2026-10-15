"""GIMP domain structural perturbation functions (Track B).

Handles two categories of GIMP eval tasks:

1. config_setting (check_config_status): perturb gimprc key-value pairs
   (theme, undo-levels, layer-new-name, tile-cache-size, etc.)
2. image_resize (check_image_size): perturb target dimensions

Usage:
    from lite.gym.envs.lite.osworld.src.gen.train.perturb.gimp import GIMP_PERTURB_FNS
    rows = GIMP_PERTURB_FNS["config_setting"](eval_row, rng)
"""

from __future__ import annotations

import copy
import json
import logging
import random
import re
import textwrap

from lite.gym.envs.lite.osworld.src.gen.train.perturb._utils import make_perturb_row, oracle_actions_of

logger = logging.getLogger(__name__)

_GIMP_CONFIG_DIR = "/home/user/.config/GIMP/2.10"

# Robust gimp-family kill for image-op / misc oracles.
#
# ROOT CAUSE (image_op + misc triangle 0.0s): the task `config` LAUNCHES gimp
# with the ORIGINAL source image and leaves it running. The evaluator
# `postconfig` (inherited from the eval base) fires Shift+Ctrl+E → type the
# result filename → Enter, EXPORTING gimp's currently-open image to result_path.
# In the agent path that captures the agent's edit; in the ORACLE path (which
# short-circuits by cp'ing a PIL gold to result_path) that export re-writes the
# still-open ORIGINAL over the cp'd gold → check_structure_sim / triangle_center
# score 0 for every structure-ALTERING op (contrast, saturation, fill, rotate,
# triangle-center). Structure-PRESERVING ops (brightness, mirror, resize) pass
# by SSIM leniency, which is exactly why only a subset failed.
#
# The old `killall gimp 2>/dev/null; sleep 1` missed the running process two
# ways: (1) wrong comm — the process is `gimp-2.10` / `script-fu`, not `gimp`
# (the eval base uses `pkill -f gimp` / `pgrep -if gimp`, and the postconfig's
# own wait-loop polls `pgrep -x gimp-2.10`); (2) no wait — under load the
# config-launched gimp is still booting when `sleep 1` elapses, so the kill runs
# before the process exists. We KILL the whole gimp family by exact comm
# (`killall`, no self-match) repeatedly across the boot window so a late-booting
# gimp is caught, guaranteeing gimp is dead before the postconfig export fires →
# the export is a no-op and the cp'd gold survives. Oracle-only: the postconfig
# is untouched, so agent rollouts still export their own edit normally.
_GIMP_KILL_CMD = (
    "for i in $(seq 1 12); do "
    "killall -q -9 gimp gimp-2.10 script-fu python-fu 2>/dev/null; "
    "sleep 1; done; true"
)

# ---------------------------------------------------------------------------
# config_setting: perturb gimprc / sessionrc key-value pairs
# ---------------------------------------------------------------------------

# Maps gimprc key -> perturbation spec
_GIMPRC_SETTINGS: dict[str, dict] = {
    "theme": {
        "values": ['"Light"', '"Dark"', '"System"'],
        "instruction_templates": [
            "Could you switch GIMP's theme to {value_display}? My eyes need a break.",
            "Please change the GIMP appearance theme to {value_display} so it matches my desktop.",
            "Can you set GIMP's theme to {value_display} — I'd like a more comfortable look while editing.",
            "Set the GIMP theme to {value_display} so the workspace feels easier on the eyes.",
            "Change GIMP's color theme to {value_display} for a fresher editing environment.",
            # D5: short polite variant (8 words) — pulls perturb p25 down toward eval p25=11.
            "Could you switch GIMP's theme to {value_display}?",
        ],
    },
    "undo-levels": {
        "values": ["25", "50", "75", "100", "125", "150", "200", "250", "300"],
        "instruction_templates": [
            "Could you bump GIMP's minimum undo steps up to {value_display}? I make a lot of mistakes.",
            "Please set GIMP's undo history limit to {value_display} steps for more room to roll back edits.",
            "Can you raise the undo levels in GIMP to {value_display} — I want a longer safety net.",
            "Set the minimum number of undo steps in GIMP to {value_display} for safer edits.",
            "Change the GIMP undo history limit to {value_display} steps so deeper rollbacks are possible.",
            # D5: short polite variant (8 words).
            "Please set GIMP's undo limit to {value_display} steps.",
        ],
    },
    "layer-new-name": {
        "values": ['"Circle"', '"Triangle"', '"Rectangle"', '"Diamond"', '"Star"', '"Oval"', '"Square"', '"Hexagon"'],
        # NOTE: GIMP has NO Preferences UI for
        # layer-new-name. The only way to update gimprc's default is via
        # the New Layer dialog (Layer → New Layer…) — when you type a
        # name there and click OK, gimprc records that name as the new
        # default. Templates now direct the agent through that dialog.
        "instruction_templates": [
            "Could you open Layer → New Layer… in GIMP, type {value_display} as the name, and click OK?",
            "Please use GIMP's Layer → New Layer… dialog to create a layer named {value_display} for me.",
            "Can you open the New Layer dialog (Layer → New Layer…) and add a layer named {value_display}?",
            "Open GIMP's New Layer dialog and create a layer named {value_display} so future layers default to that name.",
            "Add a new layer named {value_display} via Layer → New Layer… in GIMP for the next composition.",
        ],
    },
    "tile-cache-size": {
        "values": ["1073741824", "2147483648", "4294967296", "536870912"],
        "display_map": {
            "1073741824": "1 GB",
            "2147483648": "2 GB",
            "4294967296": "4 GB",
            "536870912": "512 MB",
        },
        "instruction_templates": [
            "Could you set GIMP's tile cache size to {value_display}? I'm working with bigger images now.",
            "Please bump GIMP's tile cache to {value_display} so large photos don't keep slowing down.",
            "Can you adjust GIMP's cache memory to {value_display} — performance has been sluggish.",
            "Change GIMP's tile cache size to {value_display} for better performance on big images.",
            "Set the GIMP tile cache to {value_display} so large compositions stay responsive.",
            # D5: short imperative variant (7 words) — terse style for variety.
            "Set GIMP's tile cache to {value_display}.",
        ],
    },
    "hide-docks": {
        "values": ["yes", "no"],
        "instruction_templates": [
            "Could you {verb_lower} the docks in GIMP for me? I want a {dock_reason} workspace.",
            "Please {verb_lower} GIMP's side tool docks — I'd like {dock_reason} screen space.",
            "Can you {verb_lower} the side docks in the GIMP window? It'll let me focus on the canvas.",
            "{verb} the side panels (docks) in GIMP to give the canvas a {dock_reason} feel.",
            "{verb} GIMP's tool docks so the workspace feels {dock_reason} for this editing session.",
        ],
    },
    "default-grid": {
        "values": ["8", "10", "16", "20", "24", "32", "48", "64"],
        "instruction_templates": [
            "Could you set GIMP's default grid spacing to {value_display} pixels for finer alignment?",
            "Please change the GIMP grid spacing to {value_display} pixels for my next layout.",
            "Can you adjust GIMP's default grid to {value_display}px — I want a tighter snap while designing.",
            "Change GIMP's default grid spacing to {value_display} pixels for more precise alignment.",
            "Set the default grid spacing in GIMP to {value_display} pixels so layouts snap cleanly.",
            # D5: short polite variant (9 words).
            "Could you set GIMP's grid spacing to {value_display} pixels?",
        ],
    },
}


def _get_gimprc_key_from_evaluator(eval_row: dict) -> tuple[str, str, str]:
    """Extract gimprc key, current value, and config file from evaluator rules."""
    evaluator = eval_row["metadata"]["evaluator"]
    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    key = rules.get("key", "")
    value = rules.get("value", "")

    result = evaluator.get("result", {})
    file_name = result.get("file_name", "gimprc")
    return key, value, file_name


def _build_gimprc_oracle(key: str, value: str, file_name: str) -> list[dict]:
    """Build oracle commands that set a gimprc key-value pair."""
    config_path = f"{_GIMP_CONFIG_DIR}/{file_name}"
    # gimprc uses s-expression format: (key value)
    return [
        {"type": "execute", "parameters": {
            "command": "killall gimp 2>/dev/null; sleep 1; true", "shell": True,
        }},
        {"type": "execute", "parameters": {
            "command": f"mkdir -p {_GIMP_CONFIG_DIR}", "shell": True,
        }},
        {"type": "execute", "parameters": {
            "command": (
                f"python3 << 'PYEOF'\n"
                f"import os\n"
                f"path = '{config_path}'\n"
                f"lines = []\n"
                f"if os.path.exists(path):\n"
                f"    with open(path) as f:\n"
                f"        lines = f.readlines()\n"
                f"lines = [l for l in lines if not l.strip().startswith('({key} ')]\n"
                f"lines.append('({key} {value})\\n')\n"
                f"with open(path, 'w') as f:\n"
                f"    f.writelines(lines)\n"
                f"PYEOF"
            ),
            "shell": True,
        }},
    ]


def perturb_gimp_config(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for GIMP config_setting tasks (check_config_status)."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "check_config_status":
        return []

    key, orig_value, file_name = _get_gimprc_key_from_evaluator(eval_row)
    if not key:
        return []

    spec = _GIMPRC_SETTINGS.get(key)
    if not spec:
        return []

    candidates = [v for v in spec["values"] if v != orig_value]
    if not candidates:
        return []

    # R1: when the candidate pool is too small (e.g. hide-docks orig=yes leaves
    # only "no"), fall back to multiple paraphrase variants of the same value
    # so the base still emits 3+ training rows. Otherwise, preserve the
    # existing behaviour of emitting 1 row per distinct candidate value (up
    # to cap=4) so the per-base row counts stay near the pre-R1 distribution.
    cap = 4
    n_templates = len(spec["instruction_templates"])
    rng.shuffle(candidates)
    pairs: list[tuple[str, int]] = []
    if len(candidates) >= cap:
        # Plenty of distinct values — 1 random template per value.
        for v in candidates[:cap]:
            pairs.append((v, rng.randrange(n_templates)))
    else:
        # Small candidate pool → fill up to `cap` rows with paraphrase variants
        # cycling through templates per value.
        template_idx = list(range(n_templates))
        rng.shuffle(template_idx)
        for v in candidates:
            for t in template_idx:
                pairs.append((v, t))
                if len(pairs) >= cap:
                    break
            if len(pairs) >= cap:
                break

    display_map = spec.get("display_map", {})
    rows = []
    seen_instructions: set[str] = set()
    value_use_count: dict[str, int] = {}
    for new_value, t_idx in pairs:
        value_display = display_map.get(new_value, new_value.strip('"'))
        template = spec["instruction_templates"][t_idx]

        # Build instruction
        if key == "hide-docks":
            verb = "Hide" if new_value == "yes" else "Show"
            verb_lower = "hide" if new_value == "yes" else "show"
            dock_reason = "more spacious" if new_value == "yes" else "more familiar"
            instruction = template.format(
                verb=verb, verb_lower=verb_lower, dock_reason=dock_reason,
            )
        else:
            instruction = template.format(value_display=value_display)

        # Skip duplicates (rare — different templates with identical formatted
        # output) so V4d uniqueness check passes.
        if instruction in seen_instructions:
            continue
        seen_instructions.add(instruction)

        new_evaluator = copy.deepcopy(evaluator)
        exp = new_evaluator.get("expected", {})
        if isinstance(exp, list):
            exp = exp[0]
        exp["rules"]["value"] = new_value

        # NOTE: original flush_postconfig
        # used escape + pkill TERM/KILL, which caused SIGTERM to race with any
        # "save image?" dialog — GIMP hung on the dialog, then got SIGKILL before
        # it could write gimprc. Replace with the same synth-style graceful quit:
        # activate-window → ctrl+q → sleep 2 → Return (dismiss save dialog) → wait.
        # This matches synth_gimp_set_theme_* and reliably flushes gimprc.
        new_evaluator["postconfig"] = [
            # Validation note: match by WM_CLASS instead of title
            # substring. With no image loaded, GIMP's title is "GNU Image
            # Manipulation Program" (no "GIMP" substring), so the prior
            # window_name="GIMP" form silently failed → ctrl+q misrouted →
            # gimprc/sessionrc never flushed. WM_CLASS "Gimp" is stable
            # across image-loaded and image-less GIMP windows.
            {"type": "activate_window", "parameters": {"window_name": "Gimp", "by_class": True}},
            {"type": "key", "parameters": {"key": "ctrl+q"}},
            {"type": "sleep", "parameters": {"seconds": 2}},
            # validation+ audit (a746add2 / 7b7617bd / d52d6308 / 7767eef2):
            # GIMP's "Quit GIMP — Save changes?" dialog defaults focus to
            # Cancel, so a bare `Return` = cancel quit → GIMP keeps running →
            # gimprc never flushed (gimprc only persists on graceful exit).
            # The Discard button has accelerator alt+d (text "_Discard Changes").
            # Try alt+d first (no-op if no dialog), then Return as fallback for
            # other dialog variants. Sleep brief to let dialog dismiss.
            {"type": "execute", "parameters": {
                "command": (
                    "WID=$(xdotool search --name 'Quit GIMP' 2>/dev/null | head -1); "
                    "if [ -n \"$WID\" ]; then xdotool windowactivate \"$WID\" key alt+d; fi; true"
                ),
                "shell": True,
            }},
            {"type": "sleep", "parameters": {"seconds": 1}},
            {"type": "key", "parameters": {"key": "Return"}},
            {
                "type": "execute",
                "parameters": {
                    "command": (
                        "for i in $(seq 1 30); do "
                        "  if ! pgrep -x gimp > /dev/null 2>&1 && "
                        "     ! pgrep -x gimp-2.10 > /dev/null 2>&1; then "
                        "    break; "
                        "  fi; "
                        "  sleep 1; "
                        "done; "
                        "pkill -f gimp 2>/dev/null || true; sleep 1"
                    ),
                    "shell": True,
                },
            },
        ]

        oracle = _build_gimprc_oracle(key, new_value, file_name)

        # Pre-config: set the gimprc key to a value DIFFERENT from new_value,
        # so the agent has actual work to do. Without this, the original
        # gimprc may already match new_value (e.g. fresh GIMP session has
        # `hide-docks` absent → reads as default "no", so a perturb with
        # new_value="no" passes vacuously without agent action).
        init_value = orig_value if orig_value and orig_value != new_value else (
            spec["values"][0] if spec["values"][0] != new_value else spec["values"][1]
        )
        config_path = f"{_GIMP_CONFIG_DIR}/{file_name}"
        pre_config_steps = [
            {"type": "execute", "parameters": {"command": f"mkdir -p {_GIMP_CONFIG_DIR}", "shell": True}},
            {"type": "execute", "parameters": {
                "command": (
                    f"python3 << 'PYEOF'\n"
                    f"import os\n"
                    f"path = '{config_path}'\n"
                    f"lines = []\n"
                    f"if os.path.exists(path):\n"
                    f"    with open(path) as f:\n"
                    f"        lines = f.readlines()\n"
                    f"lines = [l for l in lines if not l.strip().startswith('({key} ')]\n"
                    f"lines.append('({key} {init_value})\\n')\n"
                    f"with open(path, 'w') as f:\n"
                    f"    f.writelines(lines)\n"
                    f"PYEOF"
                ),
                "shell": True,
            }},
        ]

        # Add paraphrase index to knob_assignment so duplicate-value rows
        # (R1 fallback for small candidate pools) hash to distinct task_ids.
        # When we use a given value only once we keep the original assignment
        # shape ({key: value}) for backward-compatibility with audit/V4 scripts.
        knob: dict = {key: new_value}
        n_used = value_use_count.get(new_value, 0)
        if n_used > 0:
            knob["paraphrase_idx"] = n_used
        value_use_count[new_value] = n_used + 1

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment=knob,
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            pre_config_steps=pre_config_steps,
        ))

    return rows


# ---------------------------------------------------------------------------
# image_resize: perturb target dimensions for check_image_size tasks
# ---------------------------------------------------------------------------

_RESIZE_VALUES = [256, 384, 512, 640, 768, 800, 1024, 1280]  # 128 dropped: SSIM unstable at very small sizes

_RESIZE_TEMPLATES = [
    # validation finding: source XCFs (e.g. d16c99dc dog_with_background_two_layers.xcf)
    # have multiple layers — "image layer" sent the agent to Layer→Scale Layer
    # which only resizes the active layer (usually the empty top one) and leaves
    # the export height unchanged. Reworded to "the image" so the agent uses
    # Image→Scale Image which scales all layers + canvas uniformly.
    "Could you resize the image to {size} pixels in height while keeping the original aspect ratio?",
    "Please adjust the image height to {size} pixels — keep the proportions so nothing looks stretched.",
    "Can you resize the image to a height of {size} pixels, maintaining the same aspect ratio?",
    "Resize the image to a height of {size} pixels, keeping the proportions intact.",
    "Adjust the image height to {size} pixels with the original aspect ratio preserved.",
    # D5: short imperative variant (10 words) — pulls perturb p25 down.
    "Resize the image to {size}px height, keeping aspect ratio.",
]


def perturb_gimp_image_resize(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for GIMP image resize tasks (check_image_size)."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")

    # Handle compound funcs like ['check_image_size', 'check_structure_sim_resized']
    funcs = func if isinstance(func, list) else [func]
    if "check_image_size" not in funcs:
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        exp_size = expected[0] if expected else {}
    else:
        exp_size = expected

    if not isinstance(exp_size, dict):
        return []
    rules = exp_size.get("rules", {})
    orig_height = rules.get("height")
    if not orig_height or not isinstance(orig_height, int):
        return []

    candidates = [s for s in _RESIZE_VALUES if s != orig_height]
    if not candidates:
        return []

    rows = []
    for new_size in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_RESIZE_TEMPLATES).format(size=new_size)

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp[0]["rules"]["height"] = new_size
        else:
            new_exp["rules"]["height"] = new_size

        # Update oracle: replace old size with new size in oracle commands
        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and str(orig_height) in cmd:
                action["parameters"]["command"] = cmd.replace(
                    str(orig_height), str(new_size),
                )

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"image_height": new_size},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# filter_action: perturb GIMP filter/action history tasks
# ---------------------------------------------------------------------------

_GIMP_FILTERS = [
    ("filters-vignette", "Vignette", "Filters > Light and Shadow > Vignette"),
    ("filters-gaussian-blur", "Gaussian Blur", "Filters > Blur > Gaussian Blur"),
    ("filters-unsharp-mask", "Unsharp Mask (Sharpen)", "Filters > Enhance > Unsharp Mask"),
    ("filters-emboss", "Emboss", "Filters > Distorts > Emboss"),
    ("filters-edge-detect", "Difference of Gaussians (Edge-Detect)", "Filters > Edge-Detect > Difference of Gaussians"),
    ("filters-posterize", "Posterize", "Colors > Posterize"),
    ("filters-pixelize", "Pixelize", "Filters > Blur > Pixelize"),
]

_FILTER_TEMPLATES = [
    "Could you apply the {display_name} filter to this image in GIMP via {menu_path}?",
    "Please run the {display_name} filter on my image in GIMP using {menu_path} — I want to see the effect.",
    "Can you add the {display_name} filter to the image in GIMP? Use {menu_path}.",
    "Apply the {display_name} filter to the image in GIMP ({menu_path}) for a stylized look.",
    "Use the {display_name} filter on this image in GIMP via {menu_path} to dress it up a bit.",
]


def perturb_gimp_filter_action(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for GIMP filter/action-history tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "check_include_exclude":
        return []

    result = evaluator.get("result", {})
    result_cmd = result.get("command", "")
    if isinstance(result_cmd, list):
        result_cmd = " ".join(result_cmd)
    if "action-history" not in result_cmd:
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    includes = rules.get("include", [])
    if not includes:
        return []
    orig_filter = includes[0]

    candidates = [(fid, name, path) for fid, name, path in _GIMP_FILTERS if fid != orig_filter]
    if not candidates:
        return []

    rows = []
    for filter_id, display_name, menu_path in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_FILTER_TEMPLATES).format(
            display_name=display_name, menu_path=menu_path,
        )

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["include"] = [filter_id]

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_filter in cmd:
                action["parameters"]["command"] = cmd.replace(orig_filter, filter_id)

        # Ensure GIMP is fully dead before writing action-history, so GIMP
        # cannot overwrite it on exit.  Use SIGKILL (-9) since SIGTERM lets
        # GIMP save its own action-history.
        kill_step = {"type": "execute", "parameters": {
            "command": "killall -9 gimp 2>/dev/null; sleep 1; true", "shell": True,
        }}
        if not any("killall" in json.dumps(a) for a in oracle):
            oracle.insert(0, kill_step)

        # NOTE: action-history is only written when
        # GIMP exits gracefully.  Use the same synth-style postconfig as
        # perturb_gimp_config: activate window → Ctrl+Q → sleep 2 → Return
        # (dismiss any unsaved-changes dialog) → wait-for-exit loop.
        new_evaluator["postconfig"] = [
            # Validation note: match by WM_CLASS instead of title
            # substring. With no image loaded, GIMP's title is "GNU Image
            # Manipulation Program" (no "GIMP" substring), so the prior
            # window_name="GIMP" form silently failed → ctrl+q misrouted →
            # gimprc/sessionrc never flushed. WM_CLASS "Gimp" is stable
            # across image-loaded and image-less GIMP windows.
            {"type": "activate_window", "parameters": {"window_name": "Gimp", "by_class": True}},
            {"type": "key", "parameters": {"key": "ctrl+q"}},
            {"type": "sleep", "parameters": {"seconds": 2}},
            # validation+ audit (a746add2 / 7b7617bd / d52d6308 / 7767eef2):
            # GIMP's "Quit GIMP — Save changes?" dialog defaults focus to
            # Cancel, so a bare `Return` = cancel quit → GIMP keeps running →
            # gimprc never flushed (gimprc only persists on graceful exit).
            # The Discard button has accelerator alt+d (text "_Discard Changes").
            # Try alt+d first (no-op if no dialog), then Return as fallback for
            # other dialog variants. Sleep brief to let dialog dismiss.
            {"type": "execute", "parameters": {
                "command": (
                    "WID=$(xdotool search --name 'Quit GIMP' 2>/dev/null | head -1); "
                    "if [ -n \"$WID\" ]; then xdotool windowactivate \"$WID\" key alt+d; fi; true"
                ),
                "shell": True,
            }},
            {"type": "sleep", "parameters": {"seconds": 1}},
            {"type": "key", "parameters": {"key": "Return"}},
            {
                "type": "execute",
                "parameters": {
                    "command": (
                        "for i in $(seq 1 30); do "
                        "  if ! pgrep -x gimp > /dev/null 2>&1 && "
                        "     ! pgrep -x gimp-2.10 > /dev/null 2>&1; then "
                        "    break; "
                        "  fi; "
                        "  sleep 1; "
                        "done; "
                        "pkill -f gimp 2>/dev/null || true; sleep 1"
                    ),
                    "shell": True,
                },
            },
        ]

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"filter": filter_id},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# image_op: TYPE_1 + TYPE_2 perturbations using PIL gold (source+gold pattern).
# Replaces eval's op-specific evaluator with check_structure_sim against a PIL-
# computed gold. The agent reproduces the op in GIMP UI; oracle short-circuits
# by cp'ing the gold to result_path.
# ---------------------------------------------------------------------------

# Per-base-task spec: maps tid -> source path / result path / TYPE_1 op + pool.
# `t1_pool` excludes the eval's own param value (no-leakage). `feasible_t2_ops`
# is the candidate set for TYPE_2 (op different from t1_op, image properties OK).
_IMAGE_TASKS: dict[str, dict] = {
    # NOTE: TYPE_1 pools cover BOTH directions for brightness/contrast/saturation —
    # since the perturb evaluator is `check_structure_sim` (against PIL gold), not
    # the original directional eval, the agent's task is "match this specific
    # transformation" regardless of direction. Direction-agnostic instructions in
    # `_OP_INSTR_TEMPLATES` describe the target via param/pct, not increase/decrease.
    # Pool feasibility note (validation-driven):
    # - brightness 0.5 fails GIMP "Brightness-Contrast slider" (additive) ↔ PIL (multiplicative) SSIM≥0.9
    #   when agent uses the obvious tool. Range tightened to [0.6, 1.5].
    # - saturation 2.0 borderline (HSV-slider vs PIL.Color SSIM≈0.90). Capped at 1.8.
    # - contrast all factors are GIMP-friendly (mean-anchor vs 128-anchor SSIM≥0.90).
    # D3 audit (Phase 3-B): per-base check_structure_sim trimmed from 32 → ~14
    # rows to deflate over-representation (Audit 5 ratio 6×eval). Pool sizes
    # below preserve op diversity (each base still emits ≥1 T1 + 1 T2). Op-
    # specific T1 pool kept to 1-2 high-quality params (slider-friendly,
    # SSIM ≥ 0.9 vs GIMP UI); T2 cap = 1 instead of 2 to avoid duplicating
    # neighbouring bases' coverage of the same op.
    "7a4deb26": {  # was: brightness↓
        "src": "/home/user/Desktop/woman_sitting_by_the_tree.png",
        "result": "/home/user/Desktop/edited_darker.png",
        "t1_op": "brightness",
        "t1_pool": [0.7],   # 1 T1 darker + 1 T2 → 2 rows
        "feasible_t2_ops": ("contrast", "saturation", "mirror", "rotate"),
    },
    "f723c744": {  # was: contrast↑
        "src": "/home/user/Desktop/berries.png",
        "result": "/home/user/Desktop/berries_contrast.png",
        "t1_op": "contrast",
        # validation: 1.8 dropped → 1.5. check_structure_sim uses SSIM threshold
        # 0.9 (upstream default). PIL ImageEnhance.Contrast is mean-anchored
        # (out = mean + (in-mean)*f) whereas GIMP's Brightness-Contrast slider is
        # 128-anchored — at f=1.8 on this high-key berries image the two curves
        # diverge enough to push SSIM < 0.9, so a correct GIMP edit scored 0. No
        # single GIMP slider value reproduces PIL contrast, so we can't specify an
        # exact slider; instead pick a GIMP-reproducible factor (mirrors brightness
        # 0.5→range tighten / saturation 2.0→cap fixes above). 1.5 ⇒ "150%" instr.
        "t1_pool": [1.5],   # 1 T1 boost + 1 T2 → 2 rows
        "feasible_t2_ops": ("brightness", "saturation", "mirror", "rotate"),
    },
    "554785e9": {  # was: saturation↑
        "src": "/home/user/Desktop/woman_sitting_by_the_tree2.png",
        "result": "/home/user/Desktop/edited_colorful.png",
        "t1_op": "saturation",
        "t1_pool": [0.5],   # 1 T1 desaturate + 1 T2 → 2 rows
        "feasible_t2_ops": ("brightness", "contrast", "mirror", "rotate"),
    },
    "72f83cdc": {  # mirror H
        "src": "/home/user/Desktop/berry.png",
        "result": "/home/user/Desktop/berry_mirror.png",
        "t1_op": "mirror",
        "t1_pool": ["TB"],   # eval orig is LR; only TB (vertical) as alternative
        "feasible_t2_ops": ("brightness", "contrast", "saturation", "rotate"),
    },
    "06ca5602": {  # mode → palette (Indexed). Alt: Grayscale.
        "src": "/home/user/Desktop/computer.png",
        # Separate output file (NOT in-place). The perturb metric is check_structure_sim
        # (no mode check), so an in-place output == source would trivially pass for a
        # grayscale/mode op (grayscale-of-X is structurally ≈ X). A distinct file means
        # the untouched start has no result → non-trivial. (The eval BASE uses in-place
        # computer.png + check_palette_and_structure_sim, whose mode=='P' check keeps it
        # non-trivial — but perturb can't rely on that, hence the separate path here.)
        "result": "/home/user/Desktop/palette_computer.png",
        "t1_op": "mode",
        "t1_pool": ["L"],    # only Grayscale; orig P (Indexed) excluded
        "feasible_t2_ops": ("brightness", "contrast", "saturation", "mirror", "rotate"),
    },
    "734d6579": {  # bg → green (object stays black). Replace bg with other color.
        "src": "/home/user/Desktop/white_background_with_object.png",
        "result": "/home/user/Desktop/green_background_with_object.png",
        "t1_op": "fill_color",
        # D3: trimmed 6 → 1 color (was full pool, deflated fill_color from 6
        # rows to 1). Eval base alone tied up 8 of 32 structure_sim rows.
        "t1_pool": ["red"],
        "feasible_t2_ops": ("brightness", "contrast", "saturation", "mirror", "rotate"),
    },
}

# Color name → RGB triple, used for the fill_color op.
_FILL_COLORS: dict[str, tuple[int, int, int]] = {
    "red":    (255, 0,   0  ),
    "green":  (0,   255, 0  ),
    "blue":   (0,   0,   255),
    "yellow": (255, 255, 0  ),
    "purple": (128, 0,   128),
    "orange": (255, 165, 0  ),
    "cyan":   (0,   255, 255),
}

# TYPE_2 candidate-param pools, indexed by op. Each entry holds a small set of
# realistic param values used when the op is sampled as a TYPE_2 variant for any
# base task. Pools are cross-base-task (op meaning is image-content-independent).
_T2_POOLS: dict[str, list] = {
    "brightness": [0.7, 1.3, 1.5],     # 0.5 dropped: SSIM<0.9 vs naive GIMP UI
    "contrast":   [1.2, 1.5, 1.8, 2.0],
    "saturation": [0.5, 1.5, 1.8],     # 2.0 dropped: SSIM<0.9 borderline
    "mirror":     ["LR", "TB"],
    "rotate":     [90, 180, 270],
}

_OP_INSTR_TEMPLATES: dict[str, list[str]] = {
    # Direction-agnostic for enhance ops: pct = param×100. param<1.0 ⇒ darker /
    # less / decrease; param>1.0 ⇒ brighter / more / increase. All templates
    # include `{out}` (the filename to export to) so the agent knows where to
    # save the result — feasibility validation found this missing in v1.
    # ≥50% polite-starting (Could you / Please / Can you / I'd like / I want)
    # to match eval style (gimp eval is 77% polite). Motivation/context kept
    # short to stay within 14-22 word target.
    "brightness": [
        "Could you adjust the brightness to {pct}% of the source and export as '{out}' on the Desktop?",
        "Please scale the brightness by {param}× so the photo looks better, then export as '{out}' on the Desktop.",
        "Can you change the brightness to {pct}% of the source and save as '{out}' on the Desktop?",
        "Apply a brightness multiplier of {param} to the image and export as '{out}' on the Desktop.",
        "Adjust the image brightness to {pct}% of the source and export as '{out}' on the Desktop.",
    ],
    "contrast": [
        "Could you push the contrast to {pct}% of the original and export as '{out}' on the Desktop?",
        "Please scale the contrast by {param}× using Curves or Levels and export as '{out}' on the Desktop.",
        "Can you change the contrast to {pct}% of the source so details read clearly, exporting as '{out}' on the Desktop?",
        "Adjust the image contrast to {pct}% of the source for a punchier look, exporting as '{out}' on the Desktop.",
        "Multiply the contrast by {param} (around the mean) and export the result as '{out}' on the Desktop.",
    ],
    "saturation": [
        "Could you change the color saturation to {pct}% of the source and export as '{out}' on the Desktop?",
        "Please scale the saturation by {param}× — I want a different mood — and export as '{out}' on the Desktop.",
        "Can you set the saturation to {pct}% of the source for a more pleasing look, exporting as '{out}' on the Desktop?",
        "Apply a saturation multiplier of {param} to the image and export as '{out}' on the Desktop.",
        "Adjust the image saturation to {pct}% of the source and export the result as '{out}' on the Desktop.",
    ],
    "mirror_LR": [
        "Could you mirror this image horizontally (left-right flip) and export as '{out}' on the Desktop?",
        "Please flip the photo left-to-right — I need the reverse orientation — and save as '{out}' on the Desktop.",
        "Can you apply a horizontal flip to the image and export as '{out}' on the Desktop?",
        "Mirror the image horizontally (left-to-right flip) and export the result as '{out}' on the Desktop.",
        "Flip the image left-to-right and export the result as '{out}' on the Desktop.",
    ],
    "mirror_TB": [
        "Could you flip this image vertically (top-to-bottom mirror) and export as '{out}' on the Desktop?",
        "Please mirror the picture upside-down — I want a vertical reflection — and save as '{out}' on the Desktop.",
        "Can you apply a top-bottom flip to the image and export as '{out}' on the Desktop?",
        "Mirror the image vertically (top-to-bottom flip) and export the result as '{out}' on the Desktop.",
        "Flip the image upside down (vertical mirror) and export as '{out}' on the Desktop.",
    ],
    "rotate": [
        "Could you rotate this image {param}° counter-clockwise (expand canvas to fit) and export as '{out}' on the Desktop?",
        "Please rotate the picture by {param} degrees, letting the canvas grow, and save as '{out}' on the Desktop.",
        "Can you turn the image {param}° counter-clockwise and export as '{out}' on the Desktop?",
        "Rotate the image {param} degrees counter-clockwise (expand canvas to fit) and export as '{out}' on the Desktop.",
        "Apply a {param}-degree counter-clockwise rotation to the image and export as '{out}' on the Desktop.",
    ],
    "mode": [
        "Could you convert this image to grayscale (L mode) and export as '{out}' on the Desktop?",
        "Please change the color mode to grayscale — I want a black-and-white version — and save as '{out}' on the Desktop.",
        "Can you desaturate the image into grayscale and export as '{out}' on the Desktop?",
        "Convert the image to grayscale (L mode) and export the result as '{out}' on the Desktop.",
        "Desaturate the image fully — turn it grayscale — and export as '{out}' on the Desktop.",
    ],
    "fill_color": [
        "Could you repaint the background solid {param} while keeping the black object intact, exporting as '{out}' on the Desktop?",
        "Please change the background to {param}, leaving the dark object as-is, and save as '{out}' on the Desktop.",
        "Can you fill the non-object area with {param} and export as '{out}' on the Desktop?",
        "Repaint everything except the black silhouette in {param} and export as '{out}' on the Desktop.",
        "Make the non-object area {param} (object pixels stay black) and export as '{out}' on the Desktop.",
    ],
}


def _build_pil_expected_py(op: str, param, src_path: str, expected_path: str) -> str:
    """Generate Python heredoc body that writes gold image to expected_path."""
    _ENH = {"brightness": "Brightness", "contrast": "Contrast", "saturation": "Color"}
    if op in _ENH:
        return textwrap.dedent(f"""\
            from PIL import Image, ImageEnhance
            img = Image.open({src_path!r})
            out = ImageEnhance.{_ENH[op]}(img).enhance({param})
            out.save({expected_path!r})
        """)
    if op == "mirror":
        method = "FLIP_LEFT_RIGHT" if param == "LR" else "FLIP_TOP_BOTTOM"
        return textwrap.dedent(f"""\
            from PIL import Image
            img = Image.open({src_path!r})
            out = img.transpose(Image.{method})
            out.save({expected_path!r})
        """)
    if op == "rotate":
        return textwrap.dedent(f"""\
            from PIL import Image
            img = Image.open({src_path!r})
            out = img.rotate({int(param)}, expand=True)
            out.save({expected_path!r})
        """)
    if op == "mode":
        # PIL convert(): 'L'=Grayscale, 'P'=Indexed (palette). Save back as PNG.
        return textwrap.dedent(f"""\
            from PIL import Image
            img = Image.open({src_path!r})
            out = img.convert({param!r})
            out.save({expected_path!r})
        """)
    if op == "fill_color":
        # Replace non-black pixels with target color (preserves object outline).
        # Mirrors original 734d6579 oracle, generalized over color.
        rgb = _FILL_COLORS[param]
        return textwrap.dedent(f"""\
            from PIL import Image
            import numpy as np
            img = Image.open({src_path!r})
            px = np.array(img)
            mask = ~((px[..., 0] == 0) & (px[..., 1] == 0) & (px[..., 2] == 0))
            px[mask, 0] = {rgb[0]}
            px[mask, 1] = {rgb[1]}
            px[mask, 2] = {rgb[2]}
            Image.fromarray(px).save({expected_path!r})
        """)
    raise ValueError(f"image_op: unknown op {op!r}")


def _build_image_op_evaluator(eval_row: dict, expected_path: str, result_path: str) -> dict:
    """Replace eval's op-specific evaluator with check_structure_sim vs PIL gold.

    The result path is the perturb's OWN ``result_path`` (the file the oracle cp's to
    and the instruction names), NOT the eval base's evaluator result — those can differ
    (e.g. gimp 06ca5602's eval base now reads in-place computer.png, but the perturb
    writes a separate palette_computer.png to stay non-trivial under check_structure_sim).
    """
    new_ev = {
        "func": "check_structure_sim",
        "expected": {"type": "vm_file", "path": expected_path,
                     "dest": expected_path.rsplit("/", 1)[-1]},
        "result": {"type": "vm_file", "path": result_path,
                   "dest": result_path.rsplit("/", 1)[-1]},
    }
    def _retarget_export_filename(step: dict, filename_stem: str) -> None:
        params = step.get("parameters", {})
        cmd = params.get("command")

        def rewrite(text: str) -> str:
            return re.sub(
                r"pyautogui\.write\((?:'[^']*'|\"[^\"]*\")\)",
                f"pyautogui.write({filename_stem!r})",
                text,
            )

        if isinstance(cmd, list):
            params["command"] = [rewrite(part) if isinstance(part, str) else part for part in cmd]
        elif isinstance(cmd, str):
            params["command"] = rewrite(cmd)

    # Preserve postconfig (export-and-save GIMP UI sequence) if present.
    pc = eval_row["metadata"]["evaluator"].get("postconfig")
    if pc:
        new_ev["postconfig"] = copy.deepcopy(pc)
        result_stem = result_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        # validation finding: 6 image_op bases (7a4deb26, f723c744,
        # 554785e9, 72f83cdc, 06ca5602, 734d6579) inherit eval-base postconfigs
        # that do Shift+Ctrl+E → pyautogui.write("<basename>") → Enter. The
        # Name field is pre-filled with whatever was last there; write appends
        # → garbled filename → eval-target file never written. validation patched
        # this in misc_image_op for triangle_center; lift the same Ctrl+A +
        # Delete clear-field step to the image_op path so all 6 bases get the
        # field cleared before the write fires.
        new_pc = []
        clear_step = {"type": "execute", "parameters": {
            "command": [
                "python3", "-c",
                "import time; import pyautogui; "
                "pyautogui.hotkey(['ctrl', 'a']); time.sleep(0.3); "
                "pyautogui.press(['delete']); time.sleep(0.3)"
            ],
        }}
        inserted = False
        for step in new_ev["postconfig"]:
            cmd = step.get("parameters", {}).get("command")
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else (cmd or "")
            if not inserted and "pyautogui.write(" in cmd_str:
                new_pc.append(clear_step)
                inserted = True
            if "pyautogui.write(" in cmd_str:
                _retarget_export_filename(step, result_stem)
            new_pc.append(step)
        if inserted:
            new_ev["postconfig"] = new_pc
    return new_ev


def _build_image_op_row(
    *, eval_row: dict, perturb_type: str, op: str, param,
    src_path: str, result_path: str, expected_path: str,
    instruction: str,
) -> dict:
    """Assemble one perturb row for an image-op (TYPE_1 or TYPE_2)."""
    py_code = _build_pil_expected_py(op, param, src_path, expected_path)
    perturb_config_step = {"type": "execute", "parameters": {
        "command": f"python3 << 'PYEOF'\n{py_code}\nPYEOF", "shell": True,
    }}
    oracle = [
        {"type": "execute", "parameters": {"command": _GIMP_KILL_CMD, "shell": True}},
        {"type": "execute", "parameters": {"command": f"mkdir -p '{result_path.rsplit('/',1)[0]}'", "shell": True}},
        {"type": "execute", "parameters": {"command": f"cp '{expected_path}' '{result_path}'", "shell": True}},
    ]
    new_evaluator = _build_image_op_evaluator(eval_row, expected_path, result_path)
    return make_perturb_row(
        eval_row=eval_row,
        knob_assignment={"perturb_type": perturb_type, "op": op, "param": str(param)},
        new_instruction=instruction,
        new_oracle=oracle,
        new_evaluator=new_evaluator,
        perturb_config_step=perturb_config_step,
    )


def _format_op_instruction(op: str, param, out_filename: str, rng: random.Random) -> str:
    """Render an instruction string for (op, param). Includes target out filename."""
    if op in ("brightness", "contrast", "saturation"):
        pct = int(round(param * 100))
        instr = rng.choice(_OP_INSTR_TEMPLATES[op]).format(pct=pct, param=param, out=out_filename)
    elif op == "mirror":
        instr = rng.choice(_OP_INSTR_TEMPLATES[f"mirror_{param}"]).format(out=out_filename)
    elif op == "mode":
        instr = rng.choice(_OP_INSTR_TEMPLATES[op]).format(out=out_filename)
    elif op == "fill_color":
        instr = rng.choice(_OP_INSTR_TEMPLATES[op]).format(param=param, out=out_filename)
    elif op == "rotate":
        instr = rng.choice(_OP_INSTR_TEMPLATES[op]).format(param=int(param), out=out_filename)
    else:
        raise ValueError(f"Unknown op {op!r}")
    return instr


def perturb_gimp_image_op(eval_row: dict, rng: random.Random) -> list[dict]:
    """TYPE_1 + TYPE_2 perturb for image-op tasks. PIL gold + check_structure_sim."""
    short = eval_row["task_id"].split("_")[-1][:8]
    spec = _IMAGE_TASKS.get(short)
    if spec is None:
        return []

    src_path = spec["src"]
    result_path = spec["result"]

    rows: list[dict] = []

    # The agent must save the result at this exact path; instructions name the
    # filename explicitly so the agent can match the eval's result.path.
    result_filename = result_path.rsplit("/", 1)[-1]

    # ── TYPE_1: same op as eval, different param ────────────────────────────
    t1_op = spec["t1_op"]
    t1_pool = list(spec["t1_pool"])
    rng.shuffle(t1_pool)
    for param in t1_pool:
        instr = _format_op_instruction(t1_op, param, result_filename, rng)
        exp_path = f"/tmp/perturb_expected_{short}_t1.png"
        rows.append(_build_image_op_row(
            eval_row=eval_row, perturb_type="type1", op=t1_op, param=param,
            src_path=src_path, result_path=result_path,
            expected_path=exp_path, instruction=instr,
        ))

    # ── TYPE_2: different op (assigned by spec.feasible_t2_ops) ─────────────
    # For each TYPE_2 row we use a NEW op-appropriate output filename — the
    # base-task's result_path (e.g. "palette_computer.png" for 06ca5602) is
    # semantically tied to the base op and would confuse an agent told to do
    # a different op. Override result_path to match the new op's intent so
    # `evaluator.result.path` and the instruction's `{out}` agree.
    src_basename = src_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]   # e.g. "computer"
    out_dir = result_path.rsplit("/", 1)[0]                          # e.g. "/home/user/Desktop"
    _T2_FNAME_BY_OP = {
        "brightness": "brightened",
        "contrast":   "contrasted",
        "saturation": "saturated",
        "mirror":     "mirrored",
        "rotate":     "rotated",
    }

    t2_ops = list(spec["feasible_t2_ops"])
    rng.shuffle(t2_ops)
    # D3: TYPE_2 sample count cut from 2 → 1 per base to deflate
    # check_structure_sim total. Op coverage remains complete (every op still
    # appears across the 6 bases × 1 T2 = 6 distinct ops sampled).
    n_t2 = min(1, len(t2_ops))
    for op in t2_ops[:n_t2]:
        param_pool = _T2_POOLS.get(op, [])
        if not param_pool:
            continue
        param = rng.choice(param_pool)
        # Op-appropriate result path: "<verb>_<src_basename>.png"
        t2_filename = f"{_T2_FNAME_BY_OP[op]}_{src_basename}.png"
        t2_result_path = f"{out_dir}/{t2_filename}"
        instr = _format_op_instruction(op, param, t2_filename, rng)
        exp_path = f"/tmp/perturb_expected_{short}_t2_{op}.png"
        # Override evaluator.result.path so it points at the new filename.
        # _build_image_op_row deep-copies eval_row.evaluator.result, so we
        # must override AFTER the copy — patch the evaluator post-hoc.
        row = _build_image_op_row(
            eval_row=eval_row, perturb_type="type2", op=op, param=param,
            src_path=src_path, result_path=t2_result_path,
            expected_path=exp_path, instruction=instr,
        )
        row["metadata"]["evaluator"]["result"]["path"] = t2_result_path
        row["metadata"]["evaluator"]["result"]["dest"] = t2_filename
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# misc_image_op (P3-3 Phase 3-B): close 4 gimp eval gap bases that don't fit
# the existing `image_op` pattern.
#
# These bases use specialized evaluators (transparency, file_exists+SSIM,
# textbox_on_leftside, triangle_position) that constrain the *image content*
# rather than the operation type. Each archetype emits a small variant pool
# (3 rows / base) backed by a PIL-computed gold image.
#
# Pattern mirrors `image_op`:
#   perturb_config_step → PIL writes gold to /tmp/...
#   oracle              → killall gimp + mkdir + cp gold → result_path
#   evaluator           → eval's original func, but `expected` redirected to
#                         the local PIL gold (or, for `check_structure_sim`,
#                         to the gold; for params-only evals like
#                         `check_textbox_on_leftside` the expected is null
#                         and the gold satisfies the function intrinsically).
# ---------------------------------------------------------------------------

# Per-gap-base specs. Each gap base needs:
#   src   : source image present on the VM after eval config (download)
#   result: agent-target file path (matches eval evaluator.result.path)
#   variants: list of (op, param, instruction_template_idx) tuples
_MISC_IMAGE_TASKS: dict[str, dict] = {
    # 2a729ded: REMOVED during validation.
    # Step 0 ground truth: dog_with_background.png is mode=RGBA but mean_rgb
    # [135, 141, 145] — gray-blue scene, NOT a near-white background. Luminance
    # threshold 230 / 215 alpha-masks only 0.3% / 0.65% of pixels, so the PIL
    # gold is virtually identical to the source (SSIM ≈ 1.0 against the
    # original). At threshold 200, only 16% of pixels are masked (sky/highlights,
    # not "the background"). The instruction labels — "remove the white
    # background", "off-white background", "bright background" — are
    # semantically misaligned with the actual image content.
    # V2 still passes (oracle cp gold→result yields SSIM=1.0 trivially), but
    # during real-agent rollouts an agent that performs proper background
    # removal in GIMP produces a result with SSIM << 0.9 against the
    # near-original gold, which inverts the training signal. The eval base
    # itself is brittle for the same reason; perturbation cannot rescue it
    # without changing the evaluator definition. Mark infeasible by removing
    # from this dispatch table — perturb_gimp_misc_image returns [] for it.
    # 77b8ab4d: file_exists + structure_sim. Eval renames a photo to export.jpg.
    # We vary the export filename + a tiny SSIM-safe re-encode on the gold.
    "77b8ab4d": {
        "src": "/home/user/Desktop/The_Lost_River_Of_Dreams.jpg",
        "result": "/home/user/Desktop/export.jpg",
        "op_kind": "rename_export",
        "variants": [
            {"out_filename": "export.jpg",       "label": "export.jpg"},
            {"out_filename": "river_export.jpg", "label": "river_export.jpg"},
            {"out_filename": "photo_final.jpg",  "label": "photo_final.jpg"},
        ],
        "instruction_templates": [
            "Could you put my photo on the Desktop and save it as '{label}' for me?",
            "Please place the image on the Desktop and rename the export to '{label}'.",
            "Export this photo to the Desktop with the filename '{label}'.",
        ],
    },
    # NOTE: dropped "e2dd0213" (textbox_left) — eval threshold
    # (leftmost-dark-pixel within 5% of image width) is too tight to reproduce
    # under agent execution; PIL gold passes but actual agent output never
    # matches due to GIMP's anti-aliased text rendering.
    # f4aec372: triangle in middle. Eval base XCF has a triangle off to the
    # side. The evaluator picks the second-most-common color as the triangle,
    # computes its centroid, and passes if it's within 5% of image center.
    # We PIL-generate a gold with a centred triangle in 3 size variants.
    "f4aec372": {
        "src": "/home/user/Desktop/Triangle_On_The_Side.xcf",
        "result": "/home/user/Desktop/Triangle_In_The_Middle.png",
        "op_kind": "triangle_center",
        "variants": [
            {"size": 120, "color": (200, 80, 80),  "label": "centre the triangle on the canvas"},
            {"size": 160, "color": (80, 80, 200),  "label": "move the triangle to the middle of the image"},
            {"size": 100, "color": (80, 200, 80),  "label": "shift the triangle so it sits dead-centre"},
        ],
        "instruction_templates": [
            "Could you {label} and export as 'Triangle_In_The_Middle.png' on the Desktop?",
            "Please {label} so it's centred horizontally and vertically — save as 'Triangle_In_The_Middle.png' on the Desktop.",
            "{label_cap} and export the result as 'Triangle_In_The_Middle.png' on the Desktop.",
        ],
    },
}


def _build_misc_pil_py(op_kind: str, variant: dict, src_path: str, expected_path: str) -> str:
    """Generate Python heredoc body that writes a misc-archetype gold image."""
    if op_kind == "transparency":
        thresh = int(variant["thresh"])
        # Open source, build alpha mask where R+G+B mean > thresh → transparent.
        return textwrap.dedent(f"""\
            from PIL import Image
            import numpy as np
            img = Image.open({src_path!r}).convert('RGBA')
            arr = np.array(img)
            lum = arr[..., :3].mean(axis=2)
            mask_bg = lum > {thresh}
            arr[mask_bg, 3] = 0
            Image.fromarray(arr).save({expected_path!r})
        """)
    if op_kind == "rename_export":
        # Re-save source as JPEG (file_exists + structure_sim — gold matches
        # the eval's reference, just under the new filename).
        return textwrap.dedent(f"""\
            from PIL import Image
            img = Image.open({src_path!r}).convert('RGB')
            img.save({expected_path!r}, 'JPEG', quality=95)
        """)
    # NOTE: dropped textbox_left branch — only base e2dd0213
    # used it, and that base was dropped (eval threshold too tight; PIL gold
    # didn't reproducibly satisfy the leftmost-dark-pixel-within-5% rule).
    if op_kind == "triangle_center":
        size = int(variant["size"])
        r, g, b = variant["color"]
        # Validation fix: previous version used vertices (cx, cy-half),
        # (cx±half, cy+half) so the GEOMETRIC centroid sat at
        # (cx, cy + half/3) — for size=160 that's 26.7 px below center,
        # within 30 px tolerance but margin-thin and at risk of
        # rasterization jitter. Use a vertex layout whose centroid lands
        # exactly at (cx, cy): (cx, cy - 2*half/3), (cx - half, cy + half/3),
        # (cx + half, cy + half/3). y-mean = (-2h/3 + h/3 + h/3)/3 + cy = cy.
        return textwrap.dedent(f"""\
            from PIL import Image, ImageDraw
            W, H = 800, 600
            img = Image.new('RGB', (W, H), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            cx, cy = W // 2, H // 2
            half = {size} // 2
            # Isoceles triangle whose geometric centroid is at (cx, cy):
            # vertices (cx, cy - 2h/3), (cx - h, cy + h/3), (cx + h, cy + h/3).
            top_y    = cy - (2 * half) // 3
            bottom_y = cy + half // 3
            draw.polygon([
                (cx,        top_y),
                (cx - half, bottom_y),
                (cx + half, bottom_y),
            ], fill=({r}, {g}, {b}))
            img.save({expected_path!r})
        """)
    raise ValueError(f"misc_image_op: unknown op_kind {op_kind!r}")


def _build_misc_evaluator(eval_row: dict, expected_path: str, op_kind: str) -> dict:
    """Build evaluator for misc gap base — preserve eval func + redirect expected."""
    eval_func = eval_row["metadata"]["evaluator"].get("func", "")
    src_evaluator = eval_row["metadata"]["evaluator"]
    new_ev: dict = {
        "func": eval_func,
        "result": copy.deepcopy(src_evaluator["result"]),
    }
    if op_kind in ("transparency", "rename_export"):
        # SSIM-style evaluators read `expected`. Redirect to local PIL gold.
        new_ev["expected"] = {
            "type": "vm_file",
            "path": expected_path,
            "dest": expected_path.rsplit("/", 1)[-1],
        }
    else:
        # triangle_center: original evaluator takes only the result file
        # (no `expected`). Preserve the eval's expected field (typically
        # null) so loaders see the same shape.
        if "expected" in src_evaluator:
            new_ev["expected"] = copy.deepcopy(src_evaluator["expected"])
    pc = src_evaluator.get("postconfig")
    if pc:
        new_ev["postconfig"] = copy.deepcopy(pc)
        # Validation note: triangle_center base postconfig fires
        # Ctrl+Shift+E → pyautogui.write("Triangle_In_The_Middle") → Enter.
        # The Name field is pre-filled with the basename of whatever file is
        # active in GIMP (the agent's edited .xcf or scratch); pyautogui.write
        # appends to existing text → garbled "Triangle_On_The_SideTriangle_In_
        # The_Middle". Eval reads the exact `Triangle_In_The_Middle.png` path
        # which never gets written → all 3 variants FAIL. Insert a Ctrl+A +
        # Delete step AFTER Shift+Ctrl+E (which opens the dialog and focuses
        # the Name field) and BEFORE the existing write, so the field is empty
        # when pyautogui.write fires.
        if op_kind == "triangle_center" and new_ev["postconfig"]:
            pc_list = new_ev["postconfig"]
            # Find the first execute step whose command includes the write call
            # (heuristic: matches `pyautogui.write(`). Insert clear-field +
            # short sleep before it.
            clear_steps = [
                {"type": "execute", "parameters": {
                    "command": [
                        "python3", "-c",
                        "import time; import pyautogui; "
                        "pyautogui.hotkey(['ctrl', 'a']); time.sleep(0.3); "
                        "pyautogui.press(['delete']); time.sleep(0.3)"
                    ],
                }},
            ]
            for i, step in enumerate(pc_list):
                cmd = step.get("parameters", {}).get("command")
                cmd_str = " ".join(cmd) if isinstance(cmd, list) else (cmd or "")
                if "pyautogui.write(" in cmd_str:
                    new_ev["postconfig"] = pc_list[:i] + clear_steps + pc_list[i:]
                    break
    return new_ev


def perturb_gimp_misc_image(eval_row: dict, rng: random.Random) -> list[dict]:
    """P3-3: 3 gap-base archetypes — transparency / rename_export / triangle_center."""
    short = eval_row["task_id"].split("_")[-1][:8]
    spec = _MISC_IMAGE_TASKS.get(short)
    if spec is None:
        return []

    src_path = spec["src"]
    op_kind = spec["op_kind"]
    rows: list[dict] = []

    variants = list(spec["variants"])
    rng.shuffle(variants)
    template_pool = list(spec["instruction_templates"])
    rng.shuffle(template_pool)

    for idx, variant in enumerate(variants):
        # rename_export uses variant-specific result path; others use spec.result.
        if op_kind == "rename_export":
            out_filename = variant["out_filename"]
            result_path = f"/home/user/Desktop/{out_filename}"
        else:
            result_path = spec["result"]

        expected_path = f"/tmp/perturb_misc_{short}_v{idx}.png" if op_kind != "rename_export" \
            else f"/tmp/perturb_misc_{short}_v{idx}.jpg"

        py_code = _build_misc_pil_py(op_kind, variant, src_path, expected_path)
        perturb_config_step = {"type": "execute", "parameters": {
            "command": f"python3 << 'PYEOF'\n{py_code}\nPYEOF", "shell": True,
        }}
        oracle = [
            {"type": "execute", "parameters": {
                "command": _GIMP_KILL_CMD, "shell": True}},
            {"type": "execute", "parameters": {
                "command": f"mkdir -p '{result_path.rsplit('/', 1)[0]}'", "shell": True}},
            {"type": "execute", "parameters": {
                "command": f"cp '{expected_path}' '{result_path}'", "shell": True}},
        ]
        new_evaluator = _build_misc_evaluator(eval_row, expected_path, op_kind)
        # rename_export: result.path varies per variant.
        if op_kind == "rename_export":
            new_evaluator["result"]["path"] = result_path
            new_evaluator["result"]["dest"] = result_path.rsplit("/", 1)[-1]

        # Format instruction. `label_cap` lets imperative templates uppercase
        # the verb without losing the polite-templates' lower-case fragment.
        template = template_pool[idx % len(template_pool)]
        label = variant["label"]
        label_cap = label[:1].upper() + label[1:] if label else label
        instruction = template.format(label=label, label_cap=label_cap)

        knob = {"misc_op": op_kind, "variant_idx": idx}

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment=knob,
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=perturb_config_step,
        ))

    return rows


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

_INTERNAL_FNS = [
    perturb_gimp_config,
    perturb_gimp_image_resize,
    perturb_gimp_filter_action,
    perturb_gimp_image_op,
    perturb_gimp_misc_image,
]


def perturb_gimp_per_task(
    eval_row: dict,
    rng: random.Random,
    max_type1: int = 8,
) -> list[dict]:
    """Iterate internal op fns; each contributes up to max_type1 unique rows.

    Default cap=8 (vs other domains' 4) accommodates `perturb_gimp_image_op`
    which emits TYPE_1 (≤6) + TYPE_2 (≤2) = up to 8 variants per base task.
    Single-op fns (config / resize / filter) still cap themselves at 4 internally.
    """
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
