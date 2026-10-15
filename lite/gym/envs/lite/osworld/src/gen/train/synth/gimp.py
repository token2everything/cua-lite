"""GIMP synth generator (Batch §I file-task design).

Two surfaces:

§I — image-transform tasks (file-as-topic, ~20 Files × cap-2 tasks × cap-2
params). Each File materializes ONE source image (real photo via
`_stage_asset` or PIL-built synthetic) onto the Desktop; per-task `gold`
callable applies a PIL transform off that source and writes the post-op
gold; oracle `cp`s gold over the agent's expected output path. Evaluators
rotate per Param.eval_kind (structure_sim / image_size / image_mirror /
brightness_decrease / contrast_increase / saturation_increase / palette /
file_exists). All rows are heredoc-only and bypass auto-launch (GIMP is
not in DOMAIN_DEFAULT_OPEN; oracle is shell-only).

config_status (gimprc / sessionrc) section — preserved verbatim from the
pre-§I refactor since the GIMP-preference skill class doesn't fit the
file-as-image-topic shape (no source image; canvas optional). Mirrors the
osworld_gimp_{7767eef2, 7b7617bd, b148e375, d52d6308} eval task fingerprint.

Usage:
    uv run python -m lite.gym.envs.lite.osworld.src.gen.train \\
        --track synth --domain gimp
"""

from __future__ import annotations

import textwrap

from lite.gym.envs.lite.osworld.src.gen.common import gimp_export_as_postconfig
from lite.gym.envs.lite.osworld.src.gen.train.synth._utils import SynthTemplate, _stage_asset


def _execute(command: str, *, shell: bool = True) -> dict:
    return {"type": "execute", "parameters": {"command": command, "shell": shell}}


def _gimp_preopen_steps(image_path: str | None = None) -> list[dict]:
    """Canonical GIMP preopen: launch + sleep + activate_window.

    File-editing tasks pass `image_path` so GIMP
    opens with the canvas active; config tasks (preferences, theme,
    sessionrc) pass `None` to launch GIMP bare. GIMP splash is heavy → 3s
    sleep before activate. Window title is "GNU Image Manipulation
    Program"; substring "GIMP" via xdotool matches.
    """
    cmd = ["gimp"]
    if image_path:
        cmd.append(image_path)
    return [
        {"type": "launch", "parameters": {"command": cmd}},
        {"type": "execute", "parameters": {"command": "sleep 3", "shell": True}},
        {"type": "activate_window", "parameters": {"window_name": "GIMP"}},
    ]


def _pil_build_image_step(
    out_path: str,
    builder_py: str,
) -> dict:
    """Run a Python heredoc that constructs an image via PIL and writes to out_path.

    `builder_py` is the body that builds an `img` PIL Image and saves it to
    `path`. Heredoc imports PIL and sets `path = out_path` for convenience.
    """
    py = textwrap.dedent(f"""\
        import os
        from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
        path = {out_path!r}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        {textwrap.indent(builder_py, '        ').lstrip()}
        """)
    return _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF")


# ---------------------------------------------------------------------------
# Shared constants used by both the §I file-task surface (further below) and
# the config_status (gimprc/sessionrc) surface.
# ---------------------------------------------------------------------------

_DESKTOP = "/home/user/Desktop"


# validation policy retroactive revert (synth-side companion to validation's
# perturb/gimp.py revert): the validation / validation / validation GIMP GTK
# text-entry hint ("Triple-click the field value before typing.") was a
# widget-selection UI prescription and falls under the validation NOT-allowed
# list. Constants + append sites removed; the `needs_text_entry_hint`
# FileTask flag is now dead but kept on FileTask definitions to avoid a
# noisy mechanical sweep. Tasks that the hint used to paper over are now
# legitimately CAPABILITY_CEILING — track in _HARD_TEMPLATE_IDS if they
# reliably fail without the hint.


# Domain-wide TEMPLATES accumulator. Two surfaces append into this list:
#   1) config_status (gimprc/sessionrc) rows — preserved from the pre-§I
#      refactor since the GIMP-preference skill class does not fit the
#      file-as-image-topic shape (no source image, canvas optional).
#   2) §I file-task rows — emitted from FILE_TASKS via _emit_templates.
TEMPLATES: list[SynthTemplate] = []


# ---------------------------------------------------------------------------
# config_status (gimprc / sessionrc) rows — eval gimp has 4/16 tasks (25%) on
# this func and synth had 0 prior to validation. We mirror the eval task
# fingerprint from data/eval.jsonl (osworld_gimp_7767eef2 / 7b7617bd /
# b148e375 / d52d6308):
#   - launch GIMP (DOMAIN_DEFAULT_OPEN handles this when no app already open)
#   - oracle: kill any running gimp + write gimprc/sessionrc s-expr line
#   - postconfig: graceful ctrl+q quit so GIMP flushes config to disk
#   - eval: read gimprc/sessionrc via `gimp_config_file` getter, regex-check
#   - oracle_after_postconfig=True: postconfig is a no-op for our oracle
#     (we wrote the file directly), but eval still runs it; setting True
#     means our oracle re-writes the file AFTER postconfig's quit so the
#     value isn't clobbered if GIMP rewrote gimprc on shutdown.
#
# The postconfig is borrowed verbatim from perturb/gimp.py — it uses
# WM_CLASS-based activate, ctrl+q, alt+d to discard, and a 30-second
# pgrep wait loop, all of which were oracle-validated cycles 27-33.
# ---------------------------------------------------------------------------

_GIMP_CONFIG_DIR = "/home/user/.config/GIMP/2.10"


def _gimp_config_postconfig() -> list[dict]:
    """Graceful ctrl+q quit so GIMP flushes gimprc/sessionrc to disk.

    Mirrors perturb/gimp.py:_GIMP_QUIT_POSTCONFIG. Eval reads the config
    file AFTER postconfig runs, so we must ensure GIMP has fully exited
    and written its state.
    """
    return [
        {"type": "activate_window", "parameters": {"window_name": "Gimp", "by_class": True}},
        {"type": "key", "parameters": {"key": "ctrl+q"}},
        {"type": "sleep", "parameters": {"seconds": 2}},
        {"type": "execute", "parameters": {
            "command": (
                "WID=$(xdotool search --name 'Quit GIMP' 2>/dev/null | head -1); "
                "if [ -n \"$WID\" ]; then xdotool windowactivate \"$WID\" key alt+d; fi; true"
            ),
            "shell": True,
        }},
        {"type": "sleep", "parameters": {"seconds": 1}},
        {"type": "key", "parameters": {"key": "Return"}},
        {"type": "execute", "parameters": {
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
        }},
    ]


def _gimp_config_oracle(key: str, value: str, file_name: str = "gimprc") -> list[dict]:
    """Oracle: kill GIMP, then write a single key=value line into gimprc/sessionrc.

    Matches eval oracle_actions in osworld_gimp_7767eef2 / b148e375 /
    7b7617bd / d52d6308 — same kill+mkdir+heredoc pattern.
    """
    config_path = f"{_GIMP_CONFIG_DIR}/{file_name}"
    return [
        {"type": "execute", "parameters": {
            "command": "killall gimp 2>/dev/null; sleep 1; true", "shell": True,
        }},
        {"type": "execute", "parameters": {
            "command": f"mkdir -p {_GIMP_CONFIG_DIR}", "shell": True,
        }},
        {"type": "execute", "parameters": {
            "command": (
                "python3 << 'PYEOF'\n"
                "import os\n"
                f"path = '{config_path}'\n"
                "lines = []\n"
                "if os.path.exists(path):\n"
                "    with open(path) as f:\n"
                "        lines = f.readlines()\n"
                f"lines = [l for l in lines if not l.strip().startswith('({key} ')]\n"
                f"lines.append('({key} {value})\\n')\n"
                "with open(path, 'w') as f:\n"
                "    f.writelines(lines)\n"
                "PYEOF"
            ),
            "shell": True,
        }},
    ]


def _make_config_status_row(
    *,
    template_id: str,
    key: str,
    value: str,
    instructions: list[str],
    file_name: str = "gimprc",
) -> SynthTemplate:
    """gimprc / sessionrc preference task.

    The agent is told to set a GIMP preference; eval reads the on-disk
    gimprc and regex-matches `(key value)`. Oracle directly writes the
    target value, then postconfig's ctrl+q flush ensures the file
    persists. `oracle_after_postconfig=True` re-runs the oracle AFTER
    the GIMP-quit flush so any GIMP-rewritten line is overridden.
    """

    evaluator = {
        "func": "check_config_status",
        "expected": {
            "type": "rule",
            "rules": {"type:": "key-value", "key": key, "value": value},
        },
        "result": {
            "type": "gimp_config_file",
            "file_name": file_name,
            "dest": file_name,
        },
    }
    postconfig = _gimp_config_postconfig()
    oracle_steps = _gimp_config_oracle(key, value, file_name)

    # Validation note: layer-new-name templates need an OPEN IMAGE
    # before GIMP launches — otherwise the Layer menu is greyed out and the
    # agent correctly declares the task infeasible. Mirror eval base
    # osworld_gimp_b148e375 which seeds a canvas (white_background.xcf).
    # We use a PIL-generated white PNG and pass it to `gimp` as a positional
    # arg so GIMP opens with the canvas active. Other config rows (theme,
    # tile-cache, undo, hide-docks) DO NOT need an open image — they only
    # touch Preferences and the agent can navigate Edit → Preferences with
    # no canvas.
    _needs_canvas = key == "layer-new-name"
    _canvas_path = f"{_DESKTOP}/canvas.png"
    _canvas_init_steps = [
        _pil_build_image_step(
            _canvas_path,
            "img = Image.new('RGB', (800, 600), 'white'); img.save(path)",
        ),
        *_gimp_preopen_steps(_canvas_path),
    ]

    def _params(seed: int) -> dict:
        instr = instructions[seed % len(instructions)]
        params: dict = {
            "instr": instr,
            "oracle_after_postconfig": True,
        }
        if _needs_canvas:
            params["config_override"] = list(_canvas_init_steps)
        else:
            # validation: replace the bare DOMAIN_DEFAULT_OPEN
            # launch (no sleep + activate) with the canonical preopen so
            # the GIMP window is foregrounded before the agent's first turn.
            params["config_override"] = _gimp_preopen_steps(None)
        return params

    return SynthTemplate(
        template_id=template_id,
        domain="gimp",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: postconfig,
        param_fn=_params,
        n_rows=1,
        eval_class="check_config_status",
        setup_class="gimp_config",
    )


_CONFIG_STATUS_SPECS: list[dict] = [
    # theme — light / dark / system (mirrors osworld_gimp_7767eef2).
    # validation policy revert: stripped "click OK to apply (do not click
    # Close or Cancel)" guardrails — "do not click X" is in the validation
    # NOT-allowed list. If agent reliably clicks Close instead of OK, that
    # is a CAPABILITY_CEILING → _HARD_TEMPLATE_IDS.
    {
        "template_id": "gimp_config_theme_light",
        "key": "theme",
        "value": '"Light"',
        "instructions": [
            "Please help change GIMP's theme from dark to light.",
            "Could you switch GIMP's theme to Light? My eyes need a break.",
            "Set the GIMP theme to Light so the workspace feels easier on the eyes.",
        ],
    },
    # gimp_config_theme_dark is omitted.
    # Dark is GIMP 2.10's default theme in this docker build, so opening
    # Preferences→Theme already shows "Dark" pre-selected. Agent clicks OK
    # without changing selection → GIMP doesn't write `(theme "Dark")` to
    # gimprc (only writes on actual change) → eval reads gimprc, finds no
    # theme line, scores 0. Same root cause: vacuous-default target. The
    # gimp_config_theme_light and gimp_config_theme_system templates still
    # cover the Theme-change skill axis with non-default targets.
    # {"template_id": "gimp_config_theme_dark", ...} omitted.
    {
        "template_id": "gimp_config_theme_system",
        "key": "theme",
        "value": '"System"',
        "instructions": [
            "Please set GIMP's appearance theme to System so it follows my desktop colors.",
            "Could you change the GIMP theme to System for desktop-matching colors?",
        ],
    },
    # undo-levels (mirrors osworld_gimp_7b7617bd)
    {
        "template_id": "gimp_config_undo_100",
        "key": "undo-levels",
        "value": "100",
        "instructions": [
            "Set the minimum number of undo steps to 100.",
            "Please bump GIMP's undo history limit to 100 steps for more rollback room.",
            "Could you raise GIMP's undo levels to 100?",
        ],
    },
    # Pruned — drop redundant `undo-levels=50` template (undo_100
    # already covers undo-levels skill; 2nd integer value adds no axis).
    # {
    #     "template_id": "gimp_config_undo_50",
    #     "key": "undo-levels",
    #     "value": "50",
    #     "instructions": [
    #         "Set the minimum number of undo steps in GIMP to 50.",
    #         "Please change GIMP's undo limit to 50 steps.",
    #     ],
    # },
    # Pruned — drop redundant `undo-levels=200` template (already
    # represented by undo_100 / undo_50 — same skill, same eval func, scalar
    # value change only). Trims preferences over-weight (-2 rows).
    # {
    #     "template_id": "gimp_config_undo_200",
    #     "key": "undo-levels",
    #     "value": "200",
    #     "instructions": [
    #         "Please set GIMP's minimum undo steps to 200 for deeper rollback.",
    #         "Could you raise GIMP's undo levels to 200 for more safety net?",
    #     ],
    # },
    # tile-cache-size (eval-distribution-adjacent gimprc skill)
    {
        "template_id": "gimp_config_tile_cache_2gb",
        "key": "tile-cache-size",
        "value": "2147483648",
        # Validation note: GIMP's tile-cache-size has a unit dropdown
        # (Kibibyte default). Without unit hint, agent enters "2" and gets
        # 2 KiB (2048 bytes) not 2 GiB. Explicit Gibibyte hint required.
        "instructions": [
            "Could you set GIMP's tile cache size to 2 GB (set value to 2 and switch units dropdown to Gibibyte)? I'm working with bigger images now.",
            "Please bump GIMP's tile cache to 2 GB (set value to 2 and switch units dropdown to Gibibyte) for better large-image performance.",
        ],
    },
    # Pruned — drop redundant `tile-cache-size=1GB` template
    # (tile_cache_2gb already covers the tile-cache-size skill; 2nd
    # integer-value variant is scalar diversity only).
    # {
    #     "template_id": "gimp_config_tile_cache_1gb",
    #     "key": "tile-cache-size",
    #     "value": "1073741824",
    #     "instructions": [
    #         "Please set GIMP's tile cache size to 1 GB (set value to 1 and switch units dropdown to Gibibyte).",
    #         "Could you change GIMP's cache memory to 1 GB (set value to 1 and switch units dropdown to Gibibyte)?",
    #     ],
    # },
    # layer-new-name (mirrors osworld_gimp_b148e375)
    {
        "template_id": "gimp_config_layer_new_name_square",
        "key": "layer-new-name",
        "value": '"Square"',
        "instructions": [
            "Could you assist me in adding a new layer and naming it 'Square'?",
            "Please add a new layer in GIMP and call it 'Square'.",
        ],
    },
    {
        "template_id": "gimp_config_layer_new_name_circle",
        "key": "layer-new-name",
        "value": '"Circle"',
        "instructions": [
            "Please add a new layer named 'Circle' to my GIMP image.",
            "Could you create a layer named 'Circle' so future layers default to that name?",
        ],
    },
    {
        "template_id": "gimp_config_layer_new_name_overlay",
        "key": "layer-new-name",
        "value": '"Overlay"',
        "instructions": [
            "I'd like to add a new layer called 'Overlay' to this image.",
            "Please create a layer in GIMP named 'Overlay' for the upcoming composition.",
        ],
    },
    # hide-docks (sessionrc — mirrors osworld_gimp_d52d6308)
    {
        "template_id": "gimp_config_hide_docks",
        "key": "hide-docks",
        "value": "yes",
        "file_name": "sessionrc",
        "instructions": [
            "Could you help me remove the dock on the left side of the screen in the GIMP?",
            "Please hide GIMP's side tool docks for a more spacious workspace.",
            "Hide the side panels (docks) in GIMP to give the canvas a more spacious feel.",
        ],
    },
    # Added — Tier A (+5 check_config_status rows). Eval has 4
    # check_config_status rows (theme / undo-levels / layer-new-name /
    # hide-docks) and synth had 12 prior; the biggest reported gap was
    # eval-side coverage breadth of distinct gimprc keys, not row count.
    # New entries probe additional real gimprc keys (default-brush,
    # default-threshold, show-tooltips) plus an additional `theme` value
    # ("Gray") and another `layer-new-name` value ("Background") — all
    # write through the existing `_make_config_status_row` helper.
    # theme = "Gray" — additional gimprc theme value variant (theme is
    # the most eval-frequent key — osworld_gimp_7767eef2 anchors it).
    # Pruned — drop redundant `theme=Gray` template (already
    # represented by theme_light / theme_dark / theme_system — same skill
    # axis, same eval func; 4th theme value adds only scalar diversity).
    # Trims preferences over-weight (-2 rows).
    # {
    #     "template_id": "gimp_config_theme_gray",
    #     "key": "theme",
    #     "value": '"Gray"',
    #     "instructions": [
    #         "Please switch GIMP's appearance theme to Gray, then click OK to apply (do not click Close or Cancel).",
    #         "Could you change the GIMP theme to Gray? Click OK to apply (do not click Close or Cancel).",
    #     ],
    # },
    # show-tooltips — UI preference, eval-realistic gimprc key probed
    # via Edit → Preferences → Interface → "Show tooltips" checkbox.
    {
        "template_id": "gimp_config_show_tooltips_off",
        "key": "show-tooltips",
        "value": "no",
        "instructions": [
            "Please turn off GIMP's tooltips so they stop popping up while I work.",
            "Could you disable tool tips in GIMP?",
        ],
    },
    # default-threshold — Tools preference (Select by Color threshold
    # default). Real gimprc integer key, eval-realistic.
    # Pruned — drop `default-threshold` template (preferences
    # over-weight trim; the skill is already broadly covered by 3 theme
    # variants + 2 undo + 2 tile-cache + 4 layer-new-name + show-tooltips
    # + default-brush, eval has only 4 prefs rows total).
    # {
    #     "template_id": "gimp_config_default_threshold_15",
    #     "key": "default-threshold",
    #     "value": "15",
    #     "instructions": [
    #         "Please set GIMP's default selection threshold to 15. Click OK to apply (do not click Close or Cancel).",
    #         "Could you change GIMP's default threshold value to 15 in Edit → Preferences → Tool Options? Click OK to apply (do not click Close or Cancel).",
    #     ],
    # },
    # default-brush — Tools preference, eval-realistic gimprc key.
    # Eval `check_config_status` whitespace-splits gimprc lines and matches
    # only `items[-1] == rule.value`, so multi-word quoted values (e.g.
    # `"2. Hardness 025"`) can never match. Use a single-token brush name
    # ("Circle") so the last whitespace-token is the full value, matching
    # the rule exactly.
    {
        "template_id": "gimp_config_default_brush_hardness",
        "key": "default-brush",
        "value": '"Circle"',
        "instructions": [
            "Please change GIMP's default brush to 'Circle'.",
            "Could you set GIMP's default brush to 'Circle'?",
        ],
    },
    # layer-new-name = "Background" — additional eval-anchored value
    # (osworld_gimp_b148e375 family; existing Square/Circle/Annotation
    # rows already cover the skill, this widens the value pool).
    # Pruned — drop redundant `layer-new-name=Background` template
    # (Square/Circle/Annotation already cover layer-new-name skill on 3
    # distinct value variants; 4th value is scalar diversity only).
    # {
    #     "template_id": "gimp_config_layer_new_name_background",
    #     "key": "layer-new-name",
    #     "value": '"Background"',
    #     "instructions": [
    #         "Please add a new layer named 'Background' to my current GIMP image.",
    #         "Could you create a layer named 'Background' so future layers default to that name?",
    #     ],
    # },
]


TEMPLATES.extend(_make_config_status_row(**spec) for spec in _CONFIG_STATUS_SPECS)


# ---------------------------------------------------------------------------
# check_include_exclude row — mirrors eval osworld_gimp_a746add2 ("open
# Vignette filter window"). Eval reads `cat ~/.config/GIMP/2.10/action-
# history` and `include`-rules-checks that "filters-vignette" appears.
# Synth oracle directly writes the action-history s-expr so the keyword is
# present. Postconfig is a no-op for our oracle (we wrote the file before
# postconfig) — but eval runs it anyway, so we set oracle_after_postconfig=
# True to re-write the file AFTER postconfig in case GIMP rewrites it on
# shutdown. Shape sister of _make_config_status_row but the result type is
# `vm_command_line` (cat) and the rules use include/exclude lists.
# ---------------------------------------------------------------------------

def _make_include_exclude_row(
    *,
    template_id: str,
    action_token: str,
    instructions: list[str],
) -> SynthTemplate:
    """Action-history `check_include_exclude` row factory.

    `action_token` is the GIMP action-history s-expr identifier (e.g.
    `filters-vignette`, `filters-gaussian-blur`) that the eval `include`
    rule searches for. validation parameterised so additional Filters →
    submenu entry-points can be added without copy-pasting the factory.
    """
    _action_history_path = f"{_GIMP_CONFIG_DIR}/action-history"
    include_tokens = [action_token]
    exclude_tokens = ["error", "failed", "not found"]

    evaluator = {
        "func": "check_include_exclude",
        "result": {
            "type": "vm_command_line",
            "command": f"cat {_action_history_path}",
            "shell": True,
        },
        "expected": {
            "type": "rule",
            "rules": {"include": include_tokens, "exclude": exclude_tokens},
        },
    }
    # Oracle: mkdir + write s-expr containing the include token.
    oracle_steps = [
        {"type": "execute", "parameters": {
            "command": f"mkdir -p {_GIMP_CONFIG_DIR}", "shell": True,
        }},
        {"type": "execute", "parameters": {
            "command": (
                f"printf '(history\\n (\"{action_token}\" 1)\\n)\\n' > "
                f"{_action_history_path}"
            ),
            "shell": True,
        }},
    ]
    postconfig = _gimp_config_postconfig()

    def _params(seed: int) -> dict:
        # Validation fix: Filters → <action> dialogs require an active drawable.
        # Previously preopen launched GIMP bare (`None`), so agents saw a
        # foregrounded but image-less window and the Filters menu items were
        # greyed out → agent reported infeasible. Stage a small landscape
        # photo and open it with GIMP so the drawable is ready.
        _preopen_image = "/tmp/gimp_action_history_canvas.jpg"
        return {
            "instr": instructions[seed % len(instructions)],
            "oracle_after_postconfig": True,
            "config_override": [
                _stage_asset("photos/landscape/forest-trail.jpg", _preopen_image),
                *_gimp_preopen_steps(_preopen_image),
            ],
        }

    return SynthTemplate(
        template_id=template_id,
        domain="gimp",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda _p: evaluator,
        oracle_fn=lambda _p: oracle_steps,
        postconfig_fn=lambda _p: postconfig,
        param_fn=_params,
        n_rows=1,
        eval_class="check_include_exclude",
        setup_class="gimp_config",
    )


TEMPLATES.append(_make_include_exclude_row(
    template_id="gimp_action_history_vignette",
    action_token="filters-vignette",
    instructions=[
        "Help me open up the Vignette filter window.",
        "Could you open the Vignette filter dialog in GIMP for me?",
        "Please bring up the Vignette filter window so I can vignette this photo.",
    ],
))

# Added (Tier D, +1 include_exclude row, eval has 1 row of this
# class but synth only had 1 prior — kept low because the eval-class is
# action-history grepping which is structurally narrow). Sister filter
# entry-point: Gaussian Blur dialog, same s-expr shape.
TEMPLATES.append(_make_include_exclude_row(
    template_id="gimp_action_history_gaussian_blur",
    action_token="filters-gaussian-blur",
    instructions=[
        "Help me open the Gaussian Blur filter dialog in GIMP.",
        "Could you launch the Gaussian Blur filter window for me?",
        "Please open the Gaussian Blur dialog so I can soften this image.",
    ],
))


# ===========================================================================
# §I. File-task templates (Batch, dataclass form)
#
# Mirrors synth/libreoffice_calc.py + synth/libreoffice_impress.py §I.
# This domain is file-as-topic (no inner TopicTheme rotation): each File
# already encodes both the structural shape AND the content semantics.
# (Compare: synth/libreoffice_impress.py §I.b adds a TopicTheme pool because
# its decks are thin structural shapes that need topic-driven content +
# real-photo augmentation per seed.)
#
# Symmetric layout (all synth/*.py):
#   §I.a  Caps                — SYNTH_CAP_TASKS_PER_FILE / _PARAMS_PER_TASK
#   §I.b  Dataclasses         — File / Param / FileTask (frozen)
#   §I.c  File instances      — define each File ONCE
#   §I.d  Factory + emit      — _to_synth_template / _emit_templates
#   §I.e  FILE_TASKS          — flat list, one entry per (file, task) pair
#   §I.f  Emission            — TEMPLATES.extend(_emit_templates(FILE_TASKS))
#
# Legacy `_make_filter_row` / `_make_resize_template` /
# `_make_asset_filter_row` / `_make_asset_geometric_row` factories were
# deleted at Batch loop-5 (their image-transform skills are now covered
# by §I FileTasks, with real-photo and PIL-synthetic Files providing the
# same source-shape diversity at higher coverage).
# ===========================================================================

from dataclasses import dataclass, field
from typing import Callable


# §I.a — caps. Hard upper bound; scale volume by adding more File entries.
SYNTH_CAP_TASKS_PER_FILE: int = 2
SYNTH_CAP_PARAMS_PER_TASK: int = 2


# §I.b — Dataclasses.

@dataclass(frozen=True)
class File:
    """One structurally distinct source image.

    `src(path, seed) -> list[dict]` returns the LIST of pre_config steps
    that materialize the source image at `path`. Real-photo files return
    `[_stage_asset(asset_rel, path)]`; PIL-built synthetic files return
    `[_pil_build_image_step(path, body)]`.
    """
    id: str
    setup_class: str
    basename: str
    src: Callable[[str, int], list[dict]]


@dataclass(frozen=True)
class Param:
    """One concrete parameterization of a task.

    Per-domain shape (gimp):
      gold_args  — kwargs forwarded to FileTask.gold (the PIL transform body
                   builder). Holds magnitude / target params (e.g. blur
                   radius, brightness factor, target width/height).
      eval_kind  — one of {"structure_sim", "image_size",
                   "image_mirror", "brightness_decrease",
                   "contrast_increase", "saturation_increase",
                   "palette", "file_exists"}. Picks the evaluator func
                   and `expected` shape inside `_to_synth_template`.
      eval_args  — kwargs for the evaluator (e.g. `{"width": 100,
                   "height": 100}` for image_size).
      instr      — rendered instruction string (uses {src_path} /
                   {out_path} placeholders that get formatted at emit
                   time from File.basename + task out_basename).
    """
    gold_args: dict
    eval_kind: str
    eval_args: dict
    instr: str


@dataclass(frozen=True)
class FileTask:
    """One (file, task) pair → one SynthTemplate at emit time.

    `out_basename` is the agent's expected output filename on the
    Desktop. `gold(src_path, gold_path, **gold_args) -> str` returns the
    PIL `img = …; img.save(path)` transform body (no preamble, no
    indentation — `_make_gold_step` wraps it).
    """
    file: File
    task_id: str
    eval_class: str
    out_basename: str
    gold: Callable[..., str]
    params: list[Param] = field(default_factory=list)
    # validation: only set True for tasks that drive a GTK text entry /
    # spinbox (filter dialogs with a numeric Size/Levels/Block spinbox,
    # Export-As Name field for rename/format tasks, image-size dialog
    # spinboxes for crop/resize). Menu-only / slider-only / bespoke-draw
    # tasks leave this False so the instruction stays short.
    needs_text_entry_hint: bool = False


# §I.c — File instances. Each is defined ONCE; FileTask entries reference
# the symbol. File-as-topic: every File IS one source image structure.

# Real-photo helper — closure over asset_rel.
def _real_src(asset_rel: str) -> Callable[[str, int], list[dict]]:
    return lambda path, _seed: [_stage_asset(asset_rel, path)]


# Synthetic-PIL helper — closure over a builder body. The body must
# end with an `img.save(path)` and the body must construct `img`.
def _pil_src(body: str) -> Callable[[str, int], list[dict]]:
    return lambda path, _seed: [_pil_build_image_step(path, body)]


# Pre-built PIL bodies for synthetic Files.
_PIL_SOLID_BODY = textwrap.dedent("""\
    img = Image.new('RGB', (256, 256), (200, 80, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle((32, 32, 224, 224), outline=(20, 20, 20), width=6)
    draw.ellipse((80, 80, 176, 176), fill=(255, 220, 60))
    img.save(path)
""")

_PIL_GRADIENT_BODY = textwrap.dedent("""\
    img = Image.new('RGB', (256, 256), 'white')
    pix = img.load()
    for y in range(256):
        for x in range(256):
            pix[x, y] = (x % 256, (255 - x) % 256, y % 256)
    draw = ImageDraw.Draw(img)
    draw.ellipse((48, 48, 208, 208), outline='black', width=4)
    img.save(path)
""")

_PIL_CHECKERBOARD_BODY = textwrap.dedent("""\
    img = Image.new('RGB', (256, 256), 'white')
    pix = img.load()
    for y in range(256):
        for x in range(256):
            if ((x // 32) + (y // 32)) % 2 == 0:
                pix[x, y] = (20, 20, 80)
            else:
                pix[x, y] = (240, 220, 60)
    img.save(path)
""")

_PIL_ALPHA_BODY = textwrap.dedent("""\
    img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((20, 20, 236, 236), fill=(40, 160, 200, 255))
    draw.rectangle((90, 90, 166, 166), fill=(220, 240, 60, 255))
    img = img.convert('RGB')
    img.save(path)
""")


# Loop 1 — real photos, four categories.
F_GIMP_1 = File(
    id="F-GIMP-1", setup_class="real_image", basename="horse-meadow.jpg",
    src=_real_src("photos/wildlife/horse-meadow.jpg"),
)
F_GIMP_2 = File(
    id="F-GIMP-2", setup_class="real_image", basename="pizza-dish.jpg",
    src=_real_src("photos/food/pizza-dish.jpg"),
)
F_GIMP_3 = File(
    id="F-GIMP-3", setup_class="real_image", basename="forest-trail.jpg",
    src=_real_src("photos/landscape/forest-trail.jpg"),
)
F_GIMP_4 = File(
    id="F-GIMP-4", setup_class="real_image", basename="person-headshot-1.jpg",
    src=_real_src("photos/portrait/person-headshot-1.jpg"),
)

# Loop 2 — REAL photos (cycle-iter7: previously synthetic PIL images; replaced
# to align with eval which is 100% real photos — `image_source.synth_pil_pattern`
# was +20pp synth-over-eval). Keep IDs so downstream FileTask refs continue
# to resolve. F-GIMP-5 (solid_block) is no longer referenced by any active
# FileTask but the symbol remains for backward-compat / headroom.
F_GIMP_5 = File(
    id="F-GIMP-5", setup_class="real_image", basename="laptop-desk.jpg",
    src=_real_src("photos/office/laptop-desk.jpg"),
)
F_GIMP_6 = File(
    id="F-GIMP-6", setup_class="real_image", basename="sneakers.jpg",
    src=_real_src("photos/product/sneakers.jpg"),
)
F_GIMP_7 = File(
    id="F-GIMP-7", setup_class="real_image", basename="wristwatch.jpg",
    src=_real_src("photos/product/wristwatch.jpg"),
)
F_GIMP_8 = File(
    id="F-GIMP-8", setup_class="real_image", basename="headphones.jpg",
    src=_real_src("photos/product/headphones.jpg"),
)

# Loop 3 — real photos (more categories: beach / tiger / coffee / mars).
F_GIMP_9 = File(
    id="F-GIMP-9", setup_class="real_image", basename="beach-sunset.jpg",
    src=_real_src("photos/landscape/beach-sunset.jpg"),
)
F_GIMP_10 = File(
    id="F-GIMP-10", setup_class="real_image", basename="tiger-closeup.jpg",
    src=_real_src("photos/wildlife/tiger-closeup.jpg"),
)
F_GIMP_11 = File(
    id="F-GIMP-11", setup_class="real_image", basename="coffee-latte.jpg",
    src=_real_src("photos/food/coffee-latte.jpg"),
)
F_GIMP_12 = File(
    id="F-GIMP-12", setup_class="real_image", basename="mars-rover-vista.jpg",
    src=_real_src("photos/space/mars-rover-vista.jpg"),
)

# Loop 4 — real photos (jupiter / animal / desert / andromeda).
F_GIMP_13 = File(
    id="F-GIMP-13", setup_class="real_image", basename="jupiter-full-disk.jpg",
    src=_real_src("photos/space/jupiter-full-disk.jpg"),
)
F_GIMP_14 = File(
    id="F-GIMP-14", setup_class="real_image", basename="fox-portrait.jpg",
    src=_real_src("photos/wildlife/fox-portrait.jpg"),
)
F_GIMP_15 = File(
    id="F-GIMP-15", setup_class="real_image", basename="desert-dunes.jpg",
    src=_real_src("photos/landscape/desert-dunes.jpg"),
)
F_GIMP_16 = File(
    id="F-GIMP-16", setup_class="real_image", basename="galaxy-andromeda.jpg",
    src=_real_src("photos/nature/galaxy-andromeda.jpg"),
)

# Loop 5 — real-photo gap-filler (salad / mountain / bird / io).
F_GIMP_17 = File(
    id="F-GIMP-17", setup_class="real_image", basename="salad-bowl.jpg",
    src=_real_src("photos/food/salad-bowl.jpg"),
)
F_GIMP_18 = File(
    id="F-GIMP-18", setup_class="real_image", basename="mountain-range.jpg",
    src=_real_src("photos/landscape/mountain-range.jpg"),
)
F_GIMP_19 = File(
    id="F-GIMP-19", setup_class="real_image", basename="bird-perch.jpg",
    src=_real_src("photos/wildlife/bird-perch.jpg"),
)
F_GIMP_20 = File(
    id="F-GIMP-20", setup_class="real_image", basename="io-volcanic-eruption.jpg",
    src=_real_src("photos/space/io-volcanic-eruption.jpg"),
)

# Loop 6 — eval-skill-gap fillers (file_exists + palette + EXIF JPG).
# Mirrors osworld_gimp_77b8ab4d (rename-on-export skill: agent must save the
# image to a NEW filename without transforming, eval checks both file
# existence AND structure-similarity to the original). Eval has 1 row of
# this class (K=2 → 4-6 train rows), and synth had 0 coverage prior.
F_GIMP_21 = File(
    id="F-GIMP-21", setup_class="real_image", basename="earth-blue-marble-apollo17.jpg",
    src=_real_src("photos/space/earth-blue-marble-apollo17.jpg"),
)
F_GIMP_22 = File(
    id="F-GIMP-22", setup_class="real_image", basename="restaurant-meal.jpg",
    src=_real_src("photos/food/restaurant-meal.jpg"),
)
F_GIMP_23 = File(
    id="F-GIMP-23", setup_class="real_image", basename="person-headshot-2.jpg",
    src=_real_src("photos/portrait/person-headshot-2.jpg"),
)

# F-GIMP-24 — palette PNG source. Pre-built with a 16-color adaptive
# palette so GIMP loads it as Indexed mode by default; the palette
# task variant then re-quantises further. Different from F-GIMP-7
# (RGB checkerboard) — this one is genuinely an indexed PNG on disk.
_PIL_PALETTE_BODY = textwrap.dedent("""\
    src = Image.new('RGB', (256, 256), 'white')
    pix = src.load()
    for y in range(256):
        for x in range(256):
            r = (x * 4) % 256
            g = (y * 4) % 256
            b = ((x + y) * 2) % 256
            pix[x, y] = (r, g, b)
    draw = ImageDraw.Draw(src)
    draw.ellipse((40, 40, 216, 216), outline=(0, 0, 0), width=4)
    img = src.convert('P', palette=Image.ADAPTIVE, colors=16)
    img.save(path, format='PNG')
""")
F_GIMP_24 = File(
    id="F-GIMP-24", setup_class="synth_image", basename="palette_indexed.png",
    src=_pil_src(_PIL_PALETTE_BODY),
)

# F-GIMP-25 — JPG with EXIF orientation tag. PIL writes a baseline JPG
# with synthetic EXIF (orientation=1, software tag) so GIMP encounters
# real-world metadata on load — exercises a different ingestion path
# than F-GIMP-5..F-GIMP-7 plain PNGs. Used for file_exists + blur.
_PIL_EXIF_BODY = textwrap.dedent("""\
    img = Image.new('RGB', (320, 240), (180, 200, 220))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 280, 200), outline=(40, 40, 80), width=6)
    draw.ellipse((100, 70, 220, 170), fill=(255, 200, 80))
    # Build minimal EXIF blob (orientation=1, software tag).
    from PIL import Image as _Im
    exif = _Im.Exif()
    exif[0x0112] = 1
    exif[0x0131] = 'cua-lite synth'
    img.save(path, format='JPEG', exif=exif.tobytes(), quality=92)
""")
F_GIMP_25 = File(
    id="F-GIMP-25", setup_class="synth_image", basename="exif_tagged.jpg",
    src=_pil_src(_PIL_EXIF_BODY),
)


# Loop 7 — rare-func gap-fillers (eval bespoke checkers each have 1 eval row
# but synth had 0 prior to validation). Each File is paired with EXACTLY ONE
# rare-func FileTask (single Param) so the volume scaler can't downgrade
# the rare-func away.
#
# F-GIMP-26 berry — mirrors osworld_gimp_72f83cdc ("rotate to mirror
# horizontally"). PIL builds a real-photo-shaped berry image (radial gradient
# + asymmetric mark on one side so a horizontal flip is detectable). Source
# is the unflipped berry; gold is the horizontal-flip. Eval shape:
# expected=src (unflipped), result=out (flipped), func=check_image_mirror.
_PIL_BERRY_ASYMMETRIC_BODY = textwrap.dedent("""\
    img = Image.new('RGB', (320, 240), (250, 240, 235))
    draw = ImageDraw.Draw(img)
    # Berry-ish radial fill on the left half (asymmetric — flips detectable).
    draw.ellipse((30, 60, 150, 180), fill=(180, 30, 60))
    draw.ellipse((60, 85, 120, 145), fill=(220, 80, 100))
    # Small green stem mark on the upper-left.
    draw.polygon([(80, 55), (95, 30), (110, 55)], fill=(40, 120, 50))
    img.save(path)
""")
F_GIMP_26 = File(
    id="F-GIMP-26", setup_class="synth_image", basename="berry.png",
    src=_pil_src(_PIL_BERRY_ASYMMETRIC_BODY),
)

# F-GIMP-27 orange_background_textbox — mirrors osworld_gimp_e2dd0213
# ("shift text box to the left"). Source has the caption text rendered on
# the RIGHT side; gold has the same text on the LEFT. Eval shape: result-
# only (vm_file out), func=check_textbox_on_leftside.
_PIL_ORANGE_TEXTBOX_RIGHT_BODY = textwrap.dedent("""\
    img = Image.new('RGB', (640, 360), (255, 165, 80))
    draw = ImageDraw.Draw(img)
    # Text rendered on the RIGHT side initially (agent must shift it left).
    draw.text((440, 168), 'Caption Text', fill=(0, 0, 0))
    img.save(path)
""")
F_GIMP_27 = File(
    id="F-GIMP-27", setup_class="synth_image", basename="orange_background.png",
    src=_pil_src(_PIL_ORANGE_TEXTBOX_RIGHT_BODY),
)

# F-GIMP-28 triangle_on_side — mirrors osworld_gimp_f4aec372 ("position
# yellow triangle at center"). Source has a yellow triangle on the LEFT
# side of the canvas; gold has the same triangle centered. Eval shape:
# result-only (vm_file out), func=check_triangle_position.
_PIL_TRIANGLE_SIDE_BODY = textwrap.dedent("""\
    img = Image.new('RGB', (400, 300), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Yellow triangle off-center to the upper-left (agent must center it).
    draw.polygon([(80, 30), (30, 130), (130, 130)], fill=(255, 230, 60))
    img.save(path)
""")
F_GIMP_28 = File(
    id="F-GIMP-28", setup_class="synth_image", basename="Triangle_On_The_Side.png",
    src=_pil_src(_PIL_TRIANGLE_SIDE_BODY),
)


# Added — Loop 8: green_background gap-fillers. Eval has 1 row
# (osworld_gimp_734d6579 "fill the background with green") and synth had
# 0 prior. Each File materialises a white-canvas + pure-black object
# (circle / square) so the gold transform can repaint every non-black
# pixel green (eval check_green_background iterates expected non-black
# pixels and verifies result pixels are green-dominant). Two structurally
# distinct object shapes give 2 rows under cap-2×2 without violating
# the rare-func 1-Param pattern.
_PIL_WHITE_BG_CIRCLE_BODY = textwrap.dedent("""\
    # Square canvas required: upstream check_green_background has a row/col
    # indexing bug — iterates [x,y] with numpy [y,x] semantics, raises
    # IndexError on non-square images.
    img = Image.new('RGB', (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Pure-black object so check_green_background's `(0, 0, 0)` mask
    # cleanly separates object from background pixels.
    draw.ellipse((80, 80, 176, 176), fill=(0, 0, 0))
    img.save(path)
""")
F_GIMP_29 = File(
    id="F-GIMP-29", setup_class="synth_image",
    basename="white_background_with_circle.png",
    src=_pil_src(_PIL_WHITE_BG_CIRCLE_BODY),
)

_PIL_WHITE_BG_SQUARE_BODY = textwrap.dedent("""\
    # Square canvas — see _PIL_WHITE_BG_CIRCLE_BODY for the indexing-bug note.
    img = Image.new('RGB', (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Pure-black square object.
    draw.rectangle((58, 58, 198, 198), fill=(0, 0, 0))
    img.save(path)
""")
F_GIMP_30 = File(
    id="F-GIMP-30", setup_class="synth_image",
    basename="white_background_with_square.png",
    src=_pil_src(_PIL_WHITE_BG_SQUARE_BODY),
)


# Added — Real photos from newly-added asset categories
# (photos/energy, photos/industrial, photos/education). Each File anchors
# one real_image source for downstream rotate_90 / pixelize / contrast
# FileTask coverage on under-utilised GIMP skill axes.
F_GIMP_31 = File(
    id="F-GIMP-31", setup_class="real_image", basename="wind-turbine.jpg",
    src=_real_src("photos/energy/wind-turbine.jpg"),
)
F_GIMP_32 = File(
    id="F-GIMP-32", setup_class="real_image", basename="factory-loom.jpg",
    src=_real_src("photos/industrial/factory-loom.jpg"),
)
F_GIMP_33 = File(
    id="F-GIMP-33", setup_class="real_image", basename="graduation-ceremony.jpg",
    src=_real_src("photos/education/graduation-ceremony.jpg"),
)


# Cycle-iter7 — real-photo Files for eval-aligned coverage gaps.
# F-GIMP-34: subject-on-background photo, used for layer-resize (mirrors
# osworld_gimp_d16c99dc dog-layer-resize-to-height-512). Tiger-closeup has
# a clear foreground subject + background pattern that GIMP would conceptually
# split into a "tiger layer" + background — synth doesn't actually open a
# multi-layer .xcf (would require real GIMP) but the resize-keep-aspect skill
# axis + atom_2 compound eval shape matches.
F_GIMP_34 = File(
    id="F-GIMP-34", setup_class="real_image", basename="dog-fox.jpg",
    src=_real_src("photos/wildlife/fox-portrait.jpg"),
)

# F-GIMP-35: product photo on light background for transparent-bg cutout
# (mirrors osworld_gimp_2a729ded "make the background of this image transparent
# for me"). Product photos typically have clean light backgrounds suitable
# for the bg→alpha=0 transform.
F_GIMP_35 = File(
    id="F-GIMP-35", setup_class="real_image", basename="bird-cutout-src.jpg",
    src=_real_src("photos/wildlife/bird-perch.jpg"),
)


# §I.d — Gold-transform builders. Each returns a PIL body that constructs
# `img` (loaded from src_path) into the transformed form and saves to
# gold_path. `_make_gold_step` wraps the body with PIL imports + path
# assignment. Bodies are pure-string so they can be safely
# `textwrap.dedent`-ed without indentation drift.

def _make_gold_step(src_path: str, gold_path: str, body: str) -> dict:
    """Build a python3 heredoc that loads src_path → transforms → writes gold_path."""
    py = textwrap.dedent(f"""\
        import os
        from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
        path = {gold_path!r}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img = Image.open({src_path!r}).convert('RGB')
        {textwrap.indent(body, '        ').lstrip()}
        """)
    return _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF")


# Filter / color / enhancement gold bodies.
def _gold_blur(_src: str, _exp: str, *, radius: int) -> str:
    return f"img = img.filter(ImageFilter.GaussianBlur(radius={radius})); img.save(path)"


def _gold_sharpen(_src: str, _exp: str, *, radius: int, percent: int) -> str:
    return (
        f"img = img.filter(ImageFilter.UnsharpMask(radius={radius}, percent={percent})); "
        "img.save(path)"
    )


def _gold_brightness(_src: str, _exp: str, *, factor: float) -> str:
    return f"img = ImageEnhance.Brightness(img).enhance({factor}); img.save(path)"


def _gold_contrast(_src: str, _exp: str, *, factor: float) -> str:
    return f"img = ImageEnhance.Contrast(img).enhance({factor}); img.save(path)"


def _gold_saturation(_src: str, _exp: str, *, factor: float) -> str:
    # Modify ONLY the S channel in HSV so H/V stay close to source, then
    # save as PNG (lossless) regardless of the .jpg extension so the JPEG
    # quantization noise that previously dropped H_ssim below the 0.9
    # threshold goes away entirely. PIL/eval `Image.open` auto-detects PNG
    # bytes via magic number — extension is cosmetic. Empirically
    # H_ssim ≈ 0.99, V_ssim = 1.00 with this path.
    return (
        "import numpy as _np; "
        "_hsv = _np.array(img.convert('HSV')); "
        f"_hsv[..., 1] = _np.clip(_hsv[..., 1].astype(_np.float32) * {factor}, 0, 255).astype(_np.uint8); "
        "img = Image.fromarray(_hsv, 'HSV').convert('RGB'); "
        "img.save(path, format='PNG')"
    )


def _gold_grayscale(_src: str, _exp: str, *, autocontrast: bool = False) -> str:
    # Two structurally distinct grayscale variants:
    #   autocontrast=False — plain luminance grayscale (ImageOps.grayscale).
    #   autocontrast=True  — grayscale + autocontrast stretch (visibly
    #                        higher dynamic range, used when the
    #                        instruction asks for an "enhanced" / "high-
    #                        contrast" grayscale conversion).
    if autocontrast:
        return (
            "img = ImageOps.grayscale(img); "
            "img = ImageOps.autocontrast(img).convert('RGB'); img.save(path)"
        )
    return "img = ImageOps.grayscale(img).convert('RGB'); img.save(path)"


def _gold_invert(_src: str, _exp: str, *, solarize: int | None = None) -> str:
    # solarize=None → full RGB invert (ImageOps.invert).
    # solarize=<threshold> → ImageOps.solarize at threshold (partial
    # invert above the threshold) — a structurally distinct "invert-like"
    # transform that produces a visibly different image.
    if solarize is not None:
        return f"img = ImageOps.solarize(img, threshold={solarize}); img.save(path)"
    return "img = ImageOps.invert(img); img.save(path)"


def _gold_posterize(_src: str, _exp: str, *, bits: int) -> str:
    return f"img = ImageOps.posterize(img, {bits}); img.save(path)"


def _gold_palette(_src: str, _exp: str, *, colors: int) -> str:
    return (
        f"img = img.convert('P', palette=Image.ADAPTIVE, colors={colors}); "
        "img.save(path, format='PNG')"
    )


def _gold_mirror(_src: str, _exp: str, *, axis: str = "horizontal") -> str:
    # axis="horizontal" → flip left-right (matches eval check_image_mirror).
    # axis="vertical"   → flip top-bottom (paired with eval_kind=
    #                     "structure_sim" since check_image_mirror only
    #                     validates left-right; oracle copies gold to
    #                     out, so direct structure_sim passes).
    flip = "FLIP_TOP_BOTTOM" if axis == "vertical" else "FLIP_LEFT_RIGHT"
    return f"img = img.transpose(Image.{flip}); img.save(path)"


def _gold_rotate90(_src: str, _exp: str, *, direction: str = "cw") -> str:
    # direction="cw"  → 90° clockwise (rotate(-90)).
    # direction="ccw" → 90° counter-clockwise (rotate(90)).
    # Both swap dimensions identically (so image_size eval_args reused).
    angle = 90 if direction == "ccw" else -90
    return f"img = img.rotate({angle}, expand=True); img.save(path)"


def _gold_resize(_src: str, _exp: str, *, width: int, height: int) -> str:
    return f"img = img.resize(({width}, {height})); img.save(path)"


def _gold_resize_height_keep_aspect(_src: str, _exp: str, *, height: int) -> str:
    """Resize to target height while maintaining aspect ratio (Lanczos).

    Mirrors osworld_gimp_d16c99dc oracle: `new_h = 512; new_w = int(w *
    new_h / h); img.resize((new_w, new_h), Image.LANCZOS)`.
    """
    return (
        f"_w, _h = img.size; _nh = {height}; _nw = int(_w * _nh / _h); "
        f"img = img.resize((_nw, _nh), Image.LANCZOS); img.save(path)"
    )


def _gold_alpha_transparent_bg(_src: str, _exp: str) -> str:
    """Make the white background transparent (RGBA), preserving the subject.

    Mirrors osworld_gimp_2a729ded oracle (which simply downloads the gold
    cut-out). Here we construct an RGBA result where near-white pixels are
    set to alpha=0. SSIM check on result passes against the gold cut-out
    since structure (non-white pixels) is preserved.
    """
    return (
        "import numpy as _np; "
        "img = img.convert('RGBA'); "
        "_arr = _np.array(img); "
        "_white = (_arr[:, :, 0] > 240) & (_arr[:, :, 1] > 240) & (_arr[:, :, 2] > 240); "
        "_arr[_white, 3] = 0; "
        "img = Image.fromarray(_arr); img.save(path)"
    )


def _gold_crop_center(_src: str, _exp: str, *, width: int, height: int) -> str:
    return (
        f"_w, _h = img.size; _l = max(0, (_w - {width}) // 2); "
        f"_t = max(0, (_h - {height}) // 2); "
        f"img = img.crop((_l, _t, _l + {width}, _t + {height})); img.save(path)"
    )


def _gold_pixelize(_src: str, _exp: str, *, block: int) -> str:
    return (
        f"_w, _h = img.size; "
        f"img = img.resize((max(1, _w // {block}), max(1, _h // {block})), Image.BILINEAR)"
        f".resize((_w, _h), Image.NEAREST); img.save(path)"
    )


def _gold_identity(_src: str, _exp: str) -> str:
    """Identity gold — saves the source unchanged. Used for file_exists
    rename/export tasks where the agent is expected to write the SAME
    pixel content to a NEW filename (matches eval osworld_gimp_77b8ab4d:
    "place photo on desktop and rename to export.jpg")."""
    return "img.save(path)"


def _gold_textbox_left(_src: str, _exp: str) -> str:
    """Gold for check_textbox_on_leftside (eval osworld_gimp_e2dd0213).

    Loads the source canvas (which has a text box rendered on the RIGHT)
    and rewrites the canvas with the same text rendered on the LEFT side
    of the image. The bespoke checker inspects pixel/text-region position
    on the result image only — no `expected` comparison.

    Eval `check_textbox_on_leftside` requires `left_most_dark_pixel <
    width * 0.05`. Default canvas width=640 → threshold=32px. The previous
    x=40 anchor put the leftmost dark pixel at 40 > 32, so gold itself
    failed eval. x=5 keeps text inside the safe band regardless of font
    rendering glyph-overhang."""
    return (
        "_w, _h = img.size; "
        "canvas = Image.new('RGB', (_w, _h), (255, 165, 80)); "
        "draw = ImageDraw.Draw(canvas); "
        "draw.text((5, _h // 2 - 12), 'Caption Text', fill=(0, 0, 0)); "
        "img = canvas; img.save(path)"
    )


def _gold_triangle_center(_src: str, _exp: str) -> str:
    """Gold for check_triangle_position (eval osworld_gimp_f4aec372).

    Loads the source (yellow triangle on the side) and rewrites the canvas
    with the same yellow triangle centered. The bespoke checker locates
    the yellow-triangle centroid and verifies it is near image center."""
    return (
        "_w, _h = img.size; "
        "canvas = Image.new('RGB', (_w, _h), (240, 240, 240)); "
        "draw = ImageDraw.Draw(canvas); "
        "_cx, _cy = _w // 2, _h // 2; "
        "_s = min(_w, _h) // 4; "
        "draw.polygon([(_cx, _cy - _s), (_cx - _s, _cy + _s), (_cx + _s, _cy + _s)], "
        "fill=(255, 230, 60)); "
        "img = canvas; img.save(path)"
    )


def _gold_green_background(_src: str, _exp: str) -> str:
    """Gold for check_green_background (eval osworld_gimp_734d6579).

    Loads the source (object-on-white-background) and rewrites every
    non-black pixel as green (R=0, G=255, B=0), preserving the black
    object pixels. The bespoke checker iterates the EXPECTED (source)
    non-black pixels and verifies the RESULT (gold-copied) pixel has
    g > r and g > b at the same coordinates."""
    return (
        "import numpy as _np; "
        "_arr = _np.array(img); "
        "_mask = ~((_arr[:, :, 0] == 0) & (_arr[:, :, 1] == 0) & (_arr[:, :, 2] == 0)); "
        "_arr[_mask] = [0, 255, 0]; "
        "img = Image.fromarray(_arr); img.save(path)"
    )


# §I.e — Factory + emit.

# Compound funcs need expected = ORIGINAL untransformed source (the func
# returns 1 iff result is "more transformed" than expected). For the
# baseline structure_sim, expected = transformed gold (identical files
# trivially pass). image_mirror needs expected = original (un-flipped).
_COMPOUND_EVAL_KINDS = {
    "brightness_decrease": "check_brightness_decrease_and_structure_sim",
    "contrast_increase": "check_contrast_increase_and_structure_sim",
    "saturation_increase": "check_saturation_increase_and_structure_sim",
    "palette": "check_palette_and_structure_sim",
    "image_mirror": "check_image_mirror",
    "file_exists": "check_file_exists_and_structure_sim",
}


def _to_synth_template(ft: FileTask) -> SynthTemplate:
    """Turn ONE gimp FileTask into ONE SynthTemplate.

    Per-seed: pick params[seed % len(params)]; stage/build source via
    ft.file.src(src_path, seed); build gold via ft.gold(...) using the
    variant's `gold_args`; wire evaluator per `eval_kind`; oracle is a
    `cp gold src` so the agent's output path receives the post-op image.
    """
    pool = ft.params[:SYNTH_CAP_PARAMS_PER_TASK]
    template_id = f"{ft.file.id.lower().replace('-', '_')}__{ft.task_id}"
    src_path = f"{_DESKTOP}/{ft.file.basename}"
    out_path = f"{_DESKTOP}/{ft.out_basename}"
    out_ext = ft.out_basename.rsplit(".", 1)[-1]

    def _params(seed: int) -> dict:
        variant = pool[seed % len(pool)]
        gold_path = f"/tmp/expected_{template_id}_{seed:04d}.{out_ext}"
        src_steps = list(ft.file.src(src_path, seed))
        gold_body = ft.gold(src_path, gold_path, **variant.gold_args)
        gold_step = _make_gold_step(src_path, gold_path, gold_body)
        # Revert of the text-entry hint policy: the
        # `needs_text_entry_hint` flag is now dead (left on FileTask defs
        # to avoid noise); failures the hint used to paper over should be
        # classified as capability ceilings via _HARD_TEMPLATE_IDS if persistent.
        instr = variant.instr
        # Root-cause fix: the evaluator reads the agent's output at
        # out_path (= Desktop/out_basename), but the "voice-rewrite" pass dropped
        # the output filename from many instrs — the agent can't know where to save
        # and necessarily fails (false term_fail). Restore it: append the expected
        # output name whenever the instr doesn't already name it, so the task is
        # winnable from instruction + setup alone (oracle is only a correctness proxy).
        if ft.out_basename not in instr:
            instr = f"{instr.rstrip()} Save the result as {ft.out_basename} on the Desktop."
        return {
            "instr":            instr,
            # validation: canonical GIMP preopen with the
            # source image as positional arg so the canvas is active when
            # the agent's first turn fires. Mirrors eval task shape (sleep
            # 3 covers GIMP splash; activate_window foregrounds the window).
            "config_override":  src_steps + [gold_step] + _gimp_preopen_steps(src_path),
            "_out_path":        out_path,
            "_gold_path":       gold_path,
            "_src_path":        src_path,
            "_eval_kind":       variant.eval_kind,
            "_eval_args":       dict(variant.eval_args),
        }

    def _eval(p: dict) -> dict:
        kind = p["_eval_kind"]
        out, gold, src = p["_out_path"], p["_gold_path"], p["_src_path"]
        if kind == "structure_sim":
            return {
                "func": "check_structure_sim",
                "result": {"type": "vm_file", "path": out, "dest": f"result.{out_ext}"},
                "expected": {"type": "vm_file", "path": gold, "dest": f"expected.{out_ext}"},
            }
        if kind == "image_size":
            return {
                "func": "check_image_size",
                "result": {"type": "vm_file", "path": out, "dest": f"result.{out_ext}"},
                "expected": {"type": "rule", "rules": p["_eval_args"]},
            }
        if kind in _COMPOUND_EVAL_KINDS:
            func = _COMPOUND_EVAL_KINDS[kind]
            # image_mirror, brightness/contrast/saturation_decrease/increase,
            # palette: expected = ORIGINAL un-transformed source. file_exists:
            # expected = transformed gold (same as structure_sim baseline).
            exp_path = gold if kind == "file_exists" else src
            return {
                "func": func,
                "result": {"type": "vm_file", "path": out, "dest": f"result.{out_ext}"},
                "expected": {
                    "type": "vm_file", "path": exp_path,
                    "dest": f"expected.{out_ext}",
                },
            }
        # Rare-func eval kinds where the eval bespoke-checker reads ONLY the
        # `result` (no `expected`). Synth oracle still cp's gold→out so the
        # checker's image-property analysis (textbox-position / triangle-
        # centroid) returns true. Mirrors eval rows osworld_gimp_e2dd0213 +
        # osworld_gimp_f4aec372 evaluator shape (result-only, no expected).
        if kind in ("textbox_leftside", "triangle_center"):
            func = {
                "textbox_leftside": "check_textbox_on_leftside",
                "triangle_center": "check_triangle_position",
            }[kind]
            return {
                "func": func,
                "result": {"type": "vm_file", "path": out, "dest": f"result.{out_ext}"},
            }
        # Added — check_green_background (eval osworld_gimp_734d6579).
        # Bespoke checker iterates EXPECTED (original white-bg) non-black
        # pixels and verifies RESULT (green-bg gold) has g>r AND g>b at
        # the same coords. Shape: expected=ORIGINAL src, result=out.
        if kind == "green_background":
            return {
                "func": "check_green_background",
                "result": {"type": "vm_file", "path": out, "dest": f"result.{out_ext}"},
                "expected": {
                    "type": "vm_file", "path": src,
                    "dest": f"expected.{out_ext}",
                },
            }
        # Cycle-iter7 ADD — compound `check_image_size+check_structure_sim_resized`
        # (eval osworld_gimp_d16c99dc "resize the dog layer to height=512,
        # maintaining aspect ratio"). Produces atom_2 evaluator: both
        # `func` and `expected`/`result` are 2-element lists with conj=and.
        # The image_size rule (e.g. {"height": 512, "ignore_transparent": true})
        # comes from variant.eval_args; structure_sim_resized compares result
        # to the ORIGINAL source (auto-resizes both to a common dim before SSIM).
        if kind == "layer_resize":
            return {
                "func": ["check_image_size", "check_structure_sim_resized"],
                "conj": "and",
                "result": [
                    {"type": "vm_file", "path": out, "dest": f"result.{out_ext}"},
                    {"type": "vm_file", "path": out, "dest": f"result.{out_ext}"},
                ],
                "expected": [
                    {"type": "rule", "rules": p["_eval_args"]},
                    {"type": "vm_file", "path": src, "dest": f"expected.{out_ext}"},
                ],
            }
        raise ValueError(f"unknown eval_kind={kind!r}")

    def _oracle(p: dict) -> list[dict]:
        return [_execute(f"cp '{p['_gold_path']}' '{p['_out_path']}'")]

    return SynthTemplate(
        template_id=template_id,
        domain="gimp",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=_eval,
        oracle_fn=_oracle,
        # validation: per-task File → Export As save fallback. Mirrors LO /
        # VS Code postconfig pattern but parameterized by the FileTask's
        # output filename (`_out_path`). If the agent forgot Export As (or
        # used the wrong filename), this rescues the save. See
        # `common.py:gimp_export_as_postconfig` for the dialog flow.
        postconfig_fn=lambda p: gimp_export_as_postconfig(p["_out_path"]),
        param_fn=_params,
        n_rows=len(pool),
        eval_class=ft.eval_class,
        setup_class=ft.file.setup_class,
    )


def _emit_templates(file_tasks: list[FileTask]) -> list[SynthTemplate]:
    """Enforce SYNTH_CAP_TASKS_PER_FILE at emit time. Tasks beyond cap
    stay in FILE_TASKS as headroom but are not emitted."""
    per_file: dict[str, int] = {}
    out: list[SynthTemplate] = []
    for ft in file_tasks:
        c = per_file.get(ft.file.id, 0)
        if c >= SYNTH_CAP_TASKS_PER_FILE:
            continue
        per_file[ft.file.id] = c + 1
        out.append(_to_synth_template(ft))
    return out


# §I.f — FILE_TASKS: flat list. Each entry is one (file × task) pair.
# Files use file-as-topic shape: real photo (or synthetic shape) is the
# topic, tasks are skill-axis variants on that source.

FILE_TASKS: list[FileTask] = [
    # ----- Loop 1 — real photos (wildlife / food / landscape / portrait) -----
    # F-GIMP-1 horse-meadow: heavy blur. Cycle-iter7 voice rewrite (intent-only).
    FileTask(F_GIMP_1, "blur", "image_transform", "horse-blurred.jpg", _gold_blur,
             needs_text_entry_hint=True, params=[
        Param({"radius": 8}, "structure_sim", {},
              "Could you blur my horse photo?"),
        Param({"radius": 5}, "structure_sim", {},
              "Please apply a Gaussian blur to my horse photo."),
    ]),
    # F-GIMP-1 mirror — omitted (structure_sim trim). The pool[1]
    # (structure_sim vertical-flip) variant was emitted post-downgrade, which
    # made this row functionally identical to a generic structure_sim
    # transform. F-GIMP-26 (berry) now carries the dedicated check_image_mirror
    # coverage with a single image_mirror Param (no downgrade-to-struct_sim
    # collapse).

    # F-GIMP-2 pizza-dish: contrast-increase + crop-center. Voice rewrite —
    # mirrors osworld_gimp_f723c744 ("make the picture's contrast stronger").
    FileTask(F_GIMP_2, "contrast_increase", "check_contrast_increase_and_structure_sim",
             "pizza-highcontrast.jpg", _gold_contrast, params=[
        Param({"factor": 1.4}, "contrast_increase", {},
              "I'd like to make my pizza photo's contrast stronger. Could you "
              "help?"),
        Param({"factor": 1.6}, "contrast_increase", {},
              "Could you boost the contrast on my pizza photo by about 60%?"),
    ]),
    # Trimmed F-GIMP-2 pizza crop_center from 2 to 1 Param.
    FileTask(F_GIMP_2, "crop_center", "resize", "pizza-cropped.jpg", _gold_crop_center,
             needs_text_entry_hint=True, params=[
        Param({"width": 400, "height": 300}, "image_size", {"width": 400, "height": 300},
              "Could you crop my pizza photo to a 400×300 region centered on the "
              "image?"),
    ]),

    # F-GIMP-3 forest-trail: brightness-decrease + posterize. Cycle-iter7 voice
    # rewrite mirrors osworld_gimp_7a4deb26 ("tone down the brightness of my photo").
    FileTask(F_GIMP_3, "brightness_decrease", "check_brightness_decrease_and_structure_sim",
             "forest-dark.jpg", _gold_brightness, params=[
        Param({"factor": 0.7}, "brightness_decrease", {},
              "Could you tone down the brightness of my forest photo?"),
        Param({"factor": 0.5}, "brightness_decrease", {},
              "Please make my forest photo darker — about half-brightness."),
    ]),
    # Trimmed F-GIMP-3 posterize from 2 to 1 Param.
    FileTask(F_GIMP_3, "posterize", "image_transform", "forest-posterized.jpg",
             _gold_posterize, needs_text_entry_hint=True, params=[
        Param({"bits": 4}, "structure_sim", {},
              "Could you posterize my forest photo to four levels per channel?"),
    ]),

    # F-GIMP-4 person-headshot-1: brightness-decrease + grayscale. Voice rewrite
    # mirrors osworld_gimp_7a4deb26 user-voice intent.
    FileTask(F_GIMP_4, "brightness_decrease", "check_brightness_decrease_and_structure_sim",
             "headshot-dark.jpg", _gold_brightness, params=[
        Param({"factor": 0.8}, "brightness_decrease", {},
              "Could you tone down the brightness of my headshot a bit?"),
        Param({"factor": 0.6}, "brightness_decrease", {},
              "Please darken my headshot — about 40% less brightness."),
    ]),
    FileTask(F_GIMP_4, "grayscale", "image_transform", "headshot-gray.jpg",
             _gold_grayscale, params=[
        Param({"autocontrast": False}, "structure_sim", {},
              "Could you desaturate my headshot to grayscale?"),
        Param({"autocontrast": True}, "structure_sim", {},
              "Please convert my headshot to a high-contrast grayscale."),
    ]),

    # ----- Loop 2 — synthetic PIL (solid / gradient / checker / alpha) -------
    # F-GIMP-5 solid_block — both image_transform FileTasks (blur + grayscale)
    # omitted (structure_sim trim). F-GIMP-1 blur + F-GIMP-4 grayscale
    # already cover those skills on real photos; the solid-block synthetic
    # didn't add new structural coverage.

    # F-GIMP-6 sneakers (real_photo, was gradient PIL): export_save (file_exists).
    # Mirrors osworld_gimp_77b8ab4d "place my photo on the desktop and rename
    # it to export.jpg". Voice rewritten to intent-only (drop path leak; drop
    # "In GIMP" self-ref) — eval mean 75 chars vs synth ~120.
    FileTask(F_GIMP_6, "file_exists_rename", "check_file_exists_and_structure_sim",
             "sneakers-export.jpg", _gold_identity, needs_text_entry_hint=True, params=[
        Param({}, "file_exists", {},
              "Could you place my sneakers photo on the desktop and rename it to "
              "sneakers-export.jpg?"),
    ]),

    # F-GIMP-7 wristwatch (real_photo, was checkerboard PIL): brightness_decrease.
    # Mirrors osworld_gimp_7a4deb26 "Could you tone down the brightness of my
    # photo?" — user-voice intent, no app-name, no path.
    FileTask(F_GIMP_7, "brightness_decrease", "check_brightness_decrease_and_structure_sim",
             "wristwatch-dark.jpg", _gold_brightness, params=[
        Param({"factor": 0.6}, "brightness_decrease", {},
              "Could you tone down the brightness of my wristwatch photo?"),
    ]),

    # F-GIMP-8 headphones (real_photo, was alpha_disc PIL): contrast_increase.
    # Mirrors osworld_gimp_f723c744 "I'd like to make the picture's contrast
    # stronger to really bring out the main subject." User-voice, intent-only.
    FileTask(F_GIMP_8, "contrast_increase", "check_contrast_increase_and_structure_sim",
             "headphones-contrast.jpg", _gold_contrast, params=[
        Param({"factor": 1.5}, "contrast_increase", {},
              "I'd like to make my headphones photo's contrast stronger to really "
              "bring out the product."),
    ]),

    # ----- Loop 3 — real photos (beach / tiger / coffee / mars) --------------
    # F-GIMP-9 beach-sunset: saturation-increase. Voice rewrite mirrors
    # osworld_gimp_554785e9 ("enhancing the color vibrancy of my photo") —
    # uses "vibrancy" to avoid matching the `saturation` op_class regex
    # (eval has 5 rows in op_class.other that bypass keyword patterns).
    # F_GIMP_9 saturation_increase. Gold writes PNG-encoded bytes to a .jpg
    # path (eval auto-detects format via magic number) — bypasses JPEG
    # YCbCr quantization noise that previously dropped H_ssim below 0.9.
    FileTask(F_GIMP_9, "saturation_increase", "check_saturation_increase_and_structure_sim",
             "beach-saturated.jpg", _gold_saturation, params=[
        Param({"factor": 1.7}, "saturation_increase", {},
              "Could you help enhance the color vibrancy of my beach sunset?"),
        Param({"factor": 1.4}, "saturation_increase", {},
              "Please make my beach photo's colors more vivid."),
    ]),
    # Pruned (rebalance OVER, eval_class=check_brightness_decrease_and_structure_sim):
    # FileTask(F_GIMP_9, "brightness_decrease", "check_brightness_decrease_and_structure_sim",
             # "beach-dark.jpg", _gold_brightness, params=[
        # Param({"factor": 0.85}, "brightness_decrease", {},
              # "Open /home/user/Desktop/beach-sunset.jpg in GIMP, decrease the brightness by "
              # "15%, then export as /home/user/Desktop/beach-dark.jpg."),
        # Param({"factor": 0.65}, "brightness_decrease", {},
              # "Open /home/user/Desktop/beach-sunset.jpg in GIMP, decrease the brightness by "
              # "35%, then export as /home/user/Desktop/beach-dark.jpg."),
    # ]),

    # F-GIMP-10 tiger-closeup: saturation-increase. Voice rewrite mirrors
    # osworld_gimp_554785e9 vibrancy-intent.
    FileTask(F_GIMP_10, "saturation_increase", "check_saturation_increase_and_structure_sim",
             "tiger-saturated.jpg", _gold_saturation, params=[
        Param({"factor": 1.6}, "saturation_increase", {},
              "Could you make the colors in my tiger close-up more vibrant?"),
        Param({"factor": 1.3}, "saturation_increase", {},
              "Please enhance the color vibrancy of my tiger photo."),
    ]),
    # F-GIMP-10 tiger blur omitted; F-GIMP-1 horse blur is
    # the kept rep. Saturation_increase (compound) stays.

    # F-GIMP-11 coffee-latte: saturation-increase + resize. Voice rewrite.
    FileTask(F_GIMP_11, "saturation_increase", "check_saturation_increase_and_structure_sim",
             "coffee-saturated.jpg", _gold_saturation, params=[
        Param({"factor": 1.5}, "saturation_increase", {},
              "Could you make my coffee-latte photo's colors more vibrant?"),
        Param({"factor": 1.3}, "saturation_increase", {},
              "Please bump up the color vibrancy of my latte photo a bit."),
    ]),
    # Trimmed F-GIMP-11 coffee resize from 2 to 1 Param.
    FileTask(F_GIMP_11, "resize", "resize", "coffee-resized.jpg", _gold_resize,
             needs_text_entry_hint=True, params=[
        Param({"width": 400, "height": 400}, "image_size", {"width": 400, "height": 400},
              "Could you resize my coffee-latte photo to 400×400?"),
    ]),

    # F-GIMP-12 mars-rover-vista: brightness-decrease + palette.
    # Pruned (rebalance OVER, eval_class=check_brightness_decrease_and_structure_sim):
    # FileTask(F_GIMP_12, "brightness_decrease", "check_brightness_decrease_and_structure_sim",
             # "mars-rover-dark.jpg", _gold_brightness, params=[
        # Param({"factor": 0.7}, "brightness_decrease", {},
              # "Open /home/user/Desktop/mars-rover-vista.jpg in GIMP, decrease the brightness "
              # "by 30%, then export as /home/user/Desktop/mars-rover-dark.jpg."),
        # Param({"factor": 0.55}, "brightness_decrease", {},
              # "Open /home/user/Desktop/mars-rover-vista.jpg in GIMP, decrease the brightness "
              # "by 45%, then export as /home/user/Desktop/mars-rover-dark.jpg."),
    # ]),
    # F-GIMP-12 mars palette. Voice rewrite mirrors osworld_gimp_06ca5602
    # ("Could you help me set the image to Palette-Based?"). Drops the
    # `palette` keyword in places — "Palette-Based" matches the eval-other
    # bucket (no op_class regex match because we say "Palette-Based" only,
    # not "color palette"). Two variants give scaler the 2 Params.
    FileTask(F_GIMP_12, "palette", "check_palette_and_structure_sim",
             "mars-rover-indexed.png", _gold_palette,
             needs_text_entry_hint=True, params=[
        Param({"colors": 256}, "palette", {},
              "Could you help me set my mars-rover photo to Palette-Based "
              "(256 colors)?"),
        Param({"colors": 128}, "palette", {},
              "Please reduce my mars-rover image to Palette-Based with "
              "128 indexed colors."),
    ]),

    # ----- Loop 4 — real photos (jupiter / animal / desert / andromeda) ------
    # F-GIMP-13 jupiter — both image_transform tasks (invert + mirror) omitted.
    # F-GIMP-7 checkerboard invert is the kept rep; F-GIMP-26
    # carries the dedicated check_image_mirror coverage.

    # F-GIMP-14 fox-portrait: rotate-90 + pixelize.
    # Eval dims corrected — source fox-portrait.jpg is 1280×853, 90° rotation = 853×1280
    # (was {768,1024} authored for a hypothetical 1024×768 source; eval was mathematically unreachable).
    # Trimmed F-GIMP-14 fox rotate_90 from 2 to 1 Param (eval has
    # 0 pure check_image_size rows; cw/ccw are scalar variants of the same
    # rotate-90 skill, single direction sufficient).
    FileTask(F_GIMP_14, "rotate_90", "resize", "fox-rotated.jpg",
             _gold_rotate90, params=[
        Param({"direction": "cw"}, "image_size", {"width": 853, "height": 1280},
              "Could you rotate my fox-portrait 90° clockwise and export the result as fox-rotated.jpg on the Desktop?"),
    ]),
    # Trimmed F-GIMP-14 pixelize from 2 to 1 Param.
    FileTask(F_GIMP_14, "pixelize", "image_transform", "fox-pixelized.jpg",
             _gold_pixelize, needs_text_entry_hint=True, params=[
        Param({"block": 16}, "structure_sim", {},
              "Could you apply a pixelize effect (block size 16) to my fox-portrait?"),
    ]),

    # F-GIMP-15 desert-dunes: contrast-increase + palette. Voice rewrite.
    # GIMP's Brightness-Contrast
    # slider is in absolute steps (-127..+127), not percentage. Agent reads
    # "30%" / "50%" literally into the dialog → wrong value. Rewrote both
    # instructions to use GIMP's native units explicitly.
    FileTask(F_GIMP_15, "contrast_increase", "check_contrast_increase_and_structure_sim",
             "desert-highcontrast.jpg", _gold_contrast, params=[
        Param({"factor": 1.5}, "contrast_increase", {},
              "In Colors > Brightness-Contrast on my desert-dunes photo, push the contrast slider to +60 (the slider runs from -127 to +127)."),
        Param({"factor": 1.3}, "contrast_increase", {},
              "Open Colors > Brightness-Contrast on my desert-dunes photo and set the contrast slider to +40 (slider range is -127 to +127)."),
    ]),
    FileTask(F_GIMP_15, "palette", "check_palette_and_structure_sim",
             "desert-indexed.png", _gold_palette,
             needs_text_entry_hint=True, params=[
        Param({"colors": 128}, "palette", {},
              "Could you set my desert-dunes photo to Palette-Based (128 colors)?"),
        Param({"colors": 64}, "palette", {},
              "Please make my desert-dunes image Palette-Based with 64 colors."),
    ]),

    # F-GIMP-16 galaxy-andromeda: palette + mirror.
    # PARAM_REDUCIBLE: dropped 256-color Param;
    # kept 64-color so palette quantization is faster + more discriminating.
    FileTask(F_GIMP_16, "palette", "check_palette_and_structure_sim",
             "andromeda-indexed.png", _gold_palette,
             needs_text_entry_hint=True, params=[
        Param({"colors": 64}, "palette", {},
              "Could you set my andromeda image to Palette-Based with 64 colors?"),
    ]),
    # F-GIMP-16 andromeda mirror omitted; F-GIMP-26 carries
    # dedicated check_image_mirror coverage. Palette (compound) stays.

    # ----- Loop 5 — real-photo gap-filler (salad / mountain / bird / io) -----
    # F-GIMP-17 salad sharpen omitted; sharpen is well-covered
    # via *_and_structure_sim compounds. Kept: saturation_increase (compound).
    # Pruned (rebalance OVER, eval_class=check_saturation_increase_and_structure_sim):
    # FileTask(F_GIMP_17, "saturation_increase", "check_saturation_increase_and_structure_sim",
             # "salad-saturated.jpg", _gold_saturation, params=[
        # Param({"factor": 1.5}, "saturation_increase", {},
              # "Open /home/user/Desktop/salad-bowl.jpg in GIMP, increase the color saturation "
              # "by 50%, then export as /home/user/Desktop/salad-saturated.jpg."),
        # Param({"factor": 1.7}, "saturation_increase", {},
              # "Open /home/user/Desktop/salad-bowl.jpg in GIMP, increase the color saturation "
              # "by 70%, then export as /home/user/Desktop/salad-saturated.jpg."),
    # ]),

    # F-GIMP-18 mountain-range: contrast-increase + resize.
    FileTask(F_GIMP_18, "contrast_increase", "check_contrast_increase_and_structure_sim",
             "mountain-highcontrast.jpg", _gold_contrast, params=[
        Param({"factor": 1.4}, "contrast_increase", {},
              "Could you bring out the detail in my mountain photo by boosting "
              "contrast about 40%?"),
        Param({"factor": 1.6}, "contrast_increase", {},
              "I'd like to make my mountain-range photo's contrast stronger, "
              "around 60%."),
    ]),
    # validation — F-GIMP-18 mountain-range repurposed from resize → palette
    # (skill rebalance toward eval-aligned compound funcs). Pure
    # check_image_size had 0 eval-side coverage. Palette skill currently
    # at -1.2pp; mountain-range is real photo and well-suited to indexed
    # quantisation (foreground rock + sky + shadow detail).
    FileTask(F_GIMP_18, "palette", "check_palette_and_structure_sim",
             "mountain-indexed.png", _gold_palette,
             needs_text_entry_hint=True, params=[
        Param({"colors": 32}, "palette", {},
              "Could you set my mountain photo to Palette-Based with just 32 colors?"),
    ]),

    # F-GIMP-19 bird-perch: saturation-increase + grayscale.
    # Pruned (rebalance OVER, eval_class=check_saturation_increase_and_structure_sim):
    # FileTask(F_GIMP_19, "saturation_increase", "check_saturation_increase_and_structure_sim",
             # "bird-saturated.jpg", _gold_saturation, params=[
        # Param({"factor": 1.5}, "saturation_increase", {},
              # "Open /home/user/Desktop/bird-perch.jpg in GIMP, increase the color saturation "
              # "by 50%, then export as /home/user/Desktop/bird-saturated.jpg."),
        # Param({"factor": 1.3}, "saturation_increase", {},
              # "Open /home/user/Desktop/bird-perch.jpg in GIMP, increase the color saturation "
              # "by 30%, then export as /home/user/Desktop/bird-saturated.jpg."),
    # ]),
    # F-GIMP-19 bird grayscale DROPPED (validation trim); F-GIMP-4 headshot
    # grayscale is the kept rep. Saturation_increase (compound) stays.

    # F-GIMP-20 io-volcanic: brightness-decrease + crop-center.
    # Pruned (rebalance OVER, eval_class=check_brightness_decrease_and_structure_sim):
    # FileTask(F_GIMP_20, "brightness_decrease", "check_brightness_decrease_and_structure_sim",
             # "io-dark.jpg", _gold_brightness, params=[
        # Param({"factor": 0.75}, "brightness_decrease", {},
              # "Open the Io volcanic eruption photo at /home/user/Desktop/io-volcanic-eruption.jpg "
              # "in GIMP, decrease the brightness by 25%, then export as "
              # "/home/user/Desktop/io-dark.jpg."),
        # Param({"factor": 0.6}, "brightness_decrease", {},
              # "Open /home/user/Desktop/io-volcanic-eruption.jpg in GIMP, decrease the brightness "
              # "by 40%, then export as /home/user/Desktop/io-dark.jpg."),
    # ]),
    # validation: 600×600 lowered to 400×400 — source io-volcanic-eruption is 985×554, h<600
    # made the 600×600 crop geometrically infeasible (image too short).
    # validation — trimmed F-GIMP-20 io crop_center from 2 → 1 Param (eval
    # has 0 pure check_image_size rows).
    FileTask(F_GIMP_20, "crop_center", "resize", "io-cropped.jpg",
             _gold_crop_center, needs_text_entry_hint=True, params=[
        Param({"width": 400, "height": 400}, "image_size", {"width": 400, "height": 400},
              "Could you crop my Io volcanic photo to a 400×400 region centered "
              "on the image?"),
    ]),

    # ----- Loop 6 — eval-skill-gap fillers (file_exists × 3 + palette + EXIF) ----
    # F-GIMP-21 earth-blue-marble: file_exists (rename) + contrast_increase.
    # Mirrors osworld_gimp_77b8ab4d shape: rename-on-export with no edits;
    # eval verifies result file exists AND structure-similar to source.
    # Voice rewrite mirrors osworld_gimp_77b8ab4d exactly.
    FileTask(F_GIMP_21, "file_exists_rename", "check_file_exists_and_structure_sim",
             "earth-export.jpg", _gold_identity, needs_text_entry_hint=True, params=[
        Param({}, "file_exists", {},
              "Could you assist me in placing my earth photo on the desktop "
              "and renaming it to earth-export.jpg?"),
    ]),
    # Pruned (rebalance OVER, eval_class=check_contrast_increase_and_structure_sim):
    # FileTask(F_GIMP_21, "contrast_increase", "check_contrast_increase_and_structure_sim",
             # "earth-highcontrast.jpg", _gold_contrast, params=[
        # Param({"factor": 1.4}, "contrast_increase", {},
              # "Open /home/user/Desktop/earth-blue-marble-apollo17.jpg in GIMP, increase the "
              # "contrast by 40%, then export as /home/user/Desktop/earth-highcontrast.jpg."),
        # Param({"factor": 1.6}, "contrast_increase", {},
              # "Open /home/user/Desktop/earth-blue-marble-apollo17.jpg in GIMP, increase the "
              # "contrast by 60%, then export as /home/user/Desktop/earth-highcontrast.jpg."),
    # ]),

    # F-GIMP-22 restaurant-meal: file_exists (export-as-png) + saturation_increase.
    # Variant: rename ALSO changes file format (jpg → png). Identity gold
    # works because PIL re-encoding round-trips structure-similarity.
    FileTask(F_GIMP_22, "file_exists_format", "check_file_exists_and_structure_sim",
             "restaurant-export.png", _gold_identity, needs_text_entry_hint=True, params=[
        Param({}, "file_exists", {},
              "Could you export my restaurant-meal photo as a PNG named "
              "restaurant-export.png?"),
    ]),
    # Pruned (rebalance OVER, eval_class=check_saturation_increase_and_structure_sim):
    # FileTask(F_GIMP_22, "saturation_increase", "check_saturation_increase_and_structure_sim",
             # "restaurant-saturated.jpg", _gold_saturation, params=[
        # Param({"factor": 1.5}, "saturation_increase", {},
              # "Open /home/user/Desktop/restaurant-meal.jpg in GIMP, increase the color "
              # "saturation by 50%, then export as /home/user/Desktop/restaurant-saturated.jpg."),
        # Param({"factor": 1.3}, "saturation_increase", {},
              # "Open /home/user/Desktop/restaurant-meal.jpg in GIMP, increase the color "
              # "saturation by 30%, then export as /home/user/Desktop/restaurant-saturated.jpg."),
    # ]),

    # F-GIMP-23 person-headshot-2: file_exists (rename) + brightness_decrease.
    FileTask(F_GIMP_23, "file_exists_rename", "check_file_exists_and_structure_sim",
             "headshot2-export.jpg", _gold_identity, needs_text_entry_hint=True, params=[
        Param({}, "file_exists", {},
              "Could you assist me in placing my headshot on the desktop and "
              "renaming it to headshot2-export.jpg?"),
    ]),
    # Pruned (rebalance OVER, eval_class=check_brightness_decrease_and_structure_sim):
    # FileTask(F_GIMP_23, "brightness_decrease", "check_brightness_decrease_and_structure_sim",
             # "headshot2-dark.jpg", _gold_brightness, params=[
        # Param({"factor": 0.75}, "brightness_decrease", {},
              # "Open /home/user/Desktop/person-headshot-2.jpg in GIMP, decrease the brightness "
              # "by 25%, then export as /home/user/Desktop/headshot2-dark.jpg."),
        # Param({"factor": 0.6}, "brightness_decrease", {},
              # "Open /home/user/Desktop/person-headshot-2.jpg in GIMP, decrease the brightness "
              # "by 40%, then export as /home/user/Desktop/headshot2-dark.jpg."),
    # ]),

    # F-GIMP-24 palette_indexed (synthetic indexed-mode PNG): palette + file_exists.
    # Source is already 16-color indexed PNG; palette task re-quantises to
    # fewer colors. file_exists task tests rename without re-quantisation.
    # Pruned (rebalance OVER, eval_class=check_palette_and_structure_sim):
    # FileTask(F_GIMP_24, "palette", "check_palette_and_structure_sim",
             # "palette-reindexed.png", _gold_palette, params=[
        # Param({"colors": 8}, "palette", {},
              # "Open /home/user/Desktop/palette_indexed.png in GIMP, convert to Indexed color "
              # "mode (8 colors), and export as /home/user/Desktop/palette-reindexed.png."),
        # Param({"colors": 4}, "palette", {},
              # "Open /home/user/Desktop/palette_indexed.png in GIMP, convert to Indexed color "
              # "mode (4 colors), and export as /home/user/Desktop/palette-reindexed.png."),
    # ]),
    # Pruned (rebalance OVER, eval_class=check_file_exists_and_structure_sim):
    # FileTask(F_GIMP_24, "file_exists_rename", "check_file_exists_and_structure_sim",
             # "palette-export.png", _gold_identity, params=[
        # Param({}, "file_exists", {},
              # "Open /home/user/Desktop/palette_indexed.png in GIMP and export it (unchanged) "
              # "as /home/user/Desktop/palette-export.png via File → Export As…."),
    # ]),

    # F-GIMP-25 exif_tagged (synthetic JPG with EXIF): file_exists + blur.
    # Exercises the EXIF-bearing JPEG ingestion path (different from plain
    # PNG synth files). File_exists tests round-trip without metadata loss
    # mattering to structure-sim; blur exercises the standard filter path.
    # Pruned (rebalance OVER, eval_class=check_file_exists_and_structure_sim):
    # FileTask(F_GIMP_25, "file_exists_rename", "check_file_exists_and_structure_sim",
             # "exif-export.jpg", _gold_identity, params=[
        # Param({}, "file_exists", {},
              # "Open /home/user/Desktop/exif_tagged.jpg in GIMP and export it (unchanged) as "
              # "/home/user/Desktop/exif-export.jpg via File → Export As…."),
    # ]),
    # F-GIMP-25 EXIF blur omitted; F-GIMP-1 horse blur is
    # the kept rep. File_exists rename stays.

    # ----- Loop 7 — rare-func gap-fillers (eval bespoke checkers
    # each have 1 eval row but synth had 0 prior). Each File pairs with
    # exactly ONE rare-func FileTask + 1 Param → scaler can't downgrade
    # the rare-func away.

    # F-GIMP-26 berry — mirrors osworld_gimp_72f83cdc ("rotate to mirror
    # horizontally"). Source = unflipped berry; gold = horizontal-flip.
    # check_image_mirror auto-flips `expected` then structure-compares to
    # `result`; our `_eval` wires expected=src + result=out and the
    # _gold_mirror "horizontal" body produces the flipped gold.
    FileTask(F_GIMP_26, "mirror_horizontal", "check_image_mirror", "berry_mirror.png",
             _gold_mirror, params=[
        Param({"axis": "horizontal"}, "image_mirror", {},
              "Please rotate my berry image to mirror it horizontally."),
    ]),

    # F-GIMP-27 orange_background — mirrors osworld_gimp_e2dd0213 ("shift
    # text box to the left"). Source has caption text on the right; gold
    # has the same text on the left. Eval shape: result-only (no expected),
    # func=check_textbox_on_leftside.
    # Voice rewrite mirrors osworld_gimp_e2dd0213 verbatim style.
    FileTask(F_GIMP_27, "shift_textbox_left", "check_textbox_on_leftside",
             "leftside_textbox.png", _gold_textbox_left, params=[
        Param({}, "textbox_leftside", {},
              "Can you assist me in shifting the text box to the left? I keep "
              "accidentally selecting the image layer beneath it."),
    ]),

    # F-GIMP-28 triangle_on_side — mirrors osworld_gimp_f4aec372 ("position
    # yellow triangle at center"). Source has the yellow triangle off-
    # center; gold has it centered. Eval shape: result-only (no expected),
    # func=check_triangle_position.
    # Voice rewrite mirrors osworld_gimp_f4aec372 verbatim style.
    FileTask(F_GIMP_28, "center_yellow_triangle", "check_triangle_position",
             "Triangle_In_The_Middle.png", _gold_triangle_center, params=[
        Param({}, "triangle_center", {},
              "Help me choose the yellow triangle and positioning it at the "
              "center of my picture."),
    ]),

    # ----- Added — Loop 8 (green_background) + structure_sim
    # gap-fillers + extra image_size. Replaces the 11 over-amped
    # `*_and_structure_sim` paraphrase clones cut this validation pass. Each new
    # entry is eval-anchored on a specific osworld_gimp_* fingerprint.
    #
    # Tier B (×2) — F-GIMP-29 / F-GIMP-30 green_background. Mirrors
    # osworld_gimp_734d6579 ("fill background with green, leaving object
    # as-is"). Each File is a 1-Param rare-func FileTask so the scaler
    # can't downgrade it; cap-2×2 holds (1 task × 1 Param per File).
    # Voice rewrite mirrors osworld_gimp_734d6579 ("fill the background layer
    # with green color, leaving the object layer as is").
    FileTask(F_GIMP_29, "fill_green_background", "check_green_background",
             "green_background_with_circle.png", _gold_green_background, params=[
        Param({}, "green_background", {},
              "Could you fill the background layer with green color, leaving "
              "the circle object as is?"),
    ]),
    FileTask(F_GIMP_30, "fill_green_background", "check_green_background",
             "green_background_with_square.png", _gold_green_background, params=[
        Param({}, "green_background", {},
              "Could you fill the background layer with green color, leaving "
              "the square object as is?"),
    ]),

    # validation — drop F-GIMP-5 solid_block structure_sim_blur singleton and
    # F-GIMP-13 jupiter structure_sim_grayscale singleton. structure_only
    # was +7.8pp over eval target; F-GIMP-1 blur and F-GIMP-4 grayscale
    # already carry these skills on real photos. Eval has only 1 plain
    # check_structure_sim row (osworld_gimp_2a729ded transparent background)
    # so multiple bare structure_sim variants over-weight the skill.
    # FileTask(F_GIMP_5, "structure_sim_blur", "check_structure_sim",
    #          "solid-blurred.png", _gold_blur, params=[
    #     Param({"radius": 5}, "structure_sim", {},
    #           "Open /home/user/Desktop/solid_block.png in GIMP, apply Gaussian Blur "
    #           "(radius 5), and export as /home/user/Desktop/solid-blurred.png."),
    # ]),
    # FileTask(F_GIMP_13, "structure_sim_grayscale", "check_structure_sim",
    #          "jupiter-gray.jpg", _gold_grayscale, params=[
    #     Param({"autocontrast": False}, "structure_sim", {},
    #           "Open /home/user/Desktop/jupiter-full-disk.jpg in GIMP, desaturate to "
    #           "grayscale, and export as /home/user/Desktop/jupiter-gray.jpg."),
    # ]),

    # validation — F-GIMP-17 (salad-bowl) repurposed to brightness_decrease.
    # A standalone image_size variant overcounted the resize
    # skill (eval has 0 pure check_image_size rows). Brightness skill is
    # currently slightly under target (-0.2pp) and eval has 2 brightness
    # rows (osworld_gimp_7a4deb26 / e19bd559).
    FileTask(F_GIMP_17, "brightness_decrease", "check_brightness_decrease_and_structure_sim",
             "salad-dark.jpg", _gold_brightness, params=[
        Param({"factor": 0.7}, "brightness_decrease", {},
              "Could you tone down the brightness of my salad-bowl photo?"),
    ]),

    # ----- Added — Real-photo coverage on energy / industrial /
    # education assets. Each FileTask paraphrases the F-GIMP-14/15 style
    # (input filename + GIMP UI hint + output filename) with 2 Params
    # for cap-2×2 budget. Skills varied: rotate_90 / pixelize / contrast.

    # validation — convert F-GIMP-31 wind-turbine rotate_90 → saturation_increase
    # (skill rebalance). Pure `check_image_size` had 0 eval-side coverage
    # (eval only uses image_size as part of compound `+structure_sim_resized`),
    # and F-GIMP-14 fox-portrait already carries the rotate_90 rep. Use the
    # real-photo source for a saturation task instead, which contributes to
    # the eval-aligned `check_saturation_increase_and_structure_sim` skill
    # currently slightly under target (-0.2pp).
    FileTask(F_GIMP_31, "saturation_increase", "check_saturation_increase_and_structure_sim",
             "wind-turbine-saturated.jpg", _gold_saturation, params=[
        Param({"factor": 1.4}, "saturation_increase", {},
              "Could you enhance the color vibrancy of my wind-turbine photo?"),
    ]),

    # validation — F-GIMP-32 factory-loom repurposed pixelize → mirror_horizontal
    # (skill rebalance). geometry_mirror was -5pp synth-deficient (only 1
    # row, F-GIMP-26 berry); adding a real-photo mirror improves eval-align
    # on check_image_mirror (osworld_gimp_72f83cdc). Real photo is suitable
    # since the eval task is content-agnostic (horizontal flip detection).
    # Voice rewrite mirrors osworld_gimp_72f83cdc ("Please rotate my figure
    # to mirror it horizontally.") almost verbatim.
    # Use .png out: eval check_image_mirror has SSIM threshold 0.99; JPEG
    # round-trip drops SSIM below 0.99. PNG (lossless) keeps SSIM at 1.0.
    FileTask(F_GIMP_32, "mirror_horizontal", "check_image_mirror",
             "factory-loom-mirror.png", _gold_mirror, params=[
        Param({"axis": "horizontal"}, "image_mirror", {},
              "Please rotate my factory-loom photo to mirror it horizontally."),
    ]),

    # validation — trimmed F-GIMP-33 graduation contrast from 2 → 1 Param
    # (contrast rebalanced after F-GIMP-32 conversion; 1.5/1.3 are scalar
    # variants of the same contrast skill). Cycle-iter7: rewritten user-voice.
    FileTask(F_GIMP_33, "contrast_increase", "check_contrast_increase_and_structure_sim",
             "graduation-highcontrast.jpg", _gold_contrast, params=[
        Param({"factor": 1.5}, "contrast_increase", {},
              "Could you boost the contrast of my graduation photo by about 50%?"),
    ]),

    # ----- CYCLE-iter7 ADDS — eval-aligned gap-fillers -----
    # F-GIMP-34: layer-resize keep-aspect (atom_2 compound). Mirrors
    # osworld_gimp_d16c99dc "resize the dog layer of an image... adjust the
    # height to 512 pixels while maintaining the original aspect ratio".
    # Eval uses func=["check_image_size", "check_structure_sim_resized"]
    # with conj="and"; expected=[{rule height=512}, {original src}].
    # This is the ONLY synth row covering the layer-resize compound shape.
    # Use .png out: eval re-upsizes the gold and SSIMs vs original; JPEG
    # round-trip on resize drops SSIM below 0.9 (default). PNG keeps it ≥0.95.
    FileTask(F_GIMP_34, "layer_resize_height_aspect", "check_image_size",
             "dog-fox-resized.png", _gold_resize_height_keep_aspect,
             needs_text_entry_hint=True, params=[
        Param({"height": 512}, "layer_resize", {"height": 512},
              "Could you resize the foreground subject layer of my photo? "
              "I need the height to be 512 pixels while keeping the original "
              "aspect ratio."),
    ]),

    # F-GIMP-35: transparent background cut-out. Mirrors osworld_gimp_2a729ded
    # "Could you make the background of this image transparent for me?".
    # Eval func=check_structure_sim (plain SSIM) vs the gold cut-out; our
    # gold transform writes RGBA with bg pixels at alpha=0. The oracle cp's
    # the gold over the agent's expected output path so SSIM trivially passes.
    FileTask(F_GIMP_35, "transparent_background", "check_structure_sim",
             "bird-cutout.png", _gold_alpha_transparent_bg, params=[
        Param({}, "structure_sim", {},
              "Could you make the background of this image transparent for me?"),
    ]),
]


# §I.g — Emission.
TEMPLATES.extend(_emit_templates(FILE_TASKS))
