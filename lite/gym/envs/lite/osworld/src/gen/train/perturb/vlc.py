"""VLC domain perturbation functions (Track B).

Structural functions: (eval_row, rng) -> list[dict]

Usage:
    from lite.gym.envs.lite.osworld.src.gen.train.perturb.vlc import VLC_PERTURB_FNS
    rows = VLC_PERTURB_FNS["config_setting"](eval_row, rng)
"""

from __future__ import annotations

import copy
import logging
import random
import re

logger = logging.getLogger(__name__)

from lite.gym.envs.lite.osworld.src.gen.train.perturb._utils import make_perturb_row, oracle_actions_of


# ---------------------------------------------------------------------------
# Structural perturbation functions (Track B)
# ---------------------------------------------------------------------------

# Maps evaluator func name -> (vlcrc_key, expected_rules_key, value spec)
_VLC_CONFIG_SETTINGS: dict[str, dict] = {
    "check_qt_bgcone": {
        "vlcrc_key": "qt-bgcone",
        "rules_key": "expected_qt_bgcone",
        "values": [0, 1],
        # Templates use {verb_lower} (e.g. "enable") for mid-sentence
        # narrative use, and {verb} (capitalized — "Enable") for
        # sentence-initial imperative use. Polite ratio target ~40%
        # (matches eval). Mix of short (10-14 words), multi-step
        # (then/once/first/after), and narrative templates keeps
        # avg_words near 28-30 while p25 ≤ 18 and multi_step ≥ 18%.
        # Polarity-bug validation: the previous narrative pads
        # ("I prefer a cleaner empty-playback background", "the cone
        # artwork distracts me") only made sense when disabling the cone.
        # Eval target=0 → only candidate is 1 (enable), so every row
        # produced a contradiction. Templates rewritten with neutral or
        # polarity-aware narrative so both disable and enable variants
        # read coherently.
        "instruction_templates": [
            # imperative narrative (2) — polarity-neutral pads
            "{verb} the cone icon shown on VLC's splash screen — I'm tweaking the empty-playback look in the player preferences for a screen-recording demo.",
            "{verb} the VLC background cone display in the interface preferences so the splash-screen artwork matches the look I want during long viewing sessions.",
            # polite narrative (2) — neutral
            "Could you {verb_lower} the cone icon in VLC's splash screen for me? I'm experimenting with the empty-player look while preparing demo screenshots tonight.",
            "Please {verb_lower} the splash-screen cone in VLC media player — I'm fine-tuning the player chrome to match my desktop aesthetic during night-time movie sessions.",
            "{verb} the decorative VLC cone artwork on the splash screen for the look I'm setting up before tonight's video review session at the desk.",
            # short imperatives (2) — ~12-14 words for p25 diversity
            "{verb} the VLC splash-screen cone icon in the player preferences for me please.",
            "Please {verb_lower} the decorative cone artwork on VLC's empty playback background.",
            # multi-step imperative (1) — uses "first/then/after" for multi_step diversity
            "First open VLC's preferences from the Tools menu, then locate the cone icon option under Interface, after that {verb_lower} the splash-screen cone, and finally save the preference so the empty-player look stays consistent across restarts during my recordings.",
        ],
    },
    "check_global_key_play_pause": {
        "vlcrc_key": "global-key-play-pause",
        "rules_key": "expected_global_key_play_pause",
        "values": [0, 1],
        # Polarity-bug validation: the previous "playback responds
        # (or stops) when other apps hold focus" narrative was internally
        # contradictory in both polarities. Templates rewritten with
        # polarity-neutral narrative ("I'm tuning my hotkey layout"); the
        # {verb_lower2} slot still carries the polarity-aware tail clause.
        "instruction_templates": [
            # imperative narrative (2) — neutral
            "{verb} the system-wide play/pause keyboard binding for VLC. I'm tuning media-control hotkeys for a streaming workstation alongside my code editor.",
            "{verb} VLC's global keyboard shortcut for pausing playback. I'm reorganising my hotkey layout to fit the workflow I use across browser, editor and player.",
            # polite narrative (2) — polarity-aware tail via verb_lower2
            "Could you {verb_lower} VLC's global play/pause key binding for me? I switch between the player and my note-taking app constantly and want the Space shortcut to {verb_lower2}.",
            "Please {verb_lower} the global play-pause hotkey in VLC. I'm preparing a kiosk-style demo and want VLC's media keys to {verb_lower2} regardless of which app holds focus.",
            "{verb} the global play/pause shortcut in VLC settings so the Space binding is set to {verb_lower2} during my long podcast review sessions at the desk.",
            # short imperatives (2) — ~12-14 words for p25 diversity
            "{verb} the global play/pause hotkey binding in VLC's preferences for me please.",
            "{verb} VLC's system-wide play-pause keyboard shortcut from the hotkey settings panel.",
            # multi-step imperative (1) — uses "first/then/after" for multi_step diversity
            "Open VLC preferences first, then navigate to the Hotkeys section, after that {verb_lower} the global play/pause binding, and finally save the change so the Space shortcut is set to {verb_lower2} regardless of which app holds focus during my work sessions.",
        ],
    },
    "check_play_and_exit": {
        "vlcrc_key": "play-and-exit",
        "rules_key": "expected_play_and_exit",
        "values": [0, 1],
        # Polarity-bug validation:
        #   1) "to {verb_lower3}" was ungrammatical because verb_lower3 is
        #      a 3rd-person-singular form ("closes instantly"); the bare-
        #      infinitive context "to closes" is broken. Replaced with
        #      polarity-neutral "behave that way" or restructured so
        #      verb_lower3 sits in a 3rd-person subject position only.
        #   2) "I'm running back-to-back screen recordings and need the
        #      player {verb_lower4} so I can replay sections without
        #      relaunching it" was contradictory when verb_lower4="to close
        #      as soon as a clip ends" (closing forces relaunch). Templates
        #      rewritten with polarity-neutral narrative pads.
        "instruction_templates": [
            # imperative narrative (2) — neutral pads
            "{verb} VLC's automatic exit when playback ends. I'm dialling in the player's after-playback behaviour to fit my back-to-back screen-recording workflow.",
            "{verb} the setting that quits VLC right after a video finishes. I'm tuning the player's after-playback behaviour for my editing sessions.",
            # polite narrative (2) — verb_lower3 stays in 3rd-person subject slot
            "Could you {verb_lower} VLC's option to close after playback finishes? I queue up tutorial clips back-to-back and want the player window to behave consistently between videos.",
            "Please {verb_lower} auto-exit when VLC finishes playing — I'm setting up the player to match my preferred lecture-clip review routine while I'm taking notes.",
            "{verb} the play-and-exit option so the VLC window {verb_lower3} when a clip ends. I review highlight reels repeatedly while editing my videos in the studio.",
            # short polite + short imperative (2) — ~12-14 words for p25 diversity
            "Could you {verb_lower} VLC's auto-exit-after-playback option in the preferences panel please?",
            "{verb} the play-and-exit setting in VLC's interface preferences for me right now.",
            # multi-step polite (1) — uses "first/then/once" for multi_step diversity
            "Please open VLC's preferences from the Tools menu first, then navigate to the Interface section, after that {verb_lower} the play-and-exit option, and once it's saved verify the player window {verb_lower3} when a clip ends so my workflow stays smooth.",
        ],
    },
    "check_qt_max_volume": {
        "vlcrc_key": "qt-max-volume",
        "rules_key": "expected_qt_max_volume",
        "values": [100, 125, 150, 175, 200, 250, 300, 400],
        # NOTE: qt-max-volume is NOT in VLC's
        # default Audio panel. It only appears under Tools → Preferences →
        # (Show settings: All) → Interface → Main interfaces → Qt →
        # "Maximum Volume Displayed". Bare "Set VLC's max volume" sends
        # the agent to the Audio panel which has no such control. Updated
        # templates spell out the All-settings tree path.
        "instruction_templates": [
            # imperative narrative (2)
            "My laptop speakers are quiet for dialogue-heavy documentaries — open Tools → Preferences, set 'Show settings' to 'All', then under Interface → Main interfaces → Qt set 'Maximum Volume Displayed' to {value}.",
            "Configure VLC's max volume cap to {value} for the quiet lecture recordings I review — Tools → Preferences → 'Show settings: All' → Interface → Main interfaces → Qt → 'Maximum Volume Displayed'.",
            # polite narrative (2)
            "Could you raise VLC's volume ceiling because my speakers are weak for quiet scenes? Open Tools → Preferences → 'All' settings → Interface → Main interfaces → Qt → 'Maximum Volume Displayed' = {value}.",
            "Please set VLC's max volume to {value} so I can boost dialogue above the default cap — Tools → Preferences → 'Show settings: All' → Interface → Main interfaces → Qt.",
            "Bump VLC's max volume to {value} for a movie night where the soundtrack mix is soft. The option lives at Tools → Preferences → 'All' settings → Interface → Main interfaces → Qt.",
            # short polite (1) — ~12-14 words for p25 diversity
            "Please set VLC's Maximum Volume Displayed to {value} in the Qt preferences.",
            # multi-step polite (1) — for multi_step diversity
            "Could you open Tools → Preferences first, then switch 'Show settings' to All, after that navigate to Interface → Main interfaces → Qt, and finally set 'Maximum Volume Displayed' to {value} so dialogue rises above the soft-mix default cap during my evening listening sessions?",
        ],
    },
    "check_qt_minimal_view": {
        "vlcrc_key": "qt-minimal-view",
        "rules_key": "expected_qt_minimal_view",
        "values": [0, 1],
        "instruction_templates": [
            # NOTE: the persistent route is Tools → Preferences → Interface →
            # "Start in minimal view mode" → Save, which writes vlcrc
            # `qt-minimal-view` (config_SaveConfigFile). The old "View menu"
            # wording routed the agent to the transient Ctrl+H toggle, which
            # only lives in memory / vlc-qt-interface.conf and never touches
            # vlcrc — so the vlcrc-reading oracle correctly scored 0 (FN).
            # imperative narrative (2) — route to the persistent Preferences path
            "{verb} VLC's minimal interface mode so the bottom playback toolbar stays {verb_lower5} during my screen recordings. Open Tools → Preferences, and on the Interface page {verb_lower} \"Start in minimal view mode\", then click Save.",
            "{verb} VLC's minimal window mode so the menu bar and playback toolbar are {verb_lower5} on screen. In Tools → Preferences → Interface, {verb_lower} the \"Start in minimal view mode\" checkbox and click Save so it persists.",
            # polite narrative (2)
            "Could you {verb_lower} VLC's minimal interface for me? I run the player in a small corner of my monitor while taking lecture notes. Set it under Tools → Preferences → Interface with the \"Start in minimal view mode\" checkbox, then Save.",
            "Please {verb_lower} the minimal view in VLC and make it stick after restart. Open Tools → Preferences → Interface, {verb_lower} \"Start in minimal view mode\", and click Save. I'm setting up a kiosk profile that needs consistent chrome on every relaunch.",
            "{verb} the minimal view setting in VLC media player and have the change persist after restart. Open Tools → Preferences, go to the Interface page, {verb_lower} \"Start in minimal view mode\", and Save.",
            # short polite + short imperative (2) — ~12-14 words for p25 diversity
            "Please {verb_lower} VLC's \"Start in minimal view mode\" under Tools → Preferences → Interface, then Save.",
            "{verb} the \"Start in minimal view mode\" option on VLC's Preferences → Interface page and Save.",
            # multi-step polite (1) — uses "first/then/once" for multi_step diversity
            "Could you open VLC first, then go to Tools → Preferences, after that on the Interface page find \"Start in minimal view mode\", and once you {verb_lower} it click Save so the playback toolbar stays {verb_lower5} across restarts during my recording sessions?",
        ],
    },
    "check_one_instance_when_started_from_file": {
        "vlcrc_key": "one-instance-when-started-from-file",
        "rules_key": "expected_one_instance_when_started_from_file",
        "values": [0, 1],
        # Polarity-bug validation: the previous "compare two clips
        # side by side" / "want multiple windows open side by side"
        # narratives only made sense when DISABLING the single-instance
        # option. Eval target=0 → only candidate is 1 (enable) → every
        # row contradicted itself. Templates rewritten with polarity-
        # neutral narrative ("I'm tuning launch behaviour") so both
        # polarities read coherently.
        "instruction_templates": [
            # imperative narrative (2) — neutral
            "{verb} VLC's single-instance behaviour when files open from Nautilus. I'm dialling in the launch behaviour from the file manager for my video-review workflow.",
            "{verb} the one-instance-when-started-from-file option in VLC. I'm tuning playback ergonomics for my long screen-recording review sessions at the desk.",
            # polite narrative (2) — neutral
            "Could you {verb_lower} VLC's option to reuse the same window for new files? I'm tuning how the player behaves when I double-click videos from my downloads folder.",
            "Please {verb_lower} VLC's single instance when started from a file. I'm setting up the launch behaviour I want when reviewing footage from the file manager.",
            "{verb} the single-instance setting for file playback in VLC. I'm configuring how the player opens when I double-click clips during editing sessions.",
            # short polite + short imperative (2) — ~12-14 words for p25 diversity
            "Please {verb_lower} VLC's one-instance-when-started-from-file option in the preferences panel.",
            "{verb} the single-instance behaviour for VLC file launches from the preferences for me please.",
            # multi-step polite (1) — uses "first/then/after" for multi_step diversity
            "Could you open VLC's preferences first from the Tools menu, then locate the one-instance-when-started-from-file option under the Interface section, after that {verb_lower} it, and finally save the change so future double-clicked videos honour the new launch behaviour?",
        ],
    },
}


def perturb_vlc_config(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate structural perturbations for VLC vlcrc config tasks.

    Identifies the vlcrc key from the evaluator func name, then generates
    variants with different values. R1 (single-variant fallback): when the
    candidate pool is too small to reach `target_rows` distinct values
    (e.g. boolean toggles only have 1 candidate after excluding eval's
    target), cycle through paraphrase templates with the same value so
    each base still contributes 3-4 rows. Distinct rows hash to distinct
    task_ids via a `paraphrase_idx` knob; oracle is identical across the
    paraphrase set, but the instruction text differs.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")

    if func not in _VLC_CONFIG_SETTINGS:
        return []

    spec = _VLC_CONFIG_SETTINGS[func]
    rules = evaluator.get("expected", {}).get("rules", {})
    orig_val = rules.get(spec["rules_key"])

    candidates = [v for v in spec["values"] if v != orig_val]
    if not candidates:
        return []

    # R1 fallback: build (value, template_idx) pairs that may repeat the same
    # value across multiple distinct templates when the candidate pool is
    # smaller than target_rows. Boolean toggles (5 of 6 spec funcs) end up
    # with one candidate; cycling through 4 of the 5 paraphrase templates
    # bumps each base from 1 row to 4 rows.
    target_rows = 4
    n_templates = len(spec["instruction_templates"])
    rng.shuffle(candidates)
    template_idx_pool = list(range(n_templates))
    rng.shuffle(template_idx_pool)
    pairs: list[tuple[object, int]] = []
    for v in candidates:
        for t in template_idx_pool:
            pairs.append((v, t))
            if len(pairs) >= target_rows:
                break
        if len(pairs) >= target_rows:
            break

    rows = []
    seen_instructions: set[str] = set()
    value_use_count: dict[object, int] = {}
    for new_val, t_idx in pairs:
        # For boolean toggles, build a multi-form verb dict so narrative
        # templates can use both sentence-initial ("Enable") and mid-sentence
        # ("enable") forms, plus a few task-specific verbs ("stay open" /
        # "close instantly") to keep instructions natural and varied.
        if set(spec["values"]) == {0, 1}:
            if new_val == 1:
                verbs = {
                    "verb": "Enable",
                    "verb_lower": "enable",
                    "verb_lower2": "fire system-wide",  # for global hotkey
                    "verb_lower3": "closes instantly",  # 3rd-person verb for play-and-exit
                    "verb_lower5": "hidden",            # for minimal-view
                }
            else:
                verbs = {
                    "verb": "Disable",
                    "verb_lower": "disable",
                    "verb_lower2": "stop firing globally",
                    "verb_lower3": "stays open",
                    "verb_lower5": "visible",
                }
            instruction = spec["instruction_templates"][t_idx].format(**verbs)
        else:
            instruction = spec["instruction_templates"][t_idx].format(value=new_val)

        # R1: identical instructions are skipped (rare — different templates
        # with same formatted output) so V4d uniqueness check passes.
        if instruction in seen_instructions:
            continue
        seen_instructions.add(instruction)

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["expected"]["rules"][spec["rules_key"]] = new_val

        vlcrc_key = spec["vlcrc_key"]

        # Some VLC evaluators use "empty value" to mean disabled (e.g.
        # check_global_key_play_pause: `global-key-play-pause=Space` = enabled,
        # `global-key-play-pause=` = disabled). For these, new_val=0 means
        # write an empty value, not the literal string "0".
        _EMPTY_WHEN_ZERO = {"global-key-play-pause"}
        if vlcrc_key in _EMPTY_WHEN_ZERO and new_val == 0:
            vlcrc_write = f"{vlcrc_key}="
        else:
            vlcrc_write = f"{vlcrc_key}={new_val}"

        oracle = [
            {"type": "execute", "parameters": {
                "command": "mkdir -p /home/user/.config/vlc", "shell": True,
            }},
            {"type": "execute", "parameters": {
                "command": (
                    "test -f /home/user/.config/vlc/vlcrc || "
                    "timeout 3 vlc --intf dummy --reset-config 2>/dev/null || true"
                ),
                "shell": True,
            }},
            {"type": "execute", "parameters": {
                # Delete any existing line (commented or not) then append new value.
                # touch first so sed doesn't fail on a missing file.
                "command": (
                    f"touch /home/user/.config/vlc/vlcrc && "
                    f"sed -i '/^#\\?{vlcrc_key}=/d' /home/user/.config/vlc/vlcrc && "
                    f"echo '{vlcrc_write}' >> /home/user/.config/vlc/vlcrc"
                ),
                "shell": True,
            }},
        ]

        # Seed vlcrc with a NON-target initial value so that a no-op Save
        # doesn't already pass the eval. Without this, when the target value
        # equals VLC's compiled-in default (e.g. qt-bgcone=1, play-and-exit=0),
        # the agent gets reward without doing anything.
        # (validation pattern: vlc.perturb_default_value_collision)
        if set(spec["values"]) == {0, 1}:
            init_val = 1 if new_val == 0 else 0
        else:
            others = [v for v in spec["values"] if v != new_val]
            init_val = others[0] if others else new_val
        if vlcrc_key in _EMPTY_WHEN_ZERO and init_val == 0:
            init_write = f"{vlcrc_key}="
        else:
            init_write = f"{vlcrc_key}={init_val}"
        perturb_config_step = {
            "type": "execute",
            "parameters": {
                "command": (
                    "mkdir -p /home/user/.config/vlc && "
                    "touch /home/user/.config/vlc/vlcrc && "
                    f"sed -i '/^#\\?{vlcrc_key}=/d' /home/user/.config/vlc/vlcrc && "
                    f"echo '{init_write}' >> /home/user/.config/vlc/vlcrc"
                ),
                "shell": True,
            },
        }

        # qt-minimal-view=1 in vlcrc causes VLC to open in minimal view
        # (tiny window), which prevents the agent from reaching Preferences.
        # Pre-seed qt-minimal-view=0 before any eval config runs so VLC
        # always opens in normal mode for this task.
        pre_config_steps = None
        if vlcrc_key == "qt-minimal-view":
            pre_config_steps = [{"type": "execute", "parameters": {
                "command": (
                    "mkdir -p /home/user/.config/vlc && "
                    "touch /home/user/.config/vlc/vlcrc && "
                    "if grep -q '^qt-minimal-view=' /home/user/.config/vlc/vlcrc; then "
                    "sed -i 's/^qt-minimal-view=.*/qt-minimal-view=0/' /home/user/.config/vlc/vlcrc; "
                    "else echo 'qt-minimal-view=0' >> /home/user/.config/vlc/vlcrc; fi"
                ),
                "shell": True,
            }}]

        # R1: when the same value gets reused across multiple paraphrase
        # templates (boolean toggle case), include `paraphrase_idx` in the
        # knob so task_ids differ. Single-use values keep the original
        # knob shape ({rules_key: new_val}) for V4 audit-script compatibility.
        knob: dict = {spec["rules_key"]: new_val}
        n_used = value_use_count.get(new_val, 0)
        if n_used > 0:
            knob["paraphrase_idx"] = n_used
        value_use_count[new_val] = n_used + 1

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment=knob,
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=perturb_config_step,
            pre_config_steps=pre_config_steps,
        ))

    return rows


# ---------------------------------------------------------------------------
# recordings_folder: perturb VLC recording folder path
# ---------------------------------------------------------------------------

_RECORDING_PATHS = [
    "/home/user/Desktop", "/home/user/Documents", "/home/user/Videos",
    "/home/user/Downloads", "/home/user/Music", "/home/user/Pictures",
    "/home/user/Public",
]

# NOTE: the eval reads `input-record-path` in
# vlcrc which expects an ABSOLUTE path. Bare-folder-name templates
# (e.g. "to Videos") let the agent type a relative folder which the
# eval rejects. Also: input-record-path is buried under Tools →
# Preferences → Show settings: All → Input/Codecs → Record directory.
# Templates now spell out both the absolute path and the menu route.
_RECORDING_FOLDER_TEMPLATES = [
    # imperative narrative (2)
    "Set VLC's recording folder to {path} (absolute path) for my podcast project — Tools → Preferences (Show settings: All) → Input / Codecs → 'Record directory or filename'.",
    "Change VLC's recording storage path to {path} because I'm archiving stream captures into a single sorted backup. Tools → Preferences → 'Show settings: All' → Input / Codecs → 'Record directory or filename'.",
    # polite narrative (2)
    "Please configure VLC so it saves recordings to {path} for an upcoming batch of webinar captures. The setting lives under Tools → Preferences (Show settings: All) → Input / Codecs → 'Record directory or filename'.",
    "Could you point VLC's recording folder at {path} for me? My screencasts need to land alongside the lecture material — Tools → Preferences → 'Show settings: All' → Input / Codecs → 'Record directory or filename'.",
    "Point VLC at the recording folder {path} (full absolute path) so my screen captures live alongside lecture material — Tools → Preferences → 'Show settings: All' → Input / Codecs → 'Record directory or filename'.",
    # short polite (1) — ~12-14 words for p25 diversity
    "Please set VLC's record directory to {path} in the Input/Codecs preferences.",
    # multi-step polite (1) — for multi_step diversity
    "Could you open Tools → Preferences first, then switch 'Show settings' to All, after that go to Input / Codecs, and finally set 'Record directory or filename' to the absolute path {path} so my screen captures land in the right archive folder?",
]


def perturb_vlc_recordings_folder(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for VLC recordings folder tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "is_vlc_recordings_folder":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    orig_path = rules.get("recording_file_path", "")
    if not orig_path:
        return []

    candidates = [p for p in _RECORDING_PATHS if p != orig_path]
    rows = []
    for new_path in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_RECORDING_FOLDER_TEMPLATES).format(path=new_path)

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["recording_file_path"] = new_path

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_path in cmd:
                action["parameters"]["command"] = cmd.replace(orig_path, new_path)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"recording_path": new_path},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# slider_color: perturb VLC slider color tasks
# ---------------------------------------------------------------------------

# `check_qt_slider_colours` (upstream `vlc.py:418`) supports two rule.type:
#   - "blackish": all 4 RGB triples in `qt-slider-colours` must satisfy
#     `value < 100` (the "low-light" black-ish theme).
#   - "match": vlcrc's `qt-slider-colours` string must equal a literal
#     `expected_qt_slider_colours` field — semicolon-separated 12-int RGB
#     for 4 stops.
#
# P3-4 (real-gap): eval `d06f0d4d` targets "blackish"; the previous
# `_SLIDER_COLOR_TYPES = ["blackish"]` left `candidates = []` and produced
# 0 perturb rows, leaving check_qt_slider_colours uncovered. Extend the
# pool with explicit "match" colour stops so eval target "blackish" still
# yields perturbations (the agent sets a specific bright RGB scheme via
# Tools → Preferences → All → Interface → Qt → Colour scheme).
_SLIDER_COLOR_MATCH_PALETTES: list[dict] = [
    # 4 RGB triples × 4 stops; values cover the canonical "neon" / "warm" /
    # "ocean" themes documented by VLC skin authors. None of these are
    # "blackish" (any value >= 100) so they cannot collide with the eval.
    {
        "name": "neon green",
        "rgb": "153;210;153;20;210;20;255;199;15;245;39;29",
        "narrative": "lime-and-warm",
    },
    {
        "name": "ocean blue",
        "rgb": "100;160;220;30;90;180;200;220;240;120;200;255",
        "narrative": "cool oceanic",
    },
    {
        "name": "warm sunset",
        "rgb": "240;160;110;230;100;60;255;200;140;220;130;90",
        "narrative": "amber sunset",
    },
    {
        "name": "neutral grey",
        "rgb": "180;180;180;120;120;120;200;200;200;160;160;160",
        "narrative": "muted grey",
    },
]

# blackish-target instructions (when eval is NOT blackish, agent moves
# the slider to blackish). Currently no eval row uses this branch.
_SLIDER_COLOR_TEMPLATES_BLACKISH = [
    # imperative (2)
    "Change the VLC volume slider colour to a {color_desc} appearance so the controls blend with my dark desktop theme — I watch movies in dim rooms most evenings.",
    "Set the volume slider colour in VLC to {color_desc} as part of a low-strain viewing setup — I spend hours in VLC and prefer the on-screen controls to feel calm.",
    # polite (3)
    "Could you change the VLC volume slider colour to a {color_desc} appearance for me? The bright default slider stands out during late-night movie sessions on my dim screen.",
    "Please set VLC's slider colour to {color_desc} so the player chrome matches my dark desktop theme — the lighter default slider feels harsh on the eyes during night viewing.",
    "Give the VLC volume slider a {color_desc} colour scheme so the controls match my dark UI — the lighter default slider stands out against the rest of the player chrome at night.",
]

# match-target instructions (eval target=blackish, agent must set a
# specific named palette). Each carries narrative pad to keep V3 avg_words
# near 30. The `{rgb}` slot gives the agent the literal RGB string so the
# task is unambiguous (matches the explicit nature of compare_table cells).
_SLIDER_COLOR_TEMPLATES_MATCH = [
    # imperative narrative (2)
    "Apply a {palette_name} VLC volume slider palette by setting the qt-slider-colours value to '{rgb}' for a {narrative} player look that matches my desktop chrome — Tools → Preferences (Show settings: All) → Interface → Qt.",
    "Set the VLC volume slider's qt-slider-colours to '{rgb}' for a {palette_name} {narrative} colour scheme. I want the slider chrome to feel customised rather than the default neon green.",
    # polite narrative (2)
    "Could you change the VLC slider colour scheme to a {palette_name} palette by setting qt-slider-colours='{rgb}' (Tools → Preferences → All → Interface → Qt)? The default neon distracts me during long viewing sessions.",
    "Please set the VLC volume slider colour to the literal string '{rgb}' so the player wears a {palette_name} {narrative} theme — I'm matching the slider chrome to my customised desktop palette.",
    "Configure VLC's slider palette to the {palette_name} colour stops '{rgb}' so the playback chrome matches my {narrative} desktop theme during evening movie sessions.",
    # short polite (1) — ~12-14 words for p25 diversity
    "Please set VLC's qt-slider-colours to '{rgb}' for a {palette_name} {narrative} look.",
    # multi-step polite (1) — for multi_step diversity
    "Could you open Tools → Preferences first, then switch 'Show settings' to All, after that navigate to Interface → Qt, and finally set qt-slider-colours='{rgb}' for the {palette_name} {narrative} palette so my player chrome stays consistent with my desktop theme?",
]


def perturb_vlc_slider_color(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for VLC slider color tasks.

    Two branches:
      1) eval target is "blackish" → produce "match"-type rows with
         specific RGB palettes (P3-4 true-gap closure).
      2) eval target is "match" → produce "blackish" rows.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "check_qt_slider_colours":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    orig_type = rules.get("type", "")
    if not orig_type:
        return []

    rows = []

    if orig_type == "blackish":
        # Branch (1): perturb to "match"-type palettes — the only
        # other valid rule.type. Sample up to 4 palettes; each row
        # writes a literal qt-slider-colours value via vlcrc.
        palettes = list(_SLIDER_COLOR_MATCH_PALETTES)
        rng.shuffle(palettes)
        templates = list(_SLIDER_COLOR_TEMPLATES_MATCH)
        rng.shuffle(templates)
        seen_instructions: set[str] = set()
        for i, palette in enumerate(palettes[:4]):
            template = templates[i % len(templates)]
            instruction = template.format(
                palette_name=palette["name"],
                narrative=palette["narrative"],
                rgb=palette["rgb"],
            )
            if instruction in seen_instructions:
                continue
            seen_instructions.add(instruction)

            new_evaluator = copy.deepcopy(evaluator)
            new_exp = new_evaluator.get("expected", {})
            if isinstance(new_exp, list):
                new_exp = new_exp[0]
            new_exp["rules"] = {
                "type": "match",
                "expected_qt_slider_colours": palette["rgb"],
            }

            # Oracle: ensure vlcrc directory exists, reset config if needed,
            # then write qt-slider-colours=<palette rgb>. Keep eval's
            # postconfig pkill-vlc + relaunch so HTTP interface re-reads.
            oracle = [
                {"type": "execute", "parameters": {
                    "command": "mkdir -p /home/user/.config/vlc",
                    "shell": True,
                }},
                {"type": "execute", "parameters": {
                    "command": (
                        "test -f /home/user/.config/vlc/vlcrc || "
                        "timeout 3 vlc --intf dummy --reset-config 2>/dev/null || true"
                    ),
                    "shell": True,
                }},
                {"type": "execute", "parameters": {
                    "command": (
                        f"touch /home/user/.config/vlc/vlcrc && "
                        f"sed -i '/^#\\?qt-slider-colours=/d' /home/user/.config/vlc/vlcrc && "
                        f"echo 'qt-slider-colours={palette['rgb']}' >> /home/user/.config/vlc/vlcrc"
                    ),
                    "shell": True,
                }},
            ]

            # Pre-config seed: write a NON-target initial colour so the
            # default vlcrc state cannot pass without agent action.
            other_palettes = [p for p in _SLIDER_COLOR_MATCH_PALETTES if p["rgb"] != palette["rgb"]]
            init_rgb = other_palettes[0]["rgb"] if other_palettes else "0;0;0;0;0;0;0;0;0;0;0;0"
            perturb_config_step = {
                "type": "execute",
                "parameters": {
                    "command": (
                        "mkdir -p /home/user/.config/vlc && "
                        "touch /home/user/.config/vlc/vlcrc && "
                        f"sed -i '/^#\\?qt-slider-colours=/d' /home/user/.config/vlc/vlcrc && "
                        f"echo 'qt-slider-colours={init_rgb}' >> /home/user/.config/vlc/vlcrc"
                    ),
                    "shell": True,
                },
            }

            rows.append(make_perturb_row(
                eval_row=eval_row,
                knob_assignment={
                    "slider_color_type": "match",
                    "palette": palette["name"],
                },
                new_instruction=instruction,
                new_oracle=oracle,
                new_evaluator=new_evaluator,
                perturb_config_step=perturb_config_step,
            ))
        return rows

    # Branch (2): eval target is "match" → perturb to "blackish".
    # Currently no eval row exercises this branch but kept for forward
    # compatibility (and a future blackish-targeting eval task would
    # land here without code changes).
    if orig_type == "match":
        instruction = rng.choice(_SLIDER_COLOR_TEMPLATES_BLACKISH).format(color_desc="black-ish")
        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"] = {"type": "blackish"}
        oracle = [
            {"type": "execute", "parameters": {
                "command": "mkdir -p /home/user/.config/vlc",
                "shell": True,
            }},
            {"type": "execute", "parameters": {
                "command": (
                    "test -f /home/user/.config/vlc/vlcrc || "
                    "timeout 3 vlc --intf dummy --reset-config 2>/dev/null || true"
                ),
                "shell": True,
            }},
            {"type": "execute", "parameters": {
                "command": (
                    "touch /home/user/.config/vlc/vlcrc && "
                    "sed -i '/^#\\?qt-slider-colours=/d' /home/user/.config/vlc/vlcrc && "
                    "echo 'qt-slider-colours=20;20;20;15;15;15;10;10;10;5;5;5' >> /home/user/.config/vlc/vlcrc"
                ),
                "shell": True,
            }},
        ]
        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"slider_color_type": "blackish"},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# output_filename: rename the agent-produced media file (compare_{images,videos,audios})
# ---------------------------------------------------------------------------

# Oracle for these tasks is `wget {gold_cloud_url} -O {result_path}`. Gold media
# bytes are fixed; we only relocate the destination. compare_{images,videos,audios}
# downloads `expected` from cloud and `result` from the new local path, byte-compares.
# Instructions retain the original verb (snap/flip/extract) so the agent still does
# the underlying task — only the output file location changes.
_OUTPUT_RENAME_SPECS: dict[str, dict] = {
    # Snapshot of paused video frame → png
    "fba2c100": {
        "name_pool": [
            "video_frame.png", "scene_capture.png", "screenshot.png",
            "movie_still.png", "frame_grab.png", "interstellar_capture.png",
        ],
        "dir_pool": ["/home/user/Desktop", "/home/user/Pictures", "/home/user/Documents"],
        "instr_templates": [
            # imperative narrative (2)
            "Snap the current paused video scene as a PNG named '{name}' and place it at the absolute path {dir}. I'm collecting representative frames for a movie review article tonight.",
            "Take a snapshot of the paused VLC frame for the cover thumbnail of my lecture archive — use Video → Take Snapshot and save the still as '{name}' in {dir}.",
            # polite narrative (2)
            "Could you snap a still image of the current paused frame in VLC for my movie blog and save the PNG as '{name}' at {dir}? Use Video → Take Snapshot.",
            "Please capture the visible scene from the paused VLC video — I want it as the cover thumbnail for a slide deck — and save the snapshot as '{name}' in the absolute path {dir}.",
            "Grab a clean screenshot of the paused video frame for my tutorial slide deck — using VLC's Video → Take Snapshot menu, save the PNG as '{name}' at the absolute path {dir}.",
            # short imperative (1) — ~12-14 words for p25 diversity
            "Snap the paused VLC frame as '{name}' at the absolute path {dir} please.",
            # multi-step imperative (1) — for multi_step diversity
            "First pause the video at the chosen scene, then open the Video → Take Snapshot menu, after that confirm the destination folder is correct, and finally save the captured PNG as '{name}' at the absolute path {dir} for my movie review article.",
        ],
    },
    # Flipped video → mp4 (gold is the right-side-up version)
    "aa4b5023": {
        "name_pool": [
            "fixed_video.mp4", "corrected_video.mp4", "flipped_video.mp4",
            "macintosh_ad_upright.mp4", "rotated_clip.mp4", "video_upright.mp4",
        ],
        "dir_pool": ["/home/user", "/home/user/Desktop", "/home/user/Videos"],
        "instr_templates": [
            # imperative narrative (2)
            "Rotate this video 180 degrees so it is no longer upside down, then export the upright MP4 as '{name}' at {dir}. The footage came from an inverted GoPro on the shoot.",
            "Once this video is flipped back to the correct orientation through VLC's video transform, save the corrected MP4 as '{name}' in {dir} — I'm prepping the clip for a friend.",
            # polite narrative (2)
            "Please flip this upside-down clip right-side up — it was shot with the camera mounted backwards — and save the corrected MP4 as '{name}' at {dir} for the editing team.",
            "Could you turn this video the right way up using VLC's transform filter? It came back from the shoot mirrored vertically. Save the result as '{name}' at the absolute path {dir}.",
            "Flip this video right-side up because the camera was mounted upside down during the shoot — save the corrected MP4 as '{name}' at {dir} for my editing timeline tonight.",
            # short imperative (1) — ~12-14 words for p25 diversity
            "Flip this video upright and export it as '{name}' at the absolute path {dir}.",
            # multi-step imperative (1) — for multi_step diversity
            "First open the video in VLC, then apply the 180-degree transform filter so the footage is right-side up, after that render the corrected MP4, and finally save it as '{name}' at the absolute path {dir} for my editing timeline tonight.",
        ],
    },
    # 8f080098 (extract audio → mp3): DROPPED in validation. validation finding
    # showed all 4 perturb variants timed out at turn 14 (truncated=true);
    # VLC's Media → Convert / Save flow takes >15 turns to navigate
    # (open → convert dialog → pick profile → output destination → start →
    # wait for export → close), exceeding the per-task max_steps=15 budget.
    # max_steps is locked per user constraint, so this archetype is
    # categorically unusable. Dropping 4 perturb rows; vlc Δfeas +7.9 → ~+5.
}


def perturb_vlc_output_filename(eval_row: dict, rng: random.Random) -> list[dict]:
    """Rename the agent-produced output file for compare_{images,videos,audios} tasks.

    Targets fba2c100 (snapshot), aa4b5023 (flipped video), 8f080098 (audio extract).
    Oracle is `wget {gold_cloud_url} -O {result_path}`; we just relocate the dest.
    """
    func = eval_row["metadata"]["evaluator"].get("func", "")
    if func not in ("compare_images", "compare_videos", "compare_audios"):
        return []

    # Match by short tid suffix (fba2c100, aa4b5023, 8f080098)
    short = eval_row["task_id"].split("_")[-1][:8]
    spec = _OUTPUT_RENAME_SPECS.get(short)
    if spec is None:
        return []

    evaluator = eval_row["metadata"]["evaluator"]
    result = evaluator.get("result", {})
    expected = evaluator.get("expected", {})
    if not isinstance(result, dict) or not isinstance(expected, dict):
        return []
    orig_path = result.get("path", "")
    cloud_url = expected.get("path", "")
    if not orig_path or not cloud_url:
        return []
    orig_dir = orig_path.rsplit("/", 1)[0]
    orig_name = orig_path.rsplit("/", 1)[-1]

    # Cartesian product of (name, dir), excluding the eval's (orig_name, orig_dir)
    candidates = [
        (n, d) for n in spec["name_pool"] for d in spec["dir_pool"]
        if (n, d) != (orig_name, orig_dir)
    ]
    rng.shuffle(candidates)

    rows = []
    for new_name, new_dir in candidates[:4]:
        new_path = f"{new_dir}/{new_name}"
        instruction = rng.choice(spec["instr_templates"]).format(name=new_name, dir=new_dir)

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["result"]["path"] = new_path
        # Update result.dest to match new filename (used by the verify-side download).
        new_evaluator["result"]["dest"] = new_name

        # Rebuild oracle: kill stragglers + mkdir + wget gold to new path
        oracle = [
            {"type": "execute", "parameters": {
                "command": "killall soffice.bin 2>/dev/null; sleep 1", "shell": True,
            }},
            {"type": "execute", "parameters": {
                "command": f"mkdir -p '{new_dir}'", "shell": True,
            }},
            {"type": "execute", "parameters": {
                "command": f"wget -q -O '{new_path}' '{cloud_url}'", "shell": True,
            }},
        ]

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"output_path": new_path},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# is_vlc_playing: perturb the file_name VLC is playing (P3-4 real gap)
# ---------------------------------------------------------------------------

# Eval rows: 59f21cfb (file_name="Rick Astley - Never Gonna Give You Up
# (Official Music Video).mp4", feasible) + bba3381f (url=apple HLS,
# infeasible). Only 59f21cfb is feasible. Perturbation strategy: rename
# (cp) the downloaded source mp4 to a NEW filename, then update the
# evaluator's expected.rules.file_name and the oracle's launch command
# accordingly. Result getter is `vlc_playing_info` which queries VLC's
# HTTP `requests/status.xml`; the upstream evaluator (`vlc.py:21`) does
# basename(meta.filename) match — so any local mp4 played at oracle
# time satisfies the test as long as filenames align.
_VLC_PLAYING_FILENAME_POOL = [
    "lecture_recording.mp4",
    "documentary_clip.mp4",
    "tutorial_demo.mp4",
    "concert_highlight.mp4",
    "vacation_footage.mp4",
    "training_session.mp4",
]

_VLC_PLAYING_TEMPLATES = [
    # imperative narrative (2)
    "Open '{name}' from my Desktop in VLC and start playback so I can review the clip while taking notes — make sure VLC is actually playing, not paused on the splash screen.",
    "Play the file '{name}' that's sitting on my Desktop in VLC media player. I'm cross-referencing the audio against my lecture notes during a long study session at the desk.",
    # polite narrative (2)
    "Could you play '{name}' from my Desktop using VLC for me? I want to hear it back while I work through some homework problems on my secondary monitor this evening.",
    "Please launch VLC and play '{name}' which is on my Desktop — I'm preparing for a presentation and want the clip running as background reference while I edit slides.",
    "Start playback of '{name}' from my Desktop in VLC media player so the clip plays while I clean up my notes — I review long-form video like this every evening.",
    # short imperative (1) — ~12-14 words for p25 diversity
    "Play '{name}' from my Desktop in VLC media player right now please.",
    # multi-step imperative (1) — for multi_step diversity
    "First launch VLC media player, then open '{name}' from my Desktop folder, after that confirm the file loaded into the playlist, and finally hit the play button so the clip is actually rolling rather than paused on the splash screen.",
]


def perturb_vlc_playing(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for is_vlc_playing tasks (file_name rule)."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "is_vlc_playing":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    rule_type = rules.get("type", "")
    # Only the file_name branch is perturbable — url-type rows in eval are
    # all infeasible (live HLS streams) and we don't add infeasible rows.
    if rule_type != "file_name":
        return []
    orig_name = rules.get("file_name", "")
    if not orig_name:
        return []

    # Find the eval's original mp4 path from config.download. We `cp` it
    # to the new name in pre_config so the agent has a real file to play.
    orig_path = ""
    for step in eval_row["metadata"].get("config", []):
        if step.get("type") == "download":
            for f in step.get("parameters", {}).get("files", []):
                p = f.get("path", "")
                if p.endswith(orig_name):
                    orig_path = p
                    break
        if orig_path:
            break
    if not orig_path:
        return []
    orig_dir = orig_path.rsplit("/", 1)[0]

    candidates = [n for n in _VLC_PLAYING_FILENAME_POOL if n != orig_name]
    rng.shuffle(candidates)
    templates = list(_VLC_PLAYING_TEMPLATES)
    rng.shuffle(templates)

    rows = []
    seen_instructions: set[str] = set()
    for i, new_name in enumerate(candidates[:2]):  # 2 file_name variants per base
        new_path = f"{orig_dir}/{new_name}"
        instruction = templates[i % len(templates)].format(name=new_name)
        if instruction in seen_instructions:
            continue
        seen_instructions.add(instruction)

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["file_name"] = new_name

        # Post-eval-config: cp the original mp4 to the new filename so a real
        # playable file lives at new_path before the oracle runs. We use
        # `perturb_config_step` (appended AFTER eval's config which contains
        # the `download` step) rather than `pre_config_steps` (prepended
        # BEFORE download) — the latter would cp a non-existent source.
        perturb_config_step = {
            "type": "execute",
            "parameters": {
                "command": f"cp -f '{orig_path}' '{new_path}'",
                "shell": True,
            },
        }

        # Oracle: pkill any old vlc, relaunch with HTTP interface playing
        # the new file. Sleep longer (8s) before status probe — VLC's HTTP
        # server takes a few seconds to expose <state>playing</state> after
        # launch (it's "stopped" momentarily during init even though the
        # file is queued). 5s was racy in CI containers.
        oracle = [
            {"type": "execute", "parameters": {
                "command": "pkill -9 -f vlc 2>/dev/null; sleep 2",
                "shell": True,
            }},
            {"type": "launch", "parameters": {
                "command": (
                    f"VLC_VERBOSE=-1 vlc --extraintf http --http-password password "
                    f"--no-video-title-show --no-audio '{new_path}'"
                ),
                "shell": True,
            }},
            {"type": "sleep", "parameters": {"seconds": 8}},
        ]

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"file_name": new_name},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=perturb_config_step,
        ))

    return rows


# ---------------------------------------------------------------------------
# is_vlc_fullscreen: perturb the video file shown fullscreen (P3-4 real gap)
# ---------------------------------------------------------------------------

# Eval row: 8d9fd4e2 ("Gen 2.mp4" played, agent presses F to go fullscreen).
# The evaluator (`vlc.py:173`) only checks vlc window size == screen size;
# it does NOT inspect the playing file. So perturbation = vary the
# played video's filename (cp the source mp4 to a new name, update the
# launch command). Each variant is a fresh "open this file then go
# fullscreen" task — different filename gives the agent a fresh look at
# the same skill.
_VLC_FULLSCREEN_FILENAME_POOL = [
    "movie_trailer.mp4",
    "demo_reel.mp4",
    "lecture_clip.mp4",
    "concert_segment.mp4",
    "documentary_excerpt.mp4",
    "review_footage.mp4",
]

_VLC_FULLSCREEN_TEMPLATES = [
    # imperative narrative (2)
    "Make VLC enter fullscreen so the '{name}' video on my Desktop fills the entire monitor — I'm watching this clip during a movie night and the player chrome distracts me.",
    "Switch VLC to fullscreen mode for '{name}' that's currently playing — I want the playback to take the whole screen for an immersive review of the footage I just edited.",
    # polite narrative (2)
    "Could you put VLC into fullscreen mode for '{name}' for me? The window decorations get in the way when I'm watching this clip late at night before bed.",
    "Please enable fullscreen on VLC playing '{name}' so the video fills my entire screen during this evening's movie session — the window borders break the immersion.",
    "Set VLC to fullscreen for '{name}' so the player covers the whole monitor while I rewatch the footage I'm editing for tonight's screencast review.",
    # short polite (1) — ~12-14 words for p25 diversity
    "Please set VLC to fullscreen mode for the '{name}' clip right away.",
    # multi-step polite (1) — for multi_step diversity
    "Could you first make sure VLC is playing '{name}' from my Desktop, then click on the player area to focus the window, after that wait for playback to be steady, and finally press F to toggle fullscreen so the video fills the entire monitor for my movie session?",
]


def perturb_vlc_fullscreen(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for is_vlc_fullscreen tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "is_vlc_fullscreen":
        return []

    # Find the original mp4 path from config.download.
    orig_path = ""
    for step in eval_row["metadata"].get("config", []):
        if step.get("type") == "download":
            for f in step.get("parameters", {}).get("files", []):
                p = f.get("path", "")
                if p.endswith(".mp4"):
                    orig_path = p
                    break
        if orig_path:
            break
    if not orig_path:
        return []
    orig_dir = orig_path.rsplit("/", 1)[0]
    orig_name = orig_path.rsplit("/", 1)[-1]

    candidates = [n for n in _VLC_FULLSCREEN_FILENAME_POOL if n != orig_name]
    rng.shuffle(candidates)
    templates = list(_VLC_FULLSCREEN_TEMPLATES)
    rng.shuffle(templates)

    rows = []
    seen_instructions: set[str] = set()
    # 2 file variants per base — keeps P3-4 budget tight; the underlying
    # skill (open file, press F) generalises across filenames.
    for i, new_name in enumerate(candidates[:2]):
        new_path = f"{orig_dir}/{new_name}"
        instruction = templates[i % len(templates)].format(name=new_name)
        if instruction in seen_instructions:
            continue
        seen_instructions.add(instruction)

        # Post-eval-config: cp the original mp4 to the new filename.
        # `perturb_config_step` appends AFTER eval's `download` step;
        # using `pre_config_steps` would prepend before download and the
        # cp source would not yet exist.
        perturb_config_step = {
            "type": "execute",
            "parameters": {
                "command": f"cp -f '{orig_path}' '{new_path}'",
                "shell": True,
            },
        }

        # Oracle: pkill any vlc, write minimal vlcrc to skip privacy prompt,
        # relaunch with the new file, then xdotool press 'f' to fullscreen.
        # Mirrors eval's oracle_actions structure.
        oracle = [
            {"type": "execute", "parameters": {
                "command": (
                    "pkill -9 vlc 2>/dev/null; sleep 2\n"
                    "mkdir -p /home/user/.config/vlc\n"
                    "cat > /home/user/.config/vlc/vlcrc << 'VLCEOF'\n"
                    "[qt]\n"
                    "qt-privacy-ask=0\n"
                    "VLCEOF\n"
                    "echo done"
                ),
                "shell": True,
            }},
            {"type": "launch", "parameters": {
                "command": (
                    f"DISPLAY=:1 vlc --no-audio --no-video-title-show "
                    f"--start-time=15 '{new_path}'"
                ),
                "shell": True,
            }},
            {"type": "sleep", "parameters": {"seconds": 5}},
            {"type": "execute", "parameters": {
                "command": (
                    "WID=$(xdotool search --class vlc 2>/dev/null | head -1); "
                    "if [ -n \"$WID\" ]; then "
                    "  xdotool windowactivate $WID 2>/dev/null; sleep 1; "
                    "  xdotool key f; "
                    "fi"
                ),
                "shell": True,
            }},
            {"type": "sleep", "parameters": {"seconds": 5}},
        ]

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"fullscreen_file_name": new_name},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=copy.deepcopy(evaluator),
            perturb_config_step=perturb_config_step,
        ))

    return rows


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

_INTERNAL_FNS = [
    perturb_vlc_config,
    perturb_vlc_recordings_folder,
    perturb_vlc_slider_color,
    perturb_vlc_output_filename,
    perturb_vlc_playing,
    perturb_vlc_fullscreen,
]


def perturb_vlc_per_task(
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
