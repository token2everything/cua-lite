"""Chrome domain perturbation functions (Track B).

Structural functions: (eval_row, rng) -> list[dict]

Usage:
    from lite.gym.envs.lite.osworld.src.gen.train.perturb.chrome import CHROME_PERTURB_FNS
    rows = CHROME_PERTURB_FNS["browser_setting"](eval_row, rng)
"""

from __future__ import annotations

import base64
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

# Maps evaluator result type -> (value_pool, instruction_templates)
_BOOLEAN_SETTINGS: dict[str, dict] = {
    "enable_do_not_track": {
        "pref_key": "enable_do_not_track",
        "values": [True, False],
        "skip_false": True,  # Chrome normalizes DNT to off; "disable" variant is always trivial_pass
        "instruction_templates": [
            "{verb} the 'Do Not Track' feature in Chrome.",
            "Please {verb} Chrome's Do Not Track setting.",
            "Could you {verb} Do Not Track in my Chrome browser?",
            "I'd like to {verb} the Do Not Track option in Chrome settings.",
            "{verb} the Do Not Track privacy feature in Chrome for me.",
            # D5: multi-step (first → next → then) ~20 words
            "First launch Chrome, next dive into the privacy and security settings, and then {verb} the Do Not Track option there.",
        ],
    },
    "enable_safe_browsing": {
        "pref_key": "safebrowsing.enabled",
        "values": [True, False],
        "skip_false": True,  # Chrome normalizes SafeBrowsing back to enabled; "disable" is trivial_pass
        "instruction_templates": [
            "{verb} Chrome's Safe Browsing protection.",
            "Please {verb} the Safe Browsing feature in Chrome.",
            "Could you {verb} safe browsing in Chrome for me?",
            "I'd like to {verb} Chrome's Safe Browsing to {reason}.",
            "{verb} the safe browsing protection in Chrome.",
            # D5: long narrative pad (~38 words) — agent has to {verb} Safe Browsing
            "I keep landing on sketchy redirect pages while researching a school report and our IT person mentioned Chrome has a built-in shield for this. Could you please {verb} Chrome's Safe Browsing protection so I get warnings before any phishing or malware download sneaks through?",
        ],
    },
    "data_delete_automacally": {
        "pref_key": "clear_data_on_exit",
        "values": [True, False],
        "skip_false": True,  # Chrome ignores external writes to clear_data; "disable" is trivial_pass
        "instruction_templates": [
            "{verb} automatic deletion of browsing data when Chrome closes.",
            "Please {verb} Chrome's option to clear data on exit.",
            "Could you {verb} automatic data clearing on browser close?",
            "I want to {verb} the setting that deletes data when I close Chrome.",
            "{verb} the feature that removes browsing data on Chrome exit.",
            # D5: multi-step (first → next → after) ~22 words
            "First open Chrome's privacy controls, then scroll to the data-on-exit section, and after that {verb} the setting that wipes browsing data automatically.",
        ],
    },
}

_PROFILE_NAMES = [
    "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace",
    "Henry", "Isabella", "Jack", "Kate", "Leo", "Maria", "Noah",
    "Olivia", "Peter", "Quinn", "Rachel",
]

_PROFILE_NAME_TEMPLATES = [
    'Change my Chrome profile name to "{name}".',
    'Please update the username in Chrome profiles to "{name}".',
    'Could you set my Chrome profile name to "{name}"?',
    'I need my Chrome profile name changed to "{name}". Can you help?',
    'Update my Chrome profile display name to "{name}".',
    # D5: multi-step (first → then → after) ~22 words
    'First open Chrome\'s profile menu in the top-right corner, then choose Manage your Chrome profile, and after that rename the profile to "{name}".',
]

# NOTE: Chrome's Appearance > Font size dropdown only
# exposes 5 labels: Very small (9px) / Small (12px) / Medium (16px) /
# Large (20px) / Very large (24px). Other values can't be reached via
# UI. Use (label, pixels) tuple so instructions can name both — agents
# see the LABEL in the dropdown but eval reads the px value.
_FONT_SIZE_VALUES = [
    ("Very small", 9),
    ("Small", 12),
    ("Medium", 16),
    ("Large", 20),
    ("Very large", 24),
]

_FONT_SIZE_TEMPLATES = [
    "In Chrome's Settings → Appearance → Font size dropdown, choose '{label}' ({px} pixels).",
    "Open Chrome Settings, go to Appearance, and set the Font size dropdown to '{label}' (which equals {px} pixels).",
    "Set Chrome's font size to '{label}' (the {px}-pixel option in Settings → Appearance).",
    "Change Chrome's Appearance Font size dropdown selection to '{label}' ({px} pixels).",
    "In Chrome's Appearance settings, pick '{label}' on the Font size dropdown — this corresponds to {px} pixels.",
    # D5: multi-step (first → next → after) ~24 words
    "First open Chrome's settings, then click the Appearance tab, and after that change the Font size dropdown to '{label}' which corresponds to {px} pixels.",
]

_SEARCH_ENGINES = ["Google", "Microsoft Bing", "DuckDuckGo", "Yahoo", "Ecosia", "Brave", "Startpage"]

# Chrome's built-in search engine list may use short_names that differ slightly
# from our canonical names (e.g. "Yahoo!" vs "Yahoo"). Include known UI variants
# so the evaluator accepts the value Chrome actually writes to Preferences.
_CHROME_NAME_VARIANTS: dict[str, list[str]] = {
    "Yahoo": ["Yahoo!", "Yahoo", "Microsoft Yahoo"],
}

# Maps short_name → (keyword, search_url) for the oracle template_url_data patch.
# Engines not listed fall back to text-substitution (best-effort).
_SEARCH_ENGINE_DATA: dict[str, tuple[str, str]] = {
    "Google":        ("google.com",     "https://www.google.com/search?q={searchTerms}"),
    "Microsoft Bing": ("bing.com",      "https://www.bing.com/search?q={searchTerms}"),
    "Bing":          ("bing.com",       "https://www.bing.com/search?q={searchTerms}"),
    "DuckDuckGo":    ("duckduckgo.com", "https://duckduckgo.com/?q={searchTerms}"),
    "Yahoo":         ("yahoo.com",      "https://search.yahoo.com/search?p={searchTerms}"),
    "Ecosia":        ("ecosia.org",     "https://www.ecosia.org/search?q={searchTerms}"),
    "Brave":         ("search.brave.com", "https://search.brave.com/search?q={searchTerms}"),
    "Startpage":     ("startpage.com",  "https://www.startpage.com/search?q={searchTerms}"),
}

_SEARCH_ENGINE_TEMPLATES = [
    "Set {engine} as the default search engine in Chrome.",
    "Please change Chrome's default search engine to {engine}.",
    "Could you make {engine} my main search engine in Chrome?",
    "I'd like to switch my Chrome search engine to {engine}.",
    "Change the default search engine in Chrome to {engine}.",
    # D5: multi-step (first → next → then) ~24 words
    "First go into Chrome's settings, next find the search engine section, and then switch the default away from whatever it is now to {engine}.",
    # D5: long narrative pad (~32 words)
    "I've been getting weird ad-laden results from my current default search and a friend swore by {engine} — could you please flip Chrome's default search engine setting over to {engine} for me?",
]


def perturb_browser_setting(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate structural perturbations for Chrome config-editing tasks.

    Handles exact_match, check_font_size, and match_in_list evaluators by
    identifying the setting type from the result type, then generating variants
    with different values.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    result = evaluator.get("result", {})
    if not isinstance(result, dict):
        return []
    result_type = result.get("type", "")

    # -- Boolean Chrome prefs (exact_match) --
    if func == "exact_match" and result_type in _BOOLEAN_SETTINGS:
        return _perturb_boolean_setting(eval_row, rng, result_type)

    # -- Profile name (exact_match) --
    if func == "exact_match" and result_type == "profile_name":
        return _perturb_profile_name(eval_row, rng)

    # -- Font size (check_font_size) --
    if func == "check_font_size":
        return _perturb_font_size(eval_row, rng)

    # -- Default search engine (match_in_list) --
    if func == "match_in_list" and result_type == "default_search_engine":
        return _perturb_search_engine(eval_row, rng)

    return []


def _perturb_boolean_setting(
    eval_row: dict, rng: random.Random, result_type: str,
) -> list[dict]:
    """Generate variants for boolean Chrome settings (enable/disable)."""
    spec = _BOOLEAN_SETTINGS[result_type]
    orig_expected = eval_row["metadata"]["evaluator"]["expected"]["rules"]["expected"]
    orig_bool = orig_expected == "true"

    rows = []
    for new_bool in spec["values"]:
        if new_bool == orig_bool:
            continue
        if not new_bool and spec.get("skip_false"):
            continue  # Chrome normalizes this pref to off; "disable" variant always trivial_pass
        verb = "enable" if new_bool else "disable"
        reason = "enhance my privacy" if new_bool else "reduce restrictions"
        instruction = rng.choice(spec["instruction_templates"]).format(
            verb=verb.capitalize(), reason=reason,
        )

        new_evaluator = copy.deepcopy(eval_row["metadata"]["evaluator"])
        new_evaluator["expected"]["rules"]["expected"] = str(new_bool).lower()

        # data_delete_automacally: the runner getter checks
        # profile.default_content_setting_values.cookies == 4 — the key the real Chrome
        # "delete data on close" UI writes (live-confirmed on Chrome 149; the old
        # browser.clear_data.browsing_data_lifetime key is an enterprise policy the UI
        # never writes). Use a dedicated oracle that sets/clears that key.
        if result_type == "data_delete_automacally":
            prefs_path = "/home/user/chrome-data/Default/Preferences"
            if new_bool:
                # Enable: set profile.default_content_setting_values.cookies = 4 (clear-on-exit)
                oracle = [{"type": "execute", "parameters": {
                    "command": (
                        "python3 << 'PYEOF'\n"
                        "import json, os\n"
                        f"path = '{prefs_path}'\n"
                        "prefs = {}\n"
                        "if os.path.exists(path):\n"
                        "    with open(path) as f:\n"
                        "        prefs = json.load(f)\n"
                        "prefs.setdefault('profile', {}).setdefault('default_content_setting_values', {})['cookies'] = 4\n"
                        "with open(path, 'w') as f:\n"
                        "    json.dump(prefs, f)\n"
                        "PYEOF"
                    ),
                    "shell": True,
                }}]
            else:
                # Disable: remove the cookies key so the getter returns "false"
                oracle = [{"type": "execute", "parameters": {
                    "command": (
                        "python3 << 'PYEOF'\n"
                        "import json, os\n"
                        f"path = '{prefs_path}'\n"
                        "prefs = {}\n"
                        "if os.path.exists(path):\n"
                        "    with open(path) as f:\n"
                        "        prefs = json.load(f)\n"
                        "prefs.get('profile', {}).get('default_content_setting_values', {}).pop('cookies', None)\n"
                        "with open(path, 'w') as f:\n"
                        "    json.dump(prefs, f)\n"
                        "PYEOF"
                    ),
                    "shell": True,
                }}]
        else:
            oracle = copy.deepcopy(oracle_actions_of(eval_row))
            for action in oracle:
                cmd = action.get("parameters", {}).get("command", "")
                if isinstance(cmd, str):
                    if orig_bool and not new_bool:
                        action["parameters"]["command"] = cmd.replace("True", "False")
                    elif not orig_bool and new_bool:
                        action["parameters"]["command"] = cmd.replace("False", "True")

        # NOTE: the BASE eval_row's
        # config often contains a jq/python step that pre-writes the pref
        # to the BASE-task's expected value. When our perturbation flips
        # the eval rule (orig_bool ↔ new_bool), the perturb_config_step
        # appended AFTER Chrome launches writes the opposite-bool to the
        # Preferences file. However, Chrome has already loaded the base
        # task's pref value into memory at startup and will write it back
        # to the file during normal operation — so by the time the agent
        # acts, the file reflects Chrome's in-memory (original) value, not
        # our perturb write → agent passes vacuously (the key is already
        # in the desired final state before any UI action).
        #
        # Fix: pkill Chrome before writing so no running
        # instance can overwrite our change, then relaunch Chrome so the
        # agent sees a live window. Chrome will read the perturb-written
        # Preferences on its fresh start and present the correct initial
        # state to the agent.
        opposite_bool = not new_bool
        if result_type == "data_delete_automacally":
            # eval checks browser.clear_data.browsing_data_lifetime.enabled.
            if opposite_bool:
                seed_py = (
                    "    prefs.setdefault('browser', {}).setdefault('clear_data', {}).setdefault('browsing_data_lifetime', {})['enabled'] = True\n"
                )
            else:
                seed_py = (
                    "    prefs.get('browser', {}).get('clear_data', {}).get('browsing_data_lifetime', {}).pop('enabled', None)\n"
                )
        else:
            seed_py = (
                f"    pref_key = {spec['pref_key']!r}\n"
                f"    target = {opposite_bool}\n"
                "    parts = pref_key.split('.')\n"
                "    obj = prefs\n"
                "    for p in parts[:-1]:\n"
                "        obj = obj.setdefault(p, {})\n"
                "    obj[parts[-1]] = target\n"
            )
        perturb_config_step = {
            "type": "execute",
            "parameters": {
                "command": (
                    # Kill any running Chrome so it cannot write back its
                    # in-memory Preferences and undo our pref change.
                    "pkill -f 'google-chrome' 2>/dev/null; sleep 2; "
                    "python3 << 'PYEOF'\n"
                    "import json, os\n"
                    "for path in [\n"
                    "    '/home/user/.config/google-chrome/Default/Preferences',\n"
                    "    '/home/user/chrome-data/Default/Preferences',\n"
                    "]:\n"
                    "    if not os.path.exists(path):\n"
                    "        continue\n"
                    "    with open(path) as f:\n"
                    "        prefs = json.load(f)\n"
                    + seed_py +
                    "    with open(path, 'w') as f:\n"
                    "        json.dump(prefs, f)\n"
                    "PYEOF\n"
                    # Relaunch Chrome so the agent's first screenshot has a window.
                    "google-chrome --remote-debugging-port=1337 >/dev/null 2>&1 & "
                    "sleep 6"
                ),
                "shell": True,
            },
        }

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"setting": result_type, "value": str(new_bool).lower()},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=perturb_config_step,
        ))
    return rows


def _perturb_profile_name(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate variants for Chrome profile name changes."""
    orig_name = eval_row["metadata"]["evaluator"]["expected"]["rules"]["expected"]
    candidates = [n for n in _PROFILE_NAMES if n != orig_name]

    rows = []
    for name in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_PROFILE_NAME_TEMPLATES).format(name=name)

        # Mirror the base eval's evaluator exactly: postconfig restarts Chrome
        # so any in-memory state is flushed, then result.type=profile_name reads
        # Preferences with the chrome-data→google-chrome fallback path.
        new_evaluator = copy.deepcopy(eval_row["metadata"]["evaluator"])
        new_evaluator["expected"]["rules"]["expected"] = name

        # Validation fix: Chrome commits the profile-name input only on
        # blur/Enter; agents typing into the field and clicking outside
        # sometimes leave the field focused, so pkill kills before the name
        # propagates to Local State / Preferences. Prepend an xdotool blur step
        # (Tab+Escape) BEFORE the existing pkill so the rename is committed
        # to memory, and the subsequent pkill flushes it to disk.
        postconfig = new_evaluator.get("postconfig") or []
        blur_step = {
            "type": "execute",
            "parameters": {
                "command": "xdotool key Tab; xdotool key Escape; sleep 1",
                "shell": True,
            },
        }
        new_evaluator["postconfig"] = [blur_step] + list(postconfig)

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_name in cmd:
                action["parameters"]["command"] = cmd.replace(orig_name, name)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"profile_name": name},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


def _perturb_font_size(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate variants for Chrome font size changes."""
    orig_rules = eval_row["metadata"]["evaluator"]["expected"]["rules"]
    orig_min = orig_rules.get("min", 16)

    # Also exclude the pixel value actually written by the eval oracle, since the
    # eval uses a range rule (min < default < max) and the oracle may write a value
    # that differs from orig_min. Without this, perturb can generate the same px as
    # the eval oracle (leakage: agent completing the perturb also passes the eval).
    eval_oracle_cmds = "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in oracle_actions_of(eval_row)
        if isinstance(a.get("parameters", {}).get("command"), str)
    )
    oracle_px_match = re.search(r'"default_font_size":\s*(\d+)', eval_oracle_cmds)
    oracle_px = int(oracle_px_match.group(1)) if oracle_px_match else None

    # Exclude both the range-min and the oracle-written px.
    excluded = {orig_min, oracle_px} if oracle_px else {orig_min}
    # Each candidate is (label, px). Skip any excluded px so we always perturb.
    candidates = [(label, px) for (label, px) in _FONT_SIZE_VALUES if px not in excluded]
    rows = []
    for label, size in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_FONT_SIZE_TEMPLATES).format(label=label, px=size)

        # NOTE: the BASE eval uses
        # `check_font_size` rule_type="range" with strict `min < default
        # < max`. Setting `min=size, max=99999` lets the Chrome default
        # 16 vacuously pass when size=14 (14<16<99999) and lets size=20
        # fail when correct (20<20<99999 == False). Switch to exact-
        # equality "value" rule against the discrete dropdown size.
        new_evaluator = copy.deepcopy(eval_row["metadata"]["evaluator"])
        new_evaluator["expected"]["rules"] = {"type": "value", "value": size}

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str):
                # Use targeted regex so only the "default_font_size": N JSON key is
                # updated, not other numeric literals that happen to match orig_min.
                new_cmd = re.sub(
                    r'"default_font_size":\s*\d+',
                    f'"default_font_size": {size}',
                    cmd,
                )
                if new_cmd == cmd and str(orig_min) in cmd:
                    # Fallback: plain replace if regex didn't match (different key name)
                    new_cmd = cmd.replace(str(orig_min), str(size))
                action["parameters"]["command"] = new_cmd

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"font_size": size},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


def _perturb_search_engine(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate variants for default search engine changes."""
    orig_expected = eval_row["metadata"]["evaluator"]["expected"]["rules"]["expected"]
    orig_names = orig_expected if isinstance(orig_expected, list) else [orig_expected]

    # Pick the canonical name for the base engine to seed in chrome-data.
    # Use the first non-"Microsoft *" form when available (e.g. "Bing" not
    # "Microsoft Bing"), since that is what the oracle template_url_data uses.
    base_engine_name = next(
        (n for n in orig_names if not n.startswith("Microsoft ")),
        orig_names[0] if orig_names else "Google",
    )

    # Exclude Google as a perturb target: Chrome's factory default is Google, so the
    # config's Preferences write (to set a non-Google initial state) gets overridden by
    # Chrome on startup → pre-oracle eval already returns "Google" → trivial_pass.
    _GOOGLE_NAMES = {"Google", "Microsoft Google"}
    candidates = [e for e in _SEARCH_ENGINES if e not in orig_names and e not in _GOOGLE_NAMES]
    rows = []
    for engine in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_SEARCH_ENGINE_TEMPLATES).format(engine=engine)

        new_evaluator = copy.deepcopy(eval_row["metadata"]["evaluator"])
        # Include "Microsoft <engine>" prefix variant since Chrome sometimes uses it.
        # Also include known UI-form variants (e.g. "Yahoo!" for "Yahoo").
        new_evaluator["expected"]["rules"]["expected"] = _CHROME_NAME_VARIANTS.get(
            engine, [engine, f"Microsoft {engine}"]
        )

        # NOTE: replace inherited pkill-only postconfig with
        # an execute+sleep variant. Chrome writes Preferences asynchronously; the
        # original "type: launch" postconfig only waits 2 seconds (dispatch_action
        # launch sleep), which is not enough for Chrome to flush Preferences on SIGTERM.
        # Using "type: execute" with an explicit sleep 8 ensures the Preferences file
        # is written before the evaluator reads it.
        new_evaluator["postconfig"] = [
            {
                "type": "execute",
                "parameters": {
                    "command": "pkill chrome 2>/dev/null; sleep 8; true",
                    "shell": True,
                },
            }
        ]

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        # Kill Chrome before writing Preferences so Chrome can't overwrite our changes.
        # Config launches Chrome; if Chrome is running when we write Preferences it
        # detects the change and immediately overwrites with its in-memory state (the
        # original engine), causing score=0. The evaluator's postconfig restarts Chrome
        # which then reads our oracle-written Preferences correctly.
        oracle.insert(0, {"type": "execute", "parameters": {
            "command": "pkill -f 'google-chrome' || true", "shell": True,
        }})
        if engine in _SEARCH_ENGINE_DATA:
            # Patch the oracle to write the correct keyword and URL for this engine.
            # Simple name substitution leaves stale keyword/url fields, which Chrome
            # may use to normalize short_name back to the original engine on relaunch.
            keyword, url = _SEARCH_ENGINE_DATA[engine]
            for action in oracle:
                cmd = action.get("parameters", {}).get("command", "")
                if isinstance(cmd, str) and "template_url_data" in cmd:
                    # Replace entire template_url_data dict using a regex so that
                    # keyword and url are always correct regardless of orig values.
                    new_tud = json.dumps({
                        "short_name": engine,
                        "keyword": keyword,
                        "url": url,
                    })
                    cmd = re.sub(
                        r'"template_url_data":\s*\{(?:[^{}]|\{[^{}]*\})*\}',
                        f'"template_url_data": {new_tud}',
                        cmd,
                    )
                    action["parameters"]["command"] = cmd
        else:
            # Fallback: text-substitute the engine name in the oracle
            for action in oracle:
                cmd = action.get("parameters", {}).get("command", "")
                if isinstance(cmd, str):
                    for old_name in orig_names:
                        if old_name and old_name in cmd:
                            cmd = cmd.replace(old_name, engine)
                    action["parameters"]["command"] = cmd

        # NOTE: seed the BASE engine into chrome-data and
        # relaunch Chrome with --user-data-dir=/home/user/chrome-data so
        # that Chrome, the agent, and the LOCAL evaluator all read/write
        # the same Preferences file. The local eval at
        # `lite/gym/envs/lite/osworld/src/eval/runner.py:830-832`
        # reads chrome-data FIRST, falling back to the default profile,
        # so agent action on chrome-data is what eval sees.
        if base_engine_name in _SEARCH_ENGINE_DATA:
            base_kw, base_url = _SEARCH_ENGINE_DATA[base_engine_name]
        else:
            base_kw, base_url = base_engine_name.lower().replace(" ", ".") + ".com", ""
        base_tud = json.dumps({
            "short_name": base_engine_name,
            "keyword": base_kw,
            "url": base_url,
        })
        perturb_config_step = {
            "type": "execute",
            "parameters": {
                "command": (
                    # Kill any running Chrome so it can't overwrite our Preferences write.
                    "pkill -f 'google-chrome' 2>/dev/null; sleep 2; "
                    "python3 << 'PYEOF'\n"
                    "import json, os\n"
                    "path = '/home/user/chrome-data/Default/Preferences'\n"
                    "os.makedirs(os.path.dirname(path), exist_ok=True)\n"
                    "prefs = {}\n"
                    "if os.path.exists(path):\n"
                    "    with open(path) as f:\n"
                    "        prefs = json.load(f)\n"
                    f"prefs['default_search_provider_data'] = {{'template_url_data': {base_tud}}}\n"
                    "with open(path, 'w') as f:\n"
                    "    json.dump(prefs, f)\n"
                    "PYEOF\n"
                    # Relaunch Chrome with chrome-data so the agent and evaluator share
                    # the same Preferences file path.
                    "google-chrome --user-data-dir=/home/user/chrome-data "
                    "--remote-debugging-port=1337 >/dev/null 2>&1 & "
                    "sleep 6"
                ),
                "shell": True,
            },
        }

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"search_engine": engine},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=perturb_config_step,
        ))
    return rows


# ---------------------------------------------------------------------------
# bookmark_folder: perturb bookmark folder creation tasks
# ---------------------------------------------------------------------------

_BOOKMARK_FOLDER_NAMES = [
    "Work", "Reading", "Shopping", "Travel", "News", "Research",
    "Recipes", "Music", "Movies", "Sports", "Finance", "Education",
]

_BOOKMARK_FOLDER_TEMPLATES = [
    "Create a new folder called '{name}' on the bookmarks bar in Chrome.",
    "Please add a folder named '{name}' to the Chrome bookmarks bar.",
    "Could you make a new bookmark folder '{name}' on the bookmarks bar?",
    "I'd like a new folder called '{name}' on my Chrome bookmarks bar.",
    "Add a folder named '{name}' to the bookmarks bar in my browser.",
    # D5: multi-step (first → next → then) ~22 words
    "First open Chrome's bookmark manager, next right-click on the bookmarks bar, and then create a fresh folder named '{name}' there.",
]


def perturb_bookmark_folder(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for Chrome bookmark folder creation tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "is_expected_bookmarks":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    bk_type = rules.get("type", "")
    if bk_type != "bookmark_bar_folders_names":
        return []

    names = rules.get("names", [])
    orig_name = names[0] if names else ""
    if not orig_name:
        return []

    candidates = [n for n in _BOOKMARK_FOLDER_NAMES if n != orig_name]
    rows = []
    for new_name in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_BOOKMARK_FOLDER_TEMPLATES).format(name=new_name)

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["names"] = [new_name]

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_name in cmd:
                action["parameters"]["command"] = cmd.replace(orig_name, new_name)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"folder_name": new_name},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# startup_page: perturb startup page tasks
# ---------------------------------------------------------------------------

# NOTE: get_new_startup_page returns "true"
# only when restore_on_startup==5 (open New Tab Page) — NOT mode 4
# (specific URL). The previous template asked agents to set a specific
# URL, which always failed the eval. The BASE task's actual intent is
# "remove auto-open URL on launch" — so the perturb must preserve
# that intent and only vary the SEEDED nuisance URL the agent has to
# clear. The eval (and oracle) keeps mode 5 == NTP.
_STARTUP_NUISANCE_URLS = [
    "funbrain.com", "fakegamingsite.net", "annoyingads.example",
    "popups.local", "homestartup.dev", "mybadhomepage.io",
    "spammy-default.test", "weirdstartpage.com",
]

_STARTUP_TEMPLATES = [
    "Whenever I launch Chrome it always opens \"{site}\". I don't want this. Make Chrome open the New Tab Page on startup instead.",
    "Chrome auto-opens \"{site}\" every time I start it — please switch the startup setting to \"Open the New Tab page\".",
    "Could you stop Chrome from auto-loading \"{site}\" on launch? Configure it to open a fresh New Tab Page on startup.",
    "I cleared my cache but Chrome still opens \"{site}\" on startup. Please change Chrome to start with the New Tab Page.",
    "Chrome keeps launching with \"{site}\" — set it to open the New Tab Page on startup so this stops happening.",
    # D5: multi-step (first → then → after) ~28 words
    "First open Chrome's settings page, then scroll down to the On startup section, and after that change the option from opening \"{site}\" to opening a fresh New Tab Page.",
]


def perturb_startup_page(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for Chrome startup page tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    result = evaluator.get("result", {})
    if not isinstance(result, dict):
        return []
    result_type = result.get("type", "")

    if func != "exact_match" or result_type != "new_startup_page":
        return []

    rows = []
    for url in rng.sample(_STARTUP_NUISANCE_URLS, min(4, len(_STARTUP_NUISANCE_URLS))):
        instruction = rng.choice(_STARTUP_TEMPLATES).format(site=url)

        new_evaluator = copy.deepcopy(evaluator)
        # Eval unchanged: still requires restore_on_startup == 5
        # Validation fix: add postconfig pkill+restart so any agent-induced
        # in-memory pref change is flushed to disk before eval reads
        # session.restore_on_startup. Mirrors _perturb_profile_name pattern.
        new_evaluator["postconfig"] = [
            {"type": "launch", "parameters": {"command": ["pkill", "chrome"]}},
            {"type": "launch", "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]}},
            {"type": "sleep", "parameters": {"seconds": 3}},
        ]

        # Oracle clears nuisance URLs and sets mode 5 (NTP).
        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and "startup_urls" in cmd:
                action["parameters"]["command"] = re.sub(
                    r'("startup_urls":\s*)\[.*?\]', '\\1[]', cmd,
                )

        # Seed the nuisance URL + restore_on_startup=4 AFTER the base eval's
        # config (which itself rewrites Preferences via jq to funbrain.com and
        # launches Chrome). We need to pkill Chrome first so the in-memory
        # prefs don't clobber our file write on a subsequent flush, then
        # rewrite Preferences with our nuisance URL, then relaunch Chrome so
        # turn_00 starts with the nuisance URL the instruction names.
        # Using `perturb_config_step` (appended) ensures we run last; if we
        # used `pre_config_steps` the base jq step would overwrite us before
        # Chrome reads Preferences (REFERENT_MISMATCH = funbrain.com visible).
        perturb_step = {
            "type": "execute",
            "parameters": {
                "command": (
                    "pkill chrome 2>/dev/null; sleep 1; "
                    "python3 << 'PYEOF'\n"
                    "import json, os\n"
                    "for path in [\n"
                    "    '/home/user/.config/google-chrome/Default/Preferences',\n"
                    "    '/home/user/chrome-data/Default/Preferences',\n"
                    "]:\n"
                    "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
                    "    prefs = {}\n"
                    "    if os.path.exists(path):\n"
                    "        with open(path) as f:\n"
                    "            prefs = json.load(f)\n"
                    "    sess = prefs.setdefault('session', {})\n"
                    "    sess['restore_on_startup'] = 4\n"
                    f"    sess['startup_urls'] = ['https://{url}']\n"
                    "    with open(path, 'w') as f:\n"
                    "        json.dump(prefs, f)\n"
                    "PYEOF\n"
                    "google-chrome --remote-debugging-port=1337 >/dev/null 2>&1 & "
                    "sleep 3"
                ),
                "shell": True,
            },
        }

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"startup_url": url},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=perturb_step,
        ))

    return rows


# ---------------------------------------------------------------------------
# history_keyword: perturb browsing history deletion tasks
# ---------------------------------------------------------------------------

_HISTORY_KEYWORDS = ["youtube", "facebook", "twitter", "reddit", "amazon", "instagram", "linkedin", "pinterest"]

_HISTORY_TEMPLATES = [
    "Delete all {keyword} entries from my Chrome browsing history.",
    "Please remove all {keyword} related history from Chrome.",
    "Could you clear my browsing history of any {keyword} visits?",
    "I want to remove all {keyword} entries from Chrome history.",
    "Help me delete {keyword} from my Chrome browsing history.",
    # D5: multi-step (once → then) ~18 words
    "Once Chrome is open, head over to the history page and then clear every entry that mentions {keyword}.",
    # D5: multi-step (first → next → after) ~22 words
    "First open Chrome's history viewer, next filter the list for {keyword}, and after that delete each one of those matching visits.",
]

# Base browsing history seeded into Chrome before the agent's first screenshot.
# This is the canonical youtube/news/etc. history the base eval task provides.
_BASE_HISTORY_STEP: dict = {
    "type": "update_browse_history",
    "parameters": {
        "history": [
            {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)", "visit_time_from_now_in_seconds": 3600},
            {"url": "https://www.youtube.com/watch?v=9bZkp7q19f0", "title": "PSY - GANGNAM STYLE(강남스타일) M/V", "visit_time_from_now_in_seconds": 1631},
            {"url": "https://www.youtube.com/watch?v=3tmd-ClpJxA", "title": "Maroon 5 - Sugar (Official Music Video)", "visit_time_from_now_in_seconds": 900},
            {"url": "https://www.nytimes.com/", "title": "The New York Times", "visit_time_from_now_in_seconds": 300},
            {"url": "https://www.youtube.com/watch?v=OPf0YbXqDm0", "title": "Ed Sheeran - Shape of You [Official Music Video]", "visit_time_from_now_in_seconds": 1200},
            {"url": "https://www.youtube.com/watch?v=JGwWNGJdvx8", "title": "Taylor Swift - Shake It Off", "visit_time_from_now_in_seconds": 2400},
            {"url": "https://www.bbc.co.uk/", "title": "BBC", "visit_time_from_now_in_seconds": 1500},
            {"url": "https://www.youtube.com/watch?v=2Vv-BfVoq4g", "title": "Adele - Hello", "visit_time_from_now_in_seconds": 1800},
            {"url": "https://www.youtube.com/watch?v=YQHsXMglC9A", "title": "Katy Perry - Roar (Official Music Video)", "visit_time_from_now_in_seconds": 2100},
            {"url": "https://www.cnn.com/", "title": "CNN", "visit_time_from_now_in_seconds": 2700},
            {"url": "https://www.youtube.com/watch?v=ru0K8uYEZWw", "title": "Justin Bieber - Baby ft. Ludacris (Official Music Video)", "visit_time_from_now_in_seconds": 3200},
            {"url": "https://www.youtube.com/watch?v=9bZkp7q19f0", "title": "PSY - GANGNAM STYLE(강남스타일) M/V", "visit_time_from_now_in_seconds": 3700},
            {"url": "https://www.nationalgeographic.com/", "title": "National Geographic", "visit_time_from_now_in_seconds": 4000},
            {"url": "https://www.youtube.com/watch?v=OPf0YbXqDm0", "title": "Ed Sheeran - Shape of You [Official Music Video]", "visit_time_from_now_in_seconds": 4300},
            {"url": "https://www.youtube.com/watch?v=JGwWNGJdvx8", "title": "Taylor Swift - Shake It Off", "visit_time_from_now_in_seconds": 4700},
            {"url": "https://www.bbc.co.uk/", "title": "BBC", "visit_time_from_now_in_seconds": 5000},
            {"url": "https://www.youtube.com/watch?v=2Vv-BfVoq4g", "title": "Adele - Hello", "visit_time_from_now_in_seconds": 5300},
            {"url": "https://www.youtube.com/watch?v=YQHsXMglC9A", "title": "Katy Perry - Roar (Official Music Video)", "visit_time_from_now_in_seconds": 5600},
            {"url": "https://www.cnn.com/", "title": "CNN", "visit_time_from_now_in_seconds": 5900},
            {"url": "https://www.youtube.com/watch?v=ru0K8uYEZWw", "title": "Justin Bieber - Baby ft. Ludacris (Official Music Video)", "visit_time_from_now_in_seconds": 6300},
            {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)", "visit_time_from_now_in_seconds": 6700},
            {"url": "https://www.nationalgeographic.com/", "title": "National Geographic", "visit_time_from_now_in_seconds": 7000},
            {"url": "https://www.youtube.com/watch?v=OPf0YbXqDm0", "title": "Ed Sheeran - Shape of You [Official Music Video]", "visit_time_from_now_in_seconds": 7300},
            {"url": "https://www.youtube.com/watch?v=JGwWNGJdvx8", "title": "Taylor Swift - Shake It Off", "visit_time_from_now_in_seconds": 7600},
            {"url": "https://www.bbc.co.uk/", "title": "BBC", "visit_time_from_now_in_seconds": 7900},
            {"url": "https://www.youtube.com/watch?v=2Vv-BfVoq4g", "title": "Adele - Hello", "visit_time_from_now_in_seconds": 8200},
            {"url": "https://www.youtube.com/watch?v=YQHsXMglC9A", "title": "Katy Perry - Roar (Official Music Video)", "visit_time_from_now_in_seconds": 8500},
            {"url": "https://www.cnn.com/", "title": "CNN", "visit_time_from_now_in_seconds": 8800},
            {"url": "https://www.youtube.com/watch?v=ru0K8uYEZWw", "title": "Justin Bieber - Baby ft. Ludacris (Official Music Video)", "visit_time_from_now_in_seconds": 9100},
            {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)", "visit_time_from_now_in_seconds": 9400},
            {"url": "https://www.nationalgeographic.com/", "title": "National Geographic", "visit_time_from_now_in_seconds": 9700},
            {"url": "https://www.youtube.com/watch?v=OPf0YbXqDm0", "title": "Ed Sheeran - Shape of You [Official Music Video]", "visit_time_from_now_in_seconds": 10000},
            {"url": "https://www.youtube.com/watch?v=JGwWNGJdvx8", "title": "Taylor Swift - Shake It Off", "visit_time_from_now_in_seconds": 10300},
            {"url": "https://www.bbc.co.uk/", "title": "BBC", "visit_time_from_now_in_seconds": 10600},
            {"url": "https://www.youtube.com/watch?v=2Vv-BfVoq4g", "title": "Adele - Hello", "visit_time_from_now_in_seconds": 10900},
            {"url": "https://www.youtube.com/watch?v=YQHsXMglC9A", "title": "Katy Perry - Roar (Official Music Video)", "visit_time_from_now_in_seconds": 11200},
            {"url": "https://www.cnn.com/", "title": "CNN", "visit_time_from_now_in_seconds": 11500},
            {"url": "https://www.youtube.com/watch?v=ru0K8uYEZWw", "title": "Justin Bieber - Baby ft. Ludacris (Official Music Video)", "visit_time_from_now_in_seconds": 11800},
            {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)", "visit_time_from_now_in_seconds": 12100},
            {"url": "https://www.nationalgeographic.com/", "title": "National Geographic", "visit_time_from_now_in_seconds": 12400},
        ],
    },
}

# Corrected pkill command: avoids pkill -9 -f chrome which can kill the shell process.
_PKILL_CHROME_CMD = "pkill -9 chrome 2>/dev/null; pkill -9 google-chrome 2>/dev/null; sleep 2; true"


def perturb_history_keyword(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for Chrome history deletion tasks.

    NOTE: prior version only swapped the eval keyword +
    oracle pattern, but left the seeded `update_browse_history` step
    untouched. Seed has youtube/news URLs only → eval for non-youtube
    target keyword (reddit, facebook, ...) returns 1 unconditionally —
    do-nothing agent passes (SFT poison). Now also rewrites a setup
    `update_browse_history` step's URL list to inject {new_keyword}.com
    entries the agent must actually delete.

    NOTE: also transform the eval row config:
    - Insert base update_browse_history before Chrome launch (pre_config_steps)
    - Remove the sleep step between socat launch and pkill
    - Fix pkill command (pkill -9 -f chrome → pkill -9 chrome separately)
    - Replace base python3 sqlite step with one that also writes new_keyword entries
    """
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "check_history_deleted":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    keywords = rules.get("keywords", [])
    orig_keyword = keywords[0] if keywords else ""
    if not orig_keyword:
        return []

    candidates = [k for k in _HISTORY_KEYWORDS if k != orig_keyword]
    rows = []
    for new_keyword in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_HISTORY_TEMPLATES).format(keyword=new_keyword)

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["keywords"] = [new_keyword]

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_keyword in cmd:
                action["parameters"]["command"] = cmd.replace(orig_keyword, new_keyword)

        # Build new_keyword history seed entries (visit_count and days_ago are
        # drawn from rng before we sample them into the seed step below).
        seed_entries = [
            {
                "url": f"https://{new_keyword}.com/page{i+1}",
                "title": f"{new_keyword} page {i+1}",
                "visit_count": rng.randint(1, 5),
                "days_ago": rng.randint(0, 30),
            }
            for i in range(3)
        ]
        history_seed = {
            "type": "update_browse_history",
            "parameters": {"entries": seed_entries},
        }

        # NOTE: transform the eval row's own config to:
        # 1. Remove the sleep step that follows the socat launch
        # 2. Replace the old pkill -9 -f chrome command with the safe variant
        # 3. Replace the old python3 sqlite step with a new one that writes
        #    the new_keyword entries (with per-URL visit_count/days_ago)
        #    in addition to the base youtube entry.
        targets_repr = repr(
            [(e["url"], e["title"], e["visit_count"], e["days_ago"]) for e in seed_entries]
        )
        new_sqlite_cmd = (
            "python3 << 'HEOF'\n"
            "import sqlite3, datetime\n"
            "path = '/home/user/chrome-data/Default/History'\n"
            "now = datetime.datetime.now()\n"
            "epoch = datetime.datetime(1601,1,1)\n"
            "def cts(days=0, secs=0):\n"
            "    return int(((now - datetime.timedelta(days=days, seconds=secs)) - epoch).total_seconds() * 1e6)\n"
            "conn = sqlite3.connect(path)\n"
            "c = conn.cursor()\n"
            f"targets = {targets_repr}\n"
            "for url, title, vc, da in targets:\n"
            "    ts = cts(days=da)\n"
            "    c.execute('INSERT OR IGNORE INTO urls (url,title,visit_count,typed_count,last_visit_time,hidden) VALUES (?,?,?,0,?,0)', (url, title, vc, ts))\n"
            "    uid = c.lastrowid\n"
            "    if uid: c.execute('INSERT INTO visits (url,visit_time,from_visit,transition,segment_id,visit_duration) VALUES (?,?,0,805306368,0,0)', (uid, ts))\n"
            "ts = cts()\n"
            "c.execute('INSERT OR IGNORE INTO urls (url,title,visit_count,typed_count,last_visit_time,hidden) VALUES (?,?,1,0,?,0)', ('https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'Rick Astley - Never Gonna Give You Up', ts))\n"
            "uid = c.lastrowid\n"
            "if uid: c.execute('INSERT INTO visits (url,visit_time,from_visit,transition,segment_id,visit_duration) VALUES (?,?,0,805306368,0,0)', (uid, ts))\n"
            "conn.commit(); conn.close()\n"
            "print('ok')\n"
            "HEOF"
        )

        # Identify the index of socat launch so we can drop only the sleep
        # that immediately follows it (between socat and pkill), not the
        # final sleep after Chrome relaunch.
        raw_config = eval_row["metadata"]["config"]
        socat_idx = next(
            (i for i, s in enumerate(raw_config)
             if s.get("type") == "launch"
             and isinstance(s.get("parameters", {}).get("command"), list)
             and any("socat" in str(c) for c in s["parameters"]["command"])),
            -1,
        )
        fixed_config = []
        for idx, step in enumerate(raw_config):
            step = copy.deepcopy(step)
            stype = step.get("type", "")
            cmd = step.get("parameters", {}).get("command", "")
            # Remove only the sleep that follows the socat launch (between socat and pkill)
            if stype == "sleep" and socat_idx >= 0 and idx == socat_idx + 1:
                continue
            # Fix pkill command
            if stype == "execute" and isinstance(cmd, str) and "pkill" in cmd and "chrome" in cmd and "python3" not in cmd:
                step["parameters"]["command"] = _PKILL_CHROME_CMD
            # Replace old python3 sqlite step with new version
            elif stype == "execute" and isinstance(cmd, str) and "sqlite3" in cmd and "History" in cmd:
                step["parameters"]["command"] = new_sqlite_cmd
            fixed_config.append(step)

        fixed_eval_row = copy.deepcopy(eval_row)
        fixed_eval_row["metadata"]["config"] = fixed_config

        rows.append(make_perturb_row(
            eval_row=fixed_eval_row,
            knob_assignment={"history_keyword": new_keyword},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            pre_config_steps=[history_seed, _BASE_HISTORY_STEP],
        ))

    return rows


# ---------------------------------------------------------------------------
# desktop_shortcut: perturb desktop shortcut creation tasks
# ---------------------------------------------------------------------------

_SHORTCUT_NAMES = [
    "Play Puzzle Game 2048", "Google Maps", "Wikipedia",
    "YouTube Music", "Weather App", "Calculator Online",
    "Gmail Inbox", "Translate",
]

_SHORTCUT_TEMPLATES = [
    'Create a desktop shortcut named "{name}" for the current page in Chrome.',
    'Please make a shortcut on my desktop called "{name}" for the page Chrome is showing right now.',
    'Could you create a desktop shortcut "{name}" for this website?',
    'I need a desktop shortcut named "{name}" for this page.',
    'Set up a desktop shortcut called "{name}" for whatever Chrome currently has open.',
    # D5: multi-step (first → then → once) ~24 words
    'First open Chrome\'s three-dot menu, then dive into More tools and pick Create shortcut, and once the dialog appears name the shortcut "{name}".',
]


def perturb_desktop_shortcut(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for Chrome desktop shortcut tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "is_shortcut_on_desktop":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    orig_name = rules.get("name", "")
    if not orig_name:
        return []

    candidates = [n for n in _SHORTCUT_NAMES if n != orig_name]
    rows = []
    for new_name in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_SHORTCUT_TEMPLATES).format(name=new_name)

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["name"] = new_name

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_name in cmd:
                action["parameters"]["command"] = cmd.replace(orig_name, new_name)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"shortcut_name": new_name},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# Archetype B — Bookmark URL (eval base 7a5a7856)
#
# Eval `is_expected_bookmarks` rules.urls = [target_url] checks the Bookmarks
# JSON's bookmark_bar children for an exact set of URLs. Perturb retargets
# each variant at a different `target_url` (none equal to the base eval's
# URL `https://jalammar.github.io/illustrated-transformer/`) and seeds the
# matching tab(s) so the agent's only job is the Ctrl+D bookmark action.
# ---------------------------------------------------------------------------

# (target_url, distractor_extra_open_tab_or_None) tuples. distractor lets the
# agent practice picking the *active* tab among ≥2 open tabs (mirroring the
# eval base which opened 2 tabs).
_BOOKMARK_URL_VARIANTS: list[tuple[str, str | None]] = [
    ("https://www.python.org/", "https://docs.python.org/3/"),
    ("https://en.wikipedia.org/wiki/Linux", None),
    ("https://github.com/python/cpython", "https://github.com/torvalds/linux"),
    ("https://stackoverflow.com/help/asking", None),
]

_BOOKMARK_URL_INSTRUCTION_TEMPLATES = [
    "Save the page I'm currently looking at ({url}) to Chrome's bookmarks bar so I can find it again later.",
    "Please bookmark the active tab ({url}) onto the bookmarks bar in Chrome — keep only this one URL there.",
    "In Chrome, add the page {url} to the bookmarks bar so I can revisit it from any new window I open.",
    "Could you put {url} on Chrome's bookmarks bar? It's the page I have open and I want a one-click link to it.",
    "I want quick access to {url}; please bookmark this currently-open page on the Chrome bookmarks bar.",
    # D5: multi-step (first → then → after) ~26 words
    "First click on the active Chrome tab showing {url}, then press Ctrl+D to open the bookmark dialog, and after that pin the bookmark to the bookmarks bar.",
    # D5: long narrative pad (~38 words)
    "I keep losing track of {url} — it's a reference I come back to all the time and digging through history is such a chore. Could you please pin it to Chrome's bookmarks bar so it's just one click away?",
]


def _build_bookmark_url_oracle(target_url: str) -> list[dict]:
    """Write a Bookmarks JSON whose bookmark_bar.children == [{url-only entry}]."""
    bookmarks = {
        "checksum": "",
        "roots": {
            "bookmark_bar": {
                "children": [
                    {"id": "11", "name": target_url, "type": "url", "url": target_url}
                ],
                "id": "1", "name": "Bookmarks bar", "type": "folder",
            },
            "other": {"children": [], "id": "1000", "name": "Other bookmarks", "type": "folder"},
            "synced": {"children": [], "id": "2000", "name": "Mobile bookmarks", "type": "folder"},
        },
        "version": 1,
    }
    bm_b64 = base64.b64encode(json.dumps(bookmarks).encode()).decode()
    return [{"type": "execute", "parameters": {
        "command": (
            "python3 << 'PYEOF'\n"
            "import base64, os\n"
            "os.makedirs('/home/user/chrome-data/Default', exist_ok=True)\n"
            f"content = base64.b64decode('{bm_b64}').decode()\n"
            "open('/home/user/chrome-data/Default/Bookmarks', 'w').write(content)\n"
            "PYEOF"
        ),
        "shell": True,
    }}]


def _build_bookmark_url_perturb_config(target_url: str, extras: list[str]) -> dict:
    """pkill chrome → seed empty Bookmarks JSON → relaunch with target tabs."""
    urls_to_open = [target_url] + extras
    # Activate target_url last so it becomes the active tab.
    # chrome_open_tabs activates the LAST URL passed.
    if urls_to_open[-1] != target_url:
        urls_to_open = extras + [target_url]
    urls_repr = json.dumps(urls_to_open)
    return {
        "type": "execute",
        "parameters": {
            "command": (
                "pkill -f 'google-chrome' 2>/dev/null; sleep 2; "
                "python3 << 'PYEOF'\n"
                "import json, os\n"
                "os.makedirs('/home/user/chrome-data/Default', exist_ok=True)\n"
                "empty = {'checksum': '', 'roots': {'bookmark_bar': {'children': [], "
                "'id': '1', 'name': 'Bookmarks bar', 'type': 'folder'}, "
                "'other': {'children': [], 'id': '1000', 'name': 'Other bookmarks', 'type': 'folder'}, "
                "'synced': {'children': [], 'id': '2000', 'name': 'Mobile bookmarks', 'type': 'folder'}}, "
                "'version': 1}\n"
                "with open('/home/user/chrome-data/Default/Bookmarks', 'w') as f:\n"
                "    json.dump(empty, f)\n"
                "PYEOF\n"
                "google-chrome --user-data-dir=/home/user/chrome-data "
                "--remote-debugging-port=1337 "
                f"{' '.join(urls_to_open)} >/dev/null 2>&1 & "
                "sleep 6"
            ),
            "shell": True,
        },
    }


def perturb_bookmark_url(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for the bookmark-URL eval base (7a5a7856)."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "is_expected_bookmarks":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    if rules.get("type") != "bookmark_bar_websites_urls":
        return []

    orig_urls = set(rules.get("urls", []))
    candidates = [(u, e) for (u, e) in _BOOKMARK_URL_VARIANTS if u not in orig_urls]
    rows = []
    for target_url, extra in candidates:
        instruction = rng.choice(_BOOKMARK_URL_INSTRUCTION_TEMPLATES).format(url=target_url)

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["urls"] = [target_url]
        # Relaxed postconfig: pkill+sleep 8 then relaunch (validation H3 timing).
        new_evaluator["postconfig"] = [
            {"type": "execute", "parameters": {
                "command": "pkill chrome 2>/dev/null; sleep 8; true", "shell": True,
            }},
        ]

        oracle = _build_bookmark_url_oracle(target_url)
        extras = [extra] if extra else []
        perturb_config_step = _build_bookmark_url_perturb_config(target_url, extras)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"bookmark_url": target_url},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=perturb_config_step,
        ))

    return rows


# ---------------------------------------------------------------------------
# Archetype P — Preferences keys (eval bases 030eeff7 / 9656a811 / 99146c54
#                                 / 93eabf48)
#
# Each axis has 4 paraphrased instructions. Strategy: always perturb to the
# eval target value (true / 1=light) and seed the OPPOSITE state via
# perturb_config_step. Chrome resets true→false but not false→true (and not
# remove_key→add_key for browsing_data_lifetime), so this direction is stable.
# ---------------------------------------------------------------------------

# Each task's paraphrased instructions (4 each — polite/imperative mix).
_PREFS_TASKS: list[dict] = [
    {
        "axis": "dnt",
        "base_tid_suffix": "030eeff7",
        "pref_path": "enable_do_not_track",
        "seed_strategy": "set_false",
        "instructions": [
            "Please switch on the Do Not Track header in Chrome so my browsing isn't tracked across the sites I visit.",
            "Turn on Chrome's Do Not Track signal so the websites I visit are asked to respect my privacy preferences.",
            "Enable the Do Not Track option in Chrome's privacy settings so each request advertises my no-track preference.",
            "Activate Chrome's Do Not Track request on each page load by flipping the privacy toggle in Settings.",
            # D5: long narrative pad (~42 words)
            "I've been reading about how third-party trackers stitch together a profile of your browsing across every site you visit, and honestly the whole thing creeps me out — could you please open Chrome and turn on the Do Not Track request in the privacy settings for me?",
        ],
    },
    {
        "axis": "safe_browsing",
        "base_tid_suffix": "9656a811",
        "pref_path": "safebrowsing.enabled",
        "seed_strategy": "set_false",
        "instructions": [
            "Could you turn Safe Browsing back on in Chrome so I get warnings whenever I visit risky or phishing sites?",
            "Enable Chrome's Safe Browsing feature in the security settings for protection against unsafe and malicious pages.",
            "Switch on Safe Browsing in Chrome's security settings so the browser alerts me about dangerous URLs and downloads.",
            "Re-activate the Safe Browsing protection toggle in Chrome's privacy and security settings panel for me.",
            # D5: multi-step (first → next → after) ~22 words
            "First open Chrome, next click into the privacy and security panel, and after that flip the Safe Browsing toggle back on so warnings show up.",
        ],
    },
    # NOTE: dropped "clear_on_exit" (99146c54) — Chrome's
    # browsing_data_lifetime is configured via enterprise policy (not the UI),
    # so the agent has no way to flip it from the Settings panel. Confirmed
    # agent-ceiling per previous rollout validation uniform-zero.
    {
        "axis": "color_scheme",
        "base_tid_suffix": "93eabf48",
        "pref_path": "browser.theme.color_scheme",
        "seed_strategy": "set_dark",
        "instructions": [
            "Please switch Chrome out of dark mode and into the light color scheme — dark backgrounds are hard to read.",
            "Turn off Chrome's dark theme and use the light appearance instead in the appearance section of Settings.",
            "Set Chrome's appearance to the light color scheme — dark mode strains my eyes during long browsing sessions.",
            "Change Chrome's interface from dark mode to light mode in the theme settings so the UI uses light backgrounds.",
            # D5: multi-step (first → next → after) ~24 words
            "First open Chrome's settings, then jump into the Appearance section, and after that switch the color scheme out of dark mode and over to the light theme.",
        ],
    },
]


def _build_prefs_seed_step(pref_path: str, seed_strategy: str) -> dict:
    """pkill chrome → seed Preferences with the OPPOSITE value → relaunch."""
    if seed_strategy == "set_false":
        seed_py = (
            f"    pref_key = {pref_path!r}\n"
            "    parts = pref_key.split('.')\n"
            "    obj = prefs\n"
            "    for p in parts[:-1]:\n"
            "        obj = obj.setdefault(p, {})\n"
            "    obj[parts[-1]] = False\n"
        )
    elif seed_strategy == "remove_key":
        # Remove the nested key entirely so Chrome's exact_match returns "false".
        seed_py = (
            f"    pref_key = {pref_path!r}\n"
            "    parts = pref_key.split('.')\n"
            "    obj = prefs\n"
            "    for p in parts[:-1]:\n"
            "        if not isinstance(obj.get(p), dict): obj = None; break\n"
            "        obj = obj[p]\n"
            "    if obj is not None:\n"
            "        obj.pop(parts[-1], None)\n"
        )
    elif seed_strategy == "set_dark":
        # color_scheme: 1=light, 2=dark. Seed dark=2 + color_scheme2=2 (mirrors
        # eval base 93eabf48's config script).
        seed_py = (
            "    prefs.setdefault('browser', {}).setdefault('theme', {})['color_scheme'] = 2\n"
            "    prefs['browser']['theme']['color_scheme2'] = 2\n"
        )
    else:
        raise ValueError(f"unknown seed_strategy={seed_strategy}")
    return {
        "type": "execute",
        "parameters": {
            "command": (
                "pkill -f 'google-chrome' 2>/dev/null; sleep 2; "
                "python3 << 'PYEOF'\n"
                "import json, os\n"
                "for path in [\n"
                "    '/home/user/.config/google-chrome/Default/Preferences',\n"
                "    '/home/user/chrome-data/Default/Preferences',\n"
                "]:\n"
                "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
                "    prefs = {}\n"
                "    if os.path.exists(path):\n"
                "        with open(path) as f:\n"
                "            prefs = json.load(f)\n"
                + seed_py +
                "    with open(path, 'w') as f:\n"
                "        json.dump(prefs, f)\n"
                "PYEOF\n"
                "google-chrome --user-data-dir=/home/user/chrome-data "
                "--remote-debugging-port=1337 >/dev/null 2>&1 & "
                "sleep 6"
            ),
            "shell": True,
        },
    }


def _build_prefs_oracle(pref_path: str, axis: str) -> list[dict]:
    """Write the eval-target value into Preferences (matches base eval target)."""
    if axis == "dnt":
        body = (
            f"    pref_key = {pref_path!r}\n"
            "    prefs[pref_key] = True\n"
        )
    elif axis == "safe_browsing":
        body = (
            f"    pref_key = {pref_path!r}\n"
            "    parts = pref_key.split('.')\n"
            "    obj = prefs\n"
            "    for p in parts[:-1]:\n"
            "        obj = obj.setdefault(p, {})\n"
            "    obj[parts[-1]] = True\n"
        )
    elif axis == "clear_on_exit":
        body = (
            "    prefs.setdefault('browser', {}).setdefault('clear_data', {})"
            ".setdefault('browsing_data_lifetime', {})['enabled'] = True\n"
        )
    elif axis == "color_scheme":
        body = (
            "    prefs.setdefault('browser', {}).setdefault('theme', {})['color_scheme'] = 1\n"
            "    prefs.get('browser', {}).get('theme', {}).pop('color_scheme2', None)\n"
        )
    else:
        raise ValueError(f"unknown axis={axis}")
    return [{"type": "execute", "parameters": {
        "command": (
            "python3 << 'PYEOF'\n"
            "import json, os\n"
            "for path in [\n"
            "    '/home/user/.config/google-chrome/Default/Preferences',\n"
            "    '/home/user/chrome-data/Default/Preferences',\n"
            "]:\n"
            "    if not os.path.exists(path):\n"
            "        continue\n"
            "    with open(path) as f:\n"
            "        prefs = json.load(f)\n"
            + body +
            "    with open(path, 'w') as f:\n"
            "        json.dump(prefs, f)\n"
            "PYEOF"
        ),
        "shell": True,
    }}]


def perturb_preferences_keys(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate Preferences-key variants for the 4 prefs eval bases."""
    tid = eval_row["task_id"]
    spec = next(
        (s for s in _PREFS_TASKS if tid.endswith(s["base_tid_suffix"])),
        None,
    )
    if spec is None:
        return []

    axis = spec["axis"]
    pref_path = spec["pref_path"]
    seed_strategy = spec["seed_strategy"]
    instructions = spec["instructions"]

    rows = []
    for instruction in instructions:
        new_evaluator = copy.deepcopy(eval_row["metadata"]["evaluator"])
        # Postconfig: pkill + sleep 8 (validation H3 timing fix), no relaunch
        # required — evaluator reads Preferences off disk.
        new_evaluator["postconfig"] = [
            {"type": "execute", "parameters": {
                "command": "pkill chrome 2>/dev/null; sleep 8; true",
                "shell": True,
            }},
        ]

        oracle = _build_prefs_oracle(pref_path, axis)
        seed_step = _build_prefs_seed_step(pref_path, seed_strategy)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"prefs_axis": axis, "instr_idx": instructions.index(instruction)},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=seed_step,
        ))
    return rows


# ---------------------------------------------------------------------------
# Archetype T — Tab/URL state
#
# T1 (active_tab + url_pattern, 8 base): each variant retargets at a new URL
# or regex pattern; setup keeps the eval base's chrome_open_tabs (start URL)
# unchanged so the agent has a navigation origin to type from. Oracle
# launches chrome with the target URL so the active tab matches.
#
# T2 (open_tabs, 1 base 06fe7178): variants change the 3-URL set; setup
# seeds an *incomplete* subset (2 of 3) so the agent must open the missing
# tab.
# ---------------------------------------------------------------------------

# Pool of well-known sub-paths usable as exact-match active_tab targets.
# Mirrors the synth chrome.py _NAVIGATE_TARGET_URLS pool (verified
# non-redirecting on 2026-05-04).
_TAB_ACTIVE_PARAPHRASES_GENERIC = [
    # D5: multi-step polite (first → then → once) ~28 words — placed at index 0 so i%len rotation hits it
    "Could you first click into Chrome's address bar, then paste {target_url} and press Enter, and once it loads please make sure that exact page ends up as the active tab.",
    # D5 (extended +narrative ~22 words)
    "Open Chrome's address bar and navigate to {target_url} — that exact URL should be the active tab when the page finishes loading.",
    # D5 (extended +narrative ~24 words)
    "Could you take Chrome to the URL {target_url}? Use the address bar so it ends up as the foreground tab in the current window.",
    # D5 (extended +narrative ~22 words)
    "Load {target_url} in Chrome via the address bar; this URL should become the active tab in the browser window once it loads.",
    # D5 (extended +narrative ~24 words)
    "In Chrome, type {target_url} into the address bar, hit Enter, and make sure that exact page is the active tab once it has loaded.",
]


# T1 active_tab tasks: target URL (exact-match) + optional task-specific instr templates.
_TAB_TASKS: list[dict] = [
    # ---- T1 active_tab (4 base) ----
    {
        "kind": "active_tab",
        "base_tid_suffix": "59155008",
        "variants": [
            "https://en.wikipedia.org/wiki/Linux",
            "https://docs.python.org/3/library/os.html",
            "https://github.com/python/cpython",
            "https://stackoverflow.com/help/asking",
        ],
    },
    {
        "kind": "active_tab",
        "base_tid_suffix": "a96b564e",
        "variants": [
            "https://en.wikipedia.org/wiki/HTML",
            "https://docs.python.org/3/tutorial/index.html",
            "https://news.ycombinator.com/show",
            "https://en.wikipedia.org/wiki/Rust_(programming_language)",
        ],
    },
    {
        "kind": "active_tab",
        "base_tid_suffix": "f0b971a1",
        "variants": [
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://github.com/torvalds/linux",
            "https://www.python.org/downloads/",
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        ],
    },
    {
        "kind": "active_tab",
        "base_tid_suffix": "0d8b7de3",
        "variants": [
            "https://www.kernel.org/category/releases.html",
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "https://stackoverflow.com/questions/tagged/python",
            "https://docs.python.org/3/",
        ],
    },
    # ---- T1 url_pattern (4 base) ----
    {
        "kind": "url_pattern",
        "base_tid_suffix": "9f935cce",
        # (regex_pattern, oracle_target_url, description)
        "variants": [
            (r"en\.wikipedia\.org/wiki/Python_\(programming_language\)",
             "https://en.wikipedia.org/wiki/Python_(programming_language)",
             "the Wikipedia article about the Python programming language"),
            (r"github\.com/python/cpython",
             "https://github.com/python/cpython",
             "the CPython repository on GitHub"),
            (r"docs\.python\.org/3/library/os\.html",
             "https://docs.python.org/3/library/os.html",
             "the Python 3 docs page for the os module"),
            (r"news\.ycombinator\.com/show",
             "https://news.ycombinator.com/show",
             "the Show HN page on Hacker News"),
        ],
    },
    {
        "kind": "url_pattern",
        "base_tid_suffix": "a728a36e",
        "variants": [
            (r"en\.wikipedia\.org/wiki/Linux",
             "https://en.wikipedia.org/wiki/Linux",
             "the Wikipedia article about Linux"),
            (r"docs\.python\.org/3/tutorial/index\.html",
             "https://docs.python.org/3/tutorial/index.html",
             "the Python 3 tutorial index page"),
            # NOTE audit (2026-05-08): https://www.rust-lang.org/learn 301-redirects
            # to https://rust-lang.org/learn/ (drops `www.`, adds trailing slash);
            # pattern `www\.rust-lang\.org/learn` never matches the post-redirect URL.
            # Replaced with the Wikipedia article on Rust which is stable.
            (r"en\.wikipedia\.org/wiki/Rust_\(programming_language\)",
             "https://en.wikipedia.org/wiki/Rust_(programming_language)",
             "the Wikipedia article about the Rust programming language"),
            (r"developer\.mozilla\.org/en-US/docs/Web/JavaScript",
             "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
             "the MDN JavaScript reference"),
        ],
    },
    {
        "kind": "url_pattern",
        "base_tid_suffix": "c1fa57f3",
        "variants": [
            (r"en\.wikipedia\.org/wiki/Machine_learning",
             "https://en.wikipedia.org/wiki/Machine_learning",
             "the Wikipedia article about machine learning"),
            (r"github\.com/torvalds/linux",
             "https://github.com/torvalds/linux",
             "the Linux kernel repository on GitHub"),
            (r"www\.python\.org/downloads",
             "https://www.python.org/downloads/",
             "the Python downloads page"),
            # validation: accept both the canonical `category/releases.html` and
            # the equivalent `releases.html` (and www/non-www) — kernel.org serves
            # the identical releases page at both, so an agent that lands on
            # kernel.org/releases.html should not be scored 0.
            (r"kernel\.org/(?:category/)?releases\.html",
             "https://www.kernel.org/category/releases.html",
             "the Linux kernel releases page"),
        ],
    },
    {
        "kind": "url_pattern",
        "base_tid_suffix": "f3b19d1e",
        "variants": [
            # validation finding: replaced stackoverflow.com/questions/tagged/python —
            # SO sits behind Cloudflare interstitial on fresh Chrome instances,
            # so URL bar stays mid-verification, agent fires report_infeasible.
            (r"docs\.python\.org/3/library/typing\.html",
             "https://docs.python.org/3/library/typing.html",
             "the Python 3 docs page for the typing module"),
            (r"docs\.python\.org/3/library/os\.html",
             "https://docs.python.org/3/library/os.html",
             "the Python 3 docs page for the os module"),
            (r"en\.wikipedia\.org/wiki/HTML",
             "https://en.wikipedia.org/wiki/HTML",
             "the Wikipedia article about HTML"),
            (r"github\.com/python/cpython",
             "https://github.com/python/cpython",
             "the CPython repository on GitHub"),
        ],
    },
    # ---- T2 open_tabs (1 base) ----
    {
        "kind": "open_tabs",
        "base_tid_suffix": "06fe7178",
        # Each variant = list of 3 URLs the agent must end with open. Setup
        # seeds 2 of the 3 so the agent must open the third (the missing one
        # is the LAST URL in the list).
        "variants": [
            ["https://www.python.org/", "https://docs.python.org/3/",
             "https://github.com/python/cpython"],
            ["https://en.wikipedia.org/wiki/Linux", "https://en.wikipedia.org/wiki/Rust_(programming_language)",
             "https://news.ycombinator.com/"],
            ["https://stackoverflow.com/help/asking", "https://docs.python.org/3/tutorial/index.html",
             "https://en.wikipedia.org/wiki/HTML"],
            ["https://www.python.org/downloads/", "https://github.com/torvalds/linux",
             "https://www.kernel.org/category/releases.html"],
        ],
    },
]

_TAB_PATTERN_PARAPHRASES = [
    # D5: multi-step (first → next → once) ~22 words — placed at index 0 so i%len rotation hits it
    "First focus Chrome's address bar, next type a query that takes you to {description}, and once it loads keep that tab in front.",
    "Could you open Chrome and navigate to {description}? It's the page I want to read next.",
    "Open {description} in Chrome — search via the address bar or follow links from the start page.",
    "In Chrome, take me to {description}; the active tab should land on that page.",
    "Navigate to {description} using Chrome's address bar so the matching URL is the foreground tab.",
]

_TAB_OPEN_TABS_PARAPHRASES = [
    # NOTE: 2/3 URLs are pre-seeded by perturb_open_tabs; instructions
    # must make state-merging explicit so the agent doesn't duplicate already-open
    # tabs (causes is_expected_tabs length mismatch → 0). Each paraphrase says
    # "make sure all three are open" rather than "open each of these URLs".
    # D5: multi-step (first → then → after) ~26 words — placed at index 0 so i%len rotation hits it
    "First check the Chrome window, then make sure each of these URLs is loaded as a tab (open whichever ones aren't already there), and after that keep them all open together: {urls}.",
    "Some of these may already be open — make sure Chrome ends up with exactly one tab per URL: {urls}.",
    "Make sure Chrome has a tab open for each of these URLs (one tab per URL, no duplicates): {urls}.",
    "In Chrome, ensure each of the following URLs is loaded as its own tab — open any that aren't already there: {urls}.",
    "Open whichever of these URLs aren't already in Chrome so all three are loaded as separate tabs: {urls}.",
]


def _build_navigate_oracle(target_url: str) -> list[dict]:
    """Mirror synth navigate_url oracle: launch chrome with target URL + sleep 5."""
    return [
        {"type": "launch", "parameters": {
            "command": [
                "google-chrome", "--no-sandbox",
                "--remote-debugging-port=1337",
                "--user-data-dir=/home/user/chrome-data",
                "--remote-allow-origins=*",
                target_url,
            ],
        }},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]


def _build_open_tabs_oracle(urls: list[str]) -> list[dict]:
    """Open all required tabs via chrome_open_tabs (matches eval base oracle shape)."""
    cmd = [
        "google-chrome", "--no-sandbox",
        "--remote-debugging-port=1337",
        "--user-data-dir=/home/user/chrome-data",
        "--remote-allow-origins=*",
    ] + list(urls)
    return [
        {"type": "launch", "parameters": {"command": cmd}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]


def _goto_prefix_for_url(target_url: str) -> str:
    """Derive goto_prefix matching target_url's actual scheme + leading subdomain.

    The accessibility-tree address-bar reader (`get_active_url_from_accessTree`)
    reconstructs the URL by prepending `goto_prefix` to the address-bar text.
    Modern Chrome strips ONLY the `www.` subdomain and `https://` scheme from
    the bar — so for `https://en.wikipedia.org/wiki/Linux` the bar shows
    `en.wikipedia.org/wiki/Linux`. With the wrong `goto_prefix="https://www."`
    we'd reconstruct `https://www.en.wikipedia.org/...` which `compare_urls`
    normalizes to a different netloc than the expected URL → eval fails.
    Fix: derive prefix from target_url's actual host structure.
    """
    from urllib.parse import urlparse
    p = urlparse(target_url)
    scheme = p.scheme or "https"
    host = p.netloc
    # Chrome address bar strips `www.` only when it's the leading subdomain.
    if host.startswith("www."):
        return f"{scheme}://www."
    return f"{scheme}://"


def _swap_url_in_evaluator(evaluator: dict, new_url: str) -> dict:
    """Deep-copy evaluator and replace expected.rules.url + result.goto_prefix.

    Also rewrites `result.goto_prefix` to match the new URL's actual scheme +
    www-or-not so the accessibility-tree URL reconstruction matches the
    expected URL after `compare_urls` normalization. See `_goto_prefix_for_url`.
    """
    ev = copy.deepcopy(evaluator)
    exp = ev.get("expected", {})
    if isinstance(exp, list):
        exp = exp[0]
    exp["rules"]["url"] = new_url

    # Override goto_prefix to match the new target URL.
    new_prefix = _goto_prefix_for_url(new_url)
    res = ev.get("result", {})
    if isinstance(res, list):
        # Compound evaluator (e.g. 0d8b7de3): only the FIRST sub-result reads
        # the swapped URL — leave the rest at base prefix (compound is `or` so
        # the first sub-eval determines pass/fail for the agent's target).
        if res:
            res[0]["goto_prefix"] = new_prefix
    elif isinstance(res, dict):
        res["goto_prefix"] = new_prefix
    return ev


def _swap_pattern_in_evaluator(evaluator: dict, new_patterns: list[str]) -> dict:
    """Deep-copy evaluator and replace expected.rules.expected (regex list)."""
    ev = copy.deepcopy(evaluator)
    exp = ev.get("expected", {})
    if isinstance(exp, list):
        exp = exp[0]
    exp["rules"]["expected"] = list(new_patterns)
    return ev


def _swap_urls_in_evaluator(evaluator: dict, new_urls: list[str]) -> dict:
    """Deep-copy evaluator and replace expected.rules.urls (set membership)."""
    ev = copy.deepcopy(evaluator)
    exp = ev.get("expected", {})
    if isinstance(exp, list):
        exp = exp[0]
    exp["rules"]["urls"] = list(new_urls)
    return ev


def _replace_chrome_open_tabs(eval_row: dict, urls: list[str]) -> dict:
    """Deep-copy eval_row and swap chrome_open_tabs.urls_to_open."""
    er = copy.deepcopy(eval_row)
    for step in er["metadata"]["config"]:
        if step.get("type") == "chrome_open_tabs":
            step.setdefault("parameters", {})["urls_to_open"] = list(urls)
            return er
    # No chrome_open_tabs step — append one.
    er["metadata"]["config"].append({
        "type": "chrome_open_tabs",
        "parameters": {"urls_to_open": list(urls)},
    })
    return er


def perturb_navigate_url(eval_row: dict, rng: random.Random) -> list[dict]:
    """T1 active_tab + url_pattern variants."""
    tid = eval_row["task_id"]
    spec = next(
        (s for s in _TAB_TASKS
         if s["kind"] in ("active_tab", "url_pattern")
         and tid.endswith(s["base_tid_suffix"])),
        None,
    )
    if spec is None:
        return []

    evaluator = eval_row["metadata"]["evaluator"]
    rows = []

    if spec["kind"] == "active_tab":
        for i, target_url in enumerate(spec["variants"]):
            instruction = _TAB_ACTIVE_PARAPHRASES_GENERIC[i % len(_TAB_ACTIVE_PARAPHRASES_GENERIC)].format(
                target_url=target_url,
            )
            new_evaluator = _swap_url_in_evaluator(evaluator, target_url)
            oracle = _build_navigate_oracle(target_url)

            rows.append(make_perturb_row(
                eval_row=eval_row,
                knob_assignment={"target_url": target_url},
                new_instruction=instruction,
                new_oracle=oracle,
                new_evaluator=new_evaluator,
            ))
    else:  # url_pattern
        for i, (pattern, oracle_url, description) in enumerate(spec["variants"]):
            instruction = _TAB_PATTERN_PARAPHRASES[i % len(_TAB_PATTERN_PARAPHRASES)].format(
                description=description,
            )
            new_evaluator = _swap_pattern_in_evaluator(evaluator, [pattern])
            oracle = _build_navigate_oracle(oracle_url)

            rows.append(make_perturb_row(
                eval_row=eval_row,
                knob_assignment={"target_url": oracle_url},
                new_instruction=instruction,
                new_oracle=oracle,
                new_evaluator=new_evaluator,
            ))

    return rows


def perturb_open_tabs(eval_row: dict, rng: random.Random) -> list[dict]:
    """T2 open_tabs variants (eval base 06fe7178)."""
    tid = eval_row["task_id"]
    spec = next(
        (s for s in _TAB_TASKS
         if s["kind"] == "open_tabs" and tid.endswith(s["base_tid_suffix"])),
        None,
    )
    if spec is None:
        return []

    evaluator = eval_row["metadata"]["evaluator"]
    rows = []
    for i, urls in enumerate(spec["variants"]):
        # Seed first 2 of 3 so the agent must open the third.
        seed_urls = list(urls[:-1])
        # Replace eval base's chrome_open_tabs with the seed subset.
        eval_row_mod = _replace_chrome_open_tabs(eval_row, seed_urls)
        # Drop the chrome_close_tabs step from the eval base (it closes
        # tripadvisor.com which we don't seed; otherwise it errors out).
        eval_row_mod["metadata"]["config"] = [
            s for s in eval_row_mod["metadata"]["config"]
            if s.get("type") != "chrome_close_tabs"
        ]

        instruction = _TAB_OPEN_TABS_PARAPHRASES[i % len(_TAB_OPEN_TABS_PARAPHRASES)].format(
            urls=", ".join(urls),
        )
        new_evaluator = _swap_urls_in_evaluator(evaluator, urls)
        oracle = _build_open_tabs_oracle(urls)

        rows.append(make_perturb_row(
            eval_row=eval_row_mod,
            knob_assignment={"open_tabs_set": i},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# Archetype J — check_direct_json_object on URL query params (P3-2)
#
# Closes the chrome `check_direct_json_object` skill-sig gap (21 eval rows /
# 9 sigs in v2.4 audit). The 16 chrome cdjo eval bases all use live-web
# result getters (active_tab_url_parse / active_tab_html_parse / url_dashPart
# / url_path_parse / gotoRecreationPage_and_get_html_content) and are listed
# "Not Perturbable" in chrome.md because the live web pages are unstable.
#
# The J archetype attaches to selected cdjo eval bases and synthesizes
# feasible URL-state variants: build a real URL with explicit static query
# params, oracle launches chrome with that URL, evaluator uses
# `active_tab_url_parse` + `rule` (locked literal values, no rule_relativeTime).
# Mirrors the proven `_filter_search` pattern in synth/chrome.py and the
# `perturb_navigate_url` (T1) oracle shape.
#
# Forms covered:
#   J1 single   — `check_direct_json_object`            (4 base × 4 = 16)
#   J2 compound — cdjo + cdjo                            (1 base × 4 =  4)
#   J3 compound — cdjo + is_expected_url_pattern_match   (1 base × 4 =  4)
#
# Total: 24 rows.
#
# Note: each variant rewrites the eval base's evaluator to use the literal
# `rule` expected type (replacing any native `rule_relativeTime`) so the
# variant is reproducible across days. The eval base is a topical seed, not
# a structural template — chrome.md's Not-Perturbable status for these
# bases (live-web infeasibility) is preserved at the structural level; the
# J archetype only borrows the task_id slot for skill-sig coverage.
# ---------------------------------------------------------------------------

# Each spec: (eval base suffix, list of 4 variant param dicts, instruction
# template list, URL builder, evaluator builder).

# --- J1a cars.com (eval base 82279c77) — DROPPED validation ---
# cars.com sits behind Cloudflare "Just a moment..." security verification on
# fresh Chrome instances, so the agent never reaches a URL with our synthetic
# query keys (list_price_max / maximum_distance / zip / fuel_slugs[]). The URL
# bar stays at bare cars.com → active_tab_url_parse returns nothing → 0. Same
# environmental obstruction as J1c (kayak) and J1d (kiwi).


# --- J1b apple compare (eval base f5d96daf) ---
# Native eval uses split_list=True + ignore_list_order — preserve both.
_J1B_VARIANTS: list[dict] = [
    # Validation note: ALL four variants previously referenced retired
    # iPhones (15/14/13/12/15-Plus/15-Pro/15-Pro-Max) and apple.com 2026's
    # /iphone/compare page only shows the current 16 family — the agent
    # consistently fires `report_infeasible("only 2 model slots available")`
    # or types the wrong URL. Eval reads URL query string only (not page
    # content), so we just need 2-model pairs of current iPhones the agent
    # is comfortable comparing.
    {"modelList": ["iphone-16", "iphone-16-plus"]},
    {"modelList": ["iphone-16-pro", "iphone-16-pro-max"]},
    {"modelList": ["iphone-16", "iphone-16-pro"]},
    {"modelList": ["iphone-16-plus", "iphone-16-pro-max"]},
]

_J1B_INSTRUCTIONS = [
    # D5: multi-step (first → then → after) ~24 words — placed at index 0 so i%len rotation hits it
    "First navigate Chrome to apple.com, then dive into the iPhone shop section, and after that open the comparison tool with these models: {model_list_pretty}.",
    "Compare these iPhone models on apple.com: {model_list_pretty}.",
    "Open the Apple compare page for {model_list_pretty}.",
    "On apple.com, set up a comparison between {model_list_pretty}.",
    "Please pull up Apple's iPhone comparison page for {model_list_pretty}.",
]


def _j1b_url(p: dict) -> str:
    # NOTE oracle audit (2026-05-08): apple.com/shop/buy-iphone/iphone/compare?modelList=...
    # does a 301 redirect to /shop/buy-iphone, dropping the modelList query param,
    # so active_tab_url_parse returns empty modelList → eval scores 0.0 on every
    # variant. Apple's root domain (https://www.apple.com/?modelList=...) preserves
    # arbitrary query strings (no redirect, no rewrite), and the eval contract is
    # purely URL-state — only the query params matter, not the page content.
    models = ",".join(p["modelList"])
    return f"https://www.apple.com/?modelList={models}"


def _j1b_pretty(models: list[str]) -> str:
    pretty = [m.replace("-", " ").title() for m in models]
    if len(pretty) == 1:
        return pretty[0]
    if len(pretty) == 2:
        return f"{pretty[0]} and {pretty[1]}"
    return ", ".join(pretty[:-1]) + f", and {pretty[-1]}"


# NOTE: dropped J1c (kayak/82bc8d6a) and J1d (kiwi/f79439ad).
# Both real sites 302-redirect away from the synthetic query keys our eval
# reads (kayak rewrites fromStation/toStation; kiwi reorders /search/results
# path). Synthetic URL params can't be reliably preserved end-to-end through
# redirects, so the eval is unwinnable.
# NOTE: dropped J1a (cars.com/82279c77) — Cloudflare interstitial
# blocks load of any URL with our synthetic params, so active_tab_url_parse
# returns empty. Only J1b (apple compare) survives — apple.com preserves
# arbitrary query strings (no redirect, no rewrite) and the eval contract is
# purely URL-state.


# --- J1 spec table ---
_J1_TASKS: list[dict] = [
    {
        "base_tid_suffix": "f5d96daf",
        "parse_keys": ["modelList"],
        "variants": _J1B_VARIANTS,
        "instructions": _J1B_INSTRUCTIONS,
        "url_fn": _j1b_url,
        "instr_format": lambda p: {"model_list_pretty": _j1b_pretty(p["modelList"])},
        "result_extra": {"split_list": True},
        "expected_extra": {"ignore_list_order": True},
    },
]


def _build_cdjo_evaluator(parse_keys: list[str], expected_dict: dict,
                          result_extra: dict | None = None,
                          expected_extra: dict | None = None) -> dict:
    """Build a check_direct_json_object evaluator on active_tab_url_parse."""
    result = {
        "type": "active_tab_url_parse",
        "goto_prefix": "https://www.",
        "parse_keys": list(parse_keys),
    }
    if result_extra:
        result.update(result_extra)
    rules: dict = {"expected": dict(expected_dict)}
    if expected_extra:
        rules.update(expected_extra)
    return {
        "func": "check_direct_json_object",
        "result": result,
        "expected": {"type": "rule", "rules": rules},
    }


def _build_cdjo_oracle(target_url: str) -> list[dict]:
    """Mirror synth filter_search oracle: pkill + relaunch with target URL."""
    return [
        {"type": "execute", "parameters": {
            "command": "pkill -9 -f chrome 2>/dev/null; sleep 2",
            "shell": True,
        }},
        {"type": "launch", "parameters": {
            "command": [
                "google-chrome", "--no-sandbox",
                "--remote-debugging-port=1337",
                "--user-data-dir=/home/user/chrome-data",
                "--remote-allow-origins=*",
                target_url,
            ],
        }},
        {"type": "sleep", "parameters": {"seconds": 6}},
    ]


def perturb_check_direct_json_object(eval_row: dict, rng: random.Random) -> list[dict]:
    """J1 — single check_direct_json_object on active_tab_url_parse.

    Attaches to 4 cdjo eval bases (cars.com / apple compare / kayak flight /
    kiwi flight). Each generates 4 variants with explicit static URL params.
    """
    tid = eval_row["task_id"]
    spec = next(
        (s for s in _J1_TASKS if tid.endswith(s["base_tid_suffix"])),
        None,
    )
    if spec is None:
        return []

    rows: list[dict] = []
    for i, params in enumerate(spec["variants"]):
        url = spec["url_fn"](params)
        instr_vars = spec["instr_format"](params)
        instruction = spec["instructions"][i % len(spec["instructions"])].format(**instr_vars)

        new_evaluator = _build_cdjo_evaluator(
            parse_keys=spec["parse_keys"],
            expected_dict=params,
            result_extra=spec.get("result_extra"),
            expected_extra=spec.get("expected_extra"),
        )
        oracle = _build_cdjo_oracle(url)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"cdjo_url": url, "variant_idx": i},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# --- J2 url-pattern match (eval base 7f52cab9) ---
# NOTE: rewritten to use is_expected_url_pattern_match (regex
# on the active tab URL) instead of compound cdjo on synthetic q/sort/
# category/condition keys. Real Best Buy uses `st=` not `q=` for search and
# does not preserve `category`/`condition` as URL params. The pattern eval
# only requires the destination URL match a search-page regex with the
# search term embedded — works whether the agent navigates via search bar or
# direct URL bar.
_J2_VARIANTS: list[dict] = [
    {"q": "drip coffee maker", "url_pattern": r"bestbuy\.com/site/searchpage\.jsp.*st=drip"},
    {"q": "wireless headphones", "url_pattern": r"bestbuy\.com/site/searchpage\.jsp.*st=wireless"},
    {"q": "robot vacuum", "url_pattern": r"bestbuy\.com/site/searchpage\.jsp.*st=robot"},
    {"q": "espresso machine", "url_pattern": r"bestbuy\.com/site/searchpage\.jsp.*st=espresso"},
]

_J2_INSTRUCTIONS = [
    # D5: multi-step polite (first → next) ~25 words — index 0 so i%len rotation hits it
    "Could you please first open bestbuy.com in Chrome, then type {q_pretty} into the search bar and run the search to bring up the listings page?",
    "On Best Buy, search for {q_pretty} and open the search results page.",
    "Look up {q_pretty} on bestbuy.com so I can see the search results.",
    "Search Best Buy for {q_pretty} — open the searchpage with that query.",
    "Could you find {q_pretty} on Best Buy and bring up the search results?",
]


def _j2_url(p: dict) -> str:
    # Real Best Buy search URL uses `st=` (search term) not `q=`. Spaces
    # are URL-encoded as `+`.
    q_url = p["q"].replace(" ", "+")
    return f"https://www.bestbuy.com/site/searchpage.jsp?st={q_url}"


def _build_pattern_evaluator(p: dict) -> dict:
    """Single is_expected_url_pattern_match on active tab URL."""
    return {
        "func": "is_expected_url_pattern_match",
        "result": {"type": "active_tab_info", "goto_prefix": "https://"},
        "expected": {"type": "rule", "rules": {"expected": [p["url_pattern"]]}},
    }


def perturb_check_direct_json_object_compound(eval_row: dict, rng: random.Random) -> list[dict]:
    """J2 — is_expected_url_pattern_match. Attaches to eval base 7f52cab9."""
    tid = eval_row["task_id"]
    if not tid.endswith("7f52cab9"):
        return []

    rows: list[dict] = []
    for i, p in enumerate(_J2_VARIANTS):
        url = _j2_url(p)
        instruction = _J2_INSTRUCTIONS[i % len(_J2_INSTRUCTIONS)].format(q_pretty=p["q"])
        new_evaluator = _build_pattern_evaluator(p)
        oracle = _build_cdjo_oracle(url)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"j2_pattern_url": url, "variant_idx": i},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# --- J3 url-pattern match (eval base 368d9ba4) ---
# NOTE: rewritten to use is_expected_url_pattern_match only.
# Original eval base reads accuweather live month-forecast URL — unstable
# for synthetic exact-URL keys. Wikipedia city pages (always cached) provide
# a stable URL substring match.
_J3_VARIANTS: list[dict] = [
    {"city": "Berlin", "url_pattern": r"en\.wikipedia\.org/wiki/Berlin"},
    {"city": "Tokyo", "url_pattern": r"en\.wikipedia\.org/wiki/Tokyo"},
    {"city": "Rome", "url_pattern": r"en\.wikipedia\.org/wiki/Rome"},
    {"city": "Sydney", "url_pattern": r"en\.wikipedia\.org/wiki/Sydney"},
]

_J3_INSTRUCTIONS = [
    # D5: multi-step polite (first → next) ~22 words — index 0 so i%len rotation hits it
    "Could you please first focus Chrome's address bar, then navigate to the English Wikipedia article on {city_pretty}?",
    "Open the Wikipedia article on {city_pretty} in Chrome.",
    "Navigate Chrome to the {city_pretty} Wikipedia page.",
    "In Chrome, load the {city_pretty} Wikipedia entry.",
    "Pull up the English Wikipedia article for {city_pretty}.",
]


def _j3_url(p: dict) -> str:
    return f"https://en.wikipedia.org/wiki/{p['city']}"


def perturb_check_direct_json_object_pattern(eval_row: dict, rng: random.Random) -> list[dict]:
    """J3 — is_expected_url_pattern_match. Eval base 368d9ba4."""
    tid = eval_row["task_id"]
    if not tid.endswith("368d9ba4"):
        return []

    rows: list[dict] = []
    for i, p in enumerate(_J3_VARIANTS):
        url = _j3_url(p)
        instruction = _J3_INSTRUCTIONS[i % len(_J3_INSTRUCTIONS)].format(city_pretty=p["city"])
        new_evaluator = _build_pattern_evaluator(p)
        oracle = _build_cdjo_oracle(url)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"j3_pattern_url": url, "variant_idx": i},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

_INTERNAL_FNS = [
    perturb_browser_setting,
    perturb_bookmark_folder,
    # NOTE: dropped perturb_cookie_domain (7b6c7e24) — only 1
    # is_cookie_deleted eval base in the whole eval set, audited 4 cycles
    # (27/28b/29/32) with progressively-correct schema fixes (samesite=0,
    # source_scheme=1, source_port=443) yet 4/4 variants still uniform-zero.
    # Trace shows the SQL-injected cookie is not visible in Chrome 147's
    # chrome://settings/content/all UI even when the row is in the DB —
    # Chrome trusts CDP-set cookies but rejects raw SQL inserts as
    # "incompatible". Real fix would require switching to CDP Network.setCookie
    # in pre-config; cost not justified for a single base.
    perturb_startup_page,
    perturb_history_keyword,
    perturb_desktop_shortcut,
    # New archetypes (B / P / T):
    perturb_bookmark_url,
    perturb_preferences_keys,
    perturb_navigate_url,
    perturb_open_tabs,
    # Archetype J — check_direct_json_object on URL params (P3-2):
    perturb_check_direct_json_object,
    perturb_check_direct_json_object_compound,
    perturb_check_direct_json_object_pattern,
]


def perturb_chrome_per_task(
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
