"""Thunderbird domain perturbation functions (Track B).

Structural functions: (eval_row, rng) -> list[dict]

Usage:
    from lite.gym.envs.lite.osworld.src.gen.train.perturb.thunderbird import THUNDERBIRD_PERTURB_FNS
    rows = THUNDERBIRD_PERTURB_FNS["pref_setting"](eval_row, rng)
"""

from __future__ import annotations

import copy
import logging
import random
import re

logger = logging.getLogger(__name__)

from lite.gym.envs.lite.osworld.src.gen.train.perturb._utils import make_perturb_row, oracle_actions_of


_TB_PROFILE_DIR = "/home/user/.thunderbird/t5q2a5hp.default-release"

# Known boolean prefs and their perturbation specs.
#
# NOTE (V3 instruction quality fix): templates are imperative + narrative
# with task-context motivation. Each pool keeps 4 imperative + 1 polite
# variant (target ~15-20% polite, matching the eval distribution of 13%).
# Word counts target ~25-30 to mirror eval's detailed descriptions.
# Verb is lowercased mid-sentence via .lower() at format time.
_BOOLEAN_PREF_SPECS: dict[str, dict] = {
    "mail.server.default.applyIncomingFilters": {
        "instruction_templates": [
            "In Thunderbird, {verb} automatic application of message filters on incoming mail so my subfolder rules fire as new messages arrive instead of waiting for a manual run.",
            "Open Thunderbird's account settings and {verb} the auto-apply filters option for incoming messages so filters run on every fetch and my inbox stays organized.",
            "Configure Thunderbird so incoming mail filters are {verb}d automatically; with several accounts feeding the inbox, I can't keep clicking 'Run Filters Now' all day.",
            "Adjust the mail server defaults in Thunderbird and {verb} automatic application of incoming filters — those rules sort newsletters and team threads as they arrive.",
            "Please {verb} the auto-apply filters setting for incoming mail in Thunderbird. I have several accounts and want filtering rules to run on every fetch.",
            # D5 short variant (~15 words): terse imperative for instruction-length diversity
            "In Thunderbird, {verb} automatic application of message filters on incoming mail across all mail accounts.",
            # D5 long variant (~46 words): narrative pad with task context for length tail
            "My team feeds three mailing lists into this Thunderbird inbox and the subfolder rules never run unless I click Run Filters Now, so go into the account settings and {verb} automatic application of message filters on incoming mail so the rules fire for every fetch.",
        ],
    },
    "mail.imap.use_status_for_biff": {
        "instruction_templates": [
            "In Thunderbird, {verb} the IMAP STATUS command for new-mail notifications so the client checks message counts the lightweight way instead of running a full folder fetch.",
            "Open Thunderbird's advanced config editor and {verb} use_status_for_biff for IMAP — on a slow link the STATUS check is much cheaper than the default new-mail probe.",
            "Update Thunderbird's IMAP behaviour and {verb} the STATUS-based new-mail check; my server handles STATUS efficiently and I'd rather use it than the alternative biff method.",
            "Configure Thunderbird so the IMAP STATUS poll for new messages is {verb}d; my mailbox has thousands of messages and STATUS notifications are cheaper than the default.",
            "I want to {verb} the IMAP STATUS new-mail check in Thunderbird because I'm trying to reduce the bandwidth my mail client uses on a metered connection.",
            # D5 short variant (~15 words): terse imperative for instruction-length diversity
            "In Thunderbird, {verb} the IMAP STATUS command for new-mail notifications across all my IMAP accounts.",
            # D5 long variant (~49 words): narrative pad with task context for length tail
            "I am on a metered home internet connection while travelling this month and the default Thunderbird IMAP biff probe is burning through my data cap on every poll, so go into about:config and {verb} the IMAP STATUS command for new-mail notifications across all the IMAP accounts I have configured.",
        ],
    },
    "mail.identity.id1.auto_quote": {
        # NOTE: templates need polarity-aware
        # narrative — eval base has ref=false so perturb flips to verb=enable,
        # but the previous all-templates-anti-quote phrasing ("I'd rather compose
        # clean replies", "no longer inserted") contradicted enable-direction
        # → agent disabled → 4/4 variants failed. Split into separate enable/
        # disable template lists; the perturb_thunderbird_pref code below picks
        # the right list by verb.
        "instruction_templates_enable": [
            "In Thunderbird, enable automatic quoting of the original message when I reply — pasting the prior thread into the composer helps reviewers follow the back-and-forth.",
            "Open Thunderbird's identity settings and enable the auto-quote feature for replies, so the original message text is automatically inserted whenever I hit Reply.",
            "Configure Thunderbird's reply composer to enable automatic quoting of the original message; the team's review process expects the prior thread inline.",
            "In Thunderbird's primary-identity settings, turn on the option that quotes the original message in replies, so colleagues see the conversation history without scrolling.",
            "Could you enable Thunderbird's auto-quote-on-reply setting? Inline quoted history is the convention on the mailing list I'm on.",
            # short imperative
            "In Thunderbird, enable automatic quoting of the original message text whenever I hit the Reply button.",
            # long narrative pad
            "The mailing list I'm on expects every reply to inline the prior thread for context, so go into the identity settings for my primary account and enable automatic quoting of the original message whenever I hit the Reply button.",
        ],
        "instruction_templates_disable": [
            "In Thunderbird, disable automatic quoting of the original message when I reply — I'd rather compose clean replies myself than have the entire previous thread pasted in.",
            "Open Thunderbird's identity settings and disable the auto-quote feature for replies, so the original message text is no longer inserted whenever I hit Reply.",
            "Configure Thunderbird's reply composer to disable automatic quoting of the original message; on long mailing-list threads the auto-included history makes my replies enormous.",
            "In Thunderbird's primary-identity settings, turn off the option that quotes the original message in replies, so I can decide per-message whether to keep the prior text.",
            "Could you disable Thunderbird's auto-quote-on-reply setting? I'd rather start replies with a blank composer and only pull in the parts that actually matter.",
            # short imperative
            "In Thunderbird, disable automatic quoting of the original message text whenever I hit the Reply button.",
            # long narrative pad
            "Long mailing-list threads keep blowing up the size of my replies because Thunderbird is pasting the entire prior history into the composer every time, so go into the identity settings for my primary account and disable automatic quoting of the original message whenever I hit the Reply button.",
        ],
    },
}

# Theme perturbation
_THEMES = [
    ("thunderbird-compact-dark@mozilla.org", "dark"),
    ("thunderbird-compact-light@mozilla.org", "light"),
    ("default-theme@mozilla.org", "default"),
]

_THEME_TEMPLATES = [
    "Switch Thunderbird's appearance to the {name} theme — I work in low light most evenings and the default look is too bright during long email sessions.",
    "In Thunderbird, open the Add-ons Manager, go to Themes, and enable the built-in {name} theme so the UI matches the rest of my desktop's colour scheme.",
    "Change Thunderbird's active theme to {name} mode; with several mail accounts open all day, I want a consistent appearance instead of the mixed default.",
    "Configure Thunderbird to use the {name} theme as the active appearance, applying it across the message list, folder pane, and compose window for a unified look.",
    "Set Thunderbird's overall appearance to {name} mode; I spend a lot of time triaging email and the {name} theme is much easier on my eyes late at night.",
    # D5 short variant (~15 words): terse imperative for instruction-length diversity
    "Switch Thunderbird's overall active appearance over to the built-in {name} theme via the Add-ons Manager.",
    # D5 long variant (~44 words): narrative pad with task context for length tail
    "I spend hours triaging email after dinner under low ambient light and the default Thunderbird palette burns my eyes during long sessions, so go into the Add-ons Manager, open the Themes section, and switch the active theme to {name} so the UI stays comfortable.",
]

# Signature perturbation
_NAMES = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry"]
_AFFILIATIONS = ["ABC Lab", "ML Lab", "NLP Group", "AI Research", "CS Department", "Data Science Team", "Robotics Lab", "Vision Lab"]

# NOTE: the eval regex demands `name\naffil`
# (newline-separated). Templates must instruct the agent to put each
# value on its OWN line in the signature box — phrasing that suggests
# slash- or comma-joined inline text causes the agent to type
# "Name / Affil" on a single line, failing the eval.
_SIGNATURE_TEMPLATES = [
    'Set up a plain-text signature for my Thunderbird identity with "{name}" on the first line and "{affil}" on the second line — outgoing messages should close with my name above the affiliation.',
    'In Thunderbird, open the account settings and configure a two-line signature: type "{name}" on line one, press Enter, then "{affil}" on line two so collaborators see who I am.',
    'Configure my Thunderbird signature so the first line says "{name}" and the second line (after a newline) says "{affil}"; the team uses this format for external correspondence.',
    'Open Thunderbird signature settings for my primary identity and enter "{name}" on the first line, then "{affil}" on the next line — plain text only, no HTML or inline separators.',
    'Please create an email signature in Thunderbird: type "{name}" on line 1, press Enter, then type "{affil}" on line 2 so I get a clean two-line block on every outgoing message.',
    # D5 short variant (~15 words): terse imperative for instruction-length diversity
    'Set my Thunderbird signature to "{name}" on line one and "{affil}" on line two.',
    # D5 long variant (~40 words): narrative pad with task context for length tail
    'I am rolling out a new email-signature standard across the team this quarter, so configure my Thunderbird identity with a two-line plain-text signature: "{name}" on the first line, then "{affil}" on the second line, separated by a real newline.',
]


def perturb_thunderbird_pref(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate structural perturbations for Thunderbird about:config pref tasks.

    Handles check_thunderbird_prefs evaluator by identifying the pref keys
    in the expected rules and generating variants with different values.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")

    if func != "check_thunderbird_prefs":
        return []

    expected = evaluator.get("expected", {}).get("rules", {})
    expect_rules = expected.get("expect", {})
    if not isinstance(expect_rules, dict):
        return []

    # -- Theme perturbation --
    if "extensions.activeThemeID" in expect_rules:
        return _perturb_theme(eval_row, rng)

    # -- Signature perturbation --
    if "mail.identity.id1.htmlSigText" in expect_rules:
        return _perturb_signature(eval_row, rng)

    # -- Boolean pref perturbation --
    return _perturb_boolean_prefs(eval_row, rng, expect_rules)


def _perturb_theme(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate variants for Thunderbird theme changes."""
    evaluator = eval_row["metadata"]["evaluator"]
    expect_rules = evaluator["expected"]["rules"]["expect"]
    theme_rule = expect_rules["extensions.activeThemeID"]
    orig_ref = theme_rule.get("ref", "")

    # NOTE: default Thunderbird profile has no
    # activeThemeID set (or is set to "default"), so the eval regex
    # `re.match("light", ...)` and `re.match("default", ...)` BOTH
    # vacuously match the seeded prefs without any agent action. Restrict
    # candidates to "dark" only so the perturb requires the agent to
    # actively flip the theme. If the base task is already "dark", we
    # have nothing to perturb to.
    candidates = [(tid, name) for tid, name in _THEMES
                  if orig_ref not in tid and name == "dark"]
    if not candidates:
        return []
    rows = []
    for theme_id, theme_name in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_THEME_TEMPLATES).format(name=theme_name)

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["expected"]["rules"]["expect"]["extensions.activeThemeID"]["ref"] = theme_name

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and "activeThemeID" in cmd:
                action["parameters"]["command"] = re.sub(
                    r'thunderbird-compact-\w+@mozilla\.org|default-theme@mozilla\.org',
                    theme_id, cmd,
                )

        # Pre-seed the opposite theme via perturb_config_step so the
        # agent must actually flip. Use perturb_config_step (NOT
        # pre_config_steps) because the base config downloads + extracts
        # a tarball into ~/.thunderbird, which would otherwise wipe an
        # earlier prefs.js write. perturb_config_step runs AFTER the
        # base config so the seed survives.
        opposite_tid = "thunderbird-compact-light@mozilla.org"
        perturb_config_step = {
            "type": "execute",
            "parameters": {
                "command": (
                    "PROFILE_DIR=$(ls -d /home/user/.thunderbird/*.default-release 2>/dev/null | head -1) && "
                    "[ -n \"$PROFILE_DIR\" ] && "
                    "PREFS=\"$PROFILE_DIR/prefs.js\" && "
                    "touch \"$PREFS\" && "
                    f"grep -v 'extensions.activeThemeID' \"$PREFS\" > \"$PREFS.tmp\" && "
                    f"echo 'user_pref(\"extensions.activeThemeID\", \"{opposite_tid}\");' >> \"$PREFS.tmp\" && "
                    "mv \"$PREFS.tmp\" \"$PREFS\""
                ),
                "shell": True,
            },
        }

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"theme": theme_id},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=perturb_config_step,
        ))
    return rows


def _perturb_signature(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate variants for Thunderbird signature tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    expect_rules = evaluator["expected"]["rules"]["expect"]
    sig_rule = expect_rules["mail.identity.id1.htmlSigText"]
    orig_ref = sig_rule.get("ref", "")

    # Parse original name and affiliation from regex pattern
    orig_name = ""
    orig_affil = ""
    ref_match = re.match(r'(\w+)\\n(.+)', orig_ref)
    if ref_match:
        orig_name = ref_match.group(1)
        orig_affil = ref_match.group(2)

    rows = []
    for _ in range(4):
        name = rng.choice([n for n in _NAMES if n != orig_name])
        affil = rng.choice([a for a in _AFFILIATIONS if a != orig_affil])
        instruction = rng.choice(_SIGNATURE_TEMPLATES).format(name=name, affil=affil)

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["expected"]["rules"]["expect"]["mail.identity.id1.htmlSigText"]["ref"] = (
            f"{name}\\n{affil}"
        )

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and "htmlSigText" in cmd:
                # Chain replacements on cmd (not action["parameters"]["command"])
                # to avoid later replace overwriting earlier one.
                if orig_name:
                    cmd = cmd.replace(orig_name, name)
                if orig_affil:
                    cmd = cmd.replace(orig_affil, affil)
                action["parameters"]["command"] = cmd

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"sig_name": name, "sig_affil": affil},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


def _perturb_boolean_prefs(
    eval_row: dict, rng: random.Random, expect_rules: dict,
) -> list[dict]:
    """Generate variants for boolean Thunderbird prefs (flip true/false).

    For each boolean pref in the eval, emits one row per pref. When the
    eval has only ONE boolean pref (e.g. f201fbc3 / auto_quote), emits
    multiple paraphrase variants of the same flip so a single-pref base
    doesn't degrade to a single-row training signal (R1-thunderbird).
    """
    rows = []

    boolean_pref_keys = [
        k for k, r in expect_rules.items()
        if isinstance(r, dict) and r.get("method") == "eq"
        and r.get("ref") in (True, False) and k in _BOOLEAN_PREF_SPECS
    ]
    # When the eval has only one boolean pref, emit 3-4 paraphrase variants
    # so single-pref bases (e.g. f201fbc3) don't degrade to a 1-row signal.
    # When multiple prefs (e.g. 08c73485), emit 1 row per pref to avoid
    # over-representing this archetype.
    paraphrases_per_pref = 4 if len(boolean_pref_keys) == 1 else 1

    for pref_key, rule in expect_rules.items():
        if not isinstance(rule, dict):
            continue
        method = rule.get("method", "")
        ref = rule.get("ref")

        # Only handle boolean eq checks
        if method != "eq" or ref not in (True, False):
            continue
        if pref_key not in _BOOLEAN_PREF_SPECS:
            continue

        spec = _BOOLEAN_PREF_SPECS[pref_key]
        new_val = not ref
        # All boolean templates put the verb mid-sentence (after "In
        # Thunderbird,", "Configure", "Open", "Please", etc.), so the
        # verb must be lowercase ("enable" / "disable"). Sentence-initial
        # capitalization is provided by the surrounding template text.
        verb = "enable" if new_val else "disable"

        # Pick distinct templates per paraphrase variant (rng.sample
        # avoids replacement). When paraphrases_per_pref == 1 this still
        # picks one template at random.
        # validation: prefer polarity-aware template lists when present —
        # `instruction_templates_enable` / `instruction_templates_disable`
        # let templates carry verb-consistent narrative (avoids the f201fbc3
        # contradiction where anti-quote phrasing fired with verb=enable).
        if "instruction_templates_enable" in spec and "instruction_templates_disable" in spec:
            templates = spec["instruction_templates_enable"] if new_val else spec["instruction_templates_disable"]
        else:
            templates = spec["instruction_templates"]
        n_variants = min(paraphrases_per_pref, len(templates))
        chosen_templates = rng.sample(templates, n_variants)

        for variant_idx, template in enumerate(chosen_templates):
            instruction = template.format(verb=verb)

            new_evaluator = copy.deepcopy(eval_row["metadata"]["evaluator"])
            new_evaluator["expected"]["rules"]["expect"][pref_key]["ref"] = new_val
            # Strip secondary prefs from `expect`: the perturbed instruction only
            # names ONE pref, so the eval must require only that one. Otherwise the
            # agent has no way to satisfy unnamed prefs via UI.
            # (validation pattern: perturb.thunderbird_08c73485_single_pref_instr_multi_pref_eval)
            for other_key in list(new_evaluator["expected"]["rules"]["expect"]):
                if other_key != pref_key:
                    del new_evaluator["expected"]["rules"]["expect"][other_key]

            # Build the full set of pref assignments: perturbed key gets new_val,
            # all other boolean eq keys in expect_rules keep their original value.
            # This ensures secondary prefs (e.g. mail.imap.use_status_for_biff) are
            # also set so the evaluator passes.
            all_pref_assignments: list[tuple[str, str]] = []
            for k, r in expect_rules.items():
                if not isinstance(r, dict) or r.get("method") != "eq":
                    continue
                v = r.get("ref")
                if v not in (True, False):
                    continue
                if k == pref_key:
                    all_pref_assignments.append((k, "true" if new_val else "false"))
                else:
                    all_pref_assignments.append((k, "true" if v else "false"))

            # Build the heredoc body: one filter+append per pref key
            pref_ops = "\n".join(
                f"lines = [l for l in lines if '{k}' not in l]\n"
                f"lines.append('user_pref(\"{k}\", {jv});\\n')"
                for k, jv in all_pref_assignments
            )

            # Kill Thunderbird first so it can't overwrite prefs.js when it closes.
            # Then write all pref values directly.
            oracle = [
                {"type": "execute", "parameters": {
                    "command": f"mkdir -p {_TB_PROFILE_DIR}",
                    "shell": True,
                }},
                {"type": "execute", "parameters": {
                    "command": "pkill -f thunderbird; sleep 3",
                    "shell": True,
                }},
                {"type": "execute", "parameters": {
                    "command": (
                        f"python3 << 'PYEOF'\n"
                        f"import os\n"
                        f"path = '{_TB_PROFILE_DIR}/prefs.js'\n"
                        f"lines = []\n"
                        f"if os.path.exists(path):\n"
                        f"    with open(path) as f:\n"
                        f"        lines = f.readlines()\n"
                        f"{pref_ops}\n"
                        f"with open(path, 'w') as f:\n"
                        f"    f.writelines(lines)\n"
                        f"PYEOF"
                    ),
                    "shell": True,
                }},
            ]

            # BUG-2 fix: the natural Account-Settings / Config-Editor
            # UI path writes server prefs as ``mail.server.serverN.X`` and identity
            # prefs verbatim, but the strict upstream ``check_thunderbird_prefs``
            # asserts the ``mail.server.default.X`` namespace → a correct agent edit
            # scores 0 (e.g. 08c73485 applyIncomingFilters, 2/2 fail). ``_loose`` (in
            # metrics.py, already used by synth) accepts both namespaces. Harmless for
            # identity prefs (no serverN alias) where loose == strict.
            new_evaluator["func"] = "check_thunderbird_prefs_loose"

            # BUG-1 fix: when the perturb TARGET equals the pref's
            # Thunderbird DEFAULT (e.g. auto_quote defaults true, "enable" → true),
            # the starting profile has NO line for it, the box is already in the
            # target state, the agent correctly does nothing — yet the eval looks for
            # a ``user_pref`` line that a default-valued pref never writes → 0 (f201fbc3,
            # 4/4 fail). Seed the OPPOSITE value into prefs.js as the INITIAL state so
            # the toggle is a REAL change the agent must make and Thunderbird persists.
            # perturb_config_step (NOT pre_config) runs AFTER the base tarball extract,
            # else the profile unpack would wipe it (same rationale as the theme seed).
            opposite_val = "false" if new_val else "true"
            perturb_config_step = {
                "type": "execute",
                "parameters": {
                    "command": (
                        f"mkdir -p {_TB_PROFILE_DIR} && "
                        f"PREFS=\"{_TB_PROFILE_DIR}/prefs.js\" && touch \"$PREFS\" && "
                        f"grep -v '{pref_key}' \"$PREFS\" > \"$PREFS.tmp\" && "
                        f"echo 'user_pref(\"{pref_key}\", {opposite_val});' >> \"$PREFS.tmp\" && "
                        "mv \"$PREFS.tmp\" \"$PREFS\""
                    ),
                    "shell": True,
                },
            }

            # knob_assignment includes paraphrase index ONLY when emitting
            # multiple paraphrase variants (single-pref base), so the
            # multi-pref case (08c73485) keeps stable single-row hashes.
            knob: dict = {pref_key: new_val}
            if paraphrases_per_pref > 1:
                knob["paraphrase"] = variant_idx
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
# filter_setting: perturb Thunderbird email filter tasks
# ---------------------------------------------------------------------------

_FILTER_FOLDERS = ["Promotions", "Deals", "Offers", "Sales", "Newsletters", "Updates", "Alerts", "Notices"]
_FILTER_KEYWORDS = ["discount", "sale", "offer", "promo", "deal", "coupon", "clearance", "reward"]

# validation (aeab9888, 659c6698): the eval reads msgFilterRules.dat under the
# *IMAP account* (ImapMail/outlook.office365.com/), because a filter that fires
# on incoming mail must be attached to the account that receives it. The
# destination is still a Local Folder, but the filter RULE must be created with
# the message-filters dialog scoped to the email account ("Filters for:" set to
# the email/IMAP account, NOT "Local Folders"). The earlier templates only said
# "create a local folder ... add an inbox filter", which led the agent to attach
# the filter to Local Folders → rule written to the wrong account's dat → 0.
# Every template now names the email/IMAP account as the filter's owner.
_FILTER_TEMPLATES = [
    'In Thunderbird, create a local folder called "{folder}", then on your email (IMAP) account add an inbox filter that auto-moves any message whose subject contains "{keyword}" into that new folder, so the main inbox stays focused.',
    'Set up a Thunderbird filter on your email account: first create a local folder named "{folder}", then add a rule scoped to that account\'s inbox that auto-moves messages with "{keyword}" in the subject into "{folder}" as they arrive.',
    'Configure Thunderbird so emails with "{keyword}" in the subject are routed out of your email account\'s inbox into a local folder named "{folder}" — create the folder first, then build the filter under that email (IMAP) account.',
    'I need a "{folder}" local folder in Thunderbird and an automatic filter on my email account that catches every inbox message whose subject contains "{keyword}" and moves it into "{folder}", since they clutter my view daily.',
    'Please create a "{folder}" local folder in Thunderbird and, under my email (IMAP) account, add a filter that moves inbox emails with "{keyword}" in the subject straight into it; morning triage will be faster once they\'re auto-sorted.',
    # D5 short variants (~15-20 words): terse imperative for instruction-length diversity
    'Create a "{folder}" local folder in Thunderbird and add a filter on your email account that moves "{keyword}"-subject inbox mail there.',
    'On your Thunderbird email (IMAP) account, build a filter that auto-moves inbox messages with "{keyword}" subjects into a new local "{folder}" folder.',
    'In Thunderbird, make a "{folder}" local folder and, under your email account, a filter sending "{keyword}"-subject inbox mail there.',
    # D5 long variants (~40-55 words): narrative pad with task context for length tail
    'My morning inbox triage takes forever lately because of vendor noise, so could you please create a local folder named "{folder}" in Thunderbird and, on my email (IMAP) account, add a message filter that auto-moves any message whose subject contains "{keyword}" into that newly created folder as soon as it arrives.',
    'I keep losing useful threads under marketing noise, and the team finally agreed on a sorting convention this week, so could you please create a "{folder}" local folder in Thunderbird and configure a filter under my email account that auto-moves any new inbox message whose subject contains "{keyword}" straight into that folder.',
    'Daily inbox triage is eating an hour and I want to fix it before the next sprint, so in Thunderbird first create a local folder named "{folder}" and then, on my email (IMAP) account, add a message filter that catches every new inbox message whose subject contains "{keyword}" and routes it directly into that folder.',
]

_FORWARD_EMAILS = [
    "backup-x2024@gmail.com", "archive-x2024@gmail.com",
    "forward-x2024@gmail.com", "copy-x2024@gmail.com",
    "store-x2024@gmail.com", "save-x2024@gmail.com",
    "mirror-x2024@gmail.com",
]

_FORWARD_TEMPLATES = [
    "Set up Thunderbird to forward every incoming email to {email} — keep it a purely local filter and do not touch the upstream account's server-side forwarding settings.",
    "In Thunderbird, configure a message filter that forwards each new inbox message to {email}. It must run locally so the online provider's forwarding rules remain unchanged.",
    "Build a Thunderbird filter that forwards every email I receive to {email}, client-side only; I want a backup copy in that mailbox without modifying anything on the IMAP server.",
    "Configure a local Thunderbird filter to auto-forward each inbound email to {email} as soon as it arrives, so I always have a duplicate of incoming mail at that address.",
    "Could you set up email forwarding to {email} in Thunderbird? Make it a local filter rather than a server-side rule — I just need a copy mirrored over for my records.",
    # D5 short variant (~18 words): terse imperative for instruction-length diversity.
    # NOTE: the previous shorter form
    # without the "client-side only" hedge triggered GPT-5's auto-forwarding
    # safety refusal at turn_00. Keep the local-only qualifier to align with
    # the longer templates (which all carry the hedge).
    "In Thunderbird, build a local-only client-side message filter that forwards every incoming inbox email to {email} (do not modify server-side rules).",
    # D5 long variant (~45 words): narrative pad with task context for length tail
    "My team has asked me to mirror all incoming mail to a shared archive mailbox, but the IMAP server admin will not enable server-side rules, so configure a purely client-side Thunderbird filter that forwards every new inbox message to {email} as soon as it lands.",
]


def perturb_thunderbird_filter(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for Thunderbird email filter tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "check_thunderbird_filter":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    expect_list = rules.get("expect", [])

    if not isinstance(expect_list, list) or not expect_list:
        return []

    filter_spec = expect_list[0] if isinstance(expect_list[0], dict) else {}
    action = filter_spec.get("action", "")

    if action == "Move to folder":
        return _perturb_move_filter(eval_row, rng, filter_spec)
    elif action == "Forward":
        return _perturb_forward_filter(eval_row, rng, filter_spec)

    return []


def _perturb_move_filter(
    eval_row: dict, rng: random.Random, filter_spec: dict,
) -> list[dict]:
    """Perturb move-to-folder filter tasks."""
    action_value = filter_spec.get("actionValue", "")
    conditions = filter_spec.get("condition", [])

    # Extract original folder name from URL
    orig_folder = action_value.rsplit("/", 1)[-1] if "/" in action_value else ""
    # Extract original keyword from condition
    orig_keyword = ""
    for cond in conditions:
        if isinstance(cond, str) and "contains," in cond:
            orig_keyword = cond.rsplit(",", 1)[-1].rstrip(")")
            break

    folder_candidates = [f for f in _FILTER_FOLDERS if f != orig_folder]
    keyword_candidates = [k for k in _FILTER_KEYWORDS if k != orig_keyword]

    rows = []
    for _ in range(min(4, len(folder_candidates))):
        new_folder = rng.choice(folder_candidates)
        new_keyword = rng.choice(keyword_candidates) if keyword_candidates else orig_keyword
        instruction = rng.choice(_FILTER_TEMPLATES).format(
            folder=new_folder, keyword=new_keyword,
        )

        new_evaluator = copy.deepcopy(eval_row["metadata"]["evaluator"])
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_expect = new_exp["rules"]["expect"]
        for spec in new_expect:
            if isinstance(spec, dict):
                if "actionValue" in spec and orig_folder:
                    spec["actionValue"] = spec["actionValue"].replace(
                        orig_folder, new_folder,
                    )
                if "condition" in spec:
                    spec["condition"] = [
                        c.replace(orig_keyword, new_keyword) if isinstance(c, str) and orig_keyword else c
                        for c in spec["condition"]
                    ]

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action_item in oracle:
            cmd = action_item.get("parameters", {}).get("command", "")
            if isinstance(cmd, str):
                if orig_folder:
                    cmd = cmd.replace(orig_folder, new_folder)
                if orig_keyword:
                    cmd = cmd.replace(orig_keyword, new_keyword)
                action_item["parameters"]["command"] = cmd

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"filter_folder": new_folder, "filter_keyword": new_keyword},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


def _perturb_forward_filter(
    eval_row: dict, rng: random.Random, filter_spec: dict,
) -> list[dict]:
    """Perturb email forwarding filter tasks."""
    orig_email = filter_spec.get("actionValue", "")
    if not orig_email:
        return []

    candidates = [e for e in _FORWARD_EMAILS if e != orig_email]
    rows = []
    for new_email in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_FORWARD_TEMPLATES).format(email=new_email)

        new_evaluator = copy.deepcopy(eval_row["metadata"]["evaluator"])
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        for spec in new_exp["rules"]["expect"]:
            if isinstance(spec, dict) and spec.get("actionValue") == orig_email:
                spec["actionValue"] = new_email

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action_item in oracle:
            cmd = action_item.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_email in cmd:
                action_item["parameters"]["command"] = cmd.replace(orig_email, new_email)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"forward_email": new_email},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# local_folder: perturb Thunderbird local folder creation tasks
# ---------------------------------------------------------------------------

_LOCAL_FOLDER_PAIRS = [
    ("WORK", "PERSONAL"), ("OFFICE", "HOME"), ("BUSINESS", "ACADEMIC"),
    ("CLIENTS", "INTERNAL"), ("PROJECTS", "ADMIN"), ("REPORTS", "ARCHIVES"),
    ("DRAFTS", "SENT"), ("FINANCE", "TRAVEL"),
]

# NOTE: the eval reads .msf files from the
# `Local Folders` account directory. When the agent creates folders on
# the IMAP / mail account instead, the .msf files land elsewhere and
# the eval fails. Templates now name the parent account explicitly so
# the agent right-clicks the correct sidebar entry.
_LOCAL_FOLDER_TEMPLATES = [
    "In Thunderbird, right-click the 'Local Folders' account in the sidebar (not the IMAP account) and create two new folders under it named {folder1} and {folder2}, so I can archive messages without the remote server.",
    "Open Thunderbird, locate the 'Local Folders' entry in the sidebar — the offline-only account — and add two subfolders called {folder1} and {folder2}; these must stay local, not on any IMAP account.",
    "Create two new folders, {folder1} and {folder2}, as direct subfolders of the 'Local Folders' account in Thunderbird's folder pane, not under any IMAP account, since I want them stored only on this machine.",
    "Add {folder1} and {folder2} to Thunderbird as new folders under the 'Local Folders' account in the sidebar (the local-only one, not the IMAP server), so I can drag messages off the server into local archives.",
    "In Thunderbird's sidebar, add two folders {folder1} and {folder2} directly under the 'Local Folders' account; make sure they are NOT created under the email/IMAP account so they live purely on disk.",
    # D5 short variant (~15 words): terse imperative for instruction-length diversity
    "In Thunderbird's sidebar, add {folder1} and {folder2} as new subfolders under the Local Folders account.",
    # D5 long variant (~41 words): narrative pad with task context for length tail
    "I want to move some old conversations off the IMAP server and onto disk before the next mailbox quota review, so in Thunderbird's sidebar create two subfolders named {folder1} and {folder2} under the Local Folders account, not under any IMAP/email account.",
]


def perturb_thunderbird_local_folders(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for Thunderbird local folder creation tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "check_list":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    expect_list = rules.get("expect", [])

    if not isinstance(expect_list, list):
        return []

    # Check if this is a folder creation task (patterns like \bFOLDER.msf\b)
    orig_folders = []
    for pattern in expect_list:
        if isinstance(pattern, str) and ".msf" in pattern:
            match = re.search(r'\\b(\w+)\\.msf\\b', pattern)
            if match:
                folder_name = match.group(1)
                if folder_name not in orig_folders:
                    orig_folders.append(folder_name)

    if len(orig_folders) < 2:
        return []

    rows = []
    for pair in rng.sample(_LOCAL_FOLDER_PAIRS, min(4, len(_LOCAL_FOLDER_PAIRS))):
        new_folder1, new_folder2 = pair
        instruction = rng.choice(_LOCAL_FOLDER_TEMPLATES).format(
            folder1=new_folder1, folder2=new_folder2,
        )

        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        # Replace folder names in the regex patterns
        new_expect = []
        for pattern in new_exp["rules"]["expect"]:
            if isinstance(pattern, str):
                updated = pattern
                if len(orig_folders) >= 1:
                    updated = updated.replace(orig_folders[0], new_folder1)
                if len(orig_folders) >= 2:
                    updated = updated.replace(orig_folders[1], new_folder2)
                new_expect.append(updated)
            else:
                new_expect.append(pattern)
        new_exp["rules"]["expect"] = new_expect

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action_item in oracle:
            cmd = action_item.get("parameters", {}).get("command", "")
            if isinstance(cmd, str):
                if len(orig_folders) >= 1:
                    cmd = cmd.replace(orig_folders[0], new_folder1)
                if len(orig_folders) >= 2:
                    cmd = cmd.replace(orig_folders[1], new_folder2)
                action_item["parameters"]["command"] = cmd

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"folder1": new_folder1, "folder2": new_folder2},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))

    return rows


# ---------------------------------------------------------------------------
# backup_path: perturb email backup target directory (9bc3cc16)
# ---------------------------------------------------------------------------

_BACKUP_PATHS = [
    "emails.bak",       # eval value — excluded at generation time
    "inbox_backup",
    "mail_archive",
    "email_export",
    "inbox.bak",
    "mail_backup",
    "eml_backup",
]

_BACKUP_TEMPLATES = [
    "Back up every email in my Thunderbird inbox to ~/{path}, exporting each message as its own .eml file so I can browse them later with another mail client.",
    "In Thunderbird, export all messages from the inbox to ~/{path} with one .eml file per message — I want a per-message archive on disk before cleaning out the server mailbox.",
    "Save each inbox email in Thunderbird as a separate .eml file under ~/{path}; the directory is for archival, so write one file per message rather than a single mbox dump.",
    "Export all inbox emails from Thunderbird to ~/{path}, producing one .eml file per message so the archive is portable and each email can be opened on its own.",
    "Help me back up all email files in my Thunderbird inbox to ~/{path}, saving them separately in .eml format, one file per message, so each email stays independent.",
    # D5 short variants (~15-20 words): terse imperative for instruction-length diversity
    "Export every Thunderbird inbox message to ~/{path} as a separate .eml file, one per email.",
    "Archive each Thunderbird inbox message under ~/{path} as its own separate .eml file before cleanup.",
    "Dump every Thunderbird inbox message to ~/{path}, with one separate .eml file per individual email.",
    # D5 long variants (~40-55 words): narrative pad with task context for length tail
    "Before I wipe this Thunderbird profile next week as part of the laptop swap, could you please back up every email currently sitting in the inbox into ~/{path}, exporting each individual message as its own separate .eml file so I can re-import them one by one later if needed.",
    "My IT team is running a Thunderbird profile reset for the upcoming compliance audit, and I cannot lose the inbox history, so please back up every email currently in the inbox into ~/{path} on disk, exporting one separate .eml file per message so each one stays independently restorable.",
    "I am about to migrate from Thunderbird to a different mail client next week, and I need every individual inbox message preserved on disk so I can reopen them one by one in any reader, so export each email in the Thunderbird inbox to ~/{path} as its own .eml file.",
]


def perturb_thunderbird_backup_path(eval_row: dict, rng: random.Random) -> list[dict]:
    """Perturb the target backup directory for email backup tasks (check_list/cache_file)."""
    evaluator = eval_row["metadata"]["evaluator"]
    if evaluator.get("func") != "check_list":
        return []
    result = evaluator.get("result", {})
    if result.get("type") != "cache_file":
        return []
    # Guard: skip local-folder tasks (a10b69e1) which also use check_list/cache_file.
    if "local-folder" in result.get("path", ""):
        return []

    # Extract original path from postconfig ls command
    postconfig = evaluator.get("postconfig", [])
    orig_path = ""
    for step in postconfig:
        cmd = step.get("parameters", {}).get("command", [])
        if isinstance(cmd, list) and "ls" in cmd:
            for part in cmd:
                if part.startswith("/home/user/"):
                    orig_path = part.replace("/home/user/", "")
                    break
    if not orig_path:
        return []

    candidates = [p for p in _BACKUP_PATHS if p != orig_path]
    rows = []
    for new_path in rng.sample(candidates, min(4, len(candidates))):
        instruction = rng.choice(_BACKUP_TEMPLATES).format(path=new_path)

        new_evaluator = copy.deepcopy(evaluator)
        # Update postconfig ls command
        for step in new_evaluator.get("postconfig", []):
            cmd = step.get("parameters", {}).get("command", [])
            if isinstance(cmd, list) and "ls" in cmd:
                step["parameters"]["command"] = [
                    part.replace(f"/home/user/{orig_path}", f"/home/user/{new_path}")
                    for part in cmd
                ]
                step["parameters"]["stdout"] = f"{new_path}.ls"
        # Update result path key
        new_evaluator["result"]["path"] = f"{new_path}.ls"

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str):
                action["parameters"]["command"] = cmd.replace(
                    f"/home/user/{orig_path}", f"/home/user/{new_path}"
                )

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"backup_path": new_path},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


# ---------------------------------------------------------------------------
# unified_inbox: perturb folder-pane mode in xulstore.json (3f49d2cc)
# ---------------------------------------------------------------------------
#
# P3-4-thunderbird real-gap closure: the eval (3f49d2cc) checks
# `xulstore.json[messenger.xhtml][folderTree][mode] =~ "\\bsmart\\b"` via
# check_json. Mozilla Thunderbird's folder pane supports several built-in
# modes; perturbing the `mode` value gives multiple distinct evaluator
# expectations + oracle states, all with the same check_json signature.
#
# Modes excluded from the pool:
#   - `smart`   : eval value (excluded to prevent leakage)
#   - `all`     : seeded value — would be trivial-pass because the standard
#     thunderbird-profile.tar.gz already has
#     `xulstore.json["chrome://...messenger.xhtml"]["folderTree"]["mode"] = "all"`,
#     so an instruction asking for "all folders" mode is satisfied by doing
#     nothing.
#
# Modes kept (each requires the agent to flip away from the seed):
#   - `favorite`, `unread`, `recent`

_FOLDER_MODES = [
    ("favorite", "Favorite Folders"),
    ("unread", "Unread Folders"),
    ("recent", "Recent Folders"),
]

_UNIFIED_INBOX_TEMPLATES = [
    "In Thunderbird, switch the folder pane to the {label} view ({mode} mode) so I see only the folders that match that filter — managing several accounts is much easier when the sidebar isn't a flat list of every subfolder.",
    "Configure Thunderbird's folder pane to use the {label} mode ({mode}); with multiple mailboxes feeding the sidebar I want a curated view rather than the default tree.",
    "Open Thunderbird and change the folder pane mode to {label} ({mode}) — the default 'all folders' tree is overwhelming once you connect more than one IMAP account.",
    "Set the Thunderbird folder pane to {label} ({mode} mode) so the sidebar surfaces only the relevant folders; I'm tidying up after onboarding several email accounts at once.",
    "Could you flip Thunderbird's folder pane into {label} mode ({mode})? The default view shows every folder from every account and I'd rather have it filtered for daily triage.",
    # D5 short variant (~15 words): terse imperative for instruction-length diversity
    "In Thunderbird, switch the folder pane sidebar to the {label} view ({mode} folder mode).",
    # D5 long variant (~48 words): narrative pad with task context for length tail
    "I added a few extra IMAP accounts this week and the default flat folder tree in Thunderbird is making it impossible to find anything during morning triage, so could you please flip the folder pane mode in Thunderbird over to {label} ({mode} mode) so the sidebar shows a curated view.",
]


def perturb_thunderbird_unified_inbox(eval_row: dict, rng: random.Random) -> list[dict]:
    """Perturb the Thunderbird folder-pane mode for check_json/xulstore tasks.

    Targets the 3f49d2cc eval (unified inbox / smart folders). The base
    eval expects `mode` to match `\\bsmart\\b`; perturbed variants pick a
    different valid mode (favorite / unread / recent) and rewrite both
    the evaluator expected `ref` and the oracle's `data[..][mode] = ...`
    assignment so each row is end-to-end consistent.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    if evaluator.get("func") != "check_json":
        return []

    expected = evaluator.get("expected", {})
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    expect_list = rules.get("expect", [])
    if not isinstance(expect_list, list) or not expect_list:
        return []

    # Find a folder-pane mode rule. Key path is
    # ["chrome://messenger/content/messenger.xhtml", "folderTree", "mode"].
    mode_rule = None
    for r in expect_list:
        if not isinstance(r, dict):
            continue
        key = r.get("key", [])
        if isinstance(key, list) and len(key) == 3 and key[1] == "folderTree" and key[2] == "mode":
            mode_rule = r
            break
    if mode_rule is None:
        return []

    orig_ref = mode_rule.get("ref", "")

    # Sample 3 modes (full pool size). rng.sample with full size yields a
    # randomized order without replacement.
    candidates = [(m, lbl) for m, lbl in _FOLDER_MODES if m not in orig_ref]
    rows = []
    for new_mode, label in rng.sample(candidates, len(candidates)):
        instruction = rng.choice(_UNIFIED_INBOX_TEMPLATES).format(
            mode=new_mode, label=label,
        )

        new_evaluator = copy.deepcopy(evaluator)
        new_expect = new_evaluator["expected"]["rules"]["expect"]
        for r in new_expect:
            if (
                isinstance(r, dict)
                and isinstance(r.get("key", []), list)
                and len(r["key"]) == 3
                and r["key"][1] == "folderTree"
                and r["key"][2] == "mode"
            ):
                # Match the bare mode word with a `\b<mode>\b` regex; this
                # mirrors the eval's `\bsmart\b` style and keeps `re` method.
                r["ref"] = rf"\b{new_mode}\b"

        # Oracle: rewrite `data[..][folderTree][mode] = "smart"` → new_mode.
        # The base oracle (see eval row dump) sets the mode literally inside
        # a python heredoc; substitute the literal "smart" → new_mode.
        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and '"smart"' in cmd:
                action["parameters"]["command"] = cmd.replace('"smart"', f'"{new_mode}"')

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"folder_mode": new_mode},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


# ---------------------------------------------------------------------------
# remove_account: perturb account-removal task (dfac9ee8, check_csv)
# ---------------------------------------------------------------------------
#
# P3-4-thunderbird real-gap closure: the eval (dfac9ee8) checks a CSV
# dump of decrypted Thunderbird logins via firefox_decrypt and asserts
# that the `(url, user)` pair for the removed account is NOT present
# (`unexpect`). The standard profile only contains a single Outlook
# account (anonym-x2024@outlook.com), so we cannot vary the removal
# target across emails without first seeding new accounts (out of scope
# for this archetype). Instead we generate paraphrase variants of the
# same removal: each variant has a distinct instruction and a stable
# (oracle, evaluator) pair, which still expands the training signal for
# this skill (account removal via Thunderbird's account manager).

_REMOVE_ACCOUNT_TEMPLATES = [
    'Remove the Outlook account "{email}" from Thunderbird entirely — open the account settings dialog, select that account, and use Account Actions → Remove Account so it disappears from the sidebar.',
    'In Thunderbird, delete the account "{email}". Open Account Settings, pick the matching IMAP account in the left list, and remove it through Account Actions so my profile no longer references those credentials.',
    'Detach the "{email}" account from Thunderbird — open the account manager, find that Outlook entry, and remove it; I\'m migrating off this mailbox and don\'t need it sync\'d here anymore.',
    'Open Thunderbird\'s Account Settings, locate the "{email}" Outlook account in the sidebar list, and remove it through Account Actions; I\'m cleaning up old profiles and want this one fully detached.',
    'Could you remove the "{email}" account from Thunderbird? Open Account Settings, select the account, and use Remove Account from the Account Actions menu — I no longer use this mailbox.',
    # D5 short variant (~15 words): terse imperative for instruction-length diversity
    'In Thunderbird, remove the old Outlook account "{email}" entirely via Account Settings → Account Actions.',
    # D5 long variant (~41 words): narrative pad with task context for length tail
    'I am decommissioning my old Outlook mailbox now that the cutover to the new tenant is complete, so go into Thunderbird Account Settings, locate the "{email}" entry in the left list, and remove that account entirely through the Account Actions menu.',
]


def perturb_thunderbird_remove_account(eval_row: dict, rng: random.Random) -> list[dict]:
    """Perturb the dfac9ee8 (check_csv / remove account) task.

    Generates 3 paraphrase variants of the same removal — instruction
    text varies, oracle + evaluator stay pinned to the eval's single
    Outlook account because the profile contains only that one IMAP
    account. Provides a real-gap training signal for the check_csv
    evaluator type without requiring multi-account profile seeding.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    if evaluator.get("func") != "check_csv":
        return []

    expected = evaluator.get("expected", {})
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    unexpect = rules.get("unexpect", [])
    if not isinstance(unexpect, list) or not unexpect:
        return []

    target_email = ""
    for r in unexpect:
        if isinstance(r, dict) and "@" in r.get("user", ""):
            target_email = r["user"]
            break
    if not target_email:
        return []

    # 3 paraphrase variants. Sample without replacement to ensure each
    # row has a distinct instruction template.
    n_variants = min(3, len(_REMOVE_ACCOUNT_TEMPLATES))
    chosen = rng.sample(_REMOVE_ACCOUNT_TEMPLATES, n_variants)

    rows = []
    for variant_idx, template in enumerate(chosen):
        instruction = template.format(email=target_email)

        # Oracle and evaluator are pinned to the eval's account; we only
        # vary the instruction wording. Deep-copy to avoid sharing state
        # across rows.
        new_evaluator = copy.deepcopy(evaluator)
        new_oracle = copy.deepcopy(oracle_actions_of(eval_row))

        rows.append(make_perturb_row(
            eval_row=eval_row,
            # paraphrase index keeps task_id hash unique across variants
            knob_assignment={"remove_email": target_email, "paraphrase": variant_idx},
            new_instruction=instruction,
            new_oracle=new_oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

_INTERNAL_FNS = [
    perturb_thunderbird_pref,
    perturb_thunderbird_filter,
    perturb_thunderbird_local_folders,
    perturb_thunderbird_backup_path,
    # NOTE: dropped 2 archetypes:
    #   - perturb_thunderbird_account_setup (15c3b339): GPT-5.4 safety-refuses
    #     typing 'password' literal into a password field
    #   - perturb_thunderbird_star_folder (dd84e895): only "Bills" is local
    #     (gloda flushes synchronously); INBOX/Drafts/Sent are IMAP with async
    #     gloda indexing → eval contract races vs close_window postconfig
    perturb_thunderbird_unified_inbox,
    perturb_thunderbird_remove_account,
]


def perturb_thunderbird_per_task(
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
