"""Synth-specific helpers (Track A only).

Holds dataclasses, regexes, style passes, asset staging, the
`make_synth_row` row-builder, and the `_terminal_preopen_steps`
shell helper. Imported by every `synth/<domain>.py` module.

Run-as-script: this module is import-only; it has no `__main__` entry.
"""

from __future__ import annotations

import hashlib
import pathlib as _pathlib
import random
import re as _re
from dataclasses import dataclass
from typing import Callable

from lite.gym.envs.lite.osworld.src.gen.common import (
    NOISE_CANDIDATES,
)
from lite.gym.envs.lite.osworld import exclude_reasons
from lite.gym.envs.lite.osworld.src.utils.assets import asset_root


def _stable_hash(s: str) -> int:
    """Deterministic 32-bit hash of `s` (across Python invocations).

    Replaces the builtin `hash()` for seeding `random.Random` in
    template `param_fn`s — `hash()` is randomized per-process when
    PYTHONHASHSEED is unset, which makes generator output non-
    deterministic and breaks `test_train_jsonl_idempotent`.
    """
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


@dataclass
class SynthTemplate:
    """Template for generating synthetic training tasks (Track A)."""

    template_id: str  # e.g. "calc_sort"
    domain: str  # e.g. "libreoffice_calc"
    instruction_fn: Callable[[dict], str]  # params -> instruction string
    evaluator_fn: Callable[[dict], dict]  # params -> evaluator dict
    oracle_fn: Callable[[dict], list[dict]]  # params -> oracle_actions list
    postconfig_fn: Callable[[dict], list[dict] | None]  # params -> postconfig or None
    param_fn: Callable[[int], dict]  # seed -> random params dict
    n_rows: int = 30  # how many rows to generate
    setup_class: str = ""  # type of initial env state, e.g. "employee_table", "sales_table"
    eval_class: str = ""  # GUI skill / operation type, e.g. "sort_col", "chart", "bold_text"
    open_command: list[str] | str | None = None  # app launch command appended after synth step


# OSWorld Thunderbird profile used for all synth_thunderbird tasks.
# The tarball extracts to ~/.thunderbird/t5q2a5hp.default-release/ and provides
# a pre-configured account so Thunderbird opens to an inbox instead of the setup wizard.
_TB_PROFILE_URL = (
    "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache"
    "/resolve/main/thunderbird/dd84e895-72fd-4023-a336-97689ded257c"
    "/thunderbird-profile.tar.gz"
)
TB_PROFILE_DIR = "/home/user/.thunderbird/t5q2a5hp.default-release"

# Config steps prepended to every synth task for specific domains (domain → step list).
# Applied first so the environment is ready before per-template setup.
DOMAIN_PRE_CONFIG: dict[str, list[dict]] = {
    "thunderbird": [
        {
            "type": "download",
            "parameters": {"files": [{"url": _TB_PROFILE_URL, "path": "/home/user/thunderbird-profile.tar.gz"}]},
        },
        {
            "type": "execute",
            "parameters": {"command": ["tar", "-xzv", "--recursive-unlink", "-f", "/home/user/thunderbird-profile.tar.gz", "-C", "/home/user/"]},
        },
    ],
}

# Default app launch command per domain — used when neither the template nor its params
# supply an explicit open_command but the domain's app must be open for the agent.
# Each value becomes {"type": "launch", "parameters": {"command": <value>}}.
DOMAIN_DEFAULT_OPEN: dict[str, list[str] | str] = {
    "vlc": ["vlc", "--no-video-title-show"],
    "vs_code": ["code"],
    "chrome": ["google-chrome", "--remote-debugging-port=1337"],
    "thunderbird": ["thunderbird", "-profile", TB_PROFILE_DIR],
    "gimp": ["gimp"],
    "libreoffice_impress": ["libreoffice", "--impress"],
    # libreoffice_calc and libreoffice_writer always use file-specific open steps
}


# Root for asset staging — `_stage_asset(rel, dst)` resolves rel against
# `<asset_root>/<rel>`. Shared with dispatch.py's runtime root via
# ``asset_root()`` (HF-downloaded ``.cache/`` → in-repo ``data/`` fallback), so
# codegen and runtime always agree on where the bundle lives.
_ASSET_ROOT = asset_root()


def _stage_asset(rel: str, dst: str) -> dict:
    """Build a `host_push` config step that copies an asset into the container.

    `rel`  : path relative to `lite/.../data/assets/synth/` (e.g.
             "photos/space/earth-blue-marble-apollo17.jpg").
    `dst`  : container path under `/home/user/`, `/tmp/`, or `/var/tmp/`.

    Asserts the host file exists at codegen time so a typo'd asset path
    doesn't slip through to V2 as a 0-byte container file. The returned
    step is dispatched by `dispatch.py` `host_push` handler — see that
    function for the wire format and write_bytes semantics.
    """
    host_path = _ASSET_ROOT / rel
    if not host_path.is_file():
        raise FileNotFoundError(
            f"_stage_asset: host file not found: {host_path} "
            f"(rel={rel!r}). Verify against assets/synth/MANIFEST.csv."
        )
    # Store the checkout-RELATIVE rel (not the absolute host_path) so the
    # committed data is portable across checkouts; dispatch.py's host_push
    # handler resolves it against the runtime asset root. (Baking the absolute
    # codegen path here made train.synth.jsonl non-portable + broke the
    # idempotency byte-lock off the author's checkout.)
    return {
        "type": "host_push",
        "parameters": {"files": [{
            "host_path": rel,
            "dst": dst,
        }]},
    }


_SAVE_SUFFIXES = _re.compile(
    r'\s*(?:'
    r'Save the (?:file|document|presentation|spreadsheet)\.'
    r'|Save when done\.'
    r'|Save when finished\.'
    r"|Save when you're done\."
    r"|Don't forget to save\."
    r'|Save\.'
    r'|\(?\d+\)\s*[Ss]ave(?:\s+the\s+\w+)?\.?'
    r')\s*$'
)

# Drop "(in|with|using) <App>" trailing/inline phrases — eval rarely names the
# app explicitly when context implies it. Keep "Open <file>" / capitalized
# proper nouns inside URLs intact; only target redundant app-name phrases.
_APP_INLINE = _re.compile(
    r'\s+(?:in|with|using)\s+'
    r'(?:LibreOffice(?:\s+(?:Writer|Calc|Impress))?|'
    r'Microsoft\s+(?:Word|Excel|Office)|'
    r'Google\s+Chrome|Chrome|GIMP|VLC|Thunderbird)\b',
    _re.IGNORECASE,
)

# C1: drop "Open <file>.<ext> on/from the Desktop. " or "Open it from your home
# folder. " preamble. Replace with "the document" / "the spreadsheet" / etc.
# Eval has the file already open ("this document"). Train was 71% has-filepath.
_OPEN_PREAMBLE_DOCX = _re.compile(
    r'^Open\s+(?P<file>\S+?)\.docx\s+(?:on|from)\s+(?:the\s+Desktop|your\s+home\s+folder)\.\s+',
    _re.IGNORECASE,
)
_OPEN_PREAMBLE_XLSX = _re.compile(
    r'^Open\s+(?P<file>\S+?)\.xlsx\s+(?:on|from|in)\s+(?:the\s+Desktop|your\s+home\s+folder)\.\s+',
    _re.IGNORECASE,
)
_OPEN_PREAMBLE_PPTX = _re.compile(
    r'^Open\s+(?P<file>\S+?)\.pptx\s+(?:on|from|in)\s+(?:the\s+Desktop|your\s+home\s+folder)\.\s+',
    _re.IGNORECASE,
)
# C2: drop menu paths ONLY when they form a complete standalone sentence:
# "Use Format > Text > Subscript." (must end with period directly after
# the path, no compound "and Footer > Footer >..." mid-sentence). Stricter
# than v2 which broke "Use Insert > Header and Footer > Footer > Default"
# by stripping just "Use Insert > Header" and leaving the orphaned tail.
# This v3 only matches when the path goes path-element → path-element →
# ...→ "." with NO connecting words like "and".
_MENU_PATH = _re.compile(
    r'(?:^|(?<=\.\s)|(?<=\?\s)|(?<=!\s))'
    r'(?:Use|Via|Through|Go\s+to|Open)\s+'
    r'[A-Z][a-zA-Z]+(?:\s*>\s*[A-Z][a-zA-Z]+)+'
    r'\s*\.\s*',
)


def _apply_style_pass(instruction: str, seed: int) -> str:
    """Reduce style mismatch with eval set.

    Eval has 0% "Save when done", 0% menu paths, ~20% explicit app names.
    Train baseline (validation-post) was 11% / 13% / 37%.  Apply stochastic,
    seed-deterministic transforms so per-template vocab stays diverse but
    aggregate distribution shifts toward eval style.
    """
    rng = random.Random(seed ^ 0xCAFEBABE)
    # C3 (existing): strip trailing "Save…" suffix in ~80% of seeds.
    if rng.random() < 0.8:
        instruction = _SAVE_SUFFIXES.sub('', instruction).rstrip()
        if instruction and not instruction[-1] in '.!?':
            instruction += '.'
    # C5 (existing): drop redundant "in/with/using <App>" phrases in ~50%.
    if rng.random() < 0.5:
        instruction = _APP_INLINE.sub('', instruction)
    # C1 (new): drop "Open <file>.docx on/from the Desktop. " preamble in ~60%
    #   of seeds. Replace with "In the document, " or just leave a clean start.
    #   Eval rarely tells the agent to "open" the file — it's already open.
    if rng.random() < 0.6:
        for pat in (_OPEN_PREAMBLE_DOCX, _OPEN_PREAMBLE_XLSX, _OPEN_PREAMBLE_PPTX):
            new = pat.sub('', instruction)
            if new != instruction:
                # Capitalize first letter of remaining instruction.
                if new and new[0].islower():
                    new = new[0].upper() + new[1:]
                instruction = new
                break
    # C2 (new): drop "Use Format > Text > Subscript" menu paths in ~50% of seeds.
    #   Eval rarely names the menu path; agent should learn UI by intent.
    if rng.random() < 0.5:
        new = _MENU_PATH.sub('', instruction)
        if new and new != instruction:
            # Clean up double-spaces / leading/trailing punct
            new = _re.sub(r'\s{2,}', ' ', new).strip()
            if new and not new[-1] in '.!?':
                new += '.'
            if new and new[0].islower():
                new = new[0].upper() + new[1:]
            instruction = new
    return instruction


def make_synth_row(template: SynthTemplate, seed: int, params: dict) -> dict:
    """Build one train.synth.jsonl row from a template + params."""
    domain_short = template.domain.split("_")[-1]
    task_id = f"synth_{domain_short}_{template.template_id}_{seed:04d}"

    instruction = template.instruction_fn(params)
    instruction = _apply_style_pass(instruction, seed)
    # C4: add a conversational prefix to ~15% of instructions. Train variant
    # lists are already heavily polite (~61%); eval is only 35%. A higher
    # additive ratio (50% / 33%) pushed train polite *up* — wrong direction.
    # 15% is a small additive that gives the model exposure to extra polite
    # vocabulary without inflating the aggregate ratio.
    _pfx_rng = random.Random(seed ^ 0xF00D)
    if _pfx_rng.random() < 0.15:
        _PREFIXES = [
            "Could you help me with this? ",
            "Please help me — ",
            "Can you help me? I need to ",
            "I need some help: ",
            "Hey, could you please ",
        ]
        prefix = _pfx_rng.choice(_PREFIXES)
        if instruction:
            instruction = prefix + instruction[0].lower() + instruction[1:]
    evaluator = template.evaluator_fn(params)
    oracle_actions = template.oracle_fn(params)

    postconfig = template.postconfig_fn(params)
    if postconfig:
        evaluator["postconfig"] = postconfig

    out_path = params.get("out_path", f"/home/user/Desktop/{template.template_id}_{seed:04d}.xlsx")
    expected_path = params.get("expected_path", f"/tmp/expected_{seed:04d}.xlsx")
    # `synth_out` / `synth_expected` only differ from `out_path` / `expected_path`
    # in validation templates where the synth produces an INPUT file (e.g. the
    # source mp4 for vlc.extract_audio_mp3) at a different container path than
    # where the agent saves its OUTPUT. Default to out_path / expected_path so
    # the standard templates (140+) carry no extra knob.
    synth_out      = params.get("synth_out",      out_path)
    synth_expected = params.get("synth_expected", expected_path)

    # Allow full config override (for templates whose setup can't be expressed
    # by a single synth command — kept until generator refactor cycle).
    if "config_override" in params:
        config = list(params["config_override"])
    else:
        # pre_config_steps: domain-level steps prepended first (e.g. download profile)
        config = list(DOMAIN_PRE_CONFIG.get(template.domain, []))
        config.extend(params.get("pre_config_steps", []))
        # Source-state setup is encoded directly in pre_config_steps heredocs
        # and host_push steps (see synth/AGENTS.md for the host-heredoc design).
        # Add any extra config steps (e.g., second-domain file setup for multi_apps)
        for extra_step in params.get("extra_config_steps", []):
            config.append(extra_step)

    # Minimum auto-launch fallback: if no app has been launched yet AND the
    # template/params/domain supplies an `open_command`, append a single bare
    # `launch` step. The per-domain post-launch settle (sleep + activate_window
    # + body-focus etc.) lives in each `synth/<domain>.py` helper and is
    # injected via `params["post_open_config_steps"]` (see the loop below).
    # This builder deliberately does NOT special-case any domain here — the
    # per-domain pattern lives in each `synth/<domain>.py` module.
    already_has_app_open = any(s.get("type") in ("launch", "open") for s in config)
    open_cmd = (
        params.get("open_command")
        or template.open_command
        or (DOMAIN_DEFAULT_OPEN.get(template.domain) if not already_has_app_open else None)
    )
    if open_cmd and not already_has_app_open:
        config.append({"type": "launch", "parameters": {"command": open_cmd}})

    # Per-template post-open hook: runs AFTER the LO launch + activate_window
    # sequence above. Used by writer to click into the document body so the
    # agent's first Ctrl+A actually selects body text (not the first-run
    # Release Notes banner). See libreoffice_writer._make_template.
    for post_step in params.get("post_open_config_steps", []):
        config.append(post_step)

    others = {
        "domain": template.domain,
        "source": f"synthetic:{template.template_id}",
        "oracle_actions": oracle_actions,
        "oracle_after_postconfig": params.get("oracle_after_postconfig", False),
    }
    if params.get("exclude_reason"):
        others["exclude_reason"] = exclude_reasons.validate(params["exclude_reason"])

    # validation-post: max_steps removed from JSONL output. yaml's
    # env_kwargs.max_steps + domain_overrides is the single source of
    # truth for runtime cap (see scripts/configs/*/default/lite.osworld.yaml
    # and lite/gym/envs/lite/osworld/main.py:LiteOsworldEnv.__init__).
    return {
        "task_id": task_id,
        "instruction": instruction,
        "metadata": {
            "others": others,
            "config": config,
            "evaluator": evaluator,
            "noise": NOISE_CANDIDATES.get(template.domain, {"candidates": ["mouse_jitter", "desktop_files"], "max_apply": 2}),
        },
    }


def _terminal_preopen_steps() -> list[dict]:
    """Pre-open gnome-terminal directly (no WM hotkey dependency).

    Matches the upstream eval oracles (eval/os.py + eval/multi_apps.py) which
    launch `gnome-terminal -- bash -ic '...'` rather than relying on
    Ctrl+Alt+T being bound. gnome-terminal is the terminal the
    `vm_terminal_output` getter can read (AT-SPI tree limited to
    gnome-terminal-server), so this also keeps any terminal-output eval
    happy.

    Original validation BUG context still applies: without pre-open, agents
    emit `left_click@dock + type_text` in the same turn; the dock launch
    is async and keystrokes drop on the desktop. Direct launch + sleep +
    activate_window keeps the same pre-open contract.
    """
    return [
        {"type": "execute", "parameters": {
            "command": "DISPLAY=:1 gnome-terminal & sleep 2; true",
            "shell": True,
        }},
        {"type": "activate_window", "parameters": {"window_name": "Terminal"}},
    ]
