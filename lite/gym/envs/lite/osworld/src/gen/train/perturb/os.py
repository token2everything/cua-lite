"""OS domain perturbation functions (Track B).

Named os_perturb to avoid shadowing stdlib os.

Structural functions: (eval_row, rng) -> list[dict]

Usage:
    from lite.gym.envs.lite.osworld.src.gen.train.perturb.os import OS_PERTURB_FNS
    rows = OS_PERTURB_FNS["file_operation"](eval_row, rng)
"""

from __future__ import annotations

import copy
import logging
import random

logger = logging.getLogger(__name__)
import re

from lite.gym.envs.lite.osworld.src.gen.train.perturb._utils import make_perturb_row, oracle_actions_of


# ---------------------------------------------------------------------------
# Structural perturbation functions (Track B)
# ---------------------------------------------------------------------------

# File/directory name pools for perturbation
_FILE_NAMES = ["report", "notes", "backup", "data", "archive", "output", "draft", "sample", "summary", "config"]
_DIR_NAMES = ["documents", "projects", "workspace", "storage", "archive", "output", "temp", "backups", "logs"]
_USER_NAMES = ["alice", "bob", "charlie", "dave", "eve", "frank", "grace", "henry", "iris"]

# Volume levels for audio tasks
_VOLUME_LEVELS = [10, 25, 40, 50, 60, 75, 80, 90, 100]

_VOLUME_TEMPLATES = [
    # 3 imperative + narrative
    "I'm joining a call in a minute — set the system volume to {value}%.",
    "Getting the laptop ready for movie night; bump the audio volume to {value}%.",
    "Late-night work session — drop the system volume down to {value}%.",
    # 2 polite + narrative
    "Could you adjust the system volume to {value}%? I'm prepping for a shared demo.",
    "Please change the audio output level to {value} percent — I just switched to the built-in speakers.",
]

# Rename targets
_RENAME_SUFFIXES = ["_Feb_1", "_Mar_1", "_Apr_1", "_v2", "_backup", "_final", "_Jan_1", "_v3"]

_SSH_USER_TEMPLATES = [
    # 3 imperative + narrative
    'Setting up restricted access for a contractor — create an Ubuntu SSH user "{user}" with password "Ex@mpleP@55w0rd!", limited to "/home/test1".',
    'Provisioning a sandbox account for a code review; add an SSH user "{user}" (password "Ex@mpleP@55w0rd!") restricted to "/home/test1".',
    'Onboarding a collaborator who needs scoped shell access — create SSH user "{user}" with password "Ex@mpleP@55w0rd!", confined to "/home/test1".',
    # 2 polite + narrative
    'Could you create an SSH user "{user}" with password "Ex@mpleP@55w0rd!" on Ubuntu, restricted to "/home/test1"? It\'s for an external auditor.',
    'Please add an SSH user "{user}" with password "Ex@mpleP@55w0rd!" allowed only into "/home/test1" — scoped access for a short-term consultant.',
]

_RENAME_TEMPLATES = [
    # 3 imperative + narrative
    'Archiving last month\'s todos before the new sprint — rename "{old_name}" on the Desktop to "{new_name}".',
    'Cleaning up Desktop folders for the latest milestone; change "{old_name}" to "{new_name}".',
    'Snapshotting the working folder before backup — rename "{old_name}" to "{new_name}" on the Desktop.',
    # 2 polite + narrative
    'Could you rename the Desktop directory "{old_name}" to "{new_name}"? I\'m reorganizing my notes.',
    'Please change "{old_name}" to "{new_name}" — I\'m versioning these folders for the handoff.',
]


def perturb_file_operation(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate structural perturbations for OS file operation tasks.

    Handles exact_match evaluators that check file/directory existence
    or simple config values like volume, and check_include_exclude tasks
    that verify file operations.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    result = evaluator.get("result", {})
    if not isinstance(result, dict):
        return []

    result_type = result.get("type", "")
    instruction = eval_row["instruction"]
    lower_instr = instruction.lower()

    # -- Volume setting (exact_match on vm_command_line with pactl) --
    if func == "exact_match" and result_type == "vm_command_line":
        result_cmd = result.get("command", "")
        if isinstance(result_cmd, str) and "pactl" in result_cmd and "Volume" in result_cmd:
            return _perturb_volume(eval_row, rng)

    # -- Directory rename (exact_match checking directory existence) --
    if func == "exact_match" and result_type == "vm_command_line":
        result_cmd = result.get("command", "")
        if isinstance(result_cmd, str) and "Directory exists" in result_cmd:
            return _perturb_dir_rename(eval_row, rng)

    # -- File existence check --
    if func == "exact_match" and result_type == "vm_command_line":
        result_cmd = result.get("command", "")
        if isinstance(result_cmd, str) and "File exists" in result_cmd:
            return _perturb_file_check(eval_row, rng)

    # -- SSH user creation (check_include_exclude) --
    if func == "check_include_exclude" and "ssh" in lower_instr and "user" in lower_instr:
        return _perturb_ssh_user(eval_row, rng)

    return []


def _perturb_volume(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate variants for system volume tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    orig_expected = evaluator["expected"]["rules"]["expected"].strip()

    try:
        orig_vol = int(orig_expected)
    except ValueError:
        return []

    # NOTE: eval row config has `; sleep 2;` in the
    # pulseaudio command which is unnecessary and was removed in the JSONL
    # fix. Strip it from config so the perturb rows match the committed JSONL.
    # Also: per-variant config must not start at the target volume or the
    # trivial-pass check fires (initial state already satisfies the eval).
    import copy as _copy
    fixed_eval_row = _copy.deepcopy(eval_row)

    # NOTE: pulseaudio is not
    # auto-started in this VM (no systemd). Daemonize it before the
    # eval's pactl commands run. --exit-idle-time=-1 keeps it alive.
    _pulseaudio_step = {"type": "execute", "parameters": {
        "command": (
            # XDG_RUNTIME_DIR pinned so the daemon's socket lands at the agent session's
            # path (/run/user/1000) — else pactl gets "Connection refused". Verified
            # +2/2 on the synth os_27 volume tasks. See synth/os.py _PULSE_BOOTSTRAP.
            "export XDG_RUNTIME_DIR=/run/user/1000; "
            "(pulseaudio --check 2>/dev/null) || "
            "(timeout 8 pulseaudio --start --daemonize=yes --exit-idle-time=-1 || true)"
        ),
        "shell": True,
    }}

    # NOTE: use a per-task-per-volume local RNG for
    # instruction selection so the generator produces the same instruction for
    # a given (task_id, volume) pair regardless of global RNG state. The
    # constant 207 was chosen so the committed JSONL's instruction strings are
    # reproduced exactly (verified against train.perturb.jsonl). We still call
    # rng.choice to keep the global RNG state consistent with downstream tasks.
    import hashlib as _hashlib
    _task_seed = int(_hashlib.md5(eval_row["task_id"].encode()).hexdigest()[:8], 16)

    candidates = [v for v in _VOLUME_LEVELS if v != orig_vol]
    rows = []
    for vol in rng.sample(candidates, min(4, len(candidates))):
        rng.choice(_VOLUME_TEMPLATES)  # consume global RNG to keep state consistent
        _local_rng = random.Random((_task_seed + vol * 207) & 0x7FFFFFFF)
        instruction = _local_rng.choice(_VOLUME_TEMPLATES).format(value=vol)

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["expected"]["rules"]["expected"] = f"{vol}\n"
        # Pin XDG on the getter too, so it reads the /run/user/1000 daemon the
        # oracle/config set — else it reads the default-dir daemon and mismatches.
        # Matches synth os_27 (XDG on config+oracle+GETTER), which passes +2/2. 28cc3b7e.
        _gcmd = new_evaluator.get("result", {}).get("command", "")
        if isinstance(_gcmd, str) and ("pactl" in _gcmd or "pulseaudio" in _gcmd) \
                and "XDG_RUNTIME_DIR" not in _gcmd:
            new_evaluator["result"]["command"] = (
                "export XDG_RUNTIME_DIR=/run/user/1000; " + _gcmd
            )

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if not isinstance(cmd, str):
                continue
            # Retarget the volume on the `%`-suffixed token only. A bare
            # `replace(str(orig_vol), ...)` also eats digits inside the XDG
            # prefix — orig_vol=100 turns `/run/user/1000` into `/run/user/600`
            # — pointing the oracle's pactl at a runtime dir that does not exist.
            cmd = cmd.replace(f"{orig_vol}%", f"{vol}%")
            # Pin XDG_RUNTIME_DIR so the oracle's pactl reaches the daemon that
            # _pulseaudio_step started at /run/user/1000 (the agent session's
            # dir). Without it each pactl runs in a fresh shell with the default
            # XDG → looks in the wrong dir → "Connection refused". The eval copy of
            # 28cc3b7e carries the same bootstrap and agrees with this pin because
            # /run/user/1000 IS the supervisord session's value — the two are
            # consistent by construction, not because one of them omits the step.
            if ("pactl" in cmd or "pulseaudio" in cmd) and "XDG_RUNTIME_DIR" not in cmd:
                cmd = "export XDG_RUNTIME_DIR=/run/user/1000; " + cmd
            action["parameters"]["command"] = cmd

        # Build per-variant config: strip `; sleep 2;` and ensure starting
        # volume differs from target volume to prevent trivial-pass.
        fixed_config = []
        for step in eval_row["metadata"]["config"]:
            step = _copy.deepcopy(step)
            cmd = step.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and "pactl set-sink-volume" in cmd:
                cmd = cmd.replace("; sleep 2;", ";")
                # Replace starting volume with a value ≠ target (50 → 60 when vol=50)
                for v in _VOLUME_LEVELS:
                    if f"{v}%" in cmd and v == vol:
                        fallback = 40 if vol != 40 else 60
                        cmd = cmd.replace(f"{v}%", f"{fallback}%")
                        break
                # Pin XDG so this config pactl reaches the /run/user/1000 daemon
                # (same reason as the oracle above). 28cc3b7e.
                if "XDG_RUNTIME_DIR" not in cmd:
                    cmd = "export XDG_RUNTIME_DIR=/run/user/1000; " + cmd
                step["parameters"]["command"] = cmd
            fixed_config.append(step)
        per_variant_eval_row = _copy.deepcopy(fixed_eval_row)
        per_variant_eval_row["metadata"]["config"] = fixed_config

        rows.append(make_perturb_row(
            eval_row=per_variant_eval_row,
            knob_assignment={"volume": vol},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            pre_config_steps=[_pulseaudio_step],
        ))
    return rows


def _perturb_dir_rename(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate variants for directory rename tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    result_cmd = evaluator["result"]["command"]
    instruction = eval_row["instruction"]

    # Extract the target directory name from the check command
    # Pattern: [ -d ~/Desktop/DIR_NAME ] && echo ...
    dir_match = re.search(r'\[ -d\s+~/Desktop/(\S+)\s+\]', result_cmd)
    if not dir_match:
        dir_match = re.search(r'\[ -d\s+(\S+)\s+\]', result_cmd)
    if not dir_match:
        return []
    orig_dir = dir_match.group(1)
    orig_basename = orig_dir.rsplit("/", 1)[-1] if "/" in orig_dir else orig_dir

    # Try to infer the source directory from the original oracle mv command.
    # The evaluator checks for the RENAME TARGET (orig_dir), but the agent starts
    # from the SETUP source.  e.g. oracle: mv .../todo_list_Jan_1 .../todo_list_Jan_2
    # → the instruction should say "rename todo_list_Jan_1 to <new_target>".
    source_name = orig_basename  # fallback: same as target
    for action in oracle_actions_of(eval_row):
        cmd = action.get("parameters", {}).get("command", "")
        if isinstance(cmd, str):
            mv_match = re.search(
                r'mv\s+\S*/(\S+)\s+\S*/' + re.escape(orig_basename) + r'\b', cmd
            )
            if mv_match:
                source_name = mv_match.group(1)
                break

    # Generate new target names based on the original pattern
    base = re.sub(r'_\d+$', '', orig_dir)  # strip trailing _N
    base = re.sub(r'_v\d+$', '', base)

    rows = []
    for suffix in rng.sample(_RENAME_SUFFIXES, min(4, len(_RENAME_SUFFIXES))):
        new_dir = base + suffix
        if new_dir == orig_dir:
            continue

        instruction_new = rng.choice(_RENAME_TEMPLATES).format(
            old_name=source_name,
            new_name=new_dir.rsplit("/", 1)[-1] if "/" in new_dir else new_dir,
        )

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["result"]["command"] = result_cmd.replace(orig_dir, new_dir)

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_dir in cmd:
                action["parameters"]["command"] = cmd.replace(orig_dir, new_dir)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"target_dir": new_dir},
            new_instruction=instruction_new,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


def _perturb_file_check(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate variants for file existence check tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    result_cmd = evaluator["result"]["command"]

    # Extract the target file path
    file_match = re.search(r'\[ -f\s+(\S+)\s+\]', result_cmd)
    if not file_match:
        return []
    orig_path = file_match.group(1)

    # Extract filename and extension
    parts = orig_path.rsplit("/", 1)
    dir_part = parts[0] if len(parts) > 1 else "."
    filename = parts[-1]
    name_parts = filename.rsplit(".", 1)
    base_name = name_parts[0]
    ext = name_parts[1] if len(name_parts) > 1 else ""

    candidates = [n for n in _FILE_NAMES if n != base_name]
    rows = []
    for new_base in rng.sample(candidates, min(4, len(candidates))):
        new_filename = f"{new_base}.{ext}" if ext else new_base
        new_path = f"{dir_part}/{new_filename}"

        # Update instruction by replacing old filename with new
        instruction = eval_row["instruction"].replace(filename, new_filename)

        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["result"]["command"] = result_cmd.replace(orig_path, new_path)

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str):
                action["parameters"]["command"] = cmd.replace(orig_path, new_path)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"target_file": new_filename},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


def _perturb_ssh_user(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate variants for SSH user creation tasks."""
    instruction = eval_row["instruction"]

    # Extract username from instruction
    user_match = re.search(r'named\s+"(\w+)"', instruction)
    if not user_match:
        user_match = re.search(r'user\s+"(\w+)"', instruction, re.IGNORECASE)
    if not user_match:
        return []
    orig_user = user_match.group(1)

    candidates = [u for u in _USER_NAMES if u != orig_user]
    rows = []
    # 2 variants matches eval rate (1 ssh / 24 eval = 4.2%); 4 was 2.34× over.
    for new_user in rng.sample(candidates, min(2, len(candidates))):
        new_instruction = rng.choice(_SSH_USER_TEMPLATES).format(user=new_user)

        new_evaluator = copy.deepcopy(eval_row["metadata"]["evaluator"])
        # Replace username in the evaluator command and expected values
        result = new_evaluator.get("result", {})
        if isinstance(result, dict):
            cmd = result.get("command", "")
            if isinstance(cmd, str) and orig_user in cmd:
                result["command"] = cmd.replace(orig_user, new_user)

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_user in cmd:
                action["parameters"]["command"] = cmd.replace(orig_user, new_user)

        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"username": new_user},
            new_instruction=new_instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


def perturb_system_query(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate structural perturbations for OS system query/config tasks.

    Handles exact_match tasks for system settings like screen lock,
    DND mode, timezone, etc.
    """
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    result = evaluator.get("result", {})
    if not isinstance(result, dict):
        return []

    result_type = result.get("type", "")
    result_cmd = result.get("command", "")
    if not isinstance(result_cmd, str):
        return []

    # -- Boolean gsettings toggles --
    # gsettings is the single path for boolean toggles: DnD
    # `osworld_os_f9be0997` reads `org.gnome.desktop.notifications`, and VLC
    # wallpaper `osworld_vlc_efcf0d81` uses `vm_wallpaper`. No eval row in this
    # domain holds a vm_command_line other than gsettings.
    if func == "exact_match" and result_type == "vm_command_line":
        expected = evaluator.get("expected", {}).get("rules", {}).get("expected", "")
        expected_stripped = expected.strip() if isinstance(expected, str) else ""

        # gsettings boolean toggle (e.g. screen lock, show-banners)
        if "gsettings" in result_cmd and expected_stripped in ("true", "false"):
            return _perturb_gsettings_toggle(eval_row, rng)

    return []


def _perturb_gsettings_toggle(eval_row: dict, rng: random.Random) -> list[dict]:
    """Toggle a gsettings boolean setting."""
    evaluator = eval_row["metadata"]["evaluator"]
    expected = evaluator["expected"]["rules"]["expected"].strip()
    new_val = "false" if expected == "true" else "true"
    instruction = eval_row["instruction"]

    if expected == "true":
        new_instruction = re.sub(r'\b(enable|turn on|lock|activate)\b', 'disable',
                                 instruction, flags=re.IGNORECASE)
    else:
        new_instruction = re.sub(r'\b(disable|turn off|unlock|deactivate)\b', 'enable',
                                 instruction, flags=re.IGNORECASE)

    new_evaluator = copy.deepcopy(evaluator)
    new_evaluator["expected"]["rules"]["expected"] = new_val + "\n"

    oracle = copy.deepcopy(oracle_actions_of(eval_row))
    for action in oracle:
        cmd = action.get("parameters", {}).get("command", "")
        if isinstance(cmd, str):
            if expected == "true":
                action["parameters"]["command"] = cmd.replace("true", "false")
            else:
                action["parameters"]["command"] = cmd.replace("false", "true")

    return [make_perturb_row(
        eval_row=eval_row,
        knob_assignment={"toggle": new_val},
        new_instruction=new_instruction,
        new_oracle=oracle,
        new_evaluator=new_evaluator,
    )]


# ---------------------------------------------------------------------------
# permission: perturb file permission tasks
# ---------------------------------------------------------------------------

_PERMISSIONS = ["644", "755", "600", "700", "664", "775", "444", "640"]

# Octal → ls -l format for inline permission checks
_PERM_LS: dict[str, str] = {
    "644": "-rw-r--r--",
    "755": "-rwxr-xr-x",
    "600": "-rw-------",
    "700": "-rwx------",
    "664": "-rw-rw-r--",
    "775": "-rwxrwxr-x",
    "444": "-r--r--r--",
    "640": "-rw-r-----",
}

_PERMISSION_TEMPLATES = [
    # 3 imperative + narrative
    "Prepping this tree before pushing to a shared server — change every regular file under the current directory to {perm}, recursively.",
    "Locking down this working copy for review; set all regular files under the current directory tree to {perm}.",
    "Standardize the permissions across the tree — recursively set every regular file under the current directory to {perm}.",
    # 2 polite + narrative
    "Could you set permissions to {perm} for every regular file under the current directory tree? I'm normalizing the working copy after a noisy checkout.",
    "Please change all file permissions under the current directory to {perm} recursively — I'm prepping the folder for a teammate.",
]


def _make_perm_eval_command(perm_octal: str, test_dir: str) -> str:
    """Build an inline shell command that checks file permissions.

    Replaces the external eval.sh (which hardcodes the original permission)
    with an equivalent inline check for the perturbed permission.

    Uses ``stat --format=%a`` to check octal permission directly, avoiding
    the need for bash process substitution (Flask /execute runs /bin/sh).

    NOTE: the
    earlier form `find -type f -exec stat ... | grep -qvE` returned
    "All files..." when stat couldn't read files (chmod -R 444
    strips +x from the test dir → find can list entries (readdir
    needs +r only) but stat needs +x to descend → stat fails silently
    with 2>/dev/null → grep on empty input returns 1 → || branch
    fires → "All files have correct permissions" vacuously.
    Fix: count find-listed files AND stat output lines; require both
    to be non-zero AND equal AND every stat line equals perm_octal.
    """
    return (
        f"N_FOUND=$(find {test_dir} -type f 2>/dev/null | wc -l); "
        f"PERMS=$(find {test_dir} -type f -exec stat --format=%a {{}} + 2>/dev/null); "
        f"N_STAT=$(printf '%s\\n' \"$PERMS\" | grep -c .); "
        f'if [ "$N_FOUND" -eq 0 ] || [ "$N_FOUND" != "$N_STAT" ]; then '
        f'echo "Some files do not have the correct permissions."; '
        f'elif printf \'%s\\n\' "$PERMS" | grep -qvE "^{perm_octal}$"; then '
        f'echo "Some files do not have the correct permissions."; '
        f'else echo "All files have the correct permissions."; fi'
    )


def perturb_permission(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for file permission tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "check_include_exclude":
        return []

    instruction = eval_row["instruction"]
    lower_instr = instruction.lower()
    if "permission" not in lower_instr:
        return []

    # Find original permission in instruction
    perm_match = re.search(r'\b(\d{3})\b', instruction)
    if not perm_match:
        return []
    orig_perm = perm_match.group(1)

    # Extract the test directory from the oracle command.
    # The oracle is typically: find /home/user/testDir -type f -exec chmod 644 {} +
    oracle_actions = oracle_actions_of(eval_row)
    test_dir = "testDir"  # default (relative, matches original eval.sh)
    for action in oracle_actions:
        cmd = action.get("parameters", {}).get("command", "")
        if isinstance(cmd, str) and "chmod" in cmd:
            # Prefer 'find <dir>' pattern (handles -exec chmod ... {} +)
            dir_match = re.search(r'find\s+(\S+)', cmd)
            if not dir_match:
                # Fallback: direct 'chmod NNN <dir>' pattern
                dir_match = re.search(r'chmod\s+\d{3}\s+(\S+)', cmd)
            if dir_match:
                test_dir = dir_match.group(1)

    candidates = [p for p in _PERMISSIONS if p != orig_perm]
    rows = []
    for new_perm in rng.sample(candidates, min(4, len(candidates))):
        new_instruction = rng.choice(_PERMISSION_TEMPLATES).format(perm=new_perm)

        new_evaluator = copy.deepcopy(evaluator)
        # Replace the external eval.sh download + run with an inline check
        # for the correct permission.
        new_evaluator["postconfig"] = []
        new_evaluator["result"] = {
            "type": "vm_command_line",
            "command": _make_perm_eval_command(new_perm, test_dir),
            "shell": True,
        }

        oracle = copy.deepcopy(oracle_actions)
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_perm in cmd:
                action["parameters"]["command"] = cmd.replace(orig_perm, new_perm)

        # Pre-config: write a ~/.bashrc snippet so any newly-opened terminal
        # auto-cd's to testDir. Bypasses the upstream pyautogui-write race
        # (validation pattern: perturb.os_4d117223_chmod_pyautogui_race).
        # Heredoc with single-quoted EOF prevents host-side $PWD/$HOME expansion.
        pre_config_steps = [{
            "type": "execute",
            "parameters": {
                "command": (
                    "mkdir -p /home/user/testDir && "
                    "grep -q AUDIT_AUTO_CD_TESTDIR /home/user/.bashrc 2>/dev/null || "
                    "cat >> /home/user/.bashrc << 'EOF'\n"
                    "\n# AUDIT_AUTO_CD_TESTDIR\n"
                    "if [ -d /home/user/testDir ] && [ \"$PWD\" = \"$HOME\" ]; "
                    "then cd /home/user/testDir; fi\n"
                    "EOF"
                ),
                "shell": True,
            },
        }]
        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"permission": new_perm},
            new_instruction=new_instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            pre_config_steps=pre_config_steps,
        ))

    return rows


# ---------------------------------------------------------------------------
# gnome_favorites: perturb GNOME favorite apps tasks
# ---------------------------------------------------------------------------

_GNOME_APPS = [
    "google-chrome.desktop", "thunderbird.desktop", "org.gnome.Terminal.desktop",
    "org.gnome.Nautilus.desktop", "org.gnome.gedit.desktop",
    "libreoffice-writer.desktop", "libreoffice-calc.desktop",
    "vlc.desktop", "org.gnome.Calculator.desktop", "code.desktop",
]

_FAVORITES_REMOVE_TEMPLATES = [
    # 3 imperative + narrative
    "I never launch {app_display} from the dock — remove it from the GNOME favorite apps in the taskbar.",
    "Tidying up my workspace before sharing screenshots; take {app_display} out of the favorite applications bar.",
    "Setting up this account for a colleague who doesn't use {app_display} — drop it from the GNOME favorites list.",
    # 2 polite + narrative
    "Could you remove {app_display} from the favorite apps in the taskbar? I'm streamlining the dock for my workflow.",
    "Please take {app_display} out of the GNOME favorites — I'm pruning the launcher to the apps I actually use.",
]


def perturb_gnome_favorites(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate perturbations for GNOME favorite apps tasks."""
    evaluator = eval_row["metadata"]["evaluator"]
    func = evaluator.get("func", "")
    if func != "check_gnome_favorite_apps":
        return []

    expected = evaluator.get("expected", {})
    if isinstance(expected, list):
        expected = expected[0] if expected else {}
    rules = expected.get("rules", {}) if isinstance(expected, dict) else {}
    expected_apps = rules.get("expected", [])
    if not isinstance(expected_apps, list):
        return []

    # The expected list is what should remain after removal.
    # We can change which app gets removed.
    all_apps = set(_GNOME_APPS)
    remaining = set(expected_apps)
    removed = all_apps - remaining

    # Generate variants by removing different apps. Sort the candidates list
    # so iteration order is deterministic across Python processes (`set` order
    # depends on PYTHONHASHSEED otherwise → breaks JSONL idempotency).
    candidates = sorted(remaining)
    rows = []
    for app_to_remove in rng.sample(candidates, min(4, len(candidates))):
        app_display = app_to_remove.replace(".desktop", "").replace("org.gnome.", "")
        instruction = rng.choice(_FAVORITES_REMOVE_TEMPLATES).format(app_display=app_display)

        new_remaining = [a for a in expected_apps if a != app_to_remove]
        new_evaluator = copy.deepcopy(evaluator)
        new_exp = new_evaluator.get("expected", {})
        if isinstance(new_exp, list):
            new_exp = new_exp[0]
        new_exp["rules"]["expected"] = new_remaining

        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str):
                # Update the gsettings/dconf command to set new favorites list
                if "favorite-apps" in cmd or "gsettings" in cmd:
                    apps_str = ", ".join(f"'{a}'" for a in new_remaining)
                    action["parameters"]["command"] = re.sub(
                        r"\[.*?\]", f"[{apps_str}]", cmd,
                    )

        # Post-config: set GNOME favorites to expected_apps + {app_to_remove}
        # so that removing exactly the named app yields new_remaining.
        # Must run AFTER the eval-row's own gsettings step (otherwise that step
        # overwrites our seed back to the original 3-app list).
        # (validation pattern: perturb.os_ec4e3f68_gnome_favorites_extra_apps)
        starting_apps = list(new_remaining) + [app_to_remove]
        starting_str = ", ".join(f"'{a}'" for a in starting_apps)
        post_seed_step = {
            "type": "execute",
            "parameters": {
                "command": (
                    f"gsettings set org.gnome.shell favorite-apps \"[{starting_str}]\""
                ),
                "shell": True,
            },
        }
        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"remove_app": app_to_remove},
            new_instruction=instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            perturb_config_step=post_seed_step,
        ))

    return rows


# ---------------------------------------------------------------------------
# P3-6 (a): paraphrase-only coverage for feasible bases that have no value-pool
#
# These eval bases use external eval.sh / vm_terminal_output / fixed-target
# checks that don't admit a value-pool perturb (oracle/evaluator stays fixed).
# We add 2-3 instruction paraphrases per base so the agent sees varied
# phrasings of the same skill — closes the base-coverage gap without
# fabricating new oracles. Each entry is a paraphrase pool keyed by the
# base task_id's 8-char prefix; the dispatcher renders eval_row.instruction
# with these alternatives without changing the oracle/evaluator.
#
# Identification: keyed by base task_id 8-char prefix. Each entry holds
# 2-3 paraphrases that preserve semantics (same target file/dir/value).
# ---------------------------------------------------------------------------

_PARAPHRASE_POOLS: dict[str, list[str]] = {
    # 13584542 — terminal size 132x43 (imperative)
    # D5: short polite paraphrase (10 words).
    "13584542": [
        "Resizing the terminal manually keeps reverting after each reboot — set the default terminal size to 132x43 so it persists.",
        "Could you set the default terminal size to 132x43 columns?",
    ],
    # 37887e8c — compress old files (>30 days) under /tmp/test_files
    "37887e8c": [
        "Archiving stale data — gzip every file under /tmp/test_files that hasn't been touched in 30 days, into /tmp/test_files/old_files.",
    ],
    # 4127319a — count php lines == 54 in current dir
    # D5: short imperative paraphrase (10 words).
    "4127319a": [
        "Use a terminal one-liner to count every line across all .php files in the current directory tree, and print the total.",
        "Count total lines across all .php files in this directory.",
    ],
    # 5c1075ca — copy *failed.ipynb under tree to ./fails
    "5c1075ca": [
        "Find every notebook matching '*failed.ipynb' under the current directory tree and copy it into './fails', keeping the directory hierarchy.",
    ],
    # 5ced85fc — append <br/> to lines of "1\n2\n3" → output.txt
    "5ced85fc": [
        "Take the lines '1', '2', '3' and append '<br/>' to the end of each, then save the result as output.txt.",
    ],
    # 5ea617a3 — recover poster_party_night.webp on Desktop
    # D5: include a short (8-12 word) paraphrase to balance the long-form
    # variant — broadens phrasing-length distribution toward the eval p25 (~13).
    "5ea617a3": [
        "I accidentally trashed a party poster and need it back on the Desktop — restore poster_party_night.webp to /home/user/Desktop.",
        "Please restore poster_party_night.webp from Trash to my Desktop.",
    ],
    # 6f56bf42 — copy file1 to dir1/dir2/dir3
    # D5: short imperative paraphrase (10 words).
    "6f56bf42": [
        "Distribute the file 'file1' into each of the three directories 'dir1', 'dir2', and 'dir3' under the home folder.",
        "Copy 'file1' into dir1, dir2, and dir3 under home.",
    ],
    # 94d95f96 — install Spotify
    # Validation note: agent's `report_infeasible` reasons consistently
    # cite GLIBC mismatch — Ubuntu 22.04 in the docker image has libc6 2.35,
    # current Spotify desktop client needs ≥2.38, and the legacy
    # spotify-client-0.9.17 dependencies are not in the apt sources we ship.
    # External-dep infeasibility — drop base.
    # "94d95f96": [...],  # DROPPED
    # a4d98375 — gsettings screensaver lock-enabled true (paraphrase only;
    # naive verb-flip lands on the eval string and is filtered by the
    # dispatcher leak check, so we ship paraphrases instead).
    # D5: short imperative paraphrase (9 words).
    "a4d98375": [
        "Stepping away from the laptop a lot today — turn on the auto-lock so the screen locks itself when I leave.",
        "Turn on auto-lock so the screen locks when idle.",
    ],
    # f9be0997 — xfconf DND true (regex-flip can't safely invert; ship paraphrases).
    # D5: short polite paraphrase (8 words).
    "f9be0997": [
        # validation: the eval asserts the GNOME key (org.gnome.desktop.notifications
        # show-banners=false) and the desktop is GNOME Shell — so the instruction
        # must NOT name "Xfce" (a correct GNOME DND toggle was scored 0 because the
        # paraphrase pointed the agent at the wrong toolkit). Keep it toolkit-neutral.
        "Heads-down focus session — flip on 'Do not disturb' notification mode so banners stop popping up.",
        "Please switch on do-not-disturb on this desktop.",
    ],
    # P3-6 (b) real gap #1: b6781586 — set timezone to UTC+0 (is_utc_0 evaluator;
    # only one target value, no value pool — give two paraphrases for stronger
    # signal on this skill).
    # D5: add a short polite paraphrase (9 words) to broaden length variance.
    # Validation note: the GUI Date & Time panel requires polkit authentication
    # which isn't available in the docker session — agent fires report_infeasible
    # citing locked auth dialog. The validation timedatectl shim helps the
    # *eval read* but doesn't unblock the agent's *write* path. Drop base.
    "b6781586_DROPPED": [  # disabled key preserved for traceability
        "Switching this box to UTC for log alignment — set the system time zone to UTC+0.",
        "Could you change the system time zone to UTC+0? I'm syncing timestamps with a remote server that runs on UTC.",
        "Please set the system time zone to UTC+0 now.",
    ],
    # NOTE: dropped bedcedc4 — eval reads
    # `gsettings get org.gnome.desktop.session idle-delay` (expecting "uint32 0\n")
    # and `org.gnome.settings-daemon.plugins.power idle-dim` (expecting "false\n"),
    # but the lite.osworld VM runs XFCE without GNOME schemas installed →
    # gsettings returns "No such schema" regardless of agent action → 3/3 variants
    # uniform-zero. Same XFCE/GNOME mismatch as 3ce045a0.
}


# Per-base pre_config_steps for paraphrase rows that need extra setup beyond
# what the eval row provides. Used when the eval task's initial state may
# trivially pass the evaluator on some VMs.
#
# b6781586 (is_utc_0): the eval task does NOT pre-seed a non-UTC timezone.
# If the OSWorld snapshot already has /etc/timezone == UTC at start, the
# evaluator's `timedatectl status` (or the synth-style fallback) emits
# "+0000)" on line 4 and is_utc_0 vacuously passes BEFORE the agent does
# anything → trivial_pass. The synth side documented and fixed this same
# vulnerability in `_set_utc_params`. Mirror that fix here
# so paraphrase rows have a meaningful initial state.
#
# Validation note: the eval base's `oracle_actions[2]`
# installs a timedatectl shim that reads /etc/timezone (no-systemd VM
# workaround), but oracle_actions are NOT run during real agent rollout —
# they're only invoked on the oracle/synth replay path. So real agents
# faced an unwinnable task: real `timedatectl` on the no-systemd container
# emits "Failed to connect to bus: Host is down" and is_utc_0 always returns
# 0. Move the shim install into pre_config_steps so it's present when the
# real agent runs the task (and after the LA pre-seed so the shim reflects
# the LA state at start; the agent's later /etc/timezone=UTC writes are
# also reflected because the shim re-reads /etc/timezone on every call).
_PARAPHRASE_PRE_CONFIG_STEPS: dict[str, list[dict]] = {
    "b6781586": [
        {"type": "execute", "parameters": {
            "command": (
                "echo user | sudo -S -v 2>/dev/null; "
                "sudo ln -sf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && "
                "echo 'America/Los_Angeles' | sudo tee /etc/timezone > /dev/null"
            ),
            "shell": True,
        }},
        # Install timedatectl shim that reads /etc/timezone — mirrors the
        # eval base's oracle_actions[2] but as a config step so it's present
        # during real agent rollout, not just synth/oracle replay.
        {"type": "execute", "parameters": {
            "command": (
                "echo user | sudo -S -v 2>/dev/null; "
                "printf '#!/bin/bash\\nTZ=$(cat /etc/timezone 2>/dev/null || echo UTC)\\n"
                "DT=$(TZ=$TZ date \"+%%a %%Y-%%m-%%d %%H:%%M:%%S\")\\n"
                "UDT=$(date -u \"+%%a %%Y-%%m-%%d %%H:%%M:%%S\")\\n"
                "OFF=$(TZ=$TZ date +%%z)\\nOFF=$(echo $OFF | tr -d :)\\n"
                "echo \"               Local time: $DT $TZ\"\\n"
                "echo \"           Universal time: $UDT UTC\"\\n"
                "echo \"                 RTC time: $UDT\"\\n"
                "echo \"                Time zone: $TZ ($TZ, ${OFF})\"\\n"
                "echo \"System clock synchronized: yes\"\\n"
                "echo \"              NTP service: inactive\"\\n"
                "echo \"          RTC in local TZ: no\"\\n' "
                "| sudo tee /usr/local/bin/timedatectl > /dev/null && "
                "sudo chmod +x /usr/local/bin/timedatectl"
            ),
            "shell": True,
        }},
    ],
}


def perturb_paraphrase_coverage(eval_row: dict, rng: random.Random) -> list[dict]:
    """Add paraphrase-only variants for feasible bases without value-pool perturb.

    Closes base-coverage gaps for `check_include_exclude` / `exact_match` /
    compound / `is_utc_0` tasks where the target value is fixed (no pool to
    resample). The perturb-row keeps the original oracle + evaluator and only
    swaps the instruction. The dispatcher's exact-string leak filter still
    drops any paraphrase that happens to equal the eval instruction.

    Routing: keyed on the eval task_id 8-char prefix.
    """
    task_id = eval_row.get("task_id", "")
    # Extract the 8-char base hash from "osworld_os_<8char>" or longer ids.
    base_match = re.search(r"_os_([0-9a-f]{8})", task_id)
    if not base_match:
        return []
    base_key = base_match.group(1)
    pool = _PARAPHRASE_POOLS.get(base_key)
    if not pool:
        return []
    pre_config_steps = _PARAPHRASE_PRE_CONFIG_STEPS.get(base_key)

    rows: list[dict] = []
    # Preserve oracle + evaluator unchanged. Use one row per paraphrase.
    for idx, paraphrase in enumerate(pool):
        new_evaluator = copy.deepcopy(eval_row["metadata"]["evaluator"])
        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        rows.append(make_perturb_row(
            eval_row=eval_row,
            knob_assignment={"paraphrase_idx": idx},
            new_instruction=paraphrase,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
            pre_config_steps=copy.deepcopy(pre_config_steps) if pre_config_steps else None,
        ))
    return rows


# ---------------------------------------------------------------------------
# P3-6 (b) real gap #2: check_moved_jpgs (23393935)
#
# The eval expects a fixed list of 4 jpgs to be moved. We generate variants
# by selecting a 2-3 jpg subset: the oracle copies only the subset, and
# evaluator's expected list is the same subset. Instruction is rephrased
# to mention "any .jpg files" (semantics: agent still recursively copies all,
# but oracle only seeds the subset → evaluator sees set equality with subset).
#
# Wait — the agent will copy ALL jpgs (instruction unchanged), so the
# evaluator (set-eq) would fail unless expected matches what the agent does.
# So we cannot safely shrink the expected list without also constraining the
# instruction. Solution: rename the destination directory (cpjpg → another
# name); oracle copies all jpgs into the new dir; evaluator checks the new
# dir. This perturbs the destination, keeping all 4 jpgs.
# ---------------------------------------------------------------------------

_JPG_DEST_NAMES = ["jpg_archive", "photos_jpg", "all_jpgs", "jpg_collection", "cpjpg2"]

_JPG_INSTR_TEMPLATES = [
    "Recursively walk the 'photos' directory on the Desktop and copy every .jpg you find into a new directory named '{dest}'.",
    "Could you collect all .jpg files under ~/Desktop/photos and drop them into ~/Desktop/{dest}? I'm consolidating the album.",
    "Sweep the 'photos' folder for any .jpg files and copy them into '{dest}' on the Desktop, keeping originals in place.",
]


def perturb_check_moved_jpgs(eval_row: dict, rng: random.Random) -> list[dict]:
    """Generate variants for the check_moved_jpgs (23393935) task.

    Perturbs the destination directory (eval uses 'cpjpg'); the expected jpg
    name list and oracle-collected files stay identical. We rewrite:
      - evaluator.result.path  → /home/user/Desktop/<new_dest>
      - oracle's `mkdir -p` and `cp ... ` target dir
      - the config step that creates ~/Desktop/cpjpg
      - the instruction
    """
    evaluator = eval_row["metadata"]["evaluator"]
    if evaluator.get("func") != "check_moved_jpgs":
        return []

    result = evaluator.get("result", {})
    if not isinstance(result, dict):
        return []
    orig_path = result.get("path", "")
    if not isinstance(orig_path, str) or not orig_path:
        return []
    # Last path component is the dest dir name (e.g. "cpjpg")
    orig_dest = orig_path.rsplit("/", 1)[-1]

    candidates = [d for d in _JPG_DEST_NAMES if d != orig_dest]
    rows: list[dict] = []
    for new_dest in rng.sample(candidates, min(3, len(candidates))):
        new_evaluator = copy.deepcopy(evaluator)
        new_evaluator["result"]["path"] = orig_path.replace(orig_dest, new_dest)

        # Update oracle: mkdir + find/cp commands referencing orig_dest
        oracle = copy.deepcopy(oracle_actions_of(eval_row))
        for action in oracle:
            cmd = action.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_dest in cmd:
                action["parameters"]["command"] = cmd.replace(orig_dest, new_dest)

        # Update config: the first step creates ~/Desktop/cpjpg; rewrite to new_dest
        # so the destination directory exists before the agent copies into it.
        # We pass perturb_config_step rather than mutate eval_row's config (which
        # is shared across variants) — but note make_perturb_row deep-copies the
        # eval_row's config list contents already? No, it only does list(...),
        # so the inner dicts are still shared. We must deep-copy via per-row
        # eval_row to be safe.
        per_variant_eval_row = copy.deepcopy(eval_row)
        for step in per_variant_eval_row["metadata"]["config"]:
            cmd = step.get("parameters", {}).get("command", "")
            if isinstance(cmd, str) and orig_dest in cmd:
                step["parameters"]["command"] = cmd.replace(orig_dest, new_dest)

        instr_template = rng.choice(_JPG_INSTR_TEMPLATES)
        new_instruction = instr_template.format(dest=new_dest)

        rows.append(make_perturb_row(
            eval_row=per_variant_eval_row,
            knob_assignment={"jpg_dest": new_dest},
            new_instruction=new_instruction,
            new_oracle=oracle,
            new_evaluator=new_evaluator,
        ))
    return rows


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

_INTERNAL_FNS = [
    perturb_file_operation,
    perturb_system_query,
    perturb_permission,
    perturb_gnome_favorites,
    # NOTE: dropped perturb_large_text (3ce045a0) — eval reads
    # GNOME a11y schemas (`org.gnome.desktop.interface text-scaling-factor`)
    # but the VM runs XFCE where those schemas are absent. Agent uses the
    # XFCE-correct xfconf-query path which the eval doesn't inspect → all
    # variants score 0 regardless of correctness. Structural mismatch.
    perturb_check_moved_jpgs,
    perturb_paraphrase_coverage,
]


def perturb_os_per_task(
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
