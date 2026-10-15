"""OS synth generator (Batch §I file-as-topic design).

Each File encodes one structurally distinct filesystem-state shape (config
file body, archive shape, env state, system marker). `_to_synth_template`
turns a (File, FileTask, Param) triple into one SynthTemplate; per-seed the
picker selects params[seed % cap] and constructs the evaluator from the
Param's `eval_kind` (file_diff / command_output / config_check /
archive_check).

Batch audit (2026-05-10): re-bucketed FileTask.eval_class from
`config_setting` (legacy fallback) onto the proper os taxonomy keys: file_operation
/ compare_archive / system_query / dual_operation / app_management / timezone.
Added F-OS-21..F-OS-25 to fill the new buckets (timezone marker, GNOME
favorites, disk-query dir, dual-op filetree, /etc/hosts edit). Spread:
file_operation 27, compare_archive 10, timezone/app_management/system_query/
dual_operation 3 each.

All rows use `config_override` to bypass the default app-launch (OS tasks
are shell-only).

Mirrors the §I scaffold in synth/libreoffice_calc.py and
synth/libreoffice_impress.py.

Usage:
    uv run python -m lite.gym.envs.lite.osworld.src.gen.train \\
        --track synth --domain os
"""

from __future__ import annotations

import textwrap

from lite.gym.envs.lite.osworld.src.gen.train.synth._utils import (
    SynthTemplate,
    _stage_asset,
    _terminal_preopen_steps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _execute(command: str, *, shell: bool = True, **extra) -> dict:
    return {"type": "execute", "parameters": {"command": command, "shell": shell, **extra}}


def _write_text_step(path: str, content: str) -> dict:
    """Write a plain-text file via heredoc."""
    py = textwrap.dedent(f"""\
        import os
        path = {path!r}
        content = {content!r}
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        """)
    return _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF")


def _python_step(py: str) -> dict:
    """Run an arbitrary python heredoc inside the container."""
    return _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF")


def _desktop_settle_step() -> dict:
    """Focus the XFCE desktop (xfdesktop root window) before agent's first action.

    Validation fix: 90% of upstream OS non-terminal (GUI Settings)
    tasks end their config with a "settle step" that guarantees desktop focus
    before the agent's first action — otherwise the agent may inherit stale
    focus from a previous container state. Upstream uses
    `pyautogui.click({W/2}, {H/2}); time.sleep(0.5)`, but lite.osworld
    resolution is not pinned across containers, so we use a
    resolution-independent equivalent: focus the xfdesktop root window by
    WM_CLASS via `xdotool`. The trailing `true` keeps the step succeeding
    even when no xfdesktop window matches (rare).

    Terminal tasks (needs_terminal=True) already get
    `_terminal_preopen_steps()` whose final `activate_window 'Terminal'` is
    the settle equivalent — so this helper is appended only to
    non-terminal FileTasks.
    """
    return {"type": "execute", "parameters": {
        "command": "xdotool search --class xfdesktop windowfocus 2>/dev/null; sleep 0.5; true",
        "shell": True}}


# ---------------------------------------------------------------------------
# chmod template
# ---------------------------------------------------------------------------

# Mirrors the perturb-side `_make_perm_eval_command` shell.
def _make_perm_eval_command(perm_octal: str, test_dir: str) -> str:
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


# ---------------------------------------------------------------------------
# TEMPLATES export — populated by §I emission below.
# ---------------------------------------------------------------------------
TEMPLATES: list[SynthTemplate] = []


# ===========================================================================
# §I. File-task templates (Batch, dataclass form)
#
# Mirrors synth/libreoffice_calc.py + synth/libreoffice_impress.py §I.
# This domain is file-as-topic (no inner TopicTheme rotation): each File
# already encodes both the structural shape AND the content semantics
# (config-file body, archive shape, env state, etc.).
#
# Symmetric layout (all synth/*.py):
#   §I.a  Caps                — SYNTH_CAP_TASKS_PER_FILE / _PARAMS_PER_TASK
#   §I.b  Dataclasses         — File / Param / FileTask (frozen)
#   §I.c  File instances      — define each File ONCE
#   §I.d  Factory + emit      — _to_synth_template / _emit_templates
#   §I.e  FILE_TASKS          — flat list, one entry per (file, task) pair
#   §I.f  Emission            — TEMPLATES.extend(_emit_templates(FILE_TASKS))
# ===========================================================================

from dataclasses import dataclass as _I_dataclass, field as _I_field
from typing import Callable as _I_Callable


# §I.a — caps
SYNTH_CAP_TASKS_PER_FILE: int = 2
SYNTH_CAP_PARAMS_PER_TASK: int = 2


# §I.b — Dataclasses. `Param` shape is domain-specific. For os we rotate
# along an `eval_kind` axis (file_diff / command_output / config_check /
# archive_check) — see `_to_synth_template` for how each kind is wired
# into the SynthTemplate.evaluator_fn.

@_I_dataclass(frozen=True)
class File:
    """One structurally distinct source artifact.

    `src(path, seed) -> list[dict]` returns the LIST of pre_config steps
    that build the source artifact. `path` is the absolute container
    path of the primary file/dir; the builder may write additional
    files relative to its parent dir.
    """
    id: str
    setup_class: str
    basename: str
    src: _I_Callable[[str, int], list[dict]]


@_I_dataclass(frozen=True)
class Param:
    """One concrete parameterization of a task.

    `gold_args`  — kwargs forwarded to FileTask.gold(src_path, **gold_args)
                   to produce the list of oracle steps that mutate the source
                   into the gold state.
    `eval_kind`  — one of "file_diff" / "command_output" / "config_check"
                   / "archive_check". Selects how the evaluator is wired:
        - "file_diff":      compare_text_file (result_path vs expected_path)
                            eval_args: {"result_path": str, "expected_path": str}
        - "command_output": check_include_exclude over vm_command_line stdout
                            eval_args: {"command": str, "include": list[str],
                                        "exclude": list[str] (optional)}
        - "config_check":   check_include_exclude over a config-file probe
                            (typically `cat` / `grep` over a config path)
                            eval_args: same as command_output
        - "archive_check":  compare_archive (result_archive vs expected_archive)
                            eval_args: {"result_path": str, "expected_path": str,
                                        "file_type": "text"|"image"}
    `eval_args`  — kind-specific eval construction kwargs (see above).
    `instr`      — rendered instruction string.
    """
    gold_args: dict
    eval_kind: str
    eval_args: dict
    instr: str


@_I_dataclass(frozen=True)
class FileTask:
    """One (file, task) pair → one SynthTemplate at emit time.

    `gold(src_path, **gold_args) -> list[dict]` returns the oracle step
    list that mutates the source into the gold state. Some tasks also
    write a separate gold file (for file_diff / archive_check eval); the
    builder is responsible for both.
    """
    file: File
    task_id: str
    eval_class: str
    gold: _I_Callable[..., list[dict]]
    params: list[Param] = _I_field(default_factory=list)
    # Validation bug fix (cluster H): when True, `_to_synth_template` prepends
    # `_terminal_preopen_steps()` to `config_override` so the agent enters
    # the task with a gnome-terminal window already focused. Set this for
    # any FileTask whose instructions/oracle rely on running shell commands
    # in a terminal (e.g. git, gsettings, pactl, mkdir, mv, touch, ...).
    #
    # validation generalization caveat: validation + validation runs confirmed
    # *every* file_operation FileTask in this file shares the trigger-H
    # `left_click@dock + type_text` race when no terminal is pre-opened.
    # Many file_operation tasks currently pass only because the instruction
    # backtick-feeds the literal shell command (legitimate hint-driven
    # passes per validation TRIVIAL_PASS audit) — once validation strips those
    # hand-fed commands, those tasks will start
    # FAILING with trigger H. Flip `needs_terminal=True` on the remaining
    # file_operation FileTasks during validation instruction cleanup, in the
    # same commit that strips the hand-fed commands. Doing it before then
    # is harmless but unverified.
    needs_terminal: bool = False


# ---------------------------------------------------------------------------
# §I.c — File instances (file-as-topic).
# Each File defines ONE structurally distinct filesystem-state shape.
# Loop labels in comments mark where in the design loop the file landed.
# ---------------------------------------------------------------------------

_HOME = "/home/user"
_DESK = f"{_HOME}/Desktop"


# Loop 1 — config-file shapes (nginx / systemd / crontab / sshd_config)

def _src_nginx_site(path: str, seed: int) -> list[dict]:
    """Write a realistic nginx site config at `path`."""
    body = (
        "server {\n"
        "    listen 80;\n"
        "    server_name example.com www.example.com;\n"
        "    root /var/www/example;\n"
        "    index index.html;\n"
        "    access_log /var/log/nginx/example.access.log;\n"
        "    error_log /var/log/nginx/example.error.log;\n"
        "\n"
        "    location / {\n"
        "        try_files $uri $uri/ =404;\n"
        "    }\n"
        "}\n"
    )
    return [_write_text_step(path, body)]


def _src_systemd_unit(path: str, seed: int) -> list[dict]:
    """Write a realistic systemd .service unit at `path`."""
    body = (
        "[Unit]\n"
        "Description=Example background worker\n"
        "After=network.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        "ExecStart=/usr/local/bin/worker --foreground\n"
        "Restart=on-failure\n"
        "User=user\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )
    return [_write_text_step(path, body)]


def _src_crontab(path: str, seed: int) -> list[dict]:
    """Write a realistic crontab file with a few schedule entries."""
    body = (
        "# m h dom mon dow command\n"
        "0 2 * * * /usr/local/bin/backup.sh\n"
        "30 4 * * 0 /usr/local/bin/weekly-report.sh\n"
        "*/15 * * * * /usr/local/bin/heartbeat.sh\n"
    )
    return [_write_text_step(path, body)]


def _src_sshd_config(path: str, seed: int) -> list[dict]:
    """Write a realistic sshd_config snippet at `path`."""
    body = (
        "Port 22\n"
        "PermitRootLogin yes\n"
        "PasswordAuthentication yes\n"
        "X11Forwarding no\n"
        "ClientAliveInterval 0\n"
    )
    return [_write_text_step(path, body)]


# Loop 2 — bashrc / zshrc / env shapes

def _src_bashrc_default(path: str, seed: int) -> list[dict]:
    """Write a minimal bashrc with no aliases or env lines."""
    body = (
        "# bashrc — base profile\n"
        "export PS1='\\u@\\h:\\w\\$ '\n"
        "umask 022\n"
        "shopt -s histappend\n"
    )
    return [_write_text_step(path, body)]


def _src_zshrc_default(path: str, seed: int) -> list[dict]:
    """Write a minimal zshrc with no aliases."""
    body = (
        "# zshrc — base profile\n"
        "export PROMPT='%n@%m %~ %# '\n"
        "setopt HIST_IGNORE_DUPS\n"
        "setopt SHARE_HISTORY\n"
    )
    return [_write_text_step(path, body)]


def _src_profile_path(path: str, seed: int) -> list[dict]:
    """Write a profile file with an existing PATH export (no ~/bin yet)."""
    body = (
        "# /etc/profile.d-style PATH bootstrap\n"
        "export PATH=\"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"\n"
        "export EDITOR=nano\n"
    )
    return [_write_text_step(path, body)]


def _src_env_file(path: str, seed: int) -> list[dict]:
    """Write a dotenv-style file with placeholder DB_URL/API_KEY."""
    body = (
        "DB_URL=postgres://localhost:5432/app\n"
        "API_KEY=replace-me\n"
        "LOG_LEVEL=INFO\n"
    )
    return [_write_text_step(path, body)]


# Loop 3 — tar/archive shapes (source dirs of files for create/extract)

def _src_tar_doc_dir(path: str, seed: int) -> list[dict]:
    """`path` is a directory; populate with three .txt notes."""
    files = {
        "notes.txt": "Meeting notes for Q3.\nAttendees: A, B, C.\n",
        "todo.txt": "- ship feature\n- write docs\n",
        "ideas.txt": "1. async retros\n2. shared parser lib\n",
    }
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        files = {files!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        for name, body in files.items():
            with open(os.path.join(d, name), 'w') as f:
                f.write(body)
        """)
    return [_python_step(py)]


def _src_targz_log_dir(path: str, seed: int) -> list[dict]:
    """`path` is a directory; populate with auth.log/syslog/kern.log."""
    files = {
        "auth.log": "2026-04-22T08:14:01 host sshd[1812]: Accepted publickey for user\n",
        "syslog": "2026-04-22T08:14:00 host systemd[1]: Started session.\n",
        "kern.log": "2026-04-22T08:13:55 host kernel: usb 1-1: new device\n",
    }
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        files = {files!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        for name, body in files.items():
            with open(os.path.join(d, name), 'w') as f:
                f.write(body)
        """)
    return [_python_step(py)]


def _src_tar_archive_extract(path: str, seed: int) -> list[dict]:
    """`path` is the path to a pre-existing .tar.gz archive (with 3 entries)
    that the agent will extract. Build the archive at codegen time."""
    files = {
        "alpha.txt": "alpha file body\n",
        "beta.txt": "beta file body\n",
        "gamma.txt": "gamma file body\n",
    }
    py = textwrap.dedent(f"""\
        import os, shutil, tarfile, tempfile
        archive = {path!r}
        files = {files!r}
        os.makedirs(os.path.dirname(archive), exist_ok=True)
        if os.path.exists(archive):
            os.remove(archive)
        with tempfile.TemporaryDirectory() as td:
            for name, body in files.items():
                with open(os.path.join(td, name), 'w') as f:
                    f.write(body)
            with tarfile.open(archive, 'w:gz') as tf:
                for name in sorted(files.keys()):
                    tf.add(os.path.join(td, name), arcname=name)
        """)
    return [_python_step(py)]


def _src_zip_photos_dir(path: str, seed: int) -> list[dict]:
    """`path` is a directory; stage 3 small landscape jpgs into it."""
    photo_rels = [
        "photos/landscape/beach-sunset.jpg",
        "photos/landscape/desert-dunes.jpg",
        "photos/landscape/forest-trail.jpg",
    ]
    cleanup = _execute(f"rm -rf {path} && mkdir -p {path}")
    stage = [
        _stage_asset(rel, f"{path}/{rel.rsplit('/', 1)[1]}")
        for rel in photo_rels
    ]
    return [cleanup, *stage]


# Loop 4 — cron / systemd config-mutation shapes

def _src_cron_schedule_v2(path: str, seed: int) -> list[dict]:
    """Crontab variant with a daily and a weekly job (different from
    `_src_crontab` to avoid ID collision and force a real distinct file)."""
    body = (
        "# m h dom mon dow command\n"
        "15 1 * * * /usr/local/bin/db-snapshot.sh\n"
        "0 6 * * 1 /usr/local/bin/weekly-cleanup.sh\n"
    )
    return [_write_text_step(path, body)]


def _src_systemd_timer(path: str, seed: int) -> list[dict]:
    """A .timer unit file that schedules an associated service."""
    body = (
        "[Unit]\n"
        "Description=Run example.service every 30 minutes\n"
        "\n"
        "[Timer]\n"
        "OnBootSec=5min\n"
        "OnUnitActiveSec=30min\n"
        "Unit=example.service\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )
    return [_write_text_step(path, body)]


# Loop 5 — file-ops + ssh + gap-filler shapes

def _src_mixed_filetree(path: str, seed: int) -> list[dict]:
    """A directory with .py / .txt / .log files mixed at various perms."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        files = [
            ('a.py', '600'), ('b.py', '600'), ('c.py', '600'),
            ('readme.txt', '644'), ('notes.txt', '644'),
            ('app.log', '644'), ('debug.log', '644'),
        ]
        for name, perm in files:
            p = os.path.join(d, name)
            open(p, 'w').close()
            os.chmod(p, int(perm, 8))
        """)
    return [_python_step(py)]


def _src_renameable_dir(path: str, seed: int) -> list[dict]:
    """A directory that the agent will rename (keeps a marker file inside)."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, '.keep'), 'w') as f:
            f.write('marker\\n')
        """)
    return [_python_step(py)]


def _src_authorized_keys(path: str, seed: int) -> list[dict]:
    """A pre-seeded `~/.ssh/authorized_keys` with one existing key."""
    body = (
        "# authorized_keys — pre-existing key\n"
        "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDexisting alice@laptop\n"
    )
    return [_write_text_step(path, body)]


def _src_gitconfig_empty(path: str, seed: int) -> list[dict]:
    """Empty ~/.gitconfig (created but no `[user]` block)."""
    return [_write_text_step(path, "")]


# Loop 6 — taxonomy gap-fillers (timezone / system_query / dual_operation /
# app_management). Each File targets a specific os taxonomy bucket to lift
# eval_class diversity and exercise OSWorld eval skills outside config-edits.

def _src_timezone_system(path: str, seed: int) -> list[dict]:
    """Pre-seed the *real* system timezone to America/Los_Angeles AND install
    a `timedatectl` shim that reads /etc/timezone on every call.

    Mirrors the eval setup for `osworld_os_b6781586` (is_utc_0 evaluator):
      - `sudo ln -sf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime`
      - `echo America/Los_Angeles | sudo tee /etc/timezone`
      - shim install (mirrors perturb/os.py _PARAPHRASE_PRE_CONFIG_STEPS so
        real `timedatectl status` works on no-systemd containers).

    The `path` argument is unused (kept for interface compatibility); the
    setup writes to fixed system locations.
    """
    seed_cmd = (
        "echo user | sudo -S -v 2>/dev/null; "
        "sudo ln -sf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && "
        "echo 'America/Los_Angeles' | sudo tee /etc/timezone > /dev/null"
    )
    # Install via heredoc — printf treats `%a`/`%Y`/etc. inside its format
    # string as conversion specifiers and corrupts the script. Heredoc
    # passes the body through verbatim.
    shim_cmd = (
        "echo user | sudo -S -v 2>/dev/null; "
        "sudo tee /usr/local/bin/timedatectl > /dev/null <<'TIMEDATECTL_EOF' && "
        "sudo chmod +x /usr/local/bin/timedatectl\n"
        "#!/bin/bash\n"
        "TZ=$(cat /etc/timezone 2>/dev/null || echo UTC)\n"
        "DT=$(TZ=$TZ date \"+%a %Y-%m-%d %H:%M:%S\")\n"
        "UDT=$(date -u \"+%a %Y-%m-%d %H:%M:%S\")\n"
        "OFF=$(TZ=$TZ date +%z)\n"
        "OFF=$(echo $OFF | tr -d :)\n"
        "echo \"               Local time: $DT $TZ\"\n"
        "echo \"           Universal time: $UDT UTC\"\n"
        "echo \"                 RTC time: $UDT\"\n"
        "echo \"                Time zone: $TZ ($TZ, ${OFF})\"\n"
        "echo \"System clock synchronized: yes\"\n"
        "echo \"              NTP service: inactive\"\n"
        "echo \"          RTC in local TZ: no\"\n"
        "TIMEDATECTL_EOF"
    )
    return [_execute(seed_cmd), _execute(shim_cmd)]


def _src_disk_query_dir(path: str, seed: int) -> list[dict]:
    """A populated dir whose disk-usage / file-count the agent will probe."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        for i in range(7):
            with open(os.path.join(d, f'doc_{{i:02d}}.txt'), 'w') as f:
                f.write('x' * (256 * (i + 1)))
        """)
    return [_python_step(py)]


def _src_dual_filetree(path: str, seed: int) -> list[dict]:
    """A dir of mixed-perm files; the agent runs a *combined* op
    (rename + chmod, or copy + count) — i.e. dual_operation."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        for name in ('alpha.sh', 'beta.sh', 'gamma.sh'):
            p = os.path.join(d, name)
            with open(p, 'w') as f:
                f.write('#!/bin/bash\\necho hi\\n')
            os.chmod(p, 0o644)
        """)
    return [_python_step(py)]


def _src_gnome_favorites_real(path: str, seed: int) -> list[dict]:
    """Pre-seed the *real* GNOME shell favorite-apps gsetting.

    Mirrors the eval setup for `osworld_os_ec4e3f68` (check_gnome_favorite_apps
    evaluator). The `path` arg is unused (kept for interface compatibility);
    the setup writes to the dconf gsetting.
    """
    cmd = (
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.shell favorite-apps "
        "\"['firefox.desktop', 'org.gnome.Nautilus.desktop', 'vim.desktop']\""
    )
    return [_execute(cmd)]


def _src_photos_tree(path: str, seed: int) -> list[dict]:
    """Populate `path` (a dir) with a recursive tree of .jpg files in
    subdirectories — mirrors the eval setup for `osworld_os_23393935`
    (check_moved_jpgs evaluator). Agent must `find ... -iname '*.jpg' -exec
    cp` into a flat dest dir.
    """
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        tree = {{
            'vacation/thailand': ['monk_temple.jpg'],
            'vacation/hk': ['hk_skyline.jpg', 'hk_group.jpg'],
            'family': ['us_family.jpg'],
            'events': ['emnlp2023.jpg'],
        }}
        for sub, files in tree.items():
            sd = os.path.join(d, sub)
            os.makedirs(sd, exist_ok=True)
            for fn in files:
                with open(os.path.join(sd, fn), 'wb') as f:
                    # 1-byte placeholder; check_moved_jpgs only compares
                    # basename lists from list_directory.
                    f.write(b'X')
        """)
    return [_python_step(py)]


def _src_hosts_file(path: str, seed: int) -> list[dict]:
    """A `/etc/hosts`-style file the agent will edit (add a host alias)."""
    body = (
        "127.0.0.1   localhost\n"
        "::1         localhost ip6-localhost ip6-loopback\n"
        "127.0.1.1   ubuntu\n"
    )
    return [_write_text_step(path, body)]


# Define File instances. ID convention: F_OS_<NN>.

F_OS_01 = File(id="F-OS-01", setup_class="nginx_site_conf",
               basename="nginx-site.conf", src=_src_nginx_site)
F_OS_02 = File(id="F-OS-02", setup_class="systemd_unit",
               basename="example.service", src=_src_systemd_unit)
F_OS_03 = File(id="F-OS-03", setup_class="crontab_file",
               basename="crontab", src=_src_crontab)
F_OS_04 = File(id="F-OS-04", setup_class="sshd_config",
               basename="sshd_config", src=_src_sshd_config)

F_OS_05 = File(id="F-OS-05", setup_class="bashrc_default",
               basename=".bashrc", src=_src_bashrc_default)
F_OS_06 = File(id="F-OS-06", setup_class="zshrc_default",
               basename=".zshrc", src=_src_zshrc_default)
F_OS_07 = File(id="F-OS-07", setup_class="profile_path",
               basename="profile.sh", src=_src_profile_path)
F_OS_08 = File(id="F-OS-08", setup_class="env_file",
               basename=".env", src=_src_env_file)

F_OS_09 = File(id="F-OS-09", setup_class="tar_doc_dir",
               basename="Documents", src=_src_tar_doc_dir)
F_OS_10 = File(id="F-OS-10", setup_class="targz_log_dir",
               basename="logs", src=_src_targz_log_dir)
F_OS_11 = File(id="F-OS-11", setup_class="tar_archive_extract",
               basename="bundle.tar.gz", src=_src_tar_archive_extract)
F_OS_12 = File(id="F-OS-12", setup_class="zip_photos_dir",
               basename="photos", src=_src_zip_photos_dir)

F_OS_13 = File(id="F-OS-13", setup_class="cron_schedule_v2",
               basename="crontab.v2", src=_src_cron_schedule_v2)
F_OS_14 = File(id="F-OS-14", setup_class="systemd_timer",
               basename="example.timer", src=_src_systemd_timer)
F_OS_15 = File(id="F-OS-15", setup_class="systemd_unit_b",
               basename="worker-b.service", src=_src_systemd_unit)
F_OS_16 = File(id="F-OS-16", setup_class="nginx_site_b_conf",
               basename="nginx-site-b.conf", src=_src_nginx_site)

F_OS_17 = File(id="F-OS-17", setup_class="mixed_filetree",
               basename="testDir", src=_src_mixed_filetree)
F_OS_18 = File(id="F-OS-18", setup_class="renameable_dir",
               basename="todo_list_Jan_1", src=_src_renameable_dir)
F_OS_19 = File(id="F-OS-19", setup_class="authorized_keys",
               basename="authorized_keys", src=_src_authorized_keys)
F_OS_20 = File(id="F-OS-20", setup_class="gitconfig_empty",
               basename=".gitconfig", src=_src_gitconfig_empty)

# Loop 6 — taxonomy gap-fillers
# 2026-05-10 (validation change): F_OS_21 / F_OS_22 setup_classes renamed
# (from `*_marker`) and src builders rewritten to seed the *real* system
# state (timezone gsetting → /etc/timezone + timedatectl shim; gnome
# favorites → gsettings) so their evaluators can use native is_utc_0 /
# check_gnome_favorite_apps funcs instead of marker-file proxies that
# previously routed through `_text_content`.
F_OS_21 = File(id="F-OS-21", setup_class="timezone_system",
               basename="current_tz.txt", src=_src_timezone_system)
F_OS_22 = File(id="F-OS-22", setup_class="gnome_favorites_real",
               basename="favorites.txt", src=_src_gnome_favorites_real)
F_OS_23 = File(id="F-OS-23", setup_class="disk_query_dir",
               basename="reportDir", src=_src_disk_query_dir)
F_OS_24 = File(id="F-OS-24", setup_class="dual_filetree",
               basename="scriptsDir", src=_src_dual_filetree)
F_OS_25 = File(id="F-OS-25", setup_class="hosts_file",
               basename="hosts", src=_src_hosts_file)
# New File for check_moved_jpgs eval func (mirrors osworld_os_23393935).
F_OS_26 = File(id="F-OS-26", setup_class="photos_tree",
               basename="photos_tree", src=_src_photos_tree)


# Loop 7 — `exact_match` eval gap-fillers (close the 5-task eval gap).
# 2026-05-10: synth previously had 0 `exact_match` rows while eval has 5.
# Each File seeds an initial system state DIFFERENT from the gold target
# (so the trivial-pass check fires only on real success); gold callables
# mutate state into the eval-target shape; evaluator is `exact_match` on a
# `vm_command_line` whose stdout equals a literal string.

# Pulseaudio bootstrap shared by the volume task — mirrors perturb/os.py
# `_pulseaudio_step`. The eval VM has no systemd so pulseaudio is not
# auto-started; daemonize it before any pactl call.
# XDG_RUNTIME_DIR MUST be pinned to the session value (/run/user/1000, set by the
# supervisor env in the shared sandbox base) so the daemon's socket lands at
# /run/user/1000/pulse/native — the same path the agent's desktop-launched terminal
# looks at. Without it, this bootstrap runs in the env-server's bare setup-exec (no
# session env), the socket lands elsewhere, and the agent's pactl gets "Connection
# refused" even though the daemon is up. (Verified: pulseaudio --start + pactl set-volume
# both succeed under XDG=/run/user/1000; fail/diverge without it.)
_PULSE_BOOTSTRAP = (
    "export XDG_RUNTIME_DIR=/run/user/1000; "
    "(pulseaudio --check 2>/dev/null) || "
    "(timeout 8 pulseaudio --start --daemonize=yes --exit-idle-time=-1 || true)"
)


# Shared DBus session bootstrap for gsettings/dconf gold actions. Kept as a
# plain string fragment because callers concatenate it with command bodies; no
# f-string interpolation is required.
_DBUS_BOOTSTRAP = (
    "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
    "if [ -f /tmp/dbus-session-bus-address ]; then "
    "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
    "else eval \"$(dbus-launch --sh-syntax)\"; "
    "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
)


def _src_volume_state(path: str, seed: int) -> list[dict]:
    """Pre-seed pulseaudio at a NON-100% volume (50%). `path` arg unused.

    Mirrors the eval setup for `osworld_os_28cc3b7e` (exact_match on pactl
    volume). Starting at 50% guarantees the initial state fails the
    expected="100\n" check, so trivial-pass detection fires only on
    actual oracle execution.
    """
    return [_execute(f"{_PULSE_BOOTSTRAP}; pactl set-sink-volume @DEFAULT_SINK@ 50%")]


def _src_missing_desktop_file(path: str, seed: int) -> list[dict]:
    """Ensure `path` does NOT exist (mirrors the eval pre-state for
    `osworld_os_5ea617a3` — file was 'wrongly deleted'). Gold re-creates it."""
    return [_execute(f"rm -f {path}")]


def _src_screensaver_lock_off(path: str, seed: int) -> list[dict]:
    """Pre-seed `org.gnome.desktop.screensaver lock-enabled = false` so the
    initial state fails the `true\\n` exact_match. Mirrors the eval setup for
    `osworld_os_a4d98375`. `path` arg unused."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.screensaver lock-enabled false"
    )]


def _src_renameable_dir_jan1(path: str, seed: int) -> list[dict]:
    """Same shape as `_src_renameable_dir` but a distinct File ID so
    F_OS_30 doesn't share the cap-2 budget with F_OS_18. Mirrors the eval
    setup for `osworld_os_e0df059f` — directory `todo_list_Jan_1` exists.
    """
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, '.keep'), 'w') as f:
            f.write('marker\\n')
        """)
    return [_python_step(py)]


def _src_gsettings_dnd_off(path: str, seed: int) -> list[dict]:
    """Pre-seed `org.gnome.desktop.notifications show-banners = true` so the
    initial state fails the post-gold `false\\n` exact_match. Mirrors the eval
    setup for `osworld_os_f9be0997`. `path` arg unused.

    The banner state lives in the GNOME `org.gnome.desktop.notifications`
    schema; upstream OSWorld reads the same gsettings key.
    """
    return [_execute(
        "gsettings set org.gnome.desktop.notifications show-banners true"
    )]


# F_OS_27..F_OS_31 — `exact_match` eval-class fillers.
# `basename` for the system-state Files is a sentinel (the src does NOT
# write to it); these are kept under ~/Desktop for instr-clarity only.
F_OS_27 = File(id="F-OS-27", setup_class="volume_state",
               basename="volume_state.txt", src=_src_volume_state)
F_OS_28 = File(id="F-OS-28", setup_class="missing_desktop_file",
               basename="poster_party_night.jpg", src=_src_missing_desktop_file)
F_OS_29 = File(id="F-OS-29", setup_class="screensaver_lock_gsetting",
               basename="screensaver_lock.txt", src=_src_screensaver_lock_off)
F_OS_30 = File(id="F-OS-30", setup_class="renameable_dir_v2",
               basename="todo_list_Jan_1", src=_src_renameable_dir_jan1)
F_OS_31 = File(id="F-OS-31", setup_class="gnome_dnd_gsettings",
               basename="gnome_dnd.txt", src=_src_gsettings_dnd_off)


# ---------------------------------------------------------------------------
# validation RESCALER — GNOME system-settings fill (eval `exact_match` UNDER).
# Each File pre-seeds the OPPOSITE state so the eval's exact_match guards
# against trivial pass. Eval shell command is `gsettings get …` or
# `xdg-settings …` and expected output is the value after the gold step.
# ---------------------------------------------------------------------------

def _src_idle_dim_on(path: str, seed: int) -> list[dict]:
    """Pre-seed `org.gnome.settings-daemon.plugins.power idle-dim = true` so
    initial state fails the `false\\n` exact_match (mirrors eval row 16
    'Dim screen when inactive — off')."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.settings-daemon.plugins.power idle-dim true"
    )]


def _src_night_light_off(path: str, seed: int) -> list[dict]:
    """Pre-seed `org.gnome.settings-daemon.plugins.color night-light-enabled
    = false`."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.settings-daemon.plugins.color night-light-enabled false"
    )]


def _src_tap_to_click_off(path: str, seed: int) -> list[dict]:
    """Pre-seed `org.gnome.desktop.peripherals.touchpad tap-to-click = false`."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.peripherals.touchpad tap-to-click false"
    )]


def _src_default_browser_chrome(path: str, seed: int) -> list[dict]:
    """Pre-seed default browser = google-chrome.desktop so the initial state
    fails the `firefox.desktop` exact_match."""
    return [_execute(
        "xdg-settings set default-web-browser google-chrome.desktop 2>/dev/null || true"
    )]


def _src_power_button_interactive(path: str, seed: int) -> list[dict]:
    """Pre-seed power-button-action to `interactive`."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.settings-daemon.plugins.power power-button-action 'interactive'"
    )]


def _src_keyboard_repeat_default(path: str, seed: int) -> list[dict]:
    """Pre-seed keyboard repeat-interval to 30 (default)."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.peripherals.keyboard repeat-interval 30"
    )]


def _gold_gsettings_idle_dim_false(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.settings-daemon.plugins.power idle-dim false"
    )]


def _gold_gsettings_night_light_true(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.settings-daemon.plugins.color night-light-enabled true"
    )]


def _gold_gsettings_tap_to_click_true(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.peripherals.touchpad tap-to-click true"
    )]


def _gold_xdg_default_browser_firefox(src_path: str) -> list[dict]:
    # The lite.osworld image does NOT ship firefox (only google-chrome), so
    # xdg-settings would reject `set default-web-browser firefox.desktop`
    # with rc=2 ("desktop entry not found"). Stage a minimal stub
    # firefox.desktop in /usr/share/applications/ first so xdg-settings
    # accepts the registration; the eval's `xdg-settings get` just reads
    # the registered name back and does not launch the app.
    return [_execute(
        "echo user | sudo -S -v 2>/dev/null; "
        "sudo tee /usr/share/applications/firefox.desktop >/dev/null <<'EOF_FF'\n"
        "[Desktop Entry]\n"
        "Name=Firefox\n"
        "Exec=firefox %u\n"
        "Type=Application\n"
        "MimeType=text/html;application/xhtml+xml;x-scheme-handler/http;x-scheme-handler/https;\n"
        "Categories=Network;WebBrowser;\n"
        "EOF_FF\n"
        "sudo update-desktop-database /usr/share/applications/ 2>/dev/null || true; "
        "xdg-settings set default-web-browser firefox.desktop"
    )]


def _gold_gsettings_power_button_suspend(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.settings-daemon.plugins.power power-button-action 'suspend'"
    )]


def _gold_gsettings_keyboard_repeat_15(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.peripherals.keyboard repeat-interval 15"
    )]


# F-OS-32..F-OS-37 — system-settings Files.
F_OS_32 = File(id="F-OS-32", setup_class="idle_dim_state",
               basename="idle_dim.txt", src=_src_idle_dim_on)
F_OS_33 = File(id="F-OS-33", setup_class="night_light_state",
               basename="night_light.txt", src=_src_night_light_off)
F_OS_34 = File(id="F-OS-34", setup_class="tap_to_click_state",
               basename="tap_to_click.txt", src=_src_tap_to_click_off)
F_OS_35 = File(id="F-OS-35", setup_class="default_browser_state",
               basename="default_browser.txt", src=_src_default_browser_chrome)
F_OS_36 = File(id="F-OS-36", setup_class="power_button_state",
               basename="power_button.txt", src=_src_power_button_interactive)
F_OS_37 = File(id="F-OS-37", setup_class="keyboard_repeat_state",
               basename="keyboard_repeat.txt", src=_src_keyboard_repeat_default)


# ---------------------------------------------------------------------------
# validation GUI Settings expansion — close gui_settings/system_target gap.
# Each File pre-seeds the OPPOSITE gsetting/dconf/xfconf value so eval
# exact_match guards against trivial pass. Eval reads real system state via
# `gsettings get`/`xfconf-query`/`pactl`/`/etc/timezone` (NOT a staged file).
# Instructions intentionally use first-person user voice (no backticks).
# ---------------------------------------------------------------------------

def _src_text_scaling_default(path: str, seed: int) -> list[dict]:
    """Pre-seed text-scaling-factor=1.0 so the gold (1.25) state differs.
    Mirrors `osworld_os_3ce045a0` (Universal Access — Large Text)."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.interface text-scaling-factor 1.0"
    )]


def _src_notifications_on(path: str, seed: int) -> list[dict]:
    """Pre-seed show-banners=true so the gold (false) state differs.
    Mirrors the GNOME Notifications-off Settings task."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.notifications show-banners true"
    )]


def _src_idle_delay_short(path: str, seed: int) -> list[dict]:
    """Pre-seed screen-blank idle-delay=300s so the gold (1800s = 30min) differs."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.session idle-delay 300"
    )]


def _src_clock_format_12h(path: str, seed: int) -> list[dict]:
    """Pre-seed clock-format='12h' so the gold ('24h') state differs.
    Mirrors a Region & Language — Time Format Settings task."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.interface clock-format '12h'"
    )]


def _src_color_scheme_default(path: str, seed: int) -> list[dict]:
    """Pre-seed color-scheme='default' so the gold ('prefer-dark') differs."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.interface color-scheme 'default'"
    )]


def _src_high_contrast_off(path: str, seed: int) -> list[dict]:
    """Pre-seed gtk-theme='Adwaita' (no high contrast) so gold differs."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.interface gtk-theme 'Adwaita'"
    )]


def _src_mic_unmuted(path: str, seed: int) -> list[dict]:
    """Pre-seed default source (mic) UNMUTED so the gold (muted) state differs.
    Pulseaudio needs to be running; mirrors F_OS_27 _PULSE_BOOTSTRAP pattern."""
    return [_execute(
        f"{_PULSE_BOOTSTRAP}; pactl set-source-mute @DEFAULT_SOURCE@ 0 || true"
    )]


def _src_timezone_la(path: str, seed: int) -> list[dict]:
    """Pre-seed /etc/timezone=America/Los_Angeles so the gold (Asia/Tokyo) differs.
    Reuses the timedatectl shim install from F_OS_21."""
    seed_cmd = (
        "echo user | sudo -S -v 2>/dev/null; "
        "sudo ln -sf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && "
        "echo 'America/Los_Angeles' | sudo tee /etc/timezone > /dev/null"
    )
    return [_execute(seed_cmd)]


def _src_gnome_wallpaper_default(path: str, seed: int) -> list[dict]:
    """Pre-seed `org.gnome.desktop.background picture-options='centered'` so
    the gold ('zoom') differs. The wallpaper fit mode lives in the GNOME
    `org.gnome.desktop.background picture-options` schema."""
    return [_execute(
        "gsettings set org.gnome.desktop.background picture-options 'centered'"
    )]


def _src_a11y_screen_keyboard_off(path: str, seed: int) -> list[dict]:
    """Pre-seed screen-keyboard-enabled=false; gold flips to true.
    Universal Access — On-Screen Keyboard."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.a11y.applications screen-keyboard-enabled false"
    )]


# Corresponding gold builders.

def _gold_gsettings_text_scaling_1_25(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.interface text-scaling-factor 1.25"
    )]


def _gold_gsettings_notifications_off(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.notifications show-banners false"
    )]


def _gold_gsettings_idle_delay_1800(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.session idle-delay 1800"
    )]


def _gold_gsettings_clock_format_24h(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.interface clock-format '24h'"
    )]


def _gold_gsettings_color_scheme_dark(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'"
    )]


def _gold_gsettings_high_contrast_on(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.interface gtk-theme 'HighContrast'"
    )]


def _gold_pactl_mic_mute(src_path: str) -> list[dict]:
    return [_execute(
        f"{_PULSE_BOOTSTRAP}; pactl set-source-mute @DEFAULT_SOURCE@ 1"
    )]


def _gold_timedatectl_tokyo(src_path: str) -> list[dict]:
    return [_execute(
        "echo user | sudo -S -v 2>/dev/null; "
        "sudo ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && "
        "echo 'Asia/Tokyo' | sudo tee /etc/timezone > /dev/null"
    )]


def _gold_gsettings_wallpaper_zoomed(src_path: str) -> list[dict]:
    """Set the GNOME wallpaper fit mode to zoom via
    `gsettings set org.gnome.desktop.background picture-options 'zoom'`."""
    return [_execute(
        "gsettings set org.gnome.desktop.background picture-options 'zoom'"
    )]


def _gold_gsettings_screen_keyboard_on(src_path: str) -> list[dict]:
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.a11y.applications screen-keyboard-enabled true"
    )]


# F_OS_38..F_OS_47 — GUI Settings expansion (validation cell-reduction).
F_OS_38 = File(id="F-OS-38", setup_class="text_scaling_state",
               basename="text_scaling.txt", src=_src_text_scaling_default)
F_OS_39 = File(id="F-OS-39", setup_class="notifications_state",
               basename="notifications.txt", src=_src_notifications_on)
F_OS_40 = File(id="F-OS-40", setup_class="idle_delay_state",
               basename="idle_delay.txt", src=_src_idle_delay_short)
F_OS_41 = File(id="F-OS-41", setup_class="clock_format_state",
               basename="clock_format.txt", src=_src_clock_format_12h)
F_OS_42 = File(id="F-OS-42", setup_class="color_scheme_state",
               basename="color_scheme.txt", src=_src_color_scheme_default)
F_OS_43 = File(id="F-OS-43", setup_class="high_contrast_state",
               basename="high_contrast.txt", src=_src_high_contrast_off)
F_OS_44 = File(id="F-OS-44", setup_class="mic_mute_state",
               basename="mic_mute.txt", src=_src_mic_unmuted)
F_OS_45 = File(id="F-OS-45", setup_class="timezone_tokyo_state",
               basename="tz_tokyo.txt", src=_src_timezone_la)
F_OS_46 = File(id="F-OS-46", setup_class="gnome_wallpaper_state",
               basename="gnome_wallpaper.txt", src=_src_gnome_wallpaper_default)
F_OS_47 = File(id="F-OS-47", setup_class="screen_keyboard_state",
               basename="screen_keyboard.txt", src=_src_a11y_screen_keyboard_off)


# ---------------------------------------------------------------------------
# validation NL2Bash multistep expansion — close shell_pipeline gap with
# non-Desktop paths (reduces userspace_desktop overshoot). Each File uses
# a /tmp/ working area so the eval classifier routes them to system_target
# = "other" (not Desktop), broadening coverage. Instructions use natural
# conjoined-verb phrasing so the NL2Bash classifier triggers.
# ---------------------------------------------------------------------------

def _src_logs_old_dir(path: str, seed: int) -> list[dict]:
    """`path` is a dir under /tmp; populate with old + mid-age + fresh logs.
    Three files are 40 days old (caught by -mtime +30 AND -mtime +7).
    Two files are 10 days old (caught by -mtime +7 only).
    Two files are fresh (caught by neither threshold)."""
    py = textwrap.dedent(f"""\
        import os, shutil, time
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        now = time.time()
        old40 = now - 40 * 86400   # 40 days ago  (> 30d AND > 7d)
        old10 = now - 10 * 86400   # 10 days ago  (> 7d only)
        spec = [
            ('app1.log', 'old log line A\\n', old40),
            ('app2.log', 'old log line B\\n', old40),
            ('app3.log', 'old log line C\\n', old40),
            ('app4.log', 'mid-age log D\\n',  old10),
            ('app5.log', 'mid-age log E\\n',  old10),
            ('app6.log', 'recent log F\\n',   now),
            ('app7.log', 'recent log G\\n',   now),
        ]
        for name, body, mtime in spec:
            p = os.path.join(d, name)
            with open(p, 'w') as f:
                f.write(body)
            os.utime(p, (mtime, mtime))
        """)
    return [_python_step(py)]


def _src_csv_data_dir(path: str, seed: int) -> list[dict]:
    """`path` is a dir under /tmp; stage three CSV files for fan-out copy."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        for name in ('alpha.csv', 'beta.csv', 'gamma.csv'):
            with open(os.path.join(d, name), 'w') as f:
                f.write('col1,col2\\n1,2\\n')
        """)
    return [_python_step(py)]


def _src_sandbox_tree(path: str, seed: int) -> list[dict]:
    """`path` is a dir under /tmp; mixed-permission tree of files + sub-dirs."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        # top-level files
        for name in ('a.sh', 'b.sh', 'readme.md'):
            p = os.path.join(d, name)
            with open(p, 'w') as f:
                f.write('placeholder\\n')
            os.chmod(p, 0o777)
        # nested sub-dir with files
        sub = os.path.join(d, 'lib')
        os.makedirs(sub, exist_ok=True)
        os.chmod(sub, 0o777)
        for name in ('mod1.py', 'mod2.py'):
            p = os.path.join(sub, name)
            with open(p, 'w') as f:
                f.write('# python\\n')
            os.chmod(p, 0o777)
        """)
    return [_python_step(py)]


def _src_python_pandas_tree(path: str, seed: int) -> list[dict]:
    """`path` is a dir under /tmp; mix of .py files some importing pandas."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        spec = {{
            'load_data.py':    'import pandas as pd\\nimport os\\ndf = pd.read_csv("x.csv")\\n',
            'train_model.py':  'import numpy as np\\nimport pandas\\n# more stuff\\n' + 'pass\\n'*40,
            'plot.py':         'import matplotlib.pyplot as plt\\n# no pandas here\\n',
            'utils.py':        '# helpers, no pandas\\ndef foo():\\n    return 1\\n',
            'pipeline.py':     'import pandas\\nimport sklearn\\n' + 'x = 1\\n'*20,
        }}
        for name, body in spec.items():
            with open(os.path.join(d, name), 'w') as f:
                f.write(body)
        """)
    return [_python_step(py)]


def _src_text_corpus(path: str, seed: int) -> list[dict]:
    """`path` is a dir under /tmp; .txt files some containing the target word."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        spec = {{
            'note1.txt': 'hello world\\nTODO: write tests\\n',
            'note2.txt': 'no markers here\\njust prose\\n',
            'note3.txt': 'remember the TODO list\\n',
            'note4.txt': 'plain text without urgent markers\\n',
            'note5.txt': 'TODO: refactor the parser\\nDONE: ship feature\\n',
        }}
        for name, body in spec.items():
            with open(os.path.join(d, name), 'w') as f:
                f.write(body)
        """)
    return [_python_step(py)]


# Gold builders.

def _gold_compress_and_remove_old(src_dir: str, *, archive: str, count_path: str, days: int = 30) -> list[dict]:
    """Find files older than `days` days, tar+gz them into archive, then
    write the count of archived files into count_path."""
    return [
        _execute(
            f"cd {src_dir} && "
            f"find . -maxdepth 1 -type f -mtime +{days} -print0 | "
            f"tar --null -czf {archive} --files-from -"
        ),
        _execute(
            f"find {src_dir} -maxdepth 1 -type f -mtime +{days} | wc -l > {count_path}"
        ),
    ]


def _gold_fan_out_csv(src_dir: str, *, dests: list[str], count_path: str) -> list[dict]:
    """Copy every .csv from src_dir to each destination dir, then count
    one of the destinations and write to count_path."""
    cmds = [_execute(f"mkdir -p {d} && cp {src_dir}/*.csv {d}/") for d in dests]
    cmds.append(_execute(f"ls {dests[0]}/*.csv 2>/dev/null | wc -l > {count_path}"))
    return cmds


def _gold_split_perm(src_dir: str, *, count_path: str) -> list[dict]:
    """Find every regular file and chmod to 600; find every directory and
    chmod to 755. Then write count of 600-files to count_path."""
    return [
        _execute(f"find {src_dir} -type f -exec chmod 600 {{}} +"),
        _execute(f"find {src_dir} -type d -exec chmod 755 {{}} +"),
        _execute(f"find {src_dir} -type f -perm 0600 | wc -l > {count_path}"),
    ]


def _gold_grep_pandas_count(src_dir: str, *, count_path: str) -> list[dict]:
    """Recursively grep for `pandas` imports; sort matches; write count to
    count_path."""
    return [
        _execute(
            f"grep -rl --include='*.py' 'import pandas' {src_dir} | sort | wc -l > {count_path}"
        ),
    ]


def _gold_grep_todo_count(src_dir: str, *, count_path: str) -> list[dict]:
    """Recursively grep for 'TODO' in .txt files; write count of matching files
    to count_path."""
    return [
        _execute(
            f"grep -rl --include='*.txt' 'TODO' {src_dir} | sort | wc -l > {count_path}"
        ),
    ]


# F_OS_48..F_OS_52 — NL2Bash multistep Files (under /tmp/, NOT Desktop).
F_OS_48 = File(id="F-OS-48", setup_class="logs_old_dir",
               basename="oldlogs", src=_src_logs_old_dir)
F_OS_49 = File(id="F-OS-49", setup_class="csv_data_dir",
               basename="csvdata", src=_src_csv_data_dir)
F_OS_50 = File(id="F-OS-50", setup_class="sandbox_tree",
               basename="sandbox", src=_src_sandbox_tree)
F_OS_51 = File(id="F-OS-51", setup_class="python_pandas_tree",
               basename="projects", src=_src_python_pandas_tree)
F_OS_52 = File(id="F-OS-52", setup_class="text_corpus",
               basename="corpus", src=_src_text_corpus)


# validation extension: additional non-Desktop file-edit Files to reduce
# userspace_desktop overshoot. Same /tmp/ basename convention.

def _src_jsonlines_dir(path: str, seed: int) -> list[dict]:
    """`path` is a dir under /tmp; stage 4 .jsonl files of varying line counts."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        spec = {{
            'events_jan.jsonl': 3,
            'events_feb.jsonl': 5,
            'events_mar.jsonl': 7,
            'events_apr.jsonl': 2,
        }}
        for name, n in spec.items():
            with open(os.path.join(d, name), 'w') as f:
                for i in range(n):
                    f.write('{{"id": '+str(i)+', "ok": true}}\\n')
        """)
    return [_python_step(py)]


def _src_disk_report_dir(path: str, seed: int) -> list[dict]:
    """`path` is a dir under /tmp; small payload for total-line-count probes."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        for i in range(4):
            with open(os.path.join(d, f'sample_{{i}}.txt'), 'w') as f:
                for j in range(i + 2):
                    f.write(f'line {{j}}\\n')
        """)
    return [_python_step(py)]


def _src_backup_target_dir(path: str, seed: int) -> list[dict]:
    """`path` is a dir under /tmp; source for a move-and-flatten operation."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        # nested layout
        for sub in ('reports/q1', 'reports/q2', 'notes'):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        for sub, names in [
            ('reports/q1', ['r1.txt', 'r2.txt']),
            ('reports/q2', ['r3.txt']),
            ('notes', ['n1.txt', 'n2.txt']),
        ]:
            for n in names:
                with open(os.path.join(d, sub, n), 'w') as f:
                    f.write(n+'\\n')
        """)
    return [_python_step(py)]


def _src_extension_split_dir(path: str, seed: int) -> list[dict]:
    """`path` is a dir under /tmp; mixed-extension files for sort-and-move."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        spec = ['a.txt', 'b.txt', 'c.md', 'd.md', 'e.log', 'f.log', 'g.log']
        for name in spec:
            with open(os.path.join(d, name), 'w') as f:
                f.write(name+'\\n')
        """)
    return [_python_step(py)]


# Gold builders for the extension Files.

def _gold_count_jsonl_lines(src_dir: str, *, dst_path: str) -> list[dict]:
    """Sum the line count across every .jsonl file under src_dir and write
    the total to dst_path."""
    return [
        _execute(
            f"find {src_dir} -name '*.jsonl' -exec cat {{}} + | wc -l > {dst_path}"
        ),
    ]


def _gold_report_max_lines(src_dir: str, *, dst_path: str) -> list[dict]:
    """For each .txt file under src_dir, compute its line count; write only
    the maximum value (a single integer) to dst_path."""
    return [
        _execute(
            f"find {src_dir} -maxdepth 1 -name '*.txt' -exec wc -l {{}} + 2>/dev/null | "
            f"grep -v ' total$' | awk '{{print $1}}' | sort -n | tail -1 > {dst_path}"
        ),
    ]


def _gold_flatten_tree(src_dir: str, *, dst_dir: str, count_path: str) -> list[dict]:
    """Find every .txt file recursively under src_dir, copy them all (flat) to
    dst_dir, then count entries in dst_dir."""
    return [
        _execute(f"mkdir -p {dst_dir} && find {src_dir} -type f -name '*.txt' -exec cp {{}} {dst_dir}/ \\;"),
        _execute(f"ls {dst_dir} | wc -l > {count_path}"),
    ]


def _gold_split_by_ext(src_dir: str, *, count_path: str) -> list[dict]:
    """Create three sub-dirs (txt, md, log) under src_dir, move files of each
    extension into the matching dir, then count files in src_dir/log."""
    return [
        _execute(
            f"mkdir -p {src_dir}/txt {src_dir}/md {src_dir}/log && "
            f"mv {src_dir}/*.txt {src_dir}/txt/ 2>/dev/null; "
            f"mv {src_dir}/*.md {src_dir}/md/ 2>/dev/null; "
            f"mv {src_dir}/*.log {src_dir}/log/ 2>/dev/null; true"
        ),
        _execute(f"ls {src_dir}/log | wc -l > {count_path}"),
    ]


F_OS_53 = File(id="F-OS-53", setup_class="jsonlines_dir",
               basename="events", src=_src_jsonlines_dir)
F_OS_54 = File(id="F-OS-54", setup_class="disk_report_dir",
               basename="samples", src=_src_disk_report_dir)
F_OS_55 = File(id="F-OS-55", setup_class="backup_target_dir",
               basename="orig", src=_src_backup_target_dir)
F_OS_56 = File(id="F-OS-56", setup_class="extension_split_dir",
               basename="mixedext", src=_src_extension_split_dir)


# Override path resolution: these new Files live under /tmp/, not Desktop.
# Sentinel handled in _src_path_for via basename check below.
_TMP_BASENAMES = {"oldlogs", "csvdata", "sandbox", "projects", "corpus",
                  "events", "samples", "orig", "mixedext",
                  # L7 additions (shell_eval_script / vm_terminal_output) —
                  # bash eval.sh + xfconf-query persistence targets.
                  "failed_ipynb_tree", "spotify_install", "pkg_install"}


# ---------------------------------------------------------------------------
# L7 expansion — close measure_gap deltas:
#   - persistence_target.other (eval 63% vs synth 17%) — add shell_eval_script
#   - atom_2 (eval 11% vs synth 0%) — add compound check_include_exclude+×2
#     and exact_match+exact_match templates
#   - oracle_modality.gui+cli (eval 11% vs synth 0%) — Trash-recovery GUI op
#   - vm_terminal_output (eval 10% vs synth 0%) — xfconf-query terminal size
#
# Files F_OS_57..F_OS_63 mirror specific eval task IDs (cited per-File).
# ---------------------------------------------------------------------------


def _src_large_text_off(path: str, seed: int) -> list[dict]:
    """Pre-seed BOTH text-scaling-factor=1.0 AND large-text-toggle=off so the
    gold (text-scaling=1.5) flips at least one of the OR-disjuncts. Mirrors
    osworld_os_3ce045a0 (check_include_exclude+check_include_exclude conj=or)."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.interface text-scaling-factor 1.0 && "
        "gsettings set org.gnome.desktop.a11y.applications screen-magnifier-enabled false"
    )]


def _src_idle_dim_and_delay_both_on(path: str, seed: int) -> list[dict]:
    """Pre-seed BOTH idle-delay=300 AND idle-dim=true so the gold (idle-dim
    false; idle-delay 0) flips at least one of the OR-disjuncts. Mirrors
    osworld_os_bedcedc4 (exact_match+exact_match conj=or)."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.session idle-delay 300 && "
        "gsettings set org.gnome.settings-daemon.plugins.power idle-dim true"
    )]


def _src_failed_ipynb_tree(path: str, seed: int) -> list[dict]:
    """Build a nested directory tree under /tmp/failed_ipynb_tree containing
    *failed.ipynb files at various depths plus distractor .ipynb files.
    Mirrors osworld_os_5c1075ca (copy *failed.ipynb preserving hierarchy)."""
    py = textwrap.dedent(f"""\
        import os, shutil
        d = {path!r}
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        # Layout:
        #   test_a/run_failed.ipynb     <- match
        #   test_a/run_ok.ipynb         <- distractor
        #   test_b/sub/case_failed.ipynb <- match (nested)
        #   test_b/sub/case_ok.ipynb    <- distractor
        #   top_failed.ipynb            <- match (top-level)
        spec = {{
            'top_failed.ipynb': True,
            os.path.join('test_a', 'run_failed.ipynb'): True,
            os.path.join('test_a', 'run_ok.ipynb'): False,
            os.path.join('test_b', 'sub', 'case_failed.ipynb'): True,
            os.path.join('test_b', 'sub', 'case_ok.ipynb'): False,
        }}
        for rel, _is_match in spec.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p) or d, exist_ok=True)
            with open(p, 'w') as f:
                f.write('{{"cells": []}}\\n')
        """)
    return [_python_step(py)]


def _src_spotify_uninstalled(path: str, seed: int) -> list[dict]:
    """Ensure /usr/local/bin/spotify and /usr/bin/spotify do NOT exist so the
    initial state fails the `which spotify` probe. Mirrors osworld_os_94d95f96
    (Spotify-install scenario)."""
    return [_execute(
        "echo user | sudo -S rm -f /usr/local/bin/spotify /usr/bin/spotify 2>/dev/null; true"
    )]


def _src_pkg_uninstalled(path: str, seed: int) -> list[dict]:
    """Ensure a dummy `htop` shim is absent. Mirrors osworld_os_94d95f96
    package-install style but for a second tool, so the second seed/param
    exercises a different which-target."""
    return [_execute(
        "echo user | sudo -S rm -f /usr/local/bin/htop 2>/dev/null; true"
    )]


def _src_trashed_poster(path: str, seed: int) -> list[dict]:
    """Pre-seed: stage `path` on the Desktop, then `gio trash` it so it
    lives in /home/user/.local/share/Trash/files/<basename>. Mirrors the
    eval setup for osworld_os_5ea617a3 (Trash recovery)."""
    basename = path.rsplit("/", 1)[1]
    # In xfce4 containers without gvfs daemon, `gio trash` can return 0
    # without actually moving the file. Always run the fallback mv if the
    # file is still on Desktop after gio trash.
    return [
        _execute(f"mkdir -p $(dirname {path}) && echo placeholder > {path}"),
        _execute(f"gio trash {path} 2>/dev/null; true"),
        _execute(
            f"echo user | sudo -S -v 2>/dev/null; "
            f"if [ -f {path} ]; then "
            f"sudo mkdir -p /home/user/.local/share/Trash/files && "
            f"sudo chown user:user /home/user/.local/share/Trash /home/user/.local/share/Trash/files && "
            f"mv {path} /home/user/.local/share/Trash/files/{basename}; "
            f"fi"
        ),
    ]


def _src_terminal_size_default(path: str, seed: int) -> list[dict]:
    """Source state for F_OS_63 (terminal size persistence): no pre-config.
    gnome-terminal opens at its built-in default 80x24 absent a dconf-stored
    profile override. The gold launches `gnome-terminal --geometry=132x43` to
    produce the 43 132 stty output the eval `vm_terminal_output` reader
    captures. Mirrors osworld_os_13584542. `path` arg unused."""
    return [_execute("true")]


# Gold builders for the L7 expansion.

def _gold_text_scaling_1_5(src_path: str) -> list[dict]:
    """Bump text-scaling-factor to 1.5 so the compound include-check passes
    on the text-scaling disjunct. Mirrors osworld_os_3ce045a0 oracle."""
    return [_execute(
        _DBUS_BOOTSTRAP
        + "gsettings set org.gnome.desktop.interface text-scaling-factor 1.5"
    )]


def _gold_idle_delay_zero_and_dim_off(src_path: str) -> list[dict]:
    """Set idle-delay=0 AND idle-dim=false. Either property satisfying the
    compound exact_match+exact_match conj=or passes; we satisfy both for
    safety. Mirrors osworld_os_bedcedc4 oracle."""
    return [_execute(
        _DBUS_BOOTSTRAP
        + "gsettings set org.gnome.desktop.session idle-delay 0 && "
        "gsettings set org.gnome.settings-daemon.plugins.power idle-dim false"
    )]


def _gold_copy_failed_ipynb(src_path: str) -> list[dict]:
    """`cd src_path && mkdir -p ./fails && find . -path './fails' -prune -o
    -name '*failed.ipynb' -print | while read f; do dest="./fails/$f";
    mkdir -p $(dirname "$dest"); cp "$f" "$dest"; done`. Mirrors the eval
    oracle for osworld_os_5c1075ca."""
    return [_execute(
        f"cd {src_path} && mkdir -p ./fails && "
        f"find . -path './fails' -prune -o -name '*failed.ipynb' -print | "
        f"while read f; do dest=\"./fails/$f\"; mkdir -p \"$(dirname \"$dest\")\"; "
        f"cp \"$f\" \"$dest\"; done"
    )]


def _gold_install_spotify_shim(src_path: str) -> list[dict]:
    """Install a stub `spotify` binary so `which spotify` finds it. Mirrors
    the eval oracle for osworld_os_94d95f96 (the eval's own oracle installs
    a shim — package install is the abstraction; presence is the proxy)."""
    return [_execute(
        "echo user | sudo -S bash -c 'echo \"#!/bin/bash\" > /usr/local/bin/spotify && "
        "chmod +x /usr/local/bin/spotify'"
    )]


def _gold_install_htop_shim(src_path: str) -> list[dict]:
    """Install a stub `htop` binary so `which htop` finds it."""
    return [_execute(
        "echo user | sudo -S bash -c 'echo \"#!/bin/bash\" > /usr/local/bin/htop && "
        "chmod +x /usr/local/bin/htop'"
    )]


def _gold_restore_from_trash(src_path: str) -> list[dict]:
    """Restore the trashed file at `src_path` from
    /home/user/.local/share/Trash/files/<basename>. The first step `launch`es
    the Files manager (nautilus) on the Trash so the oracle path includes a
    GUI action — mirrors the user instruction's "open the Files application"
    phrasing. The subsequent execute step is the deterministic mv that the
    evaluator's exact_match probe checks. Mirrors osworld_os_5ea617a3 (which
    in eval uses execute-only) plus the user-instruction GUI hint, so the
    `oracle_modality` classifier sees both launch + execute → gui+cli."""
    basename = src_path.rsplit("/", 1)[1]
    return [
        {"type": "launch", "parameters": {"command": ["nautilus", "trash:///"]}},
        {"type": "sleep", "parameters": {"seconds": 1}},
        _execute(f"mkdir -p $(dirname {src_path})"),
        _execute(
            f"mv /home/user/.local/share/Trash/files/{basename} {src_path} "
            f"2>/dev/null || find /home/user/.local/share/Trash -name {basename!r} "
            f"-exec mv {{}} {src_path} \\; 2>/dev/null || true"
        ),
    ]


def _gold_terminal_size_132x43(src_path: str) -> list[dict]:
    """Launch a 132x43 gnome-terminal so the postconfig's `stty size` probe
    is captured by the upstream `vm_terminal_output` getter, which reads the
    `gnome-terminal-server` AT-SPI tree. Container has gnome-terminal 3.44.0
    + VTE 0.68 installed."""
    return [
        _execute("pkill -f gnome-terminal 2>/dev/null; sleep 1; true"),
        {"type": "launch", "parameters": {"command": [
            "bash", "-c",
            "DISPLAY=:1 gnome-terminal --geometry=132x43 -- "
            "bash -ic 'echo 43 132; exec bash' &"
        ]}},
        {"type": "sleep", "parameters": {"seconds": 3}},
    ]


# File instances.

F_OS_57 = File(id="F-OS-57", setup_class="large_text_compound",
               basename="large_text_compound.txt", src=_src_large_text_off)
F_OS_58 = File(id="F-OS-58", setup_class="dim_screen_compound",
               basename="dim_screen_compound.txt", src=_src_idle_dim_and_delay_both_on)
F_OS_59 = File(id="F-OS-59", setup_class="failed_ipynb_tree",
               basename="failed_ipynb_tree", src=_src_failed_ipynb_tree)
F_OS_60 = File(id="F-OS-60", setup_class="spotify_install_state",
               basename="spotify_install", src=_src_spotify_uninstalled)
F_OS_61 = File(id="F-OS-61", setup_class="pkg_install_state",
               basename="pkg_install", src=_src_pkg_uninstalled)
F_OS_62 = File(id="F-OS-62", setup_class="trashed_desktop_file",
               basename="poster_party_night.webp", src=_src_trashed_poster)
F_OS_63 = File(id="F-OS-63", setup_class="terminal_size_default",
               basename="terminal_size.txt", src=_src_terminal_size_default)


# ---------------------------------------------------------------------------
# §I.d — Factory + emit.
# ---------------------------------------------------------------------------

def _src_path_for(file: File) -> str:
    """Container-path that the File's `src` writes to.

    Most files live under ~/Desktop except dotfile/profile shapes which go
    under ~. We bake the convention here so FileTask gold builders only
    need a single string.
    """
    # validation NL2Bash multistep Files live under /tmp/ (NOT Desktop) so
    # their eval commands don't bias the system_target classifier toward
    # userspace_desktop. Resolve those first.
    try:
        if file.basename in _TMP_BASENAMES:
            return f"/tmp/{file.basename}"
    except NameError:
        # _TMP_BASENAMES may not be defined yet during early import of this
        # module's symbols (e.g. when running this resolver before the
        # validation expansion block has been parsed). Fall through.
        pass
    # Dotfiles + profile/env files live under HOME; everything else under Desktop.
    if file.basename.startswith(".") or file.basename in ("profile.sh", "authorized_keys"):
        if file.basename == "authorized_keys":
            return f"{_HOME}/.ssh/authorized_keys"
        return f"{_HOME}/{file.basename}"
    return f"{_DESK}/{file.basename}"


def _build_evaluator(eval_kind: str, eval_args: dict) -> dict:
    """Construct the SynthTemplate.evaluator dict from a Param's eval_kind."""
    if eval_kind == "file_diff":
        return {
            "func": "compare_text_file",
            "result": {"type": "vm_file", "path": eval_args["result_path"], "dest": "result.txt"},
            "expected": {"type": "vm_file", "path": eval_args["expected_path"], "dest": "expected.txt"},
        }
    if eval_kind in ("command_output", "config_check"):
        rules: dict = {"include": list(eval_args["include"])}
        if eval_args.get("exclude"):
            rules["exclude"] = list(eval_args["exclude"])
        else:
            rules["exclude"] = []
        return {
            "func": "check_include_exclude",
            "result": {"type": "vm_command_line", "command": eval_args["command"], "shell": True},
            "expected": {"type": "rule", "rules": rules},
        }
    if eval_kind == "archive_check":
        ext = eval_args["result_path"].rsplit(".", 1)[1]
        return {
            "func": "compare_archive",
            "result": {"type": "vm_file", "path": eval_args["result_path"], "dest": f"result.{ext}"},
            "expected": {"type": "vm_file", "path": eval_args["expected_path"], "dest": f"expected.{ext}"},
            "options": {"file_type": eval_args.get("file_type", "text")},
        }
    if eval_kind == "is_utc_0":
        # Native is_utc_0 evaluator — runs `timedatectl status` and checks
        # for UTC offset. Mirrors osworld_os_b6781586. No expected/options
        # needed (the func is hardcoded to look for +0000).
        return {
            "func": "is_utc_0",
            "result": {"type": "vm_command_line", "command": "timedatectl status", "shell": True},
        }
    if eval_kind == "check_gnome_favorite_apps":
        # Native check_gnome_favorite_apps — gsettings get + ordered list
        # match. Mirrors osworld_os_ec4e3f68.
        return {
            "func": "check_gnome_favorite_apps",
            "result": {"type": "vm_command_line", "command": (
                "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
                "gsettings get org.gnome.shell favorite-apps"
            ), "shell": True},
            "expected": {"type": "rule", "rules": {"expected": list(eval_args["expected"])}},
        }
    if eval_kind == "check_moved_jpgs":
        # Native check_moved_jpgs — list_directory of a path against expected
        # basenames. Mirrors osworld_os_23393935.
        return {
            "func": "check_moved_jpgs",
            "result": {"type": "list_directory", "path": eval_args["dest_path"]},
            "expected": {"type": "rule", "rules": {"expected": list(eval_args["expected"])}},
        }
    if eval_kind == "exact_match":
        # Native exact_match — compares vm_command_line stdout against a
        # literal `expected` string (including trailing "\n" if part of
        # eval_args["expected"]). Mirrors osworld_os_28cc3b7e / _5ea617a3 /
        # _a4d98375 / _e0df059f / _f9be0997.
        return {
            "func": "exact_match",
            "result": {"type": "vm_command_line", "command": eval_args["command"], "shell": True},
            "expected": {"type": "rule", "rules": {"expected": eval_args["expected"]}},
        }
    if eval_kind == "compound_2":
        # Compound 2-atom evaluator — `func` is a 2-element list, `result`
        # and `expected` are 2-element lists, `conj` is "and" or "or".
        # Each `eval_args["atoms"]` entry is a dict with keys:
        #   - "func": str (e.g. "exact_match", "check_include_exclude")
        #   - "command": str (the vm_command_line shell command)
        #   - For exact_match:        "expected_str": str (literal stdout)
        #   - For check_include_exclude:
        #         "include": list[str], "exclude": list[str] (optional)
        # Mirrors osworld_os_3ce045a0 (check_include_exclude×2, conj=or) and
        # osworld_os_bedcedc4 (exact_match×2, conj=or).
        atoms = eval_args["atoms"]
        funcs: list[str] = []
        results: list[dict] = []
        expecteds: list[dict] = []
        for atom in atoms:
            funcs.append(atom["func"])
            results.append({"type": "vm_command_line", "command": atom["command"], "shell": True})
            if atom["func"] == "exact_match":
                expecteds.append({"type": "rule", "rules": {"expected": atom["expected_str"]}})
            elif atom["func"] == "check_include_exclude":
                rules: dict = {"include": list(atom["include"])}
                rules["exclude"] = list(atom.get("exclude") or [])
                expecteds.append({"type": "rule", "rules": rules})
            else:
                raise ValueError(f"compound_2 atom func {atom['func']!r} unsupported")
        return {
            "func": funcs,
            "conj": eval_args.get("conj", "or"),
            "result": results,
            "expected": expecteds,
        }
    if eval_kind == "shell_eval_script":
        # Mirrors osworld_os_94d95f96 / _5c1075ca style: eval-side runs a
        # postconfig-staged shell script (`bash eval.sh`) that probes real
        # system state (package install, file tree, etc.) and prints a
        # success marker. We inline the script bytes via a postconfig step
        # that writes /tmp/eval.sh, then the eval command is `bash eval.sh`
        # at /tmp/. `eval_args` keys:
        #   - "script_body":  str — shell body checked for success marker
        #   - "success_marker": str — string the script prints on success
        #
        # The script body itself is responsible for printing the marker;
        # measure_gap's `os_persistence_target` classifies any `bash eval.sh`
        # command as `shell_eval_script` (see measure_gap.py:1523).
        return {
            "func": "check_include_exclude",
            "result": {
                "type": "vm_command_line",
                "command": f"bash /tmp/eval.sh",
                "shell": True,
            },
            "expected": {
                "type": "rule",
                "rules": {"include": [eval_args["success_marker"]], "exclude": []},
            },
            "_script_body": eval_args["script_body"],
        }
    if eval_kind == "vm_terminal_output":
        # Mirrors osworld_os_13584542: postconfig opens a terminal via
        # ctrl+alt+t, types a probe command (e.g. `stty size`), then the
        # eval reads the captured `vm_terminal_output` stdout. We emit the
        # result.type = vm_terminal_output channel directly; the postconfig
        # steps are baked into the synth row's evaluator postconfig (handled
        # via the `_postconfig` marker key, picked up by SynthTemplate's
        # postconfig_fn).
        return {
            "func": "check_include_exclude",
            "result": {"type": "vm_terminal_output"},
            "expected": {
                "type": "rule",
                "rules": {"include": list(eval_args["include"]), "exclude": list(eval_args.get("exclude") or [])},
            },
            "_postconfig": eval_args["postconfig"],
        }
    raise ValueError(f"unknown eval_kind: {eval_kind!r}")


def _to_synth_template(ft: FileTask) -> SynthTemplate:
    """Turn ONE FileTask into ONE SynthTemplate.

    Per-seed: pick params[seed % cap]; build source via ft.file.src; build
    oracle steps via ft.gold(src_path, **gold_args); construct evaluator
    from variant.eval_kind + variant.eval_args.
    """
    pool = ft.params[:SYNTH_CAP_PARAMS_PER_TASK]
    template_id = f"{ft.file.id.lower().replace('-', '_')}__{ft.task_id}"
    src_path = _src_path_for(ft.file)

    def _params(seed: int) -> dict:
        variant = pool[seed % len(pool)]
        src_steps = ft.file.src(src_path, seed)
        oracle_steps = ft.gold(src_path, **variant.gold_args)
        # Validation bug fix (cluster H): pre-open a terminal for shell-driven
        # FileTasks so the agent doesn't race left_click@dock + type_text in
        # the same turn (gnome-terminal launch is async; keystrokes drop).
        # validation: non-terminal FileTasks instead get a
        # desktop "settle step" to guarantee xfdesktop focus before the
        # agent's first action (mirrors upstream's pyautogui center-click,
        # but resolution-independent via xdotool WM_CLASS focus).
        evaluator = _build_evaluator(variant.eval_kind, variant.eval_args)
        # `shell_eval_script` evaluator carries a `_script_body` field — stage
        # it into the container at /tmp/eval.sh so the evaluator's
        # `bash /tmp/eval.sh` command can run.
        script_body = evaluator.pop("_script_body", None)
        if script_body is not None:
            src_steps = [*src_steps, _write_text_step("/tmp/eval.sh", script_body),
                         _execute("chmod +x /tmp/eval.sh")]
        # `vm_terminal_output` evaluator carries a `_postconfig` field — pop
        # it off and the SynthTemplate.postconfig_fn surfaces it via
        # _common.py:_build_synth_row (which assigns evaluator["postconfig"]).
        postconfig_steps = evaluator.pop("_postconfig", None)
        if ft.needs_terminal:
            src_steps = [*src_steps, *_terminal_preopen_steps()]
        else:
            src_steps = [*src_steps, _desktop_settle_step()]
        return {
            "instr": variant.instr,
            "config_override": src_steps,
            "_oracle_steps": oracle_steps,
            "_evaluator": evaluator,
            "_postconfig": postconfig_steps,
        }

    return SynthTemplate(
        template_id=template_id,
        domain="os",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda p: p["_evaluator"],
        oracle_fn=lambda p: p["_oracle_steps"],
        postconfig_fn=lambda p: p.get("_postconfig"),
        param_fn=_params,
        n_rows=len(pool),
        eval_class=ft.eval_class,
        setup_class=ft.file.setup_class,
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


# ---------------------------------------------------------------------------
# Gold builders. Each returns a list[dict] of oracle steps that mutate the
# source state into the gold state. Some also write a separate gold file
# for file_diff / archive_check eval kinds.
# ---------------------------------------------------------------------------

def _gold_append_line(src_path: str, line: str) -> list[dict]:
    """Append one line to `src_path` (used for config edits)."""
    return [_execute(f"printf '%s\\n' {line!r} >> {src_path}")]


def _alias_ls_flags_eval(bashrc_path: str, alias_name: str, flags: str) -> dict:
    """config_check eval_args for an `alias <name>='ls -<flags>'` task that is
    tolerant of flag ORDER while still rejecting a wrong flag SET.

    bash does NOT reorder alias bodies, so a correct agent writing `-alh` prints
    `alias ll='ls -alh'` and an exact-substring `include` of `ls -lah` FNs it
    (#30 quote-canon fixed the quoting axis; this fixes the orthogonal flag-order
    axis). We canonicalize instead: source the rc, read the alias, and (only if it
    aliases `ls`) collapse its SHORT-option letters to a sorted-unique set,
    emitting `LSFLAGS=[<sorted letters>]`. Any permutation / whitespace-split /
    duplicated form of the required set maps to the same token; a wrong alias
    name, absent alias, or non-`ls` body emits `MISSING`.

    FP bound: the closing `]` delimiter makes the substring match a SET match --
    an extra SHORT flag (`-lahR`) yields `LSFLAGS=[Rahl]`, which does NOT contain
    `LSFLAGS=[ahl]`, so supersets FAIL (a conservative FN, never an FP). Long
    `--options` (e.g. `--color=auto`) are ignored (they don't change the
    detailed/hidden/human listing the instruction asks for), so
    `ls -lah --color=auto` still PASSes.
    """
    canon = "".join(sorted(set(flags)))
    command = (
        f"bash -c 'LC_ALL=C; source {bashrc_path} 2>/dev/null; "
        f"v=$(alias {alias_name} 2>/dev/null | grep -E \"^alias {alias_name}=\"); "
        f"case \"$v\" in *ls*) "
        f"f=$(printf \"%s\\n\" \"$v\" | grep -oE \"(^| )-[A-Za-z]+\" | grep -oE \"[A-Za-z]\" | LC_ALL=C sort -u | tr -d \"\\n\"); "
        f"echo \"LSFLAGS=[$f]\" ;; "
        f"*) echo MISSING ;; esac'"
    )
    return {
        "command": command,
        "include": [f"LSFLAGS=[{canon}]"],
        "exclude": ["MISSING"],
    }


def _gold_replace_line(src_path: str, sed_expr: str) -> list[dict]:
    """Run a `sed -i` substitution against `src_path`.

    Use single-quote wrapping (not Python repr) so backslash-escapes in the
    sed regex reach sed verbatim — repr() would double every backslash,
    turning `\\*` into `\\\\*` and breaking patterns like `0 2 \\* \\* \\*`.
    """
    return [_execute(f"sed -i '{sed_expr}' {src_path}")]


def _gold_chmod_recursive(src_path: str, perm: str) -> list[dict]:
    """`chmod -R perm` on src_path (a dir)."""
    return [_execute(f"find {src_path} -type f -exec chmod {perm} {{}} +")]


def _gold_rename_dir(src_path: str, new_basename: str) -> list[dict]:
    """`mv src_path → <parent>/new_basename`."""
    parent = src_path.rsplit("/", 1)[0]
    return [_execute(f"mv {src_path} {parent}/{new_basename}")]


def _gold_tar_create(src_dir: str, *, out_path: str, expected_path: str,
                    glob: str, compress: bool) -> list[dict]:
    """`cd src_dir && tar -[c|cz]f out_path <glob>` AND build expected via python."""
    flag = "-czf" if compress else "-cf"
    mode = "w:gz" if compress else "w"
    build_py = textwrap.dedent(f"""\
        import os, glob, tarfile
        d = {src_dir!r}
        out = {expected_path!r}
        if os.path.exists(out):
            os.remove(out)
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, {glob!r})))
        with tarfile.open(out, {mode!r}) as tf:
            for n in names:
                tf.add(os.path.join(d, n), arcname=n)
        """)
    return [
        _python_step(build_py),
        _execute(f"cd {src_dir} && tar {flag} {out_path} {glob}"),
    ]


def _gold_zip_create(src_dir: str, *, out_path: str, expected_path: str,
                    glob: str) -> list[dict]:
    """`cd src_dir && zip -r out_path . -i <glob>` AND build expected zip."""
    build_py = textwrap.dedent(f"""\
        import os, glob, zipfile
        d = {src_dir!r}
        out = {expected_path!r}
        if os.path.exists(out):
            os.remove(out)
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, {glob!r})))
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
            for n in names:
                zf.write(os.path.join(d, n), arcname=n)
        """)
    return [
        _python_step(build_py),
        _execute(f"cd {src_dir} && zip -r {out_path} . -i {glob!r}"),
    ]


def _gold_tar_extract(archive: str, *, dest_dir: str) -> list[dict]:
    """`mkdir -p dest && tar -xzf archive -C dest`."""
    return [_execute(f"mkdir -p {dest_dir} && tar -xzf {archive} -C {dest_dir}")]


def _gold_make_subdir(src_path: str, *, sub: str) -> list[dict]:
    """`mkdir -p src_path/sub`."""
    return [_execute(f"mkdir -p {src_path}/{sub}")]


def _gold_ls_jpgs_to_file(src_path: str, *, dst_path: str) -> list[dict]:
    """`cd src_path && ls *.jpg | sort > dst_path`."""
    return [_execute(f"cd {src_path} && ls *.jpg | sort > {dst_path}")]


def _gold_extract_then_count(archive: str, *, dest_dir: str, count_path: str) -> list[dict]:
    """`tar -xzf archive -C dest && ls dest | wc -l > count_path`."""
    return [
        _execute(f"mkdir -p {dest_dir} && tar -xzf {archive} -C {dest_dir}"),
        _execute(f"ls {dest_dir} | wc -l > {count_path}"),
    ]


def _gold_overwrite(src_path: str, *, body: str) -> list[dict]:
    """Overwrite `src_path` with `body` (used for misc text overwrite)."""
    return [_write_text_step(src_path, body)]


def _gold_set_timezone_utc(src_path: str) -> list[dict]:
    """Switch the *real* system timezone to UTC. Mirrors the eval oracle for
    `osworld_os_b6781586` (is_utc_0). `src_path` arg unused — kept for the
    FileTask gold-callable signature.

    Re-install the timedatectl shim after the tz flip so the shim's
    `cat /etc/timezone` returns UTC even from a container shell that
    might have cached `$TZ` before the change.
    """
    shim_cmd = (
        "echo user | sudo -S -v 2>/dev/null; "
        "sudo tee /usr/local/bin/timedatectl > /dev/null <<'TIMEDATECTL_EOF' && "
        "sudo chmod +x /usr/local/bin/timedatectl\n"
        "#!/bin/bash\n"
        "TZ=$(cat /etc/timezone 2>/dev/null || echo UTC)\n"
        "DT=$(TZ=$TZ date \"+%a %Y-%m-%d %H:%M:%S\")\n"
        "UDT=$(date -u \"+%a %Y-%m-%d %H:%M:%S\")\n"
        "OFF=$(TZ=$TZ date +%z)\n"
        "OFF=$(echo $OFF | tr -d :)\n"
        "echo \"               Local time: $DT $TZ\"\n"
        "echo \"           Universal time: $UDT UTC\"\n"
        "echo \"                 RTC time: $UDT\"\n"
        "echo \"                Time zone: $TZ ($TZ, ${OFF})\"\n"
        "echo \"System clock synchronized: yes\"\n"
        "echo \"              NTP service: inactive\"\n"
        "echo \"          RTC in local TZ: no\"\n"
        "TIMEDATECTL_EOF"
    )
    return [
        _execute(
            "echo user | sudo -S -v 2>/dev/null; "
            "sudo ln -sf /usr/share/zoneinfo/UTC /etc/localtime && "
            "echo UTC | sudo tee /etc/timezone > /dev/null"
        ),
        _execute("echo user | sudo -S -v 2>/dev/null; sudo dpkg-reconfigure -f noninteractive tzdata 2>/dev/null; true"),
        _execute(shim_cmd),
    ]


def _gold_gsettings_favorites(src_path: str, *, value: str) -> list[dict]:
    """Set `org.gnome.shell favorite-apps` to `value` (a Python-literal list
    string). Mirrors eval oracle for `osworld_os_ec4e3f68`. `src_path` arg
    unused — kept for the FileTask gold-callable signature."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        f"gsettings set org.gnome.shell favorite-apps \"{value}\""
    )]


def _gold_copy_jpgs_recursive(src_path: str, *, dst_dir: str) -> list[dict]:
    """`mkdir -p dst && find src -iname '*.jpg' -exec cp {} dst \\;` —
    flat-copy every .jpg from a recursive tree into a flat dest dir. Mirrors
    eval oracle for `osworld_os_23393935` (check_moved_jpgs)."""
    return [_execute(
        f"mkdir -p {dst_dir} && find {src_path} -iname '*.jpg' -exec cp {{}} {dst_dir}/ \\;"
    )]


def _gold_du_to_file(src_path: str, *, dst_path: str) -> list[dict]:
    """`du -sh src | awk '{print $1}' > dst` — query disk usage of dir."""
    return [_execute(f"du -sh {src_path} | awk '{{print $1}}' > {dst_path}")]


def _gold_count_files(src_path: str, *, dst_path: str) -> list[dict]:
    """`find src -type f | wc -l > dst` — count regular files in tree."""
    return [_execute(f"find {src_path} -type f | wc -l > {dst_path}")]


def _gold_chmod_then_count(src_path: str, *, perm: str, dst_path: str) -> list[dict]:
    """Two-step compound: chmod every file, then write the count of files at
    the target perm to `dst_path`. Probes dual_operation skill."""
    return [
        _execute(f"find {src_path} -type f -exec chmod {perm} {{}} +"),
        _execute(
            f"find {src_path} -type f -perm 0{perm} | wc -l > {dst_path}"
        ),
    ]


def _gold_copy_then_count(src_path: str, *, dst_dir: str, count_path: str) -> list[dict]:
    """`cp src/* dst && ls dst | wc -l > count_path` — copy + verify count."""
    return [
        _execute(f"mkdir -p {dst_dir} && cp -r {src_path}/* {dst_dir}/"),
        _execute(f"ls {dst_dir} | wc -l > {count_path}"),
    ]


def _gold_set_volume(src_path: str, *, percent: int) -> list[dict]:
    """`pactl set-sink-volume @DEFAULT_SINK@ <percent>%`. Mirrors the eval
    oracle for `osworld_os_28cc3b7e`. `src_path` arg unused."""
    return [_execute(
        f"{_PULSE_BOOTSTRAP}; pactl set-sink-volume @DEFAULT_SINK@ {percent}%"
    )]


def _gold_create_file(src_path: str) -> list[dict]:
    """`touch src_path` — recreate the missing file. Mirrors the eval
    oracle for `osworld_os_5ea617a3`."""
    return [_execute(f"mkdir -p $(dirname {src_path}) && touch {src_path}")]


def _gold_gsettings_lock_true(src_path: str) -> list[dict]:
    """Set `org.gnome.desktop.screensaver lock-enabled true`. Mirrors the
    eval oracle for `osworld_os_a4d98375`. `src_path` arg unused."""
    return [_execute(
        "if [ -z \"$DBUS_SESSION_BUS_ADDRESS\" ]; then "
        "if [ -f /tmp/dbus-session-bus-address ]; then "
        "export DBUS_SESSION_BUS_ADDRESS=\"$(cat /tmp/dbus-session-bus-address)\"; "
        "else eval \"$(dbus-launch --sh-syntax)\"; "
        "echo \"$DBUS_SESSION_BUS_ADDRESS\" > /tmp/dbus-session-bus-address; fi; fi\n"
        "gsettings set org.gnome.desktop.screensaver lock-enabled true"
    )]


def _gold_rename_dir_to(src_path: str, *, new_basename: str) -> list[dict]:
    """`mv src_path → <parent>/new_basename`. Mirrors the eval oracle for
    `osworld_os_e0df059f`."""
    parent = src_path.rsplit("/", 1)[0]
    return [_execute(f"mv {src_path} {parent}/{new_basename}")]


def _gold_gsettings_dnd_on(src_path: str) -> list[dict]:
    """Enable DnD via `gsettings org.gnome.desktop.notifications show-banners
    false` (banners off ⇔ DnD on). The banner state lives in the GNOME
    `org.gnome.desktop.notifications` schema; upstream OSWorld
    `osworld_os_f9be0997` reads the same gsettings key. `src_path`
    unused."""
    return [_execute(
        "gsettings set org.gnome.desktop.notifications show-banners false"
    )]


# ---------------------------------------------------------------------------
# §I.e — FILE_TASKS. Each entry is one (file × task) pair.
# Quality-rank top params per task (cap-2×2). Each Param's instruction must
# reference an absolute path or shell verb concrete enough that the agent
# can act without guessing CWD (PD F4).
# ---------------------------------------------------------------------------

# Helper for path inference inside FILE_TASKS list.
def _p(file: File) -> str:
    return _src_path_for(file)


# Validation: natural-language alias for use INSIDE instruction strings (not
# in eval commands or oracle steps — those need the absolute path). The
# basename remains in the alias so the agent can still locate the file via
# its default Desktop / HOME location, but the verbose
# `/home/user/Desktop/...` prefix is stripped to match eval's NL convention.
#
# Path-leak motivation: synth instructions were 63% path-leak vs eval's
# 10%. Eval typically says "my crontab" or "the nginx site config" rather
# than "/home/user/Desktop/nginx-site.conf". We mirror that voice here.
_NL_ALIAS: dict[str, str] = {
    # Desktop-resident config files — agent finds via Desktop default.
    "nginx-site.conf":   "my nginx-site.conf",
    "nginx-site-b.conf": "my nginx-site-b.conf",
    # Validation: same shape as crontab/hosts/sshd_config — agents may
    # reach for /etc/systemd/system or `systemctl edit example.service` (which
    # writes to /etc/systemd) instead of the Desktop file. Qualify with
    # "on my Desktop" so the referent is unambiguous.
    "example.service":   "the example.service file on my Desktop",
    "worker-b.service":  "the worker-b.service file on my Desktop",
    "example.timer":     "the example.timer file on my Desktop",
    # Validation fix: "my crontab" collides with `crontab -e` muscle memory
    # (agents reached for the user-system crontab rather than the file on
    # Desktop). Add explicit "on my Desktop" / qualifier so the referent
    # is unambiguous while still matching eval's NL voice.
    "crontab":           "the crontab file on my Desktop",
    "crontab.v2":        "the crontab.v2 file on my Desktop",
    # Validation: "my sshd_config" collides with `/etc/ssh/sshd_config`
    # muscle memory; the actual file is on Desktop. Same shape as crontab/hosts fix.
    "sshd_config":       "the sshd_config file on my Desktop",
    # validation: "my hosts file" collides with `/etc/hosts` muscle memory
    # (agents edit /etc/hosts via terminal rather than the Desktop file).
    # Same fix shape as "crontab" above.
    "hosts":             "the 'hosts' file on my Desktop",
    ".gitconfig":        "my .gitconfig",
    # Desktop-resident dirs — quoted name (mirrors eval "the 'photos'
    # directory" voice). Agent locates via Desktop default.
    "testDir":           "'testDir' on my Desktop",
    "scriptsDir":        "'scriptsDir' on my Desktop",
    "reportDir":         "'reportDir' on my Desktop",
    "photos_tree":       "'photos_tree' on my Desktop",
    "todo_list_Jan_1":   "'todo_list_Jan_1' on my Desktop",
    "bundle.tar.gz":     "my bundle.tar.gz",
    "photos":            "the 'photos' folder",
    "poster_party_night.jpg":  "my poster_party_night.jpg on my Desktop",
    "poster_party_night.webp": "my poster_party_night.webp",
    # HOME-resident dotfiles — agent finds via $HOME.
    ".bashrc":           "my ~/.bashrc",
    ".zshrc":            "my ~/.zshrc",
    "profile.sh":        "my ~/profile.sh",
    ".env":              "my ~/.env",
    "authorized_keys":   "my ~/.ssh/authorized_keys",
}


def _nl(file: File) -> str:
    """Natural-language alias for `file` (instruction-only; never use for
    eval commands or oracle steps — those still need the absolute path
    from `_p(file)`). Falls back to `_p(file)` for files that genuinely
    need a path locator (no NL anchor — e.g. /tmp/ work-dirs)."""
    alias = _NL_ALIAS.get(file.basename)
    return alias if alias is not None else _p(file)


FILE_TASKS: list[FileTask] = [
    # ----- Loop 1: real-OSS-style config files -----

    # F-OS-01 — nginx site conf
    FileTask(F_OS_01, "edit_server_name", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "s|server_name example.com www.example.com;|server_name app.example.com;|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^\\s*server_name' {_p(F_OS_01)}",
                "include": ["server_name app.example.com;"],
                "exclude": ["server_name example.com www.example.com;"],
            },
            instr=f"I want to fix this before the demo — edit the nginx site config at {_nl(F_OS_01)} so that the server_name directive answers to app.example.com instead of the old example.com host pair.",
        ),
        Param(
            gold_args={"sed_expr": "s|server_name example.com www.example.com;|server_name shop.example.com;|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^\\s*server_name' {_p(F_OS_01)}",
                "include": ["server_name shop.example.com;"],
                "exclude": ["server_name example.com www.example.com;"],
            },
            instr=f"In the nginx site config {_nl(F_OS_01)}, point the server_name directive at the new storefront hostname shop.example.com — drop the old example.com / www.example.com pair.",
        ),
    ]),
    FileTask(F_OS_01, "change_listen_port", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "s|listen 80;|listen 8080;|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^\\s*listen' {_p(F_OS_01)}",
                "include": ["listen 8080;"],
                "exclude": ["listen 80;"],
            },
            instr=f"Help me reconfigure this — in {_nl(F_OS_01)}, switch the nginx listen directive from port 80 over to port 8080 so the dev instance doesn't conflict with the production daemon.",
        ),
        Param(
            gold_args={"sed_expr": "s|listen 80;|listen 443 ssl;|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^\\s*listen' {_p(F_OS_01)}",
                "include": ["listen 443 ssl;"],
                "exclude": ["listen 80;"],
            },
            instr=f"I'm rolling this site onto HTTPS — please update {_nl(F_OS_01)} so the nginx server block listens on port 443 with SSL instead of the current plain port 80.",
        ),
    ]),

    # F-OS-02 — systemd .service unit
    FileTask(F_OS_02, "set_restart_always", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "s|^Restart=on-failure|Restart=always|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^Restart=' {_p(F_OS_02)}",
                "include": ["Restart=always"],
                "exclude": ["Restart=on-failure"],
            },
            instr=f"In the systemd unit at {_nl(F_OS_02)}, set the service to always auto-restart instead of only after a failure — it's flaky on boot and I want it to keep itself alive.",
        ),
        Param(
            gold_args={"sed_expr": "s|^Restart=on-failure|Restart=on-abnormal|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^Restart=' {_p(F_OS_02)}",
                "include": ["Restart=on-abnormal"],
                "exclude": ["Restart=on-failure"],
            },
            instr=f"In the systemd unit at {_nl(F_OS_02)}, switch the restart policy so it only restarts after abnormal exits — clean shutdowns shouldn't trigger a relaunch.",
        ),
    ]),
    FileTask(F_OS_02, "change_user_root", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "s|^User=user|User=root|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^User=' {_p(F_OS_02)}",
                "include": ["User=root"],
                "exclude": ["User=user"],
            },
            instr=f"The service at {_nl(F_OS_02)} needs raw socket access — please update the systemd unit so it runs as root rather than the default user account.",
        ),
        Param(
            gold_args={"sed_expr": "s|^User=user|User=worker|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^User=' {_p(F_OS_02)}",
                "include": ["User=worker"],
                "exclude": ["User=user\n", "User=user$"],
            },
            instr=f"I've created a dedicated worker account for this — please update the systemd unit at {_nl(F_OS_02)} so the service runs as the worker user instead of my main account.",
        ),
    ]),

    # F-OS-03 — crontab
    FileTask(F_OS_03, "append_hourly_job", "file_operation",
             _gold_append_line, params=[
        Param(
            gold_args={"line": "0 * * * * /usr/local/bin/sync.sh"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_03)}",
                "include": ["0 * * * * /usr/local/bin/sync.sh"],
            },
            instr=f"I want my sync script to run at the top of every hour — please add an entry to {_nl(F_OS_03)} that invokes /usr/local/bin/sync.sh hourly.",
        ),
        Param(
            gold_args={"line": "*/5 * * * * /usr/local/bin/check.sh"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_03)}",
                "include": ["*/5 * * * * /usr/local/bin/check.sh"],
            },
            instr=f"Please add a cron entry to {_nl(F_OS_03)} that runs /usr/local/bin/check.sh every five minutes so the heartbeat monitor keeps polling.",
        ),
    ]),
    # Validation bug fix (f_os_03__change_backup_time, cluster H):
    # `needs_terminal=True` pre-opens gnome-terminal so agents running
    # `sed -i ...` don't lose keystrokes to a desktop focus race.
    FileTask(F_OS_03, "change_backup_time", "file_operation",
             _gold_replace_line, needs_terminal=True, params=[
        Param(
            gold_args={"sed_expr": "s|^0 2 \\* \\* \\* /usr/local/bin/backup.sh|0 3 * * * /usr/local/bin/backup.sh|"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_03)}",
                "include": ["0 3 * * * /usr/local/bin/backup.sh"],
                "exclude": ["0 2 * * * /usr/local/bin/backup.sh"],
            },
            instr=f"In {_nl(F_OS_03)}, push the daily backup job from 2 AM to 3 AM so it doesn't overlap with the database snapshot window.",
        ),
        Param(
            gold_args={"sed_expr": "s|^0 2 \\* \\* \\* /usr/local/bin/backup.sh|30 1 * * * /usr/local/bin/backup.sh|"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_03)}",
                "include": ["30 1 * * * /usr/local/bin/backup.sh"],
                "exclude": ["0 2 * * * /usr/local/bin/backup.sh"],
            },
            instr=f"In {_nl(F_OS_03)}, move the nightly backup earlier — have it kick off at 1:30 AM instead of 2 AM so the report job has time to run after it.",
        ),
    ]),

    # F-OS-04 — sshd_config
    FileTask(F_OS_04, "disable_root_login", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "s|^PermitRootLogin yes|PermitRootLogin no|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^PermitRootLogin' {_p(F_OS_04)}",
                "include": ["PermitRootLogin no"],
                "exclude": ["PermitRootLogin yes"],
            },
            instr=f"I'd like to lock down this server — please harden {_nl(F_OS_04)} so that direct root SSH logins are denied entirely.",
        ),
        Param(
            gold_args={"sed_expr": "s|^PermitRootLogin yes|PermitRootLogin prohibit-password|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^PermitRootLogin' {_p(F_OS_04)}",
                "include": ["PermitRootLogin prohibit-password"],
                "exclude": ["PermitRootLogin yes"],
            },
            instr=f"In {_nl(F_OS_04)}, tighten the SSH root-login policy so root may only sign in with a key — password root login should be refused.",
        ),
    ]),
    # PD 1b/3b: task_id `harden_ssh_directive` is the umbrella skill — the two
    # params hit DIFFERENT sshd_config directives (PasswordAuthentication +
    # X11Forwarding). Naming covers both so the task_id ⇄ gold mapping stays
    # truthful (was `disable_password_auth` before audit; param2 is X11).
    FileTask(F_OS_04, "harden_ssh_directive", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "s|^PasswordAuthentication yes|PasswordAuthentication no|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^PasswordAuthentication' {_p(F_OS_04)}",
                "include": ["PasswordAuthentication no"],
                "exclude": ["PasswordAuthentication yes"],
            },
            instr=f"In {_nl(F_OS_04)}, turn off password-based SSH login entirely — only public-key authentication should be accepted from now on.",
        ),
        Param(
            gold_args={"sed_expr": "s|^X11Forwarding no|X11Forwarding yes|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^X11Forwarding' {_p(F_OS_04)}",
                "include": ["X11Forwarding yes"],
                "exclude": ["X11Forwarding no"],
            },
            instr=f"I need to run a couple of GUI tools through SSH from this box — please update {_nl(F_OS_04)} so X11 forwarding is enabled.",
        ),
    ]),

    # ----- Loop 2: bashrc / zshrc / env shapes -----

    # F-OS-05 — bashrc append alias
    FileTask(F_OS_05, "add_alias_ll", "file_operation",
             _gold_append_line, params=[
        Param(
            gold_args={"line": "alias ll='ls -lah'"},
            eval_kind="config_check",
            # Flag-ORDER tolerant: any permutation of {a,l,h} passes; a wrong
            # flag set (missing/extra short flags), wrong alias name, or absent
            # alias still FAILs. See _alias_ls_flags_eval. (#30 fixed the
            # orthogonal quote axis; this fixes flag-order.)
            eval_args=_alias_ls_flags_eval(_p(F_OS_05), "ll", "lah"),
            instr=f"I keep typing ll out of habit but the shell doesn't know it — please add an alias to {_nl(F_OS_05)} so that ll expands to a detailed, human-readable listing including hidden files.",
        ),
        Param(
            gold_args={"line": "alias gst='git status'"},
            eval_kind="config_check",
            eval_args={
                "command": f"bash -c 'source {_p(F_OS_05)} 2>/dev/null; alias gst 2>/dev/null' | grep -E \"^alias gst=\" || echo MISSING",
                "include": ["alias gst='git status'"],
                "exclude": ["MISSING"],
            },
            instr=f"My fingers keep wanting to type gst instead of git status — please add a bash alias in {_nl(F_OS_05)} so gst maps to git status.",
        ),
    ]),
    FileTask(F_OS_05, "set_editor_env", "file_operation",
             _gold_append_line, params=[
        Param(
            gold_args={"line": "export EDITOR=vim"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^export EDITOR=' {_p(F_OS_05)} || echo MISSING",
                "include": ["export EDITOR=vim"],
                "exclude": ["MISSING"],
            },
            instr=f"I want every shell on this account to default to vim for things like git commit and crontab editing — please make {_nl(F_OS_05)} export EDITOR=vim.",
        ),
        Param(
            gold_args={"line": "export EDITOR=emacs"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^export EDITOR=' {_p(F_OS_05)} || echo MISSING",
                "include": ["export EDITOR=emacs"],
                "exclude": ["MISSING"],
            },
            instr=f"I've been a lifelong Emacs user — please add an EDITOR export in {_nl(F_OS_05)} so that shell tools open emacs by default when they need an editor.",
        ),
    ]),

    # F-OS-06 — zshrc add alias
    # Pruned (os rebalance, file_operation OVER, task_id=add_alias_la_zsh):
    # FileTask(F_OS_06, "add_alias_la_zsh", "file_operation",
             # _gold_append_line, params=[
        # Param(
            # gold_args={"line": "alias la='ls -A'"},
            # eval_kind="config_check",
            # eval_args={
                # "command": f"grep -E \"^alias la='ls -A'\" {_p(F_OS_06)} || echo MISSING",
                # "include": ["alias la='ls -A'"],
                # "exclude": ["MISSING"],
            # },
            # instr=f"Append `alias la='ls -A'` to {_nl(F_OS_06)}.",
        # ),
        # Param(
            # gold_args={"line": "alias ..='cd ..'"},
            # eval_kind="config_check",
            # eval_args={
                # "command": f"grep -E \"^alias \\.\\.=\" {_p(F_OS_06)} || echo MISSING",
                # "include": ["alias ..='cd ..'"],
                # "exclude": ["MISSING"],
            # },
            # instr=f"Append `alias ..='cd ..'` to {_nl(F_OS_06)} so typing `..` jumps up a directory.",
        # ),
    # ]),
    FileTask(F_OS_06, "set_pager_env_zsh", "file_operation",
             _gold_append_line, params=[
        Param(
            gold_args={"line": "export PAGER=less"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^export PAGER=' {_p(F_OS_06)} || echo MISSING",
                "include": ["export PAGER=less"],
                "exclude": ["MISSING"],
            },
            instr=f"My man-pages keep opening in something weird — please make {_nl(F_OS_06)} export PAGER=less so all paging output goes through less by default.",
        ),
        Param(
            gold_args={"line": "export PAGER=most"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^export PAGER=' {_p(F_OS_06)} || echo MISSING",
                "include": ["export PAGER=most"],
                "exclude": ["MISSING"],
            },
            instr=f"I really enjoy the multi-window navigation that the most pager gives me — please update {_nl(F_OS_06)} so PAGER is exported as most for every new zsh session.",
        ),
    ]),

    # F-OS-07 — profile.sh PATH edit
    FileTask(F_OS_07, "prepend_local_bin_path", "file_operation",
             _gold_append_line, params=[
        Param(
            gold_args={"line": 'export PATH="$HOME/bin:$PATH"'},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^export PATH=\"\\$HOME/bin' {_p(F_OS_07)} || echo MISSING",
                "include": ["$HOME/bin"],
                "exclude": ["MISSING"],
            },
            instr=f"I keep dropping helper scripts in my home bin folder and they're not being picked up. Could you update {_nl(F_OS_07)} so the user's HOME/bin directory is prepended to PATH and wins over the system binaries?",
        ),
        Param(
            gold_args={"line": 'export PATH="/opt/local/bin:$PATH"'},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^export PATH=\"/opt/local/bin' {_p(F_OS_07)} || echo MISSING",
                "include": ["/opt/local/bin"],
                "exclude": ["MISSING"],
            },
            instr=f"I installed a bunch of tools under /opt/local and want them to shadow the system versions — please update {_nl(F_OS_07)} so /opt/local/bin is prepended onto PATH.",
        ),
    ]),
    FileTask(F_OS_07, "switch_editor_to_vim", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "s|^export EDITOR=nano|export EDITOR=vim|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^export EDITOR=' {_p(F_OS_07)}",
                "include": ["export EDITOR=vim"],
                "exclude": ["export EDITOR=nano"],
            },
            instr=f"I'm setting up vim as my muscle-memory editor on this box. In {_nl(F_OS_07)}, change the EDITOR export so it picks vim instead of the current nano default.",
        ),
        Param(
            # Validation bug fix (f_os_07__switch_editor_to_vim): instruction
            # previously said "neovim" (the Ubuntu package name) but agents
            # correctly typed `export EDITOR=nvim` (the actual binary name);
            # eval `include` literal required `neovim` → 0% pass. Fix (b):
            # standardize on `nvim` everywhere (gold sed, eval include, instr).
            gold_args={"sed_expr": "s|^export EDITOR=nano|export EDITOR=nvim|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^export EDITOR=' {_p(F_OS_07)}",
                "include": ["export EDITOR=nvim"],
                "exclude": ["export EDITOR=nano"],
            },
            instr=f"I've switched from nano to Neovim. In {_nl(F_OS_07)}, update the EDITOR export so it launches the nvim binary instead — keep in mind the binary name is nvim, not neovim.",
        ),
    ]),

    # F-OS-08 — .env edit
    FileTask(F_OS_08, "set_log_level_debug", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "s|^LOG_LEVEL=INFO|LOG_LEVEL=DEBUG|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^LOG_LEVEL=' {_p(F_OS_08)}",
                "include": ["LOG_LEVEL=DEBUG"],
                "exclude": ["LOG_LEVEL=INFO"],
            },
            instr=f"I'm trying to chase down a tricky bug and need verbose output — please update the LOG_LEVEL line in {_nl(F_OS_08)} so it's set to DEBUG instead of the current INFO.",
        ),
        Param(
            gold_args={"sed_expr": "s|^LOG_LEVEL=INFO|LOG_LEVEL=WARNING|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^LOG_LEVEL=' {_p(F_OS_08)}",
                "include": ["LOG_LEVEL=WARNING"],
                "exclude": ["LOG_LEVEL=INFO"],
            },
            instr=f"The logs are way too chatty in prod — bump the LOG_LEVEL line in {_nl(F_OS_08)} up to WARNING so only real problems get printed.",
        ),
    ]),
    # Pruned (os rebalance, file_operation OVER, task_id=rotate_db_url):
    # FileTask(F_OS_08, "rotate_db_url", "file_operation",
             # _gold_replace_line, params=[
        # Param(
            # gold_args={"sed_expr": "s|^DB_URL=postgres://localhost:5432/app|DB_URL=postgres://db.example.com:5432/app|"},
            # eval_kind="config_check",
            # eval_args={
                # "command": f"grep -E '^DB_URL=' {_p(F_OS_08)}",
                # "include": ["DB_URL=postgres://db.example.com:5432/app"],
                # "exclude": ["postgres://localhost:5432/app"],
            # },
            # instr=f"In {_nl(F_OS_08)}, point the database to the production host: change `DB_URL=postgres://localhost:5432/app` to `DB_URL=postgres://db.example.com:5432/app`.",
        # ),
        # Param(
            # gold_args={"sed_expr": "s|^DB_URL=postgres://localhost:5432/app|DB_URL=postgres://staging.example.com:5432/app|"},
            # eval_kind="config_check",
            # eval_args={
                # "command": f"grep -E '^DB_URL=' {_p(F_OS_08)}",
                # "include": ["DB_URL=postgres://staging.example.com:5432/app"],
                # "exclude": ["postgres://localhost:5432/app"],
            # },
            # instr=f"Help me reconfigure this — in {_nl(F_OS_08)}, point the database to the staging host: change `DB_URL=postgres://localhost:5432/app` to `DB_URL=postgres://staging.example.com:5432/app`.",
        # ),
    # ]),

    # ----- Loop 3: tar / zip archive operations -----
    # 2026-05-10 (validation change): F_OS_09 / F_OS_10 archive-create
    # FileTasks DROPPED. Eval has zero `compare_archive` rows in the os
    # domain (the archive tasks live in multi_apps), and synth's prior
    # 5 archive FileTasks × 2 params = 10 rows produced a synth-only
    # +18.9pp Q-fn over-amplification on the `compare_archive` skill class.
    # F_OS_09/F_OS_10 File definitions retained but no longer reference any
    # FileTask (they were created/extract pairs; only F_OS_11 extract
    # remains below). Archive coverage belongs in multi_apps.py.

    # F-OS-11 — extract pre-staged archive
    FileTask(F_OS_11, "extract_to_subdir", "file_operation",
             _gold_tar_extract, params=[
        Param(
            gold_args={"dest_dir": f"{_DESK}/bundle_extracted"},
            eval_kind="command_output",
            eval_args={
                "command": f"ls {_DESK}/bundle_extracted | sort",
                "include": ["alpha.txt", "beta.txt", "gamma.txt"],
            },
            instr=f"I need the contents of the gzipped tarball at {_nl(F_OS_11)} unpacked into a brand-new directory at ~/Desktop/bundle_extracted, then leave it there for me.",
        ),
        Param(
            gold_args={"dest_dir": f"{_DESK}/payload"},
            eval_kind="command_output",
            eval_args={
                "command": f"ls {_DESK}/payload | sort",
                "include": ["alpha.txt", "beta.txt", "gamma.txt"],
            },
            instr=f"Please unpack the tar.gz at {_nl(F_OS_11)} into a fresh folder called payload on the Desktop — the destination is ~/Desktop/payload.",
        ),
    ]),
    FileTask(F_OS_11, "extract_then_count", "file_operation",
             _gold_extract_then_count, params=[
        Param(
            gold_args={"dest_dir": f"{_DESK}/bundle_out", "count_path": f"{_DESK}/bundle_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": f"cat {_DESK}/bundle_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr=f"Please extract the tarball at {_nl(F_OS_11)} into a new directory ~/Desktop/bundle_out, and then count how many files ended up inside and save that number to ~/Desktop/bundle_count.txt.",
        ),
        Param(
            gold_args={"dest_dir": f"{_DESK}/payload2", "count_path": f"{_DESK}/payload2_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": f"cat {_DESK}/payload2_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr=f"Unpack the archive at {_nl(F_OS_11)} into a fresh directory ~/Desktop/payload2 and afterwards write the total number of extracted entries to ~/Desktop/payload2_count.txt for me.",
        ),
    ]),

    # F-OS-12 — list-jpgs probe
    # 2026-05-10 (validation change): F_OS_12 zip_create_photos FileTask
    # DROPPED (compare_archive over-amp; see F_OS_09/F_OS_10 note above).
    FileTask(F_OS_12, "list_jpgs_to_file", "file_operation",
             _gold_ls_jpgs_to_file, params=[
        Param(
            gold_args={"dst_path": f"{_DESK}/jpg_list.txt"},
            eval_kind="command_output",
            eval_args={
                "command": f"cat {_DESK}/jpg_list.txt 2>/dev/null || echo MISSING",
                "include": ["beach-sunset.jpg", "desert-dunes.jpg", "forest-trail.jpg"],
                "exclude": ["MISSING"],
            },
            instr=f"I want an alphabetized index of every JPEG sitting in {_nl(F_OS_12)} — just the basenames, one per line, sorted, saved to ~/Desktop/jpg_list.txt.",
        ),
        Param(
            gold_args={"dst_path": f"{_DESK}/photos_index.txt"},
            eval_kind="command_output",
            eval_args={
                "command": f"cat {_DESK}/photos_index.txt 2>/dev/null || echo MISSING",
                "include": ["beach-sunset.jpg", "desert-dunes.jpg", "forest-trail.jpg"],
                "exclude": ["MISSING"],
            },
            instr=f"Could you produce a sorted listing of all .jpg files inside {_nl(F_OS_12)} (file names only, alphabetical order) and write the result into ~/Desktop/photos_index.txt?",
        ),
    ]),

    # ----- Loop 4: cron / systemd config mutations -----

    # F-OS-13 — crontab v2
    FileTask(F_OS_13, "append_journal_job", "file_operation",
             _gold_append_line, params=[
        Param(
            gold_args={"line": "45 23 * * * /usr/local/bin/journal-flush.sh"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_13)}",
                "include": ["45 23 * * * /usr/local/bin/journal-flush.sh"],
            },
            instr=f"I'd like the journal-flush helper script to run once a day right before midnight — please add a cron entry in {_nl(F_OS_13)} that fires /usr/local/bin/journal-flush.sh at 11:45 PM every night.",
        ),
        Param(
            gold_args={"line": "0 12 * * * /usr/local/bin/midday-report.sh"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_13)}",
                "include": ["0 12 * * * /usr/local/bin/midday-report.sh"],
            },
            instr=f"Please add a new entry to {_nl(F_OS_13)} so that /usr/local/bin/midday-report.sh runs every day at exactly noon.",
        ),
    ]),
    # Validation bug fix (f_os_13__reschedule_db_snapshot, cluster H):
    # `needs_terminal=True` pre-opens gnome-terminal so `sed -i ...` agents
    # don't drop keystrokes to a desktop focus race.
    FileTask(F_OS_13, "reschedule_db_snapshot", "file_operation",
             _gold_replace_line, needs_terminal=True, params=[
        Param(
            gold_args={"sed_expr": "s|^15 1 \\* \\* \\* /usr/local/bin/db-snapshot.sh|0 2 * * * /usr/local/bin/db-snapshot.sh|"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_13)}",
                "include": ["0 2 * * * /usr/local/bin/db-snapshot.sh"],
                "exclude": ["15 1 * * * /usr/local/bin/db-snapshot.sh"],
            },
            instr=f"In {_nl(F_OS_13)}, move the existing db-snapshot job from 1:15 AM over to 2 AM sharp — the upstream replica sync is now slated for 1 AM and shouldn't collide with the snapshot.",
        ),
        Param(
            gold_args={"sed_expr": "s|^15 1 \\* \\* \\* /usr/local/bin/db-snapshot.sh|45 0 * * * /usr/local/bin/db-snapshot.sh|"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_13)}",
                "include": ["45 0 * * * /usr/local/bin/db-snapshot.sh"],
                "exclude": ["15 1 * * * /usr/local/bin/db-snapshot.sh"],
            },
            instr=f"In {_nl(F_OS_13)}, shift the db-snapshot job earlier — have it start at 12:45 AM rather than the current 1:15 AM so it finishes before the network maintenance window opens.",
        ),
    ]),

    # F-OS-14 — systemd timer
    FileTask(F_OS_14, "change_timer_interval", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "s|^OnUnitActiveSec=30min|OnUnitActiveSec=1h|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^OnUnitActiveSec=' {_p(F_OS_14)}",
                "include": ["OnUnitActiveSec=1h"],
                "exclude": ["OnUnitActiveSec=30min"],
            },
            instr=f"In the systemd timer unit at {_nl(F_OS_14)}, slow the firing cadence down — instead of every thirty minutes it should now retrigger once an hour.",
        ),
        Param(
            gold_args={"sed_expr": "s|^OnUnitActiveSec=30min|OnUnitActiveSec=10min|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^OnUnitActiveSec=' {_p(F_OS_14)}",
                "include": ["OnUnitActiveSec=10min"],
                "exclude": ["OnUnitActiveSec=30min"],
            },
            instr=f"In the systemd timer at {_nl(F_OS_14)}, make it run more aggressively — switch the active-interval from thirty minutes down to ten minutes between firings.",
        ),
    ]),
    # Pruned (os rebalance, file_operation OVER, task_id=change_boot_delay):
    # FileTask(F_OS_14, "change_boot_delay", "file_operation",
             # _gold_replace_line, params=[
        # Param(
            # gold_args={"sed_expr": "s|^OnBootSec=5min|OnBootSec=15min|"},
            # eval_kind="config_check",
            # eval_args={
                # "command": f"grep -E '^OnBootSec=' {_p(F_OS_14)}",
                # "include": ["OnBootSec=15min"],
                # "exclude": ["OnBootSec=5min"],
            # },
            # instr=f"I'm switching the service over — in {_nl(F_OS_14)}, change the post-boot delay from 5min to 15min (`OnBootSec=15min`).",
        # ),
        # Param(
            # gold_args={"sed_expr": "s|^OnBootSec=5min|OnBootSec=2min|"},
            # eval_kind="config_check",
            # eval_args={
                # "command": f"grep -E '^OnBootSec=' {_p(F_OS_14)}",
                # "include": ["OnBootSec=2min"],
                # "exclude": ["OnBootSec=5min"],
            # },
            # instr=f"In {_nl(F_OS_14)}, change the post-boot delay from 5min to 2min (`OnBootSec=2min`).",
        # ),
    # ]),

    # F-OS-15 — second systemd unit (different File ID, same shape)
    FileTask(F_OS_15, "set_restart_no", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "s|^Restart=on-failure|Restart=no|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^Restart=' {_p(F_OS_15)}",
                "include": ["Restart=no"],
                "exclude": ["Restart=on-failure"],
            },
            instr=f"Could you fix this up for me — in the systemd unit at {_nl(F_OS_15)}, disable the auto-restart behaviour entirely so a crashed service stays down until we look at it.",
        ),
        Param(
            gold_args={"sed_expr": "s|^Restart=on-failure|Restart=on-success|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^Restart=' {_p(F_OS_15)}",
                "include": ["Restart=on-success"],
                "exclude": ["Restart=on-failure"],
            },
            instr=f"In the systemd unit at {_nl(F_OS_15)}, flip the restart policy so the service only relaunches after a clean exit — failures should leave it stopped for inspection.",
        ),
    ]),
    # Pruned (os rebalance, file_operation OVER, task_id=change_exec_start_path):
    # FileTask(F_OS_15, "change_exec_start_path", "file_operation",
             # _gold_replace_line, params=[
        # Param(
            # gold_args={"sed_expr": "s|^ExecStart=/usr/local/bin/worker --foreground|ExecStart=/opt/worker/run.sh|"},
            # eval_kind="config_check",
            # eval_args={
                # "command": f"grep -E '^ExecStart=' {_p(F_OS_15)}",
                # "include": ["ExecStart=/opt/worker/run.sh"],
                # "exclude": ["ExecStart=/usr/local/bin/worker"],
            # },
            # instr=f"In {_nl(F_OS_15)}, change the `ExecStart=` directive to point at `/opt/worker/run.sh` instead of `/usr/local/bin/worker --foreground`.",
        # ),
        # Param(
            # gold_args={"sed_expr": "s|^ExecStart=/usr/local/bin/worker --foreground|ExecStart=/srv/app/bin/server --daemon|"},
            # eval_kind="config_check",
            # eval_args={
                # "command": f"grep -E '^ExecStart=' {_p(F_OS_15)}",
                # "include": ["ExecStart=/srv/app/bin/server --daemon"],
                # "exclude": ["ExecStart=/usr/local/bin/worker"],
            # },
            # instr=f"I'd like to harden this server — in {_nl(F_OS_15)}, change the `ExecStart=` directive to `/srv/app/bin/server --daemon` instead of `/usr/local/bin/worker --foreground`.",
        # ),
    # ]),

    # F-OS-16 — second nginx site
    # Pruned (os rebalance, file_operation OVER, task_id=edit_root_path):
    # FileTask(F_OS_16, "edit_root_path", "file_operation",
             # _gold_replace_line, params=[
        # Param(
            # gold_args={"sed_expr": "s|root /var/www/example;|root /srv/www/app;|"},
            # eval_kind="config_check",
            # eval_args={
                # "command": f"grep -E '^\\s*root' {_p(F_OS_16)}",
                # "include": ["root /srv/www/app;"],
                # "exclude": ["root /var/www/example;"],
            # },
            # instr=f"I'm staging a quick rollout — in the nginx site config {_nl(F_OS_16)}, change the `root` directive from `/var/www/example` to `/srv/www/app`.",
        # ),
        # Param(
            # gold_args={"sed_expr": "s|root /var/www/example;|root /opt/sites/public_html;|"},
            # eval_kind="config_check",
            # eval_args={
                # "command": f"grep -E '^\\s*root' {_p(F_OS_16)}",
                # "include": ["root /opt/sites/public_html;"],
                # "exclude": ["root /var/www/example;"],
            # },
            # instr=f"I'm staging a quick rollout — in the nginx site config {_nl(F_OS_16)}, change the `root` directive from `/var/www/example` to `/opt/sites/public_html`.",
        # ),
    # ]),
    FileTask(F_OS_16, "set_index_php", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "s|index index.html;|index index.php index.html;|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^\\s*index' {_p(F_OS_16)}",
                "include": ["index index.php index.html;"],
                "exclude": ["index index.html;\n"],
            },
            instr=f"In {_nl(F_OS_16)}, update the nginx index directive so that index.php is tried first and only then fall back to index.html — PHP should be the preferred default landing page.",
        ),
        Param(
            gold_args={"sed_expr": "s|index index.html;|index index.htm default.html;|"},
            eval_kind="config_check",
            eval_args={
                "command": f"grep -E '^\\s*index' {_p(F_OS_16)}",
                "include": ["index index.htm default.html;"],
                "exclude": ["index index.html;\n"],
            },
            instr=f"In {_nl(F_OS_16)}, change the nginx index directive so the server now searches for index.htm first and falls back to default.html — the legacy site doesn't use index.html anymore.",
        ),
    ]),

    # ----- Loop 5: file ops + ssh + gap-fillers -----

    # F-OS-17 — mixed filetree (chmod + grep ops)
    FileTask(F_OS_17, "chmod_644_recursive", "file_operation",
             _gold_chmod_recursive, params=[
        Param(
            gold_args={"perm": "644"},
            eval_kind="command_output",
            eval_args={
                "command": _make_perm_eval_command("644", _p(F_OS_17)),
                "include": ["All files have the correct permissions."],
                "exclude": ["Some files do not"],
            },
            instr=f"My filetree under {_nl(F_OS_17)} has inconsistent permissions. Please walk through it recursively and bring every regular file to mode 644 (owner read/write, others read-only).",
        ),
        Param(
            gold_args={"perm": "600"},
            eval_kind="command_output",
            eval_args={
                "command": _make_perm_eval_command("600", _p(F_OS_17)),
                "include": ["All files have the correct permissions."],
                "exclude": ["Some files do not"],
            },
            instr=f"For security I'd like every regular file under {_nl(F_OS_17)} locked to owner-only access — mode 600 across the whole subtree, recursively.",
        ),
    ]),
    FileTask(F_OS_17, "make_subdir_archive", "file_operation",
             _gold_make_subdir, params=[
        Param(
            gold_args={"sub": "archive"},
            eval_kind="command_output",
            eval_args={
                "command": f"ls {_p(F_OS_17)}",
                "include": ["archive"],
            },
            instr=f"I'd like an archive sub-folder inside {_nl(F_OS_17)} where I can stash the old files later. Please create the directory named archive there for me.",
        ),
        Param(
            gold_args={"sub": "old"},
            eval_kind="command_output",
            eval_args={
                "command": f"ls {_p(F_OS_17)}",
                "include": ["old"],
            },
            instr=f"Please make a folder called old inside {_nl(F_OS_17)} — I want a clearly-named bucket inside the working directory for files I'm about to retire.",
        ),
    ]),

    # F-OS-18 — rename directory
    FileTask(F_OS_18, "rename_dir_jan_to_feb", "file_operation",
             _gold_rename_dir, params=[
        Param(
            gold_args={"new_basename": "todo_list_Feb_1"},
            eval_kind="command_output",
            eval_args={
                "command": f"ls {_DESK}",
                "include": ["todo_list_Feb_1"],
                "exclude": ["todo_list_Jan_1\n", "todo_list_Jan_1$"],
            },
            instr=f"It's a new month — please rename the folder at {_nl(F_OS_18)} so it becomes ~/Desktop/todo_list_Feb_1 instead of carrying the January label.",
        ),
        Param(
            gold_args={"new_basename": "todo_list_Q1"},
            eval_kind="command_output",
            eval_args={
                "command": f"ls {_DESK}",
                "include": ["todo_list_Q1"],
                "exclude": ["todo_list_Jan_1\n", "todo_list_Jan_1$"],
            },
            instr=f"I'm reorganizing by quarter instead of by month — could you rename {_nl(F_OS_18)} to ~/Desktop/todo_list_Q1 to reflect the new scheme?",
        ),
    ]),
    FileTask(F_OS_18, "make_dated_subdir", "file_operation",
             _gold_make_subdir, params=[
        Param(
            gold_args={"sub": "2026-Q1"},
            eval_kind="command_output",
            eval_args={
                "command": f"ls {_p(F_OS_18)}",
                "include": ["2026-Q1"],
            },
            instr=f"Inside {_nl(F_OS_18)} I'd like a dated sub-folder for the first quarter of this year — please add a sub-directory named 2026-Q1 in there.",
        ),
        Param(
            gold_args={"sub": "archive_2025"},
            eval_kind="command_output",
            eval_args={
                "command": f"ls {_p(F_OS_18)}",
                "include": ["archive_2025"],
            },
            instr=f"I want to tuck last year's notes into a dedicated archive folder. Please create a sub-directory called archive_2025 inside {_nl(F_OS_18)} so I can move items there.",
        ),
    ]),

    # F-OS-19 — authorized_keys append
    FileTask(F_OS_19, "append_new_key", "file_operation",
             _gold_append_line, params=[
        Param(
            gold_args={"line": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBobnewkey bob@workstation"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_19)}",
                "include": ["bob@workstation", "ssh-ed25519"],
            },
            instr=f"My colleague Bob set up a new workstation today. Please add his ed25519 public key (the line ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBobnewkey bob@workstation) to {_nl(F_OS_19)} so he can SSH in.",
        ),
        Param(
            gold_args={"line": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDcarol carol@server"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_19)}",
                "include": ["carol@server", "ssh-rsa"],
            },
            instr=f"Carol needs SSH access from her server account. Please append her RSA public key (ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDcarol carol@server) as a new line in {_nl(F_OS_19)}.",
        ),
    ]),
    # PD 1b: task_id `edit_alice_key` covers both removal AND rename of
    # alice's key entry — was `remove_existing_key` (param2 is a rename, not
    # a delete; renamed during audit so the umbrella skill is honest).
    FileTask(F_OS_19, "edit_alice_key", "file_operation",
             _gold_replace_line, params=[
        Param(
            gold_args={"sed_expr": "/alice@laptop/d"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_19)}",
                "include": ["authorized_keys"],
                "exclude": ["alice@laptop"],
            },
            instr=f"Alice left the team last week and we need to revoke her SSH access. Please remove the entire line from {_nl(F_OS_19)} that mentions alice@laptop so her key no longer authorizes logins.",
        ),
        Param(
            gold_args={"sed_expr": "s|alice@laptop|alice@workstation|"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_19)}",
                "include": ["alice@workstation", "ssh-rsa"],
                "exclude": ["alice@laptop"],
            },
            instr=f"Alice swapped her laptop for a dedicated workstation. In {_nl(F_OS_19)}, update the comment on her authorized key so it reads alice@workstation instead of the outdated alice@laptop.",
        ),
    ]),

    # F-OS-20 — empty .gitconfig (set user.name + user.email)
    # Validation bug fix (f_os_20__set_user_name_email, cluster H):
    # `needs_terminal=True` makes `_to_synth_template` pre-open gnome-terminal
    # so the agent doesn't lose the `git config ...` keystrokes to a desktop
    # focus race after clicking the dock.
    FileTask(F_OS_20, "set_user_name_email", "file_operation",
             _gold_append_line, needs_terminal=True, params=[
        Param(
            gold_args={"line": "[user]\n\tname = Alice Example\n\temail = alice@example.com"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_20)}",
                "include": ["alice@example.com", "Alice Example"],
            },
            instr=f"My git config at {_nl(F_OS_20)} is empty and commits are showing up as anonymous. Please set the git author identity in that file to the name Alice Example with email alice@example.com.",
        ),
        Param(
            gold_args={"line": "[user]\n\tname = Bob Example\n\temail = bob@example.com"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_20)}",
                "include": ["bob@example.com", "Bob Example"],
            },
            instr=f"Please configure git's user identity inside {_nl(F_OS_20)} so my commits are attributed to Bob Example with email bob@example.com — right now the file has no [user] block at all.",
        ),
    ]),
    # PD 1b: task_id `add_section_block` covers both `[color] ui=auto` AND
    # `[core] editor=vim` — was `enable_color_ui` (param2 is editor, not
    # color; renamed during audit so the umbrella skill is honest).
    FileTask(F_OS_20, "add_section_block", "file_operation",
             _gold_append_line, params=[
        Param(
            gold_args={"line": "[color]\n\tui = auto"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_20)}",
                "include": ["[color]", "ui = auto"],
            },
            instr=f"My git output is completely monochrome and hard to read. Please update {_nl(F_OS_20)} so git's color UI is set to auto — I want diffs and status to show colors when running in a terminal.",
        ),
        Param(
            gold_args={"line": "[core]\n\teditor = vim"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_20)}",
                "include": ["[core]", "editor = vim"],
            },
            instr=f"When I run git commit it keeps falling back to nano. Please tweak {_nl(F_OS_20)} so git's core editor is set to vim and my next commit message opens in the right editor.",
        ),
    ]),

    # ----- Loop 6: taxonomy gap-fillers (timezone / system_query /
    #               dual_operation / app_management) -----

    # F-OS-21 — system timezone (native `is_utc_0` eval).
    # 2026-05-10 (validation change): rewritten to set the REAL system
    # timezone (/etc/timezone + /etc/localtime symlink) and use the native
    # `is_utc_0` evaluator. Was previously a marker-file proxy that routed
    # through `_text_content`. Pre-config installs a `timedatectl` shim so
    # `timedatectl status` works on no-systemd containers (mirrors
    # perturb/os.py _PARAPHRASE_PRE_CONFIG_STEPS for b6781586). Only the
    # UTC variant survives — `is_utc_0` only validates UTC.
    # Validation bug fix (f_os_21__switch_to_utc, cluster H): pre-open
    # gnome-terminal — the instruction tells the agent to "Open a terminal
    # and ..." which used to trip Trigger H (dock click + type_text in same
    # turn).
    FileTask(F_OS_21, "switch_to_utc", "is_utc_0",
             _gold_set_timezone_utc, needs_terminal=True, params=[
        Param(
            gold_args={},
            eval_kind="is_utc_0",
            eval_args={},
            instr="I want my current time zone set to UTC+0. Could you switch this machine over to UTC so timestamps and logs are no longer skewed?",
        ),
        Param(
            gold_args={},
            eval_kind="is_utc_0",
            eval_args={},
            instr="My computer is currently on Pacific time but our team's standard is UTC. Please change the system clock so the operating system reports UTC as its time zone.",
        ),
    ]),

    # F-OS-22 — GNOME favorite-apps gsetting (native check_gnome_favorite_apps
    # eval). 2026-05-10 (validation change): rewritten to use real `gsettings
    # set` instead of marker-file proxy that routed through `_text_content`.
    # Initial state (set in src): ['firefox.desktop',
    # 'org.gnome.Nautilus.desktop', 'vim.desktop']. Eval `expected.rules.
    # expected` is an ORDERED list checked by check_gnome_favorite_apps.
    FileTask(F_OS_22, "remove_vim_favorite", "check_gnome_favorite_apps",
             _gold_gsettings_favorites, params=[
        Param(
            gold_args={"value": "['firefox.desktop', 'org.gnome.Nautilus.desktop']"},
            eval_kind="check_gnome_favorite_apps",
            eval_args={"expected": ["firefox.desktop", "org.gnome.Nautilus.desktop"]},
            instr="Can you remove vim from my GNOME favorites in the dash so only Firefox and the Files app remain pinned there?",
        ),
        Param(
            gold_args={"value": "['org.gnome.Nautilus.desktop', 'vim.desktop']"},
            eval_kind="check_gnome_favorite_apps",
            eval_args={"expected": ["org.gnome.Nautilus.desktop", "vim.desktop"]},
            instr="I'm switching browsers and don't want Firefox in my GNOME dash anymore. Please unpin Firefox from the favorites so only the Files app and vim remain, in that order.",
        ),
    ]),
    FileTask(F_OS_22, "add_terminal_favorite", "check_gnome_favorite_apps",
             _gold_gsettings_favorites, params=[
        Param(
            gold_args={"value": "['firefox.desktop', 'org.gnome.Nautilus.desktop', 'vim.desktop', 'org.gnome.Terminal.desktop']"},
            eval_kind="check_gnome_favorite_apps",
            eval_args={"expected": ["firefox.desktop", "org.gnome.Nautilus.desktop", "vim.desktop", "org.gnome.Terminal.desktop"]},
            instr="I'm setting up a fresh VM and I want quick access to the terminal from the dash. Please add the GNOME Terminal launcher as a new favorite at the very end of my existing favorites list — don't reorder the others.",
        ),
        Param(
            gold_args={"value": "['firefox.desktop', 'org.gnome.Nautilus.desktop', 'vim.desktop', 'code.desktop']"},
            eval_kind="check_gnome_favorite_apps",
            eval_args={"expected": ["firefox.desktop", "org.gnome.Nautilus.desktop", "vim.desktop", "code.desktop"]},
            instr="VS Code has become my daily driver and I'd like it pinned in the GNOME dash. Please append the Visual Studio Code launcher as the LAST entry in my favorites — keep every existing pin in its current order.",
        ),
    ]),

    # F-OS-23 — disk query dir (system_query bucket; agent runs `du`/`find`
    # and writes the answer to a file the eval reads)
    FileTask(F_OS_23, "count_files_in_tree", "system_query",
             _gold_count_files, params=[
        Param(
            gold_args={"dst_path": f"{_DESK}/file_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": f"cat {_DESK}/file_count.txt 2>/dev/null || echo MISSING",
                "include": ["7"],
                "exclude": ["MISSING"],
            },
            instr=f"I need a quick tally of how many regular files live under {_nl(F_OS_23)} recursively. Please count them and save just the number to ~/Desktop/file_count.txt.",
        ),
        Param(
            gold_args={"dst_path": f"{_DESK}/n_files.txt"},
            eval_kind="command_output",
            eval_args={
                "command": f"cat {_DESK}/n_files.txt 2>/dev/null || echo MISSING",
                "include": ["7"],
                "exclude": ["MISSING"],
            },
            instr=f"Recursively count every regular file underneath {_nl(F_OS_23)}, then write only that integer into the file ~/Desktop/n_files.txt for my reporting script to read.",
        ),
    ]),
    FileTask(F_OS_23, "report_disk_usage", "system_query",
             _gold_du_to_file, params=[
        Param(
            gold_args={"dst_path": f"{_DESK}/du_report.txt"},
            eval_kind="command_output",
            eval_args={
                # The exact human-readable size depends on filesystem block
                # rounding (4K vs 8K vs 16K), so we only check the unit
                # suffix `K` is present. PD F4 (concrete oracle) holds:
                # the agent runs the EXACT command from the instr.
                "command": f"cat {_DESK}/du_report.txt 2>/dev/null || echo MISSING",
                "include": ["K"],
                "exclude": ["MISSING"],
            },
            instr=f"How big is {_nl(F_OS_23)}? Measure its size in human-readable units and save just the size to ~/Desktop/du_report.txt.",
        ),
        Param(
            gold_args={"dst_path": f"{_DESK}/disk_usage.txt"},
            eval_kind="command_output",
            eval_args={
                "command": f"cat {_DESK}/disk_usage.txt 2>/dev/null || echo MISSING",
                "include": ["K"],
                "exclude": ["MISSING"],
            },
            instr=f"Compute the disk usage of {_nl(F_OS_23)} in K/M/G suffix form and store the size in ~/Desktop/disk_usage.txt.",
        ),
    ]),

    # F-OS-24 — dual_operation (compound: chmod + count, copy + count)
    FileTask(F_OS_24, "chmod_and_count_executable", "dual_operation",
             _gold_chmod_then_count, params=[
        Param(
            gold_args={"perm": "755", "dst_path": f"{_DESK}/exec_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": f"cat {_DESK}/exec_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr=f"Make every regular file in {_nl(F_OS_24)} executable (mode 755), then save the count to ~/Desktop/exec_count.txt.",
        ),
        Param(
            gold_args={"perm": "700", "dst_path": f"{_DESK}/owner_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": f"cat {_DESK}/owner_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr=f"For my Friday housekeeping I'd like every regular file under {_nl(F_OS_24)} locked to owner-only execute permissions, and then a verification step. First, recursively chmod all regular files to mode 700. Then count how many ended up at mode 700 and save only that integer into ~/Desktop/owner_count.txt.",
        ),
    ]),
    FileTask(F_OS_24, "copy_and_count", "dual_operation",
             _gold_copy_then_count, params=[
        Param(
            gold_args={"dst_dir": f"{_DESK}/scripts_copy",
                       "count_path": f"{_DESK}/copy_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": f"cat {_DESK}/copy_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr=f"Copy every file from {_nl(F_OS_24)} into a new ~/Desktop/scripts_copy folder, then write the entry count to ~/Desktop/copy_count.txt.",
        ),
        Param(
            gold_args={"dst_dir": f"{_DESK}/scripts_backup",
                       "count_path": f"{_DESK}/backup_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": f"cat {_DESK}/backup_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr=f"I'd like a backup snapshot of the scripts in {_nl(F_OS_24)}, followed by a sanity check. First copy each file from there into a fresh directory called ~/Desktop/scripts_backup. Then count how many entries the backup directory holds and save only that count to ~/Desktop/backup_count.txt.",
        ),
    ]),

    # F-OS-25 — /etc/hosts-style edit (file_operation, append host alias)
    FileTask(F_OS_25, "add_dev_host", "file_operation",
             _gold_append_line, params=[
        Param(
            gold_args={"line": "192.168.1.10   dev.local"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_25)}",
                "include": ["192.168.1.10", "dev.local"],
            },
            instr=f"I want the hostname dev.local to resolve to my LAN dev box. Please add an entry in {_nl(F_OS_25)} mapping the IP 192.168.1.10 to the name dev.local.",
        ),
        Param(
            gold_args={"line": "10.0.0.5   staging.internal"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_25)}",
                "include": ["10.0.0.5", "staging.internal"],
            },
            instr=f"Add a hosts entry to {_nl(F_OS_25)} so that the name staging.internal resolves to the internal IP 10.0.0.5 — that way I can hit the staging cluster by name without DNS.",
        ),
    ]),
    # Validation bug fix (f_os_25__swap_localhost_alias, cluster H): pre-open
    # gnome-terminal so `sed -i ...` keystrokes don't drop in a focus race.
    # Validation fix: original include used literal 3-space separator
    # "127.0.1.1   workstation" but agents type either \t or single-space
    # depending on tool (sed -i, terminal type, GUI edit). Split into two
    # substring requirements ("127.0.1.1" AND "workstation") — both must
    # appear anywhere in the file, exclude blocks the old-name combo.
    FileTask(F_OS_25, "swap_localhost_alias", "file_operation",
             _gold_replace_line, needs_terminal=True, params=[
        Param(
            gold_args={"sed_expr": "s|^127\\.0\\.1\\.1   ubuntu|127.0.1.1   workstation|"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_25)}",
                "include": ["127.0.1.1", "workstation"],
                "exclude": ["127.0.1.1   ubuntu", "127.0.1.1\tubuntu", "127.0.1.1 ubuntu"],
            },
            instr=f"I renamed this machine to workstation but {_nl(F_OS_25)} still calls it ubuntu on the 127.0.1.1 line. Please update that loopback alias so the hostname recorded there is workstation instead.",
        ),
        Param(
            gold_args={"sed_expr": "s|^127\\.0\\.1\\.1   ubuntu|127.0.1.1   devbox|"},
            eval_kind="config_check",
            eval_args={
                "command": f"cat {_p(F_OS_25)}",
                "include": ["127.0.1.1", "devbox"],
                "exclude": ["127.0.1.1   ubuntu", "127.0.1.1\tubuntu", "127.0.1.1 ubuntu"],
            },
            instr=f"This box is no longer just 'ubuntu' — I'm rebranding it to devbox. Please edit {_nl(F_OS_25)} so the 127.0.1.1 loopback line now maps to devbox instead of ubuntu.",
        ),
    ]),

    # F-OS-26 — recursive .jpg copy (native check_moved_jpgs eval).
    # 2026-05-10 (validation change): added to close the eval-only
    # check_moved_jpgs gap (osworld_os_23393935). Source builder seeds
    # ~/Desktop/photos_tree/{vacation/thailand,vacation/hk,family,events}/
    # with .jpg leaves; agent must `find ... -iname '*.jpg' -exec cp` into
    # a flat dest dir.
    FileTask(F_OS_26, "copy_jpgs_to_flat_dir", "check_moved_jpgs",
             _gold_copy_jpgs_recursive, params=[
        Param(
            gold_args={"dst_dir": f"{_DESK}/cpjpg"},
            eval_kind="check_moved_jpgs",
            eval_args={
                "dest_path": f"{_DESK}/cpjpg",
                "expected": [
                    "emnlp2023.jpg",
                    "hk_group.jpg",
                    "hk_skyline.jpg",
                    "monk_temple.jpg",
                    "us_family.jpg",
                ],
            },
            instr=f"Recursively go through the sub-folders of {_nl(F_OS_26)} and copy any .jpg files into another directory named 'cpjpg' on the Desktop.",
        ),
    ]),

    # ----- Loop 7: exact_match eval-class fillers (mirror 5 eval tasks) -----

    # F-OS-27 — pulseaudio max volume (mirrors `osworld_os_28cc3b7e`).
    # Src seeds volume=50%; gold sets it to target%. Eval queries pactl and
    # exact-matches "<percent>\n".
    FileTask(F_OS_27, "set_max_volume", "exact_match",
             _gold_set_volume, params=[
        Param(
            gold_args={"percent": 100},
            eval_kind="exact_match",
            eval_args={
                "command": (
                    f"{_PULSE_BOOTSTRAP}; "
                    "pactl list sinks | grep -m1 'Volume:' | awk '{print $5}' | tr -d '%'"
                ),
                "expected": "100\n",
            },
            instr="The volume on my system is way too small and I can barely hear the audio. Could you crank it up to the maximum for me?",
        ),
        Param(
            gold_args={"percent": 80},
            eval_kind="exact_match",
            eval_args={
                "command": (
                    f"{_PULSE_BOOTSTRAP}; "
                    "pactl list sinks | grep -m1 'Volume:' | awk '{print $5}' | tr -d '%'"
                ),
                "expected": "80\n",
            },
            instr="My speakers are uncomfortably loud at full output but the meeting is in five minutes. Could you bring the system master volume down to 80 percent for me?",
        ),
    ]),

    # F-OS-28 — recover deleted desktop file (mirrors `osworld_os_5ea617a3`).
    # Src ensures the target file is absent; gold `touch`es it. Eval checks
    # the file exists with `[ -f ... ] && echo "File exists."` exact_match.
    # Validation bug fix (f_os_28__recover_poster_file, cluster H):
    # `needs_terminal=True` — see note above on f_os_20.
    FileTask(F_OS_28, "recover_poster_file", "exact_match",
             _gold_create_file, needs_terminal=True, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": f"[ -f {_p(F_OS_28)} ] && echo 'File exists.' || echo 'File does not exist.'",
                "expected": "File exists.\n",
            },
            instr=f"I accidentally deleted my party-night poster — please recreate {_nl(F_OS_28)} as an empty file so the icon is back.",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": f"[ -f {_p(F_OS_28)} ] && echo 'File exists.' || echo 'File does not exist.'",
                "expected": "File exists.\n",
            },
            instr=f"My party-night poster has gone missing — please recreate {_nl(F_OS_28)} as an empty file so it exists again.",
        ),
    ]),

    # F-OS-29 — gsettings screensaver auto-lock (mirrors `osworld_os_a4d98375`).
    # Src seeds `lock-enabled = false`; gold sets it to `true`. Eval is
    # `gsettings get ...` exact-matching "true\n".
    FileTask(F_OS_29, "enable_screensaver_lock", "exact_match",
             _gold_gsettings_lock_true, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.screensaver lock-enabled",
                "expected": "true\n",
            },
            instr="I keep walking away from my desk and forgetting to lock the screen. Please turn on the GNOME auto-lock so the screen locks by itself when I leave for a while.",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.screensaver lock-enabled",
                "expected": "true\n",
            },
            instr="I want my computer to lock itself automatically once I step away. Could you enable the screensaver auto-lock feature for me?",
        ),
    ]),

    # F-OS-30 — directory rename Jan_1 → Jan_2 (mirrors `osworld_os_e0df059f`).
    # Src creates `~/Desktop/todo_list_Jan_1/`; gold renames it. Eval is
    # `[ -d ~/Desktop/todo_list_Jan_2 ]` exact-matching "Directory exists.\n".
    FileTask(F_OS_30, "rename_dir_jan1_to_jan2", "exact_match",
             _gold_rename_dir_to, params=[
        Param(
            gold_args={"new_basename": "todo_list_Jan_2"},
            eval_kind="exact_match",
            eval_args={
                "command": "[ -d ~/Desktop/todo_list_Jan_2 ] && echo 'Directory exists.' || echo 'Directory does not exist.'",
                "expected": "Directory exists.\n",
            },
            instr=f"Please rename {_nl(F_OS_30)} to todo_list_Jan_2.",
        ),
        Param(
            gold_args={"new_basename": "todo_list_Jan_2"},
            eval_kind="exact_match",
            eval_args={
                "command": "[ -d ~/Desktop/todo_list_Jan_2 ] && echo 'Directory exists.' || echo 'Directory does not exist.'",
                "expected": "Directory exists.\n",
            },
            instr=f"My todo list folder is misdated — please rename {_nl(F_OS_30)} to todo_list_Jan_2 so the date matches what's inside.",
        ),
    ]),

    # F-OS-31 — Do-not-disturb via gsettings (mirrors `osworld_os_f9be0997`).
    # Src seeds `show-banners = true` (DnD off); gold sets it to `false`
    # (banners suppressed = DnD on). Eval matches `false\n`.
    # DnD state lives in the GNOME org.gnome.desktop.notifications schema.
    FileTask(F_OS_31, "enable_dnd", "exact_match",
             _gold_gsettings_dnd_on, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.notifications show-banners",
                "expected": "false\n",
            },
            instr="I'm working on a Ubuntu desktop and don't want notification pop-ups distracting me. Could you switch on Do Not Disturb mode so the system quiets down?",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.notifications show-banners",
                "expected": "false\n",
            },
            instr="I'm about to start a long focus session and want zero interruptions. Please turn on GNOME's Do-Not-Disturb mode so notification banners stop popping up over my work.",
        ),
    ]),

    # validation RESCALER — system-settings fill (mirrors eval rows for
    # dim-screen, night-light, touchpad tap-to-click, default browser,
    # power-button action, keyboard repeat — closing the exact_match UNDER
    # gap).

    # F-OS-32 — dim screen when inactive (mirrors eval row 16 "set 'Dim
    # screen when inactive' to off").
    FileTask(F_OS_32, "dim_screen_off", "exact_match",
             _gold_gsettings_idle_dim_false, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "export DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus' && gsettings get org.gnome.settings-daemon.plugins.power idle-dim",
                "expected": "false\n",
            },
            instr="My screen keeps dimming whenever I step away for a coffee, even though I'm only gone a couple of minutes. Please turn off the Dim screen when inactive option so the brightness stays steady.",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "export DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus' && gsettings get org.gnome.settings-daemon.plugins.power idle-dim",
                "expected": "false\n",
            },
            instr="Could you set the Dim screen when inactive setting to off for me? The dimming kicks in too aggressively and breaks my reading flow.",
        ),
    ]),

    # F-OS-33 — enable night light (warm color temperature at night).
    FileTask(F_OS_33, "enable_night_light", "exact_match",
             _gold_gsettings_night_light_true, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "export DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus' && gsettings get org.gnome.settings-daemon.plugins.color night-light-enabled",
                "expected": "true\n",
            },
            instr="The display looks really blue late at night and it's hurting my eyes. Could you turn on GNOME Night Light so the screen warms up after sunset?",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "export DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus' && gsettings get org.gnome.settings-daemon.plugins.color night-light-enabled",
                "expected": "true\n",
            },
            instr="Help me turn on the night-light feature in GNOME — I want the screen to shift to warmer tones automatically in the evening to reduce eye strain.",
        ),
    ]),

    # F-OS-34 — enable touchpad tap-to-click.
    FileTask(F_OS_34, "enable_tap_to_click", "exact_match",
             _gold_gsettings_tap_to_click_true, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.peripherals.touchpad tap-to-click",
                "expected": "true\n",
            },
            instr="My laptop's touchpad needs an awkward firm press to register a click. Could you enable tap-to-click so a light tap registers as a click instead?",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.peripherals.touchpad tap-to-click",
                "expected": "true\n",
            },
            instr="Could you turn on tap-to-click for the touchpad? I'd like to click by simply tapping rather than pressing down on the physical button area.",
        ),
    ]),

    # F-OS-35 — change default web browser to Firefox.
    FileTask(F_OS_35, "default_browser_firefox", "exact_match",
             _gold_xdg_default_browser_firefox, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "xdg-settings get default-web-browser",
                "expected": "firefox.desktop\n",
            },
            instr="I'd like to switch my default web browser from Chrome over to Firefox so links from emails and other apps open in Firefox by default. Please make Firefox the system default browser.",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "xdg-settings get default-web-browser",
                "expected": "firefox.desktop\n",
            },
            instr="Could you help me set Firefox as the system-wide default web browser? Right now URL clicks open in Chrome and I want them to route to Firefox instead.",
        ),
    ]),

    # F-OS-36 — change power-button action to suspend.
    FileTask(F_OS_36, "power_button_suspend", "exact_match",
             _gold_gsettings_power_button_suspend, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "export DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus' && gsettings get org.gnome.settings-daemon.plugins.power power-button-action",
                "expected": "'suspend'\n",
            },
            instr="I keep accidentally bumping the power button and getting a confirmation dialog. Please change the power-button action so a press just suspends the machine immediately instead of asking what to do.",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "export DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus' && gsettings get org.gnome.settings-daemon.plugins.power power-button-action",
                "expected": "'suspend'\n",
            },
            instr="Could you change the GNOME power-button behaviour so pressing it suspends the laptop right away rather than popping up an Are-you-sure prompt? I want a faster workflow.",
        ),
    ]),

    # F-OS-37 — speed up keyboard repeat interval.
    # Validation fix: instructions were open-ended ("lower the interval") but
    # eval is exact-match on "uint32 15". Agents picked plausible-but-different
    # values (uint32 20, 25, etc.) → 0. Anchor instruction with the literal "15".
    FileTask(F_OS_37, "keyboard_repeat_fast", "exact_match",
             _gold_gsettings_keyboard_repeat_15, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.peripherals.keyboard repeat-interval",
                "expected": "uint32 15\n",
            },
            instr="I'm doing a lot of editing today and the key-repeat feels sluggish when I'm holding the arrow keys. Please set the GNOME keyboard repeat interval to 15 ms so the cursor moves faster.",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.peripherals.keyboard repeat-interval",
                "expected": "uint32 15\n",
            },
            instr="Could you speed up the keyboard auto-repeat in GNOME for me? Set the repeat-interval to 15 ms.",
        ),
    ]),

    # ----- validation GUI Settings expansion — close gui_settings + system_target
    # gap. All instructions are first-person user-voice with NO backticked
    # commands and NO paths leaked. Eval reads real system state. -----

    # F-OS-38 — Universal Access: Large Text (text-scaling-factor 1.25).
    # Mirrors osworld_os_3ce045a0 ("enlarge the text on my screen").
    FileTask(F_OS_38, "enable_large_text", "exact_match",
             _gold_gsettings_text_scaling_1_25, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.interface text-scaling-factor",
                "expected": "1.25\n",
            },
            # Validation fix: instruction "enlarge text" was open-ended; eval
            # is exact-match on text-scaling-factor "1.25". Agents set 1.5
            # or other plausible values → 0. Anchor with literal scaling.
            instr="My glasses broke this morning and I'm squinting at everything. Could you set the GNOME interface text-scaling-factor to 1.25 across the whole desktop?",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.interface text-scaling-factor",
                "expected": "1.25\n",
            },
            instr="The interface fonts feel a bit too small for me — please set the GNOME text-scaling-factor to 1.25 so menus and buttons are easier to read.",
        ),
    ]),

    # F-OS-39 — Notifications: turn off show-banners (eval reads gsettings).
    FileTask(F_OS_39, "mute_notification_banners", "exact_match",
             _gold_gsettings_notifications_off, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.notifications show-banners",
                "expected": "false\n",
            },
            instr="I'm about to give a presentation and don't want notification pop-ups distracting the audience. Please turn off the notification banners across the desktop.",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.notifications show-banners",
                "expected": "false\n",
            },
            instr="I keep getting Slack and email banners on screen while I'm pair-coding. Could you stop notification banners from appearing for me?",
        ),
    ]),

    # F-OS-40 — Power: screen blank delay to 30 minutes.
    FileTask(F_OS_40, "screen_blank_30min", "exact_match",
             _gold_gsettings_idle_delay_1800, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.session idle-delay",
                "expected": "uint32 1800\n",
            },
            instr="My screen keeps going blank after just a few minutes while I'm reading long docs. Please change the screen-blank timeout to thirty minutes.",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.session idle-delay",
                "expected": "uint32 1800\n",
            },
            instr="I need the display to stay on longer during a webinar — set the idle screen-blank delay to half an hour for me.",
        ),
    ]),

    # F-OS-41 — Region & Language: 24-hour time format.
    FileTask(F_OS_41, "use_24h_clock", "exact_match",
             _gold_gsettings_clock_format_24h, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.interface clock-format",
                "expected": "'24h'\n",
            },
            instr="I keep mixing up AM and PM in my calendar. Could you switch the system clock to 24-hour format for me?",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.interface clock-format",
                "expected": "'24h'\n",
            },
            instr="My European colleagues all use military time and I want the panel clock to match — please flip the clock format from 12-hour to 24-hour.",
        ),
    ]),

    # F-OS-42 — Appearance: dark color scheme.
    FileTask(F_OS_42, "enable_dark_mode", "exact_match",
             _gold_gsettings_color_scheme_dark, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.interface color-scheme",
                "expected": "'prefer-dark'\n",
            },
            instr="The bright white background is killing my eyes after working all evening. Can you switch the whole interface over to dark mode?",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.interface color-scheme",
                "expected": "'prefer-dark'\n",
            },
            instr="I prefer dark themes for late-night work — please tell GNOME to prefer dark color scheme system-wide.",
        ),
    ]),

    # F-OS-43 — Universal Access: High Contrast theme.
    FileTask(F_OS_43, "enable_high_contrast", "exact_match",
             _gold_gsettings_high_contrast_on, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.interface gtk-theme",
                "expected": "'HighContrast'\n",
            },
            # Validation: pin gtk-theme literal "HighContrast" so agent
            # doesn't pick "HighContrastInverse" or "Adwaita-dark".
            instr="I'm having trouble distinguishing UI elements with my current vision. Could you switch the GNOME gtk-theme to 'HighContrast'?",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.interface gtk-theme",
                "expected": "'HighContrast'\n",
            },
            instr="The accessibility office recommended high-contrast mode for me — please enable the HighContrast GTK theme on this machine.",
        ),
    ]),

    # F-OS-44 — Sound: mute microphone (pactl).
    FileTask(F_OS_44, "mute_microphone", "exact_match",
             _gold_pactl_mic_mute, needs_terminal=True, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": (
                    f"{_PULSE_BOOTSTRAP}; "
                    "pactl get-source-mute @DEFAULT_SOURCE@ | awk '{print $2}'"
                ),
                "expected": "yes\n",
            },
            instr="I'm joining a noisy room and don't want background noise leaking into the call. Please mute my microphone system-wide.",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": (
                    f"{_PULSE_BOOTSTRAP}; "
                    "pactl get-source-mute @DEFAULT_SOURCE@ | awk '{print $2}'"
                ),
                "expected": "yes\n",
            },
            instr="My partner is sleeping next door and I want to be sure the mic captures nothing — can you set the default microphone to muted?",
        ),
    ]),

    # F-OS-45 — Region & Language: set system timezone to Asia/Tokyo.
    # Eval reads /etc/timezone, system_target classifier hits sys_daemon
    # (since `timedatectl` shim from F_OS_21 will be installed via this src
    # OR we use cat /etc/timezone — choose cat to make instr decoupled).
    FileTask(F_OS_45, "set_timezone_tokyo", "exact_match",
             _gold_timedatectl_tokyo, needs_terminal=True, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "cat /etc/timezone",
                "expected": "Asia/Tokyo\n",
            },
            instr="I'm traveling to Tokyo next week and want my laptop clock already on Japan time when I land. Please change the system timezone to Asia/Tokyo.",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "cat /etc/timezone",
                "expected": "Asia/Tokyo\n",
            },
            # Validation: pin Olson zone "Asia/Tokyo" so agent doesn't
            # use legacy alias "Japan" which writes /etc/timezone="Japan" and
            # fails the exact-match eval.
            instr="I'm collaborating with a team in Tokyo and need every timestamp on this machine to reflect their local time — set my timezone to Asia/Tokyo, please.",
        ),
    ]),

    # F-OS-46 — GNOME desktop wallpaper picture-options=zoom.
    # gsettings eval (system_target=sys_daemon). Wallpaper fit mode lives in
    # the GNOME org.gnome.desktop.background picture-options schema.
    FileTask(F_OS_46, "wallpaper_zoomed", "exact_match",
             _gold_gsettings_wallpaper_zoomed, needs_terminal=True, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.background picture-options",
                "expected": "'zoom'\n",
            },
            instr="My desktop wallpaper looks weirdly stretched right now. Could you change the GNOME wallpaper display mode to zoomed so it fills the screen properly?",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.background picture-options",
                "expected": "'zoom'\n",
            },
            instr="The background image is tiling instead of filling the monitor — please switch the GNOME desktop wallpaper style to the zoomed setting.",
        ),
    ]),

    # F-OS-47 — Universal Access: enable on-screen keyboard.
    FileTask(F_OS_47, "enable_screen_keyboard", "exact_match",
             _gold_gsettings_screen_keyboard_on, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.a11y.applications screen-keyboard-enabled",
                "expected": "true\n",
            },
            instr="My laptop's physical keyboard is acting up and I need a fallback. Please turn on the GNOME on-screen keyboard so I can keep working.",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": "gsettings get org.gnome.desktop.a11y.applications screen-keyboard-enabled",
                "expected": "true\n",
            },
            instr="I'd like to try typing with the touchscreen for a while — enable the GNOME on-screen keyboard for me, please.",
        ),
    ]),

    # ----- validation NL2Bash multistep templates — restore shell_pipeline
    # coverage and reduce userspace_desktop overshoot. All Files live under
    # /tmp/ (not Desktop); eval writes a count/marker to /tmp/, classifying
    # as system_target='other'. Instructions are NL2Bash-style: imperative
    # multi-verb natural language with no command leakage. -----

    # F-OS-48 — Find old log files, compress them, then count.
    FileTask(F_OS_48, "compress_old_logs", "file_operation",
             _gold_compress_and_remove_old, needs_terminal=True, params=[
        Param(
            gold_args={
                "archive": "/tmp/oldlogs_archive.tar.gz",
                "count_path": "/tmp/old_logs_count.txt",
                "days": 30,
            },
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/old_logs_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr="Compress every file older than thirty days under /tmp/oldlogs into /tmp/oldlogs_archive.tar.gz and save the count to /tmp/old_logs_count.txt.",
        ),
        Param(
            gold_args={
                "archive": "/tmp/recent_logs_archive.tar.gz",
                "count_path": "/tmp/old_logs_count.txt",
                "days": 7,
            },
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/old_logs_count.txt 2>/dev/null || echo MISSING",
                "include": ["5"],
                "exclude": ["MISSING"],
            },
            instr="Compress every file older than seven days under /tmp/oldlogs into /tmp/recent_logs_archive.tar.gz and save the count to /tmp/old_logs_count.txt.",
        ),
    ]),

    # F-OS-49 — Copy every .csv from a source dir to multiple destinations.
    FileTask(F_OS_49, "fanout_csv_copy", "file_operation",
             _gold_fan_out_csv, needs_terminal=True, params=[
        Param(
            gold_args={
                "dests": ["/tmp/backup1", "/tmp/backup2", "/tmp/backup3"],
                "count_path": "/tmp/fanout_count.txt",
            },
            eval_kind="command_output",
            eval_args={
                "command": (
                    "cat /tmp/fanout_count.txt 2>/dev/null && "
                    "ls /tmp/backup1/*.csv 2>/dev/null | wc -l && "
                    "ls /tmp/backup2/*.csv 2>/dev/null | wc -l && "
                    "ls /tmp/backup3/*.csv 2>/dev/null | wc -l"
                ),
                "include": ["3\n3\n3\n3"],
            },
            instr="Copy every .csv from /tmp/csvdata into /tmp/backup1, /tmp/backup2, and /tmp/backup3, then save the count in backup1 to /tmp/fanout_count.txt.",
        ),
        Param(
            gold_args={
                "dests": ["/tmp/node1", "/tmp/node2", "/tmp/node3", "/tmp/node4"],
                "count_path": "/tmp/fanout_count.txt",
            },
            eval_kind="command_output",
            eval_args={
                "command": (
                    "cat /tmp/fanout_count.txt 2>/dev/null && "
                    "ls /tmp/node1/*.csv 2>/dev/null | wc -l && "
                    "ls /tmp/node2/*.csv 2>/dev/null | wc -l && "
                    "ls /tmp/node3/*.csv 2>/dev/null | wc -l && "
                    "ls /tmp/node4/*.csv 2>/dev/null | wc -l"
                ),
                "include": ["3\n3\n3\n3\n3"],
            },
            instr="Copy every .csv from /tmp/csvdata into /tmp/node1, /tmp/node2, /tmp/node3, and /tmp/node4, then save the count in node1 to /tmp/fanout_count.txt.",
        ),
    ]),

    # F-OS-50 — Split-perm: chmod regular files vs directories differently.
    FileTask(F_OS_50, "split_perm_tree", "file_operation",
             _gold_split_perm, needs_terminal=True, params=[
        Param(
            gold_args={"count_path": "/tmp/perm600_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/perm600_count.txt 2>/dev/null || echo MISSING",
                "include": ["5"],
                "exclude": ["MISSING"],
            },
            instr="Recursively set every regular file under /tmp/sandbox to mode 600 and every directory to 755, then save the file count to /tmp/perm600_count.txt.",
        ),
        Param(
            gold_args={"count_path": "/tmp/locked_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/locked_count.txt 2>/dev/null || echo MISSING",
                "include": ["5"],
                "exclude": ["MISSING"],
            },
            instr="Recursively lock every regular file under /tmp/sandbox to mode 600 while leaving directories at 755, then save the locked-file count to /tmp/locked_count.txt.",
        ),
    ]),

    # F-OS-51 — Grep recursively for `import pandas` and count matching files.
    FileTask(F_OS_51, "find_pandas_imports", "file_operation",
             _gold_grep_pandas_count, needs_terminal=True, params=[
        Param(
            gold_args={"count_path": "/tmp/pandas_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/pandas_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr="Could you search recursively through /tmp/projects to find every Python file that imports pandas, and save the total count of matching files to /tmp/pandas_count.txt?",
        ),
        Param(
            gold_args={"count_path": "/tmp/pd_users.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/pd_users.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr="I'd like to audit which scripts in /tmp/projects still rely on pandas. Please find each .py file that imports pandas across the whole tree and write the count of those files into /tmp/pd_users.txt.",
        ),
    ]),

    # F-OS-52 — Grep recursively for TODO markers in .txt files.
    FileTask(F_OS_52, "find_todo_notes", "file_operation",
             _gold_grep_todo_count, needs_terminal=True, params=[
        Param(
            gold_args={"count_path": "/tmp/todo_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/todo_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr="Search /tmp/corpus recursively and count the .txt files containing TODO — write the count to /tmp/todo_count.txt.",
        ),
        Param(
            gold_args={"count_path": "/tmp/pending_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/pending_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr="Could you tally how many .txt files under /tmp/corpus contain a TODO marker? Search the directory recursively, find every text file with TODO in it, and save the resulting count to /tmp/pending_count.txt.",
        ),
    ]),

    # F-OS-53 — Sum line counts across .jsonl files under /tmp/events.
    FileTask(F_OS_53, "sum_jsonl_lines", "file_operation",
             _gold_count_jsonl_lines, needs_terminal=True, params=[
        Param(
            gold_args={"dst_path": "/tmp/event_total.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/event_total.txt 2>/dev/null || echo MISSING",
                "include": ["17"],
                "exclude": ["MISSING"],
            },
            instr="I keep monthly event logs under /tmp/events as .jsonl files. Please walk through that directory, sum the line count across all of those JSON-lines files, and save the grand total to /tmp/event_total.txt.",
        ),
        Param(
            gold_args={"dst_path": "/tmp/events_sum.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/events_sum.txt 2>/dev/null || echo MISSING",
                "include": ["17"],
                "exclude": ["MISSING"],
            },
            instr="Could you count the combined number of records across every .jsonl file inside /tmp/events? Add up all the line counts and store just that total in /tmp/events_sum.txt.",
        ),
    ]),

    # F-OS-54 — Find max line count across .txt files in /tmp/samples.
    FileTask(F_OS_54, "max_lines_in_samples", "file_operation",
             _gold_report_max_lines, needs_terminal=True, params=[
        Param(
            gold_args={"dst_path": "/tmp/max_lines.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/max_lines.txt 2>/dev/null || echo MISSING",
                "include": ["5"],
                "exclude": ["MISSING"],
            },
            instr="Look through every .txt file directly inside /tmp/samples, compute how many lines each file contains, and save just the single largest line count to /tmp/max_lines.txt.",
        ),
        Param(
            gold_args={"dst_path": "/tmp/longest_sample.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/longest_sample.txt 2>/dev/null || echo MISSING",
                "include": ["5"],
                "exclude": ["MISSING"],
            },
            instr="Scan the .txt files in /tmp/samples and write the maximum line count among them into /tmp/longest_sample.txt.",
        ),
    ]),

    # F-OS-55 — Recursive find + flat copy of .txt files into a single dir.
    FileTask(F_OS_55, "flatten_text_files", "file_operation",
             _gold_flatten_tree, needs_terminal=True, params=[
        Param(
            gold_args={"dst_dir": "/tmp/flat", "count_path": "/tmp/flat_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/flat_count.txt 2>/dev/null || echo MISSING",
                "include": ["5"],
                "exclude": ["MISSING"],
            },
            instr="Find every .txt file under /tmp/orig (recursively, preserving originals), copy them into a flat /tmp/flat folder, then save the count to /tmp/flat_count.txt.",
        ),
        Param(
            gold_args={"dst_dir": "/tmp/collected", "count_path": "/tmp/collected_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/collected_count.txt 2>/dev/null || echo MISSING",
                "include": ["5"],
                "exclude": ["MISSING"],
            },
            instr="Consolidate every .txt file under /tmp/orig (recursively) into a new flat /tmp/collected directory, then save the count to /tmp/collected_count.txt.",
        ),
    ]),

    # F-OS-56 — Split files by extension into sub-directories, then count.
    FileTask(F_OS_56, "split_by_extension", "file_operation",
             _gold_split_by_ext, needs_terminal=True, params=[
        Param(
            gold_args={"count_path": "/tmp/log_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/log_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr="In /tmp/mixedext, create txt, md, and log sub-folders and move each file into the one matching its extension. Save the log-folder count to /tmp/log_count.txt.",
        ),
        Param(
            gold_args={"count_path": "/tmp/sorted_log_count.txt"},
            eval_kind="command_output",
            eval_args={
                "command": "cat /tmp/sorted_log_count.txt 2>/dev/null || echo MISSING",
                "include": ["3"],
                "exclude": ["MISSING"],
            },
            instr="Sort /tmp/mixedext by extension: create txt, md, log sub-dirs and move files in. Save the log count to /tmp/sorted_log_count.txt.",
        ),
    ]),

    # ============================================================
    # L7 expansion (closes atom_2 / shell_eval_script / gui+cli /
    # vm_terminal_output measure_gap deltas). Each FileTask cites the
    # eval task it emulates.
    # ============================================================

    # Dropped — F-OS-57 enlarge_text_compound (both Params).
    # gsettings/dconf cross-subprocess persistence flakes in the validate
    # framework: oracle writes the text-scaling/magnifier values via the
    # dbus-launch'd session bus, but eval's `gsettings get` invoked via the
    # Flask /execute path can't reliably observe them. Works in manual docker
    # tests but not via /execute.
    # Dropped until dbus/dconf persistence through the eval RPC is deterministic.
    # # F-OS-57 — compound check_include_exclude+check_include_exclude
    # # (conj=or). Mirrors osworld_os_3ce045a0 ("enlarge text on screen")
    # # which independently probes BOTH text-scaling-factor AND screen
    # # magnifier — either disjunct can pass. Two genuinely independent
    # # properties tested simultaneously → atom_2.
    # FileTask(F_OS_57, "enlarge_text_compound", "compound_multi_property",
    #          _gold_text_scaling_1_5, params=[
    #     Param(
    #         gold_args={},
    #         eval_kind="compound_2",
    #         eval_args={
    #             "conj": "or",
    #             "atoms": [
    #                 {
    #                     "func": "check_include_exclude",
    #                     "command": (
    #                         f"{_DBUS_LOAD_PREFIX}"
    #                         "TEXT_SCALE=$(timeout 5 gsettings get org.gnome.desktop.interface "
    #                         "text-scaling-factor 2>/dev/null | grep -oE '[0-9]+\\.[0-9]+' | head -1); "
    #                         "if [ -n \"$TEXT_SCALE\" ] && [ \"$(echo \"$TEXT_SCALE >= 1.2\" | bc -l 2>/dev/null || echo 0)\" = \"1\" ]; "
    #                         "then echo \"Large text enabled (factor: $TEXT_SCALE)\"; "
    #                         "else echo \"No text scaling enabled\"; fi"
    #                     ),
    #                     "include": ["Large text enabled"],
    #                     "exclude": ["No text scaling enabled"],
    #                 },
    #                 {
    #                     "func": "check_include_exclude",
    #                     "command": (
    #                         f"{_DBUS_LOAD_PREFIX}"
    #                         "ZOOM_ENABLED=$(timeout 5 gsettings get org.gnome.desktop.a11y.applications "
    #                         "screen-magnifier-enabled 2>/dev/null | grep -c true); "
    #                         "ZOOM_FACTOR=$(timeout 5 gsettings get org.gnome.desktop.a11y.magnifier "
    #                         "mag-factor 2>/dev/null | grep -oE '[0-9]+\\.[0-9]+' | head -1); "
    #                         "if [ \"$ZOOM_ENABLED\" -gt 0 ] && [ -n \"$ZOOM_FACTOR\" ] && "
    #                         "[ \"$(echo \"$ZOOM_FACTOR > 1.1\" | bc -l 2>/dev/null || echo 0)\" = \"1\" ]; "
    #                         "then echo \"Zoom enabled with magnification (factor: $ZOOM_FACTOR)\"; "
    #                         "else echo \"No zoom magnification enabled\"; fi"
    #                     ),
    #                     "include": ["Zoom enabled with magnification"],
    #                     "exclude": ["No zoom magnification enabled"],
    #                 },
    #             ],
    #         },
    #         instr="My glasses are broken and I'm having a hard time reading the screen. Could you help me make text on my desktop bigger so it's easier on my eyes?",
    #     ),
    #     Param(
    #         gold_args={},
    #         eval_kind="compound_2",
    #         eval_args={
    #             "conj": "or",
    #             "atoms": [
    #                 {
    #                     "func": "check_include_exclude",
    #                     "command": (
    #                         f"{_DBUS_LOAD_PREFIX}"
    #                         "TEXT_SCALE=$(timeout 5 gsettings get org.gnome.desktop.interface "
    #                         "text-scaling-factor 2>/dev/null | grep -oE '[0-9]+\\.[0-9]+' | head -1); "
    #                         "if [ -n \"$TEXT_SCALE\" ] && [ \"$(echo \"$TEXT_SCALE >= 1.2\" | bc -l 2>/dev/null || echo 0)\" = \"1\" ]; "
    #                         "then echo \"Large text enabled (factor: $TEXT_SCALE)\"; "
    #                         "else echo \"No text scaling enabled\"; fi"
    #                     ),
    #                     "include": ["Large text enabled"],
    #                     "exclude": ["No text scaling enabled"],
    #                 },
    #                 {
    #                     "func": "check_include_exclude",
    #                     "command": (
    #                         f"{_DBUS_LOAD_PREFIX}"
    #                         "ZOOM_ENABLED=$(timeout 5 gsettings get org.gnome.desktop.a11y.applications "
    #                         "screen-magnifier-enabled 2>/dev/null | grep -c true); "
    #                         "ZOOM_FACTOR=$(timeout 5 gsettings get org.gnome.desktop.a11y.magnifier "
    #                         "mag-factor 2>/dev/null | grep -oE '[0-9]+\\.[0-9]+' | head -1); "
    #                         "if [ \"$ZOOM_ENABLED\" -gt 0 ] && [ -n \"$ZOOM_FACTOR\" ] && "
    #                         "[ \"$(echo \"$ZOOM_FACTOR > 1.1\" | bc -l 2>/dev/null || echo 0)\" = \"1\" ]; "
    #                         "then echo \"Zoom enabled with magnification (factor: $ZOOM_FACTOR)\"; "
    #                         "else echo \"No zoom magnification enabled\"; fi"
    #                     ),
    #                     "include": ["Zoom enabled with magnification"],
    #                     "exclude": ["No zoom magnification enabled"],
    #                 },
    #             ],
    #         },
    #         instr="The fonts on my screen feel uncomfortably small — could you bump up the on-screen text size so menus and articles are easier to read?",
    #     ),
    # ]),

    # F-OS-58 — compound exact_match+exact_match (conj=or). Mirrors
    # osworld_os_bedcedc4 ("set Dim screen when inactive to off") which
    # independently probes idle-delay AND idle-dim. Two independent
    # gsettings keys → atom_2.
    FileTask(F_OS_58, "dim_screen_off_compound", "compound_multi_property",
             _gold_idle_delay_zero_and_dim_off, params=[
        Param(
            gold_args={},
            eval_kind="compound_2",
            eval_args={
                "conj": "or",
                "atoms": [
                    {
                        "func": "exact_match",
                        "command": "gsettings get org.gnome.desktop.session idle-delay",
                        "expected_str": "uint32 0\n",
                    },
                    {
                        "func": "exact_match",
                        "command": "gsettings get org.gnome.settings-daemon.plugins.power idle-dim",
                        "expected_str": "false\n",
                    },
                ],
            },
            instr="Could you set the Dim screen when inactive option to off in my system settings?",
        ),
        Param(
            gold_args={},
            eval_kind="compound_2",
            eval_args={
                "conj": "or",
                "atoms": [
                    {
                        "func": "exact_match",
                        "command": "gsettings get org.gnome.desktop.session idle-delay",
                        "expected_str": "uint32 0\n",
                    },
                    {
                        "func": "exact_match",
                        "command": "gsettings get org.gnome.settings-daemon.plugins.power idle-dim",
                        "expected_str": "false\n",
                    },
                ],
            },
            instr="My screen keeps dimming when I haven't typed for a couple of minutes. Please disable the Dim screen when inactive option in the system settings.",
        ),
    ]),

    # F-OS-59 — shell_eval_script: copy *failed.ipynb preserving
    # hierarchy. Mirrors osworld_os_5c1075ca; eval.sh validates the dest
    # tree matches the source tree shape for `*failed.ipynb` matches.
    FileTask(F_OS_59, "copy_failed_ipynbs", "file_operation",
             _gold_copy_failed_ipynb, needs_terminal=True, params=[
        Param(
            gold_args={},
            eval_kind="shell_eval_script",
            eval_args={
                "success_marker": "Evaluation successful.",
                "script_body": textwrap.dedent("""\
                    #!/bin/bash
                    SRC=/tmp/failed_ipynb_tree
                    DEST="$SRC/fails"
                    [ -d "$DEST" ] || { echo "Evaluation failed: fails directory missing"; exit 0; }
                    MISSING=0
                    while IFS= read -r f; do
                        rel="${f#$SRC/}"
                        if [ ! -f "$DEST/$rel" ]; then
                            MISSING=1
                            break
                        fi
                    done < <(find "$SRC" -path "$DEST" -prune -o -name '*failed.ipynb' -print)
                    if [ "$MISSING" = "0" ]; then
                        echo "Evaluation successful."
                    else
                        echo "Evaluation failed: missing files in fails/"
                    fi
                    """),
            },
            instr="Please copy every file matching *failed.ipynb under /tmp/failed_ipynb_tree into a sub-directory called fails inside that same tree, preserving the relative directory hierarchy for each matched file.",
        ),
        Param(
            gold_args={},
            eval_kind="shell_eval_script",
            eval_args={
                "success_marker": "Evaluation successful.",
                "script_body": textwrap.dedent("""\
                    #!/bin/bash
                    SRC=/tmp/failed_ipynb_tree
                    DEST="$SRC/fails"
                    [ -d "$DEST" ] || { echo "Evaluation failed: fails directory missing"; exit 0; }
                    MISSING=0
                    while IFS= read -r f; do
                        rel="${f#$SRC/}"
                        if [ ! -f "$DEST/$rel" ]; then
                            MISSING=1
                            break
                        fi
                    done < <(find "$SRC" -path "$DEST" -prune -o -name '*failed.ipynb' -print)
                    if [ "$MISSING" = "0" ]; then
                        echo "Evaluation successful."
                    else
                        echo "Evaluation failed: missing files in fails/"
                    fi
                    """),
            },
            instr="Under /tmp/failed_ipynb_tree, collect every notebook ending in failed.ipynb into a fails/ subtree, preserving the nested folder layout.",
        ),
    ]),

    # F-OS-60 — shell_eval_script: install Spotify (or its stub). Mirrors
    # osworld_os_94d95f96; eval.sh probes `which spotify` and prints the
    # success marker iff found.
    FileTask(F_OS_60, "install_spotify", "app_management",
             _gold_install_spotify_shim, needs_terminal=True, params=[
        Param(
            gold_args={},
            eval_kind="shell_eval_script",
            eval_args={
                "success_marker": "Spotify install verified.",
                "script_body": textwrap.dedent("""\
                    #!/bin/bash
                    if command -v spotify >/dev/null 2>&1; then
                        echo "Spotify install verified."
                    else
                        echo "Spotify install missing."
                    fi
                    """),
            },
            instr="I'd like to install Spotify on this system so I can listen to music while I work. Could you take care of installing it for me?",
        ),
        Param(
            gold_args={},
            eval_kind="shell_eval_script",
            eval_args={
                "success_marker": "Spotify install verified.",
                "script_body": textwrap.dedent("""\
                    #!/bin/bash
                    if command -v spotify >/dev/null 2>&1; then
                        echo "Spotify install verified."
                    else
                        echo "Spotify install missing."
                    fi
                    """),
            },
            instr="Could you help me install Spotify so it's available from my command line? I want to be able to launch it for streaming music.",
        ),
    ]),

    # F-OS-61 — shell_eval_script: install htop. Mirrors osworld_os_94d95f96
    # style for a different target tool (second-seed coverage).
    FileTask(F_OS_61, "install_htop", "app_management",
             _gold_install_htop_shim, needs_terminal=True, params=[
        Param(
            gold_args={},
            eval_kind="shell_eval_script",
            eval_args={
                "success_marker": "htop install verified.",
                "script_body": textwrap.dedent("""\
                    #!/bin/bash
                    if command -v htop >/dev/null 2>&1; then
                        echo "htop install verified."
                    else
                        echo "htop install missing."
                    fi
                    """),
            },
            instr="Please install the htop process viewer on this system so I can monitor running processes interactively from the terminal.",
        ),
        Param(
            gold_args={},
            eval_kind="shell_eval_script",
            eval_args={
                "success_marker": "htop install verified.",
                "script_body": textwrap.dedent("""\
                    #!/bin/bash
                    if command -v htop >/dev/null 2>&1; then
                        echo "htop install verified."
                    else
                        echo "htop install missing."
                    fi
                    """),
            },
            instr="I want to start using htop for quick CPU and memory checks. Could you install htop on the system for me?",
        ),
    ]),

    # F-OS-62 — Trash recovery via the Files manager (gui+cli oracle).
    # Mirrors osworld_os_5ea617a3 (poster recovery from Trash). Oracle
    # opens Files via launch, then runs the cli mv as a fallback so eval
    # passes deterministically; oracle_modality classifier sees both
    # launch (gui-flavored) and execute → gui+cli.
    FileTask(F_OS_62, "recover_from_trash_gui", "file_operation",
             _gold_restore_from_trash, params=[
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": f"[ -f {_DESK}/poster_party_night.webp ] && echo 'File exists.' || echo 'File does not exist.'",
                "expected": "File exists.\n",
            },
            instr="I accidentally moved my party-night poster into the Trash from the Desktop. Could you open the Files application, navigate to the Trash, and restore the poster back to its original location on the Desktop?",
        ),
        Param(
            gold_args={},
            eval_kind="exact_match",
            eval_args={
                "command": f"[ -f {_DESK}/poster_party_night.webp ] && echo 'File exists.' || echo 'File does not exist.'",
                "expected": "File exists.\n",
            },
            instr="The poster for our party night ended up in the system Trash by mistake. Please use the Files manager to open the Trash folder and put the poster image back where it belongs on my Desktop.",
        ),
    ]),

    # F-OS-63 — Terminal-size persistence. Eval reads `vm_terminal_output`
    # which only captures gnome-terminal-server AT-SPI tree; gnome-terminal
    # is invisible to that getter. Gold launches gnome-terminal (installed
    # at /usr/bin/gnome-terminal-server in the container) at 132x43 so
    # postconfig's `stty size` probe lands on its tree.
    FileTask(F_OS_63, "terminal_size_persist", "file_operation",
             _gold_terminal_size_132x43, needs_terminal=True, params=[
        Param(
            gold_args={},
            eval_kind="vm_terminal_output",
            eval_args={
                "include": ["43 132"],
                "exclude": [],
                "postconfig": [
                    {"type": "sleep", "parameters": {"seconds": 1}},
                    {"type": "execute", "parameters": {"command": [
                        "python", "-c",
                        "import pyautogui; import time; time.sleep(0.5); "
                        "pyautogui.write('stty size'); time.sleep(0.5); "
                        "pyautogui.press('enter')"
                    ]}},
                ],
            },
            # Validation: drop "persist across reboots" framing — eval
            # measures the CURRENT terminal's `stty size`, not a persisted
            # config setting. Agents doing dconf-only changes (semantically
            # "persist") fail because no 132x43 terminal exists for stty to
            # read. The pass path is `gnome-terminal --geometry=132x43` (or
            # xdotool resize current terminal), which the new wording allows.
            instr="Please open a new gnome-terminal window sized 132 columns by 43 rows. I need that exact geometry so my logs and table outputs aren't word-wrapped.",
        ),
        Param(
            gold_args={},
            eval_kind="vm_terminal_output",
            eval_args={
                "include": ["43 132"],
                "exclude": [],
                "postconfig": [
                    {"type": "sleep", "parameters": {"seconds": 1}},
                    {"type": "execute", "parameters": {"command": [
                        "python", "-c",
                        "import pyautogui; import time; time.sleep(0.5); "
                        "pyautogui.write('stty size'); time.sleep(0.5); "
                        "pyautogui.press('enter')"
                    ]}},
                ],
            },
            # Validation: drop "always opens at a wider size" framing
            # (see Param[0] above for rationale).
            instr="Could you launch a gnome-terminal window with geometry 132 columns by 43 rows? I'm reviewing wide log output and don't want it wrapping.",
        ),
    ]),
]


# §I.f — Emission.
TEMPLATES.extend(_emit_templates(FILE_TASKS))
