"""Desktop setup/evaluation functions for ``lite.cuagym``.

CUA-Gym desktop bundles ship official ``initial_setup.py`` (build files + launch a
GUI app on DISPLAY :0) and ``reward.py`` (read files, print ``REWARD: x``). We run
them verbatim inside the shared ``cua-lite/lite.cuagym`` container:

  - the scripts hardcode ``/home/user`` -> the shared lite.osworld base uses the
    same desktop user/home path;
  - the scripts hardcode ``DISPLAY=:0`` while the image's Xvnc is ``:1`` -> we
    bridge the X socket (``X0 -> X1``) so :0 reaches the running server, no script
    edit needed;
  - the scripts import doc/PDF libs -> run under the image's Python 3.12, which
    has them (openpyxl/python-docx/python-pptx/pymupdf/odfpy/reportlab/...).

Usage:
    from lite.gym.envs.lite.cuagym.src.desktop.scripts import setup_fn, evaluate_final_fn
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any, NoReturn

from lite.gym.envs.lite.cuagym.src.utils.container import write_bytes, write_text
from lite.gym.envs.lite.cuagym.src.utils.display import bridge_display
from lite.gym.envs.lite.cuagym.src.utils.reward import REWARD_RE, parse_reward
from lite.gym.envs.lite.cuagym.src.utils.routing import _SYSTEM_ABI_IMPORT_RE
from lite.gym.envs.lite.cuagym.src.utils.runtime import (
    ENV_PY,
    UNO_PY,
    bundle_needs_judge,
    install_reward_judge,
    wait_for_new_window,
    window_ids,
)
from lite.gym.errors import CuaGymTaskError
from lite.gym.sandbox.exec_stdio.client import ExecStdioError

logger = logging.getLogger(__name__)

_COMMAND_TIMEOUT = 300.0
# _SYSTEM_ABI_IMPORT_RE is the single source of truth in utils/routing.py (imported
# above); UNO/PyGObject imports pin a script to the system-Python uno-venv (UNO_PY).


def _python_for_source(source: str) -> str:
    # UNO and PyGObject are tied to the image's system Python 3.10. The
    # env-only venv combines those bindings with task-side Python libraries.
    return UNO_PY if _SYSTEM_ABI_IMPORT_RE.search(source) else ENV_PY


def _python_command_for_source(source: str) -> str:
    # The reward/setup script runs as the desktop user; the interpreter is the
    # user-readable env venv (/opt/env/venv, ENV_PY) — or UNO_PY for gi/uno sources.
    interpreter = _python_for_source(source)
    if interpreter == UNO_PY:
        # Official setup scripts sometimes generate another UNO script and
        # launch it with bare `python3`; keep those children on the same ABI.
        return f"PATH={Path(UNO_PY).parent}:$PATH {UNO_PY}"
    return ENV_PY


_DOC_KINDS = {"pptx", "docx", "xlsx"}
_OFFICE_EXTS = {".doc", ".docx", ".odt", ".xls", ".xlsx", ".ods", ".ppt", ".pptx", ".odp"}


def _invalid_bundle(
    message: str,
    *,
    phase: str,
    cause: Exception | None = None,
) -> NoReturn:
    error = CuaGymTaskError(
        f"lite.cuagym desktop {phase} bundle is invalid: {message}",
        phase=phase,
        kind="invalid_bundle",
    )
    if cause is None:
        raise error
    raise error from cause


def _task_config(task_json: Path) -> list[dict]:
    if not task_json.exists():
        return []
    try:
        value = json.loads(task_json.read_text())
        config = value.get("config", [])
        if not isinstance(config, list):
            raise TypeError("task.json config must be a list")
        return config
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        _invalid_bundle(
            f"{type(exc).__name__}: {exc}",
            phase="setup",
            cause=exc,
        )


def _local_downloads(task_json: Path) -> list[tuple[Path, str]]:
    downloads: list[tuple[Path, str]] = []
    for step in _task_config(task_json):
        if step.get("type") != "download":
            continue
        for file_spec in step.get("parameters", {}).get("files", []):
            remote_path = file_spec.get("path")
            source_url = file_spec.get("url", "")
            if remote_path and source_url.startswith("./"):
                downloads.append((task_json.parent / source_url[2:], remote_path))
    return downloads


def _open_paths(task_json: Path) -> list[str]:
    return [
        remote_path
        for step in _task_config(task_json)
        if step.get("type") == "open"
        for remote_path in [step.get("parameters", {}).get("path")]
        if remote_path
    ]


def _opener_for(remote_path: str) -> str:
    return "soffice" if Path(remote_path).suffix.lower() in _OFFICE_EXTS else "xdg-open"


#: Commands whose start raises a window. Recognised only as the resolved HEAD of a
#: backgrounded shell command (below), because a bare mention is usually the opposite
#: of a launch: over the pinned desktop bundles every
#: `subprocess.run|call|check_call(...)` naming one of these is either
#: `pkill -f <app>` (217 calls) or `code --install-extension` (76) — 0 are launches,
#: so keying the gate on those call forms would re-create the spurious hard failure
#: this gate exists to avoid.
_GUI_BINARIES = frozenset({
    "soffice", "libreoffice", "lowriter", "localc", "loimpress", "lodraw",
    "code", "codium", "gimp", "vlc", "cvlc", "nautilus", "gedit", "eog",
    "evince", "atril", "okular", "firefox", "chromium", "chromium-browser",
    "google-chrome", "google-chrome-stable", "thunderbird", "xterm",
    "gnome-terminal", "x-terminal-emulator", "gnome-system-monitor",
    "gnome-control-center", "gnome-calculator", "feh", "kid3", "kid3-qt",
    "easytag", "picard", "totem", "rhythmbox",
})

#: A setup script that starts a GUI app itself, PYTHON form: the bundles' own
#: helper is `launch_gui(cmd)` (a non-blocking `subprocess.Popen`); the bare-Popen
#: and desktop-opener forms cover the handful that inline it instead.
_GUI_LAUNCH_RE = re.compile(
    r"\blaunch_gui\s*\(|\bsubprocess\.Popen\s*\(|\bxdg-open\b|\bgtk-launch\b"
)

#: soffice/libreoffice double as batch converters (`--headless --convert-to`, and
#: `--invisible` for the UNO bridge); those runs raise no window, so a backgrounded
#: one must NOT arm the gate.
_WINDOWLESS = frozenset({"--headless", "--invisible"})


# --------------------------------------------------------------------------- #
# The SHELL form of a launch, which none of the python-shaped patterns can see:
# `code "$WORKSPACE" &`, `DISPLAY=:0 nohup soffice --calc "$F" >/dev/null 2>&1 &`,
# `( DISPLAY=:0 gimp "$XCF" || true ) &`. This is how the .sh bundles launch — and
# relaxing the window gate on one silently converts a dead GUI session into a
# reward-0 trajectory instead of a diagnosable `no_task_window`.
#
# This used to be ONE regex anchored at `^`, and the anchor was the bug. bash
# writes a command head in ways no line-anchored pattern can follow, and 23 of the
# 283 bundles the gate relaxed do launch an app through one of them:
#
#   * `"$LIBRE_BIN" --writer "$ODT" &` where `LIBRE_BIN="$(command -v libreoffice
#     || command -v soffice)` — a VARIABLE head, 12 bundles, the dominant class.
#     No regex can resolve it; it needs an assignment table.
#   * `google-chrome … "https://…/docs/#installation" & disown` — the old
#     `[^\n#]*` body stopped dead at the `#` of a URL FRAGMENT.
#   * `/usr/bin/nohup soffice …`, `/usr/bin/env DISPLAY=:0 libreoffice …` — the old
#     wrapper list only accepted BARE `nohup`/`env`.
#   * `DISPLAY=:0 setsid -f chromium … ` — detached with NO `&` at all, and the old
#     pattern structurally required one.
#   * `… && DISPLAY=:0 gnome-terminal … & disown`, `then google-chrome … & disown`,
#     `echo pw | sudo -S -u user … chromium … &` — a head that is not at `^`.
#   * `su - user -c "DISPLAY=:0 nautilus /home/user/Webpack &"`,
#     `setsid bash -c "… chromium … "`, `nohup bash -c "sleep 1; gimp \"$F\" &"` —
#     the real command lives inside a `-c` STRING.
#
# So the detector is now a small quote-aware shell scanner instead. Widening the
# regex was tried and rejected for creating FALSE POSITIVES, which are worse than
# the hole (they hard-fail a legitimate file-seeding task), and the scanner is what
# lets the three decoy shapes in the corpus stay negative for a STRUCTURAL reason
# rather than a lucky anchor:
#   * `bash -c "exec -a 'libreoffice --writer …' sleep infinity" &` — `exec -a`
#     supplies a FAKE argv[0]; the real command is `sleep`.
#   * `code "$WORKSPACE" &` inside an instructions HERE-DOC — that is data, not a
#     command, so here-doc bodies are stripped first.
#   * `STUB_VLC=/usr/local/bin/vlc` + a `sleep 3600` here-doc + `chmod +x` +
#     `nohup "$STUB_VLC" … &` — a stub the script wrote itself, so a path the
#     script makes executable is never treated as the real app.
# --------------------------------------------------------------------------- #

#: `<<TAG` / `<<-TAG` / `<<'TAG'` heredoc opener.
_HEREDOC_RE = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_]\w*)\2")

#: A path the SCRIPT makes executable is a stub it just wrote, not the real app.
_SELF_CHMOD_RE = re.compile(r"chmod\s+(?:-\w+\s+)*\+x\s+(\S+)")

#: A GUI binary named where a COMMAND NAME can legally appear — after a lookup
#: builtin or as a path. Prose that merely MENTIONS an app must not enter the
#: variable table (`MSG="Open libreoffice yourself"; "$MSG" &` is not a launch).
_NAME_CTX_RE = re.compile(
    r"(?:(?:command\s+-[vVp]+|which|type\s+-p|type|whereis)\s+|/)([\w.-]+)"
)

#: An assignment at COMMAND-WORD position — not only at the start of a line, so
#: `if command -v libreoffice; then LO="libreoffice"` and
#: `if CHROME_BIN="$(find_chrome)"; then` are both seen.
_ASSIGN_RE = re.compile(
    r"(?:^|[;&|(]|\bif\b|\bthen\b|\bdo\b|\belse\b)[ \t]*"
    r"(?:export[ \t]+|local[ \t]+|declare[ \t]+-\w+[ \t]+)?"
    r"([A-Za-z_]\w*)=([^\n;&|]*)",
    re.MULTILINE,
)

#: Places where BARE words are command names rather than prose: a `for x in …`
#: list, an array literal `TERMINALS=( … )`, and the arguments of a command
#: substitution `$(choose_cmd google-chrome chromium …)`. `MSG="Open libreoffice
#: yourself"` is none of these, which is what keeps prose out of the table.
_WORD_LIST_RE = re.compile(r"(?:\bin[ \t]|=\(|\$\()([^\n)]*)")

_FUNC_RE = re.compile(r"(?m)^[ \t]*(?:function[ \t]+)?([A-Za-z_][\w-]*)[ \t]*\(\)[ \t]*\{")
_VAR_REF_RE = re.compile(r"^\$\{?([A-Za-z_]\w*)\}?$")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")
_DURATION_RE = re.compile(r"[\d.]+[smhd]?\Z")

#: Words that are syntax, not a command name.
_KEYWORDS = frozenset({"then", "do", "else", "elif", "if", "while", "until",
                       "!", "{", "}", "("})
#: Wrappers that exec the rest of the line unchanged.
_TRANSPARENT = frozenset({"nohup", "time", "stdbuf", "ionice", "nice", "xvfb-run"})
#: Heads that consume an app NAME without starting it.
_NOT_A_LAUNCH = frozenset({"which", "type", "whereis", "pkill", "pgrep", "killall",
                           "echo", "printf", "grep", "test", "[", "cat", "sed", "awk"})
#: Wrappers that take the real command as a `-c` STRING.
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "su", "runuser"})


class _Command:
    """One simple command plus the job-control state of the list it belongs to."""

    __slots__ = ("words", "background", "substituted")

    def __init__(self, words: list[str], substituted: bool) -> None:
        self.words = words
        self.background = False
        self.substituted = substituted


def _heredoc_tags(line: str) -> list[tuple[str, bool]]:
    """`<<TAG` openers on this line that are OUTSIDE quotes (`<<<` excluded)."""
    tags: list[tuple[str, bool]] = []
    index, end, quote = 0, len(line), ""
    while index < end:
        char = line[index]
        if quote:
            if char == "\\" and quote == '"':
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in "'\"":
            quote = char
        elif char == "\\":
            index += 2
            continue
        elif line.startswith("<<<", index):
            index += 3
            continue
        elif line.startswith("<<", index):
            match = _HEREDOC_RE.match(line, index)
            if match:
                tags.append((match.group(3), bool(match.group(1))))
                index = match.end()
                continue
            index += 2
            continue
        index += 1
    return tags


def _strip_heredocs(source: str) -> str:
    """Drop heredoc BODIES — they are data (instructions, python, config), not code.

    Only when the terminator is actually present: a mis-detected tag must not
    swallow the rest of the script (and hide a launch that follows it).
    """
    kept: list[str] = []
    lines = source.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        kept.append(lines[index])
        tags = _heredoc_tags(lines[index])
        index += 1
        for tag, dash in tags:
            scan = index
            while scan < len(lines):
                candidate = lines[scan].strip() if dash else lines[scan].rstrip("\r\n")
                if candidate == tag:
                    break
                scan += 1
            if scan < len(lines):
                index = scan + 1
    return "".join(kept)


def _end_of_substitution(source: str, start: int, opener: int) -> int:
    """Index just past a `$( … )` / backtick substitution, quote- and paren-aware."""
    end = len(source)
    if opener == 1:
        index = start + 1
        while index < end:
            if source[index] == "\\":
                index += 2
                continue
            if source[index] == "`":
                return index + 1
            index += 1
        return end
    depth, index = 1, start + 2
    while index < end and depth:
        char = source[index]
        if char == "\\":
            index += 2
            continue
        if char == "'":
            close = source.find("'", index + 1)
            index = end if close < 0 else close + 1
            continue
        if char == '"':
            close = index + 1
            while close < end and source[close] != '"':
                close += 2 if source[close] == "\\" else 1
            index = close + 1
            continue
        depth += (char == "(") - (char == ")")
        index += 1
    return index


def _split_commands(source: str, substituted: bool = False) -> list[_Command]:
    """Quote-aware split into simple commands, recording job control.

    `a || b &` backgrounds the whole AND-OR list and `( a; b ) &` the whole group,
    which is why backgrounding is applied to a RANGE rather than to one command.
    `$( … )` bodies are scanned too but flagged, so a command substitution
    (`$(command -v libreoffice)`) is never mistaken for a launch site.
    """
    commands: list[_Command] = []
    words: list[str] = []
    word, started = "", False
    list_start = 0
    group_stack: list[int] = []
    last_group: tuple[int, int] | None = None
    index, end = 0, len(source)

    def flush_word() -> None:
        nonlocal word, started
        if started:
            words.append(word)
        word, started = "", False

    def flush_command(background: bool, chain: bool = False) -> None:
        nonlocal words, list_start, last_group
        flush_word()
        # Measured BEFORE appending, so `( a; b ) >/dev/null 2>&1 &` still ties the
        # `&` to the group: the redirections after `)` land in a command of their
        # own, which would otherwise push the group out of the backgrounded range.
        before = len(commands)
        if words:
            commands.append(_Command(words, substituted))
        words = []
        if background:
            begin = list_start
            if last_group is not None and last_group[1] == before:
                begin = min(begin, last_group[0])
            for command in commands[begin:]:
                command.background = True
        if not chain:
            list_start = len(commands)
            if not background:
                last_group = None

    while index < end:
        char = source[index]
        if char == "\\":
            if index + 1 < end and source[index + 1] == "\n":
                index += 2                      # line continuation: vanishes
                continue
            if index + 1 < end:
                word += source[index + 1]
                started = True
                index += 2
                continue
            index += 1
            continue
        if char == "'":
            close = source.find("'", index + 1)
            close = end if close < 0 else close
            word += source[index + 1:close]
            started = True
            index = close + 1
            continue
        if char == '"':
            close, buffer = index + 1, ""
            while close < end and source[close] != '"':
                if source[close] == "\\" and close + 1 < end:
                    buffer += source[close + 1]
                    close += 2
                    continue
                buffer += source[close]
                close += 1
            word += buffer
            started = True
            index = close + 1
            continue
        if source.startswith("$(", index) or char == "`":
            opener = 2 if char == "$" else 1
            close = _end_of_substitution(source, index, opener)
            commands.extend(_split_commands(source[index + opener:close - 1], True))
            word += "\x00"
            started = True
            index = close
            continue
        if char == "#" and not started and not word:
            newline = source.find("\n", index)
            index = end if newline < 0 else newline
            continue
        if char in " \t":
            flush_word()
            index += 1
            continue
        if source.startswith(";;", index):
            flush_command(False)
            index += 2
            continue
        if source.startswith("&&", index) or source.startswith("||", index):
            flush_command(False, chain=True)
            index += 2
            continue
        if char == "|":
            flush_command(False, chain=True)
            index += 1
            continue
        if char == "&":
            if source.startswith("&>", index):   # a redirection, not job control
                flush_word()
                word, started = "&>", True
                index += 2
                continue
            if word.endswith(">"):               # `>&`, `2>&1`
                word += char
                started = True
                index += 1
                continue
            flush_command(True)
            index += 1
            continue
        if char in ";\n":
            flush_command(False)
            index += 1
            continue
        if char in "({":
            if started:                          # `${…}`, brace expansion, `f(`
                word += char
                index += 1
                continue
            flush_command(False)
            group_stack.append(len(commands))
            index += 1
            continue
        if char in ")}":
            if started:
                word += char
                index += 1
                continue
            flush_command(False)
            if group_stack:
                last_group = (group_stack.pop(), len(commands))
            index += 1
            continue
        word += char
        started = True
        index += 1
    flush_command(False)
    return commands


def _bare_gui(words: str) -> set[str]:
    return {w.strip("\"'").rsplit("/", 1)[-1] for w in words.split()} & _GUI_BINARIES


def _gui_names_in(fragment: str) -> set[str]:
    """GUI binaries named as COMMAND NAMES anywhere in a fragment of shell."""
    names = {m.group(1) for m in _NAME_CTX_RE.finditer(fragment)} & _GUI_BINARIES
    for match in _WORD_LIST_RE.finditer(fragment):
        names |= _bare_gui(match.group(1))
    stripped = fragment.strip()
    if stripped.startswith("("):                 # `TERMINALS=( gnome-terminal … )`
        names |= _bare_gui(stripped.strip("()"))
    if stripped:
        if stripped[0] in "\"'":
            close = stripped.find(stripped[0], 1)
            first = stripped[1:close] if close > 0 else stripped[1:]
        else:
            first = stripped.split()[0]
        if first.rsplit("/", 1)[-1] in _GUI_BINARIES:
            names.add(first.rsplit("/", 1)[-1])
    return names


def _variable_table(source: str) -> dict[str, set[str]]:
    """NAME -> the GUI binaries NAME could expand to, following helper functions."""
    functions: dict[str, str] = {}
    for match in _FUNC_RE.finditer(source):
        depth, index = 1, match.end()
        while index < len(source) and depth:
            depth += (source[index] == "{") - (source[index] == "}")
            index += 1
        functions[match.group(1)] = source[match.end():index]

    table: dict[str, set[str]] = {}

    def record(name: str, names: set[str]) -> None:
        if names:
            table.setdefault(name, set()).update(names)

    def referenced(fragment: str) -> set[str]:
        return set().union(*(
            table.get(m.group(1), set())
            for m in re.finditer(r"\$\{?([A-Za-z_]\w*)", fragment)
        ), set())

    # Two passes: a candidate list reaches a head through up to two hops
    # (`TERMINALS=( … )` -> `for term in "${TERMINALS[@]}"` -> `"$term" … &`, or
    # `for c in google-chrome …` -> `CHROME_BIN="$(command -v "$c")"` -> `"$CHROME_BIN" … &`),
    # and the two hops can come in either order in the source.
    for _ in range(2):
        # `for term in "${TERMINALS[@]}"; do … "$term" … &` — a loop variable is the
        # other way a head becomes a variable; its candidates are the word list.
        for match in re.finditer(r"\bfor[ \t]+([A-Za-z_]\w*)[ \t]+in[ \t]+([^\n;]*)", source):
            record(match.group(1), _bare_gui(match.group(2)) | referenced(match.group(2)))
        for match in _ASSIGN_RE.finditer(source):
            name, value = match.group(1), match.group(2)
            if value.lstrip().startswith("(") and ")" not in value:
                # A multi-LINE array literal; `_ASSIGN_RE` stops at the newline.
                close = source.find(")", match.end(2))
                value = source[match.start(2):close + 1] if close > 0 else value
            names = _gui_names_in(value) | referenced(value)
            for function, body in functions.items():
                if re.search(rf"\b{re.escape(function)}\b", value):
                    names |= _gui_names_in(body) | referenced(body)
            record(name, names)
    return table


def _real_commands(
    words: list[str],
    background: bool,
    depth: int = 0,
) -> list[tuple[list[str], bool]]:
    """Peel keywords/env/wrappers off a command, recursing through `-c` strings.

    Returns ``[(argv, backgrounded), …]``; empty means "this starts nothing".
    """
    if depth > 3:
        return []
    forced = False
    index = 0
    while index < len(words):
        word = words[index]
        base = word.rsplit("/", 1)[-1]
        if word in _KEYWORDS or _ENV_ASSIGN_RE.match(word) or base in _TRANSPARENT:
            index += 1
            continue
        if base == "setsid":
            index += 1
            while index < len(words) and words[index].startswith("-"):
                forced = forced or words[index] in ("-f", "--fork")
                index += 1
            continue
        if base in ("env", "sudo"):
            index += 1
            while index < len(words) and (
                words[index].startswith("-") or _ENV_ASSIGN_RE.match(words[index])
            ):
                if words[index] in ("-u", "--user"):
                    index += 1
                index += 1
            continue
        if base == "command":
            index += 1
            while index < len(words) and words[index].startswith("-"):
                if words[index] in ("-v", "-V"):
                    return []                    # a LOOKUP, not a launch
                index += 1
            continue
        if base in _NOT_A_LAUNCH:
            return []
        if base == "exec":
            index += 1
            while index < len(words) and words[index].startswith("-"):
                if words[index] == "-a" and index + 1 < len(words):
                    index += 1                   # the FAKE argv[0] — never the command
                index += 1
            continue
        if base == "timeout":
            index += 1
            while index < len(words) and (
                words[index].startswith("-") or _DURATION_RE.match(words[index])
            ):
                index += 1
            continue
        if base in _SHELLS:
            for probe in range(index + 1, len(words)):
                if words[probe] == "-c" and probe + 1 < len(words):
                    inner: list[tuple[list[str], bool]] = []
                    for command in _split_commands(words[probe + 1]):
                        if command.substituted:
                            continue
                        inner += _real_commands(
                            command.words,
                            background or forced or command.background,
                            depth + 1,
                        )
                    return inner
            if base in ("su", "runuser"):
                return []
            return [(words[index:], background or forced)]
        break
    rest = words[index:]
    return [(rest, background or forced)] if rest else []


def _backgrounded_gui_launches(source: str) -> list[list[str]]:
    """Every GUI app this shell script starts in the background, as argv lists."""
    script = _strip_heredocs(source)
    variables = _variable_table(script)
    stubs = {m.group(1).strip("\"'") for m in _SELF_CHMOD_RE.finditer(source)}
    launches: list[list[str]] = []
    for command in _split_commands(script):
        if command.substituted:
            continue
        for argv, background in _real_commands(list(command.words), command.background):
            if not background or argv[0] in stubs:
                continue
            head = argv[0]
            reference = _VAR_REF_RE.match(head)
            if reference:
                resolved = variables.get(reference.group(1), set())
            else:
                base = head.rsplit("/", 1)[-1]
                resolved = {base} if base in _GUI_BINARIES else set()
            if resolved and not _WINDOWLESS.intersection(argv[1:]):
                launches.append(argv)
    return launches


def _script_launches_gui(source: str) -> bool:
    """True when the setup script starts a GUI app of its own accord."""
    return bool(_GUI_LAUNCH_RE.search(source)) or bool(
        _backgrounded_gui_launches(source)
    )


async def _apply_open_steps(computer, task_json: Path) -> bool:
    """Execute task.json open steps that are not handled by script setup."""
    opened = False
    for remote_path in _open_paths(task_json):
        opener = _opener_for(remote_path)
        result = await _run_task_command(
            computer,
            f"DISPLAY=:0 setsid {opener} {shlex.quote(remote_path)} >/dev/null 2>&1 &",
            f"open {remote_path}",
            error_type=CuaGymTaskError,
        )
        _raise_for_command(
            result,
            f"open {remote_path}",
            error_type=CuaGymTaskError,
        )
        opened = True
    return opened


def _raise_for_command(
    result,
    phase: str,
    *,
    error_type: type[CuaGymTaskError] = CuaGymTaskError,
) -> None:
    if getattr(result, "returncode", 0) == 0:
        return
    detail = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()
    raise error_type(
        f"lite.cuagym desktop {phase} failed with rc={result.returncode}: {detail[-500:]}",
        phase="setup" if phase != "reward" else "reward",
        kind="timeout" if result.returncode in {124, 137} else "command_failed",
        returncode=result.returncode,
    )


async def _run_task_command(
    computer,
    command: str,
    phase: str,
    *,
    error_type: type[CuaGymTaskError] = CuaGymTaskError,
):
    normalized_phase = "reward" if phase == "reward" else "setup"
    try:
        return await computer.interface.run_command(
            command,
            timeout=_COMMAND_TIMEOUT,
        )
    except (TimeoutError, ExecStdioError) as exc:
        if isinstance(exc, TimeoutError) or "timeoutexpired" in str(exc).lower():
            raise error_type(
                f"lite.cuagym desktop {phase} exceeded {_COMMAND_TIMEOUT:.0f}s",
                phase=normalized_phase,
                kind="timeout",
            ) from exc
        raise


async def _run_postconfig(computer, steps: list[dict]) -> None:
    for step in steps:
        kind = step.get("type")
        params = step.get("parameters") or {}
        if kind == "sleep":
            try:
                seconds = float(params.get("seconds", 0))
            except (TypeError, ValueError) as exc:
                _invalid_bundle(
                    f"invalid postconfig sleep duration: {params.get('seconds')!r}",
                    phase="setup",
                    cause=exc,
                )
            await asyncio.sleep(seconds)
            continue
        if kind != "execute":
            _invalid_bundle(
                f"unsupported postconfig type: {kind!r}",
                phase="setup",
            )
        command = params.get("command")
        if isinstance(command, list):
            command = shlex.join(str(part) for part in command)
        if not isinstance(command, str) or not command:
            _invalid_bundle(
                "postconfig execute is missing a command",
                phase="setup",
            )
        # The postconfig command runs as the desktop user (soffice/gimp/code/… run
        # as the user by construction).
        wrapped = f"cd /home/user && DISPLAY=:0 HOME=/home/user {command}"
        result = await _run_task_command(
            computer, wrapped, "postconfig",
        )
        _raise_for_command(result, "postconfig")


async def setup_fn(task, computer) -> None:
    """Set up a task in the container. Two kinds:
    - script (py/sh): run the official initial_setup script;
    - document-seed (pptx/docx/xlsx): upload the seed doc to its dest (from the
      task.json download step) and open it with LibreOffice."""
    meta = task.metadata
    await bridge_display(computer)
    # Setup runs as the desktop user, so its files are user-owned by construction —
    # no ownership-handover step, and git-commit rewards run as the user (no
    # dubious-ownership guard to work around).
    existing_windows = await window_ids(computer)
    try:
        kind = meta.get("setup_kind", "py")
        setup_path = Path(meta["setup"])
        task_json = setup_path.parent / "task.json"
        if kind in _DOC_KINDS:
            dest = meta["doc_dest"]
            downloads = _local_downloads(task_json)
            payloads = (
                [
                    (source_path.read_bytes(), remote_path)
                    for source_path, remote_path in downloads
                ]
                if downloads
                else [(setup_path.read_bytes(), dest)]
            )
        else:
            src = setup_path.read_text()
    except CuaGymTaskError:
        raise
    except (KeyError, OSError, UnicodeError, TypeError, ValueError) as exc:
        _invalid_bundle(
            f"{type(exc).__name__}: {exc}",
            phase="setup",
            cause=exc,
        )

    if kind in _DOC_KINDS:
        for payload, remote_path in payloads:
            await write_bytes(computer, payload, remote_path)
        result = await _run_task_command(
            computer,
            f"DISPLAY=:0 setsid soffice {shlex.quote(dest)} >/dev/null 2>&1 &",
            "document launch",
            error_type=CuaGymTaskError,
        )
        _raise_for_command(
            result,
            "document launch",
            error_type=CuaGymTaskError,
        )
        if not await wait_for_new_window(computer, existing_windows):
            raise CuaGymTaskError(
                "lite.cuagym desktop document window did not appear",
                phase="setup",
                kind="no_task_window",
        )
        return

    ext = "sh" if kind == "sh" else "py"
    remote = f"/home/user/initial_setup.{ext}"
    await write_text(computer, src, remote)
    # The setup script runs as the desktop user — bash for .sh, or the env interpreter
    # for .py (gi/uno → UNO_PY, else ENV_PY).
    interp = (
        "bash"
        if ext == "sh"
        else _python_command_for_source(src)
    )
    result = await _run_task_command(
        computer,
        f"cd /home/user && DISPLAY=:0 {interp} {remote}",
        "setup",
        error_type=CuaGymTaskError,
    )
    _raise_for_command(result, "setup", error_type=CuaGymTaskError)
    opened = await _apply_open_steps(computer, setup_path.parent / "task.json")
    # ...or the setup SCRIPT launched something itself. 6921 of the 8704 script-kind
    # bundles define/call `launch_gui(...)` (a non-blocking Popen) and another 929
    # background the binary straight from .sh, even though task.json declares no
    # `open` step — for those a window IS expected and its absence is a real harness
    # failure, so they must keep the hard gate.
    expects_window = opened or _script_launches_gui(src)
    # Setup ran as the desktop user → all task files are user-owned by construction;
    # no ownership handover.
    # Wait for the launched GUI app's window before the first screenshot -- but only
    # demand one when WE launched something. A class of bundles is pure file-seeding:
    # the setup script creates the CSVs/workspace and the instruction then tells the
    # AGENT to "open LibreOffice Calc"; task.json has no `open` step, so nothing was
    # ever going to raise a window. Hard-failing those turned a perfectly good task
    # into a reset error -- measured at 283 of 8704 script-kind bundles (138
    # multi_apps, 133 pdf, 11 vscode, 1 other).
    # When `opened` is true the opener is OURS, so a missing window is a
    # real harness failure and still raises; the _DOC_KINDS branch above keeps its
    # hard gate for the same reason.
    if not await wait_for_new_window(computer, existing_windows):
        if expects_window:
            raise CuaGymTaskError(
                "lite.cuagym desktop task window did not appear",
                phase="setup",
                kind="no_task_window",
            )
        logger.warning(
            "lite.cuagym: no new window after setup script %s; continuing -- the "
            "bundle declares no `open` step and its setup launches no GUI, so this "
            "is a file-seeding task whose app the agent is expected to open itself.",
            remote,
        )


async def evaluate_final_fn(task, computer) -> tuple[float, dict[str, Any]]:
    """Run the task's official reward.py inside the container, parse ``REWARD: x``.

    Returns ``(reward, eval_info)`` UNCONDITIONALLY, unlike lite.osworld /
    lite.scalecua, which take a 4th ``debug`` arg and gate the dict on it. Two
    reasons to deviate rather than grow the signature: ``SandboxBaseEnv._call_fn``
    passes only as many args as the callee declares, so this 2-arg form simply never
    receives ``debug``; and the payload is not debug telemetry. lite.osworld's reward
    is binary, so its dict only explains a failure — a CUA-Gym reward is a weighted
    rubric, so the component lines ARE the reading of a fractional score and every
    scored rollout wants them. The payload is bounded at 40 lines.
    """
    await _run_postconfig(computer, task.metadata.get("postconfig", []))
    try:
        src = Path(task.metadata["reward"]).read_text()
    except (KeyError, OSError, UnicodeError, TypeError, ValueError) as exc:
        _invalid_bundle(
            f"{type(exc).__name__}: {exc}",
            phase="reward",
            cause=exc,
        )
    # 2 of the 9138 live desktop tasks import `/tmp/reward_judge.py`; both
    # `sys.path.insert(0, '/tmp')` first, so /tmp is the right (and upstream's) home
    # even though the reward script itself lives in /home/user.
    if bundle_needs_judge(src):
        await install_reward_judge(computer)
    await write_text(computer, src, "/home/user/reward.py")
    # The reward runs as the desktop user under the env interpreter
    # (_python_command_for_source: gi/uno → UNO_PY, else ENV_PY py3.12); it reads the
    # user-owned task files + the live session (dconf via the file backend under
    # HOME=/home/user) directly, and any session/GUI tool the reward shells out to
    # runs in the user session. Missing reward deps are added to /opt/env/venv, never
    # chased on the frozen system python.
    interp = _python_command_for_source(src)
    # PYTHONIOENCODING: these bundles are LLM-authored, and at least one emits a
    # surrogate PAIR as two lone `\uXXXX` escapes in a `print()` banner
    # (`🎫`). Python compiles those to lone surrogates and, writing to the
    # redirected stdout file under LANG=en_HK.UTF-8, refuses them with
    # UnicodeEncodeError -- rc=1, and a reward that would otherwise have printed its
    # `REWARD:` line becomes a hard task error. `backslashreplace` degrades the
    # unprintable banner instead of the whole evaluation.
    r = await _run_task_command(
        computer,
        "cd /home/user && DISPLAY=:0 PYTHONIOENCODING=utf-8:backslashreplace "
        f"{interp} /home/user/reward.py",
        "reward",
    )
    _raise_for_command(r, "reward")
    out = r.stdout or ""
    if not REWARD_RE.search(out):
        raise CuaGymTaskError(
            f"lite.cuagym desktop reward produced no REWARD sentinel: {out[-500:]}",
            phase="reward",
            kind="no_reward",
        )
    try:
        # Return the reward WITH the script's own output. These bundles are
        # weighted rubrics (mean 4.4 components) that print per-component
        # `PASS:`/`FAIL:` diagnostics before the sentinel, and that text was being
        # thrown away — `info` carried only `executed_actions`, so working out WHY
        # a rollout scored 0.4 meant re-deriving the rubric from the bundle and
        # subset-summing the deficit. `SandboxBaseEnv.step` unpacks a
        # `(reward, eval_info)` tuple into `info["eval"]`, which the logger
        # persists to `04_results.json`.
        # Stored as LINES, not one blob, so `04_results.json` stays readable and
        # bounded even when a rubric prints a long diagnostic banner.
        return parse_reward(out), {"reward_stdout": out.splitlines()[-40:]}
    except ValueError as exc:
        raise CuaGymTaskError(
            f"lite.cuagym desktop reward is invalid: {exc}",
            phase="reward",
            kind="invalid_reward",
        ) from exc
