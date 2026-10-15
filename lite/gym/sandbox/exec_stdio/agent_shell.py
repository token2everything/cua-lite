"""Agent-facing persistent bash shell for sandbox-family envs.

ONE long-lived ``docker exec -i -u <agent_user> <container> bash -l`` per
container, with every ``run()`` written to its stdin and framed by a
per-command random-nonce sentinel. The command runs in a brace group ``{ ... }``
in the SAME shell (never a subshell), which is what makes ``cd``/``export``
persist across calls — the headline requirement of the bash tool.

Run a hermetic smoke locally (no docker, host bash):
    uv run python -c "import asyncio
    from lite.gym.sandbox.exec_stdio.agent_shell import AgentShellSession
    async def m():
        s = AgentShellSession('x', _local=True)
        await s.run('cd /tmp'); print((await s.run('pwd')).output); await s.close()
    asyncio.run(m())"
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from dataclasses import dataclass
from typing import Final

from lite.gym.sandbox.exec_stdio.session import DockerExecSession

#: Per-command wall clock for the agent bash tool, in seconds. Under
#: ``StepTimeoutWrapper`` the outer per-step budget may fire first; either way
#: :meth:`AgentShellSession.run` reaps the in-container command, so which one wins
#: is not load-bearing. This is the bound for callers with no outer budget; a
#: caller with its own passes ``run(..., timeout=...)``.
DEFAULT_AGENT_SHELL_TIMEOUT: Final[float] = 120.0
DEFAULT_OUTPUT_LIMIT: Final[int] = 1_000_000
#: Budget for the spawn handshake (login profile + drain sentinel). Independent
#: of the per-command timeout: a shell that cannot say hello is broken, not slow.
DEFAULT_SPAWN_TIMEOUT: Final[float] = 30.0
#: Read granularity. ``read(n)`` (NOT ``readline()``) is what makes an unbounded
#: single line truncatable instead of fatal — see :meth:`_read_until`.
_READ_CHUNK: Final[int] = 64 * 1024
#: A pending line longer than this is flushed as output mid-line. Vastly larger
#: than any sentinel line, so flushing can never eat a marker.
_PARTIAL_LINE_FLUSH: Final[int] = 64 * 1024
#: PATH prefix owned by the ENV exec-stdio side (server.py prepends
#: ``/opt/env/venv/bin:/opt/env/bin``). It must never reach the agent shell.
_ENV_PATH_PREFIX: Final[str] = "/opt/env"

#: Spawn argv for the container path. ``setsid`` makes the login shell a
#: session/process-group leader so ``kill -TERM -<pgid>`` reaps its whole tree;
#: ``-w`` keeps the wrapper alive (and the stdin pipe held) for the shell's
#: lifetime. ``command -v`` keeps an image without util-linux working — it just
#: degrades the reap to a single-pid kill (``_reap`` checks pgid == pid).
_SETSID_BOOTSTRAP: Final[str] = (
    "command -v setsid >/dev/null 2>&1 && exec setsid -w bash -l; exec bash -l"
)

#: Drop every ``/opt/env*`` component from the agent shell's PATH. ``bash -l``
#: sources /etc/profile + the user's rc, either of which can re-introduce the ENV
#: exec-stdio PATH (server.py:67) and undo the agent/env separation; the login
#: shell is kept (a real terminal is what the agent expects) and the leak is
#: scrubbed here, then ASSERTED host-side in :meth:`_after_spawn`.
_PATH_SCRUB: Final[str] = (
    '__lite_p=""; __lite_ifs=$IFS; IFS=:; for __lite_e in $PATH; do '
    f'case "$__lite_e" in {_ENV_PATH_PREFIX}*) ;; '
    '*) __lite_p="${__lite_p:+$__lite_p:}$__lite_e";; esac; done; '
    'IFS=$__lite_ifs; PATH="$__lite_p"; export PATH; '
    "unset __lite_p __lite_e __lite_ifs"
)


@dataclass(frozen=True)
class AgentShellResult:
    """Output and exit status from one persistent-shell command."""

    output: str
    returncode: int


class AgentShellTimeout(TimeoutError):
    """A command timed out after producing partial output."""

    def __init__(self, message: str, *, partial: str = "") -> None:
        super().__init__(message)
        self.partial = partial


class AgentShellDied(RuntimeError):
    """The persistent shell exited before the command sentinel arrived."""

    def __init__(self, message: str, *, partial: str = "") -> None:
        super().__init__(message)
        self.partial = partial


class AgentShellSession(DockerExecSession):
    """Persistent ``bash`` session used only for the agent-facing bash tool."""

    def __init__(
        self,
        name: str,
        *,
        exec_user: str = "user",
        timeout: float = DEFAULT_AGENT_SHELL_TIMEOUT,
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
        _local: bool = False,
    ) -> None:
        super().__init__(name, timeout=timeout, exec_user=exec_user, _local=_local)
        self._output_limit = output_limit
        #: Strong reference to an in-flight cancellation-proof cleanup, so a
        #: detached ``close()`` cannot be garbage-collected mid-reap (asyncio
        #: keeps only a weak reference to running tasks).
        self._cleanup_task: asyncio.Future | None = None
        #: The scrubbed login PATH the shell reported at spawn (diagnostics).
        self.login_path: str = ""
        #: Best-effort decoded output for the in-flight command if framing fails.
        self._partial = ""

    async def start(self) -> None:
        """Start the persistent shell and verify its scrubbed login PATH."""
        if self._alive():
            return
        if self._local:
            argv = ["bash", "--noprofile", "--norc"]
        else:
            argv = ["docker", "exec", "-i", "-u", self._exec_user, self._name,
                    "sh", "-c", _SETSID_BOOTSTRAP]
        await self._spawn_core(
            argv,
            limit=self._output_limit,
            # Belt-and-braces: the shell itself merges fd 2 into fd 1 in
            # _after_spawn (``exec 2>&1``), so the container writes on ONE
            # stream and stderr can no longer arrive after the sentinel. This
            # only catches what the docker CLI itself reports.
            stderr=asyncio.subprocess.STDOUT,
            # Local mode only: give the shell its own process group so the reap
            # can killpg it without ever signalling the test runner's group.
            start_new_session=self._local,
        )

    async def _after_spawn(self) -> None:
        """Merge stderr, land in $HOME, scrub PATH, and DRAIN the login banner.

        Without the drain sentinel the /etc/profile + MOTD bytes of ``bash -l``
        would be read as the first command's output.
        """
        assert self._proc is not None and self._proc.stdin is not None
        ready = f"__LITE_AGENT_SHELL_READY_{secrets.token_hex(12)}__"
        setup = (
            "exec 2>&1\n"
            'cd "$HOME"\n'
            f"{_PATH_SCRUB}\n"
            # pgid via ps: the shell cannot read its own pgid without it. An
            # image without procps reports "" → reap degrades to a pid kill.
            f"printf '\\n{ready}:%s:%s:%s\\n' \"$$\" "
            "\"$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')\" \"$PATH\"\n"
        )
        self._proc.stdin.write(setup.encode("utf-8"))
        await self._proc.stdin.drain()
        try:
            _banner, fields = await asyncio.wait_for(
                self._read_until(ready, 3), timeout=DEFAULT_SPAWN_TIMEOUT)
        except BaseException:
            # Same rule as run(): a handshake that did not complete leaves a
            # live shell behind, and CancelledError is not an Exception.
            await self._close_uncancellable()
            raise
        pid_s, pgid_s, path = fields
        self._pid = int(pid_s) if pid_s.strip().isdigit() else None
        self._pgid = int(pgid_s) if pgid_s.strip().isdigit() else None
        self.login_path = path
        if any(p.startswith(_ENV_PATH_PREFIX) for p in path.split(":")):
            await self.close()
            raise RuntimeError(
                f"agent shell PATH for {self._name!r} still contains "
                f"{_ENV_PATH_PREFIX} after the login scrub ({path!r}) — the env "
                "exec-stdio toolchain is leaking onto the agent's PATH"
            )
    async def run(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> AgentShellResult:
        """Run one command through the persistent shell with sentinel framing."""
        if not isinstance(command, str):
            raise TypeError("bash command must be a string")
        async with self._lock:
            if not self._alive():
                await self.start()
            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            self._partial = ""
            sentinel = f"__LITE_AGENT_SHELL_DONE_{secrets.token_hex(12)}__"
            runner = f"__lite_agent_run_{secrets.token_hex(8)}"
            status_var = f"__lite_status_{secrets.token_hex(8)}"
            out_fd_var = f"__lite_o_{secrets.token_hex(8)}"
            err_fd_var = f"__lite_e_{secrets.token_hex(8)}"
            # fd save/restore around the command so ``exec 1>&-`` (or any other
            # fd stomp) cannot break the sentinel frame. The saved fds are
            # allocated by bash ({var}>&1 picks a free fd >= 10) instead of the
            # fixed 3/4 a command is far more likely to reuse itself.
            # The command runs in a function in THIS shell, not a subshell:
            # cd/export/function definitions persist, while ``local -`` restores
            # ``set`` options without relying on post-command builtins a command
            # can shadow. The tradeoff is bash function context for rare snippets
            # that inspect FUNCNAME or use ``return``/``local`` at top level.
            # ``shopt`` state is intentionally persistent like function
            # definitions: bash offers no reliable same-shell restoration once a
            # command can define functions named command/builtin/enable/shopt.
            payload = (
                f"exec {{{out_fd_var}}}>&1 {{{err_fd_var}}}>&2\n"
                f"{runner}() {{\n"
                "local -\n"
                "{\n"
                f"{command}\n"
                "} </dev/null\n"
                "}\n"
                f"{runner}\n"
                f"{status_var}=$?\n"
                f"unset -f {runner}\n"
                f"exec 1>&${out_fd_var} 2>&${err_fd_var}\n"
                f"exec {{{out_fd_var}}}>&- {{{err_fd_var}}}>&-\n"
                # The leading \n is what makes the marker LINE-ANCHORED even
                # when the command's own output has no trailing newline;
                # _read_until strips exactly that one injected byte back off.
                f"printf '\\n{sentinel}:%s\\n' "
                f"\"${status_var}\"\n"
                f"unset {status_var} {out_fd_var} {err_fd_var}\n"
            )
            self._proc.stdin.write(payload.encode("utf-8"))
            await self._proc.stdin.drain()
            try:
                output, fields = await asyncio.wait_for(
                    self._read_until(sentinel, 1),
                    timeout=timeout if timeout is not None else self._timeout,
                )
                # Unpack, not a lenient parse: a partial frame must fail rather
                # than read as exit status 0.
                (status,) = fields
            except (asyncio.TimeoutError, TimeoutError) as e:
                partial = self._partial
                await self._close_uncancellable()
                raise AgentShellTimeout(
                    "agent bash shell timed out before command completed",
                    partial=partial,
                ) from e
            except BaseException:
                # EVERY exit that is not a completed frame leaves the same two-part
                # mess: the command still running inside the container (killing the
                # host ``docker exec`` client never reaches it) and the session
                # parked mid-frame, so the next run() would read this command's
                # residual output. close() -> _reap() fixes both.
                #
                # BaseException, not Exception, because cancellation is the common
                # case here (outer step timeout, env-server task cancellation,
                # client disconnect) and ``CancelledError`` is not an ``Exception``.
                await self._close_uncancellable()
                raise
            return AgentShellResult(output=output, returncode=int(status))

    async def _close_uncancellable(self) -> None:
        """Run ``close()`` (hence ``_reap()``) so cancellation cannot abort it.

        ``close()`` awaits several times (terminate → wait → docker exec kill), so a
        plain ``await self.close()`` on the cancellation path would be interrupted at
        the first of them, leaving the in-container tree alive. Shielding lets the
        cleanup finish, detached if we are cancelled again. Raises nothing: the
        caller is already unwinding a real failure and re-raises it.
        """
        task = asyncio.ensure_future(self.close())
        self._cleanup_task = task
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(task)

    async def _read_until(self, sentinel: str, nfields: int) -> tuple[str, list[str]]:
        """Collect output up to a LINE-ANCHORED sentinel frame.

        The frame is ``<sentinel>:<f1>:...:<fN>\\n``. Returns
        ``(output, [f1..fN])`` — ``[status]`` for :meth:`run`, ``[pid, pgid,
        PATH]`` for the spawn drain; the body is split at most ``nfields - 1``
        times, so the LAST field keeps any colons of its own (PATH does).

        The nonce is generated per call (:meth:`run`), never returned to the
        caller, and the command's stdin is ``/dev/null``, so a command cannot
        print this marker. A short frame is not checked here — it fails at the
        caller's unpack.

        Callers MUST emit the marker as ``printf '\\n<sentinel>:...\\n'``; the
        injected newline is stripped back off the returned output.

        Reads fixed-size CHUNKS, not lines: ``StreamReader.readline()`` raises
        ``ValueError`` on a line longer than the stream limit AND clears the
        buffer, so one ``cat /dev/zero`` would both destroy the session and eat
        the sentinel. Here an over-long line is flushed into the output budget
        like any other bytes, so it truncates and the session survives.
        """
        assert self._proc is not None and self._proc.stdout
        stdout = self._proc.stdout
        marker = f"{sentinel}:".encode("utf-8")
        buf = bytearray()
        out: list[bytes] = []
        total = 0
        truncated = False
        completed = False
        self._partial = ""
        # False while the middle of an over-long line is being flushed — the
        # marker only ever counts at a true line start.
        at_line_start = True

        def render(*, include_buffer: bool = False) -> str:
            text = b"".join(out).decode("utf-8", "replace")
            if include_buffer and buf and not truncated:
                text += bytes(buf).decode("utf-8", "replace")
            return text

        def emit(data: bytes) -> None:
            nonlocal total, truncated
            if truncated:
                return
            room = self._output_limit - total
            if len(data) <= room:
                out.append(data)
                total += len(data)
                return
            if room > 0:
                out.append(data[:room])
                total += room
            out.append(b"\n[truncated]\n")
            truncated = True

        try:
            while True:
                while True:
                    nl = buf.find(b"\n")
                    if nl >= 0:
                        line = bytes(buf[:nl + 1])
                        del buf[:nl + 1]
                        if at_line_start and line.startswith(marker):
                            text = render()
                            if not truncated and text.endswith("\n"):
                                text = text[:-1]  # the injected anchor newline
                            tail = line[len(marker):].decode("utf-8", "replace")
                            self._partial = text
                            completed = True
                            return text, tail.rstrip("\n").split(":", nfields - 1)
                        emit(line)
                        at_line_start = True
                        continue
                    if len(buf) >= _PARTIAL_LINE_FLUSH:
                        # No newline in 64 KiB: this cannot be the (short) sentinel
                        # line, so flush it and stop marker-matching until the next
                        # real line start.
                        emit(bytes(buf))
                        buf.clear()
                        at_line_start = False
                    break
                data = await stdout.read(_READ_CHUNK)
                if not data:
                    raise AgentShellDied(
                        "agent bash shell exited before command completed",
                        partial=render(include_buffer=True),
                    )
                buf += data
        finally:
            if not completed:
                self._partial = render(include_buffer=True)


__all__ = [
    "AgentShellDied",
    "AgentShellResult",
    "AgentShellSession",
    "AgentShellTimeout",
]
