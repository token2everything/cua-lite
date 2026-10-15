"""Shared docker-exec process lifecycle for exec_stdio package internals."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from typing import Final

#: SIGTERM → SIGKILL grace when reaping the in-container process group.
_REAP_GRACE_S: Final[float] = 0.5
#: Host-side wait for one reap ``docker exec``.
_REAP_TIMEOUT_S: Final[float] = 15.0


class DockerExecSession:
    """Spawn/close core shared by the two persistent ``docker exec -i`` channels.

    Narrow by design: identity, one subprocess, one lock, spawn/close mechanics,
    and the in-container reap. Protocol details stay with the concrete channels
    (``client.ExecStdioSession`` owns JSON-lines stdio, ``agent_shell`` owns bash
    framing). :meth:`_reap` is driven off ``self._pid``/``self._pgid``; a
    subclass's only job is to LEARN that identity at spawn time.
    """

    def __init__(
        self,
        name: str,
        *,
        timeout: float,
        exec_user: str = "user",
        _local: bool = False,
    ) -> None:
        self._name = name
        self._timeout = timeout
        self._exec_user = exec_user
        self._local = _local
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        #: In-container pid/pgid of the process this session execs, learned by the
        #: subclass's spawn handshake -- what :meth:`_reap` targets. ``None`` means
        #: identity unknown, and the reap is then skipped rather than guessed.
        self._pid: int | None = None
        self._pgid: int | None = None
        #: Identities of earlier generations, parked by :meth:`_retire`. A respawn
        #: replaces the host client WITHOUT killing what the old one left running,
        #: and the fresh handshake is about to overwrite ``_pid``/``_pgid`` — so the
        #: old identity moves here and teardown still reaps it. Retired, not lost.
        self._retired: list[tuple[int, int | None]] = []

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def _spawn_core(
        self,
        argv: list[str],
        *,
        limit: int,
        stderr: int,
        start_new_session: bool = False,
    ) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr,
            limit=limit,
            start_new_session=start_new_session,
        )
        await self._after_spawn()

    async def _after_spawn(self) -> None:
        """Post-spawn handshake on the freshly opened pipe. No-op by default."""

    async def _reap(self) -> None:
        """Kill the IN-CONTAINER process trees the host client does not own.

        Killing the host ``docker exec`` client does not signal anything inside
        the container, so without this every timed-out command leaves the
        in-container process (and its children) behind for the container's
        lifetime.

        Covers EVERY generation, current plus ``_retired`` — a respawn hands its
        old identity over instead of killing it, and this is where that debt is
        settled. ONE pair of signals for all of them (``kill`` takes several
        targets), so teardown does not get slower per respawn.

        RESIDUAL (accepted, documented): the group kill reaches everything that
        stayed in the process group — INCLUDING ``nohup``/``disown``'d
        children, which re-parent to PID 1 but keep their pgid (verified
        locally). What escapes is a child that leaves the group itself
        (``setsid``, a daemonizing service). Catching those needs a per-exec
        cgroup, which ``docker exec`` does not create.
        """
        gens = [*self._retired, (self._pid, self._pgid)]
        self._retired = []
        self._pid = self._pgid = None
        # A pid is a WIRE value (the shell's handshake line / the server's ping
        # body) and neither producer excludes 0: ``killpg(0, …)`` means "my own
        # group", i.e. the HOST harness. Reject non-positive pids here. Group kill
        # ONLY when the process leads its own group; otherwise the pgid belongs to
        # some shared group and killing it could take out unrelated container
        # processes.
        live = [(pid, pgid) for pid, pgid in gens if pid is not None and pid > 0]
        if self._local:
            pgids = [pgid for pid, pgid in live
                     if pgid == pid and pgid != os.getpgrp()]
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for pgid in pgids:
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(pgid, sig)
                if pgids and sig is signal.SIGTERM:
                    await asyncio.sleep(_REAP_GRACE_S)
            return
        targets = " ".join(f"-{pgid}" if pgid == pid else str(pid)
                           for pid, pgid in live)
        if not targets:
            return
        for sig in ("TERM", "KILL"):
            await self._docker_kill(sig, targets)
            if sig == "TERM":
                await asyncio.sleep(_REAP_GRACE_S)

    async def _docker_kill(self, sig: str, target: str) -> None:
        # `sh -c` (not the `kill` binary): the shell builtin takes a negative
        # process-group target without needing a `--` escape.
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-u", self._exec_user, self._name,
            "sh", "-c", f"kill -{sig} {target} 2>/dev/null || true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT_S)
        except Exception:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()

    async def close(self) -> None:
        """Teardown: drop the host client, then reap what it left in the container."""
        await self._drop_host_client()
        # AFTER the host client is gone, never before: the reap targets the
        # in-container tree the client never owned.
        await self._reap()

    async def _retire(self) -> None:
        """Drop the host client but LEAVE the in-container tree running.

        The respawn path (``client.ExecStdioSession._respawn``): the pipe is dead
        or desynced so the host client must go, but the command it was waiting on
        may be merely SLOW, and killing its group turns a transport fault into a
        silent DATA fault. The identity moves to ``_retired``, so teardown reaps
        it later rather than nobody reaping it ever.
        """
        await self._drop_host_client()
        if self._pid is not None:
            self._retired.append((self._pid, self._pgid))
        self._pid = self._pgid = None

    async def _drop_host_client(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.stdin and not proc.stdin.is_closing():
            with contextlib.suppress(Exception):
                proc.stdin.close()
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=5)


__all__ = ["DockerExecSession"]
