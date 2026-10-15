"""Parametrized atexit leaked-resource reaper shared across env adapters.

Not to be confused with :mod:`lite.gym.remote.reaper` — same word,
two concepts. This one is an ``atexit`` sweep of a process-local registry at
interpreter exit; that one is the env-server's steady-state,
``server_port``-scoped reconcile of the docker host.

Run: not directly. Imported by env modules that keep a process-wide registry
of live containers/processes and want them reaped on hard interpreter exit
(SIGKILL/OOM bypass ``close()``; a crash between ``reset()`` and ``close()``
would otherwise orphan the resource). ``atexit`` does NOT fire on SIGKILL —
that leak is inherently uncoverable — but normal exit / Ctrl-C / unhandled
exception all run the hook.

The skeleton every env hand-rolled is identical:

    with LOCK:
        snapshot = list(REGISTRY)
        REGISTRY.clear()          # only when teardown does not self-remove
    for entry in snapshot:
        try: teardown(entry)
        except Exception: pass

Only the per-entry ``teardown_fn`` differs (``docker rm -f`` name vs POST
``/stop`` vs ``proc.terminate()`` vs ``container.destroy()``), plus whether
the registry is cleared under the snapshot lock. :func:`reap` parametrizes
both so each env keeps a thin module-level ``@atexit.register`` hook that
re-reads its own module global (preserving monkeypatch-ability in tests).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from threading import Lock
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def reap(
    registry: Iterable[T],
    lock: Lock,
    teardown_fn: Callable[[T], None],
    *,
    clear: bool = False,
) -> None:
    """Snapshot ``registry`` under ``lock`` and best-effort ``teardown_fn`` each entry.

    Args:
        registry: The live-resource registry (``set`` or ``list``). Snapshotted
            into a ``list`` under ``lock`` so teardown runs without holding it.
        lock: Guards ``registry`` against concurrent mutation by worker threads
            (the env-server mutates these from ``asyncio.to_thread`` workers).
        teardown_fn: Per-entry best-effort reclaim. Exceptions are caught (so one
            wedged entry can't block the rest at interpreter exit) but always
            LOGGED — a silent swallow here reports a successful reap while the
            resource leaks, which is the exact failure this reaper exists to
            prevent. It also hides callback bugs (a wrong-arity ``TypeError``,
            a renamed-method ``AttributeError``) that are not "best-effort" at all.
        clear: If True, ``registry.clear()`` under the same lock as the
            snapshot. The condition, stated as a condition because a list of
            which env wants which has already gone stale once: clear when
            ``teardown_fn`` does NOT remove the entry from ``registry`` itself,
            so without the clear a second reap would retry entries already torn
            down. Leave it False when teardown self-removes, or when the
            process is exiting anyway and the registry outlives nothing.
    """
    with lock:
        snapshot = list(registry)
        if clear:
            registry.clear()  # type: ignore[attr-defined]
    for entry in snapshot:
        try:
            teardown_fn(entry)
        except Exception:
            # ``logger`` (not ``sys.stderr.write``): ``logging.shutdown`` is
            # registered by the ``logging`` module at ITS import, which always
            # precedes the importing env module's ``@atexit.register`` — and
            # atexit runs LIFO, so every hook that calls ``reap`` runs BEFORE
            # handlers are flushed and closed. (Verified to survive even the
            # inverted ordering: ``StreamHandler.close`` does not close the
            # underlying stream.) The logger buys the env-server's file handler
            # and the ``%(name)s`` origin, which a raw stderr write loses.
            logger.warning(
                "atexit reap: teardown failed for %r — resource may be LEAKED",
                entry, exc_info=True,
            )
