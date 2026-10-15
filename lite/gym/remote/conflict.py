"""Shared-backend isolation — a generic per-key readers-writer gate.

Env-agnostic. The env-server admits a
``POST /instances`` claim only if its ``metadata.others["conflict_keys"]``
don't conflict with the keys held by live instances:

  * a **writer** (``others["mutating"]=True``) takes a key **exclusively** —
    it may write any resource the key covers, so it must lock the whole thing;
  * **readers** (``mutating=False``) **share** a key and are blocked only
    by a writer (reads don't corrupt).

This contract lives in ``metadata.others`` (not typed core fields) because it
is server-mode-only and opt-in: direct-mode ``gym.make`` never reads it, and
only the few envs that share a mutable backend set it. The keys:
  * ``others["conflict_keys"]``: ``tuple[str, ...]`` — opaque labels for the
    shared resources a task touches. Absent/empty → no constraint.
  * ``others["mutating"]``: ``bool`` — True iff the task may WRITE under those
    keys → exclusive acquire.

This module knows NOTHING about browsergym / WebArena / "sites". A
``conflict_key`` is an opaque label naming a *conflict dimension*, not its
cause. ANY env that sets ``others["conflict_keys"]`` + ``others["mutating"]``
on its task metadata (and, if it mutates, a module-level ``restore_backend(
env_id, key)`` hook) gets isolation for free — browsergym is just the first
consumer.

Concurrency contract: :class:`ConflictKeyTable` is pure in-memory
bookkeeping with NO lock of its own. The caller (``server.py``) MUST hold
``State.lock`` around :meth:`try_acquire` / :meth:`release` — the same lock
that already guards ``POST /instances`` admission, so check-and-acquire is
atomic (no check-then-act race).

Usage in run code::

    # lite/gym/remote/server.py
    from lite.gym.remote.conflict import ConflictKeyTable, restore_backend_dispatch
    state.conflict = ConflictKeyTable()
    # gate, under state.lock:
    if not state.conflict.try_acquire(inst_id, keys, mutating):
        raise CapacityExhausted(..., retry_after_s=...)   # → 503 + Retry-After
    # close path (detached, off-lock), for a writer then release:
    held = state.conflict.holder(inst_id)
    if held and held[1]:
        for key in held[0]:
            await restore_backend_dispatch(env_id, key)
    async with state.lock:
        state.conflict.release(inst_id)
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import sys

logger = logging.getLogger(__name__)

#: The ``metadata.others`` keys the isolation gate reads.
#: Producer (an env's register loop) and consumer (server.py's admission
#: gate) both import THESE — a string-key typo can no longer silently
#: disable isolation. The values are wire-visible (``/tasks`` metadata,
#: ``--filter "m.others.get('mutating')"``) and MUST stay these literals.
OTHERS_CONFLICT_KEYS = "conflict_keys"
OTHERS_MUTATING = "mutating"


@dataclasses.dataclass
class _KeyState:
    """RW-lock state for one ``conflict_key``: a set of reader
    instance-ids OR one writer instance-id."""
    readers: set[str] = dataclasses.field(default_factory=set)
    writer: str | None = None


class ConflictKeyTable:
    """Per-key readers-writer locks keyed by opaque ``conflict_key``.

    Holds two maps, both mutated only under the caller's lock:
      * ``_keys``:    key → :class:`_KeyState` (who holds it now)
      * ``_holders``: instance-id → (keys, mutating) it acquired, so a
        hold can be released by id alone from any close/failure path
        (idempotent).
    """

    def __init__(self) -> None:
        self._keys: dict[str, _KeyState] = {}
        self._holders: dict[str, tuple[tuple[str, ...], bool]] = {}

    def try_acquire(
        self, inst_id: str, keys: tuple[str, ...], mutating: bool
    ) -> bool:
        """All-or-nothing acquire of ``keys`` for ``inst_id`` under the RW
        rule. Returns ``True`` on success (hold recorded), ``False`` on any
        conflict (acquiring nothing → caller should 503). Empty ``keys`` →
        ``True`` (no constraint). CALLER MUST HOLD ``State.lock``."""
        if not keys:
            return True
        # Check ALL first (atomic under the caller's lock — no TOCTOU).
        for k in keys:
            ks = self._keys.get(k)
            if ks is None:
                continue
            if ks.writer is not None:
                return False  # a writer blocks everyone
            if mutating and ks.readers:
                return False  # a writer needs the key idle
        # All clear → acquire ALL.
        for k in keys:
            ks = self._keys.setdefault(k, _KeyState())
            if mutating:
                ks.writer = inst_id
            else:
                ks.readers.add(inst_id)
        self._holders[inst_id] = (keys, mutating)
        return True

    def release(self, inst_id: str) -> None:
        """Release whatever ``inst_id`` holds. Idempotent. GCs a key once
        it has no holders. CALLER MUST HOLD ``State.lock``."""
        held = self._holders.pop(inst_id, None)
        if held is None:
            return
        keys, mutating = held
        for k in keys:
            ks = self._keys.get(k)
            if ks is None:
                continue
            if mutating and ks.writer == inst_id:
                ks.writer = None
            else:
                ks.readers.discard(inst_id)
            if ks.writer is None and not ks.readers:
                self._keys.pop(k, None)

    def holder(self, inst_id: str) -> tuple[tuple[str, ...], bool] | None:
        """Return ``(keys, mutating)`` ``inst_id`` holds, or ``None``.
        Used by the close path to decide whether to restore (writer)."""
        return self._holders.get(inst_id)

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Read-only view for metrics/admin: key → {readers, writer}."""
        return {
            k: {"readers": len(ks.readers), "writer": ks.writer}
            for k, ks in self._keys.items()
        }


async def restore_backend_dispatch(env_id: str, key: str) -> None:
    """Dispatch the per-``env_id`` ``restore_backend(env_id, key)`` module
    hook (mirrors :func:`lite.gym.registry.ensure_services` dispatch). No
    hook defined → no-op. Runs OFF ``State.lock`` (restore is slow — a WA
    stack reset takes 200-500 s) and is awaited if the hook is a coroutine.

    A hook that **raises propagates** to the caller (``_close_quietly``),
    which then keeps the conflict key held rather than re-leasing a dirty
    backend (correctness).

    NOTE: this is the one capability still discovered by getattr-by-name
    rather than a typed ``isinstance`` services dispatch. It is deliberately
    left as-is — the shared-backend isolation migration is deferred until live
    WA/VWA infra is available to validate it; migrating the conflict gate blind
    would risk a silent correctness regression."""
    for path in (
        f"lite.gym.envs.{env_id}.main",
        f"lite.gym.envs.{env_id.split('.')[0]}.main",
    ):
        mod = sys.modules.get(path)
        if mod is None:
            try:
                mod = importlib.import_module(path)
            except Exception:
                continue
        hook = getattr(mod, "restore_backend", None)
        if hook is None:
            continue
        result = hook(env_id, key)
        if inspect.isawaitable(result):
            await result
        return
