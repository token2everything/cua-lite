"""
Framework-owned reconciliation loop for the env-server.

Split by **who owns the mutated state**. The env-server
owns the instance table (``state.envs``), so it owns GHOST reconciliation —
popping instances whose backing external resource has vanished. An env owns its
external world entirely: it answers a liveness question (``live_ids``) and
reclaims its own orphans (``reap``) under its own policy. This module is that
split's framework half — backend-agnostic, ~10 lines of set-algebra. No env
imports it.

Run: imported by the env-server drift dispatcher (lite.gym.remote.server) and
the boot-recovery path. Unit-tested with fake views in
``tests/gym/remote/test_reconcile.py`` (no docker / live infra needed).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Protocol

from lite.gym.types import ReapReport

if TYPE_CHECKING:
    from lite.gym.remote.scope import ServerScope


#: Cap on ghost ids popped per env per cycle. Framework-owned because it bounds
#: the ``state.envs`` mutation. Env-var overridable; matches the legacy cap.
MAX_GHOSTS_PER_CYCLE = int(os.environ.get("CUA_LITE_DRIFT_MAX_GHOSTS_PER_CYCLE", "10"))

#: Ghost-side age guard (seconds): never pop a tracked instance whose backing
#: resource may still be booting. Protects framework state (the instance row);
#: the symmetric *orphan-side* guard lives in the env's ``reap`` (it protects
#: the external world). Env-var driven.
DEFAULT_SAFETY_MARGIN_S = float(
    os.environ.get("CUA_LITE_DRIFT_SAFETY_MARGIN_S", "120")
)


# KEEP the Protocol even though ``EnvServices`` is its only implementer: it is
# a TYPE-NARROWING seam, not a polymorphism seam. Its value is what it OMITS —
# typing the loop against the two-method view is what mechanically enforces the
# invariant stated below (and echoed at ``lite/gym/services.py``). Inlining
# ``EnvServices`` here would hand the loop ``ensure``/``shutdown`` and the
# invariant would degrade from a type error to a code-review promise.
class ReconcileView(Protocol):
    """The env's external world, as the reconcile loop sees it. An env's
    services object (``lite.gym.services.EnvServices``) satisfies this — its
    ``live_ids``/``reap`` default to no-ops. The loop is typed against this view
    ALONE, so it cannot see the env's lifecycle methods (``ensure``/``health``/
    ``shutdown``): 'the loop owns no lifecycle concern' holds by type."""

    def live_ids(self, env_id: str, scope: ServerScope) -> set[str] | None: ...
    def reap(
        self, env_id: str, scope: ServerScope, in_use: set[str], *, boot: bool = False
    ) -> int: ...


def reconcile(
    view: ReconcileView,
    env_id: str,
    scope: ServerScope,
    tracked: dict[str, tuple[str | None, float]],
    *,
    safety_margin_s: float = DEFAULT_SAFETY_MARGIN_S,
    now: float | None = None,
) -> ReapReport:
    """One reconcile cycle for one ``env_id``. Backend-agnostic.

    Args:
        view: the env's reconcile view (its services object).
        env_id: the env-id being reconciled (kept on the call so an umbrella
            services object knows which leaf it acts for).
        scope: this server run's :class:`ServerScope`.
        tracked: ``{instance_id: (external_resource_id, created_at)}`` for this
            env_id's live instances, from the StateView snapshot.
        safety_margin_s: ghost-side age guard.
        now: clock override for tests; defaults to ``time.time()``.

    Returns the env's orphan count + the GHOST ``instance_id``s (tracked
    instances whose external resource is gone AND older than ``safety_margin_s``,
    capped at :data:`MAX_GHOSTS_PER_CYCLE`). The caller pops ``ghost_ids`` from
    ``state.envs`` under its own lock *after* :func:`collect_valid_ghosts`.
    """
    # A fallible probe (docker ps / pgrep) raises ReconcileProbeError → skip this cycle
    # fail-closed (never a partial reap). The local catch keeps today's silent-on-transient
    # behavior; recovery.py's blanket except is a backstop, not the primary path.
    from lite.gym.errors import ReconcileProbeError
    try:
        live = view.live_ids(env_id, scope)
    except ReconcileProbeError:
        return ReapReport(0)
    # `is None` is identity, NOT truthiness. An empty ``set()`` means "world exists,
    # currently empty → every tracked id is a ghost"; ``None`` means "no per-instance world"
    # (the base default — a DEDICATED env now RAISES instead of returning None, so this
    # branch survives as a defensive guard for a DEDICATED that forgot to override live_ids).
    # Writing ``if not live`` would mass-false-ghost every live instance the instant a world
    # momentarily empties. This is the sharpest edge in the reconcile contract.
    if live is None:
        return ReapReport(0)

    # The ids still backing live instances — the env must NOT reap these.
    in_use = {
        ext for ext, _ in tracked.values() if ext is not None and ext in live
    }
    n_reaped = view.reap(env_id, scope, in_use, boot=False)

    if now is None:
        now = time.time()
    ghosts = tuple(
        iid
        for iid, (ext, created) in tracked.items()
        if ext is not None and ext not in live and now - created > safety_margin_s
    )[:MAX_GHOSTS_PER_CYCLE]
    return ReapReport(n_reaped, ghosts)


def recover(view: ReconcileView, env_id: str, scope: ServerScope) -> int:
    """Boot-time recovery — reclaim a prior lifetime's orphans.

    At boot there are NO tracked instances, so there are NO ghosts to detect.
    Recovery is therefore PURELY "reclaim orphans" — it calls the env's ``reap``
    DIRECTLY with ``in_use=∅``, bypassing the ``live_ids`` ghost gate.

    Why not ``reconcile(∅)``: ``reconcile`` early-returns (skipping ``reap``)
    when the probe raises ``ReconcileProbeError`` (a container env raises it on any
    ``docker ps`` hiccup) — or when ``live_ids`` returns ``None``. If boot recovery
    went through that gate, a docker glitch during boot would skip the one wide-net
    reap entirely.
    Calling ``reap`` directly here with ``boot=True`` makes boot recovery
    independent of the ghost oracle: the framework — not the env — owns the
    boot/steady split, so no env needs a hand-rolled "have I booted" latch (the
    flag IS the latch, set by which entry point called). Returns the orphan
    count.
    """
    return view.reap(env_id, scope, set(), boot=True)


def collect_valid_ghosts(
    ghost_ids: list[str] | tuple[str, ...], known_ids: set[str],
) -> list[str]:
    """Cross-env ghost validation: keep only ``ghost_ids`` present in the
    snapshot's known-id set. A buggy env's ``reap``/reconcile must never cause
    another env's instance to be popped (the failure mode — other env's clients
    404 — is silent). The dispatcher calls this over the union of all envs'
    reports before popping anything under ``state.lock``.
    """
    return [g for g in ghost_ids if g in known_ids]
