"""
Env-server integration surface for CUA-Lite environments.

These types are how an env opts into env-server features. **None of them is part
of the caller-facing :class:`lite.gym.base.LiteBaseEnv` interface.** Direct mode
uses the construction helpers only through :class:`EnvServerPoolable`'s
construction/``bind`` path.

This module holds the *per-instance* capabilities (mixed into the env class):

  * :class:`EnvServerPoolable` — explicit construction + backend recycle
    helpers retained after retiring server warm-pool task reuse.
  * :class:`EnvServerResource` — the drift-reaper's per-instance resource id.

The *per-env-id* capabilities (backend lifecycle, drift reaping) are
registered as a services object per env-id; see
:func:`lite.gym.registry.register_services`. (Shared-backend isolation is
deliberately not a typed capability — see the NOTE at the end of this module.)

See /docs/envs.md for the env-author guide.
"""

from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lite.gym.base import LiteBaseEnv

if TYPE_CHECKING:
    from lite.gym.remote.scope import ServerScope

logger = logging.getLogger(__name__)


class BackendFamily(enum.Enum):
    """The single declared "kind of env" — ONE axis: which backend lifecycle the
    env-server manages, at what cardinality. Everything else (warm strategy,
    reconcile mode, has-lifecycle) derives from this declaration.

    * ``PURE``      — no backend (static dataset / in-process).
    * ``REMOTE``    — backend lives elsewhere; the env-server manages no lifecycle.
    * ``DEDICATED`` — one local resource per instance.
    * ``SINGLETON`` — one local resource shared by all instances.

    Resource *type* (container / VM / emulator / host process) is orthogonal — this
    names lifecycle shape only. ``PURE`` vs ``REMOTE`` is NOT inferable from wiring
    (both have no local lifecycle), so it must be declared, never defaulted.
    """

    PURE = "pure"
    REMOTE = "remote"
    DEDICATED = "dedicated"
    SINGLETON = "singleton"


class WarmStrategy(enum.Enum):
    """Which pre-warm mechanism applies to an env (derived; see :func:`warm_strategy`)."""

    NONE = "none"
    EAGER = "eager"          # shared backend; optional --warm-singleton prewarm


class ReconcileMode(enum.Enum):
    """How the drift reaper reconciles an env's external resources (derived;
    see :func:`reconcile_mode`)."""

    NONE = "none"                  # no external world to reconcile
    PER_INSTANCE = "per_instance"  # diff tracked vs live per-instance ids, reap orphans
    BOOT_ONLY = "boot_only"        # steady tick = no-op; wide-net reap only on boot recovery


class EnvServerResource(ABC):
    """Mixin for an env whose instance owns an external resource the env-server's
    drift reaper must track (a docker container, leased port, browser context).

    Opt in by inheriting this and implementing :meth:`external_resource_id`. The
    server reads it via ``env.unwrapped`` + ``isinstance`` when snapshotting
    instances; an env with no external resource simply does not mix this in.

    Pairing convention: an env that mixes this in should also register an
    :class:`EnvServices` whose ``live_ids`` reports the live set in the SAME
    id-space (this property emits each instance's id; the framework reconcile
    loop diffs ``tracked`` ids against ``live_ids`` to find ghosts). Upheld by
    the env author (not enforced by :func:`lite.gym.registry.register_services`).
    """

    @property
    @abstractmethod
    def external_resource_id(self) -> str | None:
        """Opaque per-instance id of the external resource — the exact-match key
        the drift reaper reconciles against live resources. ``None`` before the
        resource is created (e.g. between ``gym.make`` and first ``reset``)."""
        ...


class EnvServerPoolable(LiteBaseEnv):
    """Construction base with explicit config validation.

    Adds two pieces of structure on top of :class:`LiteBaseEnv`:

    1. **Explicit construction**: env constructors name every accepted
       construction-time kwarg. Unknown kwargs therefore fail at Python
       parameter binding instead of silently becoming attributes.

    2. **Single soft-state entry point**: both the cold path
       (caller's construction) and any later explicit rebind in tests/tools
       bottom out in the same :meth:`bind` method, so identical kwargs
       produce identical state. No ``if X is not None`` mirror-init burden.

    Subclass contract:

      * Implement ``__init__`` with an explicit named-parameter signature for
        construction-time fields.
      * Implement :meth:`bind` — sets soft fields (and task).
        Defaults in the signature are the SINGLE source of truth
        for what each soft kwarg defaults to. UNCONDITIONAL
        assignment in the body — no ``if X is not None`` guard. Must
        be callable with no required positional args (use
        ``task=None`` or ``task_id=""``) so framework can run it once
        during ``__init__`` to populate soft defaults even when the caller
        passed only constructor kwargs.
      * Override :meth:`boot` if the backend is expensive to acquire.

    Construction entry point:

      * :meth:`__init__` initializes construction-time fields, then runs
        ``bind`` exactly once — always in that order.
    """

    def __init__(self, task: Any = None) -> None:
        """Base direct-mode entry for envs with no construction-time fields.

        Concrete envs with construction fields override ``__init__`` explicitly
        and still call ``bind`` exactly once.
        """
        self.bind(task)

    def bind(self, task: Any = None) -> None:
        """Bind a task (and apply soft kwargs). Cheap (in-memory only).
        Repeatable for explicit rebind tests/tools.

        Subclasses MUST override with an **explicit named-parameter
        signature** matching the env's accepted soft kwargs::

            def bind(self, task_id="", *, seed=None, max_steps=None):
                ...

        Two purposes for the explicit signature:

          1. Unknown soft kwargs raise ``TypeError`` at bind time instead of
             silently ``setattr``-ing a wrong attribute name.
          2. Per-env field-name remapping and resolver work lives
             next to the field declaration, not behind a generic
             ``setattr`` blob.

        **Critical invariant**: assignments must be UNCONDITIONAL
        (no ``if X is not None`` guard) and defaults in the
        signature are the SINGLE source of truth. Direct/server construction
        bottoms out here, so the body always runs with the caller's exact
        value — including ``None`` — and both modes produce byte-equal state.

        Must be callable with no required positional args
        (``task=None`` default at the base, ``task_id=""`` at most
        envs) so framework construction can always run bind once after
        construction.

        Default impl just stores task.
        """
        self._task = task

    async def boot(self) -> None:
        """Acquire the env's expensive, task-independent resource (container,
        emulator, browser context, external lease, ...).

        Override :meth:`reset` to call this lazily
        (``if not self._booted: await self.boot()``) so direct and server
        cold-spawn callers acquire the resource at the same lifecycle point.

        Contract:
          * Idempotent — calling while already booted is a no-op.
          * Task-independent — depends only on constructor state,
            NOT on a bound task.
          * Failure-safe — on exception (including CancelledError), clean up any
            partial state before re-raising. No half-acquired resource survives.

        Default no-op for envs whose backend is cheap enough to acquire entirely
        in :meth:`reset`.
        """

    # ── Cross-task reuse / recycle ──────────────────────────────────────────
    #
    # The re-task half of the lifecycle, symmetric with :meth:`boot` (the
    # acquire half). An env that can cheaply roll its backend back to
    # pristine (snapshot reload) overrides :meth:`reset_to_pristine` (+ the
    # task hooks) and calls :meth:`reset_with_recycle` at the top of its
    # ``reset()`` — the four invariants every hand-rolled copy re-derived
    # (recycle cap, first-reset gate, tear-down-BEFORE-pristine,
    # cold-spawn-skip) then live HERE, once.
    #
    # Seam ownership: the env exposes ONE handle attr, ``_current_container``
    # (what ``external_resource_id`` reads). ``boot()`` assigns it right
    # after the factory's acquire returns, BEFORE the RPC/init binding (the
    # docker-run→ready window inside acquire is covered by the reaper's
    # orphan age grace, and mobileworld additionally exposes its
    # ``_pending_container`` during that window); ``destroy_backend()``
    # nulls it only AFTER destroy() returns. "Has boot succeeded" is
    # tracked by env state, never by the handle.

    #: IN-PLACE REUSES served by the CURRENT backend (successful pristine
    #: rollbacks — cold spawns don't count and restart the budget at 0);
    #: owned by reset_with_recycle.
    _recycle_count: int = 0
    #: First-reset gate: stays False until the first reset_with_recycle
    #: completes, so an env's first reset never destroys a just-booted backend
    #: (even under cap 0).
    _recycle_first_done: bool = False

    async def reset_to_pristine(self) -> bool:
        """Cheap in-place rollback of the backend to its pristine baseline
        (snapshot reload). Return ``False`` → the framework destroys +
        reboots instead (the safe default for envs with no rollback
        action).

        Declaring this override is a CONTRACT: the env's yaml must then
        carry an explicit ``server_kwargs.max_resets_per_container`` (the
        finalizer fails the import otherwise) and the constructor must set
        ``self._max_resets_per_container`` from it — there is no safe
        universal default (0 is a 2-5 min cold-boot-per-episode cliff for
        mobileworld; unbounded is silent drift)."""
        return False

    async def tear_down_task(self) -> None:
        """Undo the previous episode's task mutations. Runs BEFORE
        :meth:`reset_to_pristine` — tear-down mutations assume the
        post-episode dirty state; running after the rollback would smear
        them onto the pristine baseline (androidworld documented a
        ``reward=0`` regression from the reversed order). No-op default;
        must tolerate "no previous task"."""

    async def init_task(self) -> None:
        """Instantiate the bound task on the pristine backend. No-op
        default."""

    async def destroy_backend(self) -> None:
        """Destroy the current backend handle, reaper-safe: keep
        ``self._current_container`` (the ``external_resource_id`` source)
        pointing at the dying container UNTIL ``destroy()`` returns, then
        null it — nulling early lets the drift reaper classify the
        still-live container as an orphan and reap it mid-teardown. No-op
        default for envs with no backend."""

    def backend_alive(self) -> bool:
        """Is a usable backend currently bound? Default reads the seam attr;
        envs whose usable-ness is a separate binding (androidworld's RPC
        ``_env``, mobileworld's ``_client``) override — the seam attr can
        point at a container whose boot never completed."""
        return self._current_container is not None

    async def reset_with_recycle(self) -> bool:
        """The framework-owned re-task sequence (call at the top of
        ``reset()``). Returns True iff a fresh cold boot happened this call
        (callers that need to skip additional post-recycle work can branch).

        Sequence (the invariants, in their load-bearing order):
          1. recycle — cap reached (and past the first-reset gate) →
             ``destroy_backend()``;
          2. cold boot if no backend — ``await self.boot()``; the fresh
             backend's reuse budget restarts at 0;
          3. surviving backend only: ``tear_down_task()`` (BEFORE pristine),
             then ``reset_to_pristine()``; a ``False`` rollback →
             destroy + fresh ``boot()`` (never a dirty backend);
          4. ``init_task()``; counters advance.
        """
        cap = getattr(self, "_max_resets_per_container", None)
        if cap is None:
            if type(self).reset_to_pristine is not EnvServerPoolable.reset_to_pristine:
                # The class-creation finalizer proves the YAML key exists, but
                # only the constructor can wire it onto the instance — a wiring
                # omission would otherwise silently degrade to cap 0 (the
                # cold-boot-per-episode cliff is not a safe silent default).
                raise RuntimeError(
                    f"{type(self).__name__} overrides reset_to_pristine but "
                    "__init__ never set self._max_resets_per_container "
                    "(wire it from server_kwargs)"
                )
            cap = 0
        if (self._recycle_first_done
                and self.backend_alive()
                and self._recycle_count >= cap):
            logger.info(
                "%s: recycle cap reached (%d in-place reuses, cap=%d) — "
                "destroying backend %s", type(self).__name__,
                self._recycle_count, cap,
                getattr(self, "external_resource_id", None),
            )
            await self.destroy_backend()

        just_cold = False
        if not self.backend_alive():
            await self.boot()
            self._recycle_count = 0
            just_cold = True

        if not just_cold:
            await self.tear_down_task()
            if not await self.reset_to_pristine():
                await self.destroy_backend()
                await self.boot()
                self._recycle_count = 0
                just_cold = True
            else:
                # cap counts IN-PLACE REUSES (successful pristine rollbacks),
                # not episodes: cap=N → N reuses = N+1 episodes per backend,
                # matching the pre-migration androidworld semantics exactly
                # (its counter incremented only on a successful snapshot
                # reload). Cold spawns restart the budget at 0 above.
                self._recycle_count += 1

        await self.init_task()
        self._recycle_first_done = True
        return just_cold

    #: The ONE backend-handle attr used by the server reaper.
    _current_container: Any = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Finalizer: an env that DECLARES the rollback capability
        (overrides :meth:`reset_to_pristine`) must carry an explicit
        ``server_kwargs.max_resets_per_container`` in its yaml — loud at
        import, never a silently-unbounded (or silently-cold-booting)
        default. Envs without the override need nothing (their framework
        path is destroy-every-reset by construction)."""
        super().__init_subclass__(**kwargs)
        if cls.reset_to_pristine is EnvServerPoolable.reset_to_pristine:
            return
        import sys
        from pathlib import Path
        mod = sys.modules.get(cls.__module__)
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            return
        try:
            from lite.gym.utils import config as _env_config
            cfg = _env_config.load(str(Path(mod_file).parent))
        except Exception:
            return  # abstract intermediates / non-env modules carry no yaml
        if "max_resets_per_container" not in cfg.server_kwargs:
            raise RuntimeError(
                f"{cls.__name__} overrides reset_to_pristine (declares the cross-task "
                f"reuse capability) but its configs/default.yaml server_kwargs has no "
                f"explicit max_resets_per_container — declare one (LARGE for slow-boot "
                f"reuse envs like mobileworld; 0 for destroy-every-reset). There is no "
                f"safe universal default."
            )


# ===========================================================================
# Per-env-id capabilities — registered as a `services` object via
# `lite.gym.registry.register_services`. The env-server discovers each
# capability by `isinstance(services, <Capability>)` (typed, never
# `getattr`-by-name). An env's services object subclasses ONLY the
# capabilities it needs; methods *within* a capability are individually
# optional (no-op defaults). None of this is touched in direct mode.
# ===========================================================================

class EnvServices:
    """Per-env-id server-side hooks, called by the env-server at fixed lifecycle
    phases:
    one object per env-id, all methods optional with safe defaults, an
    env overrides only the phases it has. Umbrella leaves register the SAME
    object. NONE of this runs in direct or client mode.

    The env contains NO ghost logic and never touches the instance table —
    GHOST reconciliation is the framework's (``lite.gym.remote.reconcile``), because
    it mutates ``state.envs``. The env owns its external world entirely: it
    answers ``live_ids`` (a liveness oracle) and reclaims its own orphans in
    ``reap`` (its full policy).

    Cohesion: the lifecycle side-effects (``ensure``/``health``/``shutdown``)
    and the reconcile view (``live_ids``/``reap``) are two concerns co-located
    only for registration. The framework loop is typed against the reconcile
    view alone (``lite.gym.remote.reconcile.ReconcileView``), so it cannot see the
    lifecycle methods.

    DEDICATED (per-instance) container backends do NOT reimplement ``live_ids``/``reap``
    — they COMPOSE a ``ContainerReaper`` (hold-and-delegate; see :class:`ContainerReaper`),
    which avoids the MRO shadow footgun a mixin would create. SINGLETON container backends
    instead subclass :class:`SingletonContainerServices` (inherit boot-only ``reap``/
    ``shutdown``, base ``live_ids`` → ``None``) — they have no per-instance world to reconcile.
    """

    # -- lifecycle side-effects --
    def register_tasks(self, env_id: str) -> None:
        """Register this env-id's task catalog into the registry on first demand
        — cheap, idempotent, NO backend acquisition.

        Fired by ``registry.task_ids`` / ``registered_env_ids`` (a catalog probe)
        and by the framework right before :meth:`ensure` boots. Kept DISTINCT
        from :meth:`ensure` precisely so a catalog probe never boots a container:
        a lazy env that loads its task list from a dataset (``webgym``,
        ``browsergym``) overrides THIS — not ``ensure`` — so ``import``-ing the
        env main stays cheap and ``task_ids`` doesn't start the backend. No-op
        default: most envs ``register()`` eagerly at module import and need no
        hook here."""

    def ensure(self, env_id: str) -> None:
        """Idempotently start/attach the shared backend on first use for this
        env-id. Raise :class:`lite.gym.errors.EnvDepsMissingError` if deps are
        absent. Lazy (demand-driven), NOT a boot step. The framework calls
        :meth:`register_tasks` before this, so ``ensure`` may assume the catalog
        is registered. No-op default."""

    def health(self, env_id: str) -> None:
        """Cheap (<10 ms) availability probe for ``GET /envs/<id>``. Raise
        :class:`lite.gym.errors.EnvDepsMissingError` if unavailable. No-op
        default (= available)."""

    def shutdown(self, env_id: str, scope: ServerScope) -> None:
        """Stop a process THIS run owns (e.g. webgym's OmniBoxes tree) at
        graceful shutdown, AFTER all per-instance ``env.close()``s. Most envs
        own nothing here (the docker daemon owns container lifetime). No-op
        default."""

    # -- reconcile view (the env's external world) --
    def live_ids(self, env_id: str, scope: ServerScope) -> set[str] | None:
        """The ids of external resources that currently EXIST for this run, in
        the SAME id-space as ``InstanceView.external_resource_id``. The framework
        uses this for GHOST detection only. Return ``None`` if the env has no
        external world to reconcile (the loop then skips it — distinct from
        'registers no services', which is also skipped).

        MUST FAIL CLOSED on a fallible/remote oracle (an HTTP master, a flaky
        CLI): on any error or a possibly-partial result, **raise**
        :class:`lite.gym.errors.ReconcileProbeError` (the loop catches it and skips
        this cycle), NEVER a partial ``set()`` and NEVER ``None`` (``None`` is
        reserved for "no per-instance world", above). A partial live set makes
        genuinely-live resources look gone → the loop ghosts their instances →
        live episodes 404. Raising is safe; a partial ``set()`` is catastrophic.

        No-op default returns ``None`` ("no external world")."""
        return None

    def reap(
        self, env_id: str, scope: ServerScope, in_use: set[str], *, boot: bool = False
    ) -> int:
        """Reclaim external resources this run owns that are NOT in ``in_use``
        (the ids still backing live instances). This is the env's ENTIRE orphan
        policy and the framework neither knows nor constrains it: age guard,
        idle-TTL pool compaction, co-tenant sparing, process-tree kill, artifact
        GC, boot wide-net. Touches ONLY the external world. Returns count
        reclaimed.

        ``boot`` is supplied by the FRAMEWORK, not latched by the env: it is
        ``True`` exactly for the one-shot boot-recovery call (``recover_all`` runs
        once at startup, before serving), ``False`` for every steady-state tick.
        A boot reap is wide-net (reclaim every prior-lifetime leftover — nothing
        races before serving); a steady reap is conservative (age guard / cap /
        spare-live). Envs read this flag instead of hand-rolling a module-global
        "have I booted yet" latch — the framework owns the boot/steady split (see
        :func:`lite.gym.remote.reconcile.recover` vs :func:`~.reconcile`).

        No-op default returns 0."""
        return 0


# NOTE: shared-backend isolation (conflict_keys / mutating / restore_backend)
# is deliberately NOT modeled as a capability ABC here. It stays on its proven
# channels — server-only keys in ``metadata.others`` (conflict_keys / mutating)
# read at claim time, and the env module's ``restore_backend`` function
# dispatched by :func:`lite.gym.remote.conflict.restore_backend_dispatch`. The
# conflict gate is correctness-critical and only validatable on live WA/VWA
# infra, so migrating it to a typed capability is deferred until that infra is
# available. Until then there's no env to subclass such an ABC, so it would be
# dead scaffolding.


# ===========================================================================
# Capability dispatch (env_id -> EnvServices / BackendFamily)
#
# These dispatchers operate on the registry's module-global state dicts
# (``_services``, ``_families``), which stay in ``lite.gym.registry`` as the
# single source of truth so that:
#   - tests that ``monkeypatch.setattr(registry, "_services", {...})``, and
#   - ``lite.gym.remote.recovery`` which reads ``registry._specs``,
# observe the same dict. Each function therefore imports the dict it needs
# from ``lite.gym.registry`` at CALL time — re-reading the module attribute on
# every call, so a monkeypatched replacement is observed here. The import has
# to stay inside the function: a top-level one would dead-lock, since
# ``lite.gym.registry`` calls back into this module while it is still being
# imported.
# ===========================================================================


def health_check(env_id: str) -> None:
    """Run the env's availability probe for ``GET /envs/{env_id}``: the typed
    EnvServices.health capability.

    No-op for envs without one — they report ``available: true`` based on
    successful task registration (the historical behavior). The probe must be
    cheap (<10 ms); it raises :class:`~lite.gym.errors.EnvDepsMissingError`
    (→ ``available: false`` + install hint) when the env's deps/assets are
    missing — surfacing runtime-only failures symmetrically with import-time
    ones.
    """
    svc = services_for(env_id)
    if svc is not None:
        svc.health(env_id)


def register_services(env_id: str, services: EnvServices) -> None:
    """Register the per-env-id capabilities object for ``env_id`` — it MUST
    subclass :class:`EnvServices`. The env-server dispatches by ``isinstance``;
    umbrella envs register the SAME instance for each leaf env_id (every
    capability method takes ``env_id``).

    Fails fast on a non-``EnvServices`` object: dispatch silently skips anything
    that isn't an ``EnvServices`` (a hangover from the old duck-typed hook model),
    so a mis-typed registration would leave the env's ``live_ids``/``reap``/
    ``ensure`` never running — orphans would leak forever with no error. Better to
    blow up at import time.
    """
    if not isinstance(services, EnvServices):
        raise TypeError(
            f"register_services({env_id!r}, ...): services must subclass "
            f"lite.gym.services.EnvServices, got {type(services).__name__}"
        )
    from lite.gym.registry import _services
    _services[env_id] = services


def services_for(env_id: str) -> object | None:
    """The registered services object for ``env_id`` (umbrella-aware:
    ``browsergym.miniwob`` falls back to ``browsergym``)."""
    from lite.gym.registry import _services
    svc = _services.get(env_id)
    if svc is None and "." in env_id:
        svc = _services.get(env_id.split(".")[0])
    return svc


# ===========================================================================
# Family declaration + derivations (the top-down "declare once, derive the rest";
# ``register_family`` is the SINGLE per-env declaration; warm strategy / reconcile
# mode / has-lifecycle all derive from it.
# ===========================================================================

def register_family(env_id: str, family: BackendFamily) -> None:
    """Declare ``env_id``'s :class:`BackendFamily` — the one fact every env states up
    front. Required (no default): PURE vs REMOTE is not wiring-inferable, so defaulting
    would be the absence-inference smell. Umbrella envs register the umbrella id (leaves
    resolve to it via :func:`family_of`)."""
    if not isinstance(family, BackendFamily):
        raise TypeError(
            f"register_family({env_id!r}, ...): family must be a BackendFamily, "
            f"got {type(family).__name__}"
        )
    from lite.gym.registry import _families
    _families[env_id] = family


def family_of(env_id: str) -> BackendFamily | None:
    """The declared :class:`BackendFamily` for ``env_id`` (umbrella-aware:
    ``browsergym.miniwob`` → ``browsergym``). ``None`` if undeclared.

    NB: the ``split(".")[0]`` fallback only fires when ``env_id`` itself isn't
    registered. ``lite.osworld`` / ``lite.demo`` register under their full id (direct
    hit wins), and no ``lite`` family exists — so a dotted leaf never wrongly resolves
    to a ``lite`` umbrella. If a ``lite.<x>`` *umbrella* with sub-leaves is ever added,
    revisit this (it would need a real ``lite`` declaration)."""
    from lite.gym.registry import _families as fams
    fam = fams.get(env_id)
    if fam is None and "." in env_id:
        fam = fams.get(env_id.split(".")[0])
    return fam


def validate_declarations(env_id: str) -> None:
    """Mode-independent declaration finalizer.

    Cross-checks the two halves of a capability declaration — the instance
    side (inheritance) and the env-id side (registration) — for ONE env after
    its module import completed. Runs in BOTH modes: the registry fires it at
    the end of every direct-mode import (fail fast at first make/task-probe,
    not silently at reap time), and the server's boot validator sweeps it for
    every registered env. This is the direct repair of the split-brain seam:
    the id-space contract used to be "upheld by the env author (not
    enforced)".

    Raises RuntimeError on:
      (a) registered ``EnvServices`` but no declared ``BackendFamily`` —
          the family gate would silently never reap the env's backend;
      (b) a lifecycle family (DEDICATED/SINGLETON) with no ``EnvServices`` —
          a backend the server cannot recover/reconcile/shutdown;
      (c) a PER_INSTANCE-reconciled (DEDICATED) env whose services do
          NOT override ``live_ids`` — the base's ``None`` means "no
          per-instance world", which silently disables ghost/orphan
          detection for a container-owning env.

    Warm-pool retirement note: DEDICATED envs always cold-spawn in server mode.
    """
    fam = family_of(env_id)
    svc = services_for(env_id)

    if svc is not None and fam is None:
        raise RuntimeError(
            f"env {env_id!r} registered services but declared no BackendFamily — add "
            f"register_family({env_id!r}, BackendFamily.X) at the env main's module "
            f"scope. The family-gated reaper would otherwise "
            f"never reap it."
        )
    if has_lifecycle(fam) and svc is None:
        raise RuntimeError(
            f"env {env_id!r} declared a lifecycle BackendFamily ({fam}) but registered "
            f"no EnvServices — register_services({env_id!r}, ...) so the server can "
            f"recover/reconcile/shutdown its backend."
        )
    # NOTE: this clause keys on the DECLARED family (PER_INSTANCE
    # reconcile), not on whether the env CLASS mixes in EnvServerResource —
    # the finalizer sees env_ids, and the registry holds entry-point
    # factories, not classes, so the mixin isn't discoverable here. A
    # SINGLETON-declared env that (wrongly) mixes in EnvServerResource is
    # caught by review + the family-declaration sweep, not by this check.
    if (svc is not None
            and reconcile_mode(fam) is ReconcileMode.PER_INSTANCE
            and type(svc).live_ids is EnvServices.live_ids):
        raise RuntimeError(
            f"env {env_id!r} is DEDICATED (per-instance reconcile) but its services "
            f"object ({type(svc).__name__}) does not override live_ids() — the drift "
            f"reaper has no reconcile view, so ghosts/orphans would silently never be "
            f"detected. Subclass ContainerServices or implement live_ids/reap."
        )

@dataclass(frozen=True)
class FamilyTraits:
    """Per-family server behavior, the SINGLE source of truth derived from the declared
    :class:`BackendFamily`. ``reconcile_mode``/``has_lifecycle``/``warm_strategy`` are thin
    readers over :data:`_FAMILY_TRAITS`. **Adding a family = ONE row** (+ a validator clause)
    instead of 3-4 scattered ``if family is X`` ladders.

    * ``reconcile`` — how the drift reaper treats this family's external resources.
    * ``has_lifecycle`` — does the env-server manage a backend lifecycle (recover/shutdown).
    * ``warms`` — the family-level warm capability. Singleton/shared-service
      prewarm is ``EAGER``; everything else is ``NONE``.
    """
    reconcile: ReconcileMode
    has_lifecycle: bool
    warms: WarmStrategy


_FAMILY_TRAITS: dict[BackendFamily, FamilyTraits] = {
    BackendFamily.DEDICATED: FamilyTraits(ReconcileMode.PER_INSTANCE, True,  WarmStrategy.NONE),
    BackendFamily.SINGLETON: FamilyTraits(ReconcileMode.BOOT_ONLY,    True,  WarmStrategy.EAGER),
    BackendFamily.PURE:      FamilyTraits(ReconcileMode.NONE,         False, WarmStrategy.NONE),
    BackendFamily.REMOTE:    FamilyTraits(ReconcileMode.NONE,         False, WarmStrategy.NONE),
}
# Undeclared family (None) → inert: no reconcile, no lifecycle, no warm. (The F3 validator
# fails fast on a services-registered env that forgot to declare a family; this keeps the
# readers total/safe for any other None lookup.)
_NO_FAMILY_TRAITS = FamilyTraits(ReconcileMode.NONE, False, WarmStrategy.NONE)


def _traits(family: BackendFamily | None) -> FamilyTraits:
    return _FAMILY_TRAITS.get(family, _NO_FAMILY_TRAITS)


def reconcile_mode(family: BackendFamily | None) -> ReconcileMode:
    """How the drift reaper reconciles this family (reader over :data:`_FAMILY_TRAITS`):
    ``DEDICATED`` → per-instance ghost-detect+reap; ``SINGLETON`` → boot-only wide-net
    (steady tick is a no-op — never reap the shared backend mid-run); ``PURE``/``REMOTE``/
    undeclared → none."""
    return _traits(family).reconcile


def has_lifecycle(family: BackendFamily | None) -> bool:
    """True iff the env-server manages a backend lifecycle (ensure/health/shutdown) for this
    family — ``DEDICATED`` or ``SINGLETON`` (not PURE/REMOTE). Reader over
    :data:`_FAMILY_TRAITS`."""
    return _traits(family).has_lifecycle


def warm_strategy(env_id: str) -> WarmStrategy:
    """How an env's backend warms.

    ``SINGLETON`` → ``EAGER``. ``DEDICATED`` server warm-pool reuse is retired,
    so DEDICATED/PURE/REMOTE/undeclared envs return ``NONE``.
    """
    return _traits(family_of(env_id)).warms
