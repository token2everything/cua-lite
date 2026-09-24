"""
Env-server framework recovery / reconcile / shutdown sweeps.

These are the per-env-id lifecycle sweeps the env-server runs around a process
lifetime — boot recovery (:func:`recover_all`), the steady-state drift loop
(:func:`reconcile_all`, ~120 s timer), and graceful teardown
(:func:`shutdown_all`). Each iterates the registered leaf env-ids, dispatches to
the env's services object (:class:`lite.gym.services.EnvServices`), and
isolates per-env exceptions so one env's docker hiccup can't abort the cycle.

Lives under ``lite.gym.remote`` (server-only surface) rather than in the
registry: it depends on the registry's spec table but the registry does not
depend on it. Imported lazily from :mod:`lite.gym.remote.server` so registry and
remote/server never form an import cycle.

Usage:
    from lite.gym.remote.recovery import reconcile_all, recover_all, shutdown_all
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from lite.gym.errors import EnvDepsMissingError, EnvServerUnavailable, EnvUnavailable
from lite.gym.services import services_for
from lite.gym.types import ReapReport, StateView
from lite.utils.registry import split_key

if TYPE_CHECKING:
    from lite.gym.remote.scope import ServerScope

logger = logging.getLogger(__name__)

#: The registry SUBMODULE. ``import lite.gym.registry as registry`` would
#: resolve via attribute access on the ``lite.gym`` package, which
#: ``lite.gym.__init__`` rebinds to the ``Registry`` *instance* (``from
#: lite.gym.registry import registry``) — so ``import ... as`` yields the
#: instance, not the module. ``import_module`` returns the real module. We read
#: its spec/services tables and import helpers through this module object (not
#: bound as locals) so the env-server — and tests that monkeypatch
#: ``registry._specs`` / ``registry._import_all`` / ``registry._services`` —
#: observe the live values.
registry = importlib.import_module("lite.gym.registry")


def _registered_leaf_env_ids() -> list[str]:
    """Env_ids the lifecycle sweeps dispatch to: every leaf with a registered
    spec PLUS every env_id with a registered services object.

    The ``_services`` union is load-bearing now that task registration is lazy:
    webgym / browsergym register their services at import but populate ``_specs``
    only on a ``task_ids`` probe, so a ``_specs``-only walk would skip their reap
    hooks after a fresh server boot — their boot-reap (WA + classifieds-volume
    cleanup, webgym container rm) would never run. Eager envs sit in both sets;
    ``set`` dedups. Using the services keys (not their leaves) keeps this cheap —
    no task-dataset load just to find what to reap."""
    return sorted({split_key(key)[0] for key in registry._specs} | set(registry._services))


def _lifecycle_env_ids(allowed_env_ids: set[str] | None) -> list[str]:
    """Import the served envs without discovering or cleaning unrelated backends.

    Explicit IDs use the registry's namespace resolution. Keep each selected
    ID when its services are registered on an umbrella: services_for/family_of
    resolve that owner without adding sibling IDs to this server's scope.
    """
    if allowed_env_ids is None:
        registry._import_all()
        return _registered_leaf_env_ids()
    selected = []
    for env_id in sorted(allowed_env_ids):
        try:
            registry._import_env(env_id)
        except (EnvDepsMissingError, EnvUnavailable, EnvServerUnavailable) as error:
            logger.debug("Env '%s' is unavailable here (%s)", env_id, error)
        else:
            selected.append(env_id)
    return selected


def _validate_families_declared(env_ids: list[str] | None = None) -> None:
    """Boot-time contract check for the family-gated reaper. Enforces the
    services⟺lifecycle contract in BOTH directions so the per-sweep code can trust it (no
    defensive ``if svc is None`` guards) — NOT placeable in ``register_services`` (envs call it
    BEFORE ``register_family``), so it runs after scoped imports, at the boot entry point.

    Delegates per-env to :func:`lite.gym.services.validate_declarations` —
    the SAME finalizer direct mode fires at import — so the two modes cannot
    drift (its clauses (a)/(b) are exactly the two directions above, plus
    the live_ids-override requirement).
    """
    from lite.gym.services import validate_declarations

    for eid in _registered_leaf_env_ids() if env_ids is None else env_ids:
        validate_declarations(eid)


def reconcile_all(
    state_view: StateView, scope: ServerScope, *, allowed_env_ids: set[str] | None = None
) -> list[ReapReport]:
    """Run the framework reconcile loop for every DEDICATED (PER_INSTANCE) env-id.

    Per env-id (NOT deduped by ``id(svc)`` — each leaf has its own external
    world and its own ``state_view`` slice): build ``tracked`` from the snapshot
    and call :func:`lite.gym.remote.reconcile.reconcile`, which asks the env's
    ``live_ids``/``reap`` and computes ghosts. Only PER_INSTANCE (DEDICATED) envs
    are reconciled here — others are skipped up front by family. A transient probe
    failure (``ReconcileProbeError``) skips that env's cycle; per-env exceptions are
    isolated (logged, empty report) so one env's docker hiccup can't abort the cycle.

    Called by the env-server drift dispatcher on a ~120 s timer. An explicit
    allow-list restricts both imports and lifecycle dispatch to those exact IDs.
    """
    from lite.gym.remote.reconcile import reconcile
    from lite.gym.services import ReconcileMode, family_of, reconcile_mode

    out: list[ReapReport] = []
    for env_id in _lifecycle_env_ids(allowed_env_ids):
        # Family is the authority: only PER_INSTANCE (DEDICATED) envs
        # get steady per-instance reconcile. SINGLETON (BOOT_ONLY) is a steady no-op (its
        # shared backend is reaped only on boot recovery); PURE/REMOTE (NONE) have no world.
        if reconcile_mode(family_of(env_id)) is not ReconcileMode.PER_INSTANCE:
            continue
        svc = services_for(env_id)  # PER_INSTANCE ⟹ has services (F3 enforces it)
        instances = state_view.by_env_id.get(env_id, ())
        tracked = {i.id: (i.external_resource_id, i.created_at) for i in instances}
        try:
            out.append(reconcile(svc, env_id, scope, tracked))
        except Exception as e:
            logger.warning("reconcile for %s failed: %s", env_id, e)
            out.append(ReapReport(orphans_reaped=0))
    return out


def recover_all(scope: ServerScope, *, allowed_env_ids: set[str] | None = None) -> int:
    """Boot-time recovery: the reconcile loop with an empty tracked set for every
    lifecycle-family (DEDICATED/SINGLETON, i.e. ``has_lifecycle``) env-id (see
    :func:`lite.gym.remote.reconcile.recover`); PURE/REMOTE are skipped by family.
    Each env's ``reap`` reclaims the orphans a prior lifetime leaked, under its
    own policy. Idempotent. Returns total reclaimed. With an explicit allow-list,
    only those exact IDs are imported, validated and dispatched.
    """
    from lite.gym.remote.reconcile import recover
    from lite.gym.services import family_of, has_lifecycle

    env_ids = _lifecycle_env_ids(allowed_env_ids)
    _validate_families_declared(env_ids)  # F3: fail-fast if a served env forgot family
    total = 0
    for env_id in env_ids:
        # has_lifecycle == DEDICATED|SINGLETON: the families whose backend the env-server owns
        # and must boot-reap. PURE/REMOTE (no lifecycle) have nothing to reclaim.
        if not has_lifecycle(family_of(env_id)):
            continue
        svc = services_for(env_id)  # has_lifecycle ⟹ has services (F3 enforces it)
        try:
            total += recover(svc, env_id, scope)
        except Exception as e:
            logger.warning("recover for %s failed: %s", env_id, e)
    return total


def shutdown_all(scope: ServerScope, *, allowed_env_ids: set[str] | None = None) -> int:
    """Graceful-shutdown teardown: call ``shutdown(env_id, scope)`` on each
    lifecycle-family (DEDICATED/SINGLETON, i.e. ``has_lifecycle``) env's services
    object, AFTER the per-instance ``env.close()`` loop; PURE/REMOTE are skipped by
    family. Deduped by ``id(svc)`` (one owned-process tree per umbrella). Most envs
    own nothing here (the docker daemon owns container lifetime). Returns the number
    of services objects dispatched to. An explicit allow-list restricts imports
    and dispatch to those exact IDs, including services owned by an umbrella.
    """
    from lite.gym.services import family_of, has_lifecycle

    n = 0
    seen: set[int] = set()
    for env_id in _lifecycle_env_ids(allowed_env_ids):
        if not has_lifecycle(family_of(env_id)):
            continue
        svc = services_for(env_id)  # has_lifecycle ⟹ has services (F3 enforces it)
        if id(svc) in seen:  # dedup: one owned-process tree per umbrella
            continue
        seen.add(id(svc))
        try:
            svc.shutdown(env_id, scope)
            n += 1
        except Exception as e:
            logger.warning("shutdown for %s failed: %s", env_id, e)
    return n
