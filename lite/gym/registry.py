"""
Gymnasium-style task registry for cua-lite gym.

Terminology:
    env_id  — environment name, e.g. "lite.demo", "lite.osworld", "webgym"
    env_key — fully-qualified task identifier: "{env_id}@{task_id}",
              e.g. "lite.demo@create_file", "lite.osworld@osworld_chrome_030eeff7"
              (called ``key`` in function signatures for brevity)

    register("lite.osworld@osworld_chrome_030eeff7", entry_point=create_env_fn, split="eval")

Splits convention:
    - "train" — tasks for training (default)
    - "eval"  — tasks for evaluation / benchmarking

Usage:
    import lite.gym as gym

    gym.registry.env_ids()                          # ['lite.demo', 'lite.osworld', 'webgym']
    gym.registry.task_ids("lite.demo")              # {'eval': ['create_file', ...]}
    gym.registry.task_ids("lite.demo", split="eval") # ['create_file', ...]
    env = gym.make("lite.demo@create_file")
"""

from __future__ import annotations

import copy
import importlib
import logging
import os
import re
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lite.core.metadata import LiteBaseMetadata
from lite.gym.base import LiteBaseEnv
from lite.gym.errors import (
    EnvDepsMissingError,
    EnvServerUnavailable,
    EnvUnavailable,
)
from lite.gym.utils.server.routing import routing_server_url
from lite.utils.registry import split_key

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EnvSpec — what we store per registered task
# ---------------------------------------------------------------------------

@dataclass
class EnvSpec:
    """Registration entry for a single task."""
    key: str                                     # "lite.demo@create_file"
    entry_point: Callable[..., LiteBaseEnv]  # factory(**kwargs) -> env
    kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: LiteBaseMetadata | None = None


# Wrapper config kwargs that make() consumes CLIENT-SIDE (NOT passed to the env entry_point),
# with their make() defaults. THE single source of truth — both the membership AND the default
# of each wrapper kwarg live here exactly once:
#   • ``lite.gym.factory.make`` pops each from ``merged`` USING this dict (`merged.pop(k, d)`),
#     so its pop set + defaults can't diverge from the membership;
#   • ``CARRIED_SPEC_KWARGS`` (the name tuple below) includes these names so they travel
#     env-server → client and REMOTE mode resolves the SAME per-task config as DIRECT (else the
#     register_from_server stub specs revert to defaults, e.g. osworld step_timeout 180->120);
#   • env-server task metadata carries the same make config as direct mode;
#     wrapper-owned kwargs are consumed by ``make()`` and never passed into strict
#     env ``bind()`` signatures.
# Env-owned make kwargs (currently ``cursor``) are carried but are NOT popped by make() and
# must be accepted and applied by the env itself.
#
# ``loop_detect``: the registry DEFAULT is 0 (off). It is observation-blind — the
# fingerprint at ``lite/gym/wrappers.py`` hashes the action only and never the
# observation, so legitimate repeated navigation on a changing screen reads as a
# stuck loop (measured 2/6 false positives on androidworld). Committed configs MAY
# still pin it per (agent, env) — several deliberately do, because some models loop
# more than others — and an explicit value always wins over this default. Override
# per run with ``--env-kwargs '{"loop_detect": N}'``.
#
# ``loop_detect_terminal_status``: the terminal status injected when
# ``loop_detect`` fires. It is wrapper-owned make config, but the score-sensitive
# meaning of the string belongs to each env's terminal-action handler.

_WRAPPER_KWARG_DEFAULTS: dict[str, Any] = {
    "step_timeout": 120.0,
    "reset_timeout": 600.0,
    "loop_detect": 0,
    "loop_detect_terminal_status": "success",
}
_ENV_MAKE_KWARGS: frozenset[str] = frozenset({
    "cursor",
})
CARRIED_SPEC_KWARGS: tuple[str, ...] = (
    tuple(_WRAPPER_KWARG_DEFAULTS) + tuple(sorted(_ENV_MAKE_KWARGS))
)

# ---------------------------------------------------------------------------
# Global registry state
# ---------------------------------------------------------------------------

_specs: dict[str, EnvSpec] = {}                    # key -> EnvSpec
_splits: dict[str, dict[str, list[str]]] = {}      # env_id -> {split: [task_ids]}
# env_id -> env-wide make-kwarg DEFAULTS (the lowest registered layer for the
# CARRIED_SPEC_KWARGS framework config — see set_env_make_kwargs / make()). Sourced
# from each env's default.yaml ``make_kwargs`` so a uniform env declares timeouts
# and env-owned capture knobs once instead of per-task; per-task register()
# kwargs and gym.make() kwargs still override. Carried to remote clients via the
# /tasks payload so direct and server modes resolve identically. Empty for envs
# that declare none.
_env_make_kwargs: dict[str, dict[str, Any]] = {}   # env_id -> {kwarg: default}
# env_id -> constructor/env kwargs that the env explicitly declares as accepted.
# Direct mode derives most of these from the env module's CFG.env_kwargs after
# import; remote mode receives the same set from /tasks and stores it here.
_env_supported_kwargs: dict[str, set[str]] = {}
_declared_env_ids: set[str] = set()                # lazy programmatic leaves
#: env_id -> BackendFamily — the SINGLE declared "kind of env":
#: warm strategy / reconcile mode / has-lifecycle all derive from this. Written by
#: :func:`lite.gym.services.register_family`; read via :func:`~.family_of`.
_families: dict[str, object] = {}
# Each registered env name → the routing mode used to populate ``_specs`` /
# ``_splits``. ``"local"`` = imported from disk; ``"remote"`` = stubs fetched
# via :func:`register_from_server`. Used by :func:`_import_env` to invalidate
# the cache when the routing mode changes (URL set vs. unset) between calls
# — otherwise an old remote stub would be returned to a caller that expects
# a local-instantiable entry_point (or vice versa).
_imported: dict[str, str] = {}                     # name -> "local" | "remote"
_imported_remote_urls: dict[str, str] = {}         # name -> server URL for remote stubs


def import_registration_modules(*, reload_cached: bool = False) -> None:
    """Import explicit out-of-tree env registration modules for direct mode."""
    raw = os.environ.get("CUA_LITE_REGISTRATION_MODULES", "").strip()
    for module_name in re.split(r"[,\s:]+", raw):
        if module_name:
            if reload_cached and module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)


def _has_env_registration(name: str) -> bool:
    prefix = f"{name}@"
    child_prefix = f"{name}."
    return (
        name in _splits
        or name in _env_make_kwargs
        or name in _env_supported_kwargs
        or any(key.startswith(child_prefix) for key in _splits)
        or any(key.startswith(child_prefix) for key in _env_make_kwargs)
        or any(key.startswith(child_prefix) for key in _env_supported_kwargs)
        or any(key.startswith(prefix) for key in _specs)
        or any(key.startswith(child_prefix) for key in _specs)
    )


def _env_name_chain(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


def _module_declared_env_kwargs(env_id: str) -> set[str]:
    """Env kwargs declared by an imported env module's config.

    Dotted sub-envs such as ``browsergym.webarena`` are often registered by an
    umbrella module (``browsergym.main``), so walk from the concrete env_id back
    toward its parent namespace and use the first imported module that exposes a
    ``CFG``. That is the only optional part of the read -- the umbrella ``main.py``
    of ``cua`` / ``lite`` / ``webharbor`` declares none; ``EnvConfig.env_kwargs`` is
    a declared non-optional ``dict``, so it is indexed rather than probed.
    """
    for env_name in _env_name_chain(env_id):
        mod = sys.modules.get(f"lite.gym.envs.{env_name}.main")
        cfg = getattr(mod, "CFG", None)
        if cfg is not None:
            return set(cfg.env_kwargs)
    return set()


def _reset_env_task_latches(name: str) -> None:
    """Clear registry and env-module lazy latches for ``name`` and parents."""
    for env_name in _env_name_chain(name):
        _tasks_registered.discard(env_name)
        _services_started.discard(env_name)
        mod = sys.modules.get(f"lite.gym.envs.{env_name}.main")
        if mod is not None and hasattr(mod, "_tasks_registered"):
            mod._tasks_registered = False


def _clear_env_registration(name: str) -> None:
    """Drop all catalog entries for one env_id before changing source."""
    child_prefix = f"{name}."
    affected_envs = {name}
    affected_envs.update(key for key in _splits if key.startswith(child_prefix))
    affected_envs.update(key for key in _env_make_kwargs if key.startswith(child_prefix))
    affected_envs.update(key for key in _env_supported_kwargs if key.startswith(child_prefix))
    affected_envs.update(key for key in _imported if key.startswith(child_prefix))
    affected_envs.update(key for key in _imported_remote_urls if key.startswith(child_prefix))
    affected_envs.update(
        key.split("@", 1)[0]
        for key in _specs
        if key.startswith(child_prefix)
    )
    for env_key in [key for key in _splits if key == name or key.startswith(child_prefix)]:
        _splits.pop(env_key, None)
    for env_key in [key for key in _env_make_kwargs if key == name or key.startswith(child_prefix)]:
        _env_make_kwargs.pop(env_key, None)
    for env_key in [
        key for key in _env_supported_kwargs
        if key == name or key.startswith(child_prefix)
    ]:
        _env_supported_kwargs.pop(env_key, None)
    for env_key in [
        key for key in _imported_remote_urls
        if key == name or key.startswith(child_prefix)
    ]:
        _imported_remote_urls.pop(env_key, None)
    for env_key in [key for key in _imported if key == name or key.startswith(child_prefix)]:
        _imported.pop(env_key, None)
    prefix = f"{name}@"
    for key in [k for k in _specs if k.startswith(prefix) or k.startswith(child_prefix)]:
        del _specs[key]
    for env_key in affected_envs:
        _reset_env_task_latches(env_key)


def _snapshot_env_registration(name: str) -> dict[str, Any]:
    """Capture current registrations so a failed source switch can roll back."""
    child_prefix = f"{name}."
    spec_prefix = f"{name}@"
    env_keys = {name}
    env_keys.update(key for key in _splits if key.startswith(child_prefix))
    env_keys.update(key for key in _env_make_kwargs if key.startswith(child_prefix))
    env_keys.update(key for key in _env_supported_kwargs if key.startswith(child_prefix))
    env_keys.update(key for key in _imported if key.startswith(child_prefix))
    env_keys.update(key for key in _imported_remote_urls if key.startswith(child_prefix))
    env_keys.update(
        key.split("@", 1)[0]
        for key in _specs
        if key.startswith(spec_prefix) or key.startswith(child_prefix)
    )
    return {
        "splits": {
            key: {split: list(ids) for split, ids in value.items()}
            for key, value in _splits.items()
            if key in env_keys
        },
        "env_make_kwargs": {
            key: dict(value)
            for key, value in _env_make_kwargs.items()
            if key in env_keys
        },
        "env_supported_kwargs": {
            key: set(value)
            for key, value in _env_supported_kwargs.items()
            if key in env_keys
        },
        "imported": {
            key: value
            for key, value in _imported.items()
            if key in env_keys
        },
        "imported_remote_urls": {
            key: value
            for key, value in _imported_remote_urls.items()
            if key in env_keys
        },
        "specs": {
            key: value
            for key, value in _specs.items()
            if key.startswith(spec_prefix) or key.startswith(child_prefix)
        },
        "tasks_registered": {key for key in env_keys if key in _tasks_registered},
        "services_started": {key for key in env_keys if key in _services_started},
    }


def _restore_env_registration(name: str, snapshot: dict[str, Any]) -> None:
    _clear_env_registration(name)
    _splits.update(snapshot["splits"])
    _env_make_kwargs.update(snapshot["env_make_kwargs"])
    _env_supported_kwargs.update(snapshot["env_supported_kwargs"])
    _imported.update(snapshot["imported"])
    _imported_remote_urls.update(snapshot["imported_remote_urls"])
    _specs.update(snapshot["specs"])
    _tasks_registered.update(snapshot["tasks_registered"])
    _services_started.update(snapshot["services_started"])


# Single-flight guard for :func:`_import_env`. In remote mode the first import of
# an env does a blocking HTTP fetch (``register_from_server``); without coalescing,
# N concurrent ``gym.make`` worker threads (the ``asyncio.to_thread`` rollout
# fan-out) all miss the ``_imported`` cache and each fire the fetch — a thundering
# herd (for webgym, N × ~95 MB). One ``threading.Lock`` per env_id (callers are
# worker threads, not coroutines) collapses the herd to a single fetch; the rest
# wait, then hit the warm cache. ``_import_locks`` is keyed by env_id and bounded
# by the (frozen) env catalog.
_import_locks: dict[str, threading.Lock] = {}
_import_locks_guard = threading.Lock()

#: env_id -> per-env-id capabilities object (subclasses any of
#: lite.gym.services.EnvServices),
#: registered via :func:`register_services`. The env-server discovers
#: each capability by ``isinstance(services, <Cap>)`` — typed dispatch,
#: no getattr-by-name. Envs without a services object (or without a
#: given capability) are simply skipped by the relevant dispatcher.
_services: dict[str, object] = {}


_ENVS_DIR = Path(__file__).parent / "envs"

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(
    key: str,
    entry_point: Callable[..., LiteBaseEnv],
    *,
    split: str = "train",
    metadata: LiteBaseMetadata | None = None,
    **kwargs: Any,
) -> None:
    """Register a single task.

    Args:
        key: "{env_id}@{task_id}" (e.g. "lite.osworld@osworld_chrome_030eeff7")
        entry_point: Factory function — called as entry_point(**kwargs) -> LiteBaseEnv
        split: Dataset split this task belongs to (default "train")
        metadata: Task metadata. Domain-specific fields go in ``others``.
            ``task_id`` and ``env_id`` are automatically set from the key under
            ``others``, so callers do not need to duplicate identity.
        **kwargs: Default kwargs passed to entry_point when make() is called

    Example:
        register("osworld@osworld_chrome_030eeff7", create_env_open_browser, split="eval",
                 metadata=LiteCUAMetadata(dims=("desktop", "use"),
                                          others={"domain": "chrome"}))
        # → metadata.others["task_id"] == "osworld_chrome_030eeff7"
        # → metadata.others["env_id"] == "osworld"

    Env-author contract (read this if you're adding a new env):

    * **Bounded internal pool / queue / port range**: if your env's
      ``reset()`` can wait on an exhaustible resource, raise
      :class:`~lite.gym.errors.CapacityExhausted` from that wait — do
      NOT block forever. The env-server's exception handler maps it to
      HTTP 503 + Retry-After; the client retries retry-safe endpoints
      such as ``reset``. Do not rely on client-side replay for ``step``:
      until request-id dedupe exists, ``step`` 503s are surfaced as
      typed immediate failures.
      See ``androidworld`` (port range via :mod:`lite.gym.utils.backend.ports`),
      ``webgym`` / ``mobilegym`` (in-container slot exhaustion surfaced
      over the RPC client) as references.

    * **Stateless / lightweight (no bounded internal resource)**: no
      typed exception needed. L1 host sensors + L2 in-flight cap on the
      env-server side handle backpressure. Reference: ``screenspot_pro``,
      ``osworld_g``.
    """
    env_id, task_id = split_key(key)

    # Inject task_id + env_id into others so they are queryable via metadata
    # filters regardless of env type.
    # Creates a new metadata object (never mutates the caller's instance —
    # callers sometimes reuse the same metadata object for multiple keys).
    if metadata is not None:
        import dataclasses
        metadata = copy.deepcopy(metadata)
        metadata = dataclasses.replace(
            metadata,
            others={**metadata.others, "task_id": task_id, "env_id": env_id},
        )

    # Idempotent re-registration: a second register() for the same key
    # overwrites the spec but must not duplicate the task in its split list
    # (task_ids() feeds --sample/--head selection and parity tests). The
    # membership scan runs ONLY on the rare re-registration path — first-time
    # registration (the 100k+-task bulk loops) stays O(1) per task.
    is_rereg = key in _specs
    _specs[key] = EnvSpec(key=key, entry_point=entry_point,
                          kwargs=kwargs, metadata=metadata)
    split_ids = _splits.setdefault(env_id, {}).setdefault(split, [])
    # (Re-registration under a DIFFERENT split would leave the task_id stale
    # in the old split's list — unreachable today: a task's split is fixed by
    # the file/list it registers from, so an identical key re-registers into
    # the same split.)
    if not is_rereg or task_id not in split_ids:
        split_ids.append(task_id)


# ---------------------------------------------------------------------------
# Discovery — lazy import of envs/{name}/main.py
# ---------------------------------------------------------------------------

def _import_env(name: str, server_url: str | None = None) -> None:
    """Populate ``_splits[name]`` either by local import or HTTP fetch.
    No-op if already imported.

    Routing is keyed on an explicit ``server_url`` when supplied, otherwise on
    ``CUA_LITE_ENV_SERVER_URL``, the same switch :func:`make` uses for instance
    creation:

    - URL set     → fetch the task list from the env server and register
                    stub specs (entry_point :func:`_remote_only_placeholder`,
                    never invoked — :func:`make` routes to
                    :class:`LiteEnvClient` instead). The (possibly dotted)
                    name is sent to the server AS-IS — the server is the
                    source of truth for env ids, and the local envs/ layout
                    is irrelevant in client mode.
    - URL not set → import the env module locally; its module-level
                    ``_register_tasks()`` (or equivalent) populates
                    ``_splits``. Dotted names resolve to nested directories
                    (``lite.osworld`` → ``envs/lite/osworld/main.py``) or,
                    when the sub-env has no directory of its own, to the
                    umbrella parent's ``main.py`` whose import registers
                    every dotted sub-env (``browsergym.*``).
                    If deps are missing, the env's ``try / except
                    ImportError`` wrapper raises
                    :class:`EnvDepsMissingError` with an install hint
                    that propagates to the caller.

    Unifying the routing eliminates the prior "mixed mode" where task
    list came from local import but instance creation went remote —
    sometimes producing local-vs-server skew when the env code differed
    between client and server.

    Plain ``ImportError`` (a genuine bug — circular import, missing
    internal module) always propagates so operators see it loudly.
    """
    server_url = server_url or routing_server_url()
    desired_mode = "remote" if server_url else "local"
    if _imported.get(name) == desired_mode and (
        not server_url or _imported_remote_urls.get(name) == server_url
    ):
        return                                     # fast path — lock-free
    # Single-flight: coalesce concurrent first-imports of the same env_id so the
    # blocking register_from_server fetch (remote mode) runs once, not once per
    # to_thread worker. Recursion into the public _import_env(parent) below takes
    # the parent's lock (child→parent, acyclic); ensure_services → _import_env
    # nests under _services_lock (direct mode) — also one-directional. No cycle.
    with _import_locks_guard:
        lock = _import_locks.setdefault(name, threading.Lock())
    with lock:
        _import_env_locked(name, server_url, desired_mode)


def _import_env_locked(name: str, server_url: str | None, desired_mode: str) -> None:
    """Body of :func:`_import_env`, run under its per-env single-flight lock.

    Double-checks the cache (a peer thread may have populated it while we waited
    on the lock), then imports the env module locally or fetches stub specs from
    the server. ``_imported[name]`` is set ONLY on success, so a failed fetch
    never poisons the cache (a retry re-attempts).
    """
    cached_mode = _imported.get(name)
    cached_url = _imported_remote_urls.get(name)
    if cached_mode == desired_mode and (
        not server_url or cached_url == server_url
    ):
        return
    has_existing_registration = _has_env_registration(name)
    refresh_registration = cached_mode is not None or (
        server_url is not None and has_existing_registration
    )
    registration_snapshot: dict[str, Any] | None = None
    if refresh_registration and server_url:
        registration_snapshot = _snapshot_env_registration(name)
    elif refresh_registration:
        # Mode flipped (URL toggled between calls). Drop the prior
        # registration, or remote URL switched. Also clear transitive child
        # registrations that may exist even when ``_imported[name]`` was never
        # marked (umbrella modules can register dotted children on import).
        # Otherwise callers can see a stub from the wrong server, stale local
        # tasks in remote mode, or an empty local catalog after switching back.
        _clear_env_registration(name)

    if server_url:
        logger.debug(
            "remote-client mode (url=%s): fetching task list for env %r via HTTP",
            server_url, name,
        )
        # The server is the source of truth for env ids in client mode —
        # ask it for the (possibly dotted) name as-is instead of probing
        # the local envs/ tree. Families that register several dotted
        # env_ids from one umbrella main.py (browsergym.*) have
        # no per-sub-env directory, so the local-layout fallback below
        # would wrongly resolve to the parent name and 404 on the server.
        # Late import: lite.gym.remote depends on lite.gym types; pulling it
        # at module-load time risks circular imports.
        from lite.gym.remote.client import register_from_server
        if refresh_registration:
            _clear_env_registration(name)
            try:
                register_from_server(name, server_url)
            except Exception:
                if registration_snapshot is not None:
                    _restore_env_registration(name, registration_snapshot)
                raise
        else:
            register_from_server(name, server_url)
        # Env-specific agent protocols/adapters (e.g. browsergym.generic) are
        # model-side bridges in ``lite/agents/extensions/`` and register via
        # ``import lite.agents.extensions`` (pulled in by ``lite.agents.models``,
        # which every rollout/RL path imports before resolving a protocol_key).
        # The env subsystem no longer reaches into ``lite.agents`` to register
        # them — keeps ``lite.gym`` free of any ``lite.agents`` dependency.
    else:
        main_py = _ENVS_DIR / Path(*name.split(".")) / "main.py"
        import_registration_modules()
        if not _has_env_registration(name):
            # Registration modules are side-effect entrypoints. If the module
            # was imported before this env's registry entries were cleared
            # (for example by a local/remote mode refresh), a plain import is a
            # cache hit and does not replay register(...). Reload only on the
            # miss path so steady-state direct make() stays cheap.
            import_registration_modules(reload_cached=True)
        if _has_env_registration(name) and not main_py.is_file():
            from lite.gym.services import validate_declarations
            validate_declarations(name)
            _imported[name] = desired_mode
            return

        if not main_py.is_file():
            # Compound name with no own main.py: try parent namespace.
            # E.g. "browsergym.miniwob" → "browsergym" (umbrella main.py
            # registers all its dotted sub-envs on import).
            if "." in name:
                parent = name.rsplit(".", 1)[0]
                parent_module = f"lite.gym.envs.{parent}.main"
                if refresh_registration and parent_module in sys.modules:
                    importlib.reload(sys.modules[parent_module])
                else:
                    _import_env(parent)
                # Declaration finalizer — validate BEFORE caching so a failed
                # check re-raises on retry instead of poisoning the fast path.
                from lite.gym.services import validate_declarations
                validate_declarations(name)
                # Mark this dotted sub-env imported: its tasks were just
                # registered transitively by the umbrella parent's main.py, so
                # the lock-free fast path trips next call. Without this,
                # browsergym.* sub-envs re-enter _import_env on every call.
                _imported[name] = desired_mode
            return
        try:
            module_name = f"lite.gym.envs.{name}.main"
            if refresh_registration and module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
        except EnvDepsMissingError as e:
            # Re-raise with a hint pointing at the remote-mode escape hatch,
            # so direct-mode users hitting a missing-deps error see the
            # alternative route without having to read the docs.
            raise EnvDepsMissingError(
                what=(
                    f"{e.what}\n"
                    f"  Alternative: set CUA_LITE_ENV_SERVER_URL=<server> "
                    f"to use a remote env-server instead of installing deps locally."
                ),
                install=e.install,
                see=e.see,
            ) from e
    if not server_url:
        # Mode-independent finalizer: cross-check the declarations
        # this import just registered (family ⟺ services ⟺ live_ids) so a
        # mis-declaration fails at first make/task-probe in DIRECT mode too —
        # not only at server boot. Validate BEFORE caching so a failure
        # re-raises on retry instead of poisoning the fast path.
        from lite.gym.services import validate_declarations
        validate_declarations(name)
    else:
        _imported_remote_urls[name] = server_url
    _imported[name] = desired_mode


def _import_all() -> None:
    """Import all env modules under envs/. Idempotent via _import_env caching.

    An env is skipped ONLY on the TYPED conditions that mean *"nothing here is
    broken; this env just is not usable on this host"*:

    - :class:`EnvDepsMissingError` (local path) — the env's own
      ``try / except ImportError`` wrapper says its optional deps or install.sh
      materials are absent. :meth:`Registry.registered_env_ids` then surfaces the
      env as ``available: false`` with the install hint.
    - :class:`EnvUnavailable` (remote path) — the env-server answered 404/501 for
      this env_id, i.e. *that server* does not serve it. Expected for every
      umbrella directory (``lite``, ``cua``, ``browsergym``, ``webharbor``),
      whose dotted sub-envs are what the server actually registers.
    - :class:`EnvServerUnavailable` (remote path) — the whole server is
      unreachable / refuses the token / is version-skewed. Not per-env
      unavailability, but a catalog probe cannot answer without a server either,
      and it is the same answer for all 16 envs, so it skips rather than aborting
      the sweep.

    Everything else PROPAGATES, loudly. A blanket ``except Exception`` here used
    to let ANY import-time failure — a syntax error, a bug, a circular import, a
    contradictory committed lock — decide that the env simply *is not on this
    host*, at ``logger.debug``: the handler chose the diagnosis, and the
    diagnosis was always "absent". The distinction it erased is the only one that
    matters here — *unavailable* is a host fact, *broken* is a bug.
    ``_import_env``'s own docstring already promised it, and
    :meth:`Registry.registered_env_ids` already applies exactly this policy one
    layer up.

    So a broken env aborts the sweep, and at server boot (``recover_all`` → here)
    it aborts boot. That is the point: booting a server whose catalog is silently
    missing an env is the failure mode this rule exists to prevent.
    """
    if not _ENVS_DIR.is_dir():
        return
    for env_dir in sorted(_ENVS_DIR.iterdir()):
        if env_dir.is_dir() and (env_dir / "main.py").exists():
            try:
                _import_env(env_dir.name)
            except (EnvDepsMissingError, EnvUnavailable, EnvServerUnavailable) as e:
                logger.debug("Env '%s' is unavailable here (%s)", env_dir.name, e)

# ---------------------------------------------------------------------------
# Lifecycle hook: ensure_services(env_id)
# ---------------------------------------------------------------------------
#
# Some envs require out-of-process auxiliary services to be up before
# any gym.make can succeed:
#   * browsergym.miniwob    — local HTTP server serving miniwob HTML
#   * browsergym.webarena   — Gitlab / shopping / reddit / wiki docker stack
#   * webgym                — OmniBoxes master + nodes + Redis
# These are NOT per-env (one shared service handles many env instances).
# Footprint is fixed (~50 MB + ~0.01 vCPU per service).
#
# Contract: an env's registered services object (register_services) may override
#   ``EnvServices.ensure(env_id: str) -> None``
# (this free function dispatches to it). The method:
#   * MUST be idempotent (probe-then-start; subsequent calls fast no-op)
#   * MUST be safe to call concurrently (file-lock or in-proc lock)
#   * MUST raise :class:`lite.gym.errors.EnvDepsMissingError` on
#     unrecoverable failure (e.g. install.sh not run, port conflict)
#   * Receives the SPECIFIC env_id ("browsergym.miniwob") so umbrella
#     envs can dispatch on sub-key.
# Envs without the hook (e.g. androidlab/world — services live inside
# the spawned docker container, no shared external service) are not
# required to define it.
#
# Call site: this module's :func:`make`, after :func:`_import_env`,
# before constructing the env. Per-env_id cache makes the steady-state
# cost one probe — typically <1 ms.

_services_started: set[str] = set()
_tasks_registered: set[str] = set()
_services_lock = threading.Lock()


def _is_registered(env_id: str) -> bool:
    """True once ``env_id``'s catalog is actually populated — the env itself
    (leaf) or any of its sub-envs (umbrella) appears in ``_splits``. Used to
    cache ``register_tasks`` only on SUCCESS: a data-absent ``register_tasks()``
    no-ops (osworld_g / screenspot_pro / webgym re-check data presence each
    call), so caching unconditionally would pin the empty result for the process
    and never re-register after the data lands."""
    return env_id in _splits or any(s.startswith(env_id + ".") for s in _splits)


def ensure_registered(env_id: str) -> None:
    """Idempotently register an env_id's task catalog (``_splits`` / ``_specs``)
    — cheap, NO backend boot.

    The catalog-probe counterpart to :func:`ensure_services`: fired by
    ``task_ids`` / ``registered_env_ids`` when an env imported OK but registered
    no tasks (a lazy env). Calls ``register_tasks(env_id)`` on the env's
    ``EnvServices`` — DISTINCT from ``ensure``, so probing a webgym/browsergym
    task list never boots its container (the trap the old eager-at-import
    registration worked around). No-op in remote-client mode (the server owns
    registration). Cached per process via ``_tasks_registered``.
    """
    if routing_server_url():
        return
    with _services_lock:
        if env_id in _tasks_registered:
            return
        _import_env(env_id)
        from lite.gym.services import EnvServices, services_for
        svc = services_for(env_id)
        if isinstance(svc, EnvServices):
            svc.register_tasks(env_id)
        if _is_registered(env_id):
            _tasks_registered.add(env_id)


def ensure_services(env_id: str) -> None:
    """Idempotently start any out-of-process services this env_id needs.

    Calls ``ensure(env_id)`` on the env's registered ``EnvServices``
    object if it has one.
    Cached per process — first call may take seconds (probe
    + subprocess); subsequent calls return immediately. See module
    docstring above for the contract.

    No-op in remote-client mode (``CUA_LITE_ENV_SERVER_URL`` set):
    service startup is the env-server's responsibility, and a typical
    slime client container lacks the host-level prerequisites
    (docker.sock, browser CLIs) needed to even probe. :func:`make`
    already skips this call in remote mode, but the guard here also
    protects callers that invoke ``ensure_services`` directly (e.g. for
    explicit warm-up before a hot path).

    In direct mode, the hook may raise :class:`EnvDepsMissingError`,
    which propagates with the env's install hint so callers know what
    to install (or can switch to remote mode by setting
    ``CUA_LITE_ENV_SERVER_URL``).
    """
    if routing_server_url():
        logger.debug(
            "remote-client mode: skipping ensure_services(%s) — server-side responsibility",
            env_id,
        )
        return
    with _services_lock:
        if env_id in _services_started:
            return
        _import_env(env_id)
        # Typed dispatch: the env's registered services object (umbrella-aware)
        # gets ``ensure(env_id)`` if it has one. Envs with no services object
        # (the simplest onboarding — most envs) are a no-op, cached so
        # subsequent make() calls skip the lookup.
        from lite.gym.services import EnvServices, services_for
        svc = services_for(env_id)
        if isinstance(svc, EnvServices):
            # Register the task catalog before acquiring the backend, so a lazy
            # env (webgym/browsergym) is registered exactly once whether it is
            # reached via a catalog probe (ensure_registered) or a make() (here).
            if env_id not in _tasks_registered:
                svc.register_tasks(env_id)
                if _is_registered(env_id):
                    _tasks_registered.add(env_id)
            svc.ensure(env_id)
        _services_started.add(env_id)


def invalidate_services(env_id: str) -> None:
    """Evict ``env_id`` from the ensure cache so the NEXT make() re-runs
    ``svc.ensure()`` — the self-heal seam for a backend that died mid-run.
    Without this, the registry-level cache short-circuits before the env's
    own ensure ever sees the death (browsergym's reset-warming eviction was
    dead code until it called this)."""
    with _services_lock:
        _services_started.discard(env_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Registry:
    """Gymnasium-style registry object.

    Usage:
        registry.env_ids()                     # ['lite.demo', 'lite.osworld', 'webgym']
        registry.task_ids("osworld")           # {'test': [...]}
        registry.task_ids("osworld", split="eval")  # [task_id, ...]
    """

    def registered_env_ids(self) -> list[str]:
        """Return all *usable* env_ids on this host: dir-scan ∪ programmatic.

        Walks :data:`_splits` (everything ``register()`` was called for,
        including programmatic sub-envs like ``browsergym.miniwob`` with no
        own dir) AND backfills two kinds of leaf envs that aren't in
        ``_splits`` yet:

        - dir-scan envs that are UNAVAILABLE here — the typed conditions
          :func:`_import_all` skips on — surfaced as ``available: false`` so
          callers see the install hint. A *broken* env is not in this set: it
          raises out of ``_import_all`` rather than being listed as absent.
        - dir-scan envs that imported OK but haven't registered any tasks
          AND have no sub-envs in ``_splits`` (i.e., leaf envs with lazy
          registration via :func:`ensure_registered` — :mod:`osworld_g`,
          :mod:`screenspot_pro`, and the now-lazy :mod:`webgym` /
          :mod:`browsergym`). Probing them through :meth:`task_ids` will
          fire ``ensure_registered`` (which calls ``register_tasks`` WITHOUT
          booting the backend) and either register or raise
          ``EnvDepsMissingError``.

        Pure umbrellas (e.g. ``browsergym``, ``lite`` — their ``main.py``
        only imports sub-envs) successfully import but never appear in
        ``_splits``; they ARE excluded here because their sub-envs are.

        Cost: triggers every env's ``main.py`` import — slow on first call
        (~seconds aggregate), fast on subsequent (cached). Use :meth:`env_ids`
        for the cheap directory-scan path (no imports, no programmatic
        sub-envs).
        """
        _import_all()
        # Task registration is lazy; fire it for every imported env so the
        # umbrella-vs-leaf test below sees the real sub-envs. Without this a lazy
        # umbrella (browsergym) has no ``browsergym.*`` in ``_splits`` → it is
        # mis-listed as a bare (non-makeable) leaf, and GET /envs/browsergym/tasks
        # then 404s. Idempotent + cached; a data-absent leaf raises
        # EnvDepsMissingError → stays unregistered and is surfaced as a lazy leaf
        # in the loop below (unchanged behaviour for osworld_g / screenspot_pro).
        for _eid in self.env_ids():
            if _eid in _imported:
                try:
                    ensure_registered(_eid)
                except EnvDepsMissingError:
                    # Data-absent lazy leaf (osworld_g / screenspot_pro before
                    # their datasets are fetched) — surfaced as a lazy leaf
                    # below. Anything else (KeyError, bad data) is a real
                    # registration bug and must propagate, not be silently
                    # swallowed as "lazy".
                    pass
        ids = set(_splits)
        ids.update(_declared_env_ids)
        for env_id in self.env_ids():
            if env_id in ids:
                continue
            # A namespace imported transitively by an umbrella may not have its
            # own _imported marker. Declared/registered children are sufficient.
            if any(s.startswith(env_id + ".") for s in ids):
                continue
            if env_id not in _imported:
                # Failed import: surface as unavailable
                ids.add(env_id)
                continue
            # Imported OK with no declared/registered children: lazy leaf.
            ids.add(env_id)
        return sorted(ids)

    def env_ids(self) -> list[str]:
        """Return env names discoverable via directory scan (no imports).

        Walks nested namespaces — a directory whose ``main.py`` is purely an
        umbrella (re-importing sub-envs) is NOT a leaf; its sub-dirs that
        contain their own ``main.py`` are. E.g. ``envs/lite/osworld/main.py``
        yields env_id ``"lite.osworld"`` (umbrella ``envs/lite/main.py`` is
        not surfaced as a leaf).

        Fast path: does not import anything. Sub-envs registered PURELY
        programmatically — e.g. ``browsergym.miniwob``, ``browsergym.webarena``,
        ``browsergym.visualwebarena`` (all registered when
        ``browsergym/main.py`` is imported, with no per-sub-env directory)
        — are NOT listed here. Use :meth:`registered_env_ids` to get the
        full list (at the cost of an import side-effect).
        """
        if not _ENVS_DIR.is_dir():
            return []
        leaves: list[str] = []
        for top in sorted(_ENVS_DIR.iterdir()):
            if not top.is_dir() or not (top / "main.py").is_file():
                continue
            sub_dirs = sorted(
                d for d in top.iterdir()
                if d.is_dir() and (d / "main.py").is_file()
            )
            if sub_dirs:
                # Umbrella: surface each sub-env as "{top}.{sub}"
                leaves.extend(f"{top.name}.{d.name}" for d in sub_dirs)
            else:
                leaves.append(top.name)
        return leaves

    def _has_registered_children(self, env_id: str) -> bool:
        """Whether anything is registered UNDER *env_id* (making it an umbrella).

        The predicate :meth:`registered_env_ids` uses to drop a pure umbrella
        (``browsergym``, ``cua.bench``, ``lite.cuaworld``): it is a namespace,
        not a makeable env, so listing it would advertise a ``gym.make`` that
        cannot work. Assumes *env_id*'s own registration already ran — only
        *env_id* can declare its children.
        """
        prefix = env_id + "."
        return any(s.startswith(prefix) for s in _splits) or any(
            s.startswith(prefix) for s in _declared_env_ids
        )

    def _registration_owner(self, env_id: str) -> str | None:
        """The one env whose registration decides *env_id*'s existence.

        A programmatic sub-env is declared by its nearest DIRECTORY-scanned
        ancestor — ``lite.cuaworld`` for ``lite.cuaworld.ardour``,
        ``cua.bench`` for ``cua.bench.local.basic``, ``browsergym`` for
        ``browsergym.miniwob``. Walking :func:`_env_name_chain` longest-first
        finds it; taking ``env_id.split(".")[0]`` does not (it yields ``lite``
        and ``cua``, which own no ``main.py`` of their own as leaves), and that
        mistake makes every ``lite.cuaworld.*`` env answer "unknown".
        """
        scanned = set(self.env_ids())
        return next((a for a in _env_name_chain(env_id) if a in scanned), None)

    def is_known_env_id(self, env_id: str) -> bool:
        """Whether *env_id* is in :meth:`registered_env_ids`, registering AT MOST ONE env.

        Same answer as ``env_id in registered_env_ids()`` — pure umbrellas
        excluded, failed imports still known — at a cost proportional to the
        env asked about. The expensive spelling fires ``ensure_registered`` for
        EVERY env, so a single-env probe paid webgym's HuggingFace load (~24 s)
        and browsergym's (~21 s); that is what ``GET /envs/{env_id}`` used to
        do, despite its docstring promising no full registry walk.

        Only *env_id*'s registration owner can settle the question, so only
        that one is imported and registered — and for a plain leaf like
        ``mobilegym`` that costs nothing, while the caller of a webgym probe
        was going to load webgym's catalog regardless.
        """
        if env_id in _splits or env_id in _declared_env_ids:
            return True
        owner = self._registration_owner(env_id)
        if owner is None:
            return False
        try:
            _import_env(owner)
            ensure_registered(owner)
        except Exception:
            # A dir-scanned env whose import/registration is broken is still
            # KNOWN: registered_env_ids surfaces it so the caller gets
            # ``available: false`` plus the install hint instead of a 404.
            pass
        if env_id in _splits or env_id in _declared_env_ids:
            return True
        return env_id == owner and not self._has_registered_children(env_id)

    def sub_env_ids(self, env_id: str) -> list[str]:
        """Sub-envs of *env_id* that :meth:`registered_env_ids` would list.

        Call :func:`ensure_registered` on *env_id* first — a lazy umbrella
        declares its children from ``register_tasks``, not at import. Reading
        the registry directly afterwards keeps the cost proportional to the env
        asked about; the natural-looking alternative
        (``[e for e in registered_env_ids() if e.startswith(env_id + ".")]``)
        registers EVERY other env to enumerate one env's children, which is how
        warming a single scoped singleton ended up loading webgym's catalog
        from HuggingFace.
        """
        prefix = env_id + "."
        ids = {e for e in _splits if e.startswith(prefix)}
        ids.update(e for e in _declared_env_ids if e.startswith(prefix))
        # A scanned child is listed only if it is a leaf; a nested umbrella
        # (``cua.bench`` under ``cua``) is dropped, mirroring registered_env_ids.
        ids.update(
            e for e in self.env_ids()
            if e.startswith(prefix) and not self._has_registered_children(e)
        )
        return sorted(ids)

    def task_ids(
        self,
        env_id: str,
        *,
        split: str | None = None,
    ) -> dict[str, list[str]] | list[str]:
        """Return task_ids for an env, optionally filtered by split.

        Args:
            env_id: Env name (e.g. "lite.demo", "lite.osworld").
            split:  If None, return ``{split: [task_ids]}`` dict for all splits.
                    If given, return flat ``[task_ids]`` for that specific split.

        Lazy envs (e.g. ``osworld_g``, ``screenspot_pro``, ``webgym``,
        ``browsergym``) defer task registration to ``ensure_registered`` —
        if the env imported OK but has no entries in ``_splits``, this method
        fires ``ensure_registered`` once (which calls the env's
        ``register_tasks`` hook WITHOUT booting the backend) to give it a
        chance to register, letting any
        :class:`lite.gym.errors.EnvDepsMissingError` propagate so the caller
        (e.g. ``GET /envs/{env_id}``) surfaces the install hint verbatim.
        """
        _import_env(env_id)

        if env_id not in _splits:
            # Lazy registration: a catalog probe registers the task list WITHOUT
            # booting the backend. ensure_registered() calls the env's
            # register_tasks() hook (which may populate _splits or raise
            # EnvDepsMissingError) — DISTINCT from ensure_services(), so probing
            # a webgym/browsergym task list never starts its OmniBoxes/docker
            # backend. Cached per process via _tasks_registered.
            ensure_registered(env_id)

        if env_id not in _splits:
            # Not makeable. Give a precise diagnosis: a namespace (its tasks live
            # under sub-envs) vs. an unknown/absent env. registered_env_ids() imports
            # everything + fires lazy registration, so sub-envs surface if present.
            makeable = self.registered_env_ids()
            subs = [e for e in makeable if e.startswith(env_id + ".")]
            if subs:
                raise KeyError(
                    f"'{env_id}' is a namespace, not a makeable env — its envs are "
                    f"{subs}. Use registry.registered_env_ids() to list makeable envs."
                )
            raise KeyError(
                f"Unknown env '{env_id}'. Makeable envs: {makeable}. "
                f"(An env whose tasks come from install.sh materials or a running "
                f"env-server won't appear until those are present — see "
                f"docs/envs.md#installation.)"
            )

        env_splits = _splits[env_id]

        if split is None:
            return {s: list(ids) for s, ids in env_splits.items()}

        if split not in env_splits:
            raise KeyError(
                f"Split '{split}' not found in env '{env_id}'. "
                f"Available splits: {list(env_splits.keys())}"
            )
        return env_splits[split]

    def task_metadata(self, env_id: str, task_id: str) -> LiteBaseMetadata | None:
        """Return metadata for a specific task, or None if not found.

        Fires lazy registration like :meth:`task_ids` (``_import_env`` +
        ``ensure_registered`` when the env isn't in ``_splits`` yet). A lazy env
        (webgym / browsergym / osworld_g / screenspot_pro) registers its catalog
        only on demand, so a bare ``_specs`` read would miss it before the first
        cold ``make``. Without this the server create hot path reads ``None`` for a
        not-yet-registered lazy-env task and silently skips the conflict gate —
        ``conflict_keys`` / ``mutating`` live in ``metadata.others``."""
        _import_env(env_id)
        if env_id not in _splits:
            ensure_registered(env_id)
        spec = _specs.get(f"{env_id}@{task_id}")
        if spec is None or spec.metadata is None:
            return None
        return copy.deepcopy(spec.metadata)

    def task_kwargs(self, env_id: str, task_id: str) -> dict[str, Any]:
        """Carried make kwargs for a task spec (see :data:`CARRIED_SPEC_KWARGS`).

        The subset of registered ``spec.kwargs`` that ``make()`` must resolve
        identically in direct and server mode. This includes wrapper-owned make
        config (timeouts / loop_detect) and env-owned make config such as
        ``cursor``.
        The env-server ships these to the client so ``register_from_server``
        rebuilds stub specs with the same resolved behavior as direct mode.
        Mirrors :meth:`task_metadata`'s lazy-registration handling."""
        _import_env(env_id)
        if env_id not in _splits:
            ensure_registered(env_id)
        spec = _specs.get(f"{env_id}@{task_id}")
        if spec is None:
            return {}
        return {k: spec.kwargs[k] for k in CARRIED_SPEC_KWARGS if k in spec.kwargs}

    def set_env_make_kwargs(self, env_id: str, kwargs: dict[str, Any]) -> None:
        """Declare an env's env-wide make-kwarg defaults (layer 2 — see make()).

        The common/uniform way for an env to set make config once from its
        default.yaml ``make_kwargs`` instead of repeating it in every per-task
        ``register()`` call. This covers wrapper-owned make config
        (step_timeout, ...) and env-owned make config (currently ``cursor``).
        Filtered to
        :data:`CARRIED_SPEC_KWARGS` so only recognized make kwargs become env
        defaults; stray keys cannot silently ride along. Per-task ``register()``
        kwargs and ``gym.make()`` kwargs still override these. Idempotent; last
        call wins."""
        _env_make_kwargs[env_id] = {
            k: v for k, v in kwargs.items() if k in CARRIED_SPEC_KWARGS
        }

    def env_make_kwargs(self, env_id: str) -> dict[str, Any]:
        """Env-wide make-kwarg defaults for *env_id* (empty if none declared)."""
        return _env_make_kwargs.get(env_id, {})

    def set_env_supported_kwargs(
        self,
        env_id: str,
        kwargs: list[str] | set[str] | tuple[str, ...],
    ) -> None:
        """Declare env-owned kwargs accepted by *env_id*.

        Direct mode usually infers these from an imported env module's
        ``CFG.env_kwargs``. Remote mode cannot import the server's env module,
        so ``register_from_server`` stores the server-advertised set here.
        """
        _env_supported_kwargs[env_id] = set(kwargs)

    def env_supported_kwargs(self, env_id: str) -> list[str]:
        """Return env-owned kwargs accepted by *env_id*.

        This is capability metadata, not construction config. It lets generic
        harness code decide whether a soft kwarg such as ``seed`` may be
        supplied without encoding env-id/action-space allowlists.
        """
        _import_env(env_id)
        if env_id not in _splits:
            ensure_registered(env_id)

        supported = set(_env_supported_kwargs.get(env_id, set()))
        supported.update(_module_declared_env_kwargs(env_id))

        prefix = f"{env_id}@"
        # SNAPSHOT, not a live view. ``_specs`` is mutated by ``register()`` on any
        # thread (``_import_locks`` is per-env_id and guards only ``_import_env``,
        # not this dict), so a concurrent registration of ANOTHER env raises
        # ``RuntimeError: dictionary changed size during iteration`` here and the
        # env-server answers 500. Observed live: two clients hitting
        # ``GET /envs/browsergym.miniwob/tasks`` at once. No host test can see it --
        # it needs two concurrent clients against a running server.
        for key, spec in list(_specs.items()):
            if key.startswith(prefix):
                supported.update(spec.kwargs)
        return sorted(supported)

    def env_supports_kwarg(self, env_id: str, kwarg: str) -> bool:
        """Whether *env_id* declares support for env-owned kwarg *kwarg*."""
        return kwarg in self.env_supported_kwargs(env_id)

    def declare_env_id(self, env_id: str) -> None:
        """Expose a programmatic lazy env before its task materials are installed."""
        _declared_env_ids.add(env_id)

registry = Registry()
