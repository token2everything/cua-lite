"""Env factory — ``make()`` construction of a registered env spec.

Split out of ``lite.gym.registry`` (which stayed the catalog + Registry class +
import machinery) so the 200-line two-path (local / remote-client) factory +
wrapper layer lives on its own. ``lite.gym.__init__`` imports ``make`` from
here — this module is its owner, not a re-export target.

Reads the registry's monkeypatchable state (``_specs``/``_splits``) and import
machinery (``_import_env``/``_import_all``/``ensure_services``) through the
registry MODULE object bound below — NOT ``import lite.gym.registry as r``,
which would hand back the ``registry`` SINGLETON that ``lite.gym.__init__``
binds over the submodule name. Module access keeps tests that
``monkeypatch.setattr(registry, "_specs", ...)`` observable here.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from typing import Any

from lite.gym.base import LiteBaseEnv
from lite.gym.utils.server.routing import routing_server_url
from lite.utils.registry import split_key

logger = logging.getLogger("lite.gym.registry")

#: The registry MODULE object (NOT the ``registry`` singleton attribute that
#: ``lite.gym.__init__`` binds over the submodule name — see the module
#: docstring). State is read through this object rather than bound as locals so
#: monkeypatched ``_specs`` / ``_splits`` replacements are observed live. Same
#: idiom as :mod:`lite.gym.remote.recovery`.
_reg = importlib.import_module("lite.gym.registry")


def make(
    key: str,
    **kwargs: Any,
) -> LiteBaseEnv:
    """Create an env from a registered spec.

    Two construction paths share one wrapper policy:
      - Local: ``spec.entry_point(**merged)`` → raw LiteBaseEnv
      - Remote: :class:`LiteEnvClient` (when ``env_server_url`` or
        ``$CUA_LITE_ENV_SERVER_URL`` is set) → raw LiteBaseEnv

    Timeout ownership is mode-specific: local construction applies
    ``StepTimeoutWrapper`` in-process, while remote construction forwards
    ``step_timeout``/``reset_timeout`` to the server and sizes the HTTP RPC
    timeout to outlive the server-side wrapper. Loop detection remains a
    client-side concern because the client wrapper already sees every action
    before it goes over the wire, so server-side detection is redundant.
    Cursor compositing is env-owned at screenshot capture time.

    Args:
        key: "{env_id}@{task_id}" (e.g. "lite.demo@create_file", "lite.osworld@osworld_chrome_030eeff7")
        **kwargs: Passed to the entry_point factory.
            Common kwargs: computer, max_steps, post_action_delay, resolution.

    Returns:
        A ready-to-use LiteBaseEnv.
    """
    if "@" not in key:
        raise ValueError(
            f"Invalid key '{key}'. Expected format: '{{env_id}}@{{task_id}}' "
            f"(e.g. 'lite.demo@create_file')"
        )

    env_id, task_id = split_key(key)
    # Resolve env_server_url early so we can route past local-only checks
    # when the caller is going to use the HTTP client anyway. ``merged``
    # is built below from spec defaults + kwargs; here we just need the
    # final resolved value, so peek at kwargs + env without consuming.
    env_server_url = (
        kwargs.get("env_server_url")
        or routing_server_url()
    )
    # Only import the requested env — avoids loading unrelated heavy modules.
    # In explicit remote mode this fetches the server task payload before any
    # local env import/dependency probe can run.
    _reg._import_env(env_id, server_url=env_server_url)
    # Start any out-of-process services this env_id needs (e.g. androidworld's
    # docker image present, webgym's OmniBoxes master, lite.osworld's docker
    # image). Idempotent: per-process cache means steady-state is one probe.
    # SKIPPED in remote-client mode: ensure_services is the server's
    # responsibility (its own startup probe runs the same hook), and a
    # slime-style client container typically has none of the local deps
    # (no docker CLI, no docker.sock, no OmniBoxes) — running these checks
    # there would falsely block an otherwise-valid HTTP-only rollout.
    if env_server_url:
        logger.debug(
            "remote-client mode (url=%s): skipping _reg.ensure_services(%s) — server-side responsibility",
            env_server_url, env_id,
        )
    elif kwargs.get("use_fake") is True:
        # Fake env construction is a local test/dev path; register lazy catalogs
        # without starting Docker/HTTP backends that the fake instance will not use.
        _reg.ensure_registered(env_id)
    else:
        _reg.ensure_services(env_id)

    env_make_kwargs = _reg.registry.env_make_kwargs(env_id)
    if env_server_url:
        with _reg._import_locks_guard:
            lock = _reg._import_locks.setdefault(env_id, threading.Lock())
        with lock:
            if _reg._imported_remote_urls.get(env_id) != env_server_url:
                raise RuntimeError(
                    f"Remote env catalog for {env_id!r} switched while constructing "
                    f"{key!r}: expected {env_server_url!r}, got "
                    f"{_reg._imported_remote_urls.get(env_id)!r}. Use one env-server "
                    "URL per env_id in a process or construct in separate processes."
                )
            spec = _reg._specs.get(key)
            splits = {
                split: list(ids)
                for split, ids in _reg._splits.get(env_id, {}).items()
            }
            env_make_kwargs = dict(_reg.registry.env_make_kwargs(env_id))
    else:
        spec = _reg._specs.get(key)
        splits = _reg._splits.get(env_id)

    if spec is None:
        if splits:
            all_ids = [tid for ids in splits.values() for tid in ids]
            preview = all_ids[:10]
            suffix = f" ... ({len(all_ids)} total)" if len(all_ids) > 10 else ""
            raise KeyError(
                f"Task '{key}' not found. Available in '{env_id}': {preview}{suffix}"
            )
        _reg._import_all()
        available = list(_reg._splits.keys())
        raise KeyError(f"Unknown key '{key}'. Available envs: {available}")
    # Resolve config through the layered precedence chain (lowest -> highest;
    # each layer overrides the ones before it):
    #   1. make()-pop defaults     — the ultimate fallback (step_timeout=120, ...)
    #   2. env_make_kwargs(env_id) — env-wide default.yaml ``make_kwargs``
    #   3. spec.kwargs             — per-task / per-subset register() kwargs
    #   4. kwargs                  — per-call gym.make() kwargs
    # Layer 2 (env-wide make_kwargs, e.g. webgym/osworld step_timeout) is read
    # from the registry singleton via the module so server mode resolves
    # identically to direct mode. (Preserved across the lite.gym.factory split.)
    merged = {**env_make_kwargs, **spec.kwargs, **kwargs}

    # Pop framework-only kwargs from ``merged`` before handing what
    # remains to ``spec.entry_point``. Grouped by consumer:
    #   (1) wrapper-owned make config — applied below in the wrapper layer
    #   (2) identity       — plumbed into the env / forwarded to server
    #   (3) routing        — picks local vs remote construction

    # (1) Wrapper-owned make config — popped from the SINGLE source
    # ``registry._WRAPPER_KWARG_DEFAULTS`` (membership + defaults in one place), so this pop set
    # can't drift from its defaults. ``CARRIED_SPEC_KWARGS`` also includes env-owned make
    # config such as cursor, which remains in ``merged`` for the env entry point.
    #   step/reset_timeout: defaults sized for contended 96-env concurrent workloads where
    #     step()/reset() may take tens of seconds (env-internal pool waits, docker-create
    #     queueing, cold boot). Callers needing tighter latency override via gym.make(...).
    #   loop_detect: upstream mobilegym ``--loop-detect`` knob generalized to
    #     all envs. 0 disables; positive N replaces the repeated action with an
    #     internal ``terminate(status=loop_detect_terminal_status)`` so the env
    #     scores through its normal terminal path, not a post-hoc
    #     ``truncated=True``. Default 0 (off) because the fingerprint is
    #     observation-blind, so it can kill legitimate repeated navigation.
    #     Override per run via ``--env-kwargs``.
    _wrapper = {k: merged.pop(k, d) for k, d in _reg._WRAPPER_KWARG_DEFAULTS.items()}
    step_timeout = _wrapper["step_timeout"]
    reset_timeout = _wrapper["reset_timeout"]
    loop_detect = _wrapper["loop_detect"]
    loop_detect_terminal_status = _wrapper["loop_detect_terminal_status"]

    # (2) Identity (session_id, token_hash, server_port).
    # Plumbed *into* the env instance — read from there, NOT from
    # os.environ / process globals (would race across concurrent
    # sessions on a multi-tenant env-server). ``token_hash`` and
    # ``server_port`` are set only by the env-server (sha256 of bearer
    # token + its listen port); local in-process gym.make leaves them
    # None and the env's external-resource scoping (naming, cleanup
    # filters, drift-reaper scope) drops the corresponding segments.
    session_id = merged.pop("session_id", None) or os.environ.get("SESSION_ID")
    token_hash = merged.pop("token_hash", None)
    server_port = merged.pop("server_port", None)
    # Selector hint accepted by registry/task-list callers. Once the concrete
    # env key is chosen, it is not an env constructor kwarg.
    merged.pop("split", None)

    # (3) Routing — env_server_url decides remote vs local construction.
    # Priority: explicit kwarg > env var (same resolution as the early
    # ``ensure_services`` gate above; the early read used ``kwargs.get``
    # non-destructively, this one pops from ``merged``).
    env_server_url = (
        merged.pop("env_server_url", None)
        or routing_server_url()
    )
    if env_server_url:
        from lite.gym.remote import LiteEnvClient
        # Forward step/reset_timeout + session_id to the server so its
        # wrapper limits + per-env naming match ours.
        server_kwargs = {
            **merged,
            "step_timeout": step_timeout,
            "reset_timeout": reset_timeout,
            # Force-off server-side loop detection: this client wraps the
            # remote env with LoopDetectWrapper below (it sees every
            # action before they go over the wire), so server-side
            # detection would just be redundant work for the same signal.
            "loop_detect": 0,
        }
        # Give the server-side reset/step wrapper time to report its own
        # timeout instead of letting the HTTP client cut the request off first.
        rpc_timeout = max(
            float(reset_timeout),
            float(step_timeout) if step_timeout is not None else 0.0,
        ) + 60.0
        env: LiteBaseEnv = LiteEnvClient(
            env_server_url, key,
            metadata=spec.metadata,
            session_id=session_id,
            timeout=rpc_timeout,
            **server_kwargs,
        )
    else:
        env = spec.entry_point(**merged)
        # Attach session_id + token_hash so env-side code reads from this
        # instance instead of os.environ / process-global state — see
        # comment above. Set as a single attribute (an ``EnvIdentity``
        # dataclass) rather than three separate underscore-prefixed
        # magic fields — env adapters read via
        # ``getattr(self, "identity", None)`` and call
        # ``identity.resolved_session_id()`` for the canonical fallback.
        if any(x is not None for x in (session_id, token_hash, server_port)):
            from lite.gym.utils.config.identity import EnvIdentity
            env.identity = EnvIdentity(
                session_id=session_id,
                token_hash=token_hash,
                server_port=server_port,
            )

    # Registry-key identity → the constructed env (both branches). The
    # LiteBaseEnv.metadata property merges these into ``others`` (the
    # same-source contract). On the remote branch the write is inert by
    # design — LiteEnvClient returns the server-computed metadata (which
    # already carries identity from the server-side env) verbatim.
    env.env_id = env_id
    env.task_id = task_id

    # Apply wrappers — innermost (timeout) → outermost (loop detector).
    # Order: LoopDetect wraps StepTimeout (so loop detection sees the
    # actions as the caller submitted them, not whatever the timeout
    # wrapper would let through). Unsupported/off-surface calls are handled by
    # the owning env's step() so the feedback can include the current observation
    # and normal step accounting. Cursor compositing is env-owned at capture time;
    # there is no registry-level cursor wrapper.
    #
    # In remote-client mode, do not also wrap the HTTP client with
    # StepTimeoutWrapper. The server owns reset/step timeout enforcement there;
    # the client RPC timeout above is deliberately longer so the typed server
    # timeout/error wins instead of a local cancellation race.
    if step_timeout is not None and not env_server_url:
        from lite.gym.wrappers import StepTimeoutWrapper
        env = StepTimeoutWrapper(env, step_timeout=step_timeout, reset_timeout=reset_timeout)

    if loop_detect > 0:
        from lite.gym.wrappers import LoopDetectWrapper
        env = LoopDetectWrapper(
            env,
            loop_threshold=loop_detect,
            terminal_status=loop_detect_terminal_status,
        )

    return env
