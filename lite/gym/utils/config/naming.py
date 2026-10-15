"""Single source of truth for env-container naming.

Producers (`lite/gym/sandbox/base.py`, `androidlab/container.py`,
`androidworld/container.py`) and consumers
(`lite/gym/remote/reaper.py`, operator `docker ps` greps)
all go through these helpers so the format can evolve without a
grep-and-replace across the repo.

Format::

    lite-env-[<server_port>-][<token_hash>-]<session_id>-<env_id>-[<task_id>-]<suffix>

Segments (outermost → innermost):

* ``server_port``  Env-server listen port (env-server mode only). Together
                    with ``token_hash`` isolates two same-token env-server
                    instances on the same host — neither reaps the other's
                    containers because the leading prefix differs.
* ``token_hash``   ``sha256(bearer)[:6]``. Strict mode + env-server mode
                    only. Lets two callers on a passthrough server share a
                    ``session_id`` without container-name collisions.
* ``session_id``   Cleanup/session label (falls back to ``$SESSION_ID`` then
                    ``"local"``). Operators grep on this to scope cleanup to one run.
* ``env_id``       Registry env_id (``"androidworld"``, ``"lite.osworld"``).
* ``task_id``      Task identifier (sanitized to docker-name-safe chars).
                    Omitted when empty (envs without per-task registration).
* ``suffix``       Per-instance disambiguator (api port, uuid hex, ...).

Run: not directly. Imported wherever a docker container is spawned.
"""

from __future__ import annotations

import os
import re

from lite.gym.utils.backend.ports import env_server_scope

_PREFIX = "lite-env"

# Docker container names allow ``[a-zA-Z0-9][a-zA-Z0-9_.-]*``; sanitize
# task_ids (which can contain arbitrary punctuation) to that charset.
_DOCKER_NAME_SAFE_CHARS = frozenset("_.-")


def _sanitize_task_id(task_id: str) -> str:
    return "".join(
        c if c.isalnum() or c in _DOCKER_NAME_SAFE_CHARS else "_"
        for c in task_id
    )


def _sanitize_session_id(session_id: str) -> str:
    """Collapse every non-alnum char (notably ``-`` and ``.``) in the session
    segment to ``_``, making it a single opaque token.

    The session segment sits between the optional ``server_port`` segment and
    ``env_id`` in the container name, and the container reaper scopes by
    ``^lite-env-<server_port>-.*-<env_id>-`` (port alone). A literal ``-`` in
    the session would forge a fake ``-<port>-`` boundary: a DIRECT-mode
    container (no ``server_port`` segment) named with session ``"30100-x"``
    becomes ``lite-env-30100-x-<env_id>-…``, which a co-resident env-server on
    port ``30100`` would match and ``docker rm -f`` while it is LIVE. Stripping
    ``-`` from the session makes the direct-vs-server name partition
    structural — a direct name can never start ``lite-env-<digits>-``. (Unlike
    ``task_id``, the session cannot keep ``-``: it precedes ``env_id``, where a
    ``-`` is load-bearing for the scope regex.)

    CONTRACT FOR SHELL CONSUMERS. Every ``lite/gym/envs/*/scripts/cleanup.sh``
    that narrows by ``SESSION_ID`` re-implements this function in shell (they
    variously use ``${SESSION_ID//[^[:alnum:]_]/_}``, ``sed``, ``tr -c``) to
    build its ``docker ps --filter name=`` segment. What those scripts may
    assume, and all they may assume, is:

      * the session occupies exactly ONE ``-``-delimited segment of the name;
      * that segment is ``[A-Za-z0-9_]*`` — every other char is collapsed to
        ``_``, one output char per input char (never dropped, never reordered);
      * an env whose container factory lowercases the WHOLE assembled name
        (e.g. ``OSWorldContainerFactory._make_name``) may lowercase its filter
        segment too; nothing else about the segment may be transformed.

    ``tests/gym/utils/config/test_naming.py`` enforces this by EXECUTING each
    cleanup.sh against an adversarial ``SESSION_ID`` under a stub ``docker``
    and comparing the filter it computed with the segment
    :func:`format_container_name` actually produced — it reads no shell source
    and pins no spelling, so changing the algorithm here (or in any script) is
    safe as long as the two still agree. Do NOT "fix" a failure there by
    re-adding a literal-substring assertion about a script's source."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in session_id)


def format_container_name(
    *,
    env_id: str | None,
    suffix: str,
    task_id: str | None = None,
    session_id: str | None = None,
    token_hash: str | None = None,
    server_port: int | None = None,
) -> str:
    """Assemble a canonical env-container name.

    ``session_id`` falls back to ``"local"`` so the assembled name is
    never missing a session segment (which would corrupt the
    grep-by-session-id convention). All other optional segments —
    ``server_port``, ``token_hash``, ``env_id``, ``task_id`` — are
    omitted when ``None`` / falsy.

    ``env_id`` is typed ``str | None`` (not required) so the rare path
    where a Sandbox env is instantiated without a registered env_id
    still produces a valid docker name; the container reaper
    (:mod:`lite.gym.remote.reaper`) matches by explicit
    env_id substring, so the missing segment is acceptable here (the
    reaper won't catch it, which mirrors the original behavior
    pre-helper-extraction — see ``sandbox/base.py``).

    The outer (scope) segments are NOT re-assembled here: the name is built by
    concatenating :func:`container_name_prefix` with the inner segments, so
    "the reaper's scope filter is a literal prefix of every name it must reap"
    holds BY CONSTRUCTION rather than by two hand-kept-in-sync assemblies.
    Adding an outer segment therefore means editing ``container_name_prefix``
    only; the old duplication made it possible to add it here and silently
    unscope the reaper.
    """
    parts: list[str] = [_sanitize_session_id(session_id) if session_id else "local"]
    if env_id:
        parts.append(env_id)
    if task_id:
        parts.append(_sanitize_task_id(task_id))
    parts.append(suffix)
    return container_name_prefix(
        server_port=server_port, token_hash=token_hash
    ) + "-".join(parts)


def container_name_prefix(
    *,
    server_port: int | None = None,
    token_hash: str | None = None,
) -> str:
    """Constant prefix shared by all containers from one env-server
    instance — and the literal head of every
    :func:`format_container_name` result, which builds on top of it
    (so this is the ONE place the outer segments are assembled). Used as
    a **scope filter** (not as a per-instance identifier) by:

      * the container reconcile view
        (:class:`lite.gym.remote.reaper.ContainerReaper`):
        narrows ``docker ps`` to in-scope containers (scoped by
        ``server_port`` alone); per-instance precision then comes from each
        env's ``external_resource_id`` (exact match), NOT from a finer prefix.

    Returns the prefix with a trailing ``-`` so callers can compose
    ``f"^{prefix}.*-{env_id}-"`` (the docker name regex) without
    juggling separators.

    DO NOT extend this with session_id / env_id / task_id for the
    purpose of matching individual instances — session_id is a batch
    label (many instances per value), and explicit test/tool rebinds can
    mutate task_id through ``env.bind`` (the container name retains the
    original task_id forever). Per-instance match must go through
    ``env.external_resource_id`` instead.
    """
    parts: list[str] = [_PREFIX]
    if server_port is not None:
        parts.append(str(server_port))
    if token_hash:
        parts.append(token_hash)
    return "-".join(parts) + "-"


def match_pattern(prefix: str, env_id: str) -> str:
    """The docker-ps ``--filter name=`` regex for in-scope containers of one
    env_id — the CONSUMER-side inverse of :func:`format_container_name`,
    defined here so producer and reaper derive from one grammar.

    ``prefix`` comes from :func:`container_name_prefix` (trailing ``-``
    included). ``env_id`` is ``re.escape``-d: ids like ``lite.osworld`` carry
    a ``.`` that would otherwise be a regex wildcard.

    The leading ``-`` before ``env_id`` is load-bearing: it requires a
    session segment BETWEEN the port and env_id, and
    :func:`_sanitize_session_id` guarantees a direct-mode session can never
    forge that boundary (a ``-``-bearing session collapses to ``_``) — so a
    direct container is structurally unmatchable by a server's scope. A
    fuller ``parse_container_name`` is deliberately NOT provided: session /
    task segments are free-form after sanitizing, so a right-to-left parse is
    ambiguous — and per-instance precision must come from
    ``env.external_resource_id`` (exact match), never from name parsing.
    """
    return f"^{prefix}.*-{re.escape(env_id)}-"


# ── Singleton shared-container scheme (webgym / mobilegym SINGLETON) ────────────
# These do NOT use the richer multi-segment ``format_container_name`` above —
# webgym/mobilegym run ONE shared container per env-server (the whole pool
# inside), so the name only needs a per-env-server scope tag, not session /
# token / task segments.


def container_scope() -> str:
    """Container-name scope tag for the shared-singleton scheme: the env-server
    port, or ``d<pid>`` in direct mode.

    The pid scope makes the direct-mode name UNIQUE PER PROCESS so concurrent
    direct rollouts (or a stray leftover) never collide — ``ensure``'s pre-run
    ``docker rm -f`` can then only hit THIS process's own stale container, never
    a co-tenant's (a shared literal would friendly-fire any same-named
    container). Same pid → same name → the process shares ONE backend.
    Trade-off: a crashed direct process's ``<svc>-d<pid>`` isn't boot-reaped by
    the next (different pid) — a bounded dev-mode leak, preferable to
    friendly-fire. Env-server mode's unique listen port already isolates."""
    return env_server_scope(default=f"d{os.getpid()}")


def container_name(service: str, server_port: int | None = None) -> str:
    """``<service>-<server_port>`` (or ``<service>-d<pid>`` in direct mode) for
    the shared-singleton scheme. ``ensure`` passes ``None`` → resolve the current
    scope; ``shutdown``/``reap`` pass the real ``scope.server_port`` (which is
    ``None`` in direct mode → the same ``d<pid>``)."""
    if server_port is not None:
        return f"{service}-{server_port}"
    return f"{service}-{container_scope()}"
