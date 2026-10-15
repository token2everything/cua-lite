"""
Direct/remote routing switch for the cua-lite gym.

A single source of truth for whether an in-process ``gym.*`` call resolves
locally or is delegated to a remote env-server over HTTP. Client processes
(rollout / eval / training) read ``CUA_LITE_ENV_SERVER_URL``; the env-server
itself suppresses it via :func:`serve_locally` so it never routes its own calls
back to itself.

Kept as a tiny leaf module (depends only on ``os``) so importers can pull the
routing switch without dragging in the heavy registry — and so the pure policy
(:func:`_route_target`) stays trivially testable without globals/env.

Usage:
    from lite.gym.utils.server.routing import routing_server_url, serve_locally
"""

from __future__ import annotations

import os

#: True once :func:`serve_locally` is called — set by the env-server's
#: ``make_app`` to mark THIS process the authoritative local registry. While
#: set, :func:`routing_server_url` ignores the ambient ``CUA_LITE_ENV_SERVER_URL``
#: so the server never routes its own ``gym.*`` calls back to itself over HTTP
#: (it commonly inherits that *client* var from the shell/profile). The server's
#: role is thus an explicit, positive fact — not inferred from the var's absence.
_serve_locally: bool = False


def serve_locally() -> None:
    """Mark this process the authoritative local registry (the env-server).

    Idempotent; called once from :func:`lite.gym.remote.server.make_app`. After
    this, :func:`routing_server_url` returns ``None`` regardless of the ambient
    ``CUA_LITE_ENV_SERVER_URL``, forcing every in-process ``gym.*`` call to
    resolve locally. Lets the server keep that client var in its environment
    (e.g. inherited from ``.zshrc``) without routing back to itself and
    self-deadlocking the single uvicorn worker — and, when the inherited URL
    points at a stale/dead remote, without hanging boot recovery
    (``recover_all`` → ``_import_all``) on a TCP connect to that dead host.
    """
    global _serve_locally
    _serve_locally = True


def routing_server_url() -> str | None:
    """The ambient env-server URL that puts a process in remote-client mode, or
    ``None`` if this process *is* the server (:func:`serve_locally`).

    Single source of truth for the direct/remote routing switch: client
    processes (rollout / eval / training) read the var; the server suppresses
    it. Every routing decision goes through here, so the server-vs-client
    distinction is decided in exactly one place. The policy itself lives in the
    pure :func:`_route_target` (state-free, so it's a plain testable truth table).
    """
    return _route_target(
        is_server=_serve_locally,
        ambient_url=os.environ.get("CUA_LITE_ENV_SERVER_URL"),
    )


def _route_target(*, is_server: bool, ambient_url: str | None) -> str | None:
    """Pure direct/remote routing policy, free of process and environment state:
    the server never delegates to a remote (so it can never route back to itself
    and deadlock); a client routes to ``ambient_url`` when one is set, else runs
    local. Kept pure so the invariant is verified without touching globals/env.
    """
    return None if is_server else ambient_url
