"""Shared HTTP keep-alive contract for remote env servers.

Both ``scripts/serve_env.py`` and ``scripts/serve_fleet.py`` expose a
client-facing HTTP endpoint used by :class:`lite.gym.remote.client.LiteEnvClient`.
The server-side idle keep-alive must outlive the client's connection pool
expiry so the client, not uvicorn, retires idle pooled connections.
"""

from __future__ import annotations

import logging

from lite.gym.remote.client import HTTPX_KEEPALIVE_EXPIRY_SEC

_KEEP_ALIVE_MARGIN_SEC = 60.0

SERVER_KEEP_ALIVE_TIMEOUT_SEC = HTTPX_KEEPALIVE_EXPIRY_SEC + _KEEP_ALIVE_MARGIN_SEC


def resolve_keep_alive_timeout(timeout_keep_alive: float | None) -> float:
    """Return the explicit keep-alive or the remote protocol default."""
    return (
        SERVER_KEEP_ALIVE_TIMEOUT_SEC
        if timeout_keep_alive is None
        else timeout_keep_alive
    )


def log_keep_alive_timeout(
    logger: logging.Logger,
    keep_alive: float,
    *,
    explicit: bool,
) -> None:
    """Log keep-alive in user-facing terms."""
    if keep_alive <= HTTPX_KEEPALIVE_EXPIRY_SEC:
        logger.warning(
            "timeout_keep_alive=%.0fs is shorter than the client keep-alive "
            "window (%.0fs); long-running requests may lose pooled connections.",
            keep_alive,
            HTTPX_KEEPALIVE_EXPIRY_SEC,
        )
    logger.info(
        "http keep-alive: %.0fs (%s)",
        keep_alive,
        "override" if explicit else "auto",
    )
