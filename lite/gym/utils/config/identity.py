"""Per-instance env identity for container naming.

Set on ``env.identity`` by :func:`lite.gym.registry.make` when running under an
env-server (carrying ``session_id`` / ``token_hash`` / ``server_port`` from the
server's ``POST /instances`` handler into the constructed env). In direct-mode
rollouts ``identity`` is ``None`` and consumers fall back to
``os.environ.get("SESSION_ID")`` or ``"local"``.

Used in BOTH direct and server mode (env adapters read it via
``getattr(self, "identity", None)``), so it is general env config/runtime
mechanism — not a server-only ``gym/remote`` concept. ``LiteBaseEnv`` does not
reference this type. The server-only run scope is its sibling
:class:`lite.gym.remote.scope.ServerScope`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvIdentity:
    session_id: str | None = None
    token_hash: str | None = None
    server_port: int | None = None

    def resolved_session_id(self) -> str:
        """Session segment for container naming with the canonical
        fallback chain (instance → ``$SESSION_ID`` → ``"local"``)."""
        return self.session_id or os.environ.get("SESSION_ID") or "local"
