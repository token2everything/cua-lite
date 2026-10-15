"""Env-server run scope, for resource-ownership scoping (server-mode only).

The per-*instance* identity (:class:`lite.gym.utils.config.identity.EnvIdentity`) lives
in ``gym/utils`` because direct mode uses it too; this module holds only the
server-*run* scope, which exists only while an env-server is running.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerScope:
    """Identity of one env-server *run*, for resource-ownership scoping.

    FRAMEWORK-OWNED and honestly typed (not "opaque"): built once at server
    startup from ``(args.port, strict-token)`` and threaded to each env's
    reconcile/shutdown methods. The framework never reads the fields back — only
    envs do, and an env may ignore them and scope by its own config instead (a
    k8s backend reads its namespace from its own env; webgym reads
    ``WEBGYM_MASTER_PORT`` from its own config). Adding a new scoping dimension
    therefore needs NO new field here.

    Distinct from :class:`lite.gym.utils.config.identity.EnvIdentity` (per *instance*,
    adds ``session_id``; ``token_hash`` is the *caller's* hash).
    ``ServerScope.token_hash`` is the *server's strict* hash, or ``None`` in
    passthrough (where the container reconcile scopes by ``server_port`` alone, so
    it needs no token hash). In passthrough the two ``token_hash`` fields share no
    value — different provenance — so the two types stay siblings, not a subclass.
    In strict single-tenant they coincide *by accident*; do not unify them on that
    basis.
    """

    server_port: int | None = None
    token_hash: str | None = None

    @classmethod
    def from_server(
        cls, *, server_port: int | None = None, strict_token: str | None = None,
    ) -> ServerScope:
        """Build the run scope. Hashes the server's *strict* bearer (the single
        ``sha256(...)[:6]`` site — replacing the per-reaper hashing). Passthrough
        passes ``strict_token=None`` → ``token_hash=None``."""
        token_hash = (
            hashlib.sha256(strict_token.encode()).hexdigest()[:6]
            if strict_token
            else None
        )
        return cls(server_port=server_port, token_hash=token_hash)
