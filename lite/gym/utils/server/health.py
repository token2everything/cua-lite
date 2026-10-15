"""Small health-check helpers shared by env service declarations."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from lite.gym.errors import EnvDepsMissingError


@dataclass(frozen=True)
class _DepsMissingSnapshot:
    what: str
    install: str
    see: str

    @classmethod
    def from_error(cls, exc: EnvDepsMissingError) -> "_DepsMissingSnapshot":
        return cls(what=exc.what, install=exc.install, see=exc.see)

    def raise_error(self) -> None:
        raise EnvDepsMissingError(what=self.what, install=self.install, see=self.see)


class CachedEnvDepsHealthCheck:
    """Cache catalog health probes without hiding launch-time checks.

    `ensure()` remains the authoritative launch gate. This helper is for
    high-frequency catalog/fleet health probes that need to report missing or
    stale env assets without running image freshness checks on every poll.
    Only deterministic :class:`EnvDepsMissingError` results are cached;
    transient failures such as :class:`CapacityExhausted` are re-raised without
    poisoning the cache. Cached failures are reconstructed from fields rather
    than reusing exception objects and their tracebacks.
    """

    def __init__(
        self,
        probe: Callable[[str], None],
        *,
        ttl_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probe = probe
        self._ttl_s = ttl_s
        self._clock = clock
        self._expires_at = 0.0
        self._cached_error: _DepsMissingSnapshot | None = None

    def __call__(self, env_id: str) -> None:
        now = self._clock()
        if now < self._expires_at:
            if self._cached_error is not None:
                self._cached_error.raise_error()
            return

        try:
            self._probe(env_id)
        except EnvDepsMissingError as exc:
            self._cached_error = _DepsMissingSnapshot.from_error(exc)
            self._expires_at = self._clock() + self._ttl_s
            raise

        self._cached_error = None
        self._expires_at = self._clock() + self._ttl_s
