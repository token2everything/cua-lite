"""Remote env protocol — HTTP client + server for :class:`LiteBaseEnv`.

Split so client- and server-side both live as library code under
:mod:`lite.gym.remote`. ``scripts/serve_env.py`` is a thin
``uvicorn.run(make_app(State(...)))`` launcher for the server half.

Public surface:
    from lite.gym.remote import LiteEnvClient              # client
    from lite.gym.remote import State, make_app            # server

Per-instance warm pool reuse is retired; ``--warm-singleton`` still prewarms
shared services.
"""

from __future__ import annotations

from typing import Any

from lite.gym.remote.client import LiteEnvClient


def __getattr__(name: str) -> Any:
    if name in {"State", "make_app"}:
        from lite.gym.remote.server import State, make_app

        return {"State": State, "make_app": make_app}[name]
    raise AttributeError(name)

__all__ = ["LiteEnvClient", "State", "make_app"]
