"""CUA-Lite Agents public facade — ``make`` / ``registry`` / ``register``.

Mirrors ``lite.gym``'s three-name surface. Everything else is imported from its
owning module (``lite.agents.core.action_space.base``,
``lite.agents.core.adapter``, ``lite.agents.core.protocol``,
``lite.agents.models.*``) — the root is a facade, not a re-export table.

The package root intentionally does not register built-in dialects at import
time. Use :func:`lite.agents.bootstrap.register_all` for explicit registration,
or run ``python -m lite.agents`` to list registered entries.
"""

from __future__ import annotations

from typing import Any

from lite.agents.factory import make

__all__ = ["make", "register", "registry"]


def __getattr__(name: str) -> Any:
    # ``registry``/``register`` resolve lazily: importing
    # ``lite.agents.core.agent`` pulls in the adapter + action-space core, whose
    # class bodies register the built-in ``lite@*`` dialects. Keeping them
    # behind ``__getattr__`` is what makes ``import lite.agents`` side-effect
    # free (tests/agents/test_package_bootstrap.py pins this). ``make`` has no
    # such cost, so it is imported eagerly.
    if name in ("registry", "register"):
        from lite.agents.core.agent import AgentRegistry
        return AgentRegistry if name == "registry" else AgentRegistry.register
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
