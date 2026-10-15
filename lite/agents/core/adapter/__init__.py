"""Adapter pillar: ``BaseAgentAdapter`` + registry + shared utilities.

Import from the package, not submodules:

    from lite.agents.core.adapter import (
        BaseAgentAdapter, AgentAdapterRegistry, AsIsAdapter,
    )

Shared turn slicing/counting belongs to ``lite.core.messages.turns``.
"""

from lite.agents.core.adapter.base import (  # noqa: F401
    AGENT_KWARGS_TOOL_SURFACE_KEYS,
    TOOL_SURFACE_KWARGS,
    AgentAdapterRegistry,
    AsIsAdapter,
    BaseAgentAdapter,
    tool_surface_agent_kwarg_names,
)

__all__ = [
    "AgentAdapterRegistry",
    "AsIsAdapter",
    "BaseAgentAdapter",
    "tool_surface_agent_kwarg_names",
]
