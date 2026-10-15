"""Agent pillar: ``BaseAgent`` + registry + runtime sampling loops.

Import from the package, not submodules:

    from lite.agents.core.agent import (
        AgentRegistry, BaseAgent, AdapterBasedAgent, AutoAdapterAgent,
    )

``PredictResult`` and ``GenerateFn`` live in ``lite.agents.types``. Rollout
records and status constants (``LiteRLStep``/``LiteRLSample``/``STATUS_*``)
live in ``lite.core``. Import them from their owning tier, not via this package.
"""

from lite.agents.core.agent.base import (  # noqa: F401
    AdapterBasedAgent,
    AgentRegistry,
    AutoAdapterAgent,
    BaseAgent,
)

__all__ = [
    "AdapterBasedAgent",
    "AgentRegistry",
    "AutoAdapterAgent",
    "BaseAgent",
]
