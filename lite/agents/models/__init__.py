"""CUA-Lite model dialect facade.

Importing this package does not register built-in model families. Call
``lite.agents.bootstrap.register_all()`` at application startup when the
built-in registry population side effects are needed.
"""

from lite.agents.core.agent import (  # noqa: F401
    AdapterBasedAgent,
    AgentRegistry,
    AutoAdapterAgent,
    BaseAgent,
)
from lite.agents.core.agent.hooks import SampleHook, SampleStepData  # noqa: F401
from lite.agents.core.agent.logger import TrajectoryLogger  # noqa: F401
from lite.agents.types import GenerateFn, PredictResult
from lite.core import (
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_TRUNCATED,
    LiteRLSample,
    LiteRLStep,
)

__all__ = [
    "AdapterBasedAgent",
    "AgentRegistry",
    "AutoAdapterAgent",
    "BaseAgent",
    "GenerateFn",
    "LiteRLSample",
    "LiteRLStep",
    "PredictResult",
    "STATUS_ABORTED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_TRUNCATED",
    "SampleHook",
    "SampleStepData",
    "TrajectoryLogger",
]
