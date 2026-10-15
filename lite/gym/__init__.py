"""cua-lite gym — Self-contained Computer Use environments.

Public API: the registry (``make`` / ``register`` / ``registry``),
:class:`LiteBaseEnv`, gym-owned env IO envelopes, the two gym facts a caller
needs *before* it can call ``make`` (``finalize_env_kwargs`` for the kwargs it
will pass, ``routing_server_url`` for whether that call resolves locally or over
HTTP), and narrow facade exports for core-owned compatibility types
(``LiteToolCall`` / ``LiteToolResult`` / Lite metadata contracts).

``finalize_env_kwargs`` and ``routing_server_url`` are exported here — rather
than left in ``lite.gym.utils`` — because ``lite.infer`` and ``lite.train`` are
gym *clients*: they must not reach past the facade into gym's internal helper
tree to answer a question gym owns.

Subsystems are imported from their canonical locations:

    from lite.gym.wrappers import EnvWrapper
    from lite.gym.sandbox import SandboxBaseEnv, SandboxTaskConfig, register_tasks
    from lite.gym.remote import LiteEnvClient, State, make_app
"""

from lite.core.metadata import (
    LiteBaseMetadata,
    LiteCUAMetadata,
    LiteGenericMetadata,
    metadata_from_dict,
)
from lite.core.tools.calls import LiteToolCall
from lite.core.tools.results import LiteToolResult
from lite.gym.base import LiteBaseEnv
from lite.gym.factory import make
from lite.gym.registry import register, registry
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.config.defaults import finalize_env_kwargs
from lite.gym.utils.server.routing import routing_server_url

__all__ = [
    "LiteBaseEnv",
    "LiteBaseMetadata",
    "LiteCUAMetadata",
    "LiteGenericMetadata",
    "LiteToolCall",
    "LiteToolResult",
    "LiteEnvObservation",
    "LiteEnvStepResult",
    "finalize_env_kwargs",
    "register",
    "make",
    "registry",
    "routing_server_url",
    "metadata_from_dict",
]
