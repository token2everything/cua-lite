"""Protocol pillar: ``BaseProtocol`` + registry + protocol window policies.

Internal layout:
    - :mod:`.base`  — ``BaseProtocol``, ``ProtocolRegistry``, ``FullHistoryProtocol``
      (the canonical default protocol), ``TurnWindowProtocol`` (the shared
      peel/group/re-attach frame for turn-windowed families).
    - :mod:`.window` — history-window content policy used by concrete history
      protocols.
    - :mod:`lite.core.messages` — provider-free message/content helpers.

External callers may import protocol-owned APIs from the package:

    from lite.agents.core.protocol import (
        BaseProtocol,
        ProtocolRegistry,
        append_with_boundary_tool_projection,
        filter_history_content,
    )

Provider-free structural message/content helpers are owned by
``lite.core.messages`` and are intentionally not re-exported here.
"""

from lite.agents.core.protocol.base import (  # noqa: F401
    BaseProtocol,
    FullHistoryProtocol,
    ProtocolRegistry,
    TurnWindowProtocol,
)
from lite.agents.core.protocol.window import (  # noqa: F401
    append_with_boundary_tool_projection,
    filter_history_content,
)

__all__ = [
    "BaseProtocol",
    "FullHistoryProtocol",
    "ProtocolRegistry",
    "TurnWindowProtocol",
    "append_with_boundary_tool_projection",
    "filter_history_content",
]
