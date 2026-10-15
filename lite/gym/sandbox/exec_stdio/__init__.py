"""exec-stdio transport: the sandbox family's host↔container path.

``client.py`` runs on the host. Sandbox images bake the stdlib-only server as
``/opt/lite/stdio_server.py``; local tests may still execute ``server.py``
directly.
"""
from lite.gym.sandbox.exec_stdio.client import (
    DEFAULT_CALL_TIMEOUT,
    DEFAULT_RUN_COMMAND_TIMEOUT,
    AgentShellSession,
    DockerProvisioner,
    ExecStdioError,
    ExecStdioInterface,
    ExecStdioSession,
    attach,
)

__all__ = [
    "AgentShellSession", "DEFAULT_CALL_TIMEOUT", "DEFAULT_RUN_COMMAND_TIMEOUT",
    "DockerProvisioner", "ExecStdioError", "ExecStdioInterface", "ExecStdioSession",
    "attach",
]
