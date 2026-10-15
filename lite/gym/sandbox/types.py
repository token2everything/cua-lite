"""
Sandbox task definition.

SandboxTaskConfig is the core data structure for Sandbox-based environments.
Each config holds task metadata AND its lifecycle functions (setup, eval, solve).

Usage:
    from lite.gym.sandbox import SandboxTaskConfig
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from typing_extensions import NotRequired


class SandboxTaskDataRow(TypedDict):
    """One row in a Sandbox JSONL task file (e.g. ``data/eval.jsonl``).

    The split (``train``/``eval``) is determined by which file the row came
    from, not by a per-row field.

    Fields:
        task_id      : registered task identifier (without env_id prefix).
        instruction  : natural-language task description shown to the agent.
        max_steps    : per-task step budget; overridden if the caller passes
                       ``gym.make(..., max_steps=N)``. Optional — falls back to
                       the env-level default when absent.
        metadata     : env-specific payload. Only one key is reserved across
                       all envs:

                       - ``metadata["others"]`` — light, queryable subdict
                         that flows directly to ``metadata.others``
                         (registry-time AND env-instance). Filter expressions
                         like ``lambda m: m.others.get("domain") == "chrome"``
                         read from this dict; keep it small (scalars / short
                         strings).

                       Every other key is env-defined runtime payload, read
                       by ``setup_fn`` / ``evaluate_fn`` via ``task.metadata``,
                       never by filter expressions.
    """

    task_id: str
    instruction: str
    max_steps: NotRequired[int]
    metadata: dict[str, Any]


@dataclass
class SandboxTaskConfig:
    """Specification for a single Sandbox task.

    Lifecycle functions are attached directly to each task. The second arg is
    the container handle (``_ContainerHandle`` from the exec-stdio transport);
    callers reach the X session via ``computer.interface`` (an
    ``ExecStdioInterface`` — same ~20-method surface the legacy cua Computer
    exposed, so existing setup/eval code transfers verbatim).
    - setup_fn:          (SandboxTaskConfig, handle) -> None
    - evaluate_step_fn:  (SandboxTaskConfig, handle) -> float | None
    - evaluate_final_fn: (SandboxTaskConfig, handle) -> float
    - solve_fn:          (SandboxTaskConfig, handle) -> None
    """

    task_id: str
    instruction: str
    # Provisioner kwargs for ``DockerProvisioner`` (lite/gym/sandbox/exec_stdio/
    # client.py): ``image``, ``memory``, ``cpu``, ``display``, ``timeout``,
    # ``ephemeral``. The ``display`` field is overridden at ``bind()`` time
    # from the env's ``display_resolution`` kwarg.
    computer: dict[str, Any]
    max_steps: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    platform: str = "desktop"  # "desktop" | "browser" | "mobile"
    # Optional extra tools surfaced to the agent (e.g. report_infeasible).
    # Flows to metadata.extra_tool_schemas at registry-time and env-instance.
    extra_tool_schemas: list[dict[str, Any]] | None = None

    # Lifecycle functions (optional, attached per-task)
    setup_fn: Callable | None = None
    evaluate_step_fn: Callable | None = None
    evaluate_final_fn: Callable | None = None
    solve_fn: Callable | None = None

    def with_display(self, display: str) -> SandboxTaskConfig:
        """Return a copy with the computer display resolution overridden."""
        import copy
        new = copy.copy(self)
        new.computer = {**self.computer, "display": display}
        return new
