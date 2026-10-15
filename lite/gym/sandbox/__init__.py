"""Sandbox utilities — task definitions, base env, and task registration.

Module layout:
    base.py     : SandboxBaseEnv (LiteBaseEnv subclass with reset/step/close)
    exec_stdio/ : host↔container transport (client.py host-side, server.py in-container)
    types.py    : SandboxTaskConfig (dataclass) + SandboxTaskDataRow (TypedDict)
    register.py : register_tasks / register_jsonl_tasks helpers
    docker/     : cua-lite/sandbox.linux — the family's minimal desktop base image

Schema convention (used by both register helpers):
    Each task carries a free-form ``metadata`` dict whose only reserved key is
    ``others``. The ``others`` subdict is light, queryable, and flows directly
    to ``metadata.others`` at both registry-time and env-instance.
    Everything else in ``metadata`` is env-specific runtime payload, read by
    ``setup_fn`` / ``evaluate_fn`` via ``task.metadata`` and never by filter
    expressions. Filter expressions like
    ``lambda m: m.others.get("domain") == "chrome"`` work uniformly across
    every Sandbox env.

JSONL row schema for envs with on-disk corpora — see ``SandboxTaskDataRow``
in ``types.py``.

Exports are lazy so type-only imports such as ``SandboxTaskConfig`` do not load
the base env and its backend/admission dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SandboxBaseEnv",
    "SandboxTaskConfig",
    "SandboxTaskDataRow",
    "lookup_task",
    "register_jsonl_tasks",
    "register_tasks",
]

_EXPORT_MODULES = {
    "SandboxBaseEnv": "lite.gym.sandbox.base",
    "SandboxTaskConfig": "lite.gym.sandbox.types",
    "SandboxTaskDataRow": "lite.gym.sandbox.types",
    "lookup_task": "lite.gym.sandbox.register",
    "register_jsonl_tasks": "lite.gym.sandbox.register",
    "register_tasks": "lite.gym.sandbox.register",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
