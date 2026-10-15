"""
Sandbox task registration helpers.

Two entry points for registering Sandbox tasks into the gym registry:

- ``register_tasks``: register a list of in-code ``SandboxTaskConfig``
  instances (used by envs whose tasks are defined inline, e.g. ``lite.demo``).
- ``register_jsonl_tasks``: read tasks from a JSONL file conforming to
  ``SandboxTaskDataRow`` and register one per row (used by envs with a
  large on-disk task corpus, e.g. ``lite.osworld``).

Both helpers register tasks via ``lite.gym.registry.register`` and produce
the same Lite metadata shape downstream — ``metadata.others`` is
sourced uniformly from ``task.metadata.get("others", {})``.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lite.gym.registry import register
from lite.gym.sandbox.base import SandboxBaseEnv
from lite.gym.sandbox.types import SandboxTaskConfig, SandboxTaskDataRow

logger = logging.getLogger(__name__)


def register_jsonl_tasks(
    jsonl_path: Path,
    *,
    env_id: str,
    split: str,
    computer: dict[str, Any],
    setup_fn: Callable | None = None,
    evaluate_step_fn: Callable | None = None,
    evaluate_final_fn: Callable | None = None,
    env_class: type[SandboxBaseEnv] = SandboxBaseEnv,
    factory_kwargs: dict[str, Any] | None = None,
    extra_tool_schemas: list[dict[str, Any]] | None = None,
    step_timeout: float | None = None,
    default_max_steps: int = 15,
    platform: str = "desktop",
) -> int:
    """Load a JSONL task file and register one gym task per row.

    Args:
        jsonl_path: path to the JSONL file (one ``SandboxTaskDataRow`` per line)
        env_id: registry env_id prefix; tasks are registered as
                ``f"{env_id}@{row['task_id']}"``
        split: split label for the registry (e.g. ``"eval"``, ``"train"``).
               The split is *not* read from the row — it is determined by the
               file the row came from.
        computer: dict passed to ``SandboxTaskConfig.computer``
        setup_fn / evaluate_step_fn / evaluate_final_fn: lifecycle functions
        env_class: env class to instantiate per task (default ``SandboxBaseEnv``)
        factory_kwargs: extra kwargs forwarded to every ``env_class(...)`` call
                        (e.g. ``{"post_action_delay": 2.0}``)
        extra_tool_schemas: optional ``metadata.extra_tool_schemas`` value (assigned onto
                     each ``SandboxTaskConfig.extra_tool_schemas``)
        step_timeout: optional registry-level step timeout
        default_max_steps: fallback used when a row omits ``max_steps``
        platform: ``SandboxTaskConfig.platform`` for every task

    Returns:
        number of tasks registered.
    """
    if not jsonl_path.exists():
        logger.warning("No JSONL data at %s", jsonl_path)
        return 0

    factory_kwargs = factory_kwargs or {}
    count = 0

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: SandboxTaskDataRow = json.loads(line)

            task = SandboxTaskConfig(
                task_id=row["task_id"],
                instruction=row["instruction"],
                computer=computer,
                max_steps=row.get("max_steps", default_max_steps),
                metadata=row["metadata"],
                platform=platform,
                extra_tool_schemas=extra_tool_schemas,
                setup_fn=setup_fn,
                evaluate_step_fn=evaluate_step_fn,
                evaluate_final_fn=evaluate_final_fn,
            )
            _register_one(task, env_id, split, env_class, factory_kwargs, step_timeout)
            count += 1

    logger.info("Loaded %d %s tasks (split=%s)", count, env_id, split)
    return count


def register_tasks(
    env_id: str,
    tasks: dict[str, list[SandboxTaskConfig]],
    *,
    env_class: type[SandboxBaseEnv] = SandboxBaseEnv,
    factory_kwargs: dict[str, Any] | None = None,
    step_timeout: float | None = None,
) -> None:
    """Batch-register in-code Sandbox tasks (grouped by split).

    Used by envs whose tasks are defined inline in Python (e.g. ``lite.demo``).
    For envs with a large on-disk corpus, prefer ``register_jsonl_tasks``.

    Args:
        env_id: Environment name, e.g. ``"lite.demo"``, ``"lite.osworld"``.
        tasks: ``{split: [SandboxTaskConfig, ...]}``.
        env_class: env class to instantiate (default ``SandboxBaseEnv``).
        factory_kwargs: extra kwargs forwarded to every ``env_class(...)`` call.
        step_timeout: optional registry-level step timeout.

    Example:
        register_tasks("lite.demo", {"eval": [SandboxTaskConfig(task_id="create_file", ...)]})
    """
    for split, task_list in tasks.items():
        for task in task_list:
            _register_one(task, env_id, split, env_class, factory_kwargs or {}, step_timeout)


#: ``env_id`` → ``task_id`` → ``SandboxTaskConfig`` lookup table,
#: populated by every successful registration. Used by Sandbox env bind paths
#: (e.g. ``LiteOsworldEnv.bind``) to resolve a ``task_id`` string back to the
#: registered task object — the factory closure that ``gym.make`` would
#: normally use can't be introspected from outside, so we mirror the binding
#: here.
_TASKS: dict[str, dict[str, SandboxTaskConfig]] = {}


def lookup_task(env_id: str, task_id: str) -> SandboxTaskConfig:
    """Return a deep copy of the registered ``SandboxTaskConfig`` for
    ``(env_id, task_id)`` or raise ``KeyError`` with the known
    task_ids in the error message.

    Copy, don't alias: this is the bind-path twin of the factory's
    ``copy.deepcopy`` (``_make_factory``) — the two exits from registration
    state. The ``_TASKS`` entry IS the factory-closure task, so handing out
    the shared object would let an instance mutation (e.g. by setup_fn) poison
    every future cold deepcopy too."""
    by_id = _TASKS.get(env_id)
    if not by_id or task_id not in by_id:
        known = list((by_id or {}))[:5]
        raise KeyError(
            f"unknown task_id={task_id!r} for env_id={env_id!r}; "
            f"known examples: {known}..."
        )
    return copy.deepcopy(by_id[task_id])


def _register_one(
    task: SandboxTaskConfig,
    env_id: str,
    split: str,
    env_class: type[SandboxBaseEnv],
    factory_kwargs: dict[str, Any],
    step_timeout: float | None,
) -> None:
    """Single-task registration shared by both helpers."""
    # Only forward step_timeout when explicitly set. A None must NOT be written into
    # spec.kwargs: it would override the env-wide make_kwargs default (and gym.make's
    # own default) at make()-merge time, leaving step_timeout=None — which disables
    # the StepTimeoutWrapper entirely. Omitting it lets the layered default apply.
    extra: dict[str, Any] = {}
    if step_timeout is not None:
        extra["step_timeout"] = step_timeout
    register(
        key=f"{env_id}@{task.task_id}",
        entry_point=_make_factory(task, env_id, env_class, factory_kwargs),
        split=split,
        # Same-source contract: the registered copy is EXACTLY the
        # env's builder output — registration and the live instance cannot
        # drift because they are one function.
        metadata=env_class._task_metadata(task),
        **extra,
    )
    # Mirror the spec into the lookup table for bind-time task resolution.
    _TASKS.setdefault(env_id, {})[task.task_id] = task


def _make_factory(
    task: SandboxTaskConfig,
    env_id: str,
    env_class: type[SandboxBaseEnv],
    factory_kwargs: dict[str, Any],
) -> Callable[..., SandboxBaseEnv]:
    """Build a factory closure that instantiates ``env_class`` for one task.

    Each ``gym.make`` call gets a fresh deep-copy of ``task`` so runtime
    mutations of ``task.metadata`` (e.g. by setup_fn) don't leak across
    instances.
    """
    def factory(**kwargs: Any) -> SandboxBaseEnv:
        merged = {**factory_kwargs, **kwargs}
        return env_class(
            task=copy.deepcopy(task),
            env_id=env_id,
            setup_fn=task.setup_fn,
            evaluate_step_fn=task.evaluate_step_fn,
            evaluate_final_fn=task.evaluate_final_fn,
            **merged,
        )
    return factory
