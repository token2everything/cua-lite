"""
lite.osworld JSONL row schema.

Documents the on-disk shape of one row in
``lite/gym/envs/lite/osworld/data/eval.jsonl``.

This is the lite.osworld *specialization* of ``SandboxTaskDataRow`` — the row
itself follows the generic ``SandboxTaskDataRow`` schema, while the
``metadata`` sub-dict has the env-specific fields described by
``LiteOsworldMetadata``.
"""

from __future__ import annotations

from typing import Any, TypedDict


class LiteOsworldTags(TypedDict, total=False):
    """``metadata.others`` contents — flows to Lite task metadata.

    Filter expressions read from this dict, e.g.::

        lambda m: m.others.get("domain") == "chrome" and m.others.get("oracle_actions")

    ``task_id`` / ``env_id`` are injected automatically by ``registry.register``
    into every registered spec's others (``_register_one`` only passes the
    metadata through) — see the reserved-keys list on ``metadata.others``.
    """

    task_id: str
    domain: str
    # "does this row have a curated oracle?" is ``bool(oracle_actions)`` — every
    # generator writes this key, so a filter on it means the same thing on every
    # split. It replaces ``oracle_verified``, which was defined as exactly
    # ``bool(oracle_actions)`` and is no longer written: a filter on that key
    # silently selects nothing (older rows under ``.logs/`` still carry it).
    oracle_actions: list[dict[str, Any]]
    oracle_after_postconfig: bool
    # "infeasible" | "google_auth" | "block: <english reason>" | None
    exclude_reason: str | None


class LiteOsworldMetadata(TypedDict):
    """``metadata`` field of a lite.osworld eval.jsonl row.

    - ``others`` is the queryable subset (see ``LiteOsworldTags``).
    - The remaining fields are runtime payload, consumed by
      ``setup_fn`` / ``evaluate_final_fn`` via ``task.metadata``.
    """

    others: LiteOsworldTags
    config: list[dict[str, Any]]                # OSWorld setup steps
    evaluator: dict[str, Any]                   # OSWorld evaluator spec
    osworld_id: str
