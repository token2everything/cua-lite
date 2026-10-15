"""Gym-layer helpers for env tool configuration.

Canonical GUI schema bodies live in ``lite.core.tools.action_space``; the
canonical finish / browser-nav / ``open_app`` / ``bash`` / ``report_infeasible``
bodies live in ``lite.core.tools.extra_tools``. This module is the env/runtime
layer: it resolves ``env_kwargs.valid_actions`` and ``env_kwargs.extra_tools``
against an env's tool set, and gates sandbox-only tools such as ``bash`` on what
the backend can actually execute. It reads schemas from the core catalogs; it
never declares one.

**The class is the AUTHORING form; ``extra_tool_schemas`` carries canonical
schema dicts.** Each env declares its env-owned extras as a ``BaseTools``
subclass — the same base ``LiteFinishToolSet`` / ``LiteBrowserNavToolSet`` / ``LiteAppLaunchToolSet``
and the GUI action sets hang off — and :func:`resolve_extra_tools` compiles a
yaml selection against it into the canonical list
``LiteBaseMetadata.extra_tool_schemas`` carries. No class ever crosses that
boundary, so ``lite.agents`` learns nothing about envs.

Canonical finish tools (``response`` / ``terminate``) are resolved from the
shared extra-tool catalog here, never repeated in an env's tool set (declaring
one there raises). Yaml's ``env_kwargs.extra_tools`` is a list of **names** that
picks a subset. Semantics:

  * ``None`` (omitted from yaml) — **block everything** (opt-in default)
  * ``[]`` — block everything (same as omission)
  * ``["name", ...]`` — explicit subset (preserves yaml order)

This is intentionally asymmetric with ``valid_actions`` (where ``None`` =
pass through native action surface): native surface should be exposed
by default, but the env's *extra* tools are additions that should require
explicit opt-in to avoid surprising the agent with tools its training
distribution never saw.

``executable`` is a SECOND, orthogonal per-env fact: a name can be declared and
still be rejected because the backend is not wired for it. It is passed beside
the tool set, never folded into it — which is also how ``bash`` (a property of
the sandbox TRANSPORT, not of any env's tool set) is granted.

Usage:
    from lite.core.tools.extra_tools import make_open_app_tool
    from lite.core.tools.schemas import BaseTools
    from lite.gym.utils.feedback.surface import resolve_extra_tools

    class MyEnvTools(BaseTools):
        _SCHEMAS = {"open_app": make_open_app_tool(_MY_ENV_APPS)}

    # in __init__ or metadata():
    extra_tools = resolve_extra_tools(
        self._extra_tools_selection,    # yaml-provided list[str] | None
        tools=MyEnvTools,
        env_name="my_env",
    )
"""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER,
    LITE_MOBILE_ACTION_NAMES_IN_SCHEMA_ORDER,
    LiteMobileActionSet,
    lite_action_names_for_action_batch_tool,
    lite_action_set_tool_names_for_metadata,
)
from lite.core.tools.extra_tools import (
    BASH_TOOL_NAME,
    LiteFinishToolSet,
    LiteShellToolSet,
)
from lite.core.tools.schemas import (
    BaseTools,
    tool_schema_name,
    validate_extra_tool_schemas,
)


def valid_action_names(platform: str, task_type: str = "use") -> frozenset[str]:
    """Names ``env_kwargs.valid_actions`` may contain for this platform/task."""
    md = LiteCUAMetadata(dims=(platform, task_type))
    names: set[str] = set()
    for top in lite_action_set_tool_names_for_metadata(md):
        children = lite_action_names_for_action_batch_tool(top)
        names |= set(children) if children else {top}
    return frozenset(names)


def resolve_valid_actions(
    valid_actions: list[str] | None,
    *,
    env_name: str,
    platform: str,
    task_type: str = "use",
) -> list[str] | None:
    """Resolve ``env_kwargs.valid_actions`` — the ONE behaviour every env shares.

    ``valid_actions`` constrains GUI actions only; standalone finish / nav /
    ``open_app`` / ``bash`` / env-local tools are surfaced through
    ``metadata.extra_tool_schemas`` instead. The two knobs are orthogonal (see
    ``lite/agents/core/action_space/base.py``), which is what makes
    ``valid_actions: []`` + ``extra_tools: ["terminate"]`` a coherent
    "no GUI, finish only" configuration.

    Semantics — the two meanings of "empty" are DIFFERENT and both load-bearing:

      * ``None`` (omitted from yaml) — no filtering; the env exposes its whole
        native GUI surface.
      * ``[]`` — deliberately no GUI actions. This strips the action-batch tool
        entirely; browsergym / webharbor.webvoyager bid-only rows depend on it.
      * ``["name", ...]`` — explicit subset, verbatim and order-preserving.

    Raises:
        ValueError: any name outside this env's native GUI vocabulary — a typo
          (``"clik"``), a finish/nav/extra tool name (``"terminate"``), or a
          action from the WRONG platform (``"tap"`` on desktop). This is the
          config boundary: a typo must never degrade into a silently smaller
          action set. Standalone tool names must fail here rather than erasing
          the agent's GUI capability while the rollout keeps running.
    """
    if valid_actions is None:
        return None
    allowed = valid_action_names(platform, task_type)
    unknown = [name for name in valid_actions if name not in allowed]
    if unknown:
        raise ValueError(
            f"{env_name} got unknown env_kwargs.valid_actions {unknown}; "
            f"valid_actions names GUI actions only and this env's surface "
            f"({platform}/{task_type}) accepts {sorted(allowed)}. Standalone "
            "tools (finish / nav / open_app / bash) are selected through "
            "env_kwargs.extra_tools instead."
        )
    return list(valid_actions)


def resolve_schema_valid_actions(
    valid_actions: list[str] | None,
    *,
    env_name: str,
    platform: str,
    supported_actions: Collection[str],
    task_type: str = "use",
) -> list[str]:
    """Resolve the GUI actions this env should advertise to agents.

    ``env_kwargs.valid_actions`` remains the schema/config surface:
    ``None`` means no env-config pruning. Some env backends still cannot
    physically execute every canonical GUI action. Their schema surface
    should not advertise those dead actions by default, but a bad model call
    should still reach env-owned unsupported-action feedback.
    """
    resolved = resolve_valid_actions(
        valid_actions,
        env_name=env_name,
        platform=platform,
        task_type=task_type,
    )
    supported = set(supported_actions)
    if resolved is None:
        return [
            name for name in valid_action_order(platform, task_type)
            if name in supported
        ]
    unsupported = [name for name in resolved if name not in supported]
    if unsupported:
        raise ValueError(
            f"{env_name} cannot advertise unsupported env_kwargs.valid_actions "
            f"{unsupported}; this env can execute {sorted(supported)}"
        )
    return list(resolved)


def valid_action_order(platform: str, task_type: str = "use") -> list[str]:
    """``valid_action_names`` in schema declaration order where order matters."""
    if task_type == "use" and platform == "mobile":
        return list(LITE_MOBILE_ACTION_NAMES_IN_SCHEMA_ORDER)
    if task_type == "use" and platform in {"desktop", "browser"}:
        return list(LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER)
    return sorted(valid_action_names(platform, task_type))


def android_supported_actions() -> frozenset[str]:
    """The mobile actions an Android backend can execute: everything but ``pinch``.

    ONE owner for the whole Android family (androidlab / androidworld /
    mobilegym / mobileworld). The four envs differ in bench harness, not in
    which canonical mobile actions their device layer can drive, so a new
    mobile action lands in all four at once — and hand-copying the subtraction
    is how one of them silently keeps advertising an action it cannot run.
    """
    return LiteMobileActionSet.get_action_names() - {"pinch"}


def copy_valid_actions(valid_actions: list[str] | None) -> list[str] | None:
    """Copy a resolved ``valid_actions`` while PRESERVING ``None`` vs ``[]``.

    Metadata builders must hand out a fresh list (never alias the module-level
    yaml constant), but the naive ``list(x)`` spelling crashes on ``None`` and
    the naive ``list(x or [])`` spelling silently collapses ``None`` (= all GUI
    actions) into ``[]`` (= no GUI actions at all) — the exact confusion this
    module exists to prevent.
    """
    return None if valid_actions is None else list(valid_actions)


def _bash_unsupported_error(env_name: str | None = None) -> ValueError:
    """The single spelling of "this env cannot honor ``bash``".

    Loud on purpose: handing an agent a shell into the WRONG machine (or a dead
    no-op tool) is worse than not offering one at all.
    """
    who = env_name or "this env"
    return ValueError(
        f"{who} cannot honor extra_tools ['{BASH_TOOL_NAME}']: the agent bash tool is a "
        "PERSISTENT shell (`docker exec -i`) into the very machine the agent sees, which "
        "only the sandbox desktop-container family (lite.osworld / lite.cuagym / "
        "lite.cuaworld) provides. Envs whose desktop lives inside a nested QEMU VM "
        "(osworld, osworld_2) cannot honor it: `docker exec` reaches the VM wrapper on "
        "the host side, not the guest the agent is looking at. Remove "
        f"'{BASH_TOOL_NAME}' from env_kwargs.extra_tools."
    )


def merge_extra_tool_schemas(
    *slices: list[dict[str, Any]] | None,
    allow_identical_duplicates: bool = True,
) -> list[dict[str, Any]]:
    """Concatenate ``extra_tool_schemas`` slices, rejecting duplicate names.

    Metadata builders that amend a base builder's output MUST use this instead of
    ``dataclasses.replace(md, extra_tool_schemas=<mine>)`` — a wholesale replace
    silently drops whatever the base resolver contributed (e.g. the ``bash``
    slice ``SandboxBaseEnv._runtime_metadata`` adds when the run opted in).

    The byte-identical-duplicate allowance is the ONLY rule this function owns:
    the same schema appearing twice covers same-source metadata mirrors, and
    collapsing it is a merge decision no validator downstream can make. Set
    ``allow_identical_duplicates=False`` to make even that case fail.

    Everything else — canonical shape, non-empty name, and the duplicate-NAME rule —
    is :func:`~lite.core.tools.schemas.validate_extra_tool_schemas`, called once
    on the result. That keeps the env-facing merge gate aligned with the
    metadata gate it feeds, including exception type and error wording.
    """
    merged: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for group in slices:
        for schema in group or []:
            fingerprint = _schema_bytes(schema)
            if allow_identical_duplicates and fingerprint in seen:
                continue
            seen.add(fingerprint)
            merged.append(schema)
    validate_extra_tool_schemas(merged, where="extra_tool_schemas")
    return merged


def _schema_bytes(schema: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except TypeError as exc:
        try:
            name = tool_schema_name(schema)
        except Exception:
            name = None
        raise TypeError(
            f"extra_tool_schemas {name!r} must be JSON-serializable"
        ) from exc


def _duplicate_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicate_set: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen:
            if name not in duplicate_set:
                duplicates.append(name)
                duplicate_set.add(name)
            continue
        seen.add(name)
    return duplicates


def _reject_duplicate_extra_tools(names: list[str]) -> None:
    duplicates = _duplicate_names(names)
    if duplicates:
        raise ValueError(f"duplicate extra_tools names: {duplicates}")


def resolve_extra_tools(
    selection: list[str] | None,
    *,
    tools: type[BaseTools],
    env_name: str | None = None,
    executable: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve a yaml selection against an env's tool set → Lite schemas.

    THE resolver. One function over ``BaseTools`` subclasses owns plain env
    extras, the sandbox ``bash`` grant, and executable-gated resolution. The
    tri-state, the duplicate rejection, the finish-tool resolution and the two
    orthogonal gates are all spelled once.

    Args:
        selection: yaml ``env_kwargs.extra_tools`` —
          - ``None`` (omitted) — block everything (opt-in default)
          - ``[]`` — block everything (explicit, same effect as omission)
          - ``["name", ...]`` — explicit subset (preserves yaml order)
        tools: the env's ``BaseTools`` subclass — what it DECLARES. Canonical
          finish tools must not be declared there; they are resolved centrally
          (declaring one raises), which is why every name predicate built on
          this set has to spell
          ``get_tool_names() | LiteFinishToolSet.get_tool_names()``.
        env_name: env id, for the failure messages only.
        executable: the SECOND, orthogonal per-env fact — what the backend is
          actually wired to run. ``None`` means "everything declared". ``bash``
          is never declared by an env (it is a property of the sandbox
          TRANSPORT), so it is admitted only by appearing here.

    Returns:
        List of canonical Lite tool schemas, in the order given by selection.
        Empty list when ``selection is None`` or ``selection == []``.

    Raises:
        ValueError: duplicate names, a name outside the declared set + finish
          catalog, or a declared-but-unwired name.

    **The vocabulary here is CLOSED but env-RELATIVE: ``declared |
    finish_names | {bash}``.** "Closed" means an env cannot name a tool nobody
    offers; it does NOT mean the set is fixed across envs, and the difference is
    the whole content of the next paragraph.

    **A model-family dialect finish spelling is therefore permitted here IF the
    env DECLARES it** — ``answer``, ``INFO``, ``finished``, ``visit_url`` are
    unknown names only to an env whose tool set does not carry them. The gym
    layer is env-relative and model-family-agnostic, so it rejects only names
    outside the env's declared vocabulary.

    **Dialect-only tool names are rejected at the row boundary.** ``lite.gym``
    does not import ``lite.data``, so env configuration remains scoped to the
    env surface. Row validation rejects dialect names:
    ``lite/data/utils/rows.py::_reject_dialect_only_tool_name`` runs over
    ``metadata.extra_tool_schemas[].name`` for BOTH row contracts, gated by no
    ``allow_*`` flag, so ``validate_canonical_rows`` and
    ``validate_raw_rollout_rows`` both refuse it. Its table covers the
    model-family dialect spellings.
    """
    names = [] if selection is None else list(selection)
    _reject_duplicate_extra_tools(names)
    finish_names = LiteFinishToolSet.get_tool_names()
    declared = tools.get_tool_names()
    overlap = sorted(declared & finish_names)
    if overlap:
        raise ValueError(
            "env extra-tool sets must not define canonical finish tools; "
            f"remove {overlap} and let the shared finish resolver handle them"
        )
    wired = declared if executable is None else frozenset(executable)
    if BASH_TOOL_NAME in names and BASH_TOOL_NAME not in wired:
        # Not "unknown" — known and deliberately withheld. Only the sandbox
        # desktop-container family lists it in ``executable``.
        raise _bash_unsupported_error(env_name)
    known = declared | finish_names | (wired & {BASH_TOOL_NAME})
    unknown = [n for n in names if n not in known]
    if unknown:
        who = f"for {env_name}" if env_name else "for this env"
        raise ValueError(
            f"unknown extra_tools: {unknown}; available {who}: {sorted(known)}"
        )
    unsupported = [
        n for n in names if n not in finish_names and n not in wired
    ]
    if unsupported:
        who = env_name or "this env"
        raise ValueError(
            f"{who} cannot execute extra_tools {unsupported}; only "
            f"{sorted(wired)} are wired"
        )
    resolved = [_resolved_extra_tool_schema(name, tools, finish_names) for name in names]
    validate_extra_tool_schemas(resolved, where="extra_tool_schemas")
    return resolved


def _resolved_extra_tool_schema(
    name: str,
    tools: type[BaseTools],
    finish_names: frozenset[str],
) -> dict[str, Any]:
    if name in finish_names:
        return LiteFinishToolSet.get_tool_schema(name)
    if name == BASH_TOOL_NAME:
        return LiteShellToolSet.get_tool_schema(BASH_TOOL_NAME)
    return tools.get_tool_schema(name)
