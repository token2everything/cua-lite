"""Build, validate, unpack, and merge canonical action-batch calls.

``computer``/``mobile`` are action-batch tools whose ``actions`` array carries
canonical child action names. This module owns that grammar end to end: the two
tool names, constructors, validators, unpacking, and adjacent-batch merge.

Usage:
    from lite.core.tools.action_space.batches import (
        make_lite_action_batch_call,
        unpack_action_batch_call,
    )

    call = make_lite_action_batch_call("computer", make_tool_call("click", {}))
    actions = unpack_action_batch_call(call)

Run:
    uv run pytest tests/gym/types/test_envelope_contract.py -q
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from lite.core.tools.calls import (
    CANONICAL_TOOL_CALL_ERROR,
    EnvAction,
    LiteToolCall,
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
    validate_lite_tool_call,
)
from lite.core.tools.schemas import (
    make_tool_schema,
    schema_type_error_reason,
    tool_arguments_match_schema_route_keys,
    tool_schema_name,
    tool_schema_parameters,
    tool_schema_validation_errors,
)

#: ``base.py`` consumes these while importing this module's constructors, so the
#: scalars live here and derived action-batch sets stay in ``base.py``.
LITE_COMPUTER_ACTION_BATCH_TOOL_NAME = "computer"
LITE_MOBILE_ACTION_BATCH_TOOL_NAME = "mobile"

# =============================================================================
# Build one action-batch
# =============================================================================


def make_lite_action_batch_call(
    action_batch_tool_name: str,
    child_call: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical action-batch call carrying one child action."""
    return make_tool_call(
        action_batch_tool_name,
        {
            "actions": [
                {"action": tool_call_name(child_call), **tool_call_arguments(child_call)}
            ]
        },
    )


def make_lite_action_batch_schema(
    *,
    action_batch_tool_name: str,
    description: str,
    child_schemas: dict[str, dict[str, Any]],
    include: list[str] | None = None,
) -> dict[str, Any] | None:
    """Build the canonical action-batch schema from action schemas."""
    include_set = set(include) if include is not None else None
    selected = [
        copy.deepcopy(schema)
        for name, schema in child_schemas.items()
        if include_set is None or name in include_set
    ]
    if not selected:
        return None

    action_prop = {
        "type": "string",
        "enum": [tool_schema_name(schema) for schema in selected],
        "description": "The GUI action to execute.",
    }
    item_props: dict[str, Any] = {"action": action_prop}
    for schema in selected:
        for name, prop in tool_schema_parameters(schema)["properties"].items():
            item_props.setdefault(name, copy.deepcopy(prop))

    return make_tool_schema(
        action_batch_tool_name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": item_props,
                        "required": ["action"],
                    },
                },
            },
            "required": ["actions"],
        },
    )


def filter_action_batch_schema(
    schemas: list[dict[str, Any]],
    *,
    action_batch_tool_name: str,
    description: str,
    child_schemas: dict[str, dict[str, Any]],
    valid_actions: list[str],
) -> list[dict[str, Any]]:
    """Apply ``valid_actions`` to a nested action-batch action enum.

    Package-internal: ``_ActionBatchActionSet.filter_child_action_enum`` in ``base.py``
    is the only caller, so this name stays off ``__all__`` and off the package
    facade.
    """
    allowed = set(valid_actions)
    out: list[dict[str, Any]] = []
    for schema in copy.deepcopy(schemas):
        if tool_schema_name(schema) != action_batch_tool_name:
            out.append(schema)
            continue
        enum = (
            tool_schema_parameters(schema)
            .get("properties", {})
            .get("actions", {})
            .get("items", {})
            .get("properties", {})
            .get("action", {})
            .get("enum")
        )
        if not isinstance(enum, list):
            continue
        exposed = set(enum)
        rebuilt = make_lite_action_batch_schema(
            action_batch_tool_name=action_batch_tool_name,
            description=description,
            child_schemas=child_schemas,
            include=[
                name
                for name in child_schemas
                if name in allowed and name in exposed
            ],
        )
        if rebuilt is not None:
            out.append(rebuilt)
    return out


# =============================================================================
# Validate one action-batch
# =============================================================================


class LiteActionBatchValidationKind(StrEnum):
    """Machine-readable failure class for an action-batch payload."""

    UNKNOWN_LITE_ACTION_BATCH_TOOL = "unknown_batch_tool"
    ARGUMENTS_NOT_OBJECT = "arguments_not_object"
    ACTIONS_NOT_LIST = "actions_not_list"
    ACTIONS_EMPTY = "actions_empty"
    CHILD_NOT_OBJECT = "child_not_object"
    CHILD_ACTION_MISSING = "child_action_missing"
    CHILD_ACTION_UNKNOWN = "child_action_unknown"
    CHILD_ARGUMENTS_INVALID = "child_arguments_invalid"


@dataclass(frozen=True)
class LiteActionBatchValidationError:
    """Structured validation error for ``computer/mobile.arguments.actions``."""

    kind: LiteActionBatchValidationKind
    action_batch_tool_name: str
    reason: str
    child_action_name: str | None = None
    child_index: int | None = None


def validate_lite_action_batch_structure(
    action_batch_tool_name: str,
    arguments: Any,
) -> tuple[list[EnvAction], LiteActionBatchValidationError | None]:
    """Validate one canonical action-batch and return env-internal bare children.

    Structure only. An ``error`` means no child could be CONSTRUCTED -- the
    arguments are not an object, ``actions`` is not a list or is empty, a child
    is not an object, or a child carries no ``action`` name -- so every caller
    rejects the whole call.

    A child naming an action this batch tool does not carry is NOT an error
    here. It is returned like any other child, because whether an action can be
    executed is the concrete env's question, and its dispatch already answers it
    with env wording on the right call id. Callers that need the stricter
    reading -- offline row validation, which must reject malformed stored data
    rather than hand it back as feedback -- call
    :func:`lite_action_batch_child_name_errors` explicitly.
    """
    # Deferred because ``base.py`` imports this module's constructors first.
    from lite.core.tools.action_space.base import lite_action_names_for_action_batch_tool

    allowed = lite_action_names_for_action_batch_tool(action_batch_tool_name)
    if allowed is None:
        reason = f"{action_batch_tool_name} is not an action-batch tool"
        return [], LiteActionBatchValidationError(
            LiteActionBatchValidationKind.UNKNOWN_LITE_ACTION_BATCH_TOOL,
            action_batch_tool_name,
            reason,
        )
    if not isinstance(arguments, dict):
        got = type(arguments).__name__
        reason = f"{action_batch_tool_name}.arguments must be an object, got {got}"
        return [], LiteActionBatchValidationError(
            LiteActionBatchValidationKind.ARGUMENTS_NOT_OBJECT,
            action_batch_tool_name,
            reason,
        )
    raw_child_items = arguments.get("actions")
    if not isinstance(raw_child_items, list):
        reason = f"{action_batch_tool_name}.arguments.actions must be a list"
        return [], LiteActionBatchValidationError(
            LiteActionBatchValidationKind.ACTIONS_NOT_LIST,
            action_batch_tool_name,
            reason,
        )
    if not raw_child_items:
        reason = f"{action_batch_tool_name}.arguments.actions must be a non-empty list"
        return [], LiteActionBatchValidationError(
            LiteActionBatchValidationKind.ACTIONS_EMPTY,
            action_batch_tool_name,
            reason,
        )

    child_calls: list[EnvAction] = []
    for index, raw_child in enumerate(raw_child_items):
        if not isinstance(raw_child, dict):
            reason = (
                f"{action_batch_tool_name}.arguments.actions[{index}] must be a dict"
            )
            return [], LiteActionBatchValidationError(
                LiteActionBatchValidationKind.CHILD_NOT_OBJECT,
                action_batch_tool_name,
                reason,
                child_index=index,
            )
        action_name = raw_child.get("action")
        if not isinstance(action_name, str) or not action_name:
            reason = (
                f"{action_batch_tool_name}.arguments.actions[{index}].action "
                "must be a non-empty string"
            )
            return [], LiteActionBatchValidationError(
                LiteActionBatchValidationKind.CHILD_ACTION_MISSING,
                action_batch_tool_name,
                reason,
                child_index=index,
            )
        child_arguments = {k: v for k, v in raw_child.items() if k != "action"}
        child_calls.append({
            "name": action_name,
            "arguments": child_arguments,
        })
    return child_calls, None


def lite_action_batch_child_name_errors(
    action_batch_tool_name: str,
    children: list[EnvAction],
) -> dict[int, LiteActionBatchValidationError]:
    """Map child INDEX -> "this batch tool does not carry that action".

    Split out of :func:`validate_lite_action_batch_structure` because the two
    consumers want opposite policies from one fact. Env ingress lets such a
    child through so the env can reject it per action, keeping its siblings
    runnable and its own result slot. Offline row validation raises, because a
    STORED row naming an impossible action is malformed data, not a live model
    mistake to answer.
    """
    from lite.core.tools.action_space.base import lite_action_names_for_action_batch_tool

    allowed = lite_action_names_for_action_batch_tool(action_batch_tool_name)
    if allowed is None:
        return {}
    out: dict[int, LiteActionBatchValidationError] = {}
    for index, child in enumerate(children):
        name = child["name"]
        if name in allowed:
            continue
        out[index] = LiteActionBatchValidationError(
            LiteActionBatchValidationKind.CHILD_ACTION_UNKNOWN,
            action_batch_tool_name,
            f"{action_batch_tool_name}.actions cannot contain {name}",
            child_action_name=name,
            child_index=index,
        )
    return out


def _child_schemas_for_lite_action_batch_tool(
    action_batch_tool_name: str,
) -> dict[str, dict[str, Any]] | None:
    from lite.core.tools.action_space.base import LiteDesktopActionSet, LiteMobileActionSet

    if action_batch_tool_name == LITE_COMPUTER_ACTION_BATCH_TOOL_NAME:
        return LiteDesktopActionSet._SCHEMAS
    if action_batch_tool_name == LITE_MOBILE_ACTION_BATCH_TOOL_NAME:
        return LiteMobileActionSet._SCHEMAS
    return None


def _child_argument_key_error_reason(
    action_batch_tool_name: str,
    child_action_name: str,
    child_index: int,
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> str | None:
    if tool_arguments_match_schema_route_keys(schema, arguments):
        return None

    item = f"{action_batch_tool_name}.arguments.actions[{child_index}]"
    parameters = tool_schema_parameters(schema)
    required = parameters.get("required", [])
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in arguments:
                return f"{item}.{name} is required for {child_action_name}"

    properties = parameters.get("properties")
    if isinstance(properties, dict) and parameters.get("additionalProperties") is not True:
        known = {name for name in properties if isinstance(name, str)}
        unknown = sorted(name for name in arguments if name not in known)
        if unknown:
            if len(unknown) == 1:
                return (
                    f"{item} has unknown argument {unknown[0]!r} "
                    f"for {child_action_name}"
                )
            return (
                f"{item} has unknown arguments {unknown!r} "
                f"for {child_action_name}"
            )

    return f"{item} arguments do not match the {child_action_name} schema"


def _child_schema_error_reason(
    action_batch_tool_name: str,
    child_index: int,
    error: Any,
) -> str:
    item = f"{action_batch_tool_name}.arguments.actions[{child_index}]"
    path = ".".join(str(part) for part in error.path)
    field = f"{item}.{path}" if path else item
    if error.validator == "type" and isinstance(error.validator_value, str):
        return schema_type_error_reason(field, error.validator_value)
    if error.validator == "required":
        missing = str(error.message).split("'", 2)[1:2]
        if missing:
            return f"{field}.{missing[0]} is required"
    if error.validator == "enum":
        return f"{field} must be one of {list(error.validator_value)!r}"
    return f"{field} {error.message}"


def validate_lite_action_batch_child_arguments(
    action_batch_tool_name: str,
    child_calls: list[EnvAction],
) -> LiteActionBatchValidationError | None:
    """Validate child-action arguments after membership/availability checks."""
    schemas = _child_schemas_for_lite_action_batch_tool(action_batch_tool_name)
    if schemas is None:
        reason = f"{action_batch_tool_name} is not an action-batch tool"
        return LiteActionBatchValidationError(
            LiteActionBatchValidationKind.UNKNOWN_LITE_ACTION_BATCH_TOOL,
            action_batch_tool_name,
            reason,
        )

    for index, child_call in enumerate(child_calls):
        child_action_name = child_call["name"]
        child_arguments = child_call["arguments"]
        schema = schemas[child_action_name]
        reason = _child_argument_key_error_reason(
            action_batch_tool_name,
            child_action_name,
            index,
            child_arguments,
            schema,
        )
        if reason is None:
            schema_errors = tool_schema_validation_errors(schema, child_arguments)
            if schema_errors:
                reason = _child_schema_error_reason(
                    action_batch_tool_name,
                    index,
                    schema_errors[0],
                )
        if reason is not None:
            return LiteActionBatchValidationError(
                LiteActionBatchValidationKind.CHILD_ARGUMENTS_INVALID,
                action_batch_tool_name,
                reason,
                child_action_name=child_action_name,
                child_index=index,
            )
    return None


# =============================================================================
# Take one apart
# =============================================================================


def unpack_action_batch_call(tool_call: LiteToolCall) -> list[dict[str, Any]]:
    """Split canonical ``computer/mobile(actions=[...])`` into child calls.

    TOTAL over model-emittable input: a structurally malformed action-batch
    payload yields ``[]`` (no actions) instead of raising. Callers sit both
    above the env and outside ``env.step`` entirely, so a
    ``KeyError``/``TypeError`` from here is not paired feedback. Yielding
    nothing is inert at every call site: no fingerprint is recorded, no
    action executes.

    Agent-wire/provider payloads still raise: those remain fail-loud infra/config
    errors, not model feedback.
    """
    if validate_lite_tool_call(tool_call, "tool_call", require_id=False) is not None:
        raise TypeError(CANONICAL_TOOL_CALL_ERROR)
    name = tool_call_name(tool_call)
    arguments = tool_call_arguments(tool_call)
    if name not in (LITE_COMPUTER_ACTION_BATCH_TOOL_NAME, LITE_MOBILE_ACTION_BATCH_TOOL_NAME):
        return [tool_call]
    child_calls, error = validate_lite_action_batch_structure(name, arguments)
    if error is not None:
        return []
    return child_calls


@dataclass(frozen=True)
class LiteActionBatchMergeResult:
    """Merged action-batch calls plus source-index provenance for each merged call."""

    tool_calls: list[dict[str, Any]]
    source_groups: list[list[int]]
    source_to_merged: list[int]


def merge_adjacent_lite_action_batches_with_provenance(
    tool_calls: list[dict[str, Any]],
) -> LiteActionBatchMergeResult:
    """Merge adjacent ``computer``/``mobile`` calls and retain source groups."""
    out: list[dict[str, Any]] = []
    source_groups: list[list[int]] = []
    source_to_merged: list[int] = []
    for source_idx, call in enumerate(tool_calls):
        name = tool_call_name(call)
        if name not in (LITE_COMPUTER_ACTION_BATCH_TOOL_NAME, LITE_MOBILE_ACTION_BATCH_TOOL_NAME):
            out.append(call)
            source_groups.append([source_idx])
            source_to_merged.append(len(out) - 1)
            continue
        if out and tool_call_name(out[-1]) == name:
            tool_call_arguments(out[-1])["actions"].extend(
                tool_call_arguments(call)["actions"]
            )
            source_groups[-1].append(source_idx)
            source_to_merged.append(len(out) - 1)
        else:
            out.append(copy.deepcopy(call))
            source_groups.append([source_idx])
            source_to_merged.append(len(out) - 1)
    return LiteActionBatchMergeResult(out, source_groups, source_to_merged)


def merge_adjacent_lite_action_batches(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent ``computer``/``mobile`` calls without crossing extras."""
    return merge_adjacent_lite_action_batches_with_provenance(tool_calls).tool_calls


__all__ = [
    # Canonical action-batch contracts, re-exported by the package facade.
    "LITE_COMPUTER_ACTION_BATCH_TOOL_NAME",
    "LITE_MOBILE_ACTION_BATCH_TOOL_NAME",
    "make_lite_action_batch_call",
    "make_lite_action_batch_schema",
    "merge_adjacent_lite_action_batches",
    "unpack_action_batch_call",
    "validate_lite_action_batch_structure",

    # Leaf-owner APIs, deliberately off the package facade. Each is imported
    # from this module by named owners: structured validation results and
    # child-argument validation by env feedback ingress
    # (``lite/gym/utils/feedback/ingress.py``) and row validation
    # (``lite/data/utils/rows.py``); merge provenance by adapter/agent
    # provenance rendering (``lite/agents/core/agent/utils/provenance.py``,
    # ``lite/agents/models/{claude,gpt}/utils/*``).
    "LiteActionBatchMergeResult",
    "LiteActionBatchValidationError",
    "LiteActionBatchValidationKind",
    "merge_adjacent_lite_action_batches_with_provenance",
    "validate_lite_action_batch_child_arguments",
]
