"""Env-side ingress validation for Lite tool calls."""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any, Literal

from lite.core.messages.final import (
    LOOP_DETECT_TERMINATE_REASON as _LOOP_DETECT_TERMINATE_REASON,
)
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    LITE_VALID_ACTION_NAMES,
    is_lite_action_name_or_action_batch_tool_name,
    lite_builtin_tool_names_for_metadata,
    validate_lite_action_batch_structure,
)
from lite.core.tools.action_space.batches import (
    LiteActionBatchValidationError,
    lite_action_batch_child_name_errors,
    LiteActionBatchValidationKind,
)
from lite.core.tools.calls import (
    RUNTIME_INTERNAL_STOP_REASON_KEY,
    RUNTIME_RESULT_CALL_ID_KEY,
    EnvAction,
    LiteToolCall,
    RuntimeEnvAction,
    make_tool_call,
    runtime_internal_stop_reason,
    runtime_rejected_reason,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
    with_runtime_internal_stop_reason,
    with_runtime_rejected_reason,
)
from lite.core.tools.extra_tools import (
    active_extra_tool_names,
    is_internal_finish_tool_call,
)
from lite.core.tools.schemas import (
    schema_type_error_reason,
    tool_arguments_match_schema_route_keys,
    tool_call_matches_schema_route_keys,
    tool_schema_name,
)
from lite.core.tools.schemas import (
    tool_schema_validation_errors as core_tool_schema_validation_errors,
)
from lite.gym.utils.feedback.errors import (
    ToolErrorFeedback as _ToolErrorFeedback,
)
from lite.gym.utils.feedback.errors import (
    append_feedback as _append_feedback,
)
from lite.gym.utils.feedback.errors import (
    current_feedback as _current_feedback,
)
from lite.gym.utils.feedback.errors import (
    error_only_feedback as _error_only_feedback,
)
from lite.gym.utils.feedback.errors import (
    invalid_action_arguments_message as _invalid_action_arguments_message,
)
from lite.gym.utils.feedback.errors import (
    invalid_action_not_available_message as _invalid_action_not_available_message,
)
from lite.gym.utils.feedback.errors import (
    unavailable_action_message as _unavailable_action_message,
)
from lite.gym.utils.feedback.errors import (
    unknown_tool_message as _unknown_tool_message,
)
from lite.gym.utils.feedback.errors import (
    unsupported_action_message as _unsupported_action_message,
)
from lite.gym.utils.feedback.results import (
    assert_safe_tool_call_envelopes,
    canonical_result_call_id,
    invalid_env_tool_call_envelope_message,
    validate_result_call_id,
)

ToolCallAvailability = Literal["active", "inactive", "unknown", "not_standalone"]

__all__ = [
    "ToolCallAvailability",
    "action_batch_structure_error_message",
    "action_satisfies_extra_tool_schema",
    "active_extra_tool_names",
    "assert_safe_tool_call_envelopes",
    "classify_standalone_tool_call",
    "extra_tool_schema_argument_error",
    "invalid_action_message",
    "invalid_env_tool_call_envelope_message",
    "invalid_top_level_action_message",
    "is_active_extra_tool_call",
    "is_lite_action_name_or_action_batch_tool_name",
    "is_inactive_tool_call",
    "is_internal_finish_tool_call",
    "is_loop_detect_terminate",
    "is_unknown_tool_call",
    "make_internal_terminate_action",
    "nested_extra_tool_action_batch_child_message",
    "prepare_env_tool_calls",
    "standalone_tool_call_feedback",
    "standalone_tool_call_feedback_with_reason",
    "unsupported_env_action_message",
]


def _as_env_action(action: RuntimeEnvAction) -> EnvAction:
    """Project canonical protocol call to the env-internal bare action layer.

    Runtime sidecars carry over unchanged: the bare action is what concrete envs
    execute, and they still need to know an internal finish from model output
    and which model call it answers.
    """
    out: dict[str, Any] = {
        "name": tool_call_name(action),
        "arguments": {
            k: v for k, v in tool_call_arguments(action).items() if v is not None
        },
    }
    if call_id := tool_call_id(action):
        out["call_id"] = call_id
    if internal_reason := runtime_internal_stop_reason(action):
        out[RUNTIME_INTERNAL_STOP_REASON_KEY] = internal_reason
    if RUNTIME_RESULT_CALL_ID_KEY in action:
        out[RUNTIME_RESULT_CALL_ID_KEY] = action[RUNTIME_RESULT_CALL_ID_KEY]
    return out


def is_active_extra_tool_call(
    action: EnvAction,
    extra_tool_schemas: list[dict[str, Any]] | None,
) -> bool:
    """True when env-internal ``action`` is enabled by ``metadata.extra_tool_schemas``.

    Most extra tools are name-distinct from GUI actions, so name matching is
    enough and preserves the historical runtime behavior. BrowserGym/WebHarbor
    can expose standalone extras such as ``click(index=...)`` whose names
    collide with GUI actions. For those collisions, require the arguments to
    match the active extra schema so ``click(coordinate=...)`` does not bypass
    the GUI gate merely because a DOM-index ``click`` extra is active.
    """
    name = action["name"]
    if name not in active_extra_tool_names(extra_tool_schemas):
        return False

    matching = [
        schema for schema in extra_tool_schemas or []
        if tool_schema_name(schema) == name
    ]
    if not is_lite_action_name_or_action_batch_tool_name(name):
        return True

    arguments = action["arguments"]
    return any(tool_arguments_match_schema_route_keys(schema, arguments) for schema in matching)


def is_loop_detect_terminate(action: RuntimeEnvAction) -> bool:
    """True for the private terminate injected by ``LoopDetectWrapper``."""
    return (
        is_internal_finish_tool_call(action)
        and tool_call_name(action) == "terminate"
        and runtime_internal_stop_reason(action) == _LOOP_DETECT_TERMINATE_REASON
    )


def make_internal_terminate_action(
    *,
    status: str = "success",
    reason: str = _LOOP_DETECT_TERMINATE_REASON,
    internal_reason: str | None = None,
    result_call_id: str | None = None,
) -> RuntimeEnvAction:
    """Build an env-private terminal call below the model tool surface."""
    action = with_runtime_internal_stop_reason(
        make_tool_call(
            "terminate",
            {"status": status, "reason": reason},
        ),
        internal_reason or reason,
    )
    if result_call_id is not None:
        action[RUNTIME_RESULT_CALL_ID_KEY] = validate_result_call_id(result_call_id)
    return action


def classify_standalone_tool_call(
    action: EnvAction,
    known_tool_names: Collection[str],
    extra_tool_schemas: list[dict[str, Any]] | None,
    *,
    is_standalone_action_tool: Callable[[EnvAction], bool] | None = None,
) -> ToolCallAvailability:
    """Classify an env-internal route against this env's standalone tool surface.

    This helper is deliberately pure: it decides only active vs known-inactive
    vs literal-unknown vs GUI/internal. The default inactive/unknown feedback
    mapping lives in :func:`standalone_tool_call_feedback`; concrete envs still
    own execution, step accounting, noop metadata, and rare carrier overrides.
    Classification is env-owned for the same reason: wrappers, adapters, and
    remote transport must not pre-reject calls before the concrete env has a
    chance to attach the current observation and apply its own step accounting.

    ``is_standalone_action_tool`` covers browser envs whose DOM tools share names with
    GUI actions, e.g. ``click(index=...)`` versus ``click(coordinate=...)``.
    """
    if runtime_internal_stop_reason(action):
        return "not_standalone"
    if is_active_extra_tool_call(action, extra_tool_schemas):
        return "active"

    name = action["name"]
    if is_lite_action_name_or_action_batch_tool_name(name) and not (
        is_standalone_action_tool is not None and is_standalone_action_tool(action)
    ):
        return "not_standalone"
    return "inactive" if name in known_tool_names else "unknown"


def standalone_tool_call_feedback_with_reason(
    action: EnvAction,
    known_tool_names: Collection[str],
    extra_tool_schemas: list[dict[str, Any]] | None,
    *,
    is_standalone_action_tool: Callable[[EnvAction], bool] | None = None,
) -> tuple[_ToolErrorFeedback | None, str | None]:
    """Default feedback and noop reason for inactive/unknown standalone calls.

    Answers three questions in one place, because every env asks them in the
    same order at the top of its action loop:

    1. **Did ingress already reject this action?** A batch child whose name the
       batch tool does not carry, a standalone extra nested inside a batch, or a
       name the task config withholds. It must NOT be dispatched -- without this
       arm an env that does not re-derive the check runs it. It is a GUI-surface
       fault, so it carries the current observation (R2a).
    2. Inactive known standalone tool, and 3. literal unknown tool: both are the
       TEXT surface -- error only, no observation, because there is no screen
       answer to give.

    Returns ``(None, None)`` for active tools, GUI/internal calls, and
    non-standalone shapes so the concrete env can continue with its own
    execution path.
    """
    if rejected_reason := runtime_rejected_reason(action):
        return _current_feedback(rejected_reason), rejected_reason

    availability = classify_standalone_tool_call(
        action,
        known_tool_names,
        extra_tool_schemas,
        is_standalone_action_tool=is_standalone_action_tool,
    )
    if availability == "inactive":
        return (
            _error_only_feedback(_unavailable_action_message(action["name"])),
            "inactive extra tool",
        )
    if availability == "unknown":
        return _error_only_feedback(_unknown_tool_message(action["name"])), "unknown tool"
    return None, None


def standalone_tool_call_feedback(
    action: EnvAction,
    known_tool_names: Collection[str],
    extra_tool_schemas: list[dict[str, Any]] | None,
    *,
    is_standalone_action_tool: Callable[[EnvAction], bool] | None = None,
) -> _ToolErrorFeedback | None:
    """Default env feedback for inactive/unknown standalone tool calls."""
    feedback, _reason = standalone_tool_call_feedback_with_reason(
        action,
        known_tool_names,
        extra_tool_schemas,
        is_standalone_action_tool=is_standalone_action_tool,
    )
    return feedback


def unsupported_env_action_message(
    name: str,
    supported_actions: Collection[str],
) -> str | None:
    """Return env feedback wording for a GUI action this env cannot execute.

    Returns ``None`` when the env can execute *name*.

    Envs own the set of GUI actions their low-level backend actually supports.
    The shared catalog owns the question "is this a GUI action at all" and the
    canonical wording. This keeps envs from each carrying a local copy of the
    same GUI-name check.

    The gate, not the wording: ``errors.unsupported_action_message`` is the
    unconditional string this returns. ``_message`` (never ``_feedback``) is
    this package's suffix for a string; ``ToolErrorFeedback`` /
    ``current_feedback`` are the carrier the caller wraps the string in.
    """
    if name in supported_actions:
        return None
    if is_lite_action_name_or_action_batch_tool_name(name):
        return _unsupported_action_message(name)
    return None


def is_inactive_tool_call(
    action: EnvAction,
    known_tool_names: Collection[str],
    extra_tool_schemas: list[dict[str, Any]] | None,
) -> bool:
    """True for a known standalone route not active in this task surface."""
    return (
        classify_standalone_tool_call(
            action,
            known_tool_names,
            extra_tool_schemas,
        )
        == "inactive"
    )


def is_unknown_tool_call(
    action: EnvAction,
    known_tool_names: Collection[str],
    extra_tool_schemas: list[dict[str, Any]] | None,
) -> bool:
    """True for a literal standalone tool name this env does not know."""
    return (
        classify_standalone_tool_call(
            action,
            known_tool_names,
            extra_tool_schemas,
        )
        == "unknown"
    )


def invalid_action_message(
    action: EnvAction,
    valid_actions: list[str] | None,
) -> str | None:
    """Return an env-side ``valid_actions`` error for env-internal GUI actions.

    ``valid_actions`` gates the ACTION layer, so this asks the action question
    directly: action-batch tool names are a different layer and are never
    subtracted back out of the answer here.
    """
    if valid_actions is None:
        return None
    name = action["name"]
    if name not in LITE_VALID_ACTION_NAMES:
        return None
    if name in valid_actions:
        return None
    return _invalid_action_not_available_message(name)


def invalid_top_level_action_message(
    action: RuntimeEnvAction,
    metadata: LiteCUAMetadata,
) -> str | None:
    """Return an error for a top-level GUI call outside this task surface."""
    name = tool_call_name(action)
    allowed = lite_builtin_tool_names_for_metadata(metadata)
    if name in allowed:
        if name in LITE_ACTION_BATCH_TOOL_NAMES:
            return None
        return invalid_action_message(_as_env_action(action), metadata.valid_actions)
    # Compound on purpose: an action-batch tool aimed at the wrong platform
    # (``mobile`` on a desktop surface) and an action outside this surface
    # (``tap``) are both "the GUI channel, but not here".
    if is_lite_action_name_or_action_batch_tool_name(name):
        return _invalid_action_not_available_message(name)
    return None


def action_batch_structure_error_message(error: LiteActionBatchValidationError) -> str:
    """Project a structural action-batch error into env-owned feedback text.

    Structural only, because ``validate_lite_action_batch_structure`` is the only
    producer that reaches env ingress. Child-ARGUMENT validation against the
    canonical child schema is owned by data-row validation
    (``lite/data/utils/rows.py``), not by this layer: at runtime the concrete env
    owns whether it can execute the child action at all, and its own dispatch
    already reports argument faults with child-action wording and the right
    carrier.
    """
    if error.kind == LiteActionBatchValidationKind.CHILD_ACTION_UNKNOWN:
        return f"invalid action: {error.child_action_name}; {error.reason}"
    action_batch_tool_name = error.action_batch_tool_name
    if error.kind == LiteActionBatchValidationKind.UNKNOWN_LITE_ACTION_BATCH_TOOL:
        return _unsupported_action_message(action_batch_tool_name)
    return _invalid_action_arguments_message(action_batch_tool_name, error.reason)


def nested_extra_tool_action_batch_child_message(
    action_batch_tool_name: str,
    action: EnvAction,
    extra_tool_schemas: list[dict[str, Any]] | None,
) -> str | None:
    """Reject standalone extra-tool calls nested inside action-batch calls."""
    if not is_active_extra_tool_call(action, extra_tool_schemas):
        return None
    name = action["name"]
    return (
        f"invalid action: {name}; "
        f"{action_batch_tool_name}.actions cannot contain standalone extra tool {name}"
    )


def extra_tool_schema_argument_error(
    action: LiteToolCall,
    schemas: list[dict[str, Any]],
) -> str:
    """Format a model-visible schema error for an active extra tool."""
    name = tool_call_name(action)
    arguments = tool_call_arguments(action)
    for schema in schemas:
        schema_errors = core_tool_schema_validation_errors(schema, arguments)
        if not schema_errors:
            continue
        err = schema_errors[0]
        path = ".".join(str(part) for part in err.path)
        field = f"{name}.arguments.{path}" if path else f"{name}.arguments"
        if err.validator == "type" and isinstance(err.validator_value, str):
            return _invalid_action_arguments_message(
                name,
                schema_type_error_reason(field, err.validator_value),
            )
        if err.validator == "required":
            missing = str(err.message).split("'", 2)[1:2]
            if missing:
                return _invalid_action_arguments_message(
                    name,
                    f"{name}.arguments.{missing[0]} is required",
                )
        return _invalid_action_arguments_message(name, f"{field} {err.message}")
    return _invalid_action_arguments_message(
        name,
        f"{name}.arguments do not match its tool schema",
    )


def action_satisfies_extra_tool_schema(
    action: LiteToolCall,
    schema: dict[str, Any],
) -> bool:
    """True when one active extra-tool schema fully accepts the arguments.

    NOT a duplicate of ``core.tools.schemas.tool_call_satisfies_schema(call,
    schema)``, which spells the same parameter ``call``: that one ANDs in
    ``tool_call_matches_schema_route_keys`` first, and this one deliberately does not,
    because its only caller (``prepare_env_tool_calls`` below) has already
    filtered to the key-matching schemas and would otherwise run the key gate
    twice. Adding a rule to either would not force editing the other, so the
    consolidation bar is not met and they stay two predicates.

    The parameter spelling (``action`` here, ``call`` there) is an OPEN owner
    decision, left alone on purpose rather than made
    locally consistent — a parameter is reachable from outside its function body,
    so respelling it lays the undecided layer vocabulary down a second time.
    ``action``/``actions`` is by far the dominant spelling across ``lite/``.
    """
    return not core_tool_schema_validation_errors(schema, tool_call_arguments(action))


def prepare_env_tool_calls(
    actions: list[RuntimeEnvAction],
    metadata: LiteCUAMetadata,
    *,
    validate_top_level_action: bool = False,
    defer_extra_schema_validation_for: Collection[str] | None = None,
) -> tuple[list[tuple[EnvAction, str | None]], dict[str, _ToolErrorFeedback]]:
    """Validate model-emittable structure, then expand action-batch calls.

    This is intentionally a small env-side ingress helper. It owns only the
    parts that are universal across envs: canonical envelope shape, malformed
    ``computer/mobile.actions`` payloads, action-batch child membership, and
    child ``valid_actions`` checks. It does NOT decide whether
    ``click(index=...)`` is a BrowserGym/WebVoyager native tool or
    ``click(coordinate=...)`` is GUI; concrete envs keep that name+args routing,
    and with it child-argument validation -- only the env knows which canonical
    actions its backend can run and which extra arguments it consumes.
    """
    out: list[tuple[EnvAction, str | None]] = []
    errors: dict[str, _ToolErrorFeedback] = {}
    deferred_extra_schema_names = frozenset(defer_extra_schema_validation_for or ())
    active_extra_names = active_extra_tool_names(metadata.extra_tool_schemas)
    assert_safe_tool_call_envelopes(actions)

    def record_error(call_id: str | None, message: str) -> None:
        if call_id is None:
            return
        _append_feedback(errors, call_id, _current_feedback(message))

    for action in actions:
        call_id = canonical_result_call_id(action)
        invalid = invalid_env_tool_call_envelope_message(action)
        if invalid:
            record_error(call_id, invalid)
            continue

        parent_call_id = call_id
        name = tool_call_name(action)
        arguments = tool_call_arguments(action)
        matching_extra_schemas = [
            schema for schema in metadata.extra_tool_schemas
            if tool_schema_name(schema) == name
        ] if name in active_extra_names else []
        if name in LITE_ACTION_BATCH_TOOL_NAMES:
            if validate_top_level_action:
                invalid = invalid_top_level_action_message(action, metadata)
                if invalid:
                    record_error(parent_call_id, invalid)
                    continue
            child_actions, batch_error = validate_lite_action_batch_structure(
                name, arguments,
            )
            if batch_error is not None:
                # Envelope-level: no child could be built, so there is nothing to
                # run or to frame. The whole call is answered with one error.
                record_error(
                    parent_call_id,
                    action_batch_structure_error_message(batch_error),
                )
                continue

            # Per-child faults are MODEL faults. Each costs its own child its
            # execution and nothing more: siblings still run, and the rejected
            # child still travels to the env so it keeps its slot in the batch --
            # one action the model emitted still owes one frame and one reason.
            # A child fault is a MODEL fault: it costs its own child its
            # execution and nothing more. The child still travels to the env
            # carrying its reason, so it keeps its slot -- one action the model
            # emitted still owes one answer and one frame.
            child_name_errors = lite_action_batch_child_name_errors(
                name, child_actions,
            )
            for index, child_action in enumerate(child_actions):
                name_error = child_name_errors.get(index)
                reason = (
                    (
                        action_batch_structure_error_message(name_error)
                        if name_error is not None
                        else None
                    )
                    or nested_extra_tool_action_batch_child_message(
                        name,
                        child_action,
                        metadata.extra_tool_schemas,
                    )
                    or invalid_action_message(child_action, metadata.valid_actions)
                )
                if reason:
                    child_action = with_runtime_rejected_reason(child_action, reason)
                out.append((child_action, parent_call_id))
            continue

        elif validate_top_level_action:
            invalid = invalid_top_level_action_message(action, metadata)
            if invalid:
                record_error(parent_call_id, invalid)
                continue

        if matching_extra_schemas and name not in deferred_extra_schema_names:
            key_matching_extra_schemas = [
                schema
                for schema in matching_extra_schemas
                if tool_call_matches_schema_route_keys(action, schema)
            ]
            if key_matching_extra_schemas:
                if any(
                    action_satisfies_extra_tool_schema(action, schema)
                    for schema in key_matching_extra_schemas
                ):
                    out.append((_as_env_action(action), parent_call_id))
                    continue
                record_error(
                    parent_call_id,
                    extra_tool_schema_argument_error(action, key_matching_extra_schemas),
                )
                continue
            if not is_lite_action_name_or_action_batch_tool_name(name):
                record_error(
                    parent_call_id,
                    extra_tool_schema_argument_error(action, matching_extra_schemas),
                )
                continue

        out.append((_as_env_action(action), parent_call_id))
    return out, errors
