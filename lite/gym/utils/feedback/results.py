"""Env step tool-result assembly and call-id pairing helpers."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from lite.core.tools.calls import (
    CANONICAL_TOOL_CALL_ERROR,
    RUNTIME_INTERNAL_STOP_REASON_KEY,
    RUNTIME_REJECTED_REASON_KEY,
    RUNTIME_RESULT_CALL_ID_KEY,
    RUNTIME_SIDECAR_KEYS,
    RuntimeEnvAction,
    canonical_lite_tool_call,
    validate_lite_tool_call,
)
from lite.core.tools.extra_tools import (
    is_internal_finish_tool_call as _is_internal_finish_tool_call,
)
from lite.core.tools.results import LiteToolResult, make_tool_result
from lite.gym.types import (
    LiteEnvStepResult,
)
from lite.gym.utils.feedback.errors import (
    ToolErrorFeedback,
)


def invalid_tool_call_envelope_message(action: Any) -> str | None:
    """Return an error for a structurally malformed canonical tool call.

    A canonical call is ``{"id"?, "type": "function", "function": ...}``. A model
    can emit a bad envelope through an adapter that passes its tool-call JSON
    through (e.g. ``arguments`` as a list), so this is MODEL feedback, not an
    infra fault: the caller pairs it to the call instead of letting an
    ``AttributeError``/``KeyError`` escape ``step()`` and kill the rollout.

    Wording only: the clause set is owned by
    :func:`lite.core.tools.calls.validate_lite_tool_call`; this function only
    wraps the canonical reason in env feedback wording.
    """
    reason = validate_lite_tool_call(
        canonical_lite_tool_call(action),
        "tool_call",
        require_id=False,
    )
    if reason is None:
        return None
    return f"invalid tool call: {reason}"


def validate_result_call_id(call_id: Any) -> str:
    """Return ``call_id`` when it is a non-empty string, else raise ``ValueError``."""
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("tool call id must be a non-empty string when present")
    return call_id


def canonical_result_call_id(tool_call: Any) -> str | None:
    """Return the id a role:tool result must pair against, or ``None``.

    An internal finish call answers through its
    ``RUNTIME_RESULT_CALL_ID_KEY`` sidecar instead of its own ``id`` (it has
    none on the model surface); every other call pairs by canonical ``id``.
    """
    if not isinstance(tool_call, dict):
        return None
    try:
        if (
            RUNTIME_RESULT_CALL_ID_KEY in tool_call
            and _is_internal_finish_tool_call(tool_call)
        ):
            return validate_result_call_id(tool_call[RUNTIME_RESULT_CALL_ID_KEY])
    except (KeyError, TypeError):
        return None
    if "id" not in tool_call:
        return None
    return validate_result_call_id(tool_call["id"])


#: Env feedback wording for a runtime sidecar a model-emitted call may never
#: spell, one entry per key of ``RUNTIME_SIDECAR_KEYS``. Written out rather than
#: formatted so each reservation stays greppable from the key it names.
_RESERVED_SIDECAR_MESSAGES = MappingProxyType({
    RUNTIME_INTERNAL_STOP_REASON_KEY: (
        "invalid tool call: _internal_stop_reason is reserved"
    ),
    RUNTIME_RESULT_CALL_ID_KEY: "invalid tool call: _result_call_id is reserved",
    RUNTIME_REJECTED_REASON_KEY: "invalid tool call: _rejected_reason is reserved",
})


def _reserved_runtime_sidecar_key(action: Any) -> str | None:
    """Return a runtime-only key this call must not carry, or ``None``.

    Runtime sidecars are stamped below the model surface, so only a runtime
    internal finish may carry one. Every model-emitted call is disqualified by
    the canonical ``id`` it was stamped with, which is what stops a model from
    spelling itself an env-internal terminal action.
    """
    if not isinstance(action, dict):
        return None
    present = sorted(RUNTIME_SIDECAR_KEYS & action.keys())
    if not present:
        return None
    try:
        if _is_internal_finish_tool_call(action):
            return None
    except (KeyError, TypeError):
        pass
    return present[0]


def _runtime_sidecar_error(action: Any) -> str | None:
    if (reserved := _reserved_runtime_sidecar_key(action)) is not None:
        return _RESERVED_SIDECAR_MESSAGES[reserved]
    if not isinstance(action, dict) or RUNTIME_RESULT_CALL_ID_KEY not in action:
        return None
    try:
        validate_result_call_id(action[RUNTIME_RESULT_CALL_ID_KEY])
    except ValueError as e:
        return str(e)
    return None


def invalid_env_tool_call_envelope_message(action: Any) -> str | None:
    """Return model-visible feedback for pairable env-ingress envelope defects.

    This is not the full direct-ingress gate; noncanonical outer keys and bad
    ids are infra-shape errors handled by ``assert_safe_tool_call_envelopes``.
    """
    if sidecar_error := _runtime_sidecar_error(action):
        return sidecar_error
    return invalid_tool_call_envelope_message(action)


def assert_safe_tool_call_envelopes(actions: list[RuntimeEnvAction]) -> None:
    """Fail loud for malformed calls that cannot be paired to a result.

    Model-emitted argument errors are useful feedback only when the standard
    agent loop can bind them back to a tool call. A malformed Lite call without
    a canonical ``id`` cannot be answered with a role:tool message, so letting
    it flow into env-owned feedback would create unpaired non-terminal output.
    """
    for action in actions:
        call_id = canonical_result_call_id(action)
        invalid = invalid_env_tool_call_envelope_message(action)
        if _reserved_runtime_sidecar_key(action) is not None:
            raise TypeError(invalid)
        try:
            internal_finish = _is_internal_finish_tool_call(action)
        except (KeyError, TypeError):
            internal_finish = False
        if not internal_finish:
            shape_error = validate_lite_tool_call(action, "action", require_id=False)
            if shape_error is not None:
                pairable = invalid is not None and call_id is not None
                if not pairable:
                    raise TypeError(CANONICAL_TOOL_CALL_ERROR)
                noncanonical_shape = (
                    "noncanonical" in shape_error
                    or "tool_call_id" in shape_error
                    or ".id must be" in shape_error
                )
                if noncanonical_shape:
                    raise TypeError(shape_error)
        if invalid and call_id is None:
            raise TypeError(CANONICAL_TOOL_CALL_ERROR)


def ordered_tool_call_ids(tool_calls: list[RuntimeEnvAction]) -> list[str | None]:
    """Extract result IDs in assistant call order.

    Public model tool calls are paired by canonical ``id`` only.
    """
    assert_safe_tool_call_envelopes(tool_calls)
    ordered: list[str | None] = []
    seen: set[str] = set()
    for tool_call in tool_calls:
        call_id = canonical_result_call_id(tool_call)
        if call_id is None:
            ordered.append(None)
            continue
        if call_id in seen:
            raise ValueError(f"duplicate tool call id: {call_id}")
        seen.add(call_id)
        ordered.append(call_id)
    return ordered


def build_tool_results_from_decisions(
    result: LiteEnvStepResult,
    *,
    ordered_call_ids: Sequence[str | None],
    continue_call_ids: Collection[str | None] | None = None,
    images: Sequence[bytes] | None = None,
    text: str | None = None,
    metadata: dict[str, Any] | None = None,
    feedback: Mapping[str, ToolErrorFeedback] | None = None,
    explicit_results: Mapping[str, LiteToolResult] | None = None,
) -> LiteEnvStepResult:
    """Populate results from explicit env-owned per-call decisions.

    The concrete env decides which call IDs receive the current observation,
    which receive current-carrier errors, which are error-only, and which have
    env-built explicit results such as bash output. This helper only assembles
    those decisions in assistant call order.
    """
    feedback = feedback or {}
    explicit_results = explicit_results or {}
    ordered = list(ordered_call_ids)
    for call_id in ordered:
        if call_id is not None:
            validate_result_call_id(call_id)
    for call_id in feedback:
        validate_result_call_id(call_id)
    for call_id in explicit_results:
        validate_result_call_id(call_id)
    ordered_ids = [cid for cid in ordered if cid is not None]
    ordered_set = set(ordered_ids)
    if len(ordered_ids) != len(ordered_set):
        duplicates = sorted(cid for cid in ordered_set if ordered_ids.count(cid) > 1)
        raise ValueError(f"duplicate tool call ids: {duplicates}")
    unknown_feedback = sorted(set(feedback) - ordered_set)
    if unknown_feedback:
        raise ValueError(f"feedback for unknown tool call ids: {unknown_feedback}")
    unknown_explicit = sorted(set(explicit_results) - ordered_set)
    if unknown_explicit:
        raise ValueError(f"explicit results for unknown tool call ids: {unknown_explicit}")
    conflicts = sorted(set(feedback) & set(explicit_results))
    if conflicts:
        raise ValueError(f"conflicting feedback and explicit results for call ids: {conflicts}")
    current_ids = frozenset(continue_call_ids or ())
    for call_id in current_ids:
        if call_id is not None:
            validate_result_call_id(call_id)
    unknown_current = sorted(
        cid for cid in current_ids if cid is not None and cid not in ordered_set
    )
    if unknown_current:
        raise ValueError(f"continue results for unknown tool call ids: {unknown_current}")
    if result.results:
        if feedback or explicit_results or current_ids:
            raise ValueError(
                "prebuilt tool results cannot be combined with feedback, "
                "explicit results, or continue call ids"
            )
        return result
    # Per-action result frames, in action order: one per EXECUTED action. The
    # agent layer stores every one and shows only the subset its adapter selects
    # (default: the last). A path where nothing executed still owes the model one
    # observation, so its caller passes a single-element list.
    current_images: list[bytes] = list(images or [])
    results: list[LiteToolResult] = []

    def current_result(call_id: str | None, *, error: str | None = None) -> LiteToolResult:
        return make_tool_result(
            tool_call_id=call_id,
            images=list(current_images),
            text=text,
            metadata=metadata,
            error=error,
        )

    for call_id in ordered:
        if call_id in explicit_results:
            explicit = explicit_results[call_id]
            if explicit.tool_call_id != call_id:
                raise ValueError(
                    "explicit result tool_call_id mismatch: "
                    f"key {call_id!r} != result.tool_call_id {explicit.tool_call_id!r}"
                )
            results.append(explicit)
            continue
        if call_id in feedback:
            item = feedback[call_id]
            # Binary dispatch, not a third copy of the carrier vocabulary:
            # ``ToolErrorFeedback.__post_init__`` is the one gate and it already
            # rejected anything outside ``get_args(ToolErrorCarrier)``. The
            # ``else: raise`` that used to sit here was unreachable (the type is
            # frozen, so no construction path skips the gate) and duplicated the
            # gate's message verbatim.
            if item.carrier == "current":
                results.append(current_result(call_id, error=item.message))
            else:
                results.append(make_tool_result(tool_call_id=call_id, error=item.message))
            continue
        if call_id in current_ids:
            if call_id is None and (result.terminated or result.truncated):
                continue
            results.append(current_result(call_id))

    result.results = results
    return result
