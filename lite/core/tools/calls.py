"""Canonical Lite tool-call shapes, constructors, and id stamping.

Owns the call<->result PAIRING INVARIANT for persisted assistant calls: one
result per call, keyed by the call's ``id`` and the result message's
``tool_call_id``.

Runtime-only env actions may carry sidecar keys while crossing env ingress.
Those keys are not canonical ``LiteToolCall`` fields, so their names carry a
``RUNTIME_`` prefix and they stay out of ``__all__``: ``RuntimeEnvAction``,
``RUNTIME_RESULT_CALL_ID_KEY``, ``RUNTIME_INTERNAL_STOP_REASON_KEY``,
``RUNTIME_SIDECAR_KEYS``, ``canonical_lite_tool_call()``,
``runtime_internal_stop_reason()``, and ``with_runtime_internal_stop_reason()``.
``validate_lite_tool_call()`` rejects the sidecar keys, which is what keeps a
runtime action off persisted assistant calls and stored rows.

``EnvAction`` is the other side of env ingress: the BARE ``{"name",
"arguments"}`` shape concrete envs execute. It is declared here so the envelope
and the action lowered out of it have one owner and one spelling.

The two *derived* helpers that read the annotation together with the finish-tool
catalog live in :mod:`lite.core.tools.extra_tools`, because they need
``LiteFinishToolSet`` and that module already imports this one.

Run:
    uv run pytest \
        tests/core/tools/test_lite_tool_call_shape.py \
        tests/core/tools/test_tool_call_ids.py \
        tests/core/tools/test_internal_calls.py -q
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Literal, TypedDict

from lite.core.errors import ToolCallValidationError

CANONICAL_TOOL_CALL_ERROR = (
    "env.step expects canonical Lite tool calls "
    "{'id?', 'type', 'function'}, not bare or provider payloads"
)


class _LiteToolCallOptional(TypedDict, total=False):
    id: str


class LiteToolCallFunction(TypedDict):
    """Function invocation inside a canonical Lite tool call."""

    name: str
    arguments: dict[str, Any]


class LiteToolCall(_LiteToolCallOptional):
    """Canonical persisted CUA-Lite tool call.

    Parser/action-space code may construct pre-stamp dicts without ``id``. Once
    a call is persisted on an assistant turn, ``id`` is the pairing key for the
    corresponding role:tool result's ``tool_call_id``.
    """

    type: Literal["function"]
    function: LiteToolCallFunction


#: Every key a canonical Lite tool call may carry. DERIVED from the two
#: TypedDicts above rather than re-listed, so the shape is declared exactly once:
#: a new field on ``LiteToolCall`` is admitted by every key-set check without an
#: edit, and a field removed from it stops being admitted.
_LITE_TOOL_CALL_KEYS = MappingProxyType({
    "outer": LiteToolCall.__required_keys__ | LiteToolCall.__optional_keys__,
    "function": (
        LiteToolCallFunction.__required_keys__
        | LiteToolCallFunction.__optional_keys__
    ),
})

#: Runtime sidecar carrying the id of the model call an internal env action must
#: still answer, so an intercepted call gets a paired role:tool result.
RUNTIME_RESULT_CALL_ID_KEY = "_result_call_id"

#: Runtime sidecar stamped on an action built below the model surface. Its
#: VALUES are owned by ``lite.core.messages.final.INTERNAL_STOP_REASONS``; this
#: module owns only the key, so annotating an action costs no messages import.
RUNTIME_INTERNAL_STOP_REASON_KEY = "_internal_stop_reason"

#: Runtime sidecar marking an action the MODEL got wrong -- a name the batch tool
#: does not carry, a standalone extra tool nested inside a batch, a name the task
#: config withholds. The env must NOT dispatch it, and must NOT drop it: it still
#: occupies one slot of the batch the model emitted, so it still owes one frame
#: and one model-visible reason. Carrying the reason on the action keeps a single
#: recorder -- whoever detected the fault wrote it here, and the env only
#: replays it.
RUNTIME_REJECTED_REASON_KEY = "_rejected_reason"


class RuntimeEnvAction(LiteToolCall, total=False):
    """A canonical call plus the runtime facts of an action built below the
    model surface: which model call it must still answer, and why it stopped.

    The sidecars are ordinary dict keys, so they reach a remote env-server
    through ``/step`` JSON exactly as they cross a direct in-process call. They
    are NOT canonical shape: :func:`validate_lite_tool_call` rejects them, which
    is what keeps them off persisted assistant calls and stored rows. Env
    ingress (``lite.gym.utils.feedback``) is their only reader, and it reserves
    them -- a call carrying an ``id`` is model output and may never spell one.
    """

    _internal_stop_reason: str
    _result_call_id: str
    _rejected_reason: str


#: What a runtime action adds on top of a canonical call. DERIVED from
#: :class:`RuntimeEnvAction` so the runtime shape is declared exactly once.
RUNTIME_SIDECAR_KEYS = frozenset(
    RuntimeEnvAction.__optional_keys__ - LiteToolCall.__optional_keys__
)

#: The env-internal BARE action layer: ``{"name", "arguments"}`` plus an optional
#: ``call_id`` and the runtime sidecars. Action-batch children and every concrete
#: env's post-ingress action list are this shape, so the layer has one spelling
#: instead of one alias per module.
#:
#: NOT :class:`RuntimeEnvAction`, which is the canonical ``{"type", "function"}``
#: envelope crossing ``env.step()``. These are the two sides of the lowering
#: ``lite.gym.utils.feedback.ingress`` applies, not two names for one fact.
EnvAction = dict[str, Any]


def canonical_lite_tool_call(action: Any) -> Any:
    """Return ``action`` with its runtime sidecars removed.

    The projection env ingress applies before asking a canonical-shape question
    about a :class:`RuntimeEnvAction`: the sidecars belong to a different layer,
    so they must not be answered as noncanonical outer keys. Non-dict inputs
    pass through unchanged -- this is a layer projection, and
    :func:`validate_lite_tool_call` owns "is this a dict at all".
    """
    if not isinstance(action, dict) or not RUNTIME_SIDECAR_KEYS & action.keys():
        return action
    return {k: v for k, v in action.items() if k not in RUNTIME_SIDECAR_KEYS}


def validate_lite_tool_call(
    call: Any,
    where: str,
    *,
    require_id: bool,
) -> str | None:
    """Return why ``call`` is not a canonical Lite tool call, or ``None``.

    The one owner of "what shape is a Lite tool call at this boundary". It
    answers with a message rather than raising because its callers disagree
    about the consequence: a publication gate raises ``ValueError``, an
    env-ingress gate turns the same fact into model-visible feedback. Only the
    *fact* is shared.

    ``require_id`` is the pre-stamp/post-stamp distinction
    :class:`LiteToolCall` already documents: parsers build calls without one, and
    ``id`` becomes mandatory the moment a call is persisted on an assistant turn
    and needs a role:tool result to pair against.

    ``where`` is the caller's positional label (e.g.
    ``messages[3].tool_calls[0]``); it prefixes the message so callers do not
    each re-derive one.
    """
    if not isinstance(call, dict):
        return f"{where} must be a dict"
    if set(call) == {"name", "arguments"}:
        return f"{where} is a bare model-function projection, not a canonical Lite tool call"
    if call.get("type") == "function_call":
        return f"{where} is a provider function-call payload, not a canonical Lite tool call"
    extra_keys = sorted(set(call) - _LITE_TOOL_CALL_KEYS["outer"])
    if "tool_call_id" in extra_keys:
        return f"{where} must use id on tool calls, not tool_call_id"
    if extra_keys:
        return f"{where} has noncanonical outer keys {extra_keys}"
    if call.get("type") != "function":
        return f"{where}.type must be 'function'"
    call_id = call.get("id")
    if require_id and (not isinstance(call_id, str) or not call_id):
        return f"{where} missing non-empty id"
    if "id" in call and (not isinstance(call_id, str) or not call_id):
        return f"{where}.id must be a non-empty string when present"
    function = call.get("function")
    if not isinstance(function, dict):
        return f"{where}.function must be an object"
    inner_extra_keys = sorted(set(function) - _LITE_TOOL_CALL_KEYS["function"])
    if inner_extra_keys:
        return f"{where}.function has noncanonical keys {inner_extra_keys}"
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return f"{where}.function.name must be a non-empty string"
    if function.get("arguments") is None:
        return f"{where}.function missing arguments dict"
    if not isinstance(function["arguments"], dict):
        if isinstance(function["arguments"], str):
            return f"{where}.function.arguments must be a dict per §1.4, not a JSON string"
        return (
            f"{where}.function.arguments must be an object, "
            f"got {type(function['arguments']).__name__}"
        )
    return None


def make_tool_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Build a canonical Lite tool-call dict, dropping ``None`` arguments."""
    args = {k: v for k, v in (arguments or {}).items() if v is not None}
    call: dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": args},
    }
    if call_id is not None:
        call = {"id": call_id, **call}
    return call


def tool_call_id(call: dict[str, Any]) -> str | None:
    """Return a canonical Lite tool call's id when present."""
    call_id = call.get("id")
    return call_id if isinstance(call_id, str) else None


def tool_call_name(call: dict[str, Any]) -> str:
    """Return a canonical Lite tool call's function name."""
    return call["function"]["name"]


def tool_call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical Lite tool call's function arguments."""
    return call["function"]["arguments"]


def runtime_internal_stop_reason(action: dict[str, Any]) -> str | None:
    """Return the stop reason stamped on a :class:`RuntimeEnvAction`, if any.

    Reads plain canonical calls too, and answers ``None`` for them: a caller
    holding either shape asks this one question.
    """
    reason = action.get(RUNTIME_INTERNAL_STOP_REASON_KEY)
    return reason if isinstance(reason, str) else None


def with_runtime_internal_stop_reason(
    action: dict[str, Any],
    reason: str,
) -> RuntimeEnvAction:
    """Return a copy of ``action`` promoted to a :class:`RuntimeEnvAction`.

    Copies rather than mutates, so the canonical call a caller may still persist
    never grows the sidecar.
    """
    return {**action, RUNTIME_INTERNAL_STOP_REASON_KEY: reason}


def runtime_rejected_reason(action: dict[str, Any]) -> str | None:
    """Return why the model got this action wrong, or ``None`` to dispatch it.

    Reads plain canonical calls too, and answers ``None`` for them: a caller
    holding either shape asks this one question.
    """
    reason = action.get(RUNTIME_REJECTED_REASON_KEY)
    return reason if isinstance(reason, str) else None


def with_runtime_rejected_reason(
    action: dict[str, Any],
    reason: str,
) -> RuntimeEnvAction:
    """Return a copy of ``action`` marked as a model fault the env must not run.

    Copies rather than mutates, so the canonical call a caller may still persist
    never grows the sidecar.
    """
    return {**action, RUNTIME_REJECTED_REASON_KEY: reason}


def _is_lite_tool_call_dict(value: dict[str, Any]) -> bool:
    """Sniff a Lite call apart from a message dict, by presence only.

    Keys come from :class:`LiteToolCall` rather than being re-listed, so this
    stays the same key set the validator above enforces. It is deliberately a
    PRESENCE test, not a validation: it runs on the pre-stamp inputs the stamper
    is about to fix up.
    """
    return LiteToolCall.__required_keys__ <= value.keys() and "tool_calls" not in value


def _message_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    if _is_lite_tool_call_dict(message):
        raise TypeError("message must be a message dict, not a Lite tool-call dict")
    if "role" not in message:
        raise TypeError("message must be a message dict with a role")
    calls = message["tool_calls"] if "tool_calls" in message else []
    if not isinstance(calls, list):
        raise TypeError('message["tool_calls"] must be a list of dict calls')
    return calls


def _messages_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{i}] must be a dict")
        if _is_lite_tool_call_dict(message):
            raise TypeError(
                f"messages[{i}] must be a message dict, not a Lite tool-call dict"
            )
        if "role" not in message:
            raise TypeError(f"messages[{i}] must be a message dict with a role")
        if "tool_calls" not in message:
            message_calls = []
        else:
            message_calls = message["tool_calls"]
        if not isinstance(message_calls, list):
            raise TypeError(f'messages[{i}]["tool_calls"] must be a list of dict calls')
        calls.extend(message_calls)
    return calls


def _validate_present_call_ids(
    tool_calls: list[dict[str, Any]],
    *,
    reject_duplicates: bool,
) -> set[str]:
    used: set[str] = set()
    for i, call in enumerate(tool_calls):
        shape_error = validate_lite_tool_call(
            call,
            f"tool_calls[{i}]",
            require_id=False,
        )
        if shape_error is not None:
            raise ToolCallValidationError(shape_error)
        if "id" not in call:
            continue
        call_id = call["id"]
        if reject_duplicates and call_id in used:
            raise ToolCallValidationError(f"duplicate id {call_id!r}")
        used.add(call_id)
    return used


def _rewrite_call_id_in_order(call: dict[str, Any], call_id: str) -> None:
    """Rewrite one call in canonical key order: ``id, type, function``."""
    function = call["function"]
    call.clear()
    call["id"] = call_id
    call["type"] = "function"
    call["function"] = function


def stamp_tool_call_list_ids(
    tool_calls: list[dict[str, Any]],
    *,
    preserve: bool,
    start: int = 0,
) -> list[dict[str, Any]]:
    """Stamp canonical ``id`` values on a Lite ``tool_calls`` list.

    Both modes validate call shape first. Only ``preserve=True`` also rejects
    duplicate existing ids, because only that mode keeps them: it fills the gaps
    around ids the caller wants to survive, so two calls claiming one id is a
    caller error. ``preserve=False`` restamps the whole list from ``start``, so
    whatever ids arrived -- duplicates included -- are about to be replaced.
    """
    if not isinstance(tool_calls, list):
        raise TypeError("tool_calls must be a list of dict calls")

    for i, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            raise TypeError(f"tool_calls[{i}] must be a dict")
        if "role" in call or "tool_calls" in call:
            raise TypeError(f"tool_calls[{i}] must be a Lite tool-call dict")
    used = _validate_present_call_ids(
        tool_calls,
        reject_duplicates=preserve,
    )

    if not preserve:
        for offset, call in enumerate(tool_calls):
            _rewrite_call_id_in_order(call, f"call_{start + offset:04d}")
        return tool_calls

    next_id = start
    for call in tool_calls:
        if isinstance(call.get("id"), str) and call["id"]:
            call_id = call["id"]
        else:
            while True:
                call_id = f"call_{next_id:04d}"
                next_id += 1
                if call_id not in used:
                    break
        _rewrite_call_id_in_order(call, call_id)
        used.add(call_id)
    return tool_calls


def stamp_message_tool_call_ids(
    message: dict[str, Any],
    *,
    preserve: bool,
    start: int = 0,
) -> dict[str, Any]:
    """Stamp canonical ``id`` values on one Lite message in place."""
    stamp_tool_call_list_ids(
        _message_tool_calls(message),
        preserve=preserve,
        start=start,
    )
    return message


def stamp_messages_tool_call_ids(
    messages: list[dict[str, Any]],
    *,
    preserve: bool,
    start: int = 0,
) -> list[dict[str, Any]]:
    """Stamp canonical ``id`` values across a Lite message list in place."""
    stamp_tool_call_list_ids(
        _messages_tool_calls(messages),
        preserve=preserve,
        start=start,
    )
    return messages


__all__ = [
    "CANONICAL_TOOL_CALL_ERROR",
    "EnvAction",
    "LiteToolCall",
    "make_tool_call",
    "validate_lite_tool_call",
    "stamp_message_tool_call_ids",
    "stamp_messages_tool_call_ids",
    "stamp_tool_call_list_ids",
    "tool_call_arguments",
    "tool_call_id",
    "tool_call_name",
]
