"""Env step error feedback carrier and wording helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from lite.gym.errors import FailureCategory

UNSUPPORTED_ACTION_PREFIX = "unsupported action: "
UNKNOWN_TOOL_PREFIX = "unknown tool: "
UNAVAILABLE_ACTION_SUFFIX = " is not available in this task."
BATCH_ABORT_PREFIX = "batch aborted: "
#: Told to a tool call whose OWN actions never ran because a rejected action in
#: an earlier call of the same step aborted execution. Without it the sibling
#: call comes back error-free and reads as a success it never was.
BATCH_ABORT_SIBLING_MESSAGE = (
    f"{BATCH_ABORT_PREFIX}not executed, because an earlier action in this step was rejected"
)
MODEL_ACTION_ERROR_TYPES = (ValueError, TypeError, IndexError, KeyError)
ToolErrorCarrier = Literal["current", "error_only"]
ActionErrorKind = Literal["model_action", "tool_execution", "unsupported_action"]
#: DERIVED from :data:`ActionErrorKind`, never hand-listed. The two used to sit
#: on adjacent lines spelling the same three strings twice, so a fourth kind had
#: to be added in both places or the parser silently rejected every record
#: carrying it. The ``Literal`` is now the single declaration.
ACTION_ERROR_KINDS: frozenset[str] = frozenset(get_args(ActionErrorKind))


@dataclass(frozen=True)
class ToolErrorFeedback:
    """Env-owned per-call error plus the carrier decision for that call."""

    message: str
    carrier: ToolErrorCarrier

    def __post_init__(self) -> None:
        # THE carrier gate. Derived from the annotation on ``carrier`` above, so
        # the vocabulary is declared exactly once; and because this runs on every
        # construction of a frozen dataclass, every downstream consumer may treat
        # ``carrier`` as already being one of ``get_args(ToolErrorCarrier)``.
        if self.carrier not in get_args(ToolErrorCarrier):
            raise ValueError(f"invalid tool error carrier: {self.carrier!r}")


@dataclass(frozen=True)
class BackendExecutionErrorDetail:
    """Error detail that a caller has already classified as backend-only."""

    detail: BaseException | str


@dataclass(frozen=True)
class ContainerActionErrorRecord:
    """Typed per-action error record returned by env containers."""

    index: int
    kind: ActionErrorKind
    name: str
    error: str
    message: str
    call_id: str | None = None


@dataclass(frozen=True)
class ModelVisibleErrorDetail:
    """Error detail that a caller has already classified as prompt-safe."""

    message: str


# OPEN — owner decision, NOT a settled shape. The two helpers
# below are one 2-field frozen dataclass wearing two names, and collapsing them
# into ``ToolErrorFeedback(message, carrier=...)`` is a mechanical 69-call-site
# edit whose only cost is readability at the call site. What the "two competing
# idioms" reading gets wrong:
# ``[measured over lite/, build/lib excluded: ``ToolErrorFeedback(`` is
# constructed at 5 sites and ALL 5 are in this file — zero direct constructions
# anywhere else in the tree. Two of those 5 are these helpers (literal carrier);
# the other three (``append_feedback``, ``record_model_action_error``,
# ``record_tool_execution_error``) all pass a *variable*, never a literal. The
# helpers have 69 call sites in lite/, 68 of them outside this package.]``
# So there is no rival direct-constructor idiom to converge on: the choice is
# "two names for one dataclass" vs "one constructor spelled 69 times", and it is
# the owner's to make. Do not collapse these on a passing-suite argument alone.
def current_feedback(message: str) -> ToolErrorFeedback:
    """Return feedback that should carry the env's current observation payload."""
    return ToolErrorFeedback(message=message, carrier="current")


def error_only_feedback(message: str) -> ToolErrorFeedback:
    """Return feedback that should not inherit the env's current observation."""
    return ToolErrorFeedback(message=message, carrier="error_only")


def _record_index(
    record: dict[str, Any],
    result_call_ids: Sequence[str | None],
) -> int | None:
    index = record.get("index")
    if isinstance(index, int):
        return index
    call_id = record.get("call_id")
    if isinstance(call_id, str) and call_id in result_call_ids:
        return result_call_ids.index(call_id)
    return None


def parse_container_action_error_record(
    record: Any,
    *,
    result_call_ids: Sequence[str | None] = (),
    fallback_name: str | None = None,
) -> ContainerActionErrorRecord | None:
    """Parse the shared typed env-container ``action_errors[]`` record shape.

    In-container servers emit ``{index, kind, name, error, message}`` plus an
    optional ``call_id``; each fact has exactly one spelling on the wire.
    ``result_call_ids`` resolves the index for a record that carries only
    ``call_id``, and ``fallback_name`` names an action whose envelope was too
    malformed for the container to read a name from.
    """
    if not isinstance(record, dict):
        return None
    index = _record_index(record, result_call_ids)
    if index is None:
        return None
    kind = record.get("kind")
    if kind not in ACTION_ERROR_KINDS:
        return None
    name = str(record.get("name") or fallback_name or "action")
    error = str(record.get("error") or "")
    message = str(record.get("message") or f"{name}: {error}")
    call_id = record.get("call_id")
    return ContainerActionErrorRecord(
        index=index,
        kind=kind,
        name=name,
        error=error,
        message=message,
        call_id=call_id if isinstance(call_id, str) else None,
    )


def append_feedback(
    feedback: dict[str, ToolErrorFeedback],
    call_id: str,
    item: ToolErrorFeedback,
) -> None:
    """Append per-call feedback, preserving current-carrier precedence."""
    existing = feedback.get(call_id)
    if existing is None:
        feedback[call_id] = item
        return
    carrier: ToolErrorCarrier = (
        "current" if "current" in (existing.carrier, item.carrier) else "error_only"
    )
    feedback[call_id] = ToolErrorFeedback(
        message=f"{existing.message}\n{item.message}",
        carrier=carrier,
    )


def unsupported_action_message(name: str) -> str:
    """D6 wording for an action the env physically cannot execute.

    Known inactive extra tools use ``unavailable_action_message`` instead; a
    literal unknown tool name uses ``unknown_tool_message``.
    """
    return f"{UNSUPPORTED_ACTION_PREFIX}{name}"


def unavailable_action_message(name: str) -> str:
    """D6 wording for a known tool/action that is inactive for this task."""
    return f"{name}{UNAVAILABLE_ACTION_SUFFIX}"


def unknown_tool_message(name: str) -> str:
    """D6 wording for a literal tool name this env does not know."""
    return f"{UNKNOWN_TOOL_PREFIX}{name}"


def invalid_action_arguments_message(name: str, reason: BaseException | str) -> str:
    """D6 wording for model-visible malformed arguments on a known action."""
    return f"invalid arguments for {name}: {reason}"


def tool_execution_error_message(name: str, reason: BaseException | str) -> str:
    """D6 wording for a routed action whose backend execution failed."""
    return f"{name} failed: {reason}"


def invalid_action_not_available_message(name: str) -> str:
    """Concise model-visible wording for a task-surface action rejection."""
    return f"invalid action: {name}; choose an available action for this task"


def _quoted_values(text: str) -> list[str]:
    values: list[str] = []
    parts = text.split("'")
    for index in range(1, len(parts), 2):
        values.append(parts[index])
    return values


def _model_visible_key_error_detail(detail: str) -> str | None:
    """Translate backend key-PROJECTION wording into agent-useful guidance.

    Scope is exactly the five ``keys.to_*`` raises, and that is a stated
    contract, not an omission: ``model_inputs.project_model_keys`` is the only
    production entry into ``lite.gym.utils.backend.keys``, and its own docstring
    pins what a key error reaching here may say. Everything it raises itself is
    already model-safe and falls through untouched; ``keys._require_canonical``
    cannot raise on that path at all, so translating its wording would be dead
    code (it was, for seven pinned test rows).

    ``quoted`` is likewise non-empty by construction, not by luck: every
    ``keys.to_*`` raise interpolates ``{token!r}``, and ``repr`` of a ``str``
    always contains at least one apostrophe -- as a delimiter, or, when the
    delimiters switch to ``"``, because the token itself holds one.
    """
    if not detail.startswith("keys.to_"):
        return None
    quoted = _quoted_values(detail)
    if (
        "no X keysym for" in detail
        or "no Playwright key for" in detail
        or "no pynput key for" in detail
        or "no WebDriver key for" in detail
    ):
        return f"unknown key token {quoted[-1]!r}"
    if "not in pyautogui.KEYBOARD_KEYS" in detail:
        return f"unsupported key token {quoted[0]!r}"
    return None


def _model_visible_coordinate_error_detail(detail: str) -> str | None:
    """Translate coordinate parser wording into agent-useful guidance."""
    prefix = "malformed normalized coordinate: "
    if detail.startswith(prefix):
        raw_value = detail.removeprefix(prefix)
        if raw_value == "None":
            return "coordinate is required"
        return "coordinate must be a [x, y] pair"
    if detail.startswith("invalid normalized coordinate values: "):
        return "coordinate values must be finite numbers"
    return None


def model_visible_error_detail(
    reason: BaseException | str | BackendExecutionErrorDetail | ModelVisibleErrorDetail,
    *,
    fallback: str,
) -> str:
    """Return model-safe detail for env feedback.

    Raw backend exception text can stay in debug metadata, logs, and ``info``.
    The paired ``LiteToolResult.error`` is prompt-visible, so backend callers
    must opt into prompt text with ``ModelVisibleErrorDetail`` rather than
    relying on substring guesses.
    """
    if isinstance(reason, BackendExecutionErrorDetail):
        return fallback
    if isinstance(reason, ModelVisibleErrorDetail):
        return reason.message or fallback
    if getattr(reason, "failure_category", None) == FailureCategory.INFRA_FAILURE:
        return fallback

    detail = str(reason)
    if not detail:
        return fallback
    key_detail = _model_visible_key_error_detail(detail)
    if key_detail is not None:
        return key_detail
    coordinate_detail = _model_visible_coordinate_error_detail(detail)
    if coordinate_detail is not None:
        return coordinate_detail
    if detail.startswith("keys.to_xdotool: "):
        detail = detail.removeprefix("keys.to_xdotool: ")
    return detail


def _tool_execution_visible_detail(
    error: BaseException | str | BackendExecutionErrorDetail | ModelVisibleErrorDetail,
    *,
    fallback: str,
) -> str:
    """Prompt-visible detail for an error already classified as tool execution."""
    if isinstance(error, ModelVisibleErrorDetail):
        return error.message or fallback
    if getattr(error, "failure_category", None) == FailureCategory.MODEL_ACTION_ERROR:
        return model_visible_error_detail(error, fallback=fallback)
    return fallback


def record_model_action_error(
    errors: dict[str, ToolErrorFeedback],
    call_id: str | None,
    error: BaseException | str | BackendExecutionErrorDetail | ModelVisibleErrorDetail,
    *,
    carrier: ToolErrorCarrier = "current",
    action_name: str | None = None,
) -> None:
    """Record a model-caused action argument error for this tool call."""
    if call_id:
        detail = model_visible_error_detail(
            error,
            fallback="arguments could not be interpreted",
        )
        message = (
            invalid_action_arguments_message(action_name, detail)
            if action_name
            else f"invalid action arguments: {detail}"
        )
        append_feedback(
            errors,
            call_id,
            ToolErrorFeedback(message=message, carrier=carrier),
        )


def record_tool_execution_error(
    errors: dict[str, ToolErrorFeedback],
    call_id: str | None,
    error: BaseException | str | BackendExecutionErrorDetail | ModelVisibleErrorDetail,
    *,
    carrier: ToolErrorCarrier = "current",
    action_name: str | None = None,
) -> None:
    """Record an env/backend execution failure for this tool call."""
    if call_id:
        detail = _tool_execution_visible_detail(
            error,
            fallback="execution failed",
        )
        message = tool_execution_error_message(action_name, detail) if action_name else detail
        append_feedback(
            errors,
            call_id,
            ToolErrorFeedback(message=message, carrier=carrier),
        )


def batch_abort_message(dropped: int) -> str:
    """D6 wording for the actions a rejected action took down with it."""
    plural = "action was" if dropped == 1 else "actions were"
    return (
        f"{BATCH_ABORT_PREFIX}the {dropped} later {plural} not executed; "
        "a batch stops at the first rejected action"
    )


def record_batch_abort(
    errors: dict[str, ToolErrorFeedback],
    call_id: str | None,
    dropped: Sequence[tuple[Any, str | None]],
    *,
    carrier: ToolErrorCarrier = "current",
) -> None:
    """Tell the model that a rejected action aborted the rest of this step.

    THE explicit half of the abort policy. Every env that ``break``s its action
    loop on a rejected action must call this, because an action-batch call
    collapses onto ONE ``call_id``: without it the model is told only that
    action *k* was bad and is never told that actions *k+1..* never ran, so it
    re-emits the identical action-batch forever.

    ``dropped`` is the UNCONSUMED tail of the env's ``(action, result_call_id)``
    loop sequence -- pass ``actions[index + 1:]`` straight in. The tail can span
    MORE than the failing call (a step may carry several action-batch calls),
    and a sibling call whose actions silently never ran would otherwise come
    back clean and read as a success it never was.
    """
    if not dropped:
        return
    if call_id:
        append_feedback(
            errors,
            call_id,
            ToolErrorFeedback(message=batch_abort_message(len(dropped)), carrier=carrier),
        )
    seen: set[str] = set()
    for _action, dropped_call_id in dropped:
        if not dropped_call_id or dropped_call_id == call_id or dropped_call_id in seen:
            continue
        seen.add(dropped_call_id)
        append_feedback(
            errors,
            dropped_call_id,
            ToolErrorFeedback(message=BATCH_ABORT_SIBLING_MESSAGE, carrier=carrier),
        )


__all__ = [
    "ACTION_ERROR_KINDS",
    "BATCH_ABORT_PREFIX",
    "BATCH_ABORT_SIBLING_MESSAGE",
    "MODEL_ACTION_ERROR_TYPES",
    "ActionErrorKind",
    "BackendExecutionErrorDetail",
    "ContainerActionErrorRecord",
    "ModelVisibleErrorDetail",
    "ToolErrorCarrier",
    "ToolErrorFeedback",
    "append_feedback",
    "batch_abort_message",
    "current_feedback",
    "error_only_feedback",
    "invalid_action_arguments_message",
    "invalid_action_not_available_message",
    "model_visible_error_detail",
    "parse_container_action_error_record",
    "record_batch_abort",
    "record_model_action_error",
    "record_tool_execution_error",
    "tool_execution_error_message",
    "unavailable_action_message",
    "unknown_tool_message",
    "unsupported_action_message",
]
