"""Source-neutral message/final-turn helpers for data processing."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from lite.core.messages import keep_model_visible_content, no_tool_call_final_text
from lite.core.messages.final import (
    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY,
    CONTENT_ONLY_FINAL_TEXT,
    canonicalize_no_tool_call_final_message,
    no_tool_call_final_content,
)
from lite.core.metadata import LiteBaseMetadata, LiteCUAMetadata
from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    validate_lite_action_batch_structure,
)
from lite.core.tools.calls import (
    stamp_messages_tool_call_ids,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
)
from lite.core.tools.extra_tools import (
    APP_LAUNCH_TOOL_NAME,
    FINISH_TOOL_ORDER,
    LiteFinishToolSet,
    make_open_app_tool,
)
from lite.core.tools.schemas import tool_schema_name

__all__ = [
    "extra_tool_schemas_for_messages",
    "finalize_use_messages",
    "normalize_content_only_final",
    "pop_terminal_terminate",
    "strip_raw_response_if_message_changed",
    "structural_final_message",
    "terminate_outcome_others",
    "validate_content_only_finals",
]

#: The standalone extra tools a stored ``use``/grounding row may persist a call
#: to, in schema declaration order. Names and order come from
#: ``core.tools.extra_tools``; this module only decides *which families* of that
#: catalog preproc rows can carry (app-launch and finish, never browser-nav).
_SCHEMAS_BY_NAME = {
    APP_LAUNCH_TOOL_NAME: make_open_app_tool(),
    **{name: LiteFinishToolSet.get_tool_schema(name) for name in FINISH_TOOL_ORDER},
}


def structural_final_message(text: str = CONTENT_ONLY_FINAL_TEXT) -> dict[str, Any]:
    """Return the default content-only assistant final turn."""
    content = (
        no_tool_call_final_content()
        if text == CONTENT_ONLY_FINAL_TEXT
        else [{"type": "text", "text": text}]
    )
    return {"role": "assistant", "content": content}


def normalize_content_only_final(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Replace a no-tool-call final assistant turn's content with one clean ``text``.

    The last assistant turn's model-facing content becomes exactly
    :func:`structural_final_message`, and every trainable part it previously held
    -- ``inline_reasoning``, ``action_description``, any other channel, or several
    at once -- is dropped. **Unconditional**: it does not branch on what the turn
    contained, because branching is precisely what re-creates the downstream
    question *"does this turn have a trainable target?"*, i.e. the empty-SFT-target
    bug. A compact internal diagnostic sidecar records which source shape was
    normalized; ``stage`` removes that sidecar before publish.

    ``action_description`` is by definition the "narration accompanying an action"
    channel (see ``lite.core.messages.final.no_tool_call_final_text``), so a turn
    with no action has nothing to narrate and the field is invalid there. That
    docstring already states the reciprocal producer contract; this function is how
    the collect side honours it.

    A final turn that HAS ``tool_calls`` is returned untouched -- the
    ``response``-submitted answer of a QA task, a ``terminate``, or an ordinary
    action. Preprocessors that build rows from scratch append
    :func:`structural_final_message` directly and never need this; it exists for the
    ``devs/data/**`` collect filters, which receive a provider's raw final turn.

    Idempotent and does NOT mutate the input. Returns ``(messages, normalized)``.
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "assistant":
            continue
        if messages[i].get("tool_calls"):
            return messages, False
        canonical = structural_final_message()
        allowed_canonical_keys = {"role", "content", CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY}
        if (
            messages[i].get("content") == canonical["content"]
            and set(messages[i]) <= allowed_canonical_keys
        ):
            return messages, False  # no rewrite needed
        final = canonicalize_no_tool_call_final_message(messages[i])
        out = list(messages)
        out[i] = final
        return out, True
    return messages, False  # no assistant turn (not a real trajectory)


TERMINATE_TOOL_NAME = "terminate"

#: ``metadata.others`` keys that carry a dropped ``terminate``'s payload.
#:
#: Deliberately NOT ``task_completed``: opencua already publishes that key as an
#: *external* judgement of whether the task was accomplished, and on published
#: rows it disagrees with the demonstrator's own ``status`` (5/48 rows read
#: ``task_completed=False`` alongside ``status="success"``). These two keys are
#: the demonstrator's *self-report* at termination, so they get their own names
#: and never overwrite an external label.
TERMINATE_STATUS_KEY = "terminate_status"
TERMINATE_REASON_KEY = "terminate_reason"


def pop_terminal_terminate(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Remove a trailing ``terminate`` call from *messages*, returning it.

    ``use`` rows never persist a ``terminate`` call: the episode always ends on
    :func:`structural_final_message`, and whatever the call asserted moves to
    ``metadata.others`` via :func:`terminate_outcome_others`. Callers therefore
    pop first and append the structural final second.

    Mutates *messages* in place. A turn left with no calls is dropped whole --
    its content (opencua's reflection, its ``last_step_*`` sidecars) goes with
    it, because the replacement is a fresh two-key ``{role, content}`` dict.
    A turn that still holds real actions keeps them, and only sheds the
    terminate (old published cagui / guiodyssey / mind2web rows fold the
    terminate onto the last action turn, so migration needs this branch).
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    calls = messages[-1].get("tool_calls") or []
    if not calls or tool_call_name(calls[-1]) != TERMINATE_TOOL_NAME:
        return None
    dropped = calls[-1]
    remaining = calls[:-1]
    if remaining:
        messages[-1]["tool_calls"] = remaining
    else:
        messages.pop()
    return dropped


def terminate_outcome_others(call: dict[str, Any] | None) -> dict[str, Any]:
    """``metadata.others`` fragment preserving a dropped ``terminate``'s payload.

    Emitted only for a **non-success** status. ``status="success"`` asserts
    nothing beyond "the episode ended", which the ``Done.`` final already
    carries, so recording it would add bytes to ~99.7% of rows and no
    information -- and it would be unrecoverable on the migration side, where a
    published mind2web/gui360 row carries a synthetic trailing
    ``terminate(success)`` that the current preproc never emits at all.
    Absence therefore means "not a self-reported failure".

    ``reason`` is emitted only when the source authored non-blank text; an empty
    reason is not information, and every layer must agree on that or the
    byte-equality bar breaks.
    """
    if call is None or tool_call_name(call) != TERMINATE_TOOL_NAME:
        return {}
    arguments = tool_call_arguments(call)
    status = arguments.get("status", "success")
    if status == "success":
        return {}
    out: dict[str, Any] = {TERMINATE_STATUS_KEY: status}
    reason = arguments.get("reason")
    if isinstance(reason, str) and reason.strip():
        out[TERMINATE_REASON_KEY] = reason
    return out


def extra_tool_schemas_for_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lite schemas required by persisted standalone extra-tool calls.

    The decision is intentionally derived from the stored ``messages`` only. A
    structural content-only ``Done.`` final does not imply a ``terminate`` schema;
    preprocessors that consume source terminal markers should move any non-success
    payload into ``metadata.others`` before calling this helper.
    """
    names = {
        tool_call_name(tc)
        for message in messages
        for tc in (message.get("tool_calls") or [])
        if tool_call_name(tc) in _SCHEMAS_BY_NAME
    }
    return [
        schema
        for schema in _SCHEMAS_BY_NAME.values()
        if tool_schema_name(schema) in names
    ]


def finalize_use_messages(
    messages: list[dict[str, Any]],
    *,
    result_boundary_tools: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Stamp calls and convert post-action screenshot users to role:tool rows.

    Only Lite action-batch calls and explicitly declared standalone
    result-boundary tools consume the following screenshot-only ``user``
    message. Finish tools
    such as ``terminate`` do not: preprocessors should pop terminal markers and
    append the structural ``Done.`` final before calling this helper, preserving
    any non-success payload in ``metadata.others``.
    """
    stamp_messages_tool_call_ids(messages, preserve=True)
    explicit_boundary_tools = frozenset(result_boundary_tools or ())

    finalized: list[dict[str, Any]] = []
    pending_assistant: dict[str, Any] | None = None
    for message in messages:
        if message.get("role") == "assistant":
            finalized.append(message)
            pending_assistant = message if message.get("tool_calls") else None
            continue

        if (
            pending_assistant is not None
            and message.get("role") == "user"
            and _is_screenshot_only_user(message)
        ):
            call = _screen_result_call(
                pending_assistant.get("tool_calls") or [],
                result_boundary_tools=explicit_boundary_tools,
            )
            if call is not None:
                finalized.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id(call),
                    "content": message["content"],
                })
                pending_assistant = None
                continue

        finalized.append(message)
        pending_assistant = None

    return finalized


def strip_raw_response_if_message_changed(
    original: dict[str, Any],
    updated: dict[str, Any],
) -> dict[str, Any]:
    """Drop adapter replay text when a filter changed the assistant message.

    Local raw rollout export may replay ``raw_response`` byte-for-byte for
    same-family tool-call turns. Once a filter rewrites ``content`` or
    ``tool_calls``, that sidecar describes the pre-filter action text and must
    not shadow the canonical cleaned message.
    """
    if original.get("raw_response") is None:
        return updated
    before = dict(original)
    before.pop("raw_response", None)
    after = dict(updated)
    after.pop("raw_response", None)
    if before == after:
        return updated
    stripped = dict(updated)
    stripped.pop("raw_response", None)
    return stripped


def validate_content_only_finals(
    messages: list[dict[str, Any]],
    metadata: LiteBaseMetadata,
    *,
    require_visible_text: bool = True,
) -> list[dict[str, Any]]:
    """Validate CUA ``use`` no-tool-call response/final turns.

    A no-tool assistant turn is either a visible response attempt followed by
    env/user feedback and another assistant turn, or the terminal content-only
    final. Canonical publication/export requires visible text for both shapes.
    Raw rollout validation may accept empty live model content while preserving
    the same ordering constraints.
    """
    if not isinstance(metadata, LiteCUAMetadata) or metadata.task_type != "use":
        return messages
    last_index = len(messages) - 1
    for mi, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if msg.get("tool_calls"):
            continue
        text = no_tool_call_final_text(msg)
        if mi != last_index:
            next_msg = messages[mi + 1]
            if next_msg.get("role") != "user":
                raise ValueError(
                    "content-only assistant turn must be followed by env feedback "
                    "or be the final message "
                    f"(messages[{mi}])"
                )
            if not _message_has_model_visible_content(next_msg):
                raise ValueError(
                    "content-only assistant turn must be followed by model-visible "
                    "env feedback "
                    f"(messages[{mi}])"
                )
            if not any(later.get("role") == "assistant" for later in messages[mi + 2:]):
                raise ValueError(
                    "content-only assistant turn followed by env feedback must "
                    "continue to another assistant turn "
                    f"(messages[{mi}])"
                )
            if require_visible_text and not text:
                raise ValueError(
                    "content-only response assistant turn must contain non-empty "
                    "plain text "
                    f"(messages[{mi}])"
                )
            continue
        if require_visible_text and not text:
            raise ValueError(
                "content-only final assistant turn must contain non-empty plain "
                "text "
                f"(messages[{mi}])"
            )
    return messages


def _message_has_model_visible_content(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return True
    if not isinstance(content, list):
        return False
    visible = keep_model_visible_content([message])[0].get("content")
    return isinstance(visible, list) and bool(visible)


def _is_screenshot_only_user(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        isinstance(content, list)
        and len(content) == 1
        and isinstance(content[0], dict)
        and content[0].get("type") == "image"
    )


def _screen_result_call(
    tool_calls: list[dict[str, Any]],
    *,
    result_boundary_tools: frozenset[str],
) -> dict[str, Any] | None:
    for call in tool_calls:
        name = tool_call_name(call)
        if _is_action_wrapper_result_boundary(call):
            return call
        if name in result_boundary_tools:
            if not isinstance(tool_call_arguments(call), dict):
                raise ValueError("canonical standalone tool_call.arguments must be a dict")
            return call
    return None


def _is_action_wrapper_result_boundary(call: dict[str, Any]) -> bool:
    """True when *call* is a Lite action-batch call whose batch consumes the next screenshot.

    The batch *shape* is not restated here: ``validate_lite_action_batch_structure``
    owns that grammar, and re-checking it by hand only bought a second, laxer copy.
    Child-action **membership** is deliberately not enforced -- pairing keys on the
    batch's structure, so a batch naming an action this catalog does not yet know
    still consumes its screenshot. Publication rejects that call later, at
    ``validate_canonical_rows``; deciding here would make pairing a name whitelist.
    """
    name = tool_call_name(call)
    if name not in LITE_ACTION_BATCH_TOOL_NAMES:
        return False
    _, error = validate_lite_action_batch_structure(name, tool_call_arguments(call))
    if error is not None:
        raise ValueError(error.reason)
    return True
