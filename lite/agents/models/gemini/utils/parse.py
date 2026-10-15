"""Gemini ``generateContent`` response parsing.

Owns the one pass that turns a native response into a canonical Lite assistant
message plus the provider-call provenance the loop replays.

Three things are specific to the native surface and easy to get wrong:

* ``thought`` is a **boolean flag on a text Part**, not a part type. A text part
  with ``thought: true`` is a thinking summary; ``thoughtSignature`` is a
  Part-level sibling of ``functionCall``, not a field inside it.
* ``functionCall.args`` is already a dict -- there is no JSON string to decode.
* ``intent`` and ``safety_decision`` ride *inside* ``args`` next to the real
  action parameters. They are lifted here so the action space only ever sees
  action parameters.

Error direction (easy to invert): a malformed argument bag is recorded as a
*string* ``error`` and becomes model-visible feedback; ``ModelToolCallParseError``
is what the action space **raises** and this module **catches**; a wrong object
type is a caller bug and raises a bare ``ValueError``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.agent.utils.provenance import provider_call_provenance_from_merge
from lite.core import LiteMessage
from lite.core.messages.final import mark_model_output_error
from lite.core.tools.action_space import LITE_ACTION_BATCH_TOOL_NAMES
from lite.core.tools.action_space.batches import (
    merge_adjacent_lite_action_batches_with_provenance,
)
from lite.core.tools.calls import make_tool_call, stamp_tool_call_list_ids

if TYPE_CHECKING:  # annotations only -- avoids an import cycle at runtime
    from lite.agents.models.gemini.action_space import (
        GeminiDesktopActionSpace,
        GeminiMobileActionSpace,
    )

logger = logging.getLogger(__name__)

#: Non-action keys Gemini puts inside ``functionCall.args``.
_NON_ACTION_ARG_KEYS = ("intent", "safety_decision")

#: ``finishReason`` values that mean *we* built a bad request. These must stay
#: loud: routing them to ``mark_model_output_error`` would blame the model for
#: our own replay bug.
#:
#: ``UNEXPECTED_TOOL_CALL`` is deliberately NOT here -- it means the model
#: called a tool it was not given, which is a model output error on a supported
#: configuration (``valid_actions: []`` drops the tool by design).
_REQUEST_BUG_FINISH_REASONS = frozenset({"MISSING_THOUGHT_SIGNATURE"})

#: ``finishReason`` values that are a normal completion. Everything else with no
#: parsed call is model-visible output error -- including values Google adds
#: later. Hard-raising on an unrecognised value would crash a rollout the day
#: the enum grows.
_NORMAL_FINISH_REASONS = frozenset({"STOP", "MAX_TOKENS"})

_NO_CANDIDATES_ERROR = "provider returned no candidates (prompt-level block); no action taken"


@dataclass(frozen=True)
class GeminiProviderCall:
    """One Gemini ``functionCall`` plus its canonical Lite call provenance."""

    provider_id: str
    name: str
    #: Acknowledged back to the provider in the ``functionResponse``.
    safety_decision: dict[str, Any] | None
    error: str | None
    canonical_call_id: str | None
    is_final_for_canonical: bool


@dataclass(frozen=True)
class GeminiParseResult:
    """A Gemini response parsed once, for Lite storage and provider replay."""

    message: LiteMessage
    provider_calls: tuple[GeminiProviderCall, ...]
    #: The model turn's raw ``parts``, replayed verbatim into provider history.
    #: Without this the replayed assistant message would have no content.
    replay_parts: tuple[dict[str, Any], ...]
    finish_reason: str | None


def _provider_parse_error(error: str) -> str:
    """Model-visible feedback for a call that was rejected before execution."""
    return (
        f"model output error: {error}. This functionCall was not executed. "
        "Return a valid functionCall in the required format, or plain final "
        "text if finished."
    )


def _strip_non_action_keys(
    args: dict[str, Any],
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    """Split ``args`` into (action parameters, intent, safety_decision).

    Non-destructive: ``args`` is left intact. The verbatim ``replay_parts``
    replay is what returns ``intent`` to the model, so the raw bag must keep it.
    """
    action_args = {k: v for k, v in args.items() if k not in _NON_ACTION_ARG_KEYS}
    intent = args.get("intent")
    safety_decision = args.get("safety_decision")
    return (
        action_args,
        intent if isinstance(intent, str) else None,
        safety_decision if isinstance(safety_decision, dict) else None,
    )


def _no_candidates_message() -> LiteMessage:
    """Assistant message for a response carrying no candidate at all.

    Reachable: a prompt-level safety block returns ``promptFeedback.blockReason``
    and **no** ``candidates`` key -- and therefore no ``finishReason`` to
    classify either. proto3 JSON omits empty repeated fields, so "present but
    empty" is the same observable as "absent"; one branch covers both.
    """
    message: LiteMessage = {"role": "assistant", "content": [], "tool_calls": []}
    mark_model_output_error(message, _NO_CANDIDATES_ERROR)
    return message


def _model_parts(response: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    """The model turn's parts and finishReason, or ``([], None)`` if absent."""
    if not isinstance(response, dict):
        raise ValueError(
            f"Gemini response must be a dict from resp.json(); got {type(response).__name__}"
        )
    candidates = response.get("candidates")
    if not candidates:
        return [], None
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    return list(parts), candidate.get("finishReason")


def parse_response_with_provenance(
    response: dict[str, Any],
    *,
    action_space: GeminiDesktopActionSpace | GeminiMobileActionSpace,
    admissible_verbs: frozenset[str],
    extra_tool_names: frozenset[str] = frozenset(),
    call_id_start: int = 0,
    pass_through_unknown: bool = False,
) -> GeminiParseResult:
    """Parse one ``generateContent`` response.

    ``admissible_verbs`` is the post-exclusion verb set from the action space
    (``visible_predefined_verbs``); a static declared-name set could not see the
    exclusion, so it is passed in rather than derived here.

    ``pass_through_unknown`` follows the mobile precedent: ``*@mobile`` spaces
    let a named non-native call through verbatim, ``*@desktop`` spaces reject it.
    """
    parts, finish_reason = _model_parts(response)
    if not parts and finish_reason is None:
        return GeminiParseResult(
            message=_no_candidates_message(),
            provider_calls=(),
            replay_parts=(),
            finish_reason=None,
        )

    if finish_reason in _REQUEST_BUG_FINISH_REASONS:
        raise RuntimeError(
            f"Gemini rejected the request we built: finishReason={finish_reason}. "
            "This is a request-construction bug (thought-signature replay), not "
            "model output -- see the family's history replay."
        )

    content_parts: list[dict[str, Any]] = []
    intents: list[str] = []
    unmerged: list[dict[str, Any]] = []
    source_groups: list[list[int] | None] = []
    records: list[tuple[dict[str, Any], str | None]] = []
    parse_errors: list[str] = []

    for index, part in enumerate(parts):
        if "functionCall" not in part:
            text = part.get("text")
            if text:
                # A text part flagged ``thought`` is a thinking SUMMARY, not the
                # native trace, so per the LiteMessage contract it belongs in an
                # inline_reasoning part rather than top-level reasoning_content.
                kind = "inline_reasoning" if part.get("thought") else "text"
                content_parts.append({"type": kind, "text": text})
            continue

        function_call = part["functionCall"]
        name = function_call.get("name") or ""
        raw_args = function_call.get("args")
        provider_id = function_call.get("id") or ""

        error: str | None = None
        if not isinstance(raw_args, dict):
            # Malformed argument bag is model output, not a caller bug: record
            # a string error, never raise.
            error = f"{name or '<unknown>'} arguments must be an object"
            raw_args = {}
        action_args, intent, safety_decision = _strip_non_action_keys(raw_args)
        if intent:
            intents.append(intent)
        if not provider_id and error is None:
            error = f"missing provider id for {name or '<unknown>'}"

        source_start = len(unmerged)
        if error is not None:
            pass
        elif name in LITE_ACTION_BATCH_TOOL_NAMES:
            # Canonical action-batch tool name, not a provider verb.
            error = f"unexpected canonical batch tool name {name!r}"
        elif name in extra_tool_names:
            # Env extras route to the env by name; env ingress owns admission,
            # so no schema validation here.
            unmerged.append(make_tool_call(name, action_args))
        elif name in admissible_verbs:
            try:
                unmerged.extend(
                    action_space.convert_tool_calls_from_agent(
                        [{"name": name, "arguments": action_args}]
                    )
                )
            except ModelToolCallParseError as exc:
                logger.warning("Ignoring malformed Gemini functionCall: %s", exc)
                error = str(exc)
        elif pass_through_unknown:
            unmerged.append(make_tool_call(name, action_args))
        else:
            error = f"undeclared functionCall {name or '<unknown>'}"

        if error is not None:
            parse_errors.append(error)
        records.append(
            (
                {
                    "provider_id": provider_id,
                    "name": name,
                    "safety_decision": safety_decision,
                },
                _provider_parse_error(error) if error else None,
            )
        )
        source_groups.append(list(range(source_start, len(unmerged))) or None)

    merge = merge_adjacent_lite_action_batches_with_provenance(unmerged)
    lite_tool_calls = merge.tool_calls
    stamp_tool_call_list_ids(lite_tool_calls, start=call_id_start, preserve=False)
    provenance = provider_call_provenance_from_merge(source_groups, merge)

    if intents:
        # One joined part, not N: ``get_action_description`` returns only the
        # first non-empty part, so N parts would silently drop N-1 -- and this
        # string is persisted into the RL step.
        content_parts.append({"type": "action_description", "text": "; ".join(intents)})

    message: LiteMessage = {
        "role": "assistant",
        "content": content_parts,
        "tool_calls": lite_tool_calls,
    }
    if not lite_tool_calls:
        if parse_errors:
            mark_model_output_error(message, "; ".join(parse_errors))
        elif finish_reason and finish_reason not in _NORMAL_FINISH_REASONS:
            # Content stop (SAFETY / RECITATION / MALFORMED_FUNCTION_CALL /
            # anything new) with nothing parsed: model-visible, not a crash.
            mark_model_output_error(message, f"provider stopped: {finish_reason}")

    return GeminiParseResult(
        message=message,
        provider_calls=tuple(
            GeminiProviderCall(
                error=error,
                canonical_call_id=prov.canonical_call_id,
                is_final_for_canonical=prov.is_final_for_canonical,
                **record,
            )
            for (record, error), prov in zip(records, provenance, strict=True)
        ),
        replay_parts=tuple(parts),
        finish_reason=finish_reason,
    )


__all__ = [
    "GeminiParseResult",
    "GeminiProviderCall",
    "parse_response_with_provenance",
]
