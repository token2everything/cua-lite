"""Gemini provider-history assembly.

Two histories run in parallel and diverge on purpose:

* ``contents`` -- the native list posted to ``generateContent``. Action batches
  show only their final frame here, and image truncation may drop older ones.
* ``trajectory`` -- durable Lite storage, which keeps every frame the env
  returned.

Native shapes that differ from BOTH precedents (Anthropic/OpenAI):

* Roles are ``user`` / ``model`` only. There is no ``assistant`` and no ``tool``.
* Tool results are ``functionResponse`` parts inside a ``user`` Content, with
  ``response`` a JSON **object** -- not a string. They carry the call's ``id``
  when it has one; ``name`` is what matches a call the provider sent without an
  id (see the error branch below).
* The screenshot rides *inside* ``functionResponse.parts[].inlineData``, not as
  a sibling Part. That binding is what lets N parallel calls each answer with
  their own frame.
* ``inlineData.data`` is bare base64 -- no ``data:`` URL prefix.

⚠️ Every ``functionResponse`` for one model turn goes into **one** ``user``
Content. Interleaving ``FC1, FR1, FC2, FR2`` is a documented 400 once thought
signatures are replayed; the legal order is ``FC1, FC2, FR1, FR2``.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from PIL import Image

from lite.agents.core.agent.utils.image_markers import (
    mark_provider_visible_image_index,
)
from lite.agents.core.agent.utils.tool_results import (
    append_tool_result_images,
    canonical_tool_result_messages,
    canonical_unpaired_response_feedback_messages,
    latest_step_feedback,
    tool_result_text_for_call_id,
)
from lite.core.tools.action_space import LITE_ACTION_BATCH_TOOL_NAMES
from lite.core.tools.calls import tool_call_id, tool_call_name

if TYPE_CHECKING:
    from lite.agents.models.gemini.utils.parse import GeminiProviderCall
    from lite.core.samples import LiteSample
    from lite.core.tools.results import LiteToolResult
    from lite.gym.types import LiteEnvStepResult

#: Keys inside ``functionResponse.response``. Defined once so the normal, error
#: and safety paths cannot drift apart.
RESPONSE_OUTPUT_KEY = "output"
RESPONSE_ERROR_KEY = "error"
RESPONSE_SAFETY_ACK_KEY = "safety_acknowledgement"

_REDACTED = "[base64 image omitted]"
_OMITTED = "[Image omitted]"


# =============================================================================
# Images
# =============================================================================


def _resize_png(png_bytes: bytes, target: tuple[int, int] | None) -> bytes:
    """Resize for the wire. ``target=None`` sends the env-native frame."""
    if target is None:
        return png_bytes
    image = Image.open(io.BytesIO(png_bytes))
    resized = image.resize(target)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    return buffer.getvalue()


async def image_b64(png_bytes: bytes, target: tuple[int, int] | None) -> str:
    """PNG bytes -> bare base64 (no ``data:`` prefix, unlike the precedents)."""
    resized = await asyncio.to_thread(_resize_png, png_bytes, target)
    return base64.b64encode(resized).decode()


async def inline_data_parts(
    image_pairs: Sequence[tuple[bytes, int]],
    *,
    target: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    """``inlineData`` parts carrying the local-only image-index marker.

    ⚠️ The marker is a top-level sibling key on the Part, and the native REST
    surface **rejects a Part with an unknown field**. Stripping it before the
    request is therefore correctness-critical, not hygiene.
    """
    parts: list[dict[str, Any]] = []
    for png_bytes, image_index in image_pairs:
        part = {"inlineData": {"mimeType": "image/png", "data": await image_b64(png_bytes, target)}}
        parts.append(mark_provider_visible_image_index(part, image_index))
    return parts


def _provider_visible_result_image_pairs(
    *,
    results_by_call_id: dict[str, LiteToolResult],
    result_image_indices_by_call_id: dict[str, tuple[int, ...]],
    canonical_call_id: str | None,
    is_action_batch: bool,
    is_final_provider_call: bool,
) -> tuple[tuple[bytes, int], ...]:
    """Frames from one env result the provider may see.

    Durable storage keeps every returned frame; an action batch shows only its
    final one, because the intermediate frames were never a separate model turn.
    """
    if canonical_call_id is None:
        return ()
    if is_action_batch and not is_final_provider_call:
        return ()
    result = results_by_call_id.get(canonical_call_id)
    if result is None or not result.images:
        return ()
    pairs = tuple(
        zip(result.images, result_image_indices_by_call_id.get(canonical_call_id, ()), strict=True)
    )
    return pairs[-1:] if is_action_batch else pairs


# =============================================================================
# Log redaction
# =============================================================================


def strip_images(payload: dict[str, Any]) -> dict[str, Any]:
    """A copy of the request payload with base64 replaced, for the RL-step log.

    Takes the whole payload -- ``contents`` alone would leave the tool config
    and system instruction out of the recorded prompt.
    """
    redacted = copy.deepcopy(payload)
    for content in redacted.get("contents") or []:
        for part in content.get("parts") or []:
            _redact_part(part)
    return redacted


def _redact_part(part: dict[str, Any]) -> None:
    if "inlineData" in part:
        part["inlineData"] = dict(part["inlineData"], data=_REDACTED)
    function_response = part.get("functionResponse")
    if function_response:
        for nested in function_response.get("parts") or []:
            _redact_part(nested)


# =============================================================================
# Image budget
# =============================================================================


def filter_to_n_most_recent_images(
    contents: list[dict[str, Any]],
    images_to_keep: int | None,
    *,
    min_removal_threshold: int = 1,
) -> list[dict[str, Any]]:
    """A NEW contents list keeping only the N most recent images.

    ⚠️ Assign the result to the outgoing payload, never back onto the persistent
    ``contents`` -- writing it back destroys frames permanently, and the RL
    step's ``image_indices`` would then claim frames the request never carried.
    Run this BEFORE harvesting markers, for the same reason.
    """
    if images_to_keep is None:
        return contents
    total = sum(_count_images(content) for content in contents)
    to_remove = total - images_to_keep
    if min_removal_threshold > 0:
        to_remove -= to_remove % min_removal_threshold
    if to_remove <= 0:
        return contents

    out = copy.deepcopy(contents)
    removed = 0
    for content in out:
        for part in content.get("parts") or []:
            removed = _omit_images(part, removed, to_remove)
    return out


def _count_images(content: dict[str, Any]) -> int:
    count = 0
    for part in content.get("parts") or []:
        count += _count_part_images(part)
    return count


def _count_part_images(part: dict[str, Any]) -> int:
    count = 1 if "inlineData" in part else 0
    function_response = part.get("functionResponse")
    if function_response:
        for nested in function_response.get("parts") or []:
            count += _count_part_images(nested)
    return count


def _omit_images(part: dict[str, Any], removed: int, to_remove: int) -> int:
    if "inlineData" in part and removed < to_remove:
        # Replace the part wholesale rather than popping ``inlineData`` alone.
        # A dropped frame must also lose its provider-visible image-index
        # marker: the harvest that follows truncation reads the marker, not the
        # payload, so a marker left on a now-text-only part would make
        # ``LiteRLStep.image_indices`` claim a frame the request never carried.
        part.clear()
        part["text"] = _OMITTED
        removed += 1
    function_response = part.get("functionResponse")
    if function_response:
        kept: list[dict[str, Any]] = []
        for nested in function_response.get("parts") or []:
            if "inlineData" in nested and removed < to_remove:
                removed += 1
                continue
            kept.append(nested)
        if kept:
            function_response["parts"] = kept
        else:
            function_response.pop("parts", None)
    return removed


# =============================================================================
# Assistant replay
# =============================================================================


def append_provider_assistant_message(
    contents: list[dict[str, Any]],
    *,
    replay_parts: Sequence[dict[str, Any]],
) -> None:
    """Replay the model turn from the SAME parse pass -- never re-read the response.

    The parts are replayed verbatim, which is also what carries
    ``thoughtSignature`` and ``intent`` back to the provider. Gemini validates
    the signature on the first ``functionCall`` part of each step in the current
    turn, so rebuilding the parts field-by-field risks dropping it.
    """
    if not replay_parts:
        return
    contents.append({"role": "model", "parts": [copy.deepcopy(p) for p in replay_parts]})


# =============================================================================
# Tool-result feedback
# =============================================================================


async def append_canonical_step_feedback(
    trajectory: LiteSample,
    step_result: LiteEnvStepResult,
    lite_tool_calls: list[dict[str, Any]],
):
    """Write the canonical (durable) side of one env step.

    Delegates to the core owner rather than re-implementing it: ``gpt`` already
    calls ``canonical_tool_result_messages`` this way, so a third copy would be
    the divergence, not the reuse.
    """
    stored = await append_tool_result_images(trajectory, step_result.results)
    if lite_tool_calls:
        feedback_messages = canonical_tool_result_messages(
            step_result.results,
            lite_tool_calls,
            result_image_indices_by_call_id=stored.by_call_id,
        )
    else:
        feedback_messages = canonical_unpaired_response_feedback_messages(
            step_result.results,
            result_image_indices_by_position=stored.by_position,
        )
    trajectory.messages.extend(feedback_messages)
    return stored


async def append_gemini_provider_feedback(
    *,
    contents: list[dict[str, Any]],
    trajectory: LiteSample,
    step_result: LiteEnvStepResult,
    lite_tool_calls: list[dict[str, Any]],
    provider_calls: Sequence[GeminiProviderCall],
    resize_target: tuple[int, int] | None,
) -> tuple[bytes | None, int | None]:
    """Write both histories for one env step; return the latest frame for reuse.

    ONE ``user`` Content carrying every ``functionResponse`` -- see the module
    docstring on FC/FR ordering.

    The set of calls answered here MUST match the set replayed by
    ``append_provider_assistant_message``: a ``functionCall`` with no matching
    ``functionResponse`` makes the next request malformed.
    """
    stored = await append_canonical_step_feedback(trajectory, step_result, lite_tool_calls)

    if not provider_calls:
        image_bytes, text, _ = latest_step_feedback(step_result)
        latest_index: int | None = None
        parts: list[dict[str, Any]] = []
        if text is not None:
            parts.append({"text": text})
        if image_bytes:
            latest_index = next(
                (indices[-1] for indices in reversed(stored.by_position) if indices),
                None,
            )
            image_part = {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": await image_b64(image_bytes, resize_target),
                }
            }
            parts.append(mark_provider_visible_image_index(image_part, latest_index))
        if parts:
            contents.append({"role": "user", "parts": parts})
        return image_bytes, latest_index

    results_by_call_id = {
        result.tool_call_id: result for result in step_result.results if result.tool_call_id
    }
    batch_call_ids = {
        tool_call_id(call)
        for call in lite_tool_calls
        if tool_call_name(call) in LITE_ACTION_BATCH_TOOL_NAMES
    }

    parts: list[dict[str, Any]] = []
    latest_bytes: bytes | None = None
    latest_index: int | None = None

    for call in provider_calls:
        if call.error is not None:
            # Answered even when ``provider_id`` is empty. The replay above is
            # verbatim over every part, so skipping one here would leave a
            # ``functionCall`` with no ``functionResponse`` and malform the next
            # request. A call with no id always lands here -- the parser assigns
            # it an error precisely because it has none -- and Gemini matches
            # such a response by ``name``, so the rejection stays model-visible
            # instead of vanishing.
            response_body: dict[str, Any] = {"name": call.name}
            if call.provider_id:
                response_body["id"] = call.provider_id
            response_body["response"] = {RESPONSE_ERROR_KEY: call.error}
            parts.append({"functionResponse": response_body})
            continue

        text = tool_result_text_for_call_id(step_result, call_id=call.canonical_call_id)
        response: dict[str, Any] = {RESPONSE_OUTPUT_KEY: text if text is not None else "ok"}
        if call.safety_decision is not None:
            # Model-visible by construction -- no side channel. A rollout is
            # unattended, so there is nobody to prompt: the acknowledgement
            # rides in the same functionResponse the model reads, and the
            # decision itself is already on the durable trajectory.
            response[RESPONSE_SAFETY_ACK_KEY] = True

        image_pairs = _provider_visible_result_image_pairs(
            results_by_call_id=results_by_call_id,
            result_image_indices_by_call_id=stored.by_call_id,
            canonical_call_id=call.canonical_call_id,
            is_action_batch=call.canonical_call_id in batch_call_ids,
            is_final_provider_call=call.is_final_for_canonical,
        )
        function_response: dict[str, Any] = {
            "id": call.provider_id,
            "name": call.name,
            "response": response,
        }
        if image_pairs:
            function_response["parts"] = await inline_data_parts(image_pairs, target=resize_target)
            latest_bytes, latest_index = image_pairs[-1]
        parts.append({"functionResponse": function_response})

    if parts:
        contents.append({"role": "user", "parts": parts})
    return latest_bytes, latest_index


__all__ = [
    "RESPONSE_ERROR_KEY",
    "append_canonical_step_feedback",
    "RESPONSE_OUTPUT_KEY",
    "RESPONSE_SAFETY_ACK_KEY",
    "append_gemini_provider_feedback",
    "append_provider_assistant_message",
    "filter_to_n_most_recent_images",
    "image_b64",
    "inline_data_parts",
    "strip_images",
]
