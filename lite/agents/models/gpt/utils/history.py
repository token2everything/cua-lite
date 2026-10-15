"""Responses-API history helpers for GPT agents.

Owns GPT provider-history assembly: the per-turn feedback items appended before
the next Responses call, the client-managed history compaction used when
``chain_previous_response=False``, and the base64 redaction applied before
diagnostic logging.
"""

from __future__ import annotations

import asyncio
from typing import Any

from lite.agents.core.agent.utils.image_markers import mark_provider_visible_image_index
from lite.agents.core.agent.utils.tool_results import (
    append_tool_result_images,
    canonical_tool_result_messages,
    canonical_unpaired_response_feedback_messages,
    latest_step_feedback,
    tool_result_text_for_call_id,
)
from lite.agents.models.gpt.utils.image_io import _IMAGE_BLOCK_TYPES, _resize_for_api
from lite.agents.models.gpt.utils.parse import GPTProviderCall
from lite.core import LiteSample
from lite.core.messages.image_refs import referenced_image_indices_in_message_order
from lite.core.tools.action_space import LITE_ACTION_BATCH_TOOL_NAMES
from lite.core.tools.calls import tool_call_id, tool_call_name
from lite.gym.types import LiteEnvStepResult


def _strip_images(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip base64 image data from input items for logging."""

    def _strip_block(block: Any) -> Any:
        if isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES:
            return {"type": block["type"], "image_url": "[base64 image omitted]"}
        return block

    stripped = []
    for item in items:
        new_item = dict(item)
        for key in ("content", "output"):
            val = new_item.get(key)
            if isinstance(val, list):
                new_item[key] = [_strip_block(b) for b in val]
            elif isinstance(val, dict):
                new_item[key] = _strip_block(val)
        stripped.append(new_item)
    return stripped


def _format_computer_action(call: dict[str, Any]) -> str:
    """Render a computer_call action summary for compacted history."""
    actions = call.get("actions")
    if not isinstance(actions, list):
        single = call.get("action")
        actions = [single] if isinstance(single, dict) else []
    parts: list[str] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        t = a.get("type", "action")
        if t in ("click", "double_click", "move", "mouse_move") and "x" in a:
            btn = a.get("button")
            parts.append(f"{t}({a['x']},{a['y']}" + (f",{btn}" if btn else "") + ")")
        elif t == "type":
            parts.append(f"type({(a.get('text') or '')!r})")
        elif t == "keypress":
            # API sets keys=null on non-keypress action dicts; keep list
            # spelling so literal '+' is not confused with a chord delimiter.
            parts.append(f"keypress({a.get('keys') or []!r})")
        elif t == "scroll":
            parts.append(f"scroll({a.get('scroll_x', 0)},{a.get('scroll_y', 0)})")
        else:
            parts.append(t)
    return "; ".join(parts) if parts else "(action)"


def _compress_computer_use_history(
    items: list[dict[str, Any]], full_history_size: int | None
) -> list[dict[str, Any]]:
    """Compact native computer-use history when ``chain_previous_response=False``.

    Keeps the last ``full_history_size`` call/output/screenshot steps intact.
    Earlier steps become a text summary in the leading user message, whose image
    blocks are removed. Provider call/output pairs are removed together so each
    retained output keeps its matching call. ``0`` is invalid because at least one
    full pair must remain.
    """
    if full_history_size is None or full_history_size < 0:
        return items
    if full_history_size == 0:
        raise ValueError("full_history_size must be a positive integer or None")
    call_idxs = [
        i
        for i, it in enumerate(items)
        if isinstance(it, dict) and it.get("type") == "computer_call"
    ]
    if len(call_idxs) <= full_history_size:
        return items

    # Leading developer/system/user messages form the head we keep while the
    # summary is injected into the first user message.
    head_end = 0
    while head_end < len(items) and items[head_end].get("role") in {
        "developer",
        "system",
        "user",
    }:
        head_end += 1

    # Window = last full_history_size steps. Start at that step's computer_call,
    # extended back over its own reasoning items so they aren't orphaned.
    window_start = call_idxs[len(call_idxs) - full_history_size]
    while window_start > head_end and items[window_start - 1].get("type") == "reasoning":
        window_start -= 1

    head, dropped, kept = items[:head_end], items[head_end:window_start], items[window_start:]

    lines: list[str] = []
    step = 1
    for it in dropped:
        if isinstance(it, dict) and it.get("type") == "computer_call":
            lines.append(f"Step {step}: {_format_computer_action(it)}")
            step += 1
    summary = "Previous actions:\n" + ("\n".join(lines) if lines else "None")

    new_head: list[dict[str, Any]] = []
    injected = False
    for it in head:
        if it.get("role") == "user" and isinstance(it.get("content"), list):
            content = [
                b
                for b in it["content"]
                if not (isinstance(b, dict) and b.get("type") in _IMAGE_BLOCK_TYPES)
            ]
            if not injected:
                content.append({"type": "input_text", "text": summary})
                injected = True
            new_item = dict(it)
            new_item["content"] = content
            new_head.append(new_item)
        else:
            new_head.append(it)
    if not injected:
        new_head.append({"role": "user", "content": [{"type": "input_text", "text": summary}]})

    return new_head + kept


def _append_lite_feedback_messages(
    trajectory: LiteSample,
    step_result: LiteEnvStepResult,
    lite_tool_calls: list[dict[str, Any]],
    *,
    result_image_indices_by_call_id: dict[str, tuple[int, ...]],
    result_image_indices_by_position: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    provider_visible_image_indices: tuple[int, ...]
    if lite_tool_calls:
        # ``step_result.results`` is already aligned to ``lite_tool_calls`` by
        # the shared loop's env-result boundary.
        tool_result_messages = canonical_tool_result_messages(
            step_result.results,
            lite_tool_calls,
            result_image_indices_by_call_id=result_image_indices_by_call_id,
        )
        provider_visible_image_indices = referenced_image_indices_in_message_order(
            tool_result_messages
        )
    else:
        # Content-only continuations have no assistant call to pair against, so
        # env feedback stays an unpaired role:user observation.
        tool_result_messages = canonical_unpaired_response_feedback_messages(
            step_result.results,
            result_image_indices_by_position=result_image_indices_by_position,
        )
        provider_visible_image_indices = tuple(
            indices[-1]
            for indices in result_image_indices_by_position
            if indices
        )
    trajectory.messages.extend(tool_result_messages)
    return provider_visible_image_indices


async def _append_unpaired_responses_feedback(
    *,
    input_items: list[dict[str, Any]],
    step_result: LiteEnvStepResult,
    result_image_indices_by_position: tuple[tuple[int, ...], ...],
    resolution: tuple[int, int] | None,
    detail: str,
) -> tuple[int, ...]:
    image_bytes, text, _ = latest_step_feedback(step_result)
    content: list[dict[str, Any]] = []
    if text is not None:
        content.append({"type": "input_text", "text": text})
    image_index: int | None = None
    if image_bytes:
        image_index = next(
            (
                indices[-1]
                for indices in reversed(result_image_indices_by_position)
                if indices
            ),
            None,
        )
        image_b64, _, _ = await asyncio.to_thread(
            _resize_for_api,
            image_bytes,
            resolution,
        )
        content.append(
            mark_provider_visible_image_index(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_b64}",
                    "detail": detail,
                },
                image_index,
            )
        )
    if content:
        input_items.append({"role": "user", "content": content})
    return (image_index,) if image_index is not None else ()


def _provider_parse_error_item(error: str) -> dict[str, Any]:
    """A user input item carrying a provider-call parse error.

    ``computer_call_output`` has no text form and an unidentified provider call
    has no output item at all, so parse errors reach the model as their own
    user item instead of being folded into a call acknowledgement.
    """
    return {"role": "user", "content": [{"type": "input_text", "text": error}]}


async def _input_image_blocks_for_call(
    step_result: LiteEnvStepResult,
    *,
    canonical_call_id: str | None,
    lite_action_batch_call_ids: set[str],
    result_image_indices_by_call_id: dict[str, tuple[int, ...]],
    resolution: tuple[int, int] | None,
    detail: str,
) -> list[dict[str, Any]]:
    if canonical_call_id is None:
        return []
    result = next(
        (result for result in step_result.results if result.tool_call_id == canonical_call_id),
        None,
    )
    if result is None or not result.images:
        return []

    image_indices = result_image_indices_by_call_id.get(canonical_call_id, ())
    image_pairs = list(zip(result.images, image_indices, strict=True))
    if canonical_call_id in lite_action_batch_call_ids:
        # Provider-visible result-image policy: action-batch results store every
        # env frame, but the next provider turn sees only the final frame.
        image_pairs = image_pairs[-1:]

    blocks: list[dict[str, Any]] = []
    for image_bytes, image_index in image_pairs:
        image_b64, _, _ = await asyncio.to_thread(
            _resize_for_api,
            image_bytes,
            resolution,
        )
        blocks.append(
            mark_provider_visible_image_index(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_b64}",
                    "detail": detail,
                },
                image_index,
            )
        )
    return blocks


async def _append_desktop_provider_feedback(
    *,
    input_items: list[dict[str, Any]],
    output_items: list[dict[str, Any]],
    parsed_provider_calls: tuple[GPTProviderCall, ...],
    trajectory: LiteSample,
    step_result: LiteEnvStepResult,
    lite_tool_calls: list[dict[str, Any]],
    resolution: tuple[int, int] | None,
    sent_image_b64: str | None,
    sent_image_index: int | None,
    detail: str,
    use_chaining: bool,
    model_output_error: str | None,
) -> tuple[bytes | None, tuple[int, ...]]:
    """Append desktop provider feedback for the next Responses turn.

    Mutates ``input_items`` and ``trajectory.messages``. Returns the next
    observation image bytes and provider-visible Lite image indices for the
    outer sample loop.
    """
    image_bytes, _, _ = latest_step_feedback(step_result)
    stored_result_images = await append_tool_result_images(
        trajectory,
        step_result.results,
    )
    lite_action_batch_call_ids = {
        call_id
        for tool_call in lite_tool_calls
        if (call_id := tool_call_id(tool_call))
        and tool_call_name(tool_call) in LITE_ACTION_BATCH_TOOL_NAMES
    }
    results_by_call_id = {
        result.tool_call_id: result
        for result in step_result.results
        if result.tool_call_id is not None
    }
    if not use_chaining:
        # Client-managed history replays the provider's own output verbatim,
        # except calls the provider left unidentified: without a ``call_id``
        # they can never be paired with an output item.
        unidentified_items = {
            id(provider_call.item)
            for provider_call in parsed_provider_calls
            if not provider_call.provider_call_id
        }
        input_items.extend(it for it in output_items if id(it) not in unidentified_items)

    sent_computer_text_call_ids: set[str] = set()
    for provider_call in parsed_provider_calls:
        item = provider_call.item
        if not provider_call.provider_call_id:
            # The parser records a parse error for every unidentified call, and
            # the call was not replayed above, so the error is its only feedback.
            input_items.append(_provider_parse_error_item(provider_call.error))
            continue

        if provider_call.item_type == "computer_call":
            provider_call_id = provider_call.provider_call_id
            canonical_call_id = provider_call.canonical_call_id
            result = results_by_call_id.get(canonical_call_id)
            result_image_indices = stored_result_images.by_call_id.get(canonical_call_id, ())
            result_image_pairs = (
                tuple(zip(result.images, result_image_indices, strict=True))
                if result is not None and result.images
                else ()
            )
            if not provider_call.is_final_for_canonical:
                output_image_b64 = sent_image_b64
                output_image_index = sent_image_index
            elif result_image_pairs:
                # Computer-call feedback follows the same provider-visible
                # policy as action-batches: keep every env frame in Lite storage,
                # but acknowledge the provider with the final frame from this
                # computer result, never a sibling tool's image.
                output_image_bytes, output_image_index = result_image_pairs[-1]
                output_image_b64, _, _ = await asyncio.to_thread(
                    _resize_for_api,
                    output_image_bytes,
                    resolution,
                )
            else:
                output_image_b64 = sent_image_b64
                output_image_index = sent_image_index
            output_item: dict[str, Any] = {
                "type": "computer_call_output",
                "call_id": provider_call_id,
                "output": {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{output_image_b64}",
                    "detail": detail,
                },
            }
            output_item["output"] = mark_provider_visible_image_index(
                output_item["output"],
                output_image_index,
            )
            pending_checks = item.get("pending_safety_checks")
            if pending_checks:
                output_item["acknowledged_safety_checks"] = pending_checks
            input_items.append(output_item)

            if provider_call.error is not None:
                # Parse failed, so nothing ran and the acknowledged frame is the
                # one the model already saw. The pending computer_call still
                # needs its output item, but ``computer_call_output`` has no
                # text form, so the error reaches the model as its own item.
                input_items.append(_provider_parse_error_item(provider_call.error))
                continue

            if (
                canonical_call_id is not None
                and provider_call.is_final_for_canonical
                and canonical_call_id not in sent_computer_text_call_ids
            ):
                per_call_text = tool_result_text_for_call_id(
                    step_result,
                    call_id=canonical_call_id,
                    default=None,
                )
                if per_call_text is not None:
                    input_items.append(
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": per_call_text}],
                        }
                    )
                sent_computer_text_call_ids.add(canonical_call_id)

        elif provider_call.item_type == "function_call":
            provider_call_id = provider_call.provider_call_id
            canonical_call_id = provider_call.canonical_call_id
            is_action_batch = canonical_call_id in lite_action_batch_call_ids
            if is_action_batch and not provider_call.is_final_for_canonical:
                image_blocks = []
            else:
                image_blocks = await _input_image_blocks_for_call(
                    step_result,
                    canonical_call_id=canonical_call_id,
                    lite_action_batch_call_ids=lite_action_batch_call_ids,
                    result_image_indices_by_call_id=stored_result_images.by_call_id,
                    resolution=resolution,
                    detail=detail,
                )
            if provider_call.error is not None:
                tool_result = provider_call.error
            else:
                tool_result = tool_result_text_for_call_id(
                    step_result,
                    call_id=canonical_call_id,
                    default=(
                        (
                            f"model output error: {model_output_error}. "
                            "No action was executed. Return a valid tool_call in "
                            "the required format, or plain final text if finished."
                        )
                        if model_output_error and canonical_call_id is None
                        else "ok"
                    ),
                )
            output: str | list[dict[str, Any]]
            if image_blocks:
                output = [{"type": "input_text", "text": str(tool_result)}, *image_blocks]
            else:
                output = str(tool_result)
            input_items.append(
                {"type": "function_call_output", "call_id": provider_call_id, "output": output}
            )

    unpaired_provider_visible_image_indices: tuple[int, ...] = ()
    if not parsed_provider_calls:
        unpaired_provider_visible_image_indices = await _append_unpaired_responses_feedback(
            input_items=input_items,
            step_result=step_result,
            result_image_indices_by_position=stored_result_images.by_position,
            resolution=resolution,
            detail=detail,
        )

    provider_visible_image_indices = _append_lite_feedback_messages(
        trajectory,
        step_result,
        lite_tool_calls,
        result_image_indices_by_call_id=stored_result_images.by_call_id,
        result_image_indices_by_position=stored_result_images.by_position,
    )
    if not parsed_provider_calls:
        provider_visible_image_indices = unpaired_provider_visible_image_indices
    return image_bytes, provider_visible_image_indices


async def _append_mobile_provider_feedback(
    *,
    input_items: list[dict[str, Any]],
    output_items: list[dict[str, Any]],
    parsed_provider_calls: tuple[GPTProviderCall, ...],
    trajectory: LiteSample,
    step_result: LiteEnvStepResult,
    lite_tool_calls: list[dict[str, Any]],
    resolution: tuple[int, int] | None,
    detail: str,
    use_chaining: bool,
) -> tuple[bytes, tuple[int, ...]]:
    """Append mobile provider feedback for the next Responses turn.

    Mutates ``input_items`` and ``trajectory.messages``. Returns the next
    observation image bytes and provider-visible Lite image indices for the
    outer sample loop.
    """
    image_bytes, next_text, _ = latest_step_feedback(step_result)
    if not image_bytes:
        raise RuntimeError("No image returned after env.step()")

    stored_result_images = await append_tool_result_images(
        trajectory,
        step_result.results,
    )
    lite_action_batch_call_ids = {
        call_id
        for tool_call in lite_tool_calls
        if (call_id := tool_call_id(tool_call))
        and tool_call_name(tool_call) in LITE_ACTION_BATCH_TOOL_NAMES
    }
    if not use_chaining:
        # Client-managed history replays the provider's own output verbatim,
        # except calls the provider left unidentified: without a ``call_id``
        # they can never be paired with an output item.
        unidentified_items = {
            id(provider_call.item)
            for provider_call in parsed_provider_calls
            if not provider_call.provider_call_id
        }
        input_items.extend(it for it in output_items if id(it) not in unidentified_items)

    for provider_call in parsed_provider_calls:
        if provider_call.item_type != "function_call":
            continue
        provider_call_id = provider_call.provider_call_id
        canonical_call_id = provider_call.canonical_call_id
        if not provider_call_id:
            # The parser records a parse error for every unidentified call, and
            # the call was not replayed above, so the error is its only feedback.
            input_items.append(_provider_parse_error_item(provider_call.error))
            continue
        if provider_call.error is not None:
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": provider_call_id,
                    "output": [{"type": "input_text", "text": provider_call.error}],
                }
            )
            continue

        result_text = tool_result_text_for_call_id(
            step_result,
            call_id=canonical_call_id,
            default=None,
        )
        output_blocks = [
            {
                "type": "input_text",
                "text": result_text if result_text is not None else (next_text or "ok"),
            }
        ]
        is_action_batch = canonical_call_id in lite_action_batch_call_ids
        if is_action_batch and not provider_call.is_final_for_canonical:
            image_blocks = []
        else:
            image_blocks = await _input_image_blocks_for_call(
                step_result,
                canonical_call_id=canonical_call_id,
                lite_action_batch_call_ids=lite_action_batch_call_ids,
                result_image_indices_by_call_id=stored_result_images.by_call_id,
                resolution=resolution,
                detail=detail,
            )
        if image_blocks:
            output_blocks.extend(image_blocks)
        input_items.append(
            {
                "type": "function_call_output",
                "call_id": provider_call_id,
                "output": output_blocks,
            }
        )

    unpaired_provider_visible_image_indices: tuple[int, ...] = ()
    if not parsed_provider_calls:
        unpaired_provider_visible_image_indices = await _append_unpaired_responses_feedback(
            input_items=input_items,
            step_result=step_result,
            result_image_indices_by_position=stored_result_images.by_position,
            resolution=resolution,
            detail=detail,
        )

    provider_visible_image_indices = _append_lite_feedback_messages(
        trajectory,
        step_result,
        lite_tool_calls,
        result_image_indices_by_call_id=stored_result_images.by_call_id,
        result_image_indices_by_position=stored_result_images.by_position,
    )
    if not parsed_provider_calls:
        provider_visible_image_indices = unpaired_provider_visible_image_indices
    return image_bytes, provider_visible_image_indices
