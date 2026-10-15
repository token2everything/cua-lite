"""Own reversible native GUI-message projection into image-text V3.

Every projected message carries original message boundaries in ``meta``.
Tool observations use user-role image/text parts because V3's tool content
is text-only. Calls, feedback IDs, content order and original fields remain
recoverable; the audit binds projected content to the original source.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from PIL import Image


class UnrepresentableNativeContent(ValueError):
    """A real native content type has no equivalent in the image-text V3 schema."""


def visible_content(original: dict, aliases: dict) -> str | list:
    """Map source image locations while preserving text and image order."""
    content = original["content"]
    if isinstance(content, str) or content is None:
        return content
    result = []
    for part in content:
        if part["type"] == "text":
            result.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image":
            reference = aliases[part["image"]["url"]]
            result.append(
                {
                    "type": "image",
                    "image": {
                        "url": reference["url"],
                        "image_wh": reference["image_wh"],
                    },
                }
            )
        else:
            raise UnrepresentableNativeContent(
                f"Native viewer content must be text/image: {part['type']}"
            )
    return result


def project_v3(record, params) -> dict:
    """Convert an admitted InternFlow record through the native V3 projection."""
    del params
    return project_native_v3(record.model_dump(mode="python", exclude_none=True))


def project_native_v3(source: dict) -> dict:
    """Preserve a collected native row, including terminal observations, in V3."""
    data = copy.deepcopy(source)
    aliases = data["metadata"]["image_aliases"]
    native = data["metadata"]["native_messages"]
    masks = data["metadata"]["internflow"]["use_loss_seq"]
    output = []
    observation_parts = []
    observation_sources = []
    assistant_index = 0
    for message in data["messages"]:
        boundary = message["meta"]["native_boundary"]
        source_index = boundary["source_index"]
        original = native[source_index]
        if message["role"] in {"tool", "user"}:
            content = visible_content(original, aliases)
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
            start = len(observation_parts)
            observation_parts.extend(parts)
            observation_sources.append(
                {
                    "source_index": source_index,
                    "source_message": original,
                    "content_start": start,
                    "content_end": len(observation_parts),
                    "canonical_tool_name": message.get("name"),
                }
            )
            continue
        if observation_sources:
            output.append(
                {
                    "role": "user",
                    "content": observation_parts,
                    "meta": {"native_projection": observation_sources},
                }
            )
            observation_parts = []
            observation_sources = []
        content = visible_content(original, aliases)
        if isinstance(content, list):
            if any(part["type"] != "text" for part in content):
                raise UnrepresentableNativeContent(
                    "V3 assistant/system image content is not representable"
                )
            content = "".join(part["text"] for part in content)
        projected = {
            "role": message["role"],
            "content": content,
            "meta": {
                "native_projection": [
                    {
                        "source_index": source_index,
                        "source_message": original,
                    }
                ]
            },
        }
        for name in ("name", "reasoning_content", "tool_calls"):
            if name in original:
                projected[name] = copy.deepcopy(original[name])
        if message["role"] == "assistant":
            projected["loss"] = masks[assistant_index]
            assistant_index += 1
            for call in projected.get("tool_calls", []):
                if isinstance(call["function"]["arguments"], str):
                    call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        output.append(projected)
    if observation_sources:
        output.append(
            {
                "role": "user",
                "content": observation_parts,
                "meta": {"native_projection": observation_sources},
            }
        )
    data["messages"] = output
    data["model_name"] = data["metadata"]["source_model_id"]
    data["metadata"].pop("native_messages")
    data["metadata"]["format_projection"] = {
        "version": "v2.1.0",
        "source": "JsonlTrajectoryLogger.native_viewer",
        "target": "multimodal_image_text_dialogs_v3",
        "tool_feedback": "reversible_user_observation_blocks",
        "sample_boundary": "one_complete_source_trajectory",
        "reasoning": "original_public_fields_only",
    }
    return data


def audit_roundtrip(projected: dict, native: dict, media_root: Path) -> dict:
    """Verify visible content and actions, then restore every native message."""
    if projected["model_name"] != projected["metadata"]["source_model_id"]:
        raise ValueError("V3 model_name differs from the real source model identity")
    if projected["metadata"]["scope"] == "v03_formal":
        actual_model_id = native["metadata"]["model_id"]
        if (
            projected["model_name"] != actual_model_id
            or projected["metadata"]["source_api_model_id"] != actual_model_id
        ):
            raise ValueError("Formal V3 teacher identity differs from the preserved native source")
    aliases = projected["metadata"]["image_aliases"]
    restored = []
    source_index = 0
    actual_image_sequence = []
    expected_image_sequence = []
    assistant_count = 0
    tool_responses = 0
    for message in projected["messages"]:
        boundaries = message["meta"]["native_projection"]
        expected_parts = []
        for boundary in boundaries:
            if boundary["source_index"] != source_index:
                raise ValueError("Projection reordered, duplicated or omitted a source message")
            source = native["messages"][source_index]
            if boundary["source_message"] != source:
                raise ValueError(f"Original boundary changed at source message {source_index}")
            restored.append(copy.deepcopy(boundary["source_message"]))
            expected = visible_content(source, aliases)
            if message["role"] == "user":
                if source["role"] not in {"tool", "user"}:
                    raise ValueError("Observation block absorbed an assistant or system")
                parts = (
                    [{"type": "text", "text": expected}] if isinstance(expected, str) else expected
                )
                if boundary["content_start"] != len(expected_parts):
                    raise ValueError("Source observation start boundary differs")
                expected_parts.extend(parts)
                if boundary["content_end"] != len(expected_parts):
                    raise ValueError("Source observation end boundary differs")
                tool_responses += source["role"] == "tool"
            else:
                if len(boundaries) != 1 or message["role"] != source["role"]:
                    raise ValueError("Decision/system boundary changed")
                expected_text = (
                    "".join(part["text"] for part in expected)
                    if isinstance(expected, list)
                    else expected
                )
                if message.get("content") != expected_text:
                    raise ValueError(f"Source text changed at message {source_index}")
                if source["role"] == "assistant":
                    assistant_count += 1
                    expected_calls = copy.deepcopy(source.get("tool_calls", []))
                    for call in expected_calls:
                        if isinstance(call["function"]["arguments"], str):
                            call["function"]["arguments"] = json.loads(
                                call["function"]["arguments"]
                            )
                    if message.get("tool_calls", []) != expected_calls:
                        raise ValueError(f"Assistant actions changed at message {source_index}")
                    if message.get("reasoning_content") != source.get("reasoning_content"):
                        raise ValueError(f"Reasoning changed at message {source_index}")
            if isinstance(expected, list):
                expected_image_sequence.extend(
                    part["image"]["url"] for part in expected if part["type"] == "image"
                )
            source_index += 1
        if message["role"] == "user" and message["content"] != expected_parts:
            raise ValueError("Projected observation content differs from actual source")
        if isinstance(message.get("content"), list):
            actual_image_sequence.extend(
                part["image"]["url"] for part in message["content"] if part["type"] == "image"
            )
    if restored != native["messages"]:
        raise ValueError("Round-trip native messages differ")
    if actual_image_sequence != expected_image_sequence:
        raise ValueError("Source image order differs after projection")
    for reference in aliases.values():
        path = (media_root / reference["url"]).resolve()
        path.relative_to(media_root.resolve())
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError(f"Image bytes changed: {reference['url']}")
        with Image.open(path) as image:
            image.load()
            if list(image.size) != reference["image_wh"]:
                raise ValueError(f"Image dimensions changed: {reference['url']}")
    for reference in projected["metadata"]["source_media_archive"].values():
        path = (media_root / reference["url"]).resolve()
        path.relative_to(media_root.resolve())
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError(f"Archived raw screenshot bytes changed: {reference['url']}")
    # Tool pairs are verified through original boundaries rather than disappearing
    # when their role changes at this explicit multimodal format boundary.
    calls = {}
    for index, message in enumerate(restored):
        for call in message.get("tool_calls", []):
            if call["id"] in calls:
                raise ValueError("Duplicate source tool call ID")
            calls[call["id"]] = (index, call["function"]["name"])
    matched = set()
    for index, message in enumerate(restored):
        if message["role"] != "tool":
            continue
        call_id = message["tool_call_id"]
        if call_id not in calls or calls[call_id][0] >= index or call_id in matched:
            raise ValueError("Original tool-response pairing is invalid")
        if "name" in message and message["name"] != calls[call_id][1]:
            raise ValueError("Original response name differs from its tool call")
        matched.add(call_id)
    return {
        "passed": True,
        "source_messages": len(restored),
        "assistant_decisions": assistant_count,
        "tool_calls": len(calls),
        "paired_tool_responses": tool_responses,
        "image_occurrences": len(actual_image_sequence),
        "messages_equal": True,
        "actions_equal": True,
        "image_bytes_equal": True,
        "image_order_equal": True,
        "archived_raw_images_equal": True,
        "source_model_identity_equal": True,
        "terminal_calls_without_feedback": sorted(set(calls) - matched),
    }
