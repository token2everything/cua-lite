from __future__ import annotations

import copy
import hashlib

import pytest
from clean_v3_projection import audit_roundtrip, project_native_v3
from PIL import Image


@pytest.mark.parametrize("final_assistant", [False, True])
def test_review_projection_preserves_observations_without_inventing_a_decision(
    tmp_path, final_assistant,
):
    image = tmp_path / "screen.png"
    Image.new("RGB", (12, 8), "blue").save(image)
    native_url = "/outputs/task-0001/images/screen.png"
    native = {
        "metadata": {"model_id": "qwen3.8-max"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Find the place."}]},
            {
                "role": "assistant",
                "content": "Inspect the page.",
                "reasoning_content": "The current map is still visible.",
                "tool_calls": [{
                    "id": "real-call-1", "type": "function",
                    "function": {"name": "computer", "arguments": '{"action":"screenshot"}'},
                }],
            },
            {
                "role": "tool", "name": "computer", "tool_call_id": "real-call-1",
                "content": [
                    {"type": "text", "text": "Actual screenshot."},
                    {"type": "image", "image": {"url": native_url, "image_wh": [12, 8]}},
                ],
            },
        ],
    }
    if final_assistant:
        native["messages"].append({"role": "assistant", "content": "Unable to finish."})
    source = {
        "messages": [
            {
                "role": message["role"],
                "content": message["content"],
                **({"name": message["name"]} if "name" in message else {}),
                "meta": {"native_boundary": {"source_index": index}},
            }
            for index, message in enumerate(native["messages"])
        ],
        "metadata": {
            "scope": "manual_failure_review",
            "source_model_id": "qwen3.8-max",
            "native_messages": native["messages"],
            "internflow": {"use_loss_seq": [False] * (1 + final_assistant)},
            "image_aliases": {native_url: {
                "url": image.name,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "image_wh": [12, 8],
            }},
            "source_media_archive": {},
        },
    }
    untouched = copy.deepcopy(source)
    row = project_native_v3(source)
    receipt = audit_roundtrip(row, native, tmp_path)

    assert source == untouched
    assert receipt["messages_equal"] and receipt["image_bytes_equal"]
    assert receipt["assistant_decisions"] == 1 + final_assistant
    assert receipt["paired_tool_responses"] == 1
    assert [message["role"] for message in row["messages"]] == (
        ["user", "assistant", "user", "assistant"]
        if final_assistant else ["user", "assistant", "user"]
    )
    assistant = row["messages"][1]
    assert assistant["reasoning_content"] == native["messages"][1]["reasoning_content"]
    assert assistant["tool_calls"][0]["id"] == "real-call-1"
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"action": "screenshot"}
    assert all(message["loss"] is False for message in row["messages"]
               if message["role"] == "assistant")
