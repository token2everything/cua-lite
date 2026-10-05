from __future__ import annotations

import importlib.util
import json
import zipfile
from importlib.resources import files
from pathlib import Path

from webarena_verified.core.evaluation.value_normalizer import ValueNormalizer

SCRIPT = Path(__file__).parents[2] / "scripts/webarena_verified_evaluator_fork.py"
SPEC = importlib.util.spec_from_file_location("webarena_verified_evaluator_fork", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fork = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fork)


def test_singleton_array_patch_only_adds_explicit_schemas():
    official_path = files("webarena_verified").joinpath("assets/dataset/webarena-verified.json")
    official_data = json.loads(official_path.read_text())
    patch_spec = json.loads(fork.PATCH_PATH.read_text())

    changed = fork.patched_dataset(official_data, patch_spec)

    for task_id, properties in (
        (699, {"website_ids", "customer_group_ids"}),
        (700, {"website_ids", "customer_group_ids"}),
        (811, {"$.issue.assignee_ids"}),
    ):
        original_eval = official_data[task_id]["eval"][1]
        fork_eval = changed[task_id]["eval"][1]
        assert original_eval["expected"] == fork_eval["expected"]
        assert "post_data_schema" not in original_eval
        assert set(fork_eval["post_data_schema"]["properties"]) == properties
        assert all(
            field["type"] == "array" and field["items"] == {"type": "number"}
            for field in fork_eval["post_data_schema"]["properties"].values()
        )
        assert {
            key: value for key, value in fork_eval.items() if key != "post_data_schema"
        } == original_eval

    assert official_data[811]["eval"][1].get("post_data_schema") is None
    assert official_data[0] == changed[0]


def test_patch_makes_singleton_array_normalizable():
    normalizer = ValueNormalizer()
    value = {"$.issue.assignee_ids": [2330]}
    schema = {
        "type": "object",
        "properties": {"$.issue.assignee_ids": {"type": "array", "items": {"type": "number"}}},
    }

    normalized = normalizer.normalize(value, schema, strict=True)

    assert tuple(item.normalized for item in normalized["$.issue.assignee_ids"]) == (2330.0,)


def test_trace_replay_hydrates_external_body_and_orders_requests(tmp_path):
    trace_path = tmp_path / "trace.zip"

    def event(started_at: str, status: int, post_data: dict | None = None) -> str:
        return json.dumps(
            {
                "type": "resource-snapshot",
                "snapshot": {
                    "startedDateTime": started_at,
                    "request": {
                        "method": "POST",
                        "url": "http://gitlab.example/issue",
                        "headers": [],
                        "postData": post_data,
                    },
                    "response": {"status": status, "headers": [], "redirectURL": ""},
                },
            }
        )

    with zipfile.ZipFile(trace_path, "w") as archive:
        archive.writestr(
            "trace.network",
            "\n".join(
                (
                    event("2026-10-04T09:25:50.227Z", 201),
                    event(
                        "2026-10-04T09:25:50.226Z",
                        200,
                        {
                            "mimeType": "application/json",
                            "text": "",
                            "_sha1": "request.json",
                        },
                    ),
                )
            ),
        )
        archive.writestr("resources/request.json", '{"issue":{"assignee_ids":[2330]}}')

    trace = fork.load_archived_trace(trace_path)

    assert [item.request_status for item in trace.events] == [200, 201]
    assert trace.events[0].post_data == {"issue": {"assignee_ids": [2330]}}
