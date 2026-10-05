"""WebArena-Verified pinned catalog tests (no BrowserGym import required)."""

from __future__ import annotations

import hashlib
import json
import zipfile
from types import SimpleNamespace

import pytest

from lite.gym.envs.browsergym import webarena_verified


def _clear_catalog_caches() -> None:
    webarena_verified.task_records.cache_clear()
    webarena_verified.hard_task_ids.cache_clear()
    webarena_verified.task_ids.cache_clear()
    webarena_verified._records_by_bgym_id.cache_clear()


@pytest.fixture
def synthetic_catalog(tmp_path, monkeypatch):
    root = tmp_path / "webarena_verified"
    dataset_dir = root / "assets" / "dataset"
    subset_dir = dataset_dir / "subsets"
    subset_dir.mkdir(parents=True)

    records = [
        {
            "task_id": task_id,
            "intent_template_id": 100 + task_id // 4,
            "revision": 2,
            "sites": ["shopping"],
            "intent": f"Task {task_id}",
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "expected": {
                        "task_type": "mutate" if task_id < 374 else "retrieve",
                        "status": "SUCCESS",
                    },
                }
            ],
        }
        for task_id in range(webarena_verified.TASK_COUNT)
    ]
    dataset = json.dumps(records).encode()
    hard = json.dumps({"task_ids": list(range(webarena_verified.HARD_TASK_COUNT))}).encode()
    (dataset_dir / "webarena-verified.json").write_bytes(dataset)
    (subset_dir / "webarena-verified-hard.json").write_bytes(hard)

    monkeypatch.setattr(webarena_verified, "_package_root", lambda: root)
    monkeypatch.setattr(webarena_verified, "DATASET_SHA256", hashlib.sha256(dataset).hexdigest())
    monkeypatch.setattr(webarena_verified, "HARD_SUBSET_SHA256", hashlib.sha256(hard).hexdigest())
    _clear_catalog_caches()
    yield root
    _clear_catalog_caches()


def test_catalog_identity_subsets_and_mutation_type(synthetic_catalog):
    task_ids = webarena_verified.task_ids()

    assert len(task_ids) == 812
    assert task_ids[0] == "webarena_verified.100.0.2"
    assert task_ids[-1] == "webarena_verified.302.811.2"
    assert sum(map(webarena_verified.is_hard, task_ids)) == 258
    assert sum(map(webarena_verified.is_mutating, task_ids)) == 374
    assert webarena_verified.task_record("webarena_verified.unknown") is None


def test_catalog_hash_mismatch_fails_loudly(synthetic_catalog, monkeypatch):
    monkeypatch.setattr(webarena_verified, "DATASET_SHA256", "0" * 64)
    _clear_catalog_caches()

    with pytest.raises(RuntimeError, match="catalog hash mismatch"):
        webarena_verified.task_records()


def test_dataset_distribution_version_is_pinned(monkeypatch):
    monkeypatch.setattr(webarena_verified, "version", lambda name: "1.2.2")

    with pytest.raises(RuntimeError, match="expected 1.2.3, found 1.2.2"):
        webarena_verified._package_root()


def test_browsergym_integration_distribution_is_pinned(tmp_path, monkeypatch):
    package_dir = tmp_path / "browsergym" / "webarena_verified"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    monkeypatch.setattr(
        webarena_verified,
        "version",
        lambda name: {
            "browsergym-webarena-verified": "0.14.3",
        }[name],
    )
    monkeypatch.setattr(
        webarena_verified.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(tmp_path / "browsergym")]),
    )

    assert webarena_verified.browsergym_package_root() == package_dir


def test_trace_loader_hydrates_external_request_body(tmp_path):
    pytest.importorskip("webarena_verified")
    trace_path = tmp_path / "trace.zip"
    body = '{"availability":"busy","clear_status_after":null}'
    event = {
        "type": "resource-snapshot",
        "snapshot": {
            "request": {
                "method": "PUT",
                "url": "http://gitlab/api/v4/user/status",
                "headers": [],
                "postData": {
                    "mimeType": "application/json",
                    "text": "",
                    "_sha1": "request.json",
                },
            },
            "response": {"status": 200, "headers": [], "redirectURL": ""},
        },
    }
    with zipfile.ZipFile(trace_path, "w") as archive:
        archive.writestr("trace.network", json.dumps(event))
        archive.writestr("resources/request.json", body)

    webarena_verified.install_playwright_trace_loader()
    from webarena_verified.types.tracing import NetworkTrace

    trace = NetworkTrace.from_playwright_trace(trace_path)

    assert trace.events[0].post_data == {
        "availability": "busy",
        "clear_status_after": None,
    }


def test_trace_loader_orders_events_by_request_start(tmp_path):
    pytest.importorskip("webarena_verified")
    trace_path = tmp_path / "trace.zip"

    def event(started_at, status):
        return {
            "type": "resource-snapshot",
            "snapshot": {
                "startedDateTime": started_at,
                "request": {
                    "method": "POST",
                    "url": "http://gitlab/-/profile",
                    "headers": [],
                },
                "response": {
                    "status": status,
                    "headers": [],
                    "redirectURL": "http://gitlab/-/profile" if status == 302 else "",
                },
            },
        }

    with zipfile.ZipFile(trace_path, "w") as archive:
        archive.writestr(
            "trace.network",
            "\n".join(
                [
                    json.dumps(event("2026-10-04T09:25:50.227Z", 302)),
                    json.dumps(event("2026-10-04T09:25:50.226Z", 200)),
                ]
            ),
        )

    webarena_verified.install_playwright_trace_loader()
    from webarena_verified.types.tracing import NetworkTrace

    trace = NetworkTrace.from_playwright_trace(trace_path)

    assert [event.request_status for event in trace.events] == [200, 302]
