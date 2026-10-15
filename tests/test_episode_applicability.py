"""Offline policy tests with invented evidence and the frozen official evaluators.

No task catalog, historical attempt, site, or model is used. The GUI lifecycle
tests explicitly replace browser rendering; their synthetic fixture pass is not
complete generated-task GT or proof of actual GUI execution.
"""

from __future__ import annotations

import importlib
import json
import sys
import tarfile
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from contract import ContractError, sha256_file, sha256_json

SITES = ("shopping", "shopping_admin", "reddit", "gitlab", "wikipedia", "map")
CONSUMERS = ("replay", "reference")


@pytest.fixture(scope="session")
def runtime(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    unpack = tmp_path_factory.mktemp("frozen-evaluator")
    with tarfile.open(root / "vendor/evaluator_build.tar.gz") as archive:
        archive.extractall(unpack, filter="data")
    package_root = unpack / "evaluator_build_host_v1"
    # The frozen bundle contains source only. This temporary metadata satisfies
    # imports; it is not evidence that production's mounted distribution is pinned.
    metadata = package_root / "webarena_verified-1.2.3.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: webarena-verified\nVersion: 1.2.3\n"
        "Summary: source-only offline test fixture\n"
    )
    sys.path.insert(0, str(package_root))
    try:
        yield SimpleNamespace(
            replay=importlib.import_module("replay"),
            reference=importlib.import_module("generated_env"),
            owner=importlib.import_module("gitlab_episode_evaluator"),
            renderer=importlib.import_module("native_snapshot"),
        )
    finally:
        sys.path.remove(str(package_root))


def _seal(candidate):
    candidate["baseline_sha256"] = sha256_json(candidate["baseline"])
    candidate["task_definition_sha256"] = sha256_json(candidate["task_definition"])
    candidate["grounding_sha256"] = sha256_json(candidate["grounding"])


def _write_recording(root, entries, *, with_gui=True):
    (root / "network.har").write_text(json.dumps({"log": {"entries": entries}}))
    frames = [
        {"type": "context-options"},
        {"type": "before", "callId": "fixture-call", "wallTime": 1767225600000, "startTime": 0},
        {
            "type": "frame-snapshot",
            "snapshot": {
                "isMainFrame": True,
                "callId": "fixture-call",
                "timestamp": 100,
                "frameUrl": "http://gitlab.test/fixture/project/-/issues/1",
                "snapshotName": "fixture-frame",
                "viewport": {"width": 1280, "height": 720},
                "html": ["HTML", {}, ["BODY", {}, ["H1", {}, "offline-fixture"]]],
            },
        },
    ]
    with zipfile.ZipFile(root / "trace.zip", "w") as archive:
        archive.writestr(
            "trace.network",
            "".join(
                json.dumps({"type": "resource-snapshot", "snapshot": entry}) + "\n"
                for entry in entries
            ),
        )
        archive.writestr(
            "trace.trace",
            "".join(json.dumps(frame) + "\n" for frame in (frames if with_gui else frames[:2])),
        )


def _fixture(root, site="shopping", *, declared=False):
    root.mkdir(parents=True, exist_ok=True)
    endpoint = f"http://{site}.test"
    response = {
        "task_type": "RETRIEVE",
        "status": "SUCCESS",
        "retrieved_data": ["offline-fixture"],
        "error_details": None,
    }
    path = "/fixture/project/-/issues/1.json" if declared else "/fixture/read.json"
    entries = [
        {
            "startedDateTime": "2026-01-01T00:00:00+00:00",
            "time": 10,
            "timings": {"send": 1, "wait": 8, "receive": 1},
            "request": {"method": "GET", "url": endpoint + path, "headers": [], "queryString": []},
            "response": {
                "status": 200,
                "headers": [],
                "cookies": [],
                "redirectURL": "",
                "content": {
                    "mimeType": "application/json",
                    "text": json.dumps({"value": "offline-fixture"}),
                },
            },
        }
    ]
    _write_recording(root, entries)
    for name in ("initial.png", "terminal.png", "state_audit.png"):
        (root / name).write_bytes(b"offline image placeholder; never rendered")
    (root / "initial_observation.json").write_text("{}")
    candidate = {
        "generated_task_id": "wavgen-offline-fixture",
        "baseline": {"site_url": endpoint},
        "task_definition": {
            "parameters": {
                "initial_notes": [],
                "expected_read": {"value": "offline-fixture"},
                "gui": {"h1": ["offline-fixture"]},
            }
        },
        "task_spec": {
            "task_id": 2000000,
            "intent_template_id": 2000000,
            "revision": 1,
            "sites": [site],
            "start_urls": [f"__{site.upper()}__{path}"],
            "intent": "Read the invented fixture value.",
            "intent_template": "Read the invented fixture value.",
            "instantiation_dict": {},
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "expected": response,
                    "results_schema": {"type": "array", "items": {"type": "string"}},
                },
                {
                    "evaluator": "NetworkEventEvaluator",
                    "expected": {
                        "url": f"__{site.upper()}__{path}",
                        "http_method": "GET",
                        "response_status": 200,
                        "response_content": {"value": "offline-fixture"},
                    },
                },
            ],
        },
        "reference_response": response,
        "grounding": {
            "artifacts": {
                "har": "network.har",
                "trace": "trace.zip",
                "screenshot": "terminal.png",
                "observation": "initial_observation.json",
            }
        },
    }
    if declared:
        candidate["grounding"]["episode_contract"] = {
            "version": 6,
            "site": "gitlab",
            "entities": {
                "issue": {
                    "project": "fixture/project",
                    "issue_id": 1,
                    "iid": 1,
                    "initial_notes_parameter": "initial_notes",
                    "issue_id_binding": None,
                    "iid_binding": None,
                }
            },
            "nodes": [
                {
                    "name": "read",
                    "kind": "read_issue",
                    "entity": "issue",
                    "response_parameter": "expected_read",
                    "endpoint": ".json",
                    "gui_parameter": "gui",
                    "terminal": True,
                }
            ],
            "partial_order": [],
        }
    _refresh_artifacts(root, candidate)
    (root / "private.jsonl").write_text(json.dumps(candidate) + "\n")
    (root / "private_task.json").write_text(json.dumps(candidate["task_spec"]))
    (root / "final_state.json").write_text(
        json.dumps(
            {
                "task_spec_sha256": sha256_json(candidate["task_spec"]),
                "confirmed": True,
                "reloaded_for_persistence": False,
            }
        )
    )
    return candidate, entries


def _refresh_artifacts(root, candidate):
    candidate["grounding"]["task_spec_sha256"] = sha256_json(candidate["task_spec"])
    candidate["grounding"]["artifact_sha256"] = {
        name: sha256_file(root / relative)
        for name, relative in candidate["grounding"]["artifacts"].items()
    }
    _seal(candidate)


def _boundary(root, candidate, scope, **changes):
    boundary = {
        "scope": scope,
        "recording_context_id": "fixture-context",
        "har_context_id": "fixture-context",
        "playwright_context_id": "fixture-context",
        "audit_context_ids": ["audit-context"],
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:01+00:00",
        "artifact_sha256": {
            name: sha256_file(root / relative)
            for name, relative in (("har", "network.har"), ("trace", "trace.zip"))
        },
        **changes,
    }
    path = root / "episode_boundary.json"
    path.write_text(json.dumps(boundary))
    if scope == "author_context_only":
        candidate["grounding"]["episode_boundary_artifact"] = {
            "path": path.name,
            "sha256": sha256_file(path),
        }
        _seal(candidate)


def _score(runtime, consumer, root, candidate, raw_response=None):
    if consumer == "replay":
        return runtime.replay.validate_replay(root, candidate)
    runtime.replay.install_playwright_trace_loader()
    return runtime.reference.score_reference(
        candidate,
        raw_response or json.dumps(candidate["reference_response"]),
        root,
        {candidate["task_spec"]["sites"][0]: candidate["baseline"]["site_url"]},
    )


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("site", SITES)
def test_generic_sites_use_official_gates_without_episode_policy(runtime, tmp_path, consumer, site):
    candidate, _ = _fixture(tmp_path, site)
    result = _score(runtime, consumer, tmp_path, candidate)
    assert result["status"] == "pass"
    assert result["episode_gate_status"] == "not_applicable"
    assert result["episode_results"] == {}
    assert not any(name.startswith("episode_") for name in result["checks"])
    assert all(result["checks"].values())
    positives = (
        (result["positive_har"], result["positive_playwright"])
        if consumer == "replay"
        else (result["har_results"], result["playwright_results"])
    )
    assert all(
        row["status"] == "success" and row["score"] == 1 for rows in positives for row in rows
    )
    if consumer == "replay":
        assert [row["score"] for row in result["negative_wrong_response"]] == [0, 1]
        assert [row["score"] for row in result["negative_empty_trace"]] == [1, 0]


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("site", ("gitlab", "shopping"))
@pytest.mark.parametrize("contract", (None, {}, "valid"))
def test_declared_contract_without_boundary_holds(runtime, tmp_path, consumer, site, contract):
    candidate, _ = _fixture(tmp_path, site, declared=True)
    if contract != "valid":
        candidate["grounding"]["episode_contract"] = contract
        _seal(candidate)
    result = _score(runtime, consumer, tmp_path, candidate)
    assert result["status"] == result["episode_gate_status"] == "hold"
    assert result["checks"]["episode_contract_pass"] is False
    assert result["episode_boundary_error"]
    assert all(
        row["status"] == "hold" and row["score"] is None
        for row in result["episode_results"].values()
    )


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize(
    "fault", ("missing_required_request", "wrong_network_content", "wrong_response")
)
def test_generic_official_failures_remain_blocked(runtime, tmp_path, consumer, fault):
    candidate, entries = _fixture(tmp_path)
    if fault == "missing_required_request":
        entries[0]["request"]["url"] = candidate["baseline"]["site_url"] + "/fixture/other.json"
        _write_recording(tmp_path, entries)
    elif fault == "wrong_network_content":
        entries[0]["response"]["content"]["text"] = json.dumps({"value": "wrong"})
        _write_recording(tmp_path, entries)
    else:
        candidate["reference_response"] = {
            **candidate["reference_response"],
            "retrieved_data": ["wrong"],
        }
    _refresh_artifacts(tmp_path, candidate)
    result = _score(runtime, consumer, tmp_path, candidate)
    assert result["status"] != "pass"
    assert result["episode_gate_status"] == "not_applicable"
    positives = result["positive_har"] if consumer == "replay" else result["har_results"]
    assert any(row["status"] == "failure" and row["score"] == 0 for row in positives)


@pytest.mark.parametrize(
    "fault", ("malformed_response", "unconfirmed_state", "wrong_state_task_hash")
)
def test_reference_response_and_final_state_gates_remain_required(runtime, tmp_path, fault):
    candidate, _ = _fixture(tmp_path)
    state_path = tmp_path / "final_state.json"
    if fault == "malformed_response":
        raw_response = "not-json"
    else:
        raw_response = None
        state = json.loads(state_path.read_text())
        state["confirmed" if fault == "unconfirmed_state" else "task_spec_sha256"] = (
            False if fault == "unconfirmed_state" else "wrong-task-hash"
        )
        state_path.write_text(json.dumps(state))
    result = _score(runtime, "reference", tmp_path, candidate, raw_response)
    assert result["status"] != "pass"
    assert result["episode_gate_status"] == "not_applicable"
    assert (
        result["checks"][
            "response_contract_valid" if fault == "malformed_response" else "final_state_confirmed"
        ]
        is False
    )


@pytest.fixture
def offline_renderer(runtime, monkeypatch):
    # A lifecycle/markup stub only: no CSS visibility claim or browser is used.
    instances = []

    class MarkupRenderer:
        def __init__(self, trace, terminal_url):
            self.trace = trace
            self.closed = False
            instances.append(self)

        def project_frame(self, index, markup, frame, selectors):
            from bs4 import BeautifulSoup

            document = BeautifulSoup(markup, "html.parser")
            return {
                selector: [element.get_text() for element in document.select(selector)]
                for selector in selectors
            }

        def receipt(self):
            return {"test_only_markup_renderer": True, "trace_sha256": sha256_file(self.trace)}

        def close(self):
            self.closed = True

    monkeypatch.setattr(runtime.renderer, "OriginalDOMRenderer", MarkupRenderer)
    return instances


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_declared_policy_scores_original_representations_and_closes_gui(
    runtime,
    tmp_path,
    consumer,
    offline_renderer,
):
    candidate, _ = _fixture(tmp_path, "gitlab", declared=True)
    scope = "author_context_only" if consumer == "replay" else "model_context_only"
    _boundary(tmp_path, candidate, scope)
    result = _score(runtime, consumer, tmp_path, candidate)
    assert result["episode_gate_status"] == "pass"
    for name in ("positive_har", "positive_playwright"):
        row = result["episode_results"][name]
        assert row["status"] == "pass" and row["recording_scope"] == scope
        assert row["gui_matches"]["read"]["call_id"] == "fixture-call"
    if consumer == "replay":
        assert result["episode_results"]["negative_empty_trace"]["status"] == "fail"
    assert result["original_frame_visible_GUI_projection"]["test_only_markup_renderer"] is True
    assert len(offline_renderer) == 1 and offline_renderer[0].closed


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize(
    "fault", ("hash", "mixed_context", "window", "missing_gui", "wrong_site", "invalid_contract")
)
def test_declared_policy_preserves_provenance_holds(
    runtime,
    tmp_path,
    consumer,
    fault,
    offline_renderer,
):
    candidate, entries = _fixture(
        tmp_path, "shopping" if fault == "wrong_site" else "gitlab", declared=True
    )
    scope = "author_context_only" if consumer == "replay" else "model_context_only"
    if fault == "missing_gui":
        _write_recording(tmp_path, entries, with_gui=False)
        _refresh_artifacts(tmp_path, candidate)
    elif fault == "invalid_contract":
        candidate["grounding"]["episode_contract"] = None
        _seal(candidate)
    changes = {}
    if fault == "hash":
        changes["artifact_sha256"] = {
            "har": "wrong-hash",
            "trace": sha256_file(tmp_path / "trace.zip"),
        }
    elif fault == "mixed_context":
        changes["har_context_id"] = "different-context"
    elif fault == "window":
        changes["ended_at"] = "2026-01-01T00:00:00.050+00:00"
    _boundary(tmp_path, candidate, scope, **changes)
    result = _score(runtime, consumer, tmp_path, candidate)
    assert result["status"] == result["episode_gate_status"] == "hold"
    assert result["checks"]["episode_contract_pass"] is False
    assert all(instance.closed for instance in offline_renderer)


@pytest.mark.parametrize("fault", ("hash", "escape"))
def test_author_boundary_pointer_is_bound_to_its_grounding(runtime, tmp_path, fault):
    candidate, _ = _fixture(tmp_path, "gitlab", declared=True)
    _boundary(tmp_path, candidate, "author_context_only")
    proof = candidate["grounding"]["episode_boundary_artifact"]
    proof["sha256" if fault == "hash" else "path"] = (
        "wrong-boundary-hash" if fault == "hash" else "../outside-batch.json"
    )
    _seal(candidate)
    result = _score(runtime, "replay", tmp_path, candidate)
    assert result["status"] == "hold"
    assert "does not match grounding" in result["episode_boundary_error"]


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_declared_episode_failure_blocks_passing_official_vectors(
    runtime,
    tmp_path,
    consumer,
    offline_renderer,
):
    candidate, _ = _fixture(tmp_path, "gitlab", declared=True)
    candidate["task_definition"]["parameters"]["gui"] = {"h1": ["absent fixture field"]}
    scope = "author_context_only" if consumer == "replay" else "model_context_only"
    _boundary(tmp_path, candidate, scope)
    result = _score(runtime, consumer, tmp_path, candidate)
    positives = result["positive_har"] if consumer == "replay" else result["har_results"]
    assert all(row["status"] == "success" and row["score"] == 1 for row in positives)
    assert result["episode_gate_status"] == "fail"
    assert result["status"] == ("reject" if consumer == "replay" else "fail")
    assert len(offline_renderer) == 1 and offline_renderer[0].closed


def test_official_har_trace_disagreement_cannot_pass(runtime, tmp_path):
    candidate, entries = _fixture(tmp_path)
    wrong = deepcopy(entries)
    wrong[0]["response"]["content"]["text"] = json.dumps({"value": "wrong"})
    _write_recording(tmp_path, wrong)
    (tmp_path / "network.har").write_text(json.dumps({"log": {"entries": entries}}))
    _refresh_artifacts(tmp_path, candidate)
    for consumer in CONSUMERS:
        result = _score(runtime, consumer, tmp_path, candidate)
        assert result["status"] != "pass"
        assert result["checks"]["har_playwright_results_equal"] is False


def test_grounding_hash_mismatch_still_fails_before_scoring(runtime, tmp_path):
    candidate, _ = _fixture(tmp_path)
    (tmp_path / "network.har").write_text("{}")
    with pytest.raises(ContractError, match="grounding artifact hash mismatch: har"):
        _score(runtime, "replay", tmp_path, candidate)


def test_gui_is_closed_when_episode_scoring_raises(
    runtime, tmp_path, monkeypatch, offline_renderer
):
    candidate, _ = _fixture(tmp_path, "gitlab", declared=True)
    _boundary(tmp_path, candidate, "model_context_only")

    def raise_during_scoring(*args, **kwargs):
        raise RuntimeError("test-only scoring interruption")

    monkeypatch.setattr(runtime.owner, "evaluate_episode", raise_during_scoring)
    with pytest.raises(RuntimeError, match="test-only scoring interruption"):
        _score(runtime, "reference", tmp_path, candidate)
    assert len(offline_renderer) == 1 and offline_renderer[0].closed
