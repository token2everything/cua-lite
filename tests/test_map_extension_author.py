"""Verify Map boundaries with fixtures and an owned loopback browser regression."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import author_map_extension as author
import pytest


def common_args(output):
    return [
        "--batch-root",
        str(output),
        "--site-url",
        "http://localhost:8080",
        "--viewport-width",
        "1920",
        "--viewport-height",
        "1080",
        "--comparison-viewport-width",
        "1280",
        "--comparison-viewport-height",
        "720",
    ]


@pytest.fixture
def case(tmp_path):
    ownership = {
        "site": "map",
        "run_id": "test-map",
        "session_nonce": "a" * 32,
        "sandbox_id": "owned-sandbox",
        "generated_task_id": "wavgen-test-map-readiness",
        "model_profile": "author",
        "attempt": "01",
        "container_name": "wav-v03-map-author-wavgen-test-map-readiness-01",
        "map_root": "/var/lib/wav-v03/map-executions/wavgen-test-map-readiness/author-01",
    }
    inspection = {
        key: value for key, value in ownership.items() if key not in {"sandbox_id", "map_root"}
    }
    inspection.update(
        {
            "container_name": "/" + ownership["container_name"],
            "container_id": "b" * 64,
            "running": True,
            "image": "locked-map@sha256:" + "c" * 64,
            "image_id": "sha256:" + "d" * 64,
            "ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8080"}]},
            "mounts": [
                {
                    "Destination": destination,
                    "Source": ownership["map_root"] + "/" + suffix,
                    "RW": True,
                }
                for destination, suffix in (
                    ("/data/database", "tile-db"),
                    ("/data/nominatim/postgres", "nominatim-db"),
                    ("/var/lib/postgresql/14/main", "website-db"),
                    ("/data/tiles", "tiles"),
                    ("/data/style", "style"),
                )
            ]
            + [
                {
                    "Destination": "/mnt/map-source/" + suffix,
                    "Source": "/fixed/" + suffix,
                    "RW": False,
                }
                for suffix in ("nominatim-flatnode", "routing-car", "routing-bike", "routing-foot")
            ],
        }
    )
    ownership_path, inspection_path = tmp_path / "producer.json", tmp_path / "inspect.json"
    ownership_path.write_text(json.dumps(ownership))
    inspection_path.write_text(json.dumps(inspection))
    output = tmp_path / "outputs"
    argv = common_args(output) + [
        "--mode",
        "explore",
        "--ownership-receipt",
        str(ownership_path),
        "--container-inspection",
        str(inspection_path),
        "--run-id",
        "test-map",
        "--nonce",
        "a" * 32,
        "--sandbox-id",
        "owned-sandbox",
        "--container-id",
        "b" * 64,
        "--start-path",
        "/#map=5/51.500/-0.100",
        "--attempt",
        "search-01",
        "--search-query",
        "library",
    ]
    return SimpleNamespace(
        argv=argv,
        ownership=ownership,
        inspection=inspection,
        ownership_path=ownership_path,
        inspection_path=inspection_path,
        output=output,
    )


class FakePage:
    def goto(self, *args, **kwargs):
        pass

    def set_viewport_size(self, size):
        pass

    def on(self, *args):
        pass


class FakeContext:
    def __init__(self):
        self.tracing = SimpleNamespace(start=lambda **kwargs: None, stop=lambda **kwargs: None)

    def route(self, *args):
        pass

    def set_default_timeout(self, value):
        pass

    def new_page(self):
        return FakePage()

    def close(self):
        pass


@pytest.fixture
def browser(monkeypatch):
    captures = []

    def new_context(**kwargs):
        captures.append(kwargs)
        return FakeContext()

    runtime = SimpleNamespace(
        chromium=SimpleNamespace(
            launch=lambda **kwargs: SimpleNamespace(new_context=new_context, close=lambda: None)
        )
    )

    class Runtime:
        def __enter__(self):
            return runtime

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(
        sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=Runtime)
    )
    monkeypatch.setattr(author, "_record", lambda *args: {"links": []})
    monkeypatch.setattr(
        author,
        "_search_batch",
        lambda *args: {
            "snapshot": {"results": [], "more_results_links": []},
        },
    )
    return captures


def test_anonymous_explore_archives_real_identity_without_baseline(case, browser):
    author.main(case.argv)
    out = case.output / "exploration/map/search-01"
    receipt = json.loads((out / "exploration_receipt.json").read_text())
    assert receipt["GT_emitted"] is False
    assert receipt["model_calls"] == receipt["generated_candidates"] == 0
    assert receipt["exploration_identity"]["author_baseline_created"] is False
    assert "baseline" not in receipt
    assert not (out / "baseline.json").exists()
    assert not (out / "draft.json").exists()
    assert (out / "producer_ownership.json").read_bytes() == case.ownership_path.read_bytes()
    assert receipt["exploration_identity"]["source_proofs"]["container_inspection"]["sha256"] == (
        hashlib.sha256(case.inspection_path.read_bytes()).hexdigest()
    )
    assert browser[0]["storage_state"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_nonce", "f" * 32),
        ("sandbox_id", "someone-else"),
        ("map_root", "/var/lib/wav-v03/map-executions/other/author-01"),
    ],
)
def test_explore_rejects_foreign_producer_before_browser_or_output(case, field, value):
    case.ownership[field] = value
    case.ownership_path.write_text(json.dumps(case.ownership))
    with pytest.raises(ValueError):
        author.main(case.argv)
    assert not case.output.exists()


@pytest.mark.parametrize(
    "change", ["stopped", "foreign-cid", "wrong-port", "foreign-db", "writable-seed"]
)
def test_explore_rejects_nonowned_or_mutable_container_before_output(case, change):
    if change == "stopped":
        case.inspection["running"] = False
    elif change == "foreign-cid":
        case.inspection["container_id"] = "e" * 64
    elif change == "wrong-port":
        case.inspection["ports"]["8080/tcp"][0]["HostPort"] = "9999"
    elif change == "foreign-db":
        case.inspection["mounts"][0]["Source"] = "/other/tile-db"
    else:
        case.inspection["mounts"][-1]["RW"] = True
    case.inspection_path.write_text(json.dumps(case.inspection))
    with pytest.raises(ValueError):
        author.main(case.argv)
    assert not case.output.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--baseline-receipt", "/missing/baseline.json"],
        ["--storage-state", "/missing/auth.json"],
        ["--task-config", "/missing/tasks.json"],
    ],
)
def test_anonymous_explore_cannot_accept_author_or_auth_inputs(case, extra):
    with pytest.raises(SystemExit):
        author._parse_args(case.argv + extra)


def test_explore_requires_actual_proof_inputs(tmp_path):
    with pytest.raises(SystemExit):
        author._parse_args(common_args(tmp_path) + ["--mode", "explore", "--start-path", "/"])


def test_ground_still_requires_fresh_author_baseline(tmp_path):
    ground = common_args(tmp_path / "output") + [
        "--task-config",
        str(tmp_path / "missing.json"),
        "--generated-task-id",
        "wavgen-test-map-6107001",
    ]
    with pytest.raises(SystemExit):
        author._parse_args(ground)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {"site_url": "http://localhost:8080", "fresh_from_image": False, "owned_run": True}
        )
    )
    with pytest.raises(ValueError, match="fresh fixed-image baseline"):
        author.main(ground + ["--baseline-receipt", str(baseline)])
    assert not (tmp_path / "output").exists()


def test_explore_records_viewport_difference_without_ground_truth(case, browser, monkeypatch):
    batches = iter(
        [
            {
                "snapshot": {
                    "results": [
                        {
                            "category": "Library",
                            "label": "visible A",
                            "url": "http://localhost:8080/node/1",
                        }
                    ],
                    "more_results_links": [],
                }
            },
            {"snapshot": {"results": [], "more_results_links": []}},
        ]
    )
    monkeypatch.setattr(author, "_search_batch", lambda *args: deepcopy(next(batches)))
    author.main(case.argv)
    out = case.output / "exploration/map/search-01"
    observation = json.loads((out / "observation.json").read_text())
    assert observation["searches"][0]["viewport_comparison"]["same_feature_facts"] is False
    assert not (out / "draft.json").exists()


def test_formal_ground_rejects_viewport_difference_and_preserves_failure(
    tmp_path, browser, monkeypatch
):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {"site_url": "http://localhost:8080", "fresh_from_image": True, "owned_run": True}
        )
    )
    config = tmp_path / "config.json"
    task_id = "wavgen-test-map-6107001"
    config.write_text(
        json.dumps(
            {
                "view": {"start_path": "/#map=5/51.500/-0.100"},
                "tasks": [
                    {
                        "generated_task_id": task_id,
                        "mode": "retrieve-filtered-features",
                        "parameters": {"query": "library"},
                    }
                ],
            }
        )
    )
    batches = iter(
        [
            {
                "snapshot": {
                    "results": [
                        {
                            "category": "Library",
                            "label": "visible A",
                            "url": "http://localhost:8080/node/1",
                        }
                    ],
                    "more_results_links": [],
                }
            },
            {"snapshot": {"results": [], "more_results_links": []}},
        ]
    )
    monkeypatch.setattr(author, "_search_batch", lambda *args: deepcopy(next(batches)))
    with pytest.raises(ValueError, match="Different viewports"):
        author.main(
            common_args(tmp_path / "outputs")
            + [
                "--baseline-receipt",
                str(baseline),
                "--task-config",
                str(config),
                "--generated-task-id",
                task_id,
            ]
        )
    out = tmp_path / "outputs/author" / task_id / "grounding-01"
    assert not (out / "draft.json").exists()
    observed = json.loads((out / "observation.json").read_text())
    assert observed["failure"]["type"] == "ValueError"
    assert observed["searches"][0]["viewport_comparison"]["same_feature_facts"] is False


@pytest.mark.parametrize("strict", [False, True])
def test_missing_feature_details_remain_actual_negative_evidence(tmp_path, monkeypatch, strict):
    target = "http://localhost:8080/node/1"
    page = SimpleNamespace(
        goto=lambda *args, **kwargs: SimpleNamespace(url=target, status=404),
        evaluate=lambda *args: {
            "headings": [{"text": "Not Found", "selector": "h1"}],
            "detail_rows": [],
        },
    )
    monkeypatch.setattr(author, "_record", lambda *args: {"url": target, "body_text": "Not Found"})
    if strict:
        with pytest.raises(ValueError, match="real visible identity and details"):
            author._feature_snapshot(page, tmp_path, target, "feature", strict=True)
    else:
        observed = author._feature_snapshot(page, tmp_path, target, "feature", strict=False)
        assert observed["feature_readiness"]["valid_identity_and_details"] is False
        assert "identity_target" not in observed and "detail_target" not in observed
    saved = json.loads((tmp_path / "feature.json").read_text())
    assert saved["request"]["status"] == 404
    assert saved["feature_readiness"]["identity_heading_count"] == 0


def test_explicit_feature_paths_must_come_from_matching_real_packet(case):
    source = case.ownership_path.parent / "search.json"
    source.write_text(
        json.dumps(
            {
                "exploration_identity": {
                    "run_id": "test-map",
                    "session_nonce": "a" * 32,
                    "sandbox_id": "owned-sandbox",
                    "container_id": "b" * 64,
                },
                "searches": [],
            }
        )
    )
    with pytest.raises(ValueError, match="absent from the actual recorded search packet"):
        author.main(case.argv + ["--feature-source", str(source), "--feature-path", "/node/1"])
    assert not case.output.exists()


def test_explicit_features_preserve_source_and_visit_only_its_actual_links(
    case, browser, monkeypatch
):
    source = case.ownership_path.parent / "search.json"
    row = {"category": "Library", "label": "visible A", "url": "http://localhost:8080/node/1"}
    source.write_text(
        json.dumps(
            {
                "exploration_identity": {
                    "run_id": "test-map",
                    "session_nonce": "a" * 32,
                    "sandbox_id": "owned-sandbox",
                    "container_id": "b" * 64,
                },
                "searches": [
                    {
                        "query": "library",
                        "viewport_comparison": {"same_feature_facts": True},
                        "batches": [{"snapshot": {"results": [row]}}],
                    }
                ],
            }
        )
    )
    visits = []

    def feature(page, output, target, name, *, strict):
        visits.append((target, strict))
        return {
            "feature_readiness": {"valid_identity_and_details": False},
            "request": {"status": 404},
        }

    monkeypatch.setattr(author, "_feature_snapshot", feature)
    author.main(case.argv + ["--feature-source", str(source), "--feature-path", "/node/1"])
    assert visits == [("http://localhost:8080/node/1", False)]
    out = case.output / "exploration/map/search-01"
    assert (out / "feature_source_observation.json").read_bytes() == source.read_bytes()
    observation = json.loads((out / "observation.json").read_text())
    assert observation["features"][0]["search_row"] == row
    assert observation["features"][0]["observation"]["request"]["status"] == 404
    assert not (out / "draft.json").exists()


@pytest.mark.parametrize("status,ready", [(200, True), (503, False)])
def test_route_terminal_targets_require_actual_successful_backend(
    tmp_path, monkeypatch, status, ready
):
    targets = [
        {"tag": "h2", "text": "Directions", "selector": "#sidebar_content > h2"},
        {"tag": "p", "text": "Distance: 856m. Time: 0:02.", "selector": "#route_summary"},
        {
            "tag": "table",
            "text": "1. Start on unnamed road\t20m\n2. Reach destination",
            "selector": "#directions",
        },
    ]
    row = SimpleNamespace(
        scroll_into_view_if_needed=lambda: None,
        evaluate=lambda *args: {"text": "numbered instruction", "full_row_visible": True},
    )
    page = SimpleNamespace(
        evaluate=lambda *args: targets,
        locator=lambda *args: SimpleNamespace(count=lambda: 2, nth=lambda index: row),
        screenshot=lambda **kwargs: None,
    )
    monkeypatch.setattr(
        author, "_record", lambda *args: {"url": "http://localhost:8080/directions"}
    )
    snapshot = author._route_snapshot(
        page,
        tmp_path,
        "route",
        [
            {
                "method": "GET",
                "status": status,
                "url": "http://localhost:8080/osrm/routed-car/route/v1/driving/1,2;3,4?steps=true",
            }
        ],
    )
    assert snapshot["route_readiness"]["valid_visible_directions_and_backend"] is ready
    if ready:
        assert snapshot["summary_target"]["selector"] == "#route_summary"
        assert snapshot["backend_request"]["status"] == 200
    else:
        assert "summary_target" not in snapshot
    assert (tmp_path / "route.json").is_file()


@pytest.fixture
def visible_records():
    def row(category, label, key):
        return {
            "category": category,
            "data_prefix": category,
            "label": label,
            "url": f"http://localhost:8080/node/{key}",
        }

    a = row("Library", "Library, 33, K Street, City, County, State, 11111, US", 1)
    b = row("Library", "Library, 206, J Street, City, County, State, 11111, US", 2)
    c = row("Library", "Library, 400, F Street, City, County, State, 11111, US", 3)
    d = row("Clinic", "Library, 85, H Street, Other City, County, State, 22222, US", 4)
    e = row("Library", d["label"], 4)
    f = row("Library", "Library, 10, S Street, Other City, County, State, 33333, US", 5)
    g = row("Station", "Library, Road, Village, County X, 44444, US", 6)
    h = row("Stop", g["label"], 7)
    batches = [[a, b, c, d, f, g, h], [a, b, c, e, f], [a, b, c]]
    searches = [
        {"query": query, "batches": [{"snapshot": {"results": rows, "more_results_links": []}}]}
        for query, rows in zip(("broad", "region", "city"), batches, strict=True)
    ]
    return {"searches": searches, "search": searches[0]["batches"][0]["snapshot"]}


@pytest.mark.parametrize(
    "mode,parameters,expected",
    [
        ("retrieve-record-overlap", {"queries": ["broad", "region", "city"]}, [3, 3, 3]),
        (
            "retrieve-record-membership-rank",
            {"queries": ["broad", "region", "city"], "minimum_query_count": 2},
            [3, 3, 3, 2],
        ),
        (
            "retrieve-record-union-category-summary",
            {"queries": ["broad", "region", "city"]},
            [5, 1, 1, 1],
        ),
        (
            "retrieve-record-constrained-union-statistics",
            {
                "queries": ["broad", "region", "city"],
                "category": "Library",
                "address_components": ["State"],
                "postal_component_from_end": 2,
            },
            [5, 3],
        ),
        (
            "retrieve-record-address-extreme-comparison",
            {"queries": ["broad", "region"], "category": "Library", "region": "City"},
            [33, 400, 367, 3],
        ),
    ],
)
def test_record_rules_use_visible_pairs_and_complete_windows(
    visible_records, mode, parameters, expected
):
    definition, evidence, _ = author.build_extension_definition(
        {"mode": mode, "parameters": parameters}, visible_records
    )
    values = definition["answer"]["values"]
    if mode in {"retrieve-record-overlap", "retrieve-record-membership-rank"}:
        assert [value["query_count"] for value in values] == expected
        assert values[0]["label"].startswith("Library, 206,")
    elif mode == "retrieve-record-union-category-summary":
        assert [value["record_count"] for value in values] == expected
        assert sum(value["record_count"] for value in values) == 8
    elif mode == "retrieve-record-constrained-union-statistics":
        assert list(values[0].values()) == expected
    else:
        assert [
            values[0][key]
            for key in (
                "lower_building_number",
                "higher_building_number",
                "building_number_difference",
                "matching_result_count",
            )
        ] == expected
    for item in evidence:
        source = visible_records
        for part in item["location"].strip("/").split("/"):
            source = source[int(part)] if isinstance(source, list) else source[part]
    from author import _task_spec

    task, _ = _task_spec(
        {
            "task_definition": definition,
            "site": "map",
            "task_id": 6107001,
            "intent_template_id": 6108001,
            "start_url": "__MAP__/",
            "revision": 1,
        }
    )
    assert task["sites"] == ["map"]


def test_directional_difference_does_not_merge_inconsistent_category_by_hidden_url(visible_records):
    definition, _, _ = author.build_extension_definition(
        {"mode": "retrieve-record-difference", "parameters": {"queries": ["broad", "region"]}},
        {**visible_records, "searches": visible_records["searches"][:2]},
    )
    assert [value["category"] for value in definition["answer"]["values"]] == [
        "Clinic",
        "Station",
        "Stop",
    ]


def test_category_comparison_preserves_declared_order_and_zero_counts(visible_records):
    definition, _, _ = author.build_extension_definition(
        {
            "mode": "retrieve-multi-query-category-comparison",
            "parameters": {"queries": ["broad", "region"], "categories": ["Library", "Clinic"]},
        },
        {**visible_records, "searches": visible_records["searches"][:2]},
    )
    assert definition["answer"]["values"] == [
        {
            "category": "Library",
            "first_query_count": 4,
            "second_query_count": 5,
            "second_minus_first": 1,
        },
        {
            "category": "Clinic",
            "first_query_count": 1,
            "second_query_count": 0,
            "second_minus_first": -1,
        },
    ]


def test_union_address_group_counts_visible_records_with_distinct_categories(visible_records):
    definition, _, _ = author.build_extension_definition(
        {
            "mode": "retrieve-record-union-address-summary",
            "parameters": {
                "queries": ["broad", "region", "city"],
                "categories": ["Library", "Clinic"],
                "component_from_end": 2,
            },
        },
        visible_records,
    )
    assert definition["answer"]["values"] == [
        {"address_component": "11111", "result_count": 3, "distinct_category_count": 1},
        {"address_component": "22222", "result_count": 2, "distinct_category_count": 2},
        {"address_component": "33333", "result_count": 1, "distinct_category_count": 1},
    ]


@pytest.mark.parametrize(
    "mode,parameters,prefix",
    [
        (
            "navigate-numeric-extremes-directions",
            {"query": "broad", "category": "Library", "address_components": ["City"]},
            "Library, 400,",
        ),
        (
            "navigate-dominant-group-directions",
            {
                "query": "broad",
                "categories": ["Library", "Clinic", "Station", "Stop"],
                "component_from_end": 3,
                "target_category": "Clinic",
                "from_label": "provided origin",
            },
            "Library, 85,",
        ),
        (
            "navigate-common-record-directions",
            {
                "queries": ["broad", "region"],
                "category": "Library",
                "address_components": ["City", "33"],
                "from_label": "provided origin",
            },
            "Library, 33,",
        ),
    ],
)
def test_route_endpoint_rules_select_full_visible_labels(visible_records, mode, parameters, prefix):
    selected = author._select_route_endpoints(
        {"mode": mode, "parameters": parameters}, visible_records
    )
    assert selected["to_label"].startswith(prefix)
    assert selected["sources"] == ["/searches"]
    if mode == "navigate-numeric-extremes-directions":
        assert selected["from_label"].startswith("Library, 33,")


def test_three_window_difference_ranks_multiple_actual_visible_records(visible_records):
    visible_records["searches"][2]["batches"][0]["snapshot"]["results"] = [
        visible_records["searches"][0]["batches"][0]["snapshot"]["results"][4]
    ]
    selected = author._select_route_endpoints(
        {
            "mode": "navigate-record-difference-directions",
            "parameters": {
                "queries": ["broad", "region", "other"],
                "category": "Library",
                "address_components": ["City"],
                "from_label": "provided origin",
            },
        },
        visible_records,
    )
    assert selected["to_label"].startswith("Library, 206,")


@pytest.fixture
def directions():
    return {
        "from_query": "visible origin",
        "to_query": "visible destination",
        "engine_label": "Car (OSRM)",
        "route_link_generation": {"zoom": 17},
        "navigation_request": {
            "method": "GET",
            "status": 200,
            "url": "http://localhost:8080/directions?engine=fossgis_osrm_car&route=2%2C1%3B4%2C3",
            "query_params": {"engine": ["fossgis_osrm_car"], "route": ["2,1;4,3"]},
            "is_navigation_request": True,
            "resource_type": "document",
        },
        "observation": {
            "url": "http://localhost:8080/directions?engine=fossgis_osrm_car&route=2%2C1%3B4%2C3#map=16/3/4",
            "controls": [
                {"name": "route_from", "value": "2, 1"},
                {"name": "route_to", "value": "4, 3"},
                {
                    "name": "routing_engines",
                    "value": "1",
                    "options": [{"label": "Car (OSRM)", "selected": True}],
                },
            ],
            "summary_target": {
                "text": "Distance: 1.2km. Time: 1:05.",
                "selector": "#actual_summary",
            },
            "heading_target": {"text": "Directions", "selector": "#actual_heading"},
            "instructions_target": {
                "text": "1. Start\n2. Reach destination",
                "selector": "#actual_table",
            },
            "tables": [[["", "1. Start", "2m"], ["", "2. Reach destination", ""]]],
            "backend_request": {
                "method": "GET",
                "status": 200,
                "url": "http://localhost:8080/osrm/routed-car/route/v1/driving/1,2;3,4?steps=true",
            },
        },
    }


def test_route_profile_answer_uses_displayed_units_and_instruction_rows(directions):
    definition, _, _ = author.build_extension_definition(
        {
            "mode": "retrieve-route-profile-comparison",
            "parameters": {
                "from_label": "visible origin",
                "to_label": "visible destination",
                "engine_labels": ["Car (OSRM)"],
            },
        },
        {"searches": [], "routes": [directions]},
    )
    assert definition["answer"]["values"] == [
        {
            "engine_label": "Car (OSRM)",
            "distance_metres": 1200,
            "displayed_time_minutes": 65,
            "instruction_count": 2,
        }
    ]


def test_directions_navigation_binds_real_controls_summary_and_backend_without_map_fragment(
    directions,
):
    definition, evidence, checks = author.build_extension_definition(
        {
            "mode": "navigate-directions",
            "parameters": {
                "from_label": "visible origin",
                "to_label": "visible destination",
                "engine_labels": ["Car (OSRM)"],
                "route_link_zoom": 17,
            },
        },
        {
            "searches": [],
            "routes": [directions],
            "terminal_route": directions,
            "route_endpoints": {"from_label": "visible origin", "to_label": "visible destination"},
        },
    )
    assert [item["kind"] for item in checks] == ["input_value"] * 3 + ["visible_text_exact"] * 3
    assert definition["parameters"]["engine_value"] == "1"
    assert definition["parameters"]["from_label"] == "visible origin"
    assert definition["parameters"]["from_input"] == "2, 1"
    assert definition["completion_events"][0]["response_status"] == 200
    assert definition["completion_events"][0]["url_template"] == "__MAP__/directions"
    assert definition["completion_events"][0]["http_method"] == "GET"
    assert definition["completion_events"][0]["query_params"] == {
        "engine": ["fossgis_osrm_car"], "route": ["2,1;4,3"]
    }
    assert definition["completion_events"][0]["last_event_only"] is True
    assert "#map" not in json.dumps(definition["completion_events"])
    assert {item["field"] for item in evidence} >= {
        "/task_definition/completion_events/0/url_template",
        "/task_definition/parameters/route_summary",
    }
    from author import _task_spec

    task, response = _task_spec(
        {
            "task_definition": definition,
            "site": "map",
            "task_id": 6107001,
            "intent_template_id": 6108001,
            "start_url": "__MAP__/",
            "revision": 1,
        }
    )
    assert response["retrieved_data"] is None
    assert "Distance: 1.2km" not in task["intent"]
    assert "zoom level 17" in task["intent"]
    assert "Press Go to recompute" in task["intent"]
    assert "If #map= is missing from the browser address, click the visible Zoom Out control once" in task["intent"]
    assert "restore level 17 if it changed, without pressing Go again" in task["intent"]
    assert "its generated route link in the browser address bar" in task["intent"]
    assert "2,1;4,3" not in task["intent"]


def test_numeric_extremes_navigation_binds_opened_summary_and_actual_document_query(
    directions, visible_records
):
    config = {
        "mode": "navigate-numeric-extremes-directions",
        "parameters": {
            "query": "broad", "category": "Library", "address_components": ["City"],
            "engine_labels": ["Car (OSRM)"],
            "route_link_zoom": 17,
        },
    }
    endpoints = author._select_route_endpoints(config, visible_records)
    original, opened = deepcopy(directions), deepcopy(directions)
    original["observation"]["summary_target"]["text"] = "Distance: 856m. Time: 0:02."
    opened["observation"]["summary_target"]["text"] = "Distance: 965m. Time: 0:02."
    opened["observation"]["url"] = opened["observation"]["url"].replace(
        "2%2C1%3B4%2C3", "2.00000%2C1.00000%3B4.00000%2C3.00000"
    )
    definition, evidence, checks = author.build_extension_definition(
        config, {**visible_records, "routes": [original], "terminal_route": opened,
                 "route_endpoints": endpoints},
    )
    assert definition["parameters"]["from_label"].startswith("Library, 33,")
    assert definition["parameters"]["to_label"].startswith("Library, 400,")
    assert definition["parameters"]["route_summary"] == "Distance: 965m. Time: 0:02."
    assert definition["completion_events"][0]["query_params"] == {
        "engine": ["fossgis_osrm_car"], "route": ["2,1;4,3"]
    }
    locations = {item["field"]: item["location"] for item in evidence}
    assert locations["/task_definition/parameters/route_summary"] == (
        "/terminal_route/observation/summary_target/text"
    )
    assert locations["/task_definition/completion_events/0/query_params/route"] == (
        "/terminal_route/navigation_request/query_params/route"
    )
    assert any(check["parameter"] == "route_summary" for check in checks)


@pytest.mark.parametrize(
    "fault,error",
    [
        (None, None),
        ("terminal_coordinate_projection", None),
        ("document_503", "document GET200"),
        ("fetch", "document GET200"),
        ("wrong_document_route", "document GET200"),
        ("projected_document_route", "document GET200"),
        ("backend_503", "backend HTTP200"),
        ("partial_instructions", "backend HTTP200"),
        ("wrong_backend_route", "endpoint coordinates"),
        ("wrong_input", "endpoint coordinates"),
        ("wrong_profile", "selected profile"),
        ("wrong_final_page", "endpoint coordinates"),
        ("wrong_final_origin", "endpoint coordinates"),
        ("wrong_final_engine", "endpoint coordinates"),
        ("wrong_final_coordinates", "endpoint coordinates"),
        ("duplicate_final_engine", "endpoint coordinates"),
        ("duplicate_final_route", "endpoint coordinates"),
        ("blank_final_query", "endpoint coordinates"),
        ("different_generated_query", "different route link queries"),
        ("different_viewport_route", "shared-route facts"),
        ("ui_timeout", "route panel wait timed out"),
    ],
)
def test_navigation_author_opens_ui_generated_route_and_preserves_strict_readiness(
    tmp_path, browser, monkeypatch, directions, fault, error
):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "fresh_from_image": True, "owned_run": "test-map", "site_url": "http://localhost:8080"
    }))
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({
        "view": {"start_path": "/#map=5/51.5/-0.1"},
        "tasks": [{
            "generated_task_id": "wavgen-test-map-shared-route",
            "task_id": 6107201,
            "intent_template_id": 6109021,
            "family": "navigate-directions",
            "mode": "navigate-directions",
            "parameters": {
                "from_label": "visible origin", "to_label": "visible destination",
                "engine_labels": ["Car (OSRM)"],
                "route_link_zoom": 17,
            },
            "source_artifact": "sites/google_map/tasks.jsonl",
            "source_record_id": "Google Map--19",
        }],
    }))
    visits, waits = [], []

    def goto(page, url, **kwargs):
        visits.append((url, kwargs))
        page.url = url
        actual_url = url.replace("2%2C1", "9%2C9") if fault == "wrong_document_route" else url
        if fault == "projected_document_route":
            actual_url = url.replace("2%2C1", "2.00000%2C1.00000")
        return SimpleNamespace(
            url=actual_url.split("#")[0],
            status=503 if fault == "document_503" else 200,
            request=SimpleNamespace(
                method="GET", resource_type="xhr" if fault == "fetch" else "document",
                is_navigation_request=lambda: fault != "fetch",
            ),
        )

    def wait_for_function(page, expression, **kwargs):
        waits.append((expression, kwargs))
        if fault == "ui_timeout":
            raise ValueError("route panel wait timed out")

    def snapshot(page, output, name, responses):
        result = deepcopy(directions["observation"])
        result["route_readiness"] = {
            "valid_visible_directions_and_backend": fault != "backend_503",
            "complete_instruction_row_coverage": fault != "partial_instructions",
        }
        if fault == "wrong_backend_route":
            result["backend_request"]["url"] = result["backend_request"]["url"].replace("1,2", "9,9")
        if fault == "wrong_input":
            result["controls"][0]["value"] = "9, 9"
        if fault == "wrong_profile":
            result["controls"][2]["options"][0]["label"] = "Foot (OSRM)"
        if fault == "wrong_final_page":
            result["url"] = result["url"].replace("/directions", "/search")
        if fault == "wrong_final_origin":
            result["url"] = result["url"].replace("http://localhost:8080", "http://other:8080")
        if fault == "wrong_final_engine":
            result["url"] = result["url"].replace("fossgis_osrm_car", "fossgis_osrm_foot")
        if fault == "wrong_final_coordinates":
            result["url"] = result["url"].replace("2%2C1", "9%2C9")
        if fault in {"duplicate_final_engine", "duplicate_final_route", "blank_final_query"}:
            query = {
                "duplicate_final_engine": "engine=fossgis_osrm_car",
                "duplicate_final_route": "route=2%2C1%3B4%2C3",
                "blank_final_query": "extra=",
            }[fault]
            result["url"] = result["url"].split("#")[0] + "&" + query
        if fault == "terminal_coordinate_projection":
            result["url"] = result["url"].replace(
                "2%2C1%3B4%2C3", "2.00000%2C1.00000%3B4.00000%2C3.00000"
            )
            result["controls"][0]["value"] = "2.00000, 1.00000"
            result["controls"][1]["value"] = "4.00000, 3.00000"
            result["summary_target"]["text"] = "Distance: 965m. Time: 0:02."
        if fault == "different_viewport_route" and name == "comparison_terminal_route":
            result["summary_target"]["text"] = "Distance: 1.3km. Time: 1:05."
        return result

    monkeypatch.setattr(FakePage, "goto", goto)
    monkeypatch.setattr(
        FakePage, "reload", lambda page, **kwargs: goto(page, page.url, **kwargs), raising=False
    )
    monkeypatch.setattr(FakePage, "wait_for_function", wait_for_function, raising=False)
    def collect(*args, **kwargs):
        assert kwargs["route_link_zoom"] == 17
        value = deepcopy(directions)
        if kwargs["name_prefix"] == "comparison_route":
            value["observation"]["url"] = value["observation"]["url"].split("#")[0] + "#map=17/3/4"
            if fault == "different_generated_query":
                value["observation"]["url"] = value["observation"]["url"].replace("2%2C1", "9%2C9")
        return [value]

    monkeypatch.setattr(author, "_collect_routes", collect)
    monkeypatch.setattr(author, "_route_snapshot", snapshot)
    output = tmp_path / "outputs"
    argv = common_args(output) + [
        "--baseline-receipt", str(baseline), "--task-config", str(config),
        "--generated-task-id", "wavgen-test-map-shared-route",
    ]
    directory = output / "author/wavgen-test-map-shared-route/grounding-01"
    if error:
        with pytest.raises(ValueError, match=error):
            author.main(argv)
        assert not (directory / "draft.json").exists()
        observed = json.loads((directory / "observation.json").read_text())
        assert observed["failure"]
        if fault == "ui_timeout":
            assert "observation" in observed["terminal_route"]
    else:
        author.main(argv)
        observed = json.loads((directory / "observation.json").read_text())
        shared_url = directions["observation"]["url"]
        comparison_url = observed["comparison_routes"][0]["observation"]["url"]
        assert shared_url in [url for url, _ in visits]
        assert visits[-1][0] == comparison_url
        assert all(options["wait_until"] == "domcontentloaded" for _, options in visits[-2:])
        assert len(waits) == 2
        assert all(0 < options["timeout"] <= 120_000 for _, options in waits)
        assert all("Reach destination" in expression for expression, _ in waits)
        assert observed["terminal_route"]["source_url"] == shared_url
        assert observed["comparison_terminal_route"]["source_url"] == comparison_url
        assert observed["comparison_terminal_route"]["source_pointer"] == (
            "/comparison_routes/0/observation/url"
        )
        assert observed["terminal_route"]["navigation_request"]["resource_type"] == "document"
        assert observed["shared_route_viewport_comparison"][0]["same_semantic_route"]
        draft = json.loads((directory / "draft.json").read_text())
        assert draft["grounding"]["artifacts"]["screenshot"].endswith("terminal_route.png")
        if fault == "terminal_coordinate_projection":
            assert draft["task_definition"]["parameters"]["route_summary"] == (
                "Distance: 965m. Time: 0:02."
            )
            assert draft["task_definition"]["parameters"]["from_input"] == "2.00000, 1.00000"
            assert draft["task_definition"]["completion_events"][0]["query_params"] == {
                "engine": ["fossgis_osrm_car"], "route": ["2,1;4,3"]
            }
            summary_evidence = next(
                e for e in draft["grounding"]["field_evidence"]
                if e["field"] == "/task_definition/parameters/route_summary"
            )
            assert summary_evidence["location"] == (
                "/terminal_route/observation/summary_target/text"
            )
        for evidence in draft["grounding"]["field_evidence"]:
            source = observed
            for part in evidence["location"].strip("/").split("/"):
                source = source[int(part)] if isinstance(source, list) else source[part]


@pytest.mark.parametrize(
    "backend_status,complete,coordinate_projection",
    [(200, True, False), (503, False, False), (200, False, False), (200, True, True)],
)
def test_shared_route_wait_uses_real_panel_and_backend_while_tiles_stall(
    tmp_path, monkeypatch, directions, backend_status, complete, coordinate_projection
):
    from playwright.sync_api import BrowserType, TimeoutError, sync_playwright

    executable = Path("/root/.cache/ms-playwright/chromium-1117/chrome-linux/chrome")
    if not executable.is_file():
        pytest.skip("Existing offline Chromium is unavailable; no browser installation performed")
    release_tiles, tile_started = threading.Event(), threading.Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path)
            requests.append(self.path)
            if path.path == "/tile.png":
                tile_started.set()
                release_tiles.wait(10)
                return
            if path.path.startswith("/osrm/"):
                time.sleep(0.15)
                body, status = b"route response", backend_status
            elif path.path == "/directions" and path.query:
                # The UI starts a real route request and an unrelated persistent tile.
                # The destination row arrives later than the summary and first row.
                from_input = "2.00000, 1.00000" if coordinate_projection else "2, 1"
                to_input = "4.00000, 3.00000" if coordinate_projection else "4, 3"
                summary = (
                    "Distance: 965m. Time: 0:02." if coordinate_projection
                    else "Distance: 1.2km. Time: 1:05."
                )
                body = f"""<!doctype html><title>OpenStreetMap</title>
                  <input name="route_from" value="{from_input}">
                  <input name="route_to" value="{to_input}">
                  <select name="routing_engines"><option value="1">Car (OSRM)</option></select>
                  <div id="panel"></div><script>
                  if ({str(coordinate_projection).lower()}) history.replaceState(null, '',
                    '/directions?engine=fossgis_osrm_car&route='
                    + '2.00000%2C1.00000%3B4.00000%2C3.00000');
                  fetch('/tile.png').catch(() => {{}});
                  fetch('/osrm/routed-car/route/v1/driving/1,2;3,4?steps=true').then(async r => {{
                    await r.text();
                    if (!r.ok) {{
                      panel.innerHTML = '<p class="search_results_error">Couldn\\'t find a route</p>';
                      return;
                    }}
                    panel.innerHTML = '<h2>Directions</h2><p>{summary}</p>'
                      + '<table><tr><td>1. Start</td><td>2m</td></tr></table>';
                    if ({str(complete).lower()}) setTimeout(() => {{
                      panel.querySelector('table').insertAdjacentHTML('beforeend',
                        '<tr><td>2. Reach destination</td><td></td></tr>');
                    }}, 100);
                  }});
                  </script>""".encode()
                status = 200
            else:
                body, status = b"<!doctype html><title>OpenStreetMap</title><h1>Map</h1>", 200
            self.send_response(status)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    actual = deepcopy(directions)
    actual["observation"]["url"] = directions["observation"]["url"].replace(
        "http://localhost:8080", origin
    )
    launch = BrowserType.launch
    monkeypatch.setattr(
        BrowserType, "launch",
        lambda browser_type, **kwargs: launch(
            browser_type, executable_path=str(executable), **kwargs
        ),
    )
    monkeypatch.setattr(author, "_collect_routes", lambda *args, **kwargs: [deepcopy(actual)])
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "fresh_from_image": True, "owned_run": "offline-test-map", "site_url": origin,
    }))
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"view": {"start_path": "/"}, "tasks": [{
        "generated_task_id": "wavgen-test-map-shared-route",
        "task_id": 6107201, "intent_template_id": 6109021,
        "family": "navigate-directions", "mode": "navigate-directions",
        "parameters": {"from_label": "visible origin", "to_label": "visible destination",
                       "engine_labels": ["Car (OSRM)"], "route_link_zoom": 17},
        "source_artifact": "sites/google_map/tasks.jsonl", "source_record_id": "Google Map--19",
    }]}))
    output = tmp_path / "outputs"
    argv = common_args(output) + [
        "--baseline-receipt", str(baseline), "--task-config", str(config),
        "--generated-task-id", "wavgen-test-map-shared-route", "--timeout-ms", "1500",
    ]
    argv[argv.index("--site-url") + 1] = origin
    directory = output / "author/wavgen-test-map-shared-route/grounding-01"
    try:
        if backend_status == 200 and complete:
            with sync_playwright() as runtime:
                probe = runtime.chromium.launch(headless=True)
                page = probe.new_page()
                page.goto(actual["observation"]["url"], wait_until="domcontentloaded", timeout=1500)
                assert tile_started.wait(1)
                with pytest.raises(TimeoutError):
                    page.wait_for_load_state("networkidle", timeout=250)
                probe.close()
            requests.clear()
            author.main(argv)
            observed = json.loads((directory / "observation.json").read_text())
            for name in ("terminal_route", "comparison_terminal_route"):
                terminal = observed[name]
                assert terminal["navigation_request"]["status"] == 200
                assert terminal["observation"]["backend_request"]["status"] == 200
                assert terminal["observation"]["route_readiness"][
                    "complete_instruction_row_coverage"
                ]
                assert len(terminal["observation"]["instruction_coverage"]) == 2
            assert (directory / "draft.json").exists()
            if coordinate_projection:
                draft = json.loads((directory / "draft.json").read_text())
                assert draft["task_definition"]["parameters"]["route_summary"] == (
                    "Distance: 965m. Time: 0:02."
                )
                assert draft["task_definition"]["completion_events"][0]["query_params"] == {
                    "engine": ["fossgis_osrm_car"], "route": ["2,1;4,3"]
                }
        else:
            exception = ValueError if backend_status == 503 else TimeoutError
            with pytest.raises(exception):
                author.main(argv)
            observed = json.loads((directory / "observation.json").read_text())
            assert not (directory / "draft.json").exists()
            terminal = observed["terminal_route"]["observation"]
            assert terminal["route_readiness"]["backend_responses"][-1]["status"] == backend_status
            if backend_status == 503:
                assert not terminal["route_readiness"]["valid_visible_directions_and_backend"]
                assert "Couldn't find a route" in terminal["body_text"]
            else:
                assert "Reach destination" not in terminal["instructions_target"]["text"]
        shared_visits = [path for path in requests if path.startswith("/directions?")]
        backend_visits = [path for path in requests if path.startswith("/osrm/")]
        assert len(shared_visits) == len(backend_visits) == (2 if complete else 1)
        assert tile_started.is_set() and not release_tiles.is_set()
    finally:
        release_tiles.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "route_link_zoom,error",
    [(None, KeyError), (16, ValueError), (True, ValueError), ("17", ValueError)],
)
def test_navigation_requires_explicit_zoom17_config_before_browser(
    tmp_path, route_link_zoom, error
):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "fresh_from_image": True, "owned_run": "test-map", "site_url": "http://localhost:8080",
    }))
    parameters = {"from_label": "origin", "to_label": "destination", "engine_labels": ["Car (OSRM)"]}
    if route_link_zoom is not None:
        parameters["route_link_zoom"] = route_link_zoom
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"view": {"start_path": "/"}, "tasks": [{
        "generated_task_id": "wavgen-test-map-zoom", "mode": "navigate-directions",
        "parameters": parameters,
    }]}))
    with pytest.raises(error):
        author.main(common_args(tmp_path / "outputs") + [
            "--baseline-receipt", str(baseline), "--task-config", str(config),
            "--generated-task-id", "wavgen-test-map-zoom",
        ])
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    "fault,go_keeps_map_view", [
        (None, False), ("zoom_does_not_move", False),
        ("post_go_zoom_does_not_move", False), ("different_generated_query", False),
        (None, True), ("hash_stays_missing", True),
    ]
)
def test_zoom17_route_links_use_visible_controls_and_independent_viewports(
    tmp_path, monkeypatch, fault, go_keeps_map_view
):
    from playwright.sync_api import BrowserType, TimeoutError

    executable = Path("/root/.cache/ms-playwright/chromium-1117/chrome-linux/chrome")
    if not executable.is_file():
        pytest.skip("Existing offline Chromium is unavailable; no installation performed")
    requests = []
    from_label, to_label = "visible origin", "visible destination"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            path = urlsplit(self.path)
            if path.path.startswith("/osrm/"):
                body, mime = b'{"code":"Ok"}', "application/json"
            elif path.path == "/directions":
                body = f"""<!doctype html><title>Map</title>
                  <form class="directions_form">
                    <input id="route_from" name="route_from">
                    <input id="route_to" name="route_to">
                    <select name="routing_engines"><option value="1">Car (OSRM)</option></select>
                    <input type="submit" name="commit" value="Go">
                  </form>
                  <a class="control-button zoomin" href="#" aria-label="Zoom In">+</a>
                  <a class="control-button zoomout" href="#" aria-label="Zoom Out">−</a>
                  <div id="panel"></div><script>
                  let zoom = innerWidth > 1500 ? 5 : 17;
                  const query = new URLSearchParams(location.search);
                  let points = [[2.123456,1.123456],[4.123456,3.123456]];
                  const opened = query.has('route');
                  if (opened) {{
                    points = query.get('route').split(';').map(p=>p.split(',').map(Number));
                    zoom = Number(new URLSearchParams(location.hash.slice(1)).get('map')
                      ?.split('/')[0] || 17);
                  }}
                  const from = document.getElementById('route_from');
                  const to = document.getElementById('route_to');
                  from.value = opened ? points[0].map(v=>v.toFixed(5)).join(', ') : '';
                  to.value = opened ? points[1].map(v=>v.toFixed(5)).join(', ') : '';
                  function moveend() {{
                    history.replaceState(null,'',location.pathname+location.search+'#map='+zoom+'/0/0');
                    document.dispatchEvent(new Event('moveend'));
                  }}
                  moveend();
                  document.querySelectorAll('a.control-button').forEach(button=>{{
                    button.onclick = event=>{{
                      event.preventDefault();
                      if ({str(fault == 'hash_stays_missing').lower()} && count>1) return;
                      if (!{str(fault == 'zoom_does_not_move').lower()}
                          && !({str(fault == 'post_go_zoom_does_not_move').lower()} && count>1))
                        zoom += button.classList.contains('zoomin') ? 1 : -1;
                      moveend();
                    }};
                  }});
                  let count=0;
                  function route() {{
                    const precision=Math.max(0,Math.ceil(Math.log(zoom)/Math.LN2));
                    let encoded=points.map(p=>p.map(v=>v.toFixed(precision)).join(',')).join(';');
                    if ({str(fault == 'different_generated_query').lower()}
                        && innerWidth<=1500 && count>0) encoded=encoded.replace('2.12346','9.12346');
                    history.replaceState(null,'','/directions?engine=fossgis_osrm_car&route='
                      +encodeURIComponent(encoded));
                    panel.innerHTML='<p>Loading</p>';
                    const backend=points.map(p=>[p[1],p[0]].join(',')).join(';');
                    fetch('/osrm/routed-car/route/v1/driving/'+backend+'?steps=true')
                      .then(r=>r.json()).then(()=>{{
                        const summary=opened?'Distance: 965m. Time: 0:02.':'Distance: 856m. Time: 0:02.';
                        panel.innerHTML='<h2>Directions</h2><p>'+summary+'</p>'
                          +'<table><tr><td>1. Start</td><td>2m</td></tr>'
                          +'<tr><td>2. Reach destination</td><td></td></tr></table>';
                        if (!opened && (!{str(go_keeps_map_view).lower()} || count===0)) {{
                          zoom=innerWidth>1500?18:16;moveend();
                        }}
                        count++;
                      }});
                  }}
                  document.querySelector('form').onsubmit=event=>{{event.preventDefault();route();}};
                  if (opened) route();
                  </script>""".encode()
                mime = "text/html"
            else:
                body, mime = b'<!doctype html><title>Map</title><a href="/directions">Directions</a>', "text/html"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    launch = BrowserType.launch
    monkeypatch.setattr(
        BrowserType, "launch",
        lambda browser_type, **kwargs: launch(browser_type, executable_path=str(executable), **kwargs),
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"fresh_from_image": True, "owned_run": "test-map", "site_url": origin}))
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"view": {"start_path": "/#map=5/0/0"}, "tasks": [{
        "generated_task_id": "wavgen-test-map-zoom", "task_id": 6107301,
        "intent_template_id": 6109101, "family": "navigate-directions", "mode": "navigate-directions",
        "parameters": {"from_label": from_label, "to_label": to_label,
                       "engine_labels": ["Car (OSRM)"], "route_link_zoom": 17},
        "source_artifact": "sites/google_map/tasks.jsonl", "source_record_id": "Google Map--19",
    }]}))
    output = tmp_path / "outputs"
    argv = common_args(output) + [
        "--baseline-receipt", str(baseline), "--task-config", str(config),
        "--generated-task-id", "wavgen-test-map-zoom", "--timeout-ms", "1500",
        "--settle-seconds", "0",
    ]
    argv[argv.index("--site-url") + 1] = origin
    directory = output / "author/wavgen-test-map-zoom/grounding-01"
    try:
        if fault:
            with pytest.raises(
                TimeoutError if fault in {
                    "zoom_does_not_move", "post_go_zoom_does_not_move", "hash_stays_missing"
                }
                else ValueError
            ):
                author.main(argv)
            assert not (directory / "draft.json").exists()
        else:
            author.main(argv)
        observed = json.loads((directory / "observation.json").read_text())
        if fault in {"post_go_zoom_does_not_move", "hash_stays_missing"}:
            assert len([path for path in requests if "1.123456,2.123456" in path]) == 2
            assert "terminal_route" not in observed
        if fault == "different_generated_query":
            assert not observed["route_link_viewport_comparison"]["same_generated_query"]
            assert "terminal_route" not in observed
        if not fault:
            assert observed["route_link_viewport_comparison"]["same_generated_query"]
            for name, click_selector in (("routes", "zoomout"), ("comparison_routes", "zoomin")):
                generation = observed[name][0]["route_link_generation"]
                assert generation["zoom"] == 17
                assert "#map=17/" in generation["zoom_url"]
                assert click_selector in generation["zoom_clicks"][0]["selector"]
                restoration = generation["post_recompute_zoom"]
                assert restoration["zoom"] == 17
                assert "#map=17/" in restoration["after_url"]
                assert restoration["before_query"] == restoration["after_query"]
                if go_keeps_map_view:
                    assert restoration["before_fragment"] == ""
                    assert "#" not in restoration["before_url"]
                    recovery, restore = restoration["zoom_clicks"]
                    assert "zoomout" in recovery["selector"]
                    assert recovery["before_zoom"] is None
                    assert recovery["after_zoom"] == restore["before_zoom"] == 16
                    assert "#map=16/" in recovery["after_url"]
                    assert "zoomin" in restore["selector"]
                    assert restore["after_zoom"] == 17
                else:
                    assert click_selector in restoration["zoom_clicks"][0]["selector"]
                assert "#map=17/" in observed[name][0]["observation"]["url"]
                assert observed[name][0]["observation"]["route_readiness"][
                    "complete_instruction_row_coverage"
                ]
                assert observed[name][0]["observation"]["summary_target"]["text"] == (
                    "Distance: 856m. Time: 0:02."
                )
            assert observed["routes"][0]["route_link_generation"]["initial_observation"]["url"] != (
                observed["comparison_routes"][0]["route_link_generation"]["initial_observation"]["url"]
            )
            assert len([path for path in requests if "1.123456,2.123456" in path]) == 4
            for terminal, source in (("terminal_route", "routes"),
                                     ("comparison_terminal_route", "comparison_routes")):
                assert observed[terminal]["source_url"] == observed[source][0]["observation"]["url"]
                assert observed[terminal]["source_pointer"] == f"/{source}/0/observation/url"
            draft = json.loads((directory / "draft.json").read_text())
            assert draft["task_definition"]["completion_events"][0]["query_params"] == {
                "engine": ["fossgis_osrm_car"], "route": ["2.12346,1.12346;4.12346,3.12346"]
            }
            assert draft["task_definition"]["parameters"]["route_summary"] == "Distance: 965m. Time: 0:02."
            assert draft["task_definition"]["parameters"]["route_link_zoom"] == 17
            assert "{route_link_zoom}" in draft["task_definition"]["goal_template"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "document_url,document_status,later_document,expected",
    [
        ("/directions?engine=fossgis_osrm_car&route=2%2C1%3B4%2C3", 200, False, 1.0),
        ("/directions", 200, False, 0.0),
        ("/directions?engine=fossgis_osrm_foot&route=2%2C1%3B4%2C3", 200, False, 0.0),
        ("/directions?engine=fossgis_osrm_car&route=2%2C1%3B9%2C9", 200, False, 0.0),
        ("/directions?engine=fossgis_osrm_car&route=4%2C3%3B2%2C1", 200, False, 0.0),
        ("/directions?engine=fossgis_osrm_car&route=2%2C1%3B4%2C3", 503, False, 0.0),
        ("/directions?engine=fossgis_osrm_car&route=2%2C1%3B4%2C3", 200, True, 0.0),
        (None, 200, False, 0.0),
    ],
)
def test_native_navigation_contract_requires_final_document_engine_and_both_endpoints(
    directions, document_url, document_status, later_document, expected
):
    pytest.importorskip("webarena_verified")
    from author import _task_spec
    from replay import _evaluate
    from webarena_verified.types import WebArenaSite, WebArenaVerifiedTask
    from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
    from webarena_verified.types.tracing import NetworkTrace

    definition, _, _ = author.build_extension_definition(
        {"mode": "navigate-directions", "parameters": {
            "from_label": "visible origin", "to_label": "visible destination",
            "engine_labels": ["Car (OSRM)"], "route_link_zoom": 17,
        }},
        {"searches": [], "routes": [directions], "terminal_route": directions, "route_endpoints": {
            "from_label": "visible origin", "to_label": "visible destination",
        }},
    )
    spec, response = _task_spec({
        "task_definition": definition, "site": "map", "task_id": 6107201,
        "intent_template_id": 6109021, "start_url": "__MAP__/", "revision": 1,
    })
    task = WebArenaVerifiedTask.model_validate(spec)
    config = WebArenaVerifiedConfig(
        test_data_file=__file__,
        environments={WebArenaSite.MAP: EnvironmentConfig(urls=["http://localhost:8080"])},
    )

    def event(url, status, accept):
        return {
            "request": {"method": "GET", "url": "http://localhost:8080" + url,
                        "headers": [{"name": "Accept", "value": accept}]},
            "response": {"status": status},
        }

    # Contract fixtures only: no network execution or replacement author evidence.
    entries = []
    if document_url:
        entries.append(event(document_url, document_status, "text/html"))
    entries.append(event("/osrm/routed-car/route/v1/driving/1,2;3,4?steps=true", 200, "application/json"))
    if later_document:
        entries.append(event("/search", 200, "text/html"))
    trace = NetworkTrace.from_content(entries)
    assert [row["score"] for row in _evaluate(task, json.dumps(response), trace, config)] == [1.0, expected]
    for field, wrong in (("retrieved_data", ["unexpected"]), ("status", "UNKNOWN_ERROR"), ("task_type", "RETRIEVE")):
        contrast = {**response, field: wrong}
        assert [row["score"] for row in _evaluate(task, json.dumps(contrast), trace, config)] == [0.0, expected]


def test_region_summary_preserves_query_region_order_and_visible_absence(visible_records):
    definition, _, _ = author.build_extension_definition(
        {
            "mode": "retrieve-multi-query-region-summary",
            "parameters": {
                "queries": ["broad", "region"],
                "regions": ["State", "County X"],
                "category": "Library",
            },
        },
        {**visible_records, "searches": visible_records["searches"][:2]},
    )
    assert definition["answer"]["values"] == [
        {
            "query": "broad",
            "region": "State",
            "result_count": 5,
            "distinct_category_count": 2,
            "category_count": 4,
        },
        {
            "query": "broad",
            "region": "County X",
            "result_count": 2,
            "distinct_category_count": 2,
            "category_count": 0,
        },
        {
            "query": "region",
            "region": "State",
            "result_count": 5,
            "distinct_category_count": 1,
            "category_count": 5,
        },
        {
            "query": "region",
            "region": "County X",
            "result_count": 0,
            "distinct_category_count": 0,
            "category_count": 0,
        },
    ]


def test_numeric_maximum_ties_choose_alphabetically_first_complete_label(visible_records):
    rows = visible_records["searches"][0]["batches"][0]["snapshot"]["results"]
    extra = dict(
        rows[2],
        label="Library, 400, A Street, City, County, State, 11111, US",
        url="http://localhost:8080/node/9",
    )
    rows.append(extra)
    selected = author._select_route_endpoints(
        {
            "mode": "navigate-numeric-extremes-directions",
            "parameters": {"query": "broad", "category": "Library", "address_components": ["City"]},
        },
        visible_records,
    )
    assert selected["to_label"] == extra["label"]
    definition, _, _ = author.build_extension_definition(
        {
            "mode": "retrieve-record-address-extreme-comparison",
            "parameters": {"queries": ["broad", "region"], "category": "Library", "region": "City"},
        },
        visible_records,
    )
    assert definition["answer"]["values"][0]["higher_number_label"] == extra["label"]


@pytest.mark.parametrize(
    "complete,strict,raises", [(True, True, False), (False, True, True), (False, False, False)]
)
def test_route_gui_collection_requires_full_instruction_evidence_for_grounding(
    tmp_path, monkeypatch, complete, strict, raises
):
    fields = {"route_from": "", "route_to": ""}
    controls = lambda: (
        [
            {"name": name, "value": value, "form": {"method": "get"}}
            for name, value in fields.items()
        ]
        + [
            {
                "name": "routing_engines",
                "value": "1",
                "form": {"method": "get"},
                "options": [{"label": "Car (OSRM)", "value": "1", "selected": True}],
            }
        ]
    )
    page = SimpleNamespace(
        goto=lambda *args, **kwargs: None,
        wait_for_timeout=lambda value: None,
        wait_for_load_state=lambda *args, **kwargs: None,
    )
    page.locator = lambda selector: SimpleNamespace(
        fill=lambda value: fields.__setitem__(
            "route_from" if "route_from" in selector else "route_to", value
        ),
        select_option=lambda **kwargs: None,
        press=lambda key: None,
    )
    monkeypatch.setattr(author, "_record", lambda *args: {"controls": controls()})
    monkeypatch.setattr(
        author,
        "_route_snapshot",
        lambda *args: {
            "controls": controls(),
            "route_readiness": {
                "valid_visible_directions_and_backend": True,
                "complete_instruction_row_coverage": complete,
            },
        },
    )
    args = (
        page,
        tmp_path,
        {"links": [{"url": "http://localhost:8080/directions"}]},
        "http://localhost:8080",
        "visible origin",
        "visible destination",
        ["Car (OSRM)"],
        [],
        1000,
        0,
    )
    if raises:
        with pytest.raises(ValueError, match="Fresh Directions"):
            author._collect_routes(*args, strict=strict, name_prefix="route")
        assert (tmp_path / "route_failure.json").is_file()
    else:
        routes = author._collect_routes(*args, strict=strict, name_prefix="route")
        assert routes[0]["observation"]["route_readiness"]["exact_endpoint_and_profile_inputs"]


def test_route_viewport_comparison_ignores_map_fragment_but_requires_actual_semantic_facts(
    directions,
):
    first, second = deepcopy(directions), deepcopy(directions)
    first["observation"]["url"] = "http://localhost:8080/directions#map=17/1/2"
    second["observation"]["url"] = "http://localhost:8080/directions#map=16/3/4"
    assert author._compare_route_recordings([first], [second])[0]["same_semantic_route"]
    second["observation"]["summary_target"]["text"] = "Distance: 2km. Time: 0:05."
    assert not author._compare_route_recordings([first], [second])[0]["same_semantic_route"]
    second = deepcopy(directions)
    second["to_query"] = "another destination"
    assert not author._compare_route_recordings([first], [second])[0]["same_semantic_route"]


def test_directions_raw_text_projection_matches_exact_runtime_gate(directions, monkeypatch):
    from itertools import count

    import generated_env
    from author import _task_spec

    raw = "\t1. Start\t20m\n\n\t2. Reach destination\t\n"
    directions["observation"]["instructions_target"]["text"] = raw
    definition, _, checks = author.build_extension_definition(
        {
            "mode": "navigate-directions",
            "parameters": {
                "from_label": "visible origin",
                "to_label": "visible destination",
                "engine_labels": ["Car (OSRM)"],
                "route_link_zoom": 17,
            },
        },
        {
            "searches": [],
            "routes": [directions],
            "terminal_route": directions,
            "route_endpoints": {"from_label": "visible origin", "to_label": "visible destination"},
        },
    )
    task, _ = _task_spec(
        {
            "task_definition": definition,
            "site": "map",
            "task_id": 6107001,
            "intent_template_id": 6108021,
            "start_url": "__MAP__/",
            "revision": 1,
        }
    )
    values = {check["selector"]: definition["parameters"][check["parameter"]] for check in checks}
    elements = {
        selector: SimpleNamespace(
            is_visible=lambda: True,
            inner_text=lambda value=value: value,
            input_value=lambda value=value: value,
        )
        for selector, value in values.items()
    }
    elements["body"] = SimpleNamespace(inner_text=lambda: raw)
    page = SimpleNamespace(
        url="http://localhost:8080/directions#map=16/3/4",
        title=lambda: "OpenStreetMap",
        locator=lambda selector: SimpleNamespace(
            all=lambda: [elements[selector]], inner_text=elements[selector].inner_text
        ),
        wait_for_timeout=lambda value: None,
    )
    candidate = {
        "task_spec": task,
        "task_definition": definition,
        "grounding": {"final_state_checks": checks},
    }
    assert generated_env.audit_final_state(candidate, {"map": "http://localhost:8080"}, page)[
        "confirmed"
    ]
    ticks = count(0, 10)
    monkeypatch.setattr(generated_env, "monotonic", lambda: next(ticks))
    definition["parameters"]["route_instructions"] = raw.strip()
    audit = generated_env.audit_final_state(candidate, {"map": "http://localhost:8080"}, page)
    assert not audit["confirmed"]
    assert not audit["checks"][-1]["passed"]
