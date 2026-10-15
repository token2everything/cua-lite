"""Offline cross-site evidence probes; no site, model or catalog is accessed."""

from __future__ import annotations

import json
import os
import zipfile
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from contract import (
    ContractError,
    public_projection,
    sha256_file,
    sha256_json,
    validate_candidate_shape,
)
from cross_site_evaluator import (
    capture_gui_frame,
    derive_value,
    evaluate_dependencies,
    evaluate_recording,
    parse_fact_value,
    record_gui_frame,
    seal_recording,
    validate_dependencies,
)
from generated_env import resolve_public_task

pytest_plugins = ["test_episode_applicability"]


def _fixture(runtime):
    from webarena_verified.types.tracing import NetworkTrace

    parameters = {
        "source_url": "__SHOPPING__/cart",
        "entity_url": "__SHOPPING__/item-7.html",
        "entity_label": "Invented product",
        "price": "$20.00",
        "note": "Invented product costs $20.00",
        "issue_url": "__GITLAB__/p/issues/1",
    }
    definition = {
        "task_type": "MUTATE",
        "goal_template": "Read the cart and record its item price.",
        "parameters": parameters,
        "answer": {"status": "SUCCESS", "values": None, "item_schema": None, "ordered": False},
        "completion_events": [
            {
                "url_template": "__GITLAB__/notes",
                "http_method": "POST",
                "response_status": 200,
                "post_data": {"note": "{note}"},
            }
        ],
        "dependencies": [
            {
                "name": "cart_to_issue",
                "source_site": "shopping",
                "target_site": "gitlab",
                "source_entity": {
                    "identity_parameter": "entity_url",
                    "label_parameter": "entity_label",
                },
                "source_url_parameter": "source_url",
                "fact_readouts": {
                    "entity_label": {"kind": "text", "selector": "a", "value_type": "string"},
                    "price": {"kind": "text", "selector": ".price", "value_type": "string"},
                },
                "target_event_index": 0,
            }
        ],
    }
    response = {
        "task_type": "MUTATE",
        "status": "SUCCESS",
        "retrieved_data": None,
        "error_details": None,
    }
    candidate = {
        "generated_task_id": "wavgen-cross-fixture",
        "generation_mode": "direct",
        "source": {"artifact_sha256": "fixture", "record_sha256": "fixture"},
        "task_definition": definition,
        "reference_response": response,
        "grounding": {
            "field_evidence": [
                {
                    "field": "/task_definition/parameters/note",
                    "rule": {
                        "operation": "format",
                        "parameters": ["entity_label", "price"],
                        "template": "{entity_label} costs {price}",
                    },
                }
            ]
        },
        "baseline": {
            "sites": {
                site: {"snapshot": "fixture", "image_digest": "sha256:fixture", "site_url": url}
                for site, url in {
                    "shopping": "http://shopping.test",
                    "gitlab": "http://gitlab.test",
                }.items()
            }
        },
        "task_spec": {
            "task_id": 2000000,
            "intent_template_id": 2000000,
            "revision": 1,
            "sites": ["shopping", "gitlab"],
            "start_urls": ["__SHOPPING__/cart", "__GITLAB__/p/issues/1"],
            "intent": definition["goal_template"],
            "intent_template": definition["goal_template"],
            "instantiation_dict": parameters,
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "expected": response,
                    "results_schema": {"type": "null"},
                },
                {
                    "evaluator": "NetworkEventEvaluator",
                    "last_event_only": False,
                    "expected": {
                        "url": "__GITLAB__/notes",
                        "http_method": "POST",
                        "response_status": 200,
                        "post_data": {"note": parameters["note"]},
                    },
                },
            ],
        },
    }
    source = {
        "startedDateTime": "2026-01-01T00:00:00+00:00",
        "time": 10,
        "timings": {"send": 1, "wait": 8, "receive": 1},
        "request": {
            "method": "GET",
            "url": "http://shopping.test/cart",
            "headers": [],
            "queryString": [],
        },
        "response": {
            "status": 200,
            "headers": [],
            "cookies": [],
            "redirectURL": "",
            "content": {
                "mimeType": "text/html",
                "text": '<table><tr><td><a href="/item-7.html">Invented product</a></td><td class="price">$20.00</td></tr></table>',
            },
        },
    }
    target = deepcopy(source)
    target.update(startedDateTime="2026-01-01T00:00:01+00:00")
    target["request"] = {
        "method": "POST",
        "url": "http://gitlab.test/notes",
        "headers": [],
        "queryString": [],
        "postData": {
            "mimeType": "application/x-www-form-urlencoded",
            "text": "note=Invented+product+costs+%2420.00",
        },
    }
    target["response"]["content"] = {"mimeType": "application/json", "text": "{}"}
    entries = [source, target]
    frames = [
        {
            "at_utc": "2026-01-01T00:00:00.500+00:00",
            "scope": "model_context_only",
            "context_id": "original",
            "page_url": "http://shopping.test/cart",
            "body_text": "Invented product $20.00",
            "visible_html": source["response"]["content"]["text"],
            "links": [
                {
                    "href": "http://shopping.test/item-7.html",
                    "text": "Invented product",
                    "body_text": "Invented product $20.00",
                    "visible_html": source["response"]["content"]["text"],
                }
            ],
            "screenshot_sha256": "fixture",
        }
    ]
    boundary = {
        "scope": "model_context_only",
        "recording_context_id": "original",
        "har_context_id": "original",
        "playwright_context_id": "original",
        "audit_context_ids": ["audit"],
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:02+00:00",
    }
    return candidate, entries, NetworkTrace, frames, boundary


def test_bodyless_browser_frame_is_empty_evidence_and_later_body_recovers(runtime, tmp_path):
    """Use installed Chromium; PLAYWRIGHT_CHROMIUM_EXECUTABLE selects an existing binary."""
    from playwright.sync_api import Error, sync_playwright

    candidate, entries, trace_type, _, boundary = _fixture(runtime)
    markup = entries[0]["response"]["content"]["text"]
    with sync_playwright() as playwright:
        executable = Path(os.environ.get(
            "PLAYWRIGHT_CHROMIUM_EXECUTABLE", playwright.chromium.executable_path
        ))
        if not executable.is_file():
            pytest.skip("An installed Chromium is required for the real DOM regression")
        browser = playwright.chromium.launch(executable_path=str(executable), headless=True)
        try:
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.route("**/*", lambda route: route.fulfill(
                status=200, content_type="text/html", body=markup
            ))
            page.goto("http://shopping.test/cart")
            page.evaluate("document.body.remove()")
            assert page.evaluate("document.body === null")
            page.screenshot(path=str(tmp_path / "empty.png"))
            record_gui_frame(
                tmp_path, page, "empty.png", "original", "model_context_only", Path(__file__)
            )
            assert page.evaluate("document.body === null")
            [empty] = [json.loads(line) for line in
                       (tmp_path / "cross_site_gui_frames.jsonl").read_text().splitlines()]
            assert empty["body_text"] == empty["visible_html"] == ""
            assert empty["links"] == []
            assert empty["screenshot_sha256"] == sha256_file(tmp_path / "empty.png")

            page.evaluate("""html => {
                const body = document.createElement('body');
                body.innerHTML = html;
                document.documentElement.appendChild(body);
            }""", markup)
            page.screenshot(path=str(tmp_path / "recovered.png"))
            record_gui_frame(
                tmp_path, page, "recovered.png", "original", "model_context_only", Path(__file__)
            )
            empty, recovered = [json.loads(line) for line in
                                (tmp_path / "cross_site_gui_frames.jsonl").read_text().splitlines()]
            assert "Invented product" in recovered["body_text"]
            assert "$20.00" in recovered["visible_html"]
            assert recovered["links"][0]["href"] == "http://shopping.test/item-7.html"
            assert recovered["screenshot_sha256"] == sha256_file(tmp_path / "recovered.png")

            # The real captured frame contents enter the existing dependency
            # gate with synthetic network windows enclosing both captures.
            source_time = datetime.fromisoformat(empty["at_utc"]) - timedelta(seconds=1)
            target_time = datetime.fromisoformat(recovered["at_utc"]) + timedelta(seconds=1)
            entries[0]["startedDateTime"] = source_time.isoformat()
            entries[1]["startedDateTime"] = target_time.isoformat()
            boundary["started_at"] = (source_time - timedelta(seconds=1)).isoformat()
            boundary["ended_at"] = (target_time + timedelta(seconds=1)).isoformat()
            sites = {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"}
            for frames, expected in [([empty], "fail"), ([empty, recovered], "pass")]:
                result = evaluate_dependencies(
                    candidate, trace_type.from_content(entries), sites, frames,
                    boundary, "model_context_only",
                )
                assert result["status"] == expected
            page.evaluate("""() => {
                document.createTreeWalker = () => { throw new Error('unrelated DOM failure'); };
            }""")
            with pytest.raises(Error, match="unrelated DOM failure"):
                record_gui_frame(
                    tmp_path, page, "recovered.png", "original", "model_context_only", Path(__file__)
                )
            assert len((tmp_path / "cross_site_gui_frames.jsonl").read_text().splitlines()) == 2
        finally:
            browser.close()


@pytest.mark.parametrize("change", ["hidden_template", "visible_price", "visible_input"])
def test_capture_tracks_visible_facts_without_hidden_template_drift(runtime, tmp_path, change):
    """Hidden template churn preserves evidence; visible fact races reject it."""
    from playwright.sync_api import sync_playwright

    candidate, entries, trace_type, _, boundary = _fixture(runtime)
    markup = entries[0]["response"]["content"]["text"]
    markup += '<input data-role="cart-item-qty" value="1">'
    markup += '<div id="template" style="position:absolute;top:2000px"><!-- ko --><span>Hidden</span></div>'
    with sync_playwright() as playwright:
        executable = Path(os.environ.get(
            "PLAYWRIGHT_CHROMIUM_EXECUTABLE", playwright.chromium.executable_path
        ))
        if not executable.is_file():
            pytest.skip("An installed Chromium is required for the real DOM regression")
        browser = playwright.chromium.launch(executable_path=str(executable), headless=True)
        try:
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.route("**/*", lambda route: route.fulfill(
                status=200, content_type="text/html", body=markup
            ))
            page.goto("http://shopping.test/cart")
            screenshots = []

            def capture_then_change():
                screenshots.append(page.screenshot(caret="initial"))
                if change == "hidden_template":
                    page.evaluate("""() => {
                        document.querySelector('#template').innerHTML =
                            '<!-- new ko --><div><span>Changed hidden content</span></div>';
                    }""")
                elif change == "visible_price":
                    page.locator('.price').evaluate("node => node.textContent = '$21.00'")
                else:
                    page.locator('input').evaluate("node => node.value = '2'")
                return screenshots[-1]

            image = capture_gui_frame(
                tmp_path, lambda: page, capture_then_change, "original.png",
                "original", "model_context_only", Path(__file__),
            )
            assert len(screenshots) == 1 and image == screenshots[0]
            assert (tmp_path / "original.png").read_bytes() == screenshots[0]
            [frame] = [json.loads(line) for line in
                       (tmp_path / "cross_site_gui_frames.jsonl").read_text().splitlines()]
            assert frame['screenshot_sha256'] == sha256_file(tmp_path / 'original.png')
            if change == "hidden_template":
                assert "$20.00" in frame["visible_html"]
                assert 'value="1"' in frame["visible_html"]
                assert "template" not in frame["visible_html"]
                assert "<!--" not in frame["visible_html"]
            else:
                assert frame["page_url"] == frame["visible_html"] == frame["body_text"] == ""
                assert frame["links"] == []
            frame_time = datetime.fromisoformat(frame["at_utc"])
            entries[0]["startedDateTime"] = (frame_time - timedelta(seconds=1)).isoformat()
            entries[1]["startedDateTime"] = (frame_time + timedelta(seconds=1)).isoformat()
            boundary["started_at"] = (frame_time - timedelta(seconds=2)).isoformat()
            boundary["ended_at"] = (frame_time + timedelta(seconds=2)).isoformat()
            result = evaluate_dependencies(
                candidate, trace_type.from_content(entries),
                {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"},
                [frame], boundary, "model_context_only",
            )
            assert result["status"] == ("pass" if change == "hidden_template" else "fail")
        finally:
            browser.close()


def test_capture_navigation_cannot_supply_source_facts_but_next_frame_can(runtime, tmp_path):
    from playwright.sync_api import sync_playwright

    candidate, entries, trace_type, _, boundary = _fixture(runtime)
    markup = entries[0]["response"]["content"]["text"]
    with sync_playwright() as playwright:
        executable = Path(
            os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE", playwright.chromium.executable_path)
        )
        if not executable.is_file():
            pytest.skip("An installed Chromium is required for the real DOM regression")
        browser = playwright.chromium.launch(executable_path=str(executable), headless=True)
        try:
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.route(
                "**/*",
                lambda route: route.fulfill(
                    status=200,
                    content_type="text/html",
                    body=markup
                    if route.request.url.endswith("/cart")
                    else "<body>Home minicart</body>",
                ),
            )
            page.goto("http://shopping.test/home")
            screenshots = []

            def capture_then_navigate():
                screenshots.append(page.screenshot())
                page.goto("http://shopping.test/cart")
                return screenshots[-1]

            image = capture_gui_frame(
                tmp_path,
                lambda: page,
                capture_then_navigate,
                "race.png",
                "original",
                "model_context_only",
                Path(__file__),
            )
            assert len(screenshots) == 1 and image == screenshots[0]
            assert (tmp_path / "race.png").read_bytes() == screenshots[0]
            capture_gui_frame(
                tmp_path,
                lambda: page,
                page.screenshot,
                "stable.png",
                "original",
                "model_context_only",
                Path(__file__),
            )
            race, stable = [
                json.loads(line)
                for line in (tmp_path / "cross_site_gui_frames.jsonl").read_text().splitlines()
            ]
            assert race["page_url"] == race["body_text"] == race["visible_html"] == ""
            assert race["links"] == []
            assert stable["page_url"] == "http://shopping.test/cart"
            assert "Invented product" in stable["body_text"]
            assert "$20.00" in stable["visible_html"]
            source_time = datetime.fromisoformat(race["at_utc"]) - timedelta(seconds=1)
            target_time = datetime.fromisoformat(stable["at_utc"]) + timedelta(seconds=1)
            entries[0]["startedDateTime"] = source_time.isoformat()
            entries[1]["startedDateTime"] = target_time.isoformat()
            boundary["started_at"] = (source_time - timedelta(seconds=1)).isoformat()
            boundary["ended_at"] = (target_time + timedelta(seconds=1)).isoformat()
            sites = {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"}
            for frames, expected in [([race], "fail"), ([race, stable], "pass")]:
                result = evaluate_dependencies(
                    candidate,
                    trace_type.from_content(entries),
                    sites,
                    frames,
                    boundary,
                    "model_context_only",
                )
                assert result["status"] == expected
        finally:
            browser.close()


def test_joint_contract_and_public_projection(runtime):
    candidate, *_ = _fixture(runtime)
    validate_candidate_shape(candidate)
    public = resolve_public_task(
        candidate, {"shopping": "http://fresh-shopping.test", "gitlab": "http://fresh-gitlab.test"}
    )
    assert public["start_urls"] == [
        "http://fresh-shopping.test/cart",
        "http://fresh-gitlab.test/p/issues/1",
    ]
    assert set(public) == {"generated_task_id", "sites", "start_urls", "intent"}
    assert "dependency" not in json.dumps(public)
    candidate["task_spec"]["sites"] = ["shopping", "shopping"]
    with pytest.raises(ContractError, match="distinct"):
        validate_candidate_shape(candidate)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing_source",
        "wrong_entity",
        "wrong_target",
        "order",
        "timing",
        "audit",
        "other_item_price",
        "missing_body",
        "offscreen_fact",
    ],
)
def test_source_entity_visible_fact_target_and_order(runtime, fault):
    candidate, entries, trace_type, frames, boundary = _fixture(runtime)
    if fault == "missing_source":
        entries = entries[1:]
    elif fault == "wrong_entity":
        entries[0]["response"]["content"]["text"] = entries[0]["response"]["content"][
            "text"
        ].replace("item-7", "item-8")
    elif fault == "wrong_target":
        entries[1]["request"]["postData"]["text"] = "note=wrong+but+valid+note"
    elif fault == "order":
        entries[1]["startedDateTime"] = "2026-01-01T00:00:00.005+00:00"
    elif fault == "timing":
        entries[0].pop("time")
    elif fault == "audit":
        frames[0]["context_id"] = "audit"
    elif fault == "missing_body":
        entries[0]["response"]["content"].pop("text")
    elif fault == "offscreen_fact":
        frames[0]["links"][0]["body_text"] = "Invented product"
        frames[0]["links"][0]["visible_html"] = frames[0]["links"][0]["visible_html"].replace(
            "$20.00", ""
        )
    elif fault == "other_item_price":
        entries[0]["response"]["content"]["text"] = (
            '<table><tr><td><a href="/item-7.html">Invented product</a></td><td>$99.00</td></tr><tr><td>Other</td><td>$20.00</td></tr></table>'
        )
    result = evaluate_dependencies(
        candidate,
        trace_type.from_content(entries),
        {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"},
        frames,
        boundary,
        "model_context_only",
    )
    assert result["status"] == (
        "pass" if fault is None else "hold" if fault in {"timing", "missing_body"} else "fail"
    )
    if fault is None:
        assert (
            result["matches"]["cart_to_issue"]["source_entity"]
            == "http://shopping.test/item-7.html"
        )


@pytest.mark.parametrize("target_present", [False, True])
def test_unrelated_bodyless_html_does_not_hide_missing_dependency(runtime, target_present):
    candidate, entries, trace_type, frames, boundary = _fixture(runtime)
    unrelated = deepcopy(entries[0])
    unrelated["request"]["url"] = "http://shopping.test/static/minicart/template.html"
    unrelated["response"]["content"].pop("text")
    entries = [unrelated, *entries[1:]] if target_present else [unrelated]
    result = evaluate_dependencies(
        candidate,
        trace_type.from_content(entries),
        {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"},
        frames,
        boundary,
        "model_context_only",
    )
    assert result["status"] == "fail"
    assert result["reason"] == "source entity/facts or target missing"


@pytest.mark.parametrize(
    ("missing_document", "frame_site", "expected_status"),
    [
        ("cart", "shopping", "hold"),
        ("item-7.html", "shopping", "hold"),
        ("equivalent-cart", "shopping", "hold"),
        ("equivalent-cart", "gitlab", "fail"),
    ],
)
def test_bodyless_source_document_requires_declared_or_observed_source_origin(
    runtime, missing_document, frame_site, expected_status
):
    candidate, entries, trace_type, frames, boundary = _fixture(runtime)
    entries[0]["request"]["url"] = f"http://shopping.test/{missing_document}"
    entries[0]["response"]["content"].pop("text")
    frames[0]["page_url"] = f"http://{frame_site}.test/{missing_document}"
    result = evaluate_dependencies(
        candidate,
        trace_type.from_content(entries),
        {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"},
        frames,
        boundary,
        "model_context_only",
    )
    assert result["status"] == expected_status
    assert result["reason"] == (
        "original source response body unavailable"
        if expected_status == "hold"
        else "source entity/facts or target missing"
    )


@pytest.mark.parametrize("source_document", ["cart", "equivalent-cart", "item-7.html"])
def test_complete_equivalent_source_survives_unrelated_bodyless_html(runtime, source_document):
    candidate, entries, trace_type, frames, boundary = _fixture(runtime)
    entries[0]["request"]["url"] = f"http://shopping.test/{source_document}"
    frames[0]["page_url"] = entries[0]["request"]["url"]
    unrelated = deepcopy(entries[0])
    unrelated["request"]["url"] = "http://shopping.test/static/minicart/template.html"
    unrelated["response"]["content"].pop("text")
    entries.insert(1, unrelated)
    result = evaluate_dependencies(
        candidate,
        trace_type.from_content(entries),
        {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"},
        frames,
        boundary,
        "model_context_only",
    )
    assert result["status"] == "pass"


def test_two_site_dependencies_cannot_be_empty(runtime):
    candidate, *_ = _fixture(runtime)
    candidate["task_definition"]["dependencies"] = []
    with pytest.raises(ContractError, match="require dependencies"):
        validate_dependencies(candidate["task_definition"], candidate["task_spec"]["sites"])


def test_derived_ratio_uses_visible_inputs_without_requiring_ratio_in_source(runtime):
    candidate, entries, trace_type, frames, boundary = _fixture(runtime)
    parameters = candidate["task_definition"]["parameters"]
    parameters.update(
        second_url="__SHOPPING__/item-8.html",
        second_label="Another product",
        second_price="$60.00",
        total=80.0,
        share="25.00",
        note="Invented product accounts for 25.00 percent",
    )
    rules = [
        {
            "field": "/task_definition/parameters/total",
            "rule": {"operation": "sum", "parameters": ["price", "second_price"]},
        },
        {
            "field": "/task_definition/parameters/share",
            "rule": {
                "operation": "ratio_percent",
                "parameters": ["price", "total"],
                "output_format": ".2f",
            },
        },
        {
            "field": "/task_definition/parameters/note",
            "rule": {
                "operation": "format",
                "parameters": ["entity_label", "share"],
                "template": "{entity_label} accounts for {share} percent",
            },
        },
    ]
    candidate["grounding"]["field_evidence"] = rules
    candidate["task_definition"]["dependencies"].append(
        {
            "name": "second_subtotal_to_share",
            "source_site": "shopping",
            "target_site": "gitlab",
            "source_entity": {
                "identity_parameter": "second_url",
                "label_parameter": "second_label",
            },
            "source_url_parameter": "source_url",
            "fact_readouts": {
                "second_price": {"kind": "text", "selector": ".price", "value_type": "string"}
            },
            "target_event_index": 0,
        }
    )
    entries[0]["response"]["content"]["text"] = entries[0]["response"]["content"]["text"].replace(
        "</table>",
        '<tr><td><a href="/item-8.html">Another product</a></td><td class="price">$60.00</td></tr></table>',
    )
    frames[0]["body_text"] += " Another product $60.00"
    frames[0]["links"].append(
        {
            "href": "http://shopping.test/item-8.html",
            "text": "Another product",
            "body_text": "Another product $60.00",
            "visible_html": '<tr><td><a href="/item-8.html">Another product</a></td><td class="price">$60.00</td></tr>',
        }
    )
    for record in rules:
        name = record["field"].rsplit("/", 1)[1]
        assert derive_value(record["rule"], parameters) == parameters[name]
    candidate["task_spec"]["eval"][1]["expected"]["post_data"]["note"] = parameters["note"]
    entries[1]["request"]["postData"]["text"] = "note=Invented+product+accounts+for+25.00+percent"
    result = evaluate_dependencies(
        candidate,
        trace_type.from_content(entries),
        {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"},
        frames,
        boundary,
        "model_context_only",
    )
    assert result["status"] == "pass"
    assert "25.00" not in entries[0]["response"]["content"]["text"]


@pytest.mark.parametrize("fault", [None, "tamper_frame", "trace_disagreement"])
def test_recording_hashes_and_both_trace_representations(runtime, tmp_path, fault):
    from webarena_verified.types.tracing import NetworkTrace

    candidate, entries, _, frames, boundary = _fixture(runtime)
    runtime.replay.install_playwright_trace_loader()
    (tmp_path / "network.har").write_text(json.dumps({"log": {"entries": entries}}))
    with zipfile.ZipFile(tmp_path / "trace.zip", "w") as archive:
        archive.writestr(
            "trace.network",
            "".join(
                json.dumps({"type": "resource-snapshot", "snapshot": entry}) + "\n"
                for entry in entries
            ),
        )
    (tmp_path / "frame.png").write_bytes(b"offline recorded screenshot fixture")
    frames[0].update(screenshot="frame.png", screenshot_sha256=sha256_file(tmp_path / "frame.png"))
    (tmp_path / "cross_site_gui_frames.jsonl").write_text(json.dumps(frames[0]) + "\n")
    seal_recording(
        tmp_path,
        "original",
        "model_context_only",
        boundary["started_at"],
        boundary["ended_at"],
        tmp_path / "network.har",
        tmp_path / "trace.zip",
        Path(__file__),
    )
    har = NetworkTrace.from_har(tmp_path / "network.har")
    playwright = NetworkTrace.from_playwright_trace(tmp_path / "trace.zip")
    if fault == "tamper_frame":
        (tmp_path / "frame.png").write_bytes(b"modified after sealing")
    elif fault == "trace_disagreement":
        playwright = playwright.model_copy(update={"events": playwright.events[1:]})
    result = evaluate_recording(
        candidate,
        {"positive_har": har, "positive_playwright": playwright},
        {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"},
        artifact_root=tmp_path,
        har_path=tmp_path / "network.har",
        trace_path=tmp_path / "trace.zip",
        expected_scope="model_context_only",
    )
    assert result["cross_site_dependency_status"] == ("pass" if fault is None else "hold")


def test_author_replay_includes_independent_dependency_negatives(runtime, tmp_path):
    candidate, entries, _, frames, boundary = _fixture(runtime)
    (tmp_path / "network.har").write_text(json.dumps({"log": {"entries": entries}}))
    with zipfile.ZipFile(tmp_path / "trace.zip", "w") as archive:
        archive.writestr(
            "trace.network",
            "".join(
                json.dumps({"type": "resource-snapshot", "snapshot": entry}) + "\n"
                for entry in entries
            ),
        )
    (tmp_path / "frame.png").write_bytes(b"offline original screenshot fixture")
    (tmp_path / "observation.json").write_text("{}")
    frames[0].update(
        scope="author_context_only",
        screenshot="frame.png",
        screenshot_sha256=sha256_file(tmp_path / "frame.png"),
    )
    (tmp_path / "cross_site_gui_frames.jsonl").write_text(json.dumps(frames[0]) + "\n")
    seal_recording(
        tmp_path,
        "original",
        "author_context_only",
        boundary["started_at"],
        boundary["ended_at"],
        tmp_path / "network.har",
        tmp_path / "trace.zip",
        Path(__file__),
    )
    candidate["grounding"].update(
        artifacts={
            "har": "network.har",
            "trace": "trace.zip",
            "screenshot": "frame.png",
            "observation": "observation.json",
        },
        task_spec_sha256=sha256_json(candidate["task_spec"]),
        episode_boundary_artifact={
            "path": "episode_boundary.json",
            "sha256": sha256_file(tmp_path / "episode_boundary.json"),
        },
    )
    candidate["grounding"]["artifact_sha256"] = {
        name: sha256_file(tmp_path / path)
        for name, path in candidate["grounding"]["artifacts"].items()
    }
    candidate.update(
        baseline_sha256=sha256_json(candidate["baseline"]),
        grounding_sha256=sha256_json(candidate["grounding"]),
        task_definition_sha256=sha256_json(candidate["task_definition"]),
    )
    (tmp_path / "private.jsonl").write_text(json.dumps(candidate) + "\n")
    result = runtime.replay.validate_replay(tmp_path, candidate)
    assert result["status"] == "pass"
    assert result["checks"]["cross_site_negative_probes_reject"]
    assert set(result["cross_site_negative_probes"]) == {
        "missing_source",
        "wrong_entity",
        "wrong_target",
        "wrong_order",
    }


def _native_recording(runtime, root):
    candidate, entries, _, frames, boundary = _fixture(runtime)
    (root / "network.har").write_text(json.dumps({"log": {"entries": entries}}))
    with zipfile.ZipFile(root / "trace.zip", "w") as archive:
        archive.writestr(
            "trace.network",
            "".join(
                json.dumps({"type": "resource-snapshot", "snapshot": entry}) + "\n"
                for entry in entries
            ),
        )
    (root / "desktop").mkdir()
    (root / "desktop/step_000.png").write_bytes(b"offline original screenshot fixture")
    for name in ("initial.png", "terminal.png", "state_audit.png"):
        (root / name).write_bytes(b"offline original screenshot fixture")
    (root / "initial_observation.json").write_text("{}")
    (root / "private_task.json").write_text(json.dumps(candidate["task_spec"]))
    candidate["baseline_sha256"] = sha256_json(candidate["baseline"])
    candidate["grounding"]["final_state_checks"] = [
        {
            "kind": "visible_text",
            "parameter": "note",
            "selector": "body",
            "site": "gitlab",
            "readback_url_parameter": "issue_url",
        }
    ]
    (root / "final_state.json").write_text(
        json.dumps(
            {
                "task_spec_sha256": sha256_json(candidate["task_spec"]),
                "confirmed": True,
                "reloaded_for_persistence": True,
            }
        )
    )
    frames[0].update(
        screenshot="desktop/step_000.png",
        screenshot_sha256=sha256_file(root / "desktop/step_000.png"),
    )
    (root / "cross_site_gui_frames.jsonl").write_text(json.dumps(frames[0]) + "\n")
    seal_recording(
        root,
        "original",
        "model_context_only",
        boundary["started_at"],
        boundary["ended_at"],
        root / "network.har",
        root / "trace.zip",
        Path(__file__),
    )
    runtime.replay.install_playwright_trace_loader()
    return candidate


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "episode_boundary.json",
        "episode_recording_lifecycle.json",
        "cross_site_gui_frames.jsonl",
        "desktop/step_000.png",
    ],
)
def test_native_score_artifacts_close_recording_proofs(runtime, tmp_path, missing):
    candidate = _native_recording(runtime, tmp_path)
    if missing:
        (tmp_path / missing).unlink()
    result = runtime.reference.score_reference(
        candidate,
        json.dumps(candidate["reference_response"]),
        tmp_path,
        {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"},
    )
    assert "cross_site_recording_boundary_sha256" not in result
    assert result["status"] == ("pass" if missing is None else "hold")
    assert result["checks"]["cross_site_recording_evidence_closed"] is (missing is None)
    if missing is None:
        assert {
            "episode_boundary.json",
            "episode_recording_lifecycle.json",
            "cross_site_gui_frames.jsonl",
            "desktop/step_000.png",
        } <= set(result["artifacts"].values())
        assert all(
            result["artifact_sha256"][name] == sha256_file(tmp_path / path)
            for name, path in result["artifacts"].items()
        )


@pytest.mark.parametrize("consumer", ["clean_native_trajectories", "export_review_bundle"])
@pytest.mark.parametrize(
    "proof",
    [
        "episode_boundary.json",
        "episode_recording_lifecycle.json",
        "cross_site_gui_frames.jsonl",
        "desktop/step_000.png",
    ],
)
@pytest.mark.parametrize("fault", ["modify", "remove"])
def test_formal_consumers_reject_modified_or_missing_recording_proofs(
    runtime, tmp_path, consumer, proof, fault
):
    import importlib

    candidate = _native_recording(runtime, tmp_path)
    score = runtime.reference.score_reference(
        candidate,
        json.dumps(candidate["reference_response"]),
        tmp_path,
        {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"},
    )
    assert score["status"] == "pass"
    score.update(model_id="qwen3.8-max", api_type="openai_chat")
    (tmp_path / "reference_receipt.json").write_text(json.dumps(score))
    native = {"metadata": {"model_id": "qwen3.8-max"}, "messages": [{"role": "assistant"}]}
    execution = {
        "teacher": {
            "requested_model": "qwen3.8-max",
            "returned_probe_model": "qwen3.8-max",
            "api_type": "openai_chat",
            "params": {"temperature": 0},
        }
    }
    (tmp_path / "model_execution.json").write_text(json.dumps(execution))
    requests = tmp_path / "requests"
    requests.mkdir()
    (requests / "000.request.json").write_text(
        json.dumps({"model": "qwen3.8-max", "messages": [], "tools": [], "temperature": 0})
    )
    (requests / "000.response.json").write_text(
        json.dumps({"model": "qwen3.8-max", "choices": [{"finish_reason": "stop"}]})
    )
    entry = {
        "score_receipt": str(tmp_path / "reference_receipt.json"),
        "task_id": candidate["generated_task_id"],
        "api_type": "openai_chat",
        "request_dir": str(requests),
        "model_execution": str(tmp_path / "model_execution.json"),
        "model_execution_sha256": sha256_file(tmp_path / "model_execution.json"),
    }
    verify = importlib.import_module(consumer).verify_execution_evidence
    verify(entry, native, public_projection(candidate), True)
    if fault == "modify":
        (tmp_path / proof).write_bytes(b"changed after native score was sealed")
    else:
        (tmp_path / proof).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(entry, native, public_projection(candidate), True)


@pytest.mark.parametrize(
    "source_quantity,gui_quantity", [(2, 2), (1, 2), (2, 1), (1, 1), (2, None)]
)
def test_quantity_input_is_proven_independently_of_price_digits(
    runtime, source_quantity, gui_quantity
):
    candidate, entries, trace_type, frames, boundary = _fixture(runtime)
    parameters = candidate["task_definition"]["parameters"]
    parameters.update(quantity=2, note="Invented product quantity 2")
    dependency = candidate["task_definition"]["dependencies"][0]
    dependency["fact_readouts"] = {
        "entity_label": {"kind": "text", "selector": "a", "value_type": "string"},
        "quantity": {
            "kind": "input_value",
            "selector": "input[data-role=cart-item-qty]",
            "value_type": "integer",
        },
    }
    candidate["grounding"]["field_evidence"][0]["rule"] = {
        "operation": "format",
        "parameters": ["entity_label", "quantity"],
        "template": "{entity_label} quantity {quantity}",
    }
    markup = '<tr><td><a href="/item-7.html">Invented product</a><input data-role="cart-item-qty" value="{}"></td><td class="price">$21.00</td></tr>'
    entries[0]["response"]["content"]["text"] = (
        "<table>" + markup.format(source_quantity) + "</table>"
    )
    entries[1]["request"]["postData"]["text"] = "note=Invented+product+quantity+2"
    candidate["task_spec"]["eval"][1]["expected"]["post_data"]["note"] = parameters["note"]
    frames[0]["body_text"] = "Invented product $21.00"
    frames[0]["links"][0].update(
        body_text=frames[0]["body_text"], visible_html=markup.format(gui_quantity)
    )
    if gui_quantity is None:
        frames[0]["links"][0]["visible_html"] = frames[0]["links"][0]["visible_html"].replace(
            '<input data-role="cart-item-qty" value="None">', ""
        )
    validate_dependencies(candidate["task_definition"], candidate["task_spec"]["sites"])
    result = evaluate_dependencies(
        candidate,
        trace_type.from_content(entries),
        {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"},
        frames,
        boundary,
        "model_context_only",
    )
    assert result["status"] == ("pass" if source_quantity == gui_quantity == 2 else "fail")


def test_numeric_fact_values_require_complete_declared_numbers():
    assert parse_fact_value("2", "integer") == 2
    assert parse_fact_value("2.50", "decimal") == 2.5
    assert parse_fact_value("$21.00", "string") == "$21.00"
    for value in ("$21.00", "quantity 2", "2 items", "2.5"):
        with pytest.raises(ValueError):
            parse_fact_value(value, "integer")
