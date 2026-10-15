#!/usr/bin/env python3
"""Record new fixed-view map UI evidence for generated search and navigation.

The coordinator supplies a fresh baseline and runs Playwright remotely. Example:
  python -B author_map_tasks.py --batch-root /outputs \
    --baseline-receipt /outputs/private/map-baseline.json \
    --site-url http://localhost:8080 \
    --viewport-width <actual-content-width> --viewport-height <actual-content-height> \
    --comparison-viewport-width <second-width> --comparison-viewport-height <second-height> \
    --task-config /outputs/plans/map_task_configs_grounding.json \
    --generated-task-id wavgen-v03-full300-0050

Route configurations record readiness only and require explicit --from and --to
query text. They never emit a draft or route GT. Search answers and navigation
targets derive solely from this run's newly recorded first result batch, never
from exploration. More results is recorded but never clicked. No model, catalog,
site preparation, or resource lifecycle operation belongs to this entry.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--generated-task-id", required=True)
    parser.add_argument("--attempt", default="attempt-01")
    parser.add_argument("--viewport-width", type=int, required=True)
    parser.add_argument("--viewport-height", type=int, required=True)
    parser.add_argument("--comparison-viewport-width", type=int)
    parser.add_argument("--comparison-viewport-height", type=int)
    parser.add_argument("--from", dest="from_query", help="Actual origin query typed in Directions")
    parser.add_argument(
        "--to", dest="to_query", help="Actual destination query typed in Directions"
    )
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    args = parser.parse_args(argv)
    site = urlsplit(args.site_url)
    if (
        site.scheme not in {"http", "https"}
        or not site.netloc
        or site.username
        or site.password
        or site.query
        or site.fragment
        or site.path not in {"", "/"}
    ):
        parser.error("site URL must be an explicit HTTP(S) origin without credentials")
    if not re.fullmatch(r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", args.generated_task_id):
        parser.error("task ID must use the generated namespace")
    if not re.fullmatch(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", args.attempt):
        parser.error("attempt must be a simple identifier")
    if bool(args.from_query) != bool(args.to_query):
        parser.error("Directions requires both explicit --from and --to queries")
    if args.timeout_ms <= 0:
        parser.error("timeout must be positive")
    if args.viewport_width <= 0 or args.viewport_height <= 0:
        parser.error("content viewport dimensions must be positive")
    if bool(args.comparison_viewport_width) != bool(args.comparison_viewport_height):
        parser.error("comparison viewport requires both width and height")
    if args.comparison_viewport_width is not None and (
        args.comparison_viewport_width <= 0 or args.comparison_viewport_height <= 0
    ):
        parser.error("comparison viewport dimensions must be positive")
    return args


def _record(page: Any, output: Path, name: str) -> dict:
    """Read rendered result labels/categories and actual same-site navigation links."""
    observation = page.evaluate("""() => {
      const visible = e => {
        const box = e.getBoundingClientRect();
        return !!(box.width && box.height) && getComputedStyle(e).visibility !== 'hidden';
      };
      const text = e => e.innerText.trim();
      return {
        url: location.href, title: document.title, body_text: document.body.innerText,
        headings: Array.from(document.querySelectorAll('h1, h2, h3')).filter(visible).map(text),
        results: Array.from(document.querySelectorAll(
          '.results-list > li.search_results_entry')).filter(visible).map(row => {
            const anchor = row.querySelector('a.set_position[href]');
            if (!anchor || !visible(anchor)) throw new Error('result anchor not rendered');
            return {category: Array.from(row.childNodes).filter(n => n.nodeType === 3)
                      .map(n => n.textContent).join('').trim(),
                    data_prefix: anchor.getAttribute('data-prefix'),
                    label: text(anchor), url: anchor.href};
          }),
        more_results_links: Array.from(document.querySelectorAll('.search_more a[href]'))
          .filter(visible).map(a => ({text: text(a), url: a.href})),
        controls: Array.from(document.querySelectorAll('input, select, button'))
          .filter(visible).filter(e => e.type !== 'hidden' && e.type !== 'password')
          .map(e => ({name: e.name, id: e.id, text: text(e), value: e.value,
            form: e.form ? {method: e.form.method, action: e.form.action} : null,
            options: e.tagName === 'SELECT' ? Array.from(e.options).map(o => ({
              label: o.textContent.trim(), value: o.value, selected: o.selected})) : null})),
        links: Array.from(document.querySelectorAll('a[href]')).filter(visible)
          .map(a => ({text: text(a), label: text(a) || a.title || a.getAttribute('aria-label'),
                     url: a.href})),
        tables: Array.from(document.querySelectorAll('table')).filter(visible)
          .map(table => Array.from(table.querySelectorAll('tr')).filter(visible)
            .map(tr => Array.from(tr.querySelectorAll('th, td')).map(text)))
      };
    }""")
    page.screenshot(path=str(output / f"{name}.png"))
    (output / f"{name}.json").write_text(json.dumps(observation, ensure_ascii=False, indent=2))
    return observation


def build_task_definition(config: dict, observation: dict) -> tuple[dict, list, list]:
    """Derive answer leaves and unique navigation targets from a new UI batch."""
    mode = config["mode"]
    if mode == "route-readiness":
        raise ValueError("Route readiness cannot emit GT or a generated task definition")
    batch = observation["search"]
    rows = batch["results"]
    if len(rows) < 3 or len({row["url"] for row in rows}) != len(rows):
        raise ValueError("First batch requires at least three distinct rendered features")
    if any(not row["category"] or row["category"] != row["data_prefix"] for row in rows):
        raise ValueError("Visible category differs from the recorded result prefix")
    if any(link["text"] != "More results" for link in batch["more_results_links"]):
        raise ValueError("Unrecognized result continuation control")
    parameters = dict(config["parameters"])
    evidence = []
    final_checks = []
    source_locations = []
    scope = (
        "Starting from the provided map view, search for {query} without panning or zooming "
        "before submission. Inspect only the complete initial displayed result batch, including "
        "rows reachable by scrolling that list. Do not select More results; later batches and "
        "global matches are excluded. "
    )
    if mode == "retrieve-category-summary":
        categories = sorted({row["category"] for row in rows})
        if len(categories) < 2:
            raise ValueError("Category comparison requires at least two displayed categories")
        values = [
            {
                "category": category,
                "result_count": sum(row["category"] == category for row in rows),
                "first_batch_total": len(rows),
                "more_results_available": bool(batch["more_results_links"]),
            }
            for category in categories
        ]
        goal = scope + (
            "Group all initial-batch results by their exact displayed category prefix. Return "
            "category, result_count, first_batch_total, and more_results_available. Order the "
            "groups alphabetically by category. A More results control means additional results "
            "may exist; it does not add them to this batch. Do not edit map data."
        )
        for category in categories:
            first_index = next(i for i, row in enumerate(rows) if row["category"] == category)
            source_locations.append(
                {
                    "category": [f"/search/results/{first_index}/category"],
                    "result_count": [f"/search/results/{i}/category" for i in range(len(rows))],
                    "first_batch_total": ["/search/results"],
                    "more_results_available": ["/search/more_results_links"],
                }
            )
    elif mode == "retrieve-location-summary":
        values = []
        for region in (parameters["region_1"], parameters["region_2"]):
            matching = [
                row for row in rows if region in [s.strip() for s in row["label"].split(",")]
            ]
            if not matching:
                raise ValueError("Both requested regions must have freshly displayed matches")
            values.append(
                {
                    "region": region,
                    "result_count": len(matching),
                    "distinct_category_count": len({row["category"] for row in matching}),
                    "library_count": sum(row["category"] == "Library" for row in matching),
                }
            )
            locations = [f"/search/results/{i}/label" for i in range(len(rows))]
            categories = [f"/search/results/{i}/category" for i in range(len(rows))]
            source_locations.append(
                {
                    "region": locations,
                    "result_count": locations,
                    "distinct_category_count": locations + categories,
                    "library_count": locations + categories,
                }
            )
        goal = scope + (
            "Compare results whose comma-separated address contains the exact region component "
            "{region_1} with those containing {region_2}. For each region return region, "
            "result_count, distinct_category_count, and library_count (category exactly Library). "
            "Return regions in that order. Do not edit map data."
        )
    elif mode == "retrieve-address-comparison":
        matching = []
        for index, row in enumerate(rows):
            components = [s.strip() for s in row["label"].split(",")]
            if row["category"] == parameters["category"] and parameters["region"] in components:
                if len(components) < 5 or not re.fullmatch(r"[0-9]+", components[1]):
                    raise ValueError("Matching address lacks a displayed integer building number")
                if not re.fullmatch(r"[0-9]{5}", components[-2]):
                    raise ValueError("Matching address lacks a displayed five-digit postal code")
                matching.append((int(components[1]), row, index, components[-2]))
        if len(matching) != 2 or matching[0][0] == matching[1][0]:
            raise ValueError(
                "Numeric address comparison requires exactly two distinct building numbers"
            )
        lower, higher = sorted(matching, key=lambda item: item[0])
        values = [
            {
                "lower_number_label": lower[1]["label"],
                "lower_building_number": lower[0],
                "lower_postal_code": lower[3],
                "higher_number_label": higher[1]["label"],
                "higher_building_number": higher[0],
                "higher_postal_code": higher[3],
                "building_number_difference": higher[0] - lower[0],
            }
        ]
        locations = {
            field: [f"/search/results/{item[2]}/label"]
            for prefix, item in (("lower", lower), ("higher", higher))
            for field in (
                f"{prefix}_number_label",
                f"{prefix}_building_number",
                f"{prefix}_postal_code",
            )
        }
        locations["building_number_difference"] = [
            f"/search/results/{item[2]}/label" for item in (lower, higher)
        ]
        source_locations = [locations]
        goal = scope + (
            "Consider only category {category} entries whose address contains the exact region "
            "component {region}. There must be two such entries. Compare their displayed "
            "integer building numbers (the address component directly after the feature name). "
            "Return lower_number_label, lower_building_number, lower_postal_code, "
            "higher_number_label, higher_building_number, higher_postal_code, and "
            "building_number_difference (higher minus lower). Labels and five-digit postal "
            "codes must retain the displayed text. Do not edit map data."
        )
    elif mode == "navigate-constrained-feature":
        matching = [
            (index, row)
            for index, row in enumerate(rows)
            if row["category"] == parameters["category"]
            and all(
                token in [s.strip() for s in row["label"].split(",")]
                for token in parameters["address_components"]
            )
        ]
        if len(matching) != 1:
            raise ValueError(
                "Category and address constraints do not uniquely select one fresh result"
            )
        index, selected = matching[0]
        path = urlsplit(selected["url"]).path
        if not re.fullmatch(r"/(node|way|relation)/[0-9]+", path):
            raise ValueError("Selected result is not an observed OSM feature page")
        parameters.update(
            target_url="__MAP__" + path,
            target_path_regex=re.escape(path),
            selected_label=selected["label"],
        )
        for name in ("target_url", "target_path_regex", "selected_label"):
            source_field = "label" if name == "selected_label" else "url"
            evidence.append(
                {
                    "field": f"/task_definition/parameters/{name}",
                    "artifact": "observation",
                    "location": f"/search/results/{index}/{source_field}",
                    "modality": "text",
                    "derivation": (
                        "Unique displayed category and exact address-component match; "
                        "preserve feature path."
                    ),
                }
            )
        goal = scope + (
            "Find the single result with category {category} whose comma-separated address "
            "contains every component in {address_components}. Open its local feature-details "
            "link and leave that feature page open. Do not edit map data."
        )
        return (
            {
                "task_type": "NAVIGATE",
                "goal_template": goal,
                "parameters": parameters,
                "answer": {
                    "status": "SUCCESS",
                    "values": None,
                    "item_schema": None,
                    "ordered": False,
                },
                "completion_events": [
                    {
                        "url_template": "^__MAP__{target_path_regex}(?:[?#].*)?$",
                        "http_method": "GET",
                        "response_status": 200,
                        "last_event_only": False,
                    }
                ],
            },
            evidence,
            [{"kind": "final_url", "parameter": "target_url"}],
        )
    else:
        raise ValueError(f"Unsupported explicit map author mode: {mode}")
    for index, locations in enumerate(source_locations):
        for field, pointers in locations.items():
            for pointer in pointers:
                evidence.append(
                    {
                        "field": f"/task_definition/answer/values/{index}/{field}",
                        "artifact": "observation",
                        "location": pointer,
                        "modality": "text",
                        "derivation": (
                            "Classify and count only this complete initial rendered batch, or "
                            "parse/compare its displayed address components as the query specifies."
                        ),
                    }
                )
    schema = {
        "type": "object",
        "properties": {
            field: {
                "type": "boolean"
                if type(value) is bool
                else "integer"
                if type(value) is int
                else "string"
            }
            for field, value in values[0].items()
        },
        "required": list(values[0]),
        "additionalProperties": False,
    }
    return (
        {
            "task_type": "RETRIEVE",
            "goal_template": goal,
            "parameters": parameters,
            "answer": {
                "status": "SUCCESS",
                "values": values,
                "item_schema": schema,
                "ordered": True,
            },
            "completion_events": [],
        },
        evidence,
        final_checks,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    configs = json.loads(args.task_config.read_text())
    selected = [
        task for task in configs["tasks"] if task["generated_task_id"] == args.generated_task_id
    ]
    if len(selected) != 1:
        raise ValueError("Select exactly one configured generated map task")
    config = selected[0]
    route_mode = config["mode"] == "route-readiness"
    if route_mode and not args.from_query:
        raise ValueError("Route readiness requires explicit --from and --to query inputs")
    if not route_mode and args.from_query:
        raise ValueError("Search authoring does not consume route query inputs")
    viewport = {"width": args.viewport_width, "height": args.viewport_height}
    comparison_viewport = {
        "width": args.comparison_viewport_width,
        "height": args.comparison_viewport_height,
    }
    if not route_mode and (
        args.comparison_viewport_width is None or comparison_viewport == viewport
    ):
        raise ValueError("Search authoring requires two explicit, different content viewports")
    baseline = json.loads(args.baseline_receipt.read_text())
    site_url = args.site_url.rstrip("/")
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("Authoring requires a coordinator-owned fixed image baseline")
    if baseline["site_url"].rstrip("/") != site_url:
        raise ValueError("Author endpoint differs from the fixed baseline receipt")
    output = args.batch_root / "author" / args.generated_task_id / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2))
    observed = {"scope": "new_formal_author_recording", "viewport": viewport}
    blocked = []
    origin = urlsplit(site_url)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(
            storage_state=str(args.storage_state) if args.storage_state else None,
            viewport=viewport,
            service_workers="block",
            accept_downloads=False,
            record_har_path=str(output / "network.har"),
            record_har_content="embed",
        )

        def restrict(route: Any) -> None:
            request = route.request
            target = urlsplit(request.url)
            same_origin = (target.scheme, target.netloc) == (origin.scheme, origin.netloc)
            read_method = request.method in {"GET", "HEAD"} or (
                request.method == "POST" and target.path == "/geocoder/search_osm_nominatim"
            )
            if same_origin and read_method:
                route.continue_()
            else:
                blocked.append(
                    {"method": request.method, "host": target.netloc, "path": target.path}
                )
                route.abort()

        context.route("**/*", restrict)
        context.set_default_timeout(args.timeout_ms)
        context.tracing.start(screenshots=True, snapshots=True, sources=False)
        page = context.new_page()
        try:
            start = site_url + configs["view"]["start_path"]
            page.goto(start, wait_until="networkidle", timeout=args.timeout_ms)
            observed["initial"] = _record(page, output, "initial")
            if route_mode:
                target = site_url + "/directions"
                if not any(link["url"] == target for link in observed["initial"]["links"]):
                    raise ValueError("Directions link is absent from the newly observed homepage")
                observed["route_inputs"] = {
                    "from_query": args.from_query,
                    "to_query": args.to_query,
                }
                observed["routes"] = []
                for index, engine in enumerate(config["engine_labels"]):
                    page.goto(target, wait_until="networkidle", timeout=args.timeout_ms)
                    form = _record(page, output, f"directions_{index:02}")
                    controls = {
                        control["name"]: control
                        for control in form["controls"]
                        if control["name"] in {"route_from", "route_to", "routing_engines"}
                    }
                    if set(controls) != {"route_from", "route_to", "routing_engines"}:
                        raise ValueError("Actual Directions controls are unavailable")
                    if any(
                        control["form"]["method"].upper() != "GET" for control in controls.values()
                    ):
                        raise ValueError(
                            "Directions controls are not the observed read-only GET form"
                        )
                    if engine not in {
                        option["label"] for option in controls["routing_engines"]["options"]
                    }:
                        raise ValueError("Requested engine is absent from the actual options")
                    page.locator("input#route_from:visible").fill(args.from_query)
                    page.locator("input#route_to:visible").fill(args.to_query)
                    page.locator("select[name='routing_engines']:visible").select_option(
                        label=engine
                    )
                    page.locator("input#route_to:visible").press("Enter")
                    page.wait_for_timeout(5_000)
                    snapshot = _record(page, output, f"route_{index:02}")
                    observed["routes"].append({"engine_label": engine, "observation": snapshot})
            else:
                query = config["parameters"]["query"]
                observed["search_batches"] = []
                # Re-record the UI from the public starting view at both actual
                # content sizes. Bounding boxes are evidence, not hidden gates.
                for index, size in enumerate((viewport, comparison_viewport)):
                    page.set_viewport_size(size)
                    page.goto(start, wait_until="networkidle", timeout=args.timeout_ms)
                    initial = _record(page, output, f"initial_{index:02}")
                    search_form = [
                        control for control in initial["controls"] if control["name"] == "query"
                    ]
                    if len(search_form) != 1 or search_form[0]["form"]["method"].upper() != "GET":
                        raise ValueError("Actual read-only search form is unavailable")
                    with page.expect_response(
                        lambda response: (
                            response.request.method == "POST"
                            and urlsplit(response.url).path == "/geocoder/search_osm_nominatim"
                            and parse_qs(urlsplit(response.url).query).get("query") == [query]
                        ),
                        timeout=args.timeout_ms,
                    ) as response_event:
                        control = page.locator("input#query:visible")
                        control.fill(query)
                        control.press("Enter")
                    response = response_event.value
                    if response.status != 200:
                        raise ValueError(
                            "New initial geocoder result request did not return HTTP200"
                        )
                    response.finished()
                    fields = parse_qs(response.request.post_data or "")
                    actual_view = {
                        name: float(fields[name][0])
                        for name in ("zoom", "minlon", "minlat", "maxlon", "maxlat")
                    }
                    page.locator(
                        ".results-list > li.search_results_entry a.set_position[href]"
                    ).first.wait_for()
                    page.wait_for_load_state("networkidle", timeout=args.timeout_ms)
                    snapshot = _record(page, output, f"search_{index:02}")
                    observed["search_batches"].append(
                        {
                            "viewport": size,
                            "initial": initial,
                            "request": {
                                "method": "POST",
                                "url": response.url,
                                "status": response.status,
                                "view": actual_view,
                                "post_field_names": sorted(fields),
                            },
                            "snapshot": snapshot,
                        }
                    )
                fingerprints = [
                    sorted(
                        (row["category"], row["label"], row["url"])
                        for row in batch["snapshot"]["results"]
                    )
                    for batch in observed["search_batches"]
                ]
                if fingerprints[0] != fingerprints[1]:
                    raise ValueError(
                        "Two newly recorded content viewports expose different first-batch "
                        "feature facts; no GT emitted"
                    )
                if [
                    bool(batch["snapshot"]["more_results_links"])
                    for batch in observed["search_batches"]
                ] not in ([True, True], [False, False]):
                    raise ValueError(
                        "Two newly recorded viewports disagree on result continuation; "
                        "no GT emitted"
                    )
                observed["search"] = observed["search_batches"][0]["snapshot"]
                observed["viewport_comparison"] = {
                    "UI_recordings": 2,
                    "same_feature_facts": True,
                    "same_continuation_presence": True,
                    "bbox_equality_required": False,
                    "result_ranking_used": False,
                }
                definition, _, _ = build_task_definition(config, observed)
                if config["mode"] == "navigate-constrained-feature":
                    page.set_viewport_size(viewport)
                    target = definition["parameters"]["target_url"].replace("__MAP__", site_url)
                    response = page.goto(target, wait_until="networkidle", timeout=args.timeout_ms)
                    observed["terminal"] = _record(page, output, "terminal")
                    observed["terminal_request"] = {
                        "method": "GET",
                        "url": response.url if response is not None else target,
                        "status": response.status if response is not None else None,
                    }
                    if observed["terminal_request"]["status"] != 200:
                        observed["navigation_failure"] = (
                            "Local feature details did not return HTTP200; no draft emitted"
                        )
                        raise ValueError(observed["navigation_failure"])
                    if urlsplit(observed["terminal"]["url"]).path != urlsplit(target).path:
                        raise ValueError(
                            "Navigation did not reach its observed feature-details path"
                        )
                    # A path alone can identify the site's real Not Found page.
                    # Require the named feature's visible identity and details.
                    feature_type, feature_id = urlsplit(target).path.strip("/").split("/")
                    identity = re.compile(
                        rf"\b{re.escape(feature_type)}\b.*\b{re.escape(feature_id)}\b",
                        re.IGNORECASE,
                    )
                    terminal = observed["terminal"]
                    selected_name = definition["parameters"]["selected_label"].split(",")[0]
                    if (
                        not any(identity.search(heading) for heading in terminal["headings"])
                        or selected_name not in terminal["body_text"]
                        or not any(
                            len(row) >= 2 and all(cell.strip() for cell in row[:2])
                            for table in terminal["tables"]
                            for row in table
                        )
                    ):
                        observed["navigation_failure"] = (
                            "HTTP200 page lacks the selected feature's visible identity "
                            "and rendered details; no draft emitted"
                        )
                        raise ValueError(observed["navigation_failure"])
        finally:
            context.tracing.stop(path=str(output / "trace.zip"))
            context.close()
            browser.close()
            (output / "blocked_requests.json").write_text(json.dumps(blocked, indent=2))
            (output / "observation.json").write_text(
                json.dumps(observed, ensure_ascii=False, indent=2)
            )
    if blocked:
        raise ValueError("Recorded requests violated the same-site read-only author policy")
    relative = output.relative_to(args.batch_root)
    if route_mode:
        receipt = {
            "status": "route_UI_recorded_distance_duration_steps_need_review",
            "generated_candidates": 0,
            "generated_task_id": args.generated_task_id,
            "site": "map",
            "route_inputs": observed["route_inputs"],
            "engine_labels": config["engine_labels"],
            "GT_emitted": False,
            "artifacts": {
                "har": "network.har",
                "trace": "trace.zip",
                "observation": "observation.json",
            },
            "blocked_requests": blocked,
        }
        (output / "readiness_receipt.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
        return
    definition, evidence, final_checks = build_task_definition(config, observed)
    draft = {
        "generated_task_id": args.generated_task_id,
        "task_id": config["task_id"],
        "intent_template_id": config["intent_template_id"],
        "site": "map",
        "start_url": "__MAP__" + configs["view"]["start_path"],
        "revision": 1,
        "source_artifact": config["source_artifact"],
        "source_record_id": config["source_record_id"],
        "baseline": baseline,
        "task_definition": definition,
        "grounding": {
            "captured_at": datetime.now(UTC).isoformat(),
            "family": config["family"],
            "artifacts": {
                "har": str(relative / "network.har"),
                "trace": str(relative / "trace.zip"),
                "screenshot": str(relative / ("terminal.png" if final_checks else "search_00.png")),
                "observation": str(relative / "observation.json"),
            },
            "field_evidence": evidence,
            "final_state_checks": final_checks,
        },
    }
    (output / "draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "status": "grounded",
                "generated_task_id": args.generated_task_id,
                "output": str(output),
                "GT_source": "new_formal_author_recording",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
