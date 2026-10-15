#!/usr/bin/env python3
"""Ground map tasks or collect explicit read-only exploration without emitting GT.

The coordinator supplies a fresh fixed-image baseline for grounding/readiness.
Explore instead verifies an existing owned container and records anonymous UI;
it never constructs an author baseline. The coordinator runs this remotely.
Readiness and explore start at an explicit same-site path; search and route inputs have no
default locations. It archives the existing map owner's observations, covers
every initial search row by scrolling, and records actual feature details.

Formal authoring accepts the pipeline CLI and a ``tasks`` configuration. Facts
and answers are calculated again in this recording, never copied from readiness.
Directions tasks derive endpoints from visible labels and bind actual route UI/network.
This entry calls no model, reads no database, and owns no sandbox lifecycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import parse_qs, urlsplit

from author_map_tasks import _record, build_task_definition


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("readiness", "explore", "ground"), default="ground")
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--container-inspection", type=Path)
    parser.add_argument("--ownership-receipt", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--nonce")
    parser.add_argument("--sandbox-id")
    parser.add_argument("--container-id")
    parser.add_argument("--storage-state", type=Path)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--generated-task-id")
    parser.add_argument("--attempt", default="grounding-01")
    parser.add_argument("--start-path")
    parser.add_argument("--viewport-width", type=int, required=True)
    parser.add_argument("--viewport-height", type=int, required=True)
    parser.add_argument("--comparison-viewport-width", type=int, required=True)
    parser.add_argument("--comparison-viewport-height", type=int, required=True)
    parser.add_argument("--search-query", action="append", default=[])
    parser.add_argument("--feature-limit", type=int, default=0)
    parser.add_argument("--feature-source", type=Path)
    parser.add_argument("--feature-path", action="append", default=[])
    parser.add_argument("--from-query")
    parser.add_argument("--to-query")
    parser.add_argument("--engine-label", action="append", default=[])
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    origin = urlsplit(args.site_url)
    if (
        origin.scheme not in {"http", "https"}
        or not origin.netloc
        or origin.username
        or origin.password
        or origin.path not in {"", "/"}
        or origin.query
        or origin.fragment
    ):
        parser.error("site URL must be an explicit HTTP(S) origin without credentials")
    if not re.fullmatch(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", args.attempt):
        parser.error("attempt must be a simple identifier")
    if (
        any(
            value <= 0
            for value in (
                args.viewport_width,
                args.viewport_height,
                args.comparison_viewport_width,
                args.comparison_viewport_height,
                args.timeout_ms,
            )
        )
        or args.settle_seconds < 0
    ):
        parser.error("viewport and timeout must be positive; settle seconds nonnegative")
    if (args.viewport_width, args.viewport_height) == (
        args.comparison_viewport_width,
        args.comparison_viewport_height,
    ):
        parser.error("two different content viewports are required")
    if bool(args.from_query) != bool(args.to_query):
        parser.error("route readiness requires paired explicit from/to queries")
    if args.engine_label and not args.from_query:
        parser.error("route engine labels require explicit from/to queries")
    if args.from_query and not args.engine_label:
        parser.error("route readiness requires an explicit actual engine label")
    if not 0 <= args.feature_limit <= 10 or len(args.search_query) > 6:
        parser.error("readiness is bounded to at most six queries and ten features per query")
    proof_inputs = (
        args.container_inspection,
        args.ownership_receipt,
        args.run_id,
        args.nonce,
        args.sandbox_id,
        args.container_id,
    )
    if args.mode == "explore":
        if not all(proof_inputs):
            parser.error(
                "explore requires inspection, ownership receipt, run, nonce, sandbox and CID"
            )
        if (
            args.baseline_receipt
            or args.storage_state
            or args.task_config
            or args.generated_task_id
        ):
            parser.error(
                "anonymous explore does not accept author baseline, authentication or tasks"
            )
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.run_id):
            parser.error("explore run ID must be the actual canonical identifier")
        if not re.fullmatch(r"[a-f0-9]{32}", args.nonce):
            parser.error("explore nonce must be the actual 32-hex ownership nonce")
        if not re.fullmatch(r"[a-f0-9]{64}", args.container_id):
            parser.error("explore CID must be the actual full container ID")
    else:
        if not args.baseline_receipt:
            parser.error("ground/readiness require a fresh coordinator baseline receipt")
        if any(proof_inputs):
            parser.error("existing-container ownership inputs belong exclusively to explore")
    if args.feature_source or args.feature_path:
        if args.mode != "explore" or not args.feature_source or not args.feature_path:
            parser.error(
                "explicit feature exploration requires explore and its actual source packet"
            )
        if args.feature_limit or len(args.feature_path) > 18:
            parser.error("explicit feature paths exclude feature-limit and are bounded to 18")
        if len(set(args.feature_path)) != len(args.feature_path) or any(
            not re.fullmatch(r"/(node|way|relation)/[0-9]+", path) for path in args.feature_path
        ):
            parser.error("feature paths must be distinct actual local feature links")
    if args.mode == "ground":
        if not args.task_config or not args.generated_task_id:
            parser.error("ground requires task-config and generated-task-id")
        if args.start_path or args.search_query or args.from_query or args.feature_limit:
            parser.error("ground inputs come exclusively from the selected task configuration")
        if not re.fullmatch(r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", args.generated_task_id):
            parser.error("task ID must use the generated namespace")
    elif not args.start_path:
        parser.error("readiness/explore require an explicit start-path from the observed homepage")
    return args


def _search_batch(
    page: Any, output: Path, start: str, query: str, name: str, timeout_ms: int
) -> dict[str, Any]:
    """Submit the visible search form and prove coverage of its initial batch."""
    page.goto(start, wait_until="networkidle", timeout=timeout_ms)
    initial = _record(page, output, name + "_initial")
    controls = [control for control in initial["controls"] if control["name"] == "query"]
    if len(controls) != 1 or controls[0]["form"]["method"].upper() != "GET":
        raise ValueError("Actual read-only map search form is unavailable")
    with page.expect_response(
        lambda response: (
            response.request.method == "POST"
            and urlsplit(response.url).path == "/geocoder/search_osm_nominatim"
            and parse_qs(urlsplit(response.url).query).get("query") == [query]
        ),
        timeout=timeout_ms,
    ) as response_event:
        control = page.locator("input#query:visible")
        control.fill(query)
        control.press("Enter")
    response = response_event.value
    response.finished()
    if response.status != 200:
        raise ValueError("Fresh initial geocoder request did not return HTTP200")
    page.wait_for_load_state("networkidle", timeout=timeout_ms)
    snapshot = _record(page, output, name)
    result_locator = page.locator(".results-list > li.search_results_entry")
    coverage = []
    for index, row in enumerate(snapshot["results"]):
        item = result_locator.nth(index)
        item.scroll_into_view_if_needed()
        visible = item.evaluate("""e => {
          const b = e.getBoundingClientRect();
          let left = Math.max(0, b.left), right = Math.min(innerWidth, b.right);
          let top = Math.max(0, b.top), bottom = Math.min(innerHeight, b.bottom);
          for (let p = e.parentElement; p; p = p.parentElement) {
            const s = getComputedStyle(p), r = p.getBoundingClientRect();
            if (/(auto|scroll|hidden|clip)/.test(s.overflowX)) {
              left = Math.max(left, r.left); right = Math.min(right, r.right);
            }
            if (/(auto|scroll|hidden|clip)/.test(s.overflowY)) {
              top = Math.max(top, r.top); bottom = Math.min(bottom, r.bottom);
            }
          }
          const a = e.querySelector('a.set_position[href]');
          return {label: a.innerText.trim(), url: a.href,
            visible_rect: {left, top, right, bottom},
            full_row_visible: left <= b.left + 1 && right >= b.right - 1
              && top <= b.top + 1 && bottom >= b.bottom - 1};
        }""")
        if (visible["label"], visible["url"]) != (row["label"], row["url"]):
            raise ValueError("Initial result batch changed during scroll coverage")
        screenshot = f"{name}_row_{index:02}.png"
        page.screenshot(path=str(output / screenshot))
        coverage.append({"row_index": index, **visible, "screenshot": screenshot})
    fields = parse_qs(response.request.post_data or "")
    return {
        "query": control.input_value(),
        "initial": initial,
        "request": {
            "method": "POST",
            "url": response.url,
            "status": response.status,
            "view": {
                key: float(fields[key][0])
                for key in ("zoom", "minlon", "minlat", "maxlon", "maxlat")
            },
            "post_field_names": sorted(fields),
        },
        "snapshot": snapshot,
        "scroll_coverage": coverage,
        "more_results_clicked": False,
    }


def _feature_snapshot(
    page: Any, output: Path, target: str, name: str, *, strict: bool = True
) -> dict[str, Any]:
    """Verify a displayed feature identity and archive selectors for terminal checks."""
    if not re.fullmatch(r"/(node|way|relation)/[0-9]+", urlsplit(target).path):
        raise ValueError("Feature target must be a newly observed local feature link")
    response = page.goto(target, wait_until="networkidle")
    snapshot = _record(page, output, name)
    snapshot["request"] = {
        "method": "GET",
        "url": response.url if response is not None else target,
        "status": response.status if response is not None else None,
    }
    snapshot["terminal_targets"] = page.evaluate("""() => {
      const visible = e => {
        const b = e.getBoundingClientRect();
        return !!(b.width && b.height) && getComputedStyle(e).visibility !== 'hidden';
      };
      const selector = e => {
        const parts = [];
        for (let n = e; n; n = n.parentElement) {
          if (n.id) { parts.unshift('#' + CSS.escape(n.id)); break; }
          let part = n.tagName.toLowerCase();
          const siblings = n.parentElement
            ? Array.from(n.parentElement.children).filter(s => s.tagName === n.tagName) : [n];
          if (siblings.length > 1) part += ':nth-of-type(' + (siblings.indexOf(n) + 1) + ')';
          parts.unshift(part);
        }
        return parts.join(' > ');
      };
      return {
        headings: Array.from(document.querySelectorAll('h1,h2,h3')).filter(visible)
          .map(e => ({text: e.innerText, selector: selector(e)})),
        detail_rows: Array.from(document.querySelectorAll('table tr')).filter(visible)
          .map(e => ({text: e.innerText, selector: selector(e),
            cells: Array.from(e.querySelectorAll('th,td')).map(c => c.innerText.trim())}))
      };
    }""")
    feature_type, feature_id = urlsplit(target).path.strip("/").split("/")
    identity = re.compile(
        rf"\b{re.escape(feature_type)}\b.*\b{re.escape(feature_id)}\b", re.IGNORECASE
    )
    identities = [
        row for row in snapshot["terminal_targets"]["headings"] if identity.search(row["text"])
    ]
    details = [
        row
        for row in snapshot["terminal_targets"]["detail_rows"]
        if len(row["cells"]) >= 2 and all(cell.strip() for cell in row["cells"][:2])
    ]
    ready = (
        snapshot["request"]["status"] == 200
        and urlsplit(snapshot["url"]).path == urlsplit(target).path
        and len(identities) == 1
        and bool(details)
    )
    snapshot["feature_readiness"] = {
        "valid_identity_and_details": ready,
        "requested_path": urlsplit(target).path,
        "identity_heading_count": len(identities),
        "nonempty_detail_row_count": len(details),
    }
    if ready:
        snapshot["identity_target"] = identities[0]
        snapshot["detail_target"] = details[0]
    (output / f"{name}.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    if strict and not ready:
        raise ValueError("Feature GET lacks its real visible identity and details")
    return snapshot


def _route_snapshot(page: Any, output: Path, name: str, responses: list[dict]) -> dict:
    """Record actual Directions text, selectors, controls and its backend response."""
    snapshot = _record(page, output, name)
    snapshot["terminal_targets"] = page.evaluate("""() => {
      const visible = e => {
        const b = e.getBoundingClientRect();
        return !!(b.width && b.height) && getComputedStyle(e).visibility !== 'hidden';
      };
      const selector = e => {
        const parts = [];
        for (let n = e; n; n = n.parentElement) {
          if (n.id) { parts.unshift('#' + CSS.escape(n.id)); break; }
          let part = n.tagName.toLowerCase();
          const siblings = n.parentElement
            ? Array.from(n.parentElement.children).filter(s => s.tagName === n.tagName) : [n];
          if (siblings.length > 1) part += ':nth-of-type(' + (siblings.indexOf(n) + 1) + ')';
          parts.unshift(part);
        }
        return parts.join(' > ');
      };
      return Array.from(document.querySelectorAll('h1,h2,h3,p,table')).filter(visible)
        .map(e => ({tag: e.tagName.toLowerCase(), text: e.innerText, selector: selector(e)}));
    }""")
    summaries = [
        target
        for target in snapshot["terminal_targets"]
        if target["tag"] == "p"
        and re.fullmatch(r"Distance: [^\n]+\. Time: [^\n]+\.", target["text"].strip())
    ]
    headings = [
        target
        for target in snapshot["terminal_targets"]
        if target["tag"] in {"h1", "h2", "h3"} and target["text"].strip() == "Directions"
    ]
    tables = [
        target
        for target in snapshot["terminal_targets"]
        if target["tag"] == "table" and re.search(r"1\. Start", target["text"])
    ]
    backend = [
        response
        for response in responses
        if response["method"] == "GET"
        and urlsplit(response["url"]).path.startswith("/osrm/")
        and "/route/v1/" in urlsplit(response["url"]).path
    ]
    ready = (
        len(summaries) == len(headings) == len(tables) == 1
        and bool(backend)
        and backend[-1]["status"] == 200
    )
    snapshot["route_readiness"] = {
        "valid_visible_directions_and_backend": ready,
        "summary_match_count": len(summaries),
        "heading_match_count": len(headings),
        "instruction_table_match_count": len(tables),
        "backend_responses": backend,
    }
    if ready:
        snapshot.update(
            summary_target=summaries[0],
            heading_target=headings[0],
            instructions_target=tables[0],
            backend_request=backend[-1],
        )
    if ready:
        table_rows = page.locator(snapshot["instructions_target"]["selector"] + " tr")
        snapshot["instruction_coverage"] = []
        for index in range(table_rows.count()):
            row = table_rows.nth(index)
            row.scroll_into_view_if_needed()
            visible = row.evaluate("""e => {
              const b = e.getBoundingClientRect();
              let left = Math.max(0, b.left), right = Math.min(innerWidth, b.right);
              let top = Math.max(0, b.top), bottom = Math.min(innerHeight, b.bottom);
              for (let p = e.parentElement; p; p = p.parentElement) {
                const s = getComputedStyle(p), r = p.getBoundingClientRect();
                if (/(auto|scroll|hidden|clip)/.test(s.overflowX)) {
                  left = Math.max(left, r.left); right = Math.min(right, r.right);
                }
                if (/(auto|scroll|hidden|clip)/.test(s.overflowY)) {
                  top = Math.max(top, r.top); bottom = Math.min(bottom, r.bottom);
                }
              }
              return {text: e.innerText.trim(), visible_rect: {left, top, right, bottom},
                full_row_visible: left <= b.left + 1 && right >= b.right - 1
                  && top <= b.top + 1 && bottom >= b.bottom - 1};
            }""")
            screenshot = f"{name}_instruction_{index:02}.png"
            page.screenshot(path=str(output / screenshot))
            snapshot["instruction_coverage"].append(
                {"row_index": index, **visible, "screenshot": screenshot}
            )
        snapshot["route_readiness"]["complete_instruction_row_coverage"] = bool(
            snapshot["instruction_coverage"]
        ) and all(row["full_row_visible"] for row in snapshot["instruction_coverage"])
    (output / f"{name}.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return snapshot


def _select_route_endpoints(config: dict, observed: dict) -> dict:
    """Select full displayed endpoint labels using the configured visible search rule."""
    mode, parameters = config["mode"], config["parameters"]
    if mode in {"navigate-directions", "retrieve-route-profile-comparison"}:
        return {
            "from_label": parameters["from_label"],
            "to_label": parameters["to_label"],
            "sources": ["/initial"],
        }
    rows = observed["searches"][0]["batches"][0]["snapshot"]["results"]
    sources = ["/searches"]
    if mode == "navigate-numeric-extremes-directions":
        candidates = []
        for index, row in enumerate(rows):
            parts = [part.strip() for part in row["label"].split(",")]
            if row["category"] == parameters["category"] and all(
                part in parts for part in parameters["address_components"]
            ):
                if not re.fullmatch(r"[0-9]+", parts[1]):
                    raise ValueError("Every matching endpoint must show an integer building number")
                candidates.append((int(parts[1]), row["label"], index))
        candidates.sort()
        if len(candidates) < 2 or candidates[0][0] == candidates[-1][0]:
            raise ValueError("Numeric endpoint selection needs distinct displayed extremes")
        return {
            "from_label": candidates[0][1],
            "to_label": min(candidates, key=lambda item: (-item[0], item[1]))[1],
            "sources": sources,
        }
    if mode == "navigate-dominant-group-directions":
        offset = parameters["component_from_end"]
        groups = {}
        for row in rows:
            if row["category"] in parameters["categories"]:
                part = [part.strip() for part in row["label"].split(",")][-offset]
                groups.setdefault(part, []).append(row)
        if len(groups) < 2:
            raise ValueError("Dominant-group selection needs at least two observed groups")
        largest = min(groups, key=lambda part: (-len(groups[part]), part))
        matching = [
            row for row in groups[largest] if row["category"] == parameters["target_category"]
        ]
    else:
        keys = [
            {(row["category"], row["label"]) for row in search["batches"][0]["snapshot"]["results"]}
            for search in observed["searches"]
        ]
        common = keys[0] & keys[1]
        if mode == "navigate-record-difference-directions":
            common -= keys[2]
        elif mode != "navigate-common-record-directions":
            raise ValueError(f"Unsupported route endpoint selection: {mode}")
        matching = [
            row
            for row in rows
            if (row["category"], row["label"]) in common
            and row["category"] == parameters["category"]
            and all(
                part in [piece.strip() for piece in row["label"].split(",")]
                for part in parameters["address_components"]
            )
        ]
        matching.sort(key=lambda row: (row["label"], row["category"]))
        if mode == "navigate-record-difference-directions":
            if len(matching) < 2:
                raise ValueError("Ranked difference needs multiple visible matching candidates")
            matching = matching[:1]
    if len(matching) != 1:
        raise ValueError("Route endpoint rule does not uniquely select a full displayed label")
    if parameters["from_label"] == matching[0]["label"]:
        raise ValueError("Directions endpoints must be different displayed labels")
    return {
        "from_label": parameters["from_label"],
        "to_label": matching[0]["label"],
        "sources": sources,
    }


def _collect_routes(
    page: Any,
    output: Path,
    initial: dict,
    site_url: str,
    from_label: str,
    to_label: str,
    engine_labels: list[str],
    responses: list[dict],
    timeout_ms: int,
    settle_seconds: float,
    *,
    strict: bool,
    name_prefix: str,
    route_link_zoom: int | None = None,
) -> list[dict]:
    """Enter observed Directions controls and record each actual profile result."""
    target = site_url + "/directions"
    if not any(link["url"] == target for link in initial["links"]):
        raise ValueError("Directions link is absent from the actual starting page")
    routes = []
    for index, engine in enumerate(engine_labels):
        page.goto(target, wait_until="networkidle", timeout=timeout_ms)
        form = _record(page, output, f"{name_prefix}_directions_{index:02}")
        controls = {
            control["name"]: control
            for control in form["controls"]
            if control["name"] in {"route_from", "route_to", "routing_engines"}
        }
        if set(controls) != {"route_from", "route_to", "routing_engines"} or any(
            control["form"]["method"].upper() != "GET" for control in controls.values()
        ):
            raise ValueError("Actual read-only Directions controls are unavailable")
        options = controls["routing_engines"]["options"]
        if engine not in {option["label"] for option in options}:
            raise ValueError("Requested engine is absent from the visible profile options")
        response_start = len(responses)
        page.locator("input#route_from:visible").fill(from_label)
        page.locator("input#route_to:visible").fill(to_label)
        page.locator("select[name='routing_engines']:visible").select_option(label=engine)
        page.locator("input#route_to:visible").press("Enter")
        page.wait_for_timeout(int(settle_seconds * 1000))
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
        snapshot = _route_snapshot(
            page, output, f"{name_prefix}_{index:02}", responses[response_start:]
        )
        route = {
            "from_query": from_label,
            "to_query": to_label,
            "engine_label": engine,
            "observation": snapshot,
        }
        routes.append(route)
        values = {
            control["name"]: control
            for control in snapshot["controls"]
            if control["name"] in {"route_from", "route_to", "routing_engines"}
        }
        selected = [
            option["label"] for option in values["routing_engines"]["options"] if option["selected"]
        ]
        exact_inputs = (
            values["route_from"]["value"] == from_label
            and values["route_to"]["value"] == to_label
            and selected == [engine]
        )
        snapshot["route_readiness"]["exact_endpoint_and_profile_inputs"] = exact_inputs
        if strict and (
            not exact_inputs
            or not snapshot["route_readiness"]["valid_visible_directions_and_backend"]
            or not snapshot["route_readiness"].get("complete_instruction_row_coverage", False)
        ):
            (output / "route_failure.json").write_text(
                json.dumps(route, ensure_ascii=False, indent=2)
            )
            raise ValueError(
                "Fresh Directions lacks exact visible inputs, summary or backend HTTP200"
            )
        if route_link_zoom is not None:
            # The frontend rounds its generated link at the zoom before routing,
            # then fits the map. Recompute through the public controls at the
            # task's explicit zoom so the two viewports generate their own link.
            generation = route["route_link_generation"] = {
                "initial_observation": snapshot,
                "zoom_clicks": [],
            }
            map_hash = parse_qs(urlsplit(page.url).fragment).get("map", [])
            if len(map_hash) != 1:
                raise ValueError("Complete Directions lacks its actual map zoom hash")
            zoom = int(map_hash[0].split("/", 1)[0])
            while zoom != route_link_zoom:
                next_zoom = zoom + (1 if zoom < route_link_zoom else -1)
                selector = (
                    "a.control-button.zoomin:visible" if zoom < route_link_zoom
                    else "a.control-button.zoomout:visible"
                )
                click = {"selector": selector, "before_url": page.url}
                generation["zoom_clicks"].append(click)
                page.locator(selector).click(timeout=timeout_ms)
                page.wait_for_function(
                    """zoom => {
                      const map = new URLSearchParams(location.hash.slice(1)).get('map');
                      return map && Number(map.split('/')[0]) === zoom;
                    }""",
                    arg=next_zoom,
                    timeout=timeout_ms,
                )
                zoom = int(parse_qs(urlsplit(page.url).fragment)["map"][0].split("/", 1)[0])
                if zoom != next_zoom:
                    raise ValueError("Visible map zoom control did not reach its recorded zoom")
                click["after_url"] = page.url
            generation.update(zoom=zoom, zoom_url=page.url)
            response_start = len(responses)
            generation["recompute_selector"] = ".directions_form input[name='commit']:visible"
            with page.expect_response(
                lambda response: response.request.method == "GET"
                and urlsplit(response.url).path.startswith("/osrm/")
                and "/route/v1/" in urlsplit(response.url).path,
                timeout=timeout_ms,
            ) as response_event:
                page.locator(generation["recompute_selector"]).click(timeout=timeout_ms)
            response_event.value.finished()
            page.wait_for_function(
                r"""() => {
                  const visible = e => {
                    const b = e.getBoundingClientRect();
                    return !!(b.width && b.height)
                      && getComputedStyle(e).visibility !== 'hidden';
                  };
                  if (Array.from(document.querySelectorAll('.search_results_error'))
                      .some(visible)) return true;
                  const tables = Array.from(document.querySelectorAll('table')).filter(visible);
                  return tables.length === 1
                    && /\d+\. Reach destination/.test(tables[0].innerText);
                }""",
                timeout=timeout_ms,
            )
            # Go can fit the map to a different zoom. Its generated query is
            # already fixed; restore the visible zoom without another Go so
            # opening this link uses the same coordinate precision.
            restoration = generation["post_recompute_zoom"] = {
                "before_url": page.url,
                "before_query": urlsplit(page.url).query,
                "before_fragment": urlsplit(page.url).fragment,
                "zoom_clicks": [],
            }
            map_hash = parse_qs(urlsplit(page.url).fragment).get("map", [])
            if not map_hash:
                # The route replaces the URL without a hash. When Go leaves
                # the map unchanged, only a real zoom move exposes its zoom.
                click = {
                    "selector": "a.control-button.zoomout:visible",
                    "before_url": page.url,
                    "before_zoom": None,
                }
                restoration["zoom_clicks"].append(click)
                page.locator(click["selector"]).click(timeout=timeout_ms)
                page.wait_for_function(
                    r"""() => {
                      const map = new URLSearchParams(location.hash.slice(1)).get('map');
                      return map && /^\d+\/[-+]?\d+(?:\.\d+)?\/[-+]?\d+(?:\.\d+)?$/.test(map);
                    }""",
                    timeout=timeout_ms,
                )
                map_hash = parse_qs(urlsplit(page.url).fragment)["map"]
                click.update(after_url=page.url, after_zoom=int(map_hash[0].split("/", 1)[0]))
            zoom = int(map_hash[0].split("/", 1)[0])
            while zoom != route_link_zoom:
                next_zoom = zoom + (1 if zoom < route_link_zoom else -1)
                selector = (
                    "a.control-button.zoomin:visible" if zoom < route_link_zoom
                    else "a.control-button.zoomout:visible"
                )
                click = {"selector": selector, "before_url": page.url, "before_zoom": zoom}
                restoration["zoom_clicks"].append(click)
                page.locator(selector).click(timeout=timeout_ms)
                page.wait_for_function(
                    """zoom => {
                      const map = new URLSearchParams(location.hash.slice(1)).get('map');
                      return map && Number(map.split('/')[0]) === zoom;
                    }""",
                    arg=next_zoom,
                    timeout=timeout_ms,
                )
                zoom = int(parse_qs(urlsplit(page.url).fragment)["map"][0].split("/", 1)[0])
                if zoom != next_zoom:
                    raise ValueError("Visible map zoom control did not restore its recorded zoom")
                click.update(after_url=page.url, after_zoom=zoom)
            restoration.update(zoom=zoom, after_url=page.url, after_query=urlsplit(page.url).query)
            snapshot = route["observation"] = _route_snapshot(
                page, output, f"{name_prefix}_{index:02}_link", responses[response_start:]
            )
            values = {control["name"]: control for control in snapshot["controls"]}
            selected = [
                option["label"] for option in values["routing_engines"]["options"]
                if option["selected"]
            ]
            if (
                values["route_from"]["value"] != from_label
                or values["route_to"]["value"] != to_label
                or selected != [engine]
                or not snapshot["route_readiness"]["valid_visible_directions_and_backend"]
                or not snapshot["route_readiness"].get("complete_instruction_row_coverage", False)
            ):
                raise ValueError("Recomputed route link lacks exact inputs or complete Directions")
    return routes


def _compare_route_recordings(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Compare visible endpoints/profile, route text and backend; ignore map presentation."""
    comparisons = []
    for first, second in zip(primary, secondary, strict=True):
        same = all(first[key] == second[key] for key in ("from_query", "to_query", "engine_label"))
        same = (
            same
            and first["observation"]["backend_request"] == second["observation"]["backend_request"]
        )
        same = same and all(
            first["observation"][key]["text"] == second["observation"][key]["text"]
            for key in ("summary_target", "instructions_target", "heading_target")
        )
        comparisons.append(
            {
                "engine_label": first["engine_label"],
                "same_semantic_route": same,
                "map_fragment_is_presentation": True,
            }
        )
    return comparisons


def build_extension_definition(config: dict, observed: dict) -> tuple[dict, list, list]:
    """Derive task rules from complete visible records or freshly displayed Directions."""
    mode, parameters = config["mode"], dict(config["parameters"])
    base_modes = {
        "retrieve-category-summary",
        "retrieve-location-summary",
        "retrieve-address-comparison",
        "navigate-constrained-feature",
    }
    if mode in base_modes:
        definition, evidence, checks = build_task_definition(config, observed)
    else:
        scope = (
            "Starting from the provided map view, submit each requested search without panning "
            "or zooming before submission. Inspect every row in each complete initial displayed "
            "batch, including rows reachable by scrolling. Do not select More results. Later "
            "batches and global matches are excluded. "
        )
        if mode.startswith("retrieve-record-") or mode in {
            "navigate-common-record-directions",
            "navigate-record-difference-directions",
        }:
            scope += (
                "A displayed record is the exact pair (displayed category, complete displayed "
                "label); count the same pair at most once within or across batches, even if "
                "its displayed position changes. "
            )
        rows = (
            observed["searches"][0]["batches"][0]["snapshot"]["results"]
            if observed["searches"]
            else []
        )
        indexed = {}
        for query_index, search in enumerate(observed["searches"]):
            for row_index, row in enumerate(search["batches"][0]["snapshot"]["results"]):
                key = (row["category"], row["label"])
                fact = indexed.setdefault(key, {"row": row, "queries": set(), "pointers": []})
                fact["queries"].add(query_index)
                fact["pointers"].append(
                    f"/searches/{query_index}/batches/0/snapshot/results/{row_index}"
                )
        values, sources = [], []
        sort_key = lambda item: (
            item[0].get("label", item[0].get("address_component", "")),
            item[0].get("category", ""),
        )
        goal = None
        if mode == "retrieve-filtered-features":
            for index, row in enumerate(rows):
                parts = [part.strip() for part in row["label"].split(",")]
                if row["category"] == parameters["category"] and all(
                    part in parts for part in parameters["address_components"]
                ):
                    values.append({"category": row["category"], "label": row["label"]})
                    sources.append([f"/search/results/{index}"])
            goal = scope + (
                "Search for {query}. Return all records with category exactly "
                "{category} whose comma-separated label contains every exact component "
                "in {address_components}, trimming spaces around components. Return "
                "category and the complete label, ordered alphabetically by label."
            )
        elif mode in {"retrieve-address-group-summary", "retrieve-record-union-address-summary"}:
            offset = parameters["component_from_end"]
            if type(offset) is not int or offset < 1:
                raise ValueError("Address grouping requires a positive component position")
            members = (
                list(indexed.values())
                if mode == "retrieve-record-union-address-summary"
                else [
                    {"row": row, "pointers": [f"/search/results/{index}"]}
                    for index, row in enumerate(rows)
                ]
            )
            groups = {}
            for fact in members:
                if fact["row"]["category"] in parameters["categories"]:
                    part = [piece.strip() for piece in fact["row"]["label"].split(",")][-offset]
                    groups.setdefault(part, []).append(fact)
            for part, facts in groups.items():
                values.append(
                    {
                        "address_component": part,
                        "result_count": len(facts),
                        "distinct_category_count": len({fact["row"]["category"] for fact in facts}),
                    }
                )
                sources.append([pointer for fact in facts for pointer in fact["pointers"]])
            if mode == "retrieve-address-group-summary":
                order = "Order groups alphabetically by address_component."
                lead = "Search for {query}."
            else:
                sort_key = lambda item: (-item[0]["result_count"], item[0]["address_component"])
                order = "Order groups by result_count descending, then address_component alphabetically."
                lead = "Search for every text in {queries}, returning to the starting view each time. Combine and deduplicate the displayed records from all initial batches."
            goal = (
                scope
                + lead
                + (
                    " Keep only categories in {categories}, matching exactly. "
                    "Split each complete label at commas and trim spaces. Group by "
                    "the component {component_from_end} positions from the end "
                    "(last component is position 1). Return address_component, "
                    "result_count and distinct_category_count. "
                )
                + order
            )
        elif mode in {
            "retrieve-record-overlap",
            "retrieve-record-difference",
            "retrieve-record-membership-rank",
        }:
            queries = parameters["queries"]
            if not 2 <= len(queries) <= 3 or len(set(queries)) != len(queries):
                raise ValueError("Record-set tasks require two or three different search texts")
            for fact in indexed.values():
                membership = fact["queries"]
                if mode == "retrieve-record-overlap":
                    keep = len(membership) == len(queries)
                elif mode == "retrieve-record-difference":
                    keep = 0 in membership and 1 not in membership
                else:
                    keep = len(membership) >= parameters["minimum_query_count"]
                if keep:
                    value = {"category": fact["row"]["category"], "label": fact["row"]["label"]}
                    if mode != "retrieve-record-difference":
                        value["query_count"] = len(membership)
                    values.append(value)
                    sources.append(fact["pointers"])
            goal = (
                scope
                + "Search for every text in {queries}, returning to the starting view each time. "
            )
            if mode == "retrieve-record-overlap":
                goal += (
                    "Return the records present in every initial batch. Return category, "
                    "complete label and query_count. Order by label, then category alphabetically."
                )
            elif mode == "retrieve-record-difference":
                goal += (
                    "Return records present in the first batch but absent from the second "
                    "batch. Return category and complete label, ordered by label then category alphabetically."
                )
            else:
                sort_key = lambda item: (
                    -item[0]["query_count"],
                    item[0]["label"],
                    item[0]["category"],
                )
                goal += (
                    "Keep records present in at least {minimum_query_count} search batches. "
                    "Return category, complete label and query_count. Order by query_count "
                    "descending, then label and category alphabetically."
                )
        elif mode == "retrieve-multi-query-region-summary":
            for search_index, search in enumerate(observed["searches"]):
                for region in parameters["regions"]:
                    matching = [
                        row
                        for row in search["batches"][0]["snapshot"]["results"]
                        if region in [part.strip() for part in row["label"].split(",")]
                    ]
                    values.append(
                        {
                            "query": search["query"],
                            "region": region,
                            "result_count": len(matching),
                            "distinct_category_count": len({row["category"] for row in matching}),
                            "category_count": sum(
                                row["category"] == parameters["category"] for row in matching
                            ),
                        }
                    )
                    sources.append([f"/searches/{search_index}"])
            sort_key = None
            goal = scope + (
                "Search for every text in {queries}, returning to the starting view each time. "
                "For each query compare each region in {regions}, matching an exact comma-separated "
                "label component after trimming spaces. Return query, region, result_count, "
                "distinct_category_count and category_count (category exactly {category}). "
                "Use zero for an absent region in that initial batch. Return query groups in "
                "the stated query order and regions in the stated region order within each group."
            )
        elif mode == "retrieve-multi-query-category-comparison":
            if len(parameters["queries"]) != 2:
                raise ValueError("Category comparison requires exactly two initial batches")
            for category in parameters["categories"]:
                counts = [
                    sum(
                        row["category"] == category
                        for row in search["batches"][0]["snapshot"]["results"]
                    )
                    for search in observed["searches"]
                ]
                values.append(
                    {
                        "category": category,
                        "first_query_count": counts[0],
                        "second_query_count": counts[1],
                        "second_minus_first": counts[1] - counts[0],
                    }
                )
                sources.append(["/searches"])
            sort_key = None
            goal = scope + (
                "Search for both texts in {queries}, returning to the starting view "
                "before each. For each category in {categories}, matching exactly, "
                "return category, first_query_count, second_query_count and "
                "second_minus_first. Return categories in the stated order; count "
                "every displayed row of that category, including zero if absent."
            )
        elif mode == "retrieve-record-union-category-summary":
            groups = {}
            for fact in indexed.values():
                groups.setdefault(fact["row"]["category"], []).append(fact)
            for category, facts in groups.items():
                values.append({"category": category, "record_count": len(facts)})
                sources.append([pointer for fact in facts for pointer in fact["pointers"]])
            sort_key = lambda item: (-item[0]["record_count"], item[0]["category"])
            goal = scope + (
                "Search for every text in {queries}, returning to the starting view "
                "each time. Combine all initial batches and deduplicate displayed "
                "records. Group by exact category; return category and record_count. "
                "Order by record_count descending, then category alphabetically."
            )
        elif mode == "retrieve-record-constrained-union-statistics":
            matching = [
                fact
                for fact in indexed.values()
                if fact["row"]["category"] == parameters["category"]
                and all(
                    part in [piece.strip() for piece in fact["row"]["label"].split(",")]
                    for part in parameters["address_components"]
                )
            ]
            if not matching:
                raise ValueError("Constrained union has no freshly observed matching records")
            postal = {
                tuple(piece.strip() for piece in fact["row"]["label"].split(","))[
                    -parameters["postal_component_from_end"]
                ]
                for fact in matching
            }
            values = [
                {
                    "matching_record_count": len(matching),
                    "distinct_postal_component_count": len(postal),
                }
            ]
            sources = [[pointer for fact in matching for pointer in fact["pointers"]]]
            goal = scope + (
                "Search for every text in {queries}, returning to the starting view "
                "each time. Combine and deduplicate displayed records. Keep category "
                "exactly {category} and labels containing every exact comma-separated "
                "component in {address_components}. Return matching_record_count and "
                "distinct_postal_component_count, using the component "
                "{postal_component_from_end} positions from the end as the postal "
                "component (last component is position 1)."
            )
        elif mode == "retrieve-record-address-extreme-comparison":
            matching = []
            for fact in indexed.values():
                row = fact["row"]
                parts = [part.strip() for part in row["label"].split(",")]
                if row["category"] == parameters["category"] and parameters["region"] in parts:
                    if not re.fullmatch(r"[0-9]+", parts[1]) or not re.fullmatch(
                        r"[0-9]{5}", parts[-2]
                    ):
                        raise ValueError(
                            "All constrained addresses need visible integer numbers and postal codes"
                        )
                    matching.append((int(parts[1]), row["label"], parts[-2], fact["pointers"]))
            matching.sort()
            if len(matching) < 2 or matching[0][0] == matching[-1][0]:
                raise ValueError("Address extrema comparison needs different visible numbers")
            lower = min(matching, key=lambda item: (item[0], item[1]))
            higher = min(matching, key=lambda item: (-item[0], item[1]))
            values = [
                {
                    "lower_number_label": lower[1],
                    "lower_building_number": lower[0],
                    "lower_postal_code": lower[2],
                    "higher_number_label": higher[1],
                    "higher_building_number": higher[0],
                    "higher_postal_code": higher[2],
                    "building_number_difference": higher[0] - lower[0],
                    "matching_result_count": len(matching),
                }
            ]
            sources = [[pointer for item in matching for pointer in item[3]]]
            goal = scope + (
                "Search for every text in {queries}, returning to the starting view each time. "
                "Combine and deduplicate displayed records. Keep category exactly {category} and labels "
                "containing the exact region component {region}. Compare the minimum "
                "and maximum integer building numbers, the component directly after "
                "the feature name. Break equal-number ties by full label alphabetically. "
                "Return lower_number_label, lower_building_number, lower_postal_code, "
                "higher_number_label, higher_building_number, higher_postal_code, "
                "building_number_difference (higher minus lower), and "
                "matching_result_count. Preserve full labels and five-digit postal codes."
            )
        elif mode == "retrieve-route-profile-comparison":
            from decimal import Decimal

            for index, route in enumerate(observed["routes"]):
                snapshot = route["observation"]
                summary = snapshot["summary_target"]["text"].strip()
                match = re.fullmatch(
                    r"Distance: ([0-9.]+)(km|m)\. Time: ([0-9]+):([0-9]{2})\.", summary
                )
                if not match:
                    raise ValueError(
                        "Route comparison requires the actual displayed distance and H:MM time"
                    )
                distance = Decimal(match[1]) * (1000 if match[2] == "km" else 1)
                if distance != int(distance):
                    raise ValueError(
                        "Displayed distance does not convert to an integer metre count"
                    )
                instructions = [
                    table
                    for table in snapshot["tables"]
                    if table and len(table[0]) >= 2 and table[0][1].startswith("1. Start")
                ]
                if len(instructions) != 1:
                    raise ValueError(
                        "Route comparison requires one actual visible instructions table"
                    )
                values.append(
                    {
                        "engine_label": route["engine_label"],
                        "distance_metres": int(distance),
                        "displayed_time_minutes": int(match[3]) * 60 + int(match[4]),
                        "instruction_count": len(instructions[0]),
                    }
                )
                sources.append([f"/routes/{index}"])
            sort_key = None
            goal = (
                "Open Directions and enter {from_label} as From and {to_label} as To, "
                "using the full labels exactly. Compare every profile in {engine_labels}. "
                "Return engine_label, distance_metres (convert km to metres if needed), "
                "displayed_time_minutes (interpret H:MM as hours and minutes), and "
                "instruction_count (count numbered instruction rows, including start and "
                "arrival). Use displayed rounded values rather than backend precision. "
                "Return profiles in the stated order."
            )
        elif (
            mode.startswith("navigate-") and mode.endswith("-directions")
        ) or mode == "navigate-directions":
            route = observed["terminal_route"]
            snapshot = route["observation"]
            endpoints = observed["route_endpoints"]
            parameters.update(from_label=endpoints["from_label"], to_label=endpoints["to_label"])
            if mode == "navigate-directions":
                goal = "Open Directions, enter {from_label} as From and {to_label} as To using the full labels exactly. "
            elif mode == "navigate-numeric-extremes-directions":
                goal = scope + (
                    "Search for {query}. Keep category exactly {category} and labels "
                    "containing every exact comma-separated component in "
                    "{address_components}. Select the minimum integer building number "
                    "as From and the maximum as To (number directly after the feature "
                    "name); break equal-number ties by full label alphabetically. Enter "
                    "the selected complete displayed labels in Directions. "
                )
            elif mode == "navigate-dominant-group-directions":
                goal = scope + (
                    "Search for {query}. Keep categories in {categories}. Group labels "
                    "by the component {component_from_end} positions from the end "
                    "(last component is position 1). Choose the largest group by row "
                    "count, breaking ties by component alphabetically. Within it select "
                    "the unique category {target_category} result as To. Enter its full "
                    "displayed label in Directions, with {from_label} as From. "
                )
            elif mode == "navigate-common-record-directions":
                goal = scope + (
                    "Search for both texts in {queries}, returning to the starting "
                    "view each time. Select the unique displayed record common to both "
                    "initial batches, with category {category} and every exact "
                    "comma-separated component in {address_components}. Enter its full "
                    "label as To in Directions and {from_label} as From. "
                )
            else:
                goal = scope + (
                    "Search for every text in {queries}, returning to the starting "
                    "view each time. Keep displayed records common to the first two "
                    "initial batches but absent from the third. Apply category exactly "
                    "{category} and every exact comma-separated component in "
                    "{address_components}. Select the alphabetically first complete "
                    "label (then category if tied) as To. Enter its full label in "
                    "Directions with {from_label} as From. "
                )
            goal += (
                "Use {engine_labels}. Once the complete Directions panel is displayed, use "
                "the visible Zoom In or Zoom Out controls to set the map to zoom level "
                "{route_link_zoom} (check the #map= zoom in the browser address). Press Go "
                "to recompute the route and wait for the complete Directions panel. Check "
                "the zoom again. If #map= is missing from the browser address, click the "
                "visible Zoom Out control once to show the actual zoom. Use the visible "
                "controls to restore level {route_link_zoom} "
                "if it changed, without pressing Go again. Then open "
                "its generated route "
                "link in the browser address bar and leave the complete computed Directions "
                "panel visible. Do not edit map data."
            )
            checks, evidence = [], []
            evidence.append(
                {
                    "field": "/task_definition/parameters/route_link_zoom",
                    "artifact": "observation",
                    "location": "/routes/0/route_link_generation/zoom",
                    "modality": "text",
                    "derivation": "Actual URL hash zoom reached through visible controls before Go.",
                }
            )
            for field in ("from_label", "to_label"):
                evidence.append(
                    {
                        "field": f"/task_definition/parameters/{field}",
                        "artifact": "observation",
                        "location": f"/route_endpoints/{field}",
                        "modality": "text",
                        "derivation": "Apply the visible-record rule to the fresh search windows.",
                    }
                )
            controls = snapshot["controls"]
            for name, field, selector in (
                ("route_from", "from_input", "#route_from"),
                ("route_to", "to_input", "#route_to"),
                ("routing_engines", "engine_value", "select[name='routing_engines']"),
            ):
                control_index = next(
                    index for index, control in enumerate(controls) if control["name"] == name
                )
                parameters[field] = controls[control_index]["value"]
                checks.append({"kind": "input_value", "parameter": field, "selector": selector})
                evidence.append(
                    {
                        "field": f"/task_definition/parameters/{field}",
                        "artifact": "observation",
                        "location": f"/terminal_route/observation/controls/{control_index}/value",
                        "modality": "text",
                        "derivation": "Exact visible Directions control value.",
                    }
                )
            for field, target in (
                ("route_summary", "summary_target"),
                ("route_heading", "heading_target"),
                ("route_instructions", "instructions_target"),
            ):
                parameters[field] = snapshot[target]["text"]
                checks.append(
                    {
                        "kind": "visible_text_exact",
                        "parameter": field,
                        "selector": snapshot[target]["selector"],
                    }
                )
                evidence.append(
                    {
                        "field": f"/task_definition/parameters/{field}",
                        "artifact": "observation",
                        "location": f"/terminal_route/observation/{target}/text",
                        "modality": "text",
                        "derivation": "Actual displayed Directions terminal text and its observed selector.",
                    }
                )
            navigation = route["navigation_request"]
            path = urlsplit(navigation["url"])
            definition = {
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
                        "url_template": "__MAP__" + path.path,
                        "http_method": navigation["method"],
                        "response_status": navigation["status"],
                        "query_params": navigation["query_params"],
                        "last_event_only": True,
                    }
                ],
            }
            for field, source in (
                ("url_template", "url"),
                ("http_method", "method"),
                ("response_status", "status"),
            ):
                evidence.append(
                    {
                        "field": f"/task_definition/completion_events/0/{field}",
                        "artifact": "observation",
                        "location": "/terminal_route/navigation_request/" + source,
                        "modality": "text",
                        "derivation": "Actual route document GET; path and query are scored separately.",
                    }
                )
            for field in ("engine", "route"):
                evidence.append(
                    {
                        "field": f"/task_definition/completion_events/0/query_params/{field}",
                        "artifact": "observation",
                        "location": f"/terminal_route/navigation_request/query_params/{field}",
                        "modality": "text",
                        "derivation": "Exact engine and coordinates of the opened route link.",
                    }
                )
        else:
            raise ValueError(f"Unsupported map extension mode: {mode}")
        if not mode.startswith("navigate-"):
            if not values:
                raise ValueError("Configured task has no freshly grounded visible answer")
            paired = list(zip(values, sources, strict=True))
            if sort_key is not None:
                paired.sort(key=sort_key)
            values = [value for value, _ in paired]
            evidence = [
                {
                    "field": f"/task_definition/answer/values/{index}/{field}",
                    "artifact": "observation",
                    "location": pointer,
                    "modality": "text",
                    "derivation": "Apply the stated exact visible-record, counting, ordering or displayed Directions rule.",
                }
                for index, (value, pointers) in enumerate(paired)
                for field in value
                for pointer in dict.fromkeys(
                    [
                        *pointers,
                        "/routes" if mode == "retrieve-route-profile-comparison" else "/searches",
                    ]
                )
            ]
            definition = {
                "task_type": "RETRIEVE",
                "goal_template": goal + " Do not edit map data.",
                "parameters": parameters,
                "answer": {
                    "status": "SUCCESS",
                    "values": values,
                    "item_schema": {
                        "type": "object",
                        "properties": {
                            field: {"type": "integer" if type(value) is int else "string"}
                            for field, value in values[0].items()
                        },
                        "required": list(values[0]),
                        "additionalProperties": False,
                    },
                    "ordered": True,
                },
                "completion_events": [],
            }
            checks = []
    for field in config["parameters"]:
        evidence.append(
            {
                "field": f"/task_definition/parameters/{field}",
                "artifact": "observation",
                "location": "/routes"
                if field in {"from_label", "to_label", "engine_labels"}
                else "/searches",
                "modality": "text",
                "derivation": "Actual submitted controls and complete visible source windows establish the configured rule.",
            }
        )
    return definition, evidence, checks


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    os.umask(0o077)
    site_url = args.site_url.rstrip("/")
    baseline = None
    exploration_identity = None
    if args.mode == "explore":
        inspection_bytes = args.container_inspection.read_bytes()
        ownership_bytes = args.ownership_receipt.read_bytes()
        inspection = json.loads(inspection_bytes)
        ownership = json.loads(ownership_bytes)
        if set(ownership) != {
            "site",
            "run_id",
            "session_nonce",
            "sandbox_id",
            "generated_task_id",
            "model_profile",
            "attempt",
            "container_name",
            "map_root",
        }:
            raise ValueError("Explore requires the canonical producer ownership receipt")
        expected = {
            "site": "map",
            "run_id": args.run_id,
            "session_nonce": args.nonce,
            "sandbox_id": args.sandbox_id,
        }
        if any(ownership[key] != value for key, value in expected.items()):
            raise ValueError("Explore ownership does not match the authorized run/nonce/sandbox")
        for key in (
            "site",
            "run_id",
            "session_nonce",
            "generated_task_id",
            "model_profile",
            "attempt",
        ):
            if inspection[key] != ownership[key]:
                raise ValueError("Actual container inspection differs from its producer ownership")
        task_id, profile, attempt = (
            ownership["generated_task_id"],
            ownership["model_profile"],
            ownership["attempt"],
        )
        if (
            not re.fullmatch(r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", task_id)
            or profile != "author"
            or not re.fullmatch(r"[0-9]{2}", attempt)
        ):
            raise ValueError("Explore must target an actual owned author case")
        container_name = f"wav-v03-map-{profile}-{task_id}-{attempt}"
        map_root = f"/var/lib/wav-v03/map-executions/{task_id}/{profile}-{attempt}"
        if (
            ownership["container_name"] != container_name
            or ownership["map_root"] != map_root
            or inspection["container_name"] != "/" + container_name
            or inspection["container_id"] != args.container_id
            or inspection["running"] is not True
        ):
            raise ValueError("Explore requires the exact running owned Map case and full CID")
        mounts = {mount["Destination"]: mount for mount in inspection["mounts"]}
        for destination, suffix in (
            ("/data/database", "tile-db"),
            ("/data/nominatim/postgres", "nominatim-db"),
            ("/var/lib/postgresql/14/main", "website-db"),
            ("/data/tiles", "tiles"),
            ("/data/style", "style"),
        ):
            if (
                mounts[destination]["Source"] != map_root + "/" + suffix
                or mounts[destination]["RW"] is not True
            ):
                raise ValueError("Explore database mount differs from the owned case root")
        for destination in (
            "/mnt/map-source/nominatim-flatnode",
            "/mnt/map-source/routing-car",
            "/mnt/map-source/routing-bike",
            "/mnt/map-source/routing-foot",
        ):
            if mounts[destination]["RW"] is not False:
                raise ValueError("Explore requires immutable read-only source mounts")
        endpoint = urlsplit(site_url)
        host_port = str(endpoint.port or (443 if endpoint.scheme == "https" else 80))
        if endpoint.hostname not in {"localhost", "127.0.0.1"} or not any(
            port["HostPort"] == host_port and port["HostIp"] == "127.0.0.1"
            for port in inspection["ports"]["8080/tcp"]
        ):
            raise ValueError("Explore endpoint differs from the actual owned Map UI port")
        exploration_identity = {
            "scope": "existing_owned_map_container_anonymous_read_only_UI",
            "run_id": args.run_id,
            "session_nonce": args.nonce,
            "sandbox_id": args.sandbox_id,
            "container_id": inspection["container_id"],
            "image": inspection["image"],
            "image_id": inspection["image_id"],
            "producer_ownership": ownership,
            "source_proofs": {
                "container_inspection": {
                    "path": str(args.container_inspection),
                    "sha256": hashlib.sha256(inspection_bytes).hexdigest(),
                    "artifact": "container_inspection.json",
                },
                "producer_ownership": {
                    "path": str(args.ownership_receipt),
                    "sha256": hashlib.sha256(ownership_bytes).hexdigest(),
                    "artifact": "producer_ownership.json",
                },
            },
            "account_policy": "public_anonymous",
            "author_baseline_created": False,
        }
    else:
        baseline = json.loads(args.baseline_receipt.read_text())
        if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
            raise ValueError(
                "Map authoring requires a coordinator-owned fresh fixed-image baseline"
            )
        if baseline["site_url"].rstrip("/") != site_url:
            raise ValueError("Author endpoint differs from the coordinator baseline receipt")
    feature_source_bytes = None
    feature_sources = []
    if args.feature_source:
        feature_source_bytes = args.feature_source.read_bytes()
        feature_source = json.loads(feature_source_bytes)
        identity = feature_source["exploration_identity"]
        if any(
            identity[key] != exploration_identity[key]
            for key in ("run_id", "session_nonce", "sandbox_id", "container_id")
        ):
            raise ValueError("Feature source packet belongs to a different owned container")
        per_query = {}
        for target_path in args.feature_path:
            matches = []
            for query_index, search in enumerate(feature_source["searches"]):
                if not search["viewport_comparison"]["same_feature_facts"]:
                    continue
                for row_index, row in enumerate(search["batches"][0]["snapshot"]["results"]):
                    if row["url"] == site_url + target_path:
                        matches.append(
                            {
                                "query": search["query"],
                                "search_row": row,
                                "source_pointer": (
                                    f"/searches/{query_index}/batches/0/snapshot/results/{row_index}"
                                ),
                            }
                        )
            if not matches:
                raise ValueError("Feature path is absent from the actual recorded search packet")
            source = matches[0]
            per_query[source["query"]] = per_query.get(source["query"], 0) + 1
            if per_query[source["query"]] > 3:
                raise ValueError("Feature exploration is bounded to three links per source query")
            feature_sources.append(source)
    config = None
    if args.mode == "ground":
        configs = json.loads(args.task_config.read_text())
        selected = [
            task for task in configs["tasks"] if task["generated_task_id"] == args.generated_task_id
        ]
        if len(selected) != 1:
            raise ValueError("Select exactly one configured generated map task")
        config = selected[0]
        start_path = configs["view"]["start_path"]
        parameters = config["parameters"]
        if config["mode"].startswith("navigate-") and "engine_labels" in parameters:
            if type(parameters["route_link_zoom"]) is not int or parameters["route_link_zoom"] != 17:
                raise ValueError("Directions navigation requires explicit route_link_zoom=17")
        queries = (
            parameters["queries"]
            if "queries" in parameters
            else ([parameters["query"]] if "query" in parameters else [])
        )
        output = args.batch_root / "author" / args.generated_task_id / args.attempt
    else:
        start_path = args.start_path
        queries = args.search_query
        output = args.batch_root / "exploration" / "map" / args.attempt
    path = urlsplit(start_path)
    if not start_path.startswith("/") or start_path.startswith("//") or path.netloc or path.scheme:
        raise ValueError("Starting view must be an explicitly observed same-site path")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    if baseline is not None:
        (output / "baseline.json").write_text(json.dumps(baseline, indent=2))
    else:
        (output / "container_inspection.json").write_bytes(inspection_bytes)
        (output / "producer_ownership.json").write_bytes(ownership_bytes)
        (output / "exploration_identity.json").write_text(
            json.dumps(exploration_identity, indent=2)
        )
    if feature_source_bytes is not None:
        (output / "feature_source_observation.json").write_bytes(feature_source_bytes)
    viewport = {"width": args.viewport_width, "height": args.viewport_height}
    comparison = {
        "width": args.comparison_viewport_width,
        "height": args.comparison_viewport_height,
    }
    observed = {
        "scope": (
            "existing_owned_map_container_anonymous_read_only_UI"
            if args.mode == "explore"
            else "new_formal_author_recording"
            if config
            else "read_only_map_readiness"
        ),
        "start_path": start_path,
        "viewport": viewport,
        "comparison_viewport": comparison,
        "model_calls": 0,
        "official_catalog_access": False,
        "searches": [],
    }
    if exploration_identity is not None:
        observed["exploration_identity"] = exploration_identity
    if feature_source_bytes is not None:
        observed["feature_source"] = {
            "path": str(args.feature_source),
            "sha256": hashlib.sha256(feature_source_bytes).hexdigest(),
            "artifact": "feature_source_observation.json",
        }
    blocked, responses = [], []
    origin = urlsplit(site_url)
    from playwright.sync_api import sync_playwright

    failure = None
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
        page.on(
            "response",
            lambda response: responses.append(
                {"url": response.url, "method": response.request.method, "status": response.status}
            ),
        )
        try:
            start = site_url + start_path
            page.goto(start, wait_until="networkidle", timeout=args.timeout_ms)
            observed["initial"] = _record(page, output, "initial")
            # Search from the same public starting view and cover each initial batch.
            for query_index, query in enumerate(queries):
                search = {"query": query, "batches": []}
                observed["searches"].append(search)
                for view_index, size in enumerate((viewport, comparison)):
                    page.set_viewport_size(size)
                    batch = _search_batch(
                        page,
                        output,
                        start,
                        query,
                        f"search_{query_index:02}_{view_index:02}",
                        args.timeout_ms,
                    )
                    batch["viewport"] = size
                    search["batches"].append(batch)
                fingerprints = [
                    sorted(
                        (row["category"], row["label"], row["url"])
                        for row in batch["snapshot"]["results"]
                    )
                    for batch in search["batches"]
                ]
                search["viewport_comparison"] = {
                    "same_feature_facts": fingerprints[0] == fingerprints[1],
                    "same_continuation_presence": len(
                        {
                            bool(batch["snapshot"]["more_results_links"])
                            for batch in search["batches"]
                        }
                    )
                    == 1,
                    "bbox_equality_required": False,
                    "result_ranking_used": False,
                }
                if args.mode != "explore" and not all(
                    search["viewport_comparison"][key]
                    for key in ("same_feature_facts", "same_continuation_presence")
                ):
                    raise ValueError("Different viewports expose different initial feature facts")
                if config and any(
                    not coverage["full_row_visible"]
                    for batch in search["batches"]
                    for coverage in batch["scroll_coverage"]
                ):
                    raise ValueError(
                        "Formal source has a result row without complete visible coverage"
                    )
                if not config and args.feature_limit:
                    page.set_viewport_size(viewport)
                    search["features"] = []
                    candidates = search["batches"][0]["snapshot"]["results"][: args.feature_limit]
                    for index, row in enumerate(candidates):
                        feature = _feature_snapshot(
                            page,
                            output,
                            row["url"],
                            f"feature_{query_index:02}_{index:02}",
                            strict=args.mode != "explore",
                        )
                        search["features"].append({"search_row": row, "observation": feature})
            if feature_sources:
                observed["features"] = []
                page.set_viewport_size(viewport)
                for index, source in enumerate(feature_sources):
                    snapshot = _feature_snapshot(
                        page,
                        output,
                        source["search_row"]["url"],
                        f"feature_{index:02}",
                        strict=False,
                    )
                    observed["features"].append({**source, "observation": snapshot})
            if config:
                if observed["searches"]:
                    observed["search"] = observed["searches"][0]["batches"][0]["snapshot"]
                if "engine_labels" in config["parameters"]:
                    if (
                        config["mode"].startswith("navigate-")
                        and len(parameters["engine_labels"]) != 1
                    ):
                        raise ValueError(
                            "Directions navigation requires one explicitly selected profile"
                        )
                    observed["route_endpoints"] = _select_route_endpoints(config, observed)
                    page.set_viewport_size(viewport)
                    observed["routes"] = _collect_routes(
                        page,
                        output,
                        observed["initial"],
                        site_url,
                        observed["route_endpoints"]["from_label"],
                        observed["route_endpoints"]["to_label"],
                        parameters["engine_labels"],
                        responses,
                        args.timeout_ms,
                        args.settle_seconds,
                        strict=True,
                        name_prefix="route",
                        route_link_zoom=(
                            parameters["route_link_zoom"]
                            if config["mode"].startswith("navigate-") else None
                        ),
                    )
                    page.set_viewport_size(comparison)
                    page.goto(start, wait_until="networkidle", timeout=args.timeout_ms)
                    observed["comparison_routes"] = _collect_routes(
                        page,
                        output,
                        observed["initial"],
                        site_url,
                        observed["route_endpoints"]["from_label"],
                        observed["route_endpoints"]["to_label"],
                        parameters["engine_labels"],
                        responses,
                        args.timeout_ms,
                        args.settle_seconds,
                        strict=True,
                        name_prefix="comparison_route",
                        route_link_zoom=(
                            parameters["route_link_zoom"]
                            if config["mode"].startswith("navigate-") else None
                        ),
                    )
                    comparisons = _compare_route_recordings(
                        observed["routes"], observed["comparison_routes"]
                    )
                    observed["route_viewport_comparison"] = comparisons
                    if not all(item["same_semantic_route"] for item in comparisons):
                        raise ValueError("Different viewports expose different Directions facts")
                    page.set_viewport_size(viewport)
                    if config["mode"].startswith("navigate-"):
                        # Directions changes the address using history.replaceState.
                        # Open that real route link so the final document navigation
                        # and the visible shared-route panel describe the same task.
                        generated_queries = [
                            parse_qs(urlsplit(observed[source][0]["observation"]["url"]).query,
                                     keep_blank_values=True)
                            for source in ("routes", "comparison_routes")
                        ]
                        observed["route_link_viewport_comparison"] = {
                            "same_generated_query": generated_queries[0] == generated_queries[1]
                        }
                        if not observed["route_link_viewport_comparison"]["same_generated_query"]:
                            raise ValueError("Different viewports generated different route link queries")
                        for name, size, source in (
                            ("terminal_route", viewport, "routes"),
                            ("comparison_terminal_route", comparison, "comparison_routes"),
                        ):
                            shared_url = observed[source][0]["observation"]["url"]
                            shared = urlsplit(shared_url)
                            query = parse_qs(shared.query, keep_blank_values=True)
                            if (
                                (shared.scheme, shared.netloc) != (origin.scheme, origin.netloc)
                                or shared.path != "/directions"
                                or set(query) != {"engine", "route"}
                                or any(len(values) != 1 or not values[0] for values in query.values())
                            ):
                                raise ValueError("Computed Directions lacks its real route link")
                            shared_points = [
                                tuple(float(value) for value in point.split(","))
                                for point in query["route"][0].split(";")
                            ]
                            if len(shared_points) != 2 or any(
                                len(point) != 2 for point in shared_points
                            ):
                                raise ValueError(
                                    "Generated route link must contain both endpoint coordinates"
                                )
                            page.set_viewport_size(size)
                            response_start = len(responses)
                            route_deadline = monotonic() + args.timeout_ms / 1000
                            # goto on the same document only changes its fragment.
                            # Reload an already-open route to record its real document GET.
                            if urlsplit(page.url)._replace(fragment="") == shared._replace(fragment=""):
                                page.goto(
                                    shared_url, wait_until="domcontentloaded", timeout=args.timeout_ms
                                )
                                response = page.reload(
                                    wait_until="domcontentloaded", timeout=args.timeout_ms
                                )
                            else:
                                response = page.goto(
                                    shared_url, wait_until="domcontentloaded", timeout=args.timeout_ms
                                )
                            terminal = observed[name] = {
                                "from_query": observed["route_endpoints"]["from_label"],
                                "to_query": observed["route_endpoints"]["to_label"],
                                "engine_label": parameters["engine_labels"][0],
                                "source_url": shared_url,
                                "source_pointer": f"/{source}/0/observation/url",
                                "navigation_request": {
                                    "url": response.url if response is not None else shared_url,
                                    "method": (
                                        response.request.method if response is not None else None
                                    ),
                                    "status": response.status if response is not None else None,
                                    "is_navigation_request": bool(
                                        response is not None
                                        and response.request.is_navigation_request()
                                    ),
                                    "resource_type": (
                                        response.request.resource_type
                                        if response is not None
                                        else None
                                    ),
                                },
                            }
                            navigation = terminal["navigation_request"]
                            requested = urlsplit(navigation["url"])
                            requested_query = parse_qs(requested.query)
                            navigation["query_params"] = requested_query
                            if (
                                navigation["method"] != "GET"
                                or navigation["status"] != 200
                                or not navigation["is_navigation_request"]
                                or navigation["resource_type"] != "document"
                                or (requested.scheme, requested.netloc, requested.path)
                                != (shared.scheme, shared.netloc, shared.path)
                                or requested_query != query
                            ):
                                raise ValueError(
                                    "Generated route link lacks its actual document GET200"
                                )
                            # Tiles may remain in flight after the route is complete.
                            # Wait for the rendered result or its real error, then
                            # bind the captured backend response and cover every row.
                            try:
                                page.wait_for_function(
                                    r"""() => {
                                      const visible = e => {
                                        const b = e.getBoundingClientRect();
                                        return !!(b.width && b.height)
                                          && getComputedStyle(e).visibility !== 'hidden';
                                      };
                                      if (Array.from(document.querySelectorAll(
                                        '.search_results_error')).some(visible)) return true;
                                      const headings = Array.from(document.querySelectorAll(
                                        'h1,h2,h3')).filter(visible).filter(
                                          e => e.innerText.trim() === 'Directions');
                                      const summaries = Array.from(document.querySelectorAll('p'))
                                        .filter(visible).filter(e =>
                                          /^Distance: [^\n]+\. Time: [^\n]+\.$/.test(
                                            e.innerText.trim()));
                                      const tables = Array.from(document.querySelectorAll('table'))
                                        .filter(visible).filter(e => {
                                          const rows = Array.from(e.querySelectorAll('tr'))
                                            .map(row => row.innerText.trim());
                                          return rows.length >= 2 && /1\. Start/.test(rows[0])
                                            && /\d+\. Reach destination/.test(rows.at(-1));
                                        });
                                      return headings.length === 1 && summaries.length === 1
                                        && tables.length === 1;
                                    }""",
                                    timeout=max(1, int((route_deadline - monotonic()) * 1000)),
                                )
                            finally:
                                snapshot = terminal["observation"] = _route_snapshot(
                                    page, output, name, responses[response_start:]
                                )
                            if (
                                not snapshot["route_readiness"][
                                    "valid_visible_directions_and_backend"
                                ]
                                or not snapshot["route_readiness"].get(
                                    "complete_instruction_row_coverage", False
                                )
                            ):
                                raise ValueError(
                                    "Opened route link lacks complete visible Directions "
                                    "and backend HTTP200"
                                )
                            controls = {
                                control["name"]: control for control in snapshot["controls"]
                            }
                            selected = [
                                option["label"]
                                for option in controls["routing_engines"]["options"]
                                if option["selected"]
                            ]
                            input_points = [
                                tuple(
                                    float(value.strip())
                                    for value in controls[field]["value"].split(",")
                                )
                                for field in ("route_from", "route_to")
                            ]
                            backend_path = urlsplit(snapshot["backend_request"]["url"]).path
                            backend_points = [
                                tuple(float(value) for value in point.split(","))[::-1]
                                for point in backend_path.rsplit("/", 1)[1].split(";")
                            ]
                            terminal_url = urlsplit(snapshot["url"])
                            terminal_query = parse_qs(terminal_url.query, keep_blank_values=True)
                            if (
                                selected != parameters["engine_labels"]
                                or input_points != shared_points
                                or backend_points != shared_points
                                or (terminal_url.scheme, terminal_url.netloc, terminal_url.path)
                                != (shared.scheme, shared.netloc, shared.path)
                                or set(terminal_query) != {"engine", "route"}
                                or any(
                                    len(values) != 1 or not values[0]
                                    for values in terminal_query.values()
                                )
                                or terminal_query["engine"] != query["engine"]
                                or [
                                    tuple(float(value) for value in point.split(","))
                                    for point in terminal_query["route"][0].split(";")
                                ] != shared_points
                            ):
                                raise ValueError(
                                    "Opened route link differs from its selected profile "
                                    "or endpoint coordinates"
                                )
                            snapshot["route_readiness"][
                                "matches_shared_route_coordinates_and_profile"
                            ] = True
                        observed["shared_route_viewport_comparison"] = _compare_route_recordings(
                            [observed["terminal_route"]], [observed["comparison_terminal_route"]]
                        )
                        if not observed["shared_route_viewport_comparison"][0][
                            "same_semantic_route"
                        ]:
                            raise ValueError("Different viewports expose different shared-route facts")
                        page.set_viewport_size(viewport)
                definition, evidence, checks = build_extension_definition(config, observed)
                if config["mode"] == "navigate-constrained-feature":
                    page.set_viewport_size(viewport)
                    target = definition["parameters"]["target_url"].replace("__MAP__", site_url)
                    terminal = _feature_snapshot(page, output, target, "terminal")
                    observed["terminal"] = terminal
                    selected_name = definition["parameters"]["selected_label"].split(",")[0]
                    if selected_name not in terminal["body_text"]:
                        raise ValueError("Fresh terminal feature lacks the selected displayed name")
                    # The same feature URL can serve a Not Found page; require real sidebar content.
                    for key, source in (
                        ("feature_heading", "identity_target"),
                        ("feature_detail", "detail_target"),
                    ):
                        definition["parameters"][key] = terminal[source]["text"]
                        checks.append(
                            {
                                "kind": "visible_text_exact",
                                "parameter": key,
                                "selector": terminal[source]["selector"],
                            }
                        )
                        evidence.append(
                            {
                                "field": f"/task_definition/parameters/{key}",
                                "artifact": "observation",
                                "location": f"/terminal/{source}/text",
                                "modality": "text",
                                "derivation": "Exact visible feature identity and sidebar details.",
                            }
                        )
                    for field in ("http_method", "response_status", "url_template"):
                        source = {
                            "http_method": "method",
                            "response_status": "status",
                            "url_template": "url",
                        }[field]
                        evidence.append(
                            {
                                "field": f"/task_definition/completion_events/0/{field}",
                                "artifact": "observation",
                                "location": f"/terminal/request/{source}",
                                "modality": "text",
                                "derivation": "Actual feature GET, preserving its observed path.",
                            }
                        )
            else:
                # Record the actual Directions form before accepting any route inputs.
                target = site_url + "/directions"
                if any(link["url"] == target for link in observed["initial"]["links"]):
                    page.set_viewport_size(viewport)
                    page.goto(target, wait_until="networkidle", timeout=args.timeout_ms)
                    directions = _record(page, output, "directions")
                    observed["directions"] = directions
                    if args.from_query:
                        observed["routes"] = _collect_routes(
                            page,
                            output,
                            observed["initial"],
                            site_url,
                            args.from_query,
                            args.to_query,
                            args.engine_label,
                            responses,
                            args.timeout_ms,
                            args.settle_seconds,
                            strict=False,
                            name_prefix="route",
                        )
                        if all(
                            route["observation"]["route_readiness"][
                                "valid_visible_directions_and_backend"
                            ]
                            for route in observed["routes"]
                        ):
                            page.set_viewport_size(comparison)
                            page.goto(start, wait_until="networkidle", timeout=args.timeout_ms)
                            observed["comparison_routes"] = _collect_routes(
                                page,
                                output,
                                observed["initial"],
                                site_url,
                                args.from_query,
                                args.to_query,
                                args.engine_label,
                                responses,
                                args.timeout_ms,
                                args.settle_seconds,
                                strict=False,
                                name_prefix="comparison_route",
                            )
                            if all(
                                route["observation"]["route_readiness"][
                                    "valid_visible_directions_and_backend"
                                ]
                                for route in observed["comparison_routes"]
                            ):
                                observed["route_viewport_comparison"] = _compare_route_recordings(
                                    observed["routes"], observed["comparison_routes"]
                                )
                elif args.from_query:
                    raise ValueError("Directions link is absent from the fresh homepage")
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error)}
            observed["failure"] = failure
            raise
        finally:
            observed["network_responses"] = responses
            observed["blocked_requests"] = blocked
            try:
                context.tracing.stop(path=str(output / "trace.zip"))
            finally:
                context.close()
                browser.close()
                (output / "observation.json").write_text(
                    json.dumps(observed, ensure_ascii=False, indent=2)
                )
                (output / "blocked_requests.json").write_text(json.dumps(blocked, indent=2))
                if not config:
                    receipt = {
                        "status": (
                            "failed"
                            if failure or blocked
                            else "actual_UI_recorded_backend_review_required"
                        ),
                        "scope": observed["scope"],
                        "generated_candidates": 0,
                        "GT_emitted": False,
                        "model_calls": 0,
                        "search_queries": queries,
                        "route_queries_submitted": bool(observed.get("routes")),
                        "failure": failure,
                        "blocked_requests": blocked,
                        "artifacts": {
                            "har": "network.har",
                            "trace": "trace.zip",
                            "observation": "observation.json",
                            "screenshot": "initial.png",
                        },
                    }
                    if args.mode == "explore":
                        receipt["exploration_identity"] = exploration_identity
                        receipt_name = "exploration_receipt.json"
                    else:
                        receipt["baseline"] = baseline
                        receipt_name = "readiness_receipt.json"
                    (output / receipt_name).write_text(
                        json.dumps(receipt, ensure_ascii=False, indent=2)
                    )
    if blocked:
        raise ValueError("Recorded requests violated the same-site read-only author policy")
    if not config:
        print(
            json.dumps(
                {"status": args.mode + "_recorded", "output": str(output), "GT_emitted": False}
            ),
            flush=True,
        )
        return
    relative = output.relative_to(args.batch_root)
    draft = {
        "generated_task_id": args.generated_task_id,
        "task_id": config["task_id"],
        "intent_template_id": config["intent_template_id"],
        "site": "map",
        "start_url": "__MAP__" + start_path,
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
                "screenshot": str(
                    relative
                    / (
                        "terminal_route.png"
                        if observed.get("terminal_route")
                        else "route_00.png"
                        if observed.get("routes")
                        else "terminal.png"
                        if checks
                        else "search_00_00.png"
                    )
                ),
                "observation": str(relative / "observation.json"),
            },
            "field_evidence": evidence,
            "final_state_checks": checks,
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
