#!/usr/bin/env python3
"""Record actual map search/directions readiness before authoring route tasks.

Inside the coordinator's Playwright/Chromium runtime:
  /opt/venv/bin/python -B /outputs/infra/author_map.py \
    --task-config /outputs/plans/map_task_configs_readiness.json \
    --baseline-receipt /outputs/private/map_author_baseline.json \
    --site-url http://localhost:8080 --batch-root /outputs

Optional --search-query or paired --from-query/--to-query use only text explicitly
supplied by the coordinator. There are no default addresses or coordinates.
This entry creates no draft or GT. Its private UI/HAR/trace packet determines
whether the real map supports later grounded search and route comparison tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--storage-state", type=Path)
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--attempt", default="readiness-01")
    parser.add_argument("--search-query")
    parser.add_argument("--from-query")
    parser.add_argument("--to-query")
    parser.add_argument("--engine-label", help="Actual visible routing option label")
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
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
    if bool(args.from_query) != bool(args.to_query):
        parser.error("route readiness needs both coordinator-supplied from/to queries")
    if not re.fullmatch(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", args.attempt):
        parser.error("attempt must be a simple identifier")
    if args.timeout_ms <= 0 or args.settle_seconds < 0:
        parser.error("timeout must be positive and settle seconds nonnegative")
    return args


def _record(page: Any, output: Path, name: str) -> dict[str, Any]:
    """Archive actual visible controls, option labels, and their form policies."""
    observation = page.evaluate(
        """() => {
          const visible = e => {
            const r = e.getBoundingClientRect();
            return !!(r.width && r.height) && getComputedStyle(e).visibility !== 'hidden';
          };
          const text = e => e.innerText.trim();
          return {
            url: location.href, title: document.title, body_text: document.body.innerText,
            links: Array.from(document.querySelectorAll('a[href]')).filter(visible)
              .map(a => ({text: text(a),
                label: text(a) || a.getAttribute('aria-label') || a.title, url: a.href})),
            controls: Array.from(document.querySelectorAll('input,select,textarea,button'))
              .filter(visible).map(e => ({
                tag: e.tagName.toLowerCase(), type: e.type, name: e.name, id: e.id,
                text: text(e), placeholder: e.getAttribute('placeholder'),
                aria_label: e.getAttribute('aria-label'),
                value: e.type === 'password' ? undefined : e.value,
                form: e.form ? {action: e.form.action, method: e.form.method} : null,
                options: e.tagName === 'SELECT' ? Array.from(e.options)
                  .map(o => ({label: o.textContent.trim(), value: o.value,
                    selected: o.selected})) : undefined
              }))
          };
        }"""
    )
    page.screenshot(path=str(output / f"{name}.png"))
    (output / f"{name}.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
    )
    return observation


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = json.loads(args.task_config.read_text())
    baseline = json.loads(args.baseline_receipt.read_text())
    site_url = args.site_url.rstrip("/")
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("readiness requires a coordinator-owned fixed image baseline")
    if baseline["site_url"].rstrip("/") != site_url:
        raise ValueError("readiness endpoint differs from the coordinator baseline receipt")
    os.umask(0o077)
    output = args.batch_root / "exploration" / "map" / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    site = urlsplit(site_url)
    blocked = []

    def restrict(route):
        requested = urlsplit(route.request.url)
        read_request = route.request.method in {"GET", "HEAD"} or (
            route.request.method == "POST"
            and requested.path == "/geocoder/search_osm_nominatim"
        )
        if (requested.scheme, requested.netloc) != (
            site.scheme,
            site.netloc,
        ) or not read_request:
            blocked.append(
                {"method": route.request.method, "host": requested.netloc, "path": requested.path}
            )
            route.abort()
        else:
            route.continue_()

    from playwright.sync_api import sync_playwright

    observations = {}
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = None
        trace_started = False
        try:
            context = browser.new_context(
                storage_state=str(args.storage_state) if args.storage_state else None,
                viewport={"width": 1920, "height": 1080},
                service_workers="block",
                accept_downloads=False,
                record_har_path=str(output / "network.har"),
                record_har_content="embed",
            )
            context.route("**/*", restrict)
            context.set_default_timeout(args.timeout_ms)
            context.tracing.start(screenshots=True, snapshots=True, sources=False)
            trace_started = True
            page = context.new_page()
            page.goto(site_url + "/", wait_until="domcontentloaded", timeout=args.timeout_ms)
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            initial = _record(page, output, "initial")
            observations["initial"] = initial
            search = [
                control
                for control in initial["controls"]
                if control["name"] == config["search_name"]
            ]
            if len(search) != 1 or search[0]["form"]["method"].upper() != "GET":
                raise ValueError("the actual visible read-only search form is unavailable")
            if args.search_query:
                control = page.locator(f"input[name='{config['search_name']}']:visible")
                control.fill(args.search_query)
                control.press("Enter")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                observations["search_submitted"] = _record(page, output, "search_submitted")
            target = config["directions_link"].replace("__MAP__", site_url)
            if not any(
                link["url"] == target and link["label"] == config["directions_label"]
                for link in initial["links"]
            ):
                raise ValueError("directions URL is absent from the actual visible homepage link")
            page.goto(target, wait_until="domcontentloaded", timeout=args.timeout_ms)
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            directions = _record(page, output, "directions")
            observations["directions"] = directions
            named = {
                control["name"]: control
                for control in directions["controls"]
                if control["name"]
                in {config["from_name"], config["to_name"], config["engines_name"]}
            }
            if set(named) != {config["from_name"], config["to_name"], config["engines_name"]}:
                raise ValueError(
                    "the actual visible directions inputs and engine selector are unavailable"
                )
            if any(control["form"]["method"].upper() != "GET" for control in named.values()):
                raise ValueError("directions readiness permits only the actual read-only GET form")
            options = named[config["engines_name"]]["options"]
            if args.engine_label:
                if args.engine_label not in {option["label"] for option in options}:
                    raise ValueError("requested engine is absent from actual visible options")
                page.locator(f"select[name='{config['engines_name']}']:visible").select_option(
                    label=args.engine_label
                )
            if args.from_query:
                page.locator(f"input[name='{config['from_name']}']:visible").fill(args.from_query)
                target_control = page.locator(f"input[name='{config['to_name']}']:visible")
                target_control.fill(args.to_query)
                target_control.press("Enter")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                observations["route_submitted"] = _record(page, output, "route_submitted")
        finally:
            try:
                if context is not None and trace_started:
                    context.tracing.stop(path=str(output / "trace.zip"))
            finally:
                try:
                    if context is not None:
                        context.close()
                finally:
                    browser.close()
    (output / "observation.json").write_text(
        json.dumps(observations, ensure_ascii=False, indent=2) + "\n"
    )
    receipt = {
        "status": "actual_ui_recorded_backend_grounding_pending",
        "site": "map",
        "captured_at": datetime.now(UTC).isoformat(),
        "baseline": baseline,
        "search_form_visible": True,
        "directions_form_visible": True,
        "routing_option_labels": [option["label"] for option in options],
        "search_query_submitted": bool(args.search_query),
        "route_queries_submitted": bool(args.from_query),
        "query_source": "explicit_coordinator_input_only; no default location or coordinates",
        "backend_status": "Review actual UI and HAR before any geographic task or GT is authored.",
        "network_policy": (
            "same_origin_GET_HEAD_and_POST_geocoder_search_osm_nominatim_only; "
            "the actual Rails handler fetches search XML and renders results "
            "without business writes"
        ),
        "blocked_requests": blocked,
        "generated_candidates": 0,
        "artifacts": {
            "har": "network.har",
            "trace": "trace.zip",
            "screenshot": "directions.png",
            "observation": "observation.json",
        },
    }
    (output / "readiness_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(output / "readiness_receipt.json"),
                "generated_candidates": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1) from None
