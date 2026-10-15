#!/usr/bin/env python3
"""Record bounded, read-only UI evidence on one coordinator-owned fixed site.

The coordinator supplies the site URL and runs this inside its prepared runtime.
This entry starts only its own browser. It never provisions or resets a site,
calls a model, discovers official tasks, or constructs a candidate. Observations
are author evidence for the existing ``author.compile_candidate`` pipeline.

Example (inside the configured runtime):
  /opt/venv/bin/python -B explore_single_site.py --site gitlab \
    --url http://127.0.0.1:8023 --output-dir /outputs/exploration/gitlab \
    --private-state /outputs/private/gitlab_storage_state.json \
    --login-owner /opt/api-extra/python-extra/browsergym/webarena/instance.py

Dependencies: Playwright/Chromium and, for standard login, the deployed
BrowserGym WebArena instance owner plus ``webarena.browser_env.env_config``.
Use PYTHONPATH=/opt/api-extra/python-extra:/root/cua-lite in that runtime.
Headless recording uses a 1920x1080 browser viewport; it is author UI evidence,
not a model rollout or a claim about full-desktop trajectory quality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urldefrag, urljoin, urlsplit

SITES = ("shopping", "gitlab", "reddit", "wikipedia", "map")
LOGIN_SITES = frozenset({"shopping", "gitlab", "reddit"})
ACTION_LABEL = re.compile(
    r"^(?:delete|remove|destroy|log\s*out|sign\s*out|upvote|downvote|"
    r"subscribe|unsubscribe|follow|unfollow|add\s+to\s+(?:cart|wish\s*list|compare))(?:\b|$)",
    re.I,
)
ACTION_PATH = re.compile(
    r"/(?:delete|remove|destroy|logout|sign_out|signout|upvote|downvote|"
    r"subscribe|unsubscribe|follow|unfollow|vote|addtocart)(?:/|$)|"
    r"/(?:cart|product_compare)/add(?:/|$)|/wishlist/(?:index/)?add(?:/|$)",
    re.I,
)
LINK_PRIORITY = {
    "shopping": re.compile(
        r"women|men|gear|sale|new|shirt|bag|jacket|watch|tank|pant|tee|short", re.I
    ),
    "gitlab": re.compile(
        r"project|group|explore|issue|repository|activity|member|commit|merge|wiki|milestone|label",
        re.I,
    ),
    "reddit": re.compile(r"forum|comment|post|all|hot|new|top", re.I),
    "wikipedia": re.compile(
        r"science|history|geography|technology|mathematics|physics|biology", re.I
    ),
    "map": re.compile(r"direction|history|about|data|map", re.I),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=SITES, required=True)
    parser.add_argument(
        "--url", required=True, help="Explicit public homepage/landing URL on the one fixed site"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New private author-evidence directory"
    )
    parser.add_argument(
        "--private-state",
        type=Path,
        required=True,
        help="Private output storage-state path, never printed",
    )
    parser.add_argument(
        "--storage-state",
        type=Path,
        help="Use coordinator-authenticated state instead of logging in",
    )
    parser.add_argument(
        "--login-owner",
        type=Path,
        default=Path("/opt/api-extra/python-extra/browsergym/webarena/instance.py"),
    )
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-links-per-page", type=int, default=100)
    parser.add_argument(
        "--follow-label",
        action="append",
        default=[],
        help="Optional regex selecting visible link labels",
    )
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    origin = urlsplit(args.url)
    if (
        origin.scheme not in {"http", "https"}
        or not origin.netloc
        or origin.username
        or origin.password
        or origin.query
        or origin.fragment
    ):
        parser.error("--url must be an HTTP(S) landing URL without credentials/query/fragment")
    if not 1 <= args.max_pages <= 30 or not 0 <= args.max_depth <= 3:
        parser.error("max pages must be 1..30 and max depth must be 0..3")
    if not 1 <= args.max_links_per_page <= 300 or args.timeout_ms <= 0 or args.settle_seconds < 0:
        parser.error("links and timeout must be positive; settle seconds must be nonnegative")
    for pattern in args.follow_label:
        re.compile(pattern)
    return args


def run_exploration(args: argparse.Namespace) -> dict:
    """Record rendered facts and follow only sampled visible same-site links."""
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    os.umask(0o077)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    args.private_state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    origin = urlsplit(args.url)
    phase = "authentication"
    blocked = []
    observations = []
    frontier = deque([{"url": args.url, "depth": 0, "label": "homepage", "discovered_from": None}])
    queued = {urldefrag(args.url)[0]}
    patterns = [re.compile(pattern, re.I) for pattern in args.follow_label]
    account_summary = {"policy": "public_anonymous", "username": None}
    report = {
        "site": args.site,
        "landing_url": args.url,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "single_site_fixed_initial_assets_read_only_author_ui",
        "candidate_owner": "devs/data/webarena_verified_generated/author.py:compile_candidate",
        "official_catalog_access": False,
        "model_calls": 0,
        "business_write_requests_allowed": False,
        "viewport": [1920, 1080],
        "headless": args.headless,
        "limits": {
            "pages": args.max_pages,
            "depth": args.max_depth,
            "links_per_page": args.max_links_per_page,
        },
    }

    def restrict(route):
        requested = urlsplit(route.request.url)
        same_origin = (requested.scheme, requested.netloc) == (origin.scheme, origin.netloc)
        if not same_origin or (
            phase == "observation" and route.request.method not in {"GET", "HEAD"}
        ):
            # Exclude query strings, bodies and headers from the summary. Full
            # authenticated HAR/trace remain private author artifacts.
            blocked.append(
                {
                    "phase": phase,
                    "method": route.request.method,
                    "path": requested.path,
                    "reason": "foreign_origin" if not same_origin else "non_read_method",
                }
            )
            route.abort()
        else:
            route.continue_()

    failure = None
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=args.headless)
        auth = context = None
        trace_started = False
        try:
            # Use the existing standard account/login owner without importing
            # BrowserGym's package initializer or discovering benchmark tasks.
            if args.storage_state is not None:
                state = json.loads(args.storage_state.read_text())
                account_summary = {
                    "policy": "coordinator_preauthenticated_storage_state",
                    "username": None,
                }
            elif args.site in LOGIN_SITES:
                spec = importlib.util.spec_from_file_location(
                    "wavgen_author_login_owner", args.login_owner
                )
                if spec is None or spec.loader is None:
                    raise ImportError("Cannot load the explicitly deployed login owner")
                owner = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(owner)
                for name in (
                    "SHOPPING",
                    "SHOPPING_ADMIN",
                    "REDDIT",
                    "GITLAB",
                    "WIKIPEDIA",
                    "MAP",
                    "HOMEPAGE",
                ):
                    os.environ.setdefault(f"WA_{name}", "http://127.0.0.1:9")
                os.environ[f"WA_{args.site.upper()}"] = args.url.rstrip("/")
                instance = owner.WebArenaInstance()
                account_summary = {
                    "policy": "standard_WebArenaInstance_ui_login",
                    "username": instance.credentials[args.site]["username"],
                    "login_owner_sha256": hashlib.sha256(args.login_owner.read_bytes()).hexdigest(),
                    "credential_values_exported": False,
                }
                auth = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    service_workers="block",
                )
                auth.set_default_timeout(args.timeout_ms)
                auth.route("**/*", restrict)
                login_page = auth.new_page()
                instance.ui_login(site=args.site, page=login_page)
                state = auth.storage_state()
                auth.close()
                auth = None
            else:
                state = {"cookies": [], "origins": []}
            # Login requests/password fields never enter the recording scope.
            args.private_state.write_text(json.dumps(state))
            args.private_state.chmod(0o600)
            phase = "observation"
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                storage_state=state,
                accept_downloads=False,
                service_workers="block",
                record_har_path=str(output / "network.har"),
                record_har_content="embed",
            )
            context.route("**/*", restrict)
            context.tracing.start(screenshots=True, snapshots=True, sources=False)
            trace_started = True
            page = context.new_page()
            page.set_default_timeout(args.timeout_ms)
            # Each navigation target originates in a visible sampled anchor;
            # no entity IDs, API endpoints or alternate URLs are invented.
            while frontier and len(observations) < args.max_pages:
                target = frontier.popleft()
                number = len(observations)
                name = f"page_{number:03d}"
                row = {**target, "observation": f"{name}.json", "screenshot": f"{name}.png"}
                try:
                    response = page.goto(
                        target["url"], wait_until="domcontentloaded", timeout=args.timeout_ms
                    )
                    page.wait_for_timeout(int(args.settle_seconds * 1000))
                    row.update(
                        url=page.url,
                        title=page.title(),
                        http_status=response.status if response else None,
                    )
                    row["body_text"] = page.locator("body").inner_text()
                    row["tables"] = [
                        table.inner_text() for table in page.locator("table:visible").all()
                    ]
                    controls = []
                    for control in page.locator(
                        "input:visible, textarea:visible, select:visible, button:visible"
                    ).all():
                        tag = control.evaluate("element => element.tagName.toLowerCase()")
                        kind = control.get_attribute("type") or tag
                        record = {
                            "tag": tag,
                            "type": kind,
                            "name": control.get_attribute("name"),
                            "placeholder": control.get_attribute("placeholder"),
                            "aria_label": control.get_attribute("aria-label"),
                            "text": control.inner_text() if tag == "button" else None,
                        }
                        if kind != "password" and tag in {"input", "textarea", "select"}:
                            record["value"] = control.input_value()
                        if tag == "select":
                            record["selected_text"] = control.locator("option:checked").inner_text()
                        controls.append(record)
                    row["controls"] = controls
                    links = []
                    for anchor in page.locator("a:visible").all():
                        raw = anchor.get_attribute("href") or ""
                        label = (
                            anchor.inner_text().strip()
                            or anchor.get_attribute("aria-label")
                            or anchor.get_attribute("title")
                            or ""
                        )
                        if not raw or not label or raw.startswith("#"):
                            continue
                        link_url = urldefrag(urljoin(page.url, raw))[0]
                        parsed = urlsplit(link_url)
                        if (parsed.scheme, parsed.netloc) != (origin.scheme, origin.netloc):
                            continue
                        unsafe_query = any(
                            value.lower()
                            in {"delete", "remove", "logout", "vote", "subscribe", "unsubscribe"}
                            for key, values in parse_qs(parsed.query).items()
                            if key.lower() in {"action", "do", "op"}
                            for value in values
                        )
                        allowed = not (
                            ACTION_LABEL.search(label)
                            or ACTION_PATH.search(parsed.path)
                            or unsafe_query
                            or anchor.get_attribute("download") is not None
                        )
                        links.append(
                            {"label": label, "url": link_url, "navigation_allowed": allowed}
                        )
                        if len(links) >= args.max_links_per_page:
                            break
                    row["visible_links"] = links
                    page.screenshot(path=str(output / f"{name}.png"))
                    row["status"] = "observed"
                    if target["depth"] < args.max_depth:
                        for link in sorted(
                            links,
                            key=lambda link: (
                                not bool(LINK_PRIORITY[args.site].search(link["label"]))
                            ),
                        ):
                            if not link["navigation_allowed"] or link["url"] in queued:
                                continue
                            if patterns and not any(
                                pattern.search(link["label"]) for pattern in patterns
                            ):
                                continue
                            if len(queued) >= args.max_pages:
                                break
                            queued.add(link["url"])
                            frontier.append(
                                {
                                    "url": link["url"],
                                    "depth": target["depth"] + 1,
                                    "label": link["label"],
                                    "discovered_from": name,
                                }
                            )
                except PlaywrightError as error:
                    # Exception call logs can contain entered auth values.
                    # Retain the type/phase, never print or persist that text.
                    row.update(status="failed", error_type=type(error).__name__)
                (output / f"{name}.json").write_text(
                    json.dumps(row, ensure_ascii=False, indent=2) + "\n"
                )
                observations.append(row)
            args.private_state.write_text(json.dumps(context.storage_state()))
            args.private_state.chmod(0o600)
        except Exception as error:
            failure = {"phase": phase, "error_type": type(error).__name__}
        finally:
            try:
                if context is not None and trace_started:
                    context.tracing.stop(path=str(output / "trace.zip"))
            finally:
                try:
                    if context is not None:
                        context.close()
                    if auth is not None:
                        auth.close()
                finally:
                    browser.close()
    report.update(
        status="failed"
        if failure or not observations or observations[0]["status"] != "observed"
        else "recorded",
        account=account_summary,
        pages_attempted=len(observations),
        pages_observed=sum(row["status"] == "observed" for row in observations),
        observed_pages=[
            {
                "observation": row["observation"],
                "screenshot": row["screenshot"],
                "status": row["status"],
            }
            for row in observations
        ],
        blocked_requests=blocked,
        artifacts={"har": "network.har", "trace": "trace.zip", "observations": "observations.json"},
        failure=failure,
        completed_at=datetime.now(UTC).isoformat(),
    )
    (output / "observations.json").write_text(
        json.dumps(observations, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "exploration_receipt.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        result = run_exploration(args)
    except Exception as error:
        # Provider call logs and UI login exceptions can contain secrets. The
        # coordinator receives a type only, including on recording cleanup.
        print(
            json.dumps({"site": args.site, "status": "failed", "error_type": type(error).__name__})
        )
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "site": result["site"],
                "status": result["status"],
                "pages_observed": result["pages_observed"],
                "report": str(args.output_dir / "exploration_receipt.json"),
            }
        )
    )
    if result["status"] != "recorded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
