#!/usr/bin/env python3
"""Collect grounded page evidence, HAR, trace, and screenshots for pilot tasks."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


def _load_plan(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("plan must be a JSON array of objects")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for item in _load_plan(args.plan):
            task_root = args.output_root / item["generated_task_id"]
            task_root.mkdir(parents=True, exist_ok=True)
            storage_state = None
            if header_name := item.get("bootstrap_auth_header_from_site"):
                from webarena.browser_env.env_config import ACCOUNTS

                credentials = ACCOUNTS[item["site"]]
                bootstrap_context = browser.new_context(
                    extra_http_headers={
                        header_name: f"{credentials['username']}:{credentials['password']}"
                    }
                )
                bootstrap_page = bootstrap_context.new_page()
                bootstrap_page.goto(
                    item["url"], wait_until="domcontentloaded", timeout=120_000
                )
                bootstrap_page.wait_for_timeout(2_000)
                storage_state = bootstrap_context.storage_state()
                bootstrap_context.close()
            context = browser.new_context(
                record_har_path=task_root / "network.har",
                record_har_content="embed",
                storage_state=storage_state,
            )
            context.tracing.start(screenshots=True, snapshots=True, sources=False)
            page = context.new_page()
            if item.get("login"):
                from browsergym.webarena.instance import WebArenaInstance

                WebArenaInstance().ui_login(site=item["site"], page=page)
            response = page.goto(item["url"], wait_until="domcontentloaded", timeout=120_000)
            page.wait_for_timeout(2_000)
            page.screenshot(path=task_root / "page.png", full_page=True)
            observation = {
                "generated_task_id": item["generated_task_id"],
                "site": item["site"],
                "requested_url": item["url"],
                "final_url": page.url,
                "response_status": response.status if response else None,
                "title": page.title(),
                "links": page.locator("a").evaluate_all(
                    "els => els.map(a => ({text: (a.innerText || '').trim(), href: a.href}))"
                ),
                "body_text": page.locator("body").inner_text(timeout=30_000),
                "observed_at": datetime.now(UTC).isoformat(),
            }
            (task_root / "observation.json").write_text(
                json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
            )
            context.tracing.stop(path=task_root / "trace.zip")
            context.close()
            results.append({key: observation[key] for key in observation if key != "body_text"})
        browser.close()
    (args.output_root / "observations.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"collected={len(results)} output_root={args.output_root}")


if __name__ == "__main__":
    main()
