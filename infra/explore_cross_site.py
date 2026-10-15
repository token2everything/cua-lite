#!/usr/bin/env python3
"""Record bounded visible Shopping and GitLab facts on a fresh joint baseline.

The coordinator supplies authenticated storage and a JSON site-to-URL map.
Only visible same-origin links are followed. No models or site resources are started.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit
from uuid import uuid4

from explore_single_site import ACTION_LABEL, ACTION_PATH


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-urls", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pages-per-site", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--settle-ms", type=int, default=2000)
    args = parser.parse_args()
    if not 1 <= args.max_pages_per_site <= 40 or not 1 <= args.max_depth <= 6:
        parser.error("pages must be 1..40 and depth 1..6")
    return args


def record_observation(
    page, output: Path, name: str, site: str, *, context_id: str, producer: Path
) -> dict:
    """Capture visible entity identities and fact locations for both author consumers."""
    row = page.evaluate("""() => {
      const visible = e => !!e && !!(e.getBoundingClientRect().width && e.getBoundingClientRect().height);
      const text = e => e ? e.innerText.trim() : null;
      const all = (selector, root=document) => Array.from(root.querySelectorAll(selector)).filter(visible);
      const link = e => ({title: text(e), url: e.href});
      const readouts = (root, fields) => Object.fromEntries(Object.entries(fields)
        .filter(([name, spec]) => {
          const elements = root.querySelectorAll(spec.selector);
          return elements.length === 1 && visible(elements[0]);
        }));
      const detail = document.querySelector('.product-info-main');
      return {
        url: location.href, title: document.title, body_text: document.body.innerText,
        links: all('a[href]').map(a => ({...link(a), download:a.hasAttribute('download')})),
        products: all('main .product-item-info').map(e => ({
          ...link(e.querySelector('.product-item-link')),
          price: text(e.querySelector('[data-price-type=finalPrice] .price')),
          rating: e.querySelector('.rating-result')?.getAttribute('title') || null,
          reviews: text(e.querySelector('.reviews-actions .action.view')),
          readouts:readouts(e,{
            title:{kind:'text',selector:'.product-item-link',value_type:'string'},
            price:{kind:'text',selector:'[data-price-type=finalPrice] .price',value_type:'string'},
            reviews:{kind:'text',selector:'.reviews-actions .action.view',value_type:'string'}
          })
        })),
        cart_items: all('#shopping-cart-table tbody.cart.item tr.item-info').map(e => ({
          title:text(e.querySelector('.product-item-name')),url:e.querySelector('.product-item-name a')?.href,
          price:text(e.querySelector('.col.price .price')),
          quantity:e.querySelector('input[data-role=cart-item-qty]')?.value,
          subtotal:text(e.querySelector('.col.subtotal .price')),visible_text:text(e),
          readouts:readouts(e,{
            title:{kind:'text',selector:'.product-item-name',value_type:'string'},
            price:{kind:'text',selector:'.col.price .price',value_type:'string'},
            quantity:{kind:'input_value',selector:'input[data-role=cart-item-qty]',value_type:'integer'},
            subtotal:{kind:'text',selector:'.col.subtotal .price',value_type:'string'}
          })
        })),
        orders: all('#my-orders-table tbody tr').map(e => ({
          title:text(e.querySelector('td.col.id')),url:e.querySelector('a.action.view')?.href,
          total:text(e.querySelector('td.col.total .price')),
          status:text(e.querySelector('td.col.status')),visible_text:text(e),
          readouts:readouts(e,{
            title:{kind:'text',selector:'td.col.id',value_type:'string'},
            total:{kind:'text',selector:'td.col.total .price',value_type:'string'},
            status:{kind:'text',selector:'td.col.status',value_type:'string'}
          })
        })),
        product_detail: visible(detail) ? {
          title:text(document.querySelector('h1.page-title')),url:location.href,
          sku:text(detail.querySelector('.sku .value')),
          price:text(detail.querySelector('[data-price-type=finalPrice] .price')),visible_text:text(detail)
        } : null,
        tables:all('table').map(e => ({id:e.id,headers:all('thead th',e).map(text),
          rows:all('tbody tr',e).map(tr => ({text:text(tr),cells:all('th,td',tr).map(text),
            links:all('a[href]',tr).map(link)}))})),
        notes:all('li.note-comment:not(.system-note) .note-text.md').map(e => ({text:text(e),
          id:e.closest('.note')?.id || null})),
        issue_state: {
          title:text(document.querySelector('h1.title')),
          description:text(document.querySelector('div.description')),
          assignee:text(document.querySelector('[data-testid="assignee-block-container"]')),
          labels:all('[data-testid="sidebar-labels"] .gl-label-text').map(text),
          open:all('button').some(e=>text(e)==='Close issue'),
          checklist:all('div.description input[type=checkbox]').map(e=>({checked:e.checked,
            label:text(e.closest('li'))}))
        },
        forms:all('form').map(e=>({action:e.action,method:e.method,
          fields:Array.from(e.elements).map(c=>({name:c.name,id:c.id,type:c.type}))})),
        headings:all('h1,h2,h3').map(text),
        controls:all('input,textarea,select,button').filter(e=>e.type!=='password').map(e=>({
          tag:e.tagName.toLowerCase(),type:e.type,name:e.name,id:e.id,
          placeholder:e.getAttribute('placeholder'),text:text(e),value:e.value,
          aria_label:e.getAttribute('aria-label')}))
      };
    }""")
    row["site"] = site
    row["evidence"] = {"observation": name + ".json", "screenshot": name + ".png"}
    if site == "gitlab":
        row["issues"] = []
        for anchor in row["links"]:
            match = re.fullmatch(r"/(.+)/-/issues/(\d+)", urlsplit(anchor["url"]).path)
            if match:
                row["issues"].append({**anchor, "project_path": match[1], "issue_iid": match[2]})
        current = re.fullmatch(r"/(.+)/-/issues/(\d+)", urlsplit(row["url"]).path)
        row["issue_identity"] = (
            {"project_path": current[1], "issue_iid": current[2], "url": row["url"]}
            if current
            else None
        )
    page.screenshot(path=str(output / (name + ".png")))
    from cross_site_evaluator import record_gui_frame

    record_gui_frame(output, page, name + ".png", context_id, "author_context_only", producer)
    (output / (name + ".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    return row


def run_exploration(args) -> dict:
    """Follow a bounded graph of visible links while keeping both sites in one recording."""
    from cross_site_evaluator import seal_recording
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    sites = json.loads(args.site_urls.read_text())
    baseline = json.loads(args.baseline_receipt.read_text())
    if set(sites) != {"shopping", "gitlab"} or set(baseline["sites"]) != set(sites):
        raise ValueError("This author explorer requires the declared Shopping and GitLab pair")
    if not baseline["fresh_from_image"] or not baseline["authentication_only_preparation"]:
        raise ValueError("Exploration requires a fresh authentication-only joint baseline")
    origins = {(urlsplit(url).scheme, urlsplit(url).netloc) for url in sites.values()}
    for site, url in sites.items():
        if baseline["sites"][site]["site_url"].rstrip("/") != url.rstrip("/"):
            raise ValueError("Site URL differs from its joint baseline")
    os.umask(0o077)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    blocked, observations = [], []
    context_id = str(uuid4())
    started_at = datetime.now(UTC).isoformat()

    def restrict(route):
        request = route.request
        url = urlsplit(request.url)
        shipping_estimate = (
            request.method == "POST"
            and (url.scheme, url.netloc)
            == (urlsplit(sites["shopping"]).scheme, urlsplit(sites["shopping"]).netloc)
            and url.path == "/rest/default/V1/carts/mine/estimate-shipping-methods"
        )
        if (
            (url.scheme, url.netloc) not in origins
            or (request.method not in {"GET", "HEAD"} and not shipping_estimate)
            or ACTION_PATH.search(url.path)
        ):
            blocked.append({"method": request.method, "path": url.path})
            route.abort()
        else:
            route.continue_()

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(
            storage_state=str(args.storage_state),
            viewport={"width": 1920, "height": 1080},
            service_workers="block",
            accept_downloads=False,
            record_har_path=str(output / "network.har"),
            record_har_content="embed",
        )
        context.route("**/*", restrict)
        context.tracing.start(screenshots=True, snapshots=True, sources=False)
        try:
            page = context.new_page()
            for site, root in sites.items():
                frontier = [
                    {"url": root.rstrip("/") + "/", "depth": 0, "label": "homepage", "from": None}
                ]
                queued, visited = {frontier[0]["url"]}, 0
                while frontier and visited < args.max_pages_per_site:
                    target = frontier.pop(0)
                    name = f"{site}_{visited:03d}"
                    visited += 1
                    try:
                        response = page.goto(
                            target["url"], wait_until="domcontentloaded", timeout=args.timeout_ms
                        )
                        page.wait_for_timeout(args.settle_ms)
                        row = record_observation(
                            page, output, name, site, context_id=context_id, producer=Path(__file__)
                        )
                        row.update(
                            discovered_from=target["from"],
                            depth=target["depth"],
                            http_status=response.status if response else None,
                            status="observed",
                        )
                        if target["depth"] < args.max_depth:
                            links = []
                            for anchor in row["links"]:
                                url = urldefrag(urljoin(page.url, anchor["url"]))[0]
                                parsed, root_parts = urlsplit(url), urlsplit(root)
                                if (
                                    (parsed.scheme, parsed.netloc)
                                    != (root_parts.scheme, root_parts.netloc)
                                    or url in queued
                                    or anchor["download"]
                                    or not anchor["title"]
                                    or ACTION_PATH.search(parsed.path)
                                    or ACTION_LABEL.search(anchor["title"])
                                ):
                                    continue
                                if site == "shopping":
                                    priority = (
                                        0
                                        if re.search(
                                            r"cart|orders?|showcase|search|next",
                                            anchor["title"],
                                            re.IGNORECASE,
                                        )
                                        else 2
                                    )
                                    if row["products"] and any(
                                        item["url"] == url for item in row["products"]
                                    ):
                                        priority = 1
                                else:
                                    priority = (
                                        0 if re.search(r"/-/issues(?:/\d+)?$", parsed.path) else 2
                                    )
                                    if re.search(r"projects?|explore", anchor["title"], re.IGNORECASE):
                                        priority = 1
                                links.append(
                                    {
                                        "url": url,
                                        "depth": target["depth"] + 1,
                                        "label": anchor["title"],
                                        "from": name,
                                        "priority": priority,
                                    }
                                )
                                queued.add(url)
                            frontier.extend(links)
                            frontier.sort(key=lambda item: (item["priority"], item["depth"]))
                        observations.append(row)
                    except PlaywrightError as error:
                        observations.append(
                            {
                                "site": site,
                                "url": target["url"],
                                "status": "failed",
                                "error_type": type(error).__name__,
                            }
                        )
        finally:
            try:
                context.tracing.stop(path=str(output / "trace.zip"))
            finally:
                try:
                    context.close()
                finally:
                    browser.close()
    seal_recording(
        output,
        context_id,
        "author_context_only",
        started_at,
        datetime.now(UTC).isoformat(),
        output / "network.har",
        output / "trace.zip",
        Path(__file__),
    )
    report = {
        "sites": list(sites),
        "scope": "joint_fixed_baseline_read_only_author_evidence",
        "model_calls": 0,
        "observations": observations,
        "blocked_requests": blocked,
        "limits": {"pages_per_site": args.max_pages_per_site, "depth": args.max_depth},
    }
    (output / "observations.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"output": str(output), "pages": len(observations), "model_calls": 0}))
    return report


if __name__ == "__main__":
    run_exploration(_parse_args())
