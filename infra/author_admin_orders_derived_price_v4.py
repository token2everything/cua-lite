#!/usr/bin/env python3
"""Ground fixed-baseline Magento tasks using visible author observations.

The coordinator owns baseline creation and recovery. This script opens a new
browser context, records visible grids, filters, and editor controls, and emits
grounding artifacts. Mutation modes perform the requested task after the
coordinator restores the baseline. It never prepares baseline assets or invokes
a model.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "inspect-orders",
            "retrieve-orders",
            "retrieve-orders-aggregate",
            "inspect-products",
            "inspect-customers",
            "inspect-detail",
            "retrieve-product-attributes",
            "navigate-product-attributes",
            "retrieve-simple-product-prices",
            "prepare-product-mutations",
            "mutate-product-name-price",
            "mutate-product-attributes",
            "mutate-selected-product-attributes",
            "mutate-selected-product-price-options",
            "mutate-orders-topk-derived-product-price",
        ],
        required=True,
    )
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-url", default="http://localhost:7780/admin")
    parser.add_argument("--generated-task-id", required=True)
    parser.add_argument("--attempt", default="attempt-01")
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--detail-url", help="Actual same-site link observed in the baseline UI")
    parser.add_argument("--entity-limit", type=int, default=3)
    parser.add_argument(
        "--expand-section",
        action="append",
        default=[],
        help="Actual observed editor section label to open before recording its controls",
    )
    args = parser.parse_args()
    if (
        args.mode
        in {
            "retrieve-orders",
            "retrieve-orders-aggregate",
            "retrieve-product-attributes",
            "navigate-product-attributes",
            "retrieve-simple-product-prices",
            "prepare-product-mutations",
            "mutate-product-name-price",
            "mutate-product-attributes",
            "mutate-selected-product-attributes",
            "mutate-selected-product-price-options",
            "mutate-orders-topk-derived-product-price",
        }
        and args.task_config is None
    ):
        parser.error("task grounding requires a config derived from the new baseline")
    if not args.attempt.replace("-", "").isalnum():
        parser.error("attempt must be a simple identifier")
    if not re.fullmatch(r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", args.generated_task_id):
        parser.error("generated task ID must use the generated batch namespace")
    if args.entity_limit not in range(1, 6):
        parser.error("entity limit must be between 1 and 5")
    if len(args.expand_section) > 5:
        parser.error("at most five observed editor sections may be expanded per entity")
    if args.mode == "inspect-detail":
        if not args.detail_url:
            parser.error("inspect-detail requires the actual observed detail URL")
        detail = urlsplit(args.detail_url)
        site = urlsplit(args.site_url)
        if (detail.scheme, detail.netloc) != (site.scheme, site.netloc):
            parser.error("detail URL must stay on the task's single site")
    return args


def _record(page: Any, output: Path, name: str) -> dict[str, Any]:
    """Archive rendered grid data and the controls used to obtain that data."""
    observation = page.evaluate(
        """() => {
          const visible = e => {
            const bounds = e.getBoundingClientRect();
            return !!(bounds.width && bounds.height);
          };
          const text = e => e.innerText.trim();
          const grid = Array.from(document.querySelectorAll('table.data-grid')).find(visible);
          const controls = Array.from(document.querySelectorAll(
            'input, select, textarea, button, [contenteditable=true]'))
            .filter(visible).map(e => ({
              tag: e.tagName.toLowerCase(), type: e.type, name: e.name, id: e.id,
              text: text(e), title: e.title, value: e.value,
              checked: 'checked' in e ? e.checked : undefined,
              class_name: e.className, disabled: e.disabled,
              aria_label: e.getAttribute('aria-label'),
              data_bind: e.getAttribute('data-bind'),
              labels: e.labels ? Array.from(e.labels).map(text) : undefined,
              form: e.form ? {
                id: e.form.id, action: e.form.action, method: e.form.method
              } : undefined,
              placeholder: e.getAttribute('placeholder'),
              data_index: e.closest('[data-index]')?.getAttribute('data-index'),
              options: e.tagName === 'SELECT' ? Array.from(e.options).map(o => ({
                label: o.textContent.trim(), value: o.value, selected: o.selected
              })) : undefined
            }));
          return {
            url: location.href, title: document.title, body_text: document.body.innerText,
            grid: grid ? {
              headers: Array.from(grid.querySelectorAll('thead th')).map(text),
              rows: Array.from(grid.querySelectorAll('tbody tr')).filter(visible)
                .map(tr => Array.from(tr.querySelectorAll('td')).map(text)),
              row_links: Array.from(grid.querySelectorAll('tbody tr')).filter(visible)
                .map(tr => Array.from(tr.querySelectorAll('a[href]')).filter(visible)
                  .map(a => ({text: text(a), url: a.href})))
            } : null,
            controls,
            headings: Array.from(document.querySelectorAll(
              'h1, h2, h3, [role=tab], .fieldset-wrapper-title')).filter(visible).map(text),
            links: Array.from(document.querySelectorAll('a[href]')).filter(visible)
              .map(a => ({text: text(a), url: a.href})),
            forms: Array.from(document.querySelectorAll('form')).filter(visible)
              .map(form => ({id: form.id, action: form.action, method: form.method}))
          };
        }"""
    )
    observation["frames"] = [
        {"url": frame.url, "name": frame.name, "body_text": frame.locator("body").inner_text()}
        for frame in page.frames
        if frame != page.main_frame and frame.frame_element().is_visible()
    ]
    page.screenshot(path=str(output / f"{name}.png"))
    (output / f"{name}.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
    )
    return observation


def _collect_order_pages(page: Any, output: Path) -> list[dict[str, Any]]:
    """Record every rendered order page through the disabled Next Page control."""
    pages = []
    while True:
        observation = _record(page, output, f"page_{len(pages):03}")
        pages.append(observation)
        next_page = page.locator("button[title='Next Page']:visible")
        if next_page.is_disabled():
            break
        old_rows = page.locator("table.data-grid tbody:visible").inner_text()
        next_page.click()
        page.wait_for_function(
            """previous => {
                const rows = Array.from(document.querySelectorAll('table.data-grid tbody'))
                    .find(element => element.getBoundingClientRect().height > 0);
                return rows && rows.innerText !== previous;
            }""",
            arg=old_rows,
            timeout=60_000,
        )
        page.wait_for_load_state("networkidle", timeout=60_000)
        page.wait_for_timeout(1_000)
    return pages


def _save_product(page: Any, output: Path, index: int, edit_url: str) -> dict[str, Any]:
    """Record a real Save POST and revisit the editor to read persisted values."""
    _record(page, output, f"unsaved_{index:02}")
    with page.expect_response(
        lambda response: (
            response.request.method == "POST"
            and re.search(r"/catalog/product/save(?:/|$)", urlsplit(response.url).path)
        ),
        timeout=60_000,
    ) as saved:
        page.locator("button#save-button:visible").click()
    response = saved.value
    receipt = {"url": response.url, "method": "POST", "response_status": response.status}
    (output / f"save_{index:02}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    page.wait_for_load_state("networkidle", timeout=60_000)
    page.goto(edit_url, wait_until="networkidle", timeout=120_000)
    page.reload(wait_until="networkidle", timeout=120_000)
    page.locator("input[name='product[sku]']:visible").wait_for(timeout=60_000)
    return receipt


def main() -> None:
    args = _parse_args()
    baseline = json.loads(args.baseline_receipt.read_text())
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("authoring requires a coordinator-owned fixed image baseline")
    if baseline["site_url"].rstrip("/") != args.site_url.rstrip("/"):
        raise ValueError("author endpoint differs from the coordinator baseline receipt")
    output = args.batch_root / "author" / args.generated_task_id / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    definition = None
    revision = 1
    source_artifact = "sites/amazon_jobs/tasks.jsonl"

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            storage_state=str(args.storage_state),
            viewport={"width": 1920, "height": 1080},
            record_har_path=str(output / "network.har"),
            record_har_content="embed",
        )
        context.tracing.start(screenshots=True, snapshots=True, sources=False)
        page = context.new_page()
        try:
            page.goto(args.site_url.rstrip("/") + "/", wait_until="networkidle", timeout=120_000)
            _record(page, output, "initial")
            paths = {
                "inspect-orders": "sales/order/",
                "retrieve-orders": "sales/order/",
                "retrieve-orders-aggregate": "sales/order/",
                "inspect-products": "catalog/product/",
                "inspect-customers": "customer/index/",
                "retrieve-product-attributes": "catalog/product/",
                "navigate-product-attributes": "catalog/product/",
                "retrieve-simple-product-prices": "catalog/product/",
                "prepare-product-mutations": "catalog/product/",
                "mutate-product-name-price": "catalog/product/",
                "mutate-product-attributes": "catalog/product/",
                "mutate-selected-product-attributes": "catalog/product/",
                "mutate-selected-product-price-options": "catalog/product/",
                "mutate-orders-topk-derived-product-price": "sales/order/",
            }
            page.goto(
                args.detail_url
                if args.mode == "inspect-detail"
                else args.site_url.rstrip("/") + "/" + paths[args.mode],
                wait_until="networkidle",
                timeout=120_000,
            )
            page.wait_for_timeout(1_000)
            if args.mode == "inspect-detail":
                _record(page, output, "detail")
                for index, label in enumerate(args.expand_section):
                    section_label = page.locator(
                        ".fieldset-wrapper-title:visible > .admin__collapsible-title "
                        "> span:first-child"
                    ).filter(has_text=re.compile(r"^\s*" + re.escape(label) + r"\s*$"))
                    section_label.click()
                    page.wait_for_function(
                        "element => element.dataset.stateCollapsible === 'open'",
                        arg=section_label.locator("xpath=../..").element_handle(),
                        timeout=60_000,
                    )
                    page.wait_for_load_state("networkidle", timeout=60_000)
                    _record(page, output, f"detail_section_{index:02}")
            else:
                page.locator("table.data-grid tbody tr:visible").first.wait_for(timeout=60_000)
                listing = _record(page, output, args.mode.split("-")[-1])
                if args.mode in {"inspect-products", "inspect-customers"}:
                    edit_path = (
                        "/catalog/product/edit/"
                        if args.mode == "inspect-products"
                        else "/customer/index/edit/"
                    )
                    selected = []
                    for row_index, links in enumerate(listing["grid"]["row_links"]):
                        targets = list(
                            dict.fromkeys(
                                link["url"]
                                for link in links
                                if edit_path in urlsplit(link["url"]).path
                            )
                        )
                        if targets:
                            selected.append({"row_index": row_index, "url": targets[0]})
                        if len(selected) == args.entity_limit:
                            break
                    # Only actual rendered links determine target URLs. Missing
                    # customer editor links remain an explicit exploration result.
                    (output / "selected_entities.json").write_text(
                        json.dumps(selected, ensure_ascii=False, indent=2) + "\n"
                    )
                    for index, target in enumerate(selected):
                        page.goto(target["url"], wait_until="networkidle", timeout=120_000)
                        page.wait_for_timeout(1_000)
                        _record(page, output, f"editor_{index:02}")
                        for section_index, label in enumerate(args.expand_section):
                            section_label = page.locator(
                                ".fieldset-wrapper-title:visible > .admin__collapsible-title "
                                "> span:first-child"
                            ).filter(has_text=re.compile(r"^\s*" + re.escape(label) + r"\s*$"))
                            section_label.click()
                            page.wait_for_function(
                                "element => element.dataset.stateCollapsible === 'open'",
                                arg=section_label.locator("xpath=../..").element_handle(),
                                timeout=60_000,
                            )
                            page.wait_for_load_state("networkidle", timeout=60_000)
                            _record(page, output, f"editor_{index:02}_section_{section_index:02}")
                elif args.mode in {
                    "inspect-orders",
                    "retrieve-orders",
                    "mutate-orders-topk-derived-product-price",
                }:
                    page.locator("button:visible").filter(
                        has_text=re.compile(r"^\s*Filters\s*$")
                    ).click()
                    _record(page, output, "filters")
            if args.mode in {"retrieve-orders", "mutate-orders-topk-derived-product-price"}:
                config = json.loads(args.task_config.read_text())
                parameters = config["parameters"]
                # Apply only visible filters observed on this fixed baseline.
                page.locator(config["status_selector"]).select_option(label=parameters["status"])
                date_from = datetime.strptime(parameters["purchase_date_from"], "%Y-%m-%d").date()
                date_to = datetime.strptime(parameters["purchase_date_to"], "%Y-%m-%d").date()
                page.locator("input[name='created_at[from]']").fill(date_from.strftime("%m/%d/%Y"))
                page.locator("input[name='created_at[to]']").fill(date_to.strftime("%m/%d/%Y"))
                page.locator("input[name='base_grand_total[from]']").fill(
                    str(parameters["minimum_total_usd"])
                )
                page.locator("button:visible").filter(
                    has_text=re.compile(r"^\s*Apply Filters\s*$")
                ).click()
                page.wait_for_load_state("networkidle", timeout=60_000)
                page.wait_for_timeout(1_000)

                # Visit every filtered page before ranking, retaining raw visible
                # cells and their exact locations for field-level answer evidence.
                pages = _collect_order_pages(page, output)
                columns = {
                    "order_id": "ID",
                    "purchase_date": "Purchase Date",
                    "bill_to_name": "Bill-to Name",
                    "grand_total_usd": "Grand Total (Base)",
                }
                grounded_rows = []
                for page_index, observation in enumerate(pages):
                    grid = observation["grid"]
                    indices = {
                        name: grid["headers"].index(label) for name, label in columns.items()
                    }
                    status_index = grid["headers"].index("Status")
                    for row_index, cells in enumerate(grid["rows"]):
                        row = {name: cells[index] for name, index in indices.items()}
                        total = Decimal(row["grand_total_usd"].removeprefix("$").replace(",", ""))
                        purchased = datetime.strptime(row["purchase_date"], "%b %d, %Y %I:%M:%S %p")
                        if not (
                            cells[status_index] == parameters["status"]
                            and date_from <= purchased.date() <= date_to
                            and total >= Decimal(str(parameters["minimum_total_usd"]))
                        ):
                            raise ValueError(
                                "rendered row violates the requested visible UI filters"
                            )
                        row["grand_total_usd"] = (
                            str(total)
                            if args.mode == "mutate-orders-topk-derived-product-price"
                            else float(total)
                        )
                        grounded_rows.append((row, page_index, row_index, indices))
                ids = [row[0]["order_id"] for row in grounded_rows]
                if len(ids) != len(set(ids)):
                    raise ValueError("pagination repeated an order; complete grounding is unproven")
                record_count = int(re.search(r"(\d+) records found", pages[0]["body_text"])[1])
                if record_count != len(grounded_rows):
                    raise ValueError("collected visible rows differ from the filtered grid count")
                grounded_rows.sort(
                    key=lambda record: (
                        -Decimal(str(record[0]["grand_total_usd"])),
                        int(record[0]["order_id"]),
                    )
                )
                selected = grounded_rows[: parameters["limit"]]
                if len(selected) != parameters["limit"]:
                    raise ValueError("the requested top-k exceeds the grounded qualifying orders")
                if args.mode == "retrieve-orders":
                    answer_values = [record[0] for record in selected]
                    evidence = [
                        {
                            "field": f"/task_definition/answer/values/{index}/{name}",
                            "artifact": "observation",
                            "location": (
                                f"/pages/{page_index}/grid/rows/{row_index}/{indices[name]}"
                            ),
                            "modality": "text",
                            "derivation": (
                                "Parse the displayed USD amount as a JSON number."
                                if name == "grand_total_usd"
                                else "Exact displayed grid cell."
                            ),
                        }
                        for index, (_, page_index, row_index, indices) in enumerate(selected)
                        for name in columns
                    ]
                    definition = {
                        "task_type": "RETRIEVE",
                        "goal_template": (
                            "In Orders, find orders whose Status is {status}, whose displayed "
                            "Purchase Date is from {purchase_date_from} through "
                            "{purchase_date_to} inclusive (include the entire final day), and "
                            "whose Grand Total (Base) is at least {minimum_total_usd} USD. "
                            "Return the {limit} qualifying orders with the largest Grand Total"
                            " (Base), in descending total order; for equal totals, put the "
                            "smaller numeric order ID first. For each order, report order_id "
                            "as the exact displayed ID including leading zeros, purchase_date "
                            "as the exact displayed date and time, bill_to_name, and "
                            "grand_total_usd as a number without a currency symbol. Use the "
                            "Orders grid's displayed date and time without timezone "
                            "conversion."
                        ),
                        "parameters": parameters,
                        "answer": {
                            "status": "SUCCESS",
                            "values": answer_values,
                            "ordered": True,
                            "item_schema": {
                                "type": "object",
                                "properties": {
                                    name: {
                                        "type": "number" if name == "grand_total_usd" else "string"
                                    }
                                    for name in columns
                                },
                                "required": list(columns),
                                "additionalProperties": False,
                            },
                        },
                        # Retrieval can legitimately use filtering, searching, or
                        # manual browsing; no particular request path is required.
                        "completion_events": [],
                    }
                    archived_observation = {
                        "parameters": parameters,
                        "pages": pages,
                        "filtered_count": record_count,
                        "all_pages_collected": True,
                        "ranking": "grand_total_usd descending, numeric order_id ascending",
                        "selected_rows": answer_values,
                    }
                    family = "orders_filtered_ranked_records"
                    source_record = "Amazon Jobs--5"
                    screenshot = "page_000.png"
                    final_checks = []
                if args.mode == "mutate-orders-topk-derived-product-price":
                    if type(parameters["limit"]) is not int or parameters["limit"] not in {
                        5,
                        8,
                        12,
                    }:
                        raise ValueError("order-derived price requires a top-k of 5, 8 or 12")
                    derived_price = (
                        sum(Decimal(record[0]["grand_total_usd"]) for record in selected)
                        / Decimal(parameters["limit"])
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    orders_observation = {
                        "parameters": parameters,
                        "pages": pages,
                        "filtered_count": record_count,
                        "all_pages_collected": True,
                        "ranking": "grand_total_usd descending, numeric order_id ascending",
                        "selected_rows": [record[0] for record in selected],
                    }
                    order_evidence = [
                        {
                            "field": "/task_definition/parameters/expected_price",
                            "artifact": "observation",
                            "location": f"/orders/pages/{page_index}/grid/rows/{row_index}/{cell}",
                            "modality": "text",
                            "derivation": (
                                "Validate Status, inclusive displayed Purchase Date and minimum "
                                "Grand Total (Base) for every unique collected ID. Rank all "
                                "qualifying amounts as exact decimals descending and numeric "
                                "ID ascending; sum the first requested K amounts, divide by K "
                                "and round half-up to USD cents."
                            ),
                        }
                        for _, page_index, row_index, indices in grounded_rows
                        for cell in [
                            indices["order_id"],
                            indices["purchase_date"],
                            indices["grand_total_usd"],
                            pages[page_index]["grid"]["headers"].index("Status"),
                        ]
                    ]
            elif args.mode == "retrieve-orders-aggregate":
                config = json.loads(args.task_config.read_text())
                parameters = config["parameters"]
                pages = _collect_order_pages(page, output)
                record_count = int(re.search(r"(\d+) records found", pages[0]["body_text"])[1])
                if record_count != config["expected_record_count"]:
                    raise ValueError(
                        "the formal Orders view does not cover the fixed-baseline total"
                    )
                groups = {}
                order_ids = []
                for page_index, observation in enumerate(pages):
                    grid = observation["grid"]
                    id_index = grid["headers"].index("ID")
                    status_index = grid["headers"].index("Status")
                    amount_index = grid["headers"].index(parameters["amount_column"])
                    for row_index, cells in enumerate(grid["rows"]):
                        order_ids.append(cells[id_index])
                        status = cells[status_index]
                        group = groups.setdefault(
                            status, {"amount": Decimal("0.00"), "sources": []}
                        )
                        group["amount"] += Decimal(
                            cells[amount_index].removeprefix("$").replace(",", "")
                        )
                        group["sources"].append(
                            {
                                "row": f"/pages/{page_index}/grid/rows/{row_index}",
                                "id_index": id_index,
                                "status_index": status_index,
                                "amount_index": amount_index,
                            }
                        )
                if len(order_ids) != len(set(order_ids)):
                    raise ValueError("pagination repeated an order; aggregate coverage is unproven")
                if len(order_ids) != record_count:
                    raise ValueError(
                        "collected visible rows differ from the full Orders grid count"
                    )
                ranked = sorted(groups.items(), key=lambda item: (-item[1]["amount"], item[0]))
                answer_values = [
                    {
                        "status": status,
                        "order_count": len(group["sources"]),
                        "grand_total_usd": float(group["amount"]),
                    }
                    for status, group in ranked
                ]
                evidence = []
                for index, (status, group) in enumerate(ranked):
                    first = group["sources"][0]
                    evidence.append(
                        {
                            "field": f"/task_definition/answer/values/{index}/status",
                            "artifact": "observation",
                            "location": f"{first['row']}/{first['status_index']}",
                            "modality": "text",
                            "derivation": "Exact displayed Status cell for this group.",
                        }
                    )
                    for source in group["sources"]:
                        for field, cell_index, derivation in [
                            (
                                "order_count",
                                source["id_index"],
                                "Count the distinct displayed ID cells bound to this field across "
                                "all recorded pages; each paired Status cell equals this group.",
                            ),
                            (
                                "grand_total_usd",
                                source["amount_index"],
                                "Parse all displayed USD amounts bound to this field as exact "
                                "decimals and sum them; each paired Status cell equals this group.",
                            ),
                        ]:
                            evidence.append(
                                {
                                    "field": f"/task_definition/answer/values/{index}/{field}",
                                    "artifact": "observation",
                                    "location": f"{source['row']}/{cell_index}",
                                    "modality": "text",
                                    "derivation": (
                                        derivation + f" Paired Status: {source['row']}/"
                                        f"{source['status_index']}."
                                    ),
                                }
                            )
                definition = {
                    "task_type": "RETRIEVE",
                    "goal_template": (
                        "Audit all existing Orders across every page, without applying date, "
                        "customer, amount, or status filters. Group the orders by their displayed "
                        "Status. For every status that occurs, report status as the exact "
                        "displayed "
                        "label, order_count as the number of distinct orders in that status, and "
                        "grand_total_usd as the sum of their {amount_column} amounts in USD. "
                        "Include every status present in the full Orders grid. Sort the groups "
                        "by grand_total_usd descending; for equal totals, sort status labels "
                        "alphabetically ascending. Sum the displayed USD values without currency "
                        "conversion, and return amounts as numbers without currency symbols. "
                        "Do not modify any order."
                    ),
                    "parameters": parameters,
                    "answer": {
                        "status": "SUCCESS",
                        "values": answer_values,
                        "ordered": True,
                        "item_schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "order_count": {"type": "integer"},
                                "grand_total_usd": {"type": "number"},
                            },
                            "required": ["status", "order_count", "grand_total_usd"],
                            "additionalProperties": False,
                        },
                    },
                    "completion_events": [],
                }
                archived_observation = {
                    "parameters": parameters,
                    "pages": pages,
                    "record_count": record_count,
                    "all_pages_collected": True,
                    "unique_order_ids": len(set(order_ids)),
                    "ranking": "grand_total_usd descending, status label ascending",
                    "status_groups": answer_values,
                }
                family = "orders_status_aggregate"
                source_record = "Amazon Jobs--13"
                screenshot = "page_000.png"
                final_checks = []
            if args.mode in {
                "prepare-product-mutations",
                "mutate-product-name-price",
                "mutate-product-attributes",
                "mutate-selected-product-attributes",
                "mutate-selected-product-price-options",
                "mutate-orders-topk-derived-product-price",
            }:
                config = json.loads(args.task_config.read_text())
                revision = config["revision"]
                parameters = dict(config["parameters"])
                skus = (
                    parameters["candidate_skus"]
                    if args.mode
                    in {
                        "mutate-selected-product-attributes",
                        "mutate-selected-product-price-options",
                        "mutate-orders-topk-derived-product-price",
                    }
                    else config["target_skus"]
                )
                editors = []
                indices_by_editor = []
                urls = []
                for index, sku in enumerate(skus):
                    url = config["product_links"][sku].replace(
                        "__SHOPPING_ADMIN__", args.site_url.rstrip("/")
                    )
                    page.goto(url, wait_until="networkidle", timeout=120_000)
                    page.wait_for_timeout(1_000)
                    captured = _record(page, output, f"before_{index:02}")
                    indices = {
                        control["name"]: number
                        for number, control in enumerate(captured["controls"])
                        if control.get("name", "").startswith("product[")
                    }
                    if captured["controls"][indices["product[sku]"]]["value"] != sku:
                        raise ValueError("the mutation editor link identifies a different SKU")
                    editors.append(captured)
                    indices_by_editor.append(indices)
                    urls.append(url)
                archived_observation = {"before": editors, "after": [], "save_requests": []}
                if args.mode == "mutate-orders-topk-derived-product-price":
                    archived_observation["orders"] = orders_observation
                if args.mode == "prepare-product-mutations":
                    (output / "mutation_readiness.json").write_text(
                        json.dumps(
                            {
                                "status": "initial_controls_recorded_without_saving",
                                "family": config["family"],
                                "target_skus": skus,
                                "post_body_and_persistence": "not_yet_observed",
                                "ground_truth": "not_authored",
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                else:
                    initial_fields = [
                        {
                            field: captured["controls"][indices[f"product[{field}]"]]["value"]
                            for field in ["name", "sku", "price", "weight"]
                        }
                        for captured, indices in zip(editors, indices_by_editor)
                    ]
                    evidence = (
                        list(order_evidence)
                        if args.mode == "mutate-orders-topk-derived-product-price"
                        else []
                    )
                    final_checks = []
                    if args.mode == "mutate-product-name-price":
                        prices = [Decimal(fields["price"]) for fields in initial_fields]
                        if len(prices) != 2 or prices[0] == prices[1]:
                            raise ValueError(
                                "conditional name-price revision requires two unequal prices"
                            )
                        cheaper = min(range(2), key=lambda index: prices[index])
                        mutation_indices = [cheaper]
                        expected_fields = [dict(fields) for fields in initial_fields]
                        expected_fields[cheaper]["name"] += parameters["name_suffix"]
                        expected_fields[cheaper]["price"] = str(
                            (
                                prices[cheaper]
                                * (Decimal("100") - Decimal(str(parameters["discount_percent"])))
                                / Decimal("100")
                            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                        )
                        goal = (
                            "Compare the current regular Price fields of the simple products "
                            "with SKUs {sku_1} and {sku_2}. Append the literal suffix "
                            '"{name_suffix}" to the current '
                            "Product Name of the cheaper product. Reduce that same cheaper "
                            "product's regular Price by "
                            "{discount_percent} percent, rounded to two decimal places with "
                            "half-up rounding. Save the cheaper product and finish on its "
                            "editor. Preserve its SKU and Weight fields exactly. Do not edit "
                            "the more expensive product."
                        )
                        family = "products_conditional_name_price_update"
                    else:
                        selected_index = 0
                        if args.mode == "mutate-selected-product-attributes":
                            if len(skus) not in {3, 4}:
                                raise ValueError("conditional attribute revision requires 3–4 SKUs")
                            matches = []
                            for index, (captured, indices) in enumerate(
                                zip(editors, indices_by_editor)
                            ):
                                selected_labels = {
                                    attribute: [
                                        option["label"]
                                        for option in captured["controls"][
                                            indices[f"product[{attribute}]"]
                                        ]["options"]
                                        if option["selected"]
                                    ]
                                    for attribute in ["material", "climate"]
                                }
                                if all(
                                    parameters[f"{attribute}_condition"]
                                    in selected_labels[attribute]
                                    and len(selected_labels[attribute])
                                    == parameters[f"{attribute}_selected_count"]
                                    for attribute in ["material", "climate"]
                                ):
                                    matches.append(index)
                                evidence.append(
                                    {
                                        "field": (
                                            f"/task_definition/parameters/candidate_skus/{index}"
                                        ),
                                        "artifact": "observation",
                                        "location": (
                                            f"/before/{index}/controls/"
                                            f"{indices['product[sku]']}/value"
                                        ),
                                        "modality": "text",
                                    }
                                )
                                for attribute in ["material", "climate"]:
                                    for condition in ["condition", "selected_count"]:
                                        evidence.append(
                                            {
                                                "field": (
                                                    "/task_definition/parameters/"
                                                    f"{attribute}_{condition}"
                                                ),
                                                "artifact": "observation",
                                                "location": (
                                                    f"/before/{index}/controls/"
                                                    f"{indices[f'product[{attribute}]']}/options"
                                                ),
                                                "modality": "text",
                                                "derivation": (
                                                    "Apply the requested initial selected-label "
                                                    "membership and selected-option count to "
                                                    "every formal candidate before editing."
                                                ),
                                            }
                                        )
                            if len(matches) != 1:
                                raise ValueError(
                                    "initial Material/Climate conditions do not identify "
                                    "one candidate"
                                )
                            selected_index = matches[0]
                            archived_observation["matching_candidate_indices"] = matches
                        elif args.mode in {
                            "mutate-selected-product-price-options",
                            "mutate-orders-topk-derived-product-price",
                        }:
                            if len(skus) != 2 or parameters["selection_rule"] not in {
                                "cheapest",
                                "most_expensive",
                            }:
                                raise ValueError("price-option revision requires two ranked SKUs")
                            if args.mode == "mutate-selected-product-price-options":
                                if parameters["price_operation"] not in {"increase", "decrease"}:
                                    raise ValueError("price_operation must be increase or decrease")
                                rate = Decimal(str(parameters["price_change_percent"]))
                                if not Decimal("0") < rate < Decimal("100"):
                                    raise ValueError(
                                        "price_change_percent must be between 0 and 100"
                                    )
                            prices = [Decimal(fields["price"]) for fields in initial_fields]
                            direction = 1 if parameters["selection_rule"] == "cheapest" else -1
                            selected_index = sorted(
                                range(2), key=lambda index: (direction * prices[index], skus[index])
                            )[0]
                            archived_observation["selected_candidate_index"] = selected_index
                            for index, indices in enumerate(indices_by_editor):
                                for field in ["price", "sku"]:
                                    evidence.append(
                                        {
                                            "field": "/task_definition/parameters/sku",
                                            "artifact": "observation",
                                            "location": (
                                                f"/before/{index}/controls/"
                                                f"{indices[f'product[{field}]']}/value"
                                            ),
                                            "modality": "text",
                                            "derivation": (
                                                "Compare both formal initial regular Prices; "
                                                "apply the requested price ranking and "
                                                "alphabetical SKU tie-break before any edit."
                                            ),
                                        }
                                    )
                        elif len(skus) != 1:
                            raise ValueError("selected-attribute revision requires one product")
                        mutation_indices = [selected_index]
                        expected_fields = [dict(fields) for fields in initial_fields]
                        if args.mode == "mutate-selected-product-price-options":
                            sign = 1 if parameters["price_operation"] == "increase" else -1
                            expected_fields[selected_index]["price"] = str(
                                (
                                    prices[selected_index]
                                    * (Decimal("100") + sign * rate)
                                    / Decimal("100")
                                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                            )
                        elif args.mode == "mutate-orders-topk-derived-product-price":
                            expected_fields[selected_index]["price"] = str(derived_price)
                        selected_values = {}
                        for attribute in ["material", "climate"]:
                            options = editors[selected_index]["controls"][
                                indices_by_editor[selected_index][f"product[{attribute}]"]
                            ]["options"]
                            if args.mode == "mutate-orders-topk-derived-product-price":
                                preserved = [option for option in options if option["selected"]]
                                selected_values[attribute] = [
                                    option["value"] for option in preserved
                                ]
                                parameters[f"{attribute}_preserved"] = [
                                    option["label"] for option in preserved
                                ]
                                parameters[f"{attribute}_added_count"] = 0
                                parameters[f"{attribute}_added_name"] = options[0]["label"]
                                evidence.extend(
                                    [
                                        {
                                            "field": (
                                                "/task_definition/parameters/"
                                                f"{attribute}_added_name"
                                            ),
                                            "artifact": "observation",
                                            "location": (
                                                f"/before/{selected_index}/controls/"
                                                f"{indices_by_editor[selected_index][f'product[{attribute}]']}/options/0/label"
                                            ),
                                            "modality": "text",
                                            "derivation": (
                                                "Actual offered label required by the existing "
                                                "exact-set reader; zero added_count allows no "
                                                "new labels, including this label."
                                            ),
                                        },
                                        {
                                            "field": (
                                                "/task_definition/parameters/"
                                                f"{attribute}_added_count"
                                            ),
                                            "artifact": "observation",
                                            "location": (
                                                f"/before/{selected_index}/controls/"
                                                f"{indices_by_editor[selected_index][f'product[{attribute}]']}/options"
                                            ),
                                            "modality": "text",
                                            "derivation": (
                                                "The task preserves this complete initial set "
                                                "and allows zero new members."
                                            ),
                                        },
                                    ]
                                )
                                evidence.append(
                                    {
                                        "field": (
                                            f"/task_definition/parameters/{attribute}_preserved"
                                        ),
                                        "artifact": "observation",
                                        "location": (
                                            f"/before/{selected_index}/controls/"
                                            f"{indices_by_editor[selected_index][f'product[{attribute}]']}/options"
                                        ),
                                        "modality": "text",
                                        "derivation": (
                                            "Preserve the entire initial selected-label set, "
                                            "including an empty set; allow zero added members."
                                        ),
                                    }
                                )
                                final_checks.append(
                                    {
                                        "kind": "text_set_preserved",
                                        "parameter": f"{attribute}_preserved",
                                        "selector": (
                                            f"select[name='product[{attribute}]'] option:checked"
                                        ),
                                        "added_count_parameter": f"{attribute}_added_count",
                                        "added_name_parameter": f"{attribute}_added_name",
                                    }
                                )
                                continue
                            add = next(
                                option
                                for option in options
                                if option["label"] == parameters[f"{attribute}_add"]
                            )
                            if args.mode == "mutate-selected-product-price-options":
                                if add["selected"]:
                                    raise ValueError("requested added option is already selected")
                                preserved = [option for option in options if option["selected"]]
                            else:
                                remove = next(
                                    option
                                    for option in options
                                    if option["label"] == parameters[f"{attribute}_remove"]
                                )
                                if not remove["selected"] or add["selected"]:
                                    raise ValueError(
                                        "requested attribute revision differs from the "
                                        "formal baseline"
                                    )
                                preserved = [
                                    option
                                    for option in options
                                    if option["selected"] and option != remove
                                ]
                            parameters[f"{attribute}_preserved"] = [
                                option["label"] for option in preserved
                            ]
                            parameters[f"{attribute}_added_count"] = 1
                            selected_values[attribute] = [
                                option["value"]
                                for option in options
                                if option in preserved or option == add
                            ]
                            evidence.append(
                                {
                                    "field": f"/task_definition/parameters/{attribute}_preserved",
                                    "artifact": "observation",
                                    "location": (
                                        f"/before/{selected_index}/controls/"
                                        f"{indices_by_editor[selected_index][f'product[{attribute}]']}/options"
                                    ),
                                    "modality": "text",
                                    "derivation": (
                                        "Preserve every initially selected option before adding "
                                        "the requested new member."
                                        if args.mode == "mutate-selected-product-price-options"
                                        else "Preserve initial selections except the removal."
                                    ),
                                }
                            )
                            final_checks.append(
                                {
                                    "kind": "text_set_preserved",
                                    "parameter": f"{attribute}_preserved",
                                    "selector": (
                                        f"select[name='product[{attribute}]'] option:checked"
                                    ),
                                    "added_count_parameter": f"{attribute}_added_count",
                                    "added_name_parameter": f"{attribute}_add",
                                }
                            )
                        goal = (
                            "Edit the existing product with exact SKU {sku}. In Material, "
                            "deselect {material_remove} and select {material_add}. In Climate, "
                            "deselect {climate_remove} and select {climate_add}. Keep every "
                            "other currently selected Material and Climate option selected, "
                            "and do not select any other new option. Save the product. "
                            "Preserve its Product Name, SKU, Price and Weight fields exactly "
                            "and finish on this product's editor."
                        )
                        family = "product_selected_attribute_revision"
                        if args.mode == "mutate-selected-product-attributes":
                            goal = (
                                "Among the existing products with SKUs {candidate_skus}, "
                                "find the unique product whose initially selected Material "
                                "options include {material_condition} and total exactly "
                                "{material_selected_count}, and whose initially selected "
                                "Climate options include {climate_condition} and total exactly "
                                "{climate_selected_count}. Apply these conditions to the saved "
                                "state before making any changes. Edit only that product. "
                                "In Material, deselect {material_remove} and select "
                                "{material_add}. In Climate, deselect {climate_remove} and "
                                "select {climate_add}. Keep every other initially selected "
                                "Material and Climate option selected, and do not select any "
                                "other new option. Save the selected product and revisit or "
                                "refresh its editor to confirm the persisted selections. "
                                "Preserve its Product Name, SKU, Price and Weight fields "
                                "exactly. Do not save or change any other candidate product. "
                                "Finish on the selected product's editor."
                            )
                            family = "products_attribute_selected_revision"
                        elif args.mode == "mutate-selected-product-price-options":
                            goal = (
                                "Compare the saved regular Price fields of the existing products "
                                "with SKUs {candidate_skus}. Select the {selection_rule} product: "
                                "cheapest means lowest Price and most_expensive means highest "
                                "Price. For tied Prices select the SKU alphabetically ascending. "
                                "Apply a {price_change_percent} percent {price_operation} to that "
                                "product's initial regular Price, rounded to two decimal places "
                                "with half-up rounding. Increase means add that percentage of "
                                "the initial Price; decrease means subtract it. In Material, "
                                "add {material_add} to its initially selected options. In Climate, "
                                "add {climate_add} to its initially selected options. Preserve "
                                "every other initially selected Material and Climate option and "
                                "do not select any other new option. Keep Product Name, SKU and "
                                "Weight exactly unchanged. Save only the selected product, "
                                "revisit or refresh to confirm persistence, and finish on its "
                                "editor. Do not edit or save the other candidate."
                            )
                            family = "products_price_ranked_option_addition"
                        elif args.mode == "mutate-orders-topk-derived-product-price":
                            goal = (
                                "In Orders, find every order whose Status is {status}, whose "
                                "displayed Purchase Date is from {purchase_date_from} through "
                                "{purchase_date_to} inclusive, including the entire final day, "
                                "and whose Grand Total (Base) is at least {minimum_total_usd} USD. "
                                "Use all filtered pages. Rank the qualifying orders by Grand "
                                "Total (Base) descending, with smaller numeric order ID first "
                                "for equal totals. Calculate the arithmetic mean of the first "
                                "{limit} Grand Total (Base) amounts, in USD, using exact decimal "
                                "amounts and rounding the final mean half-up to two decimal "
                                "places. Use the displayed dates without timezone conversion. "
                                "Compare the saved regular Price fields of the existing products "
                                "with SKUs {candidate_skus}. Select the {selection_rule} product: "
                                "cheapest means lowest Price and most_expensive means highest "
                                "Price; for tied Prices select the SKU alphabetically ascending. "
                                "Set only that product's regular Price to the calculated USD "
                                "mean. Preserve its Product Name, SKU, Weight and the complete "
                                "initially selected Material and Climate sets, including empty "
                                "sets; do not add or remove any option. Save only this product, "
                                "revisit or refresh its editor to confirm persistence, and "
                                "finish on its editor. Do not change any order or edit or save "
                                "the other candidate product."
                            )
                            family = "orders_topk_derived_product_price"
                    save_receipts = []
                    for result_index, index in enumerate(mutation_indices):
                        url = urls[index]
                        expected = expected_fields[index]
                        page.goto(url, wait_until="networkidle", timeout=120_000)
                        if args.mode == "mutate-product-name-price":
                            page.locator("input[name='product[name]']:visible").fill(
                                expected["name"]
                            )
                            page.locator("input[name='product[price]']:visible").fill(
                                expected["price"]
                            )
                        else:
                            if args.mode in {
                                "mutate-selected-product-price-options",
                                "mutate-orders-topk-derived-product-price",
                            }:
                                page.locator("input[name='product[price]']:visible").fill(
                                    expected["price"]
                                )
                            if args.mode != "mutate-orders-topk-derived-product-price":
                                for attribute, selected in selected_values.items():
                                    page.locator(
                                        f"select[name='product[{attribute}]']:visible"
                                    ).select_option(selected)
                        save_receipts.append(_save_product(page, output, result_index, url))
                        after = _record(page, output, f"after_{result_index:02}")
                        archived_observation["after"].append(after)
                        after_indices = {
                            control["name"]: number
                            for number, control in enumerate(after["controls"])
                            if control.get("name", "").startswith("product[")
                        }
                        for field in ["name", "sku", "price", "weight"]:
                            actual = after["controls"][after_indices[f"product[{field}]"]]["value"]
                            passed = (
                                Decimal(actual) == Decimal(expected[field])
                                if field == "price"
                                and args.mode
                                in {
                                    "mutate-product-name-price",
                                    "mutate-selected-product-price-options",
                                    "mutate-orders-topk-derived-product-price",
                                }
                                else actual == expected[field]
                            )
                            if not passed:
                                raise ValueError(
                                    f"persisted {field} differs from the requested mutation"
                                )
                            parameter = (
                                f"sku_{index + 1}"
                                if field == "sku"
                                and len(skus) == 2
                                and args.mode
                                not in {
                                    "mutate-selected-product-price-options",
                                    "mutate-orders-topk-derived-product-price",
                                }
                                else "sku"
                                if field == "sku"
                                else f"expected_{field}"
                            )
                            parameters[parameter] = actual
                            evidence.append(
                                {
                                    "field": f"/task_definition/parameters/{parameter}",
                                    "artifact": "observation",
                                    "location": (
                                        f"/before/{index}/controls/"
                                        f"{indices_by_editor[index][f'product[{field}]']}/value"
                                    ),
                                    "modality": "text",
                                    "derivation": (
                                        "Both initial product Prices determine the selected "
                                        "SKU; the bound Orders amounts independently determine "
                                        "the expected Price by exact top-k mean. Other initial "
                                        "fields are preserved exactly."
                                        if args.mode == "mutate-orders-topk-derived-product-price"
                                        else "Formal initial value used for exact preservation or "
                                        "the requested percentage price change with half-up "
                                        "cent rounding; after confirms the persisted result."
                                        if args.mode == "mutate-selected-product-price-options"
                                        else "Formal initial value used for preservation or the "
                                        "requested name suffix/price discount; the corresponding "
                                        "after observation confirms the persisted result."
                                    ),
                                }
                            )
                            evidence.append(
                                {
                                    "field": f"/task_definition/parameters/{parameter}",
                                    "artifact": "observation",
                                    "location": (
                                        f"/after/{result_index}/controls/"
                                        f"{after_indices[f'product[{field}]']}/value"
                                    ),
                                    "modality": "text",
                                    "derivation": (
                                        "Refreshed persisted value, checked against the formal "
                                        "initial value and requested task operation."
                                    ),
                                }
                            )
                            check = {
                                "kind": "input_value",
                                "parameter": parameter,
                                "selector": f"input[name='product[{field}]']",
                            }
                            final_checks.append(check)
                        if args.mode in {
                            "mutate-product-attributes",
                            "mutate-selected-product-attributes",
                            "mutate-selected-product-price-options",
                            "mutate-orders-topk-derived-product-price",
                        }:
                            for attribute, selected in selected_values.items():
                                persisted = [
                                    option["value"]
                                    for option in after["controls"][
                                        after_indices[f"product[{attribute}]"]
                                    ]["options"]
                                    if option["selected"]
                                ]
                                if persisted != selected:
                                    raise ValueError(
                                        "refreshed options differ from the requested exact set"
                                    )
                    archived_observation["save_requests"] = save_receipts
                    definition = {
                        "task_type": "MUTATE",
                        "goal_template": goal,
                        "parameters": parameters,
                        "answer": {
                            "status": "SUCCESS",
                            "values": None,
                            "item_schema": None,
                            "ordered": False,
                        },
                        "completion_events": [],
                    }
                    source_record = "Google Shopping--2"
                    source_artifact = "sites/google_shopping/tasks.jsonl"
                    screenshot = "after_00.png"
            elif args.mode == "retrieve-simple-product-prices":
                config = json.loads(args.task_config.read_text())
                parameters = config["parameters"]
                skus = [parameters["sku_1"], parameters["sku_2"]]
                editors = []
                control_indices = []
                prices = []
                for index, sku in enumerate(skus):
                    url = config["product_links"][sku].replace(
                        "__SHOPPING_ADMIN__", args.site_url.rstrip("/")
                    )
                    page.goto(url, wait_until="networkidle", timeout=120_000)
                    page.wait_for_timeout(1_000)
                    observation = _record(page, output, f"product_{index:02}")
                    editors.append(observation)
                    indices = {
                        control["name"]: number
                        for number, control in enumerate(observation["controls"])
                        if control.get("name", "").startswith("product[")
                    }
                    control_indices.append(indices)
                    controls = observation["controls"]
                    if controls[indices["product[sku]"]]["value"] != sku:
                        raise ValueError("the observed editor link identifies a different SKU")
                    prices.append(Decimal(controls[indices["product[price]"]]["value"]))
                cheapest = min(prices)
                answer_values = [
                    {
                        "sku": sku,
                        "product_name": observation["controls"][indices["product[name]"]]["value"],
                        "price_usd": float(price),
                        "price_above_cheapest_usd": float(price - cheapest),
                    }
                    for sku, observation, indices, price in zip(
                        skus, editors, control_indices, prices
                    )
                ]
                evidence = []
                for index, indices in enumerate(control_indices):
                    for field, control_name in {
                        "sku": "product[sku]",
                        "product_name": "product[name]",
                        "price_usd": "product[price]",
                    }.items():
                        evidence.append(
                            {
                                "field": f"/task_definition/answer/values/{index}/{field}",
                                "artifact": "observation",
                                "location": (
                                    f"/editors/{index}/controls/{indices[control_name]}/value"
                                ),
                                "modality": "text",
                                "derivation": (
                                    "Parse the exact visible Price input as a USD number."
                                    if field == "price_usd"
                                    else "Exact visible input value."
                                ),
                            }
                        )
                    for source_index, source_indices in enumerate(control_indices):
                        evidence.append(
                            {
                                "field": (
                                    f"/task_definition/answer/values/{index}/"
                                    "price_above_cheapest_usd"
                                ),
                                "artifact": "observation",
                                "location": (
                                    f"/editors/{source_index}/controls/"
                                    f"{source_indices['product[price]']}/value"
                                ),
                                "modality": "text",
                                "derivation": (
                                    "Parse both bound visible Price inputs as exact decimals, "
                                    f"take their minimum, and subtract it from editor {index}'s "
                                    "Price. Report the nonnegative difference as a USD number."
                                ),
                            }
                        )
                definition = {
                    "task_type": "RETRIEVE",
                    "goal_template": (
                        "Compare the existing simple products with exact SKUs {sku_1} and "
                        "{sku_2}. Open each product's editor and read its Product Name and "
                        "current Price field. Return one object per product in the SKU order "
                        "listed above, with fields sku, product_name, price_usd, and "
                        "price_above_cheapest_usd. The last field is that product's current "
                        "Price minus the lower of these two current Prices, so the cheaper "
                        "product has a difference of zero. Use the regular Price field, "
                        "without special prices, taxes, or currency conversion. Return USD "
                        "amounts as numbers without currency symbols. Do not save or change "
                        "any product's persisted data."
                    ),
                    "parameters": parameters,
                    "answer": {
                        "status": "SUCCESS",
                        "values": answer_values,
                        "ordered": True,
                        "item_schema": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "product_name": {"type": "string"},
                                "price_usd": {"type": "number"},
                                "price_above_cheapest_usd": {"type": "number"},
                            },
                            "required": list(answer_values[0]),
                            "additionalProperties": False,
                        },
                    },
                    "completion_events": [
                        {
                            "url_template": "^__SHOPPING_ADMIN__/catalog/product/save/.*$",
                            "http_method": "POST",
                            "should_not_exist": True,
                            "last_event_only": False,
                        }
                    ],
                }
                archived_observation = {"parameters": parameters, "editors": editors}
                family = "products_simple_price_comparison"
                source_record = "Amazon Jobs--13"
                screenshot = "product_00.png"
                final_checks = []
            elif args.mode in {"retrieve-product-attributes", "navigate-product-attributes"}:
                config = json.loads(args.task_config.read_text())
                parameters = dict(config["parameters"])
                skus = [parameters["sku_1"], parameters["sku_2"], parameters["sku_3"]]
                editors = []
                control_indices = []
                values = []
                for index, sku in enumerate(skus):
                    url = config["product_links"][sku].replace(
                        "__SHOPPING_ADMIN__", args.site_url.rstrip("/")
                    )
                    page.goto(url, wait_until="networkidle", timeout=120_000)
                    page.wait_for_timeout(1_000)
                    observation = _record(page, output, f"product_{index:02}")
                    editors.append(observation)
                    indices = {
                        control["name"]: number
                        for number, control in enumerate(observation["controls"])
                        if control.get("name", "").startswith("product[")
                    }
                    control_indices.append(indices)
                    controls = observation["controls"]
                    actual_sku = controls[indices["product[sku]"]]["value"]
                    if actual_sku != sku:
                        raise ValueError("the observed editor link identifies a different SKU")
                    material = controls[indices["product[material]"]]["options"]
                    climate = controls[indices["product[climate]"]]["options"]
                    values.append(
                        {
                            "sku": actual_sku,
                            "product_name": controls[indices["product[name]"]]["value"],
                            "material_count": sum(option["selected"] for option in material),
                            "climate_count": sum(option["selected"] for option in climate),
                            "spring_selected": next(
                                option["selected"]
                                for option in climate
                                if option["label"] == "Spring"
                            ),
                        }
                    )
                archived_observation = {"parameters": parameters, "editors": editors}
                screenshot = "product_00.png"
                if args.mode == "retrieve-product-attributes":
                    definition = {
                        "task_type": "RETRIEVE",
                        "goal_template": (
                            "Audit the existing product attributes for SKUs {sku_1}, {sku_2}, and "
                            "{sku_3}. For each, report its SKU and Product Name, the number of "
                            "currently selected Material options, the number of currently selected "
                            "Climate options, and whether Spring is selected in Climate. "
                            "Count only "
                            "selected options, not every option offered by the control. Return one "
                            "object per product in the SKU order listed above, using fields sku, "
                            "product_name, material_count, climate_count, and spring_selected. "
                            "Do not save or change any product's persisted data."
                        ),
                        "parameters": parameters,
                        "answer": {
                            "status": "SUCCESS",
                            "values": values,
                            "ordered": True,
                            "item_schema": {
                                "type": "object",
                                "properties": {
                                    "sku": {"type": "string"},
                                    "product_name": {"type": "string"},
                                    "material_count": {"type": "number"},
                                    "climate_count": {"type": "number"},
                                    "spring_selected": {"type": "boolean"},
                                },
                                "required": list(values[0]),
                                "additionalProperties": False,
                            },
                        },
                        "completion_events": [],
                    }
                    evidence = []
                    for index, indices in enumerate(control_indices):
                        for field, control_name in {
                            "sku": "product[sku]",
                            "product_name": "product[name]",
                            "material_count": "product[material]",
                            "climate_count": "product[climate]",
                            "spring_selected": "product[climate]",
                        }.items():
                            source = "value" if field in {"sku", "product_name"} else "options"
                            evidence.append(
                                {
                                    "field": f"/task_definition/answer/values/{index}/{field}",
                                    "artifact": "observation",
                                    "modality": "text",
                                    "location": (
                                        f"/editors/{index}/controls/{indices[control_name]}/{source}"
                                    ),
                                    "derivation": (
                                        "Exact visible input value."
                                        if source == "value"
                                        else "Count selected options."
                                        if field.endswith("_count")
                                        else "Read the selected flag of the Spring option."
                                    ),
                                }
                            )
                    family = "products_attribute_comparison"
                    source_record = "Amazon Jobs--13"
                    final_checks = []
                else:
                    matches = []
                    for index, (observation, indices) in enumerate(zip(editors, control_indices)):
                        controls = observation["controls"]
                        climate = {
                            option["label"]
                            for option in controls[indices["product[climate]"]]["options"]
                            if option["selected"]
                        }
                        style = {
                            option["label"]
                            for option in controls[indices["product[style_general]"]]["options"]
                            if option["selected"]
                        }
                        if {
                            parameters["climate_1"],
                            parameters["climate_2"],
                        } <= climate and parameters["style"] in style:
                            matches.append(index)
                    if len(matches) != 1:
                        raise ValueError(
                            "the observed attribute conditions do not identify one candidate"
                        )
                    selected_index = matches[0]
                    selected = editors[selected_index]
                    indices = control_indices[selected_index]
                    parameters["product_name"] = values[selected_index]["product_name"]
                    parameters["product_id"] = (
                        urlsplit(selected["url"]).path.split("/id/")[1].split("/")[0]
                    )
                    page.goto(selected["url"], wait_until="networkidle", timeout=120_000)
                    _record(page, output, "terminal")
                    definition = {
                        "task_type": "NAVIGATE",
                        "goal_template": (
                            "Among the products with SKUs {sku_1}, {sku_2}, and {sku_3}, find the "
                            "one whose currently selected Climate options include both {climate_1} "
                            "and {climate_2}, and whose selected Style General options include "
                            "{style}. Open that product's edit page and leave it open. Do not save "
                            "or change any product's persisted data."
                        ),
                        "parameters": parameters,
                        "answer": {
                            "status": "SUCCESS",
                            "values": None,
                            "item_schema": None,
                            "ordered": False,
                        },
                        "completion_events": [
                            {
                                "url_template": (
                                    "^__SHOPPING_ADMIN__/catalog/product/edit/id/{product_id}(?:/.*)?$"
                                ),
                                "http_method": "GET",
                                "response_status": 200,
                                "last_event_only": False,
                            }
                        ],
                    }
                    evidence = [
                        {
                            "field": "/task_definition/parameters/product_name",
                            "artifact": "observation",
                            "location": (
                                f"/editors/{selected_index}/controls/"
                                f"{indices['product[name]']}/value"
                            ),
                            "modality": "text",
                        },
                        {
                            "field": "/task_definition/parameters/product_id",
                            "artifact": "observation",
                            "location": f"/editors/{selected_index}/url",
                            "modality": "text",
                            "derivation": "Product ID segment of the actual observed editor URL.",
                        },
                    ]
                    selected_parameter = f"sku_{selected_index + 1}"
                    final_checks = [
                        {
                            "kind": "input_value",
                            "parameter": selected_parameter,
                            "selector": "input[name='product[sku]']",
                        },
                        {
                            "kind": "input_value",
                            "parameter": "product_name",
                            "selector": "input[name='product[name]']",
                        },
                    ]
                    family = "products_attribute_selected_navigation"
                    source_record = "Amazon Jobs--5"
                    screenshot = "terminal.png"
                    archived_observation["matching_candidate_indices"] = matches
                definition["completion_events"].append(
                    {
                        "url_template": "^__SHOPPING_ADMIN__/catalog/product/save(?:/.*)?$",
                        "http_method": "POST",
                        "should_not_exist": True,
                        "last_event_only": False,
                    }
                )
        finally:
            context.tracing.stop(path=str(output / "trace.zip"))
            context.close()
            browser.close()
    if definition is not None:
        if args.mode in {
            "mutate-product-name-price",
            "mutate-product-attributes",
            "mutate-selected-product-attributes",
            "mutate-selected-product-price-options",
            "mutate-orders-topk-derived-product-price",
        }:
            from webarena_verified.core.utils.network_event_utils import load_har_trace

            actual_events = load_har_trace(output / "network.har")
            fields = ["product[name]", "product[sku]"]
            if args.mode in {
                "mutate-product-name-price",
                "mutate-selected-product-price-options",
                "mutate-orders-topk-derived-product-price",
            }:
                fields += ["product[weight]", "product[price]"]
            if args.mode != "mutate-product-name-price":
                fields += [
                    f"product[{attribute}][{position}]"
                    for attribute, selected in selected_values.items()
                    for position in range(len(selected))
                ]
            for index, receipt in enumerate(save_receipts):
                actual = next(
                    event
                    for event in actual_events
                    if event.url == receipt["url"]
                    and event.http_method == "POST"
                    and event.request_status == receipt["response_status"]
                )
                if not actual.is_request_success:
                    raise ValueError("the observed product Save request did not succeed")
                body = actual.post_data
                expected_body = {field: body[field] for field in fields}
                if args.mode == "mutate-orders-topk-derived-product-price":
                    expected = expected_fields[selected_index]
                    if Decimal(str(expected_body["product[price]"])) != derived_price or any(
                        expected_body[f"product[{field}]"] != expected[field]
                        for field in ["name", "sku", "weight"]
                    ):
                        raise ValueError("actual Save body differs from the order-derived update")
                    for attribute, selected in selected_values.items():
                        if any(
                            expected_body[f"product[{attribute}][{position}]"] != value
                            for position, value in enumerate(selected)
                        ):
                            raise ValueError("actual Save body lost a preserved attribute member")
                receipt["post_data"] = expected_body
                if args.mode == "mutate-orders-topk-derived-product-price":
                    evidence.extend(
                        {
                            **binding,
                            "field": (
                                f"/task_definition/completion_events/{index}/"
                                "post_data/product[price]"
                            ),
                        }
                        for binding in order_evidence
                    )
                    for field, source in [
                        ("url_template", "url"),
                        ("response_status", "response_status"),
                    ]:
                        evidence.append(
                            {
                                "field": f"/task_definition/completion_events/{index}/{field}",
                                "artifact": "observation",
                                "location": f"/save_requests/{index}/{source}",
                                "modality": "text",
                                "derivation": (
                                    "Actual selected-product Save route and successful HTTP "
                                    "status from the same recorded POST."
                                ),
                            }
                        )
                path = (
                    urlsplit(actual.url)
                    .path.removeprefix(urlsplit(args.site_url).path.rstrip("/"))
                    .rstrip("/")
                )
                route = path[: path.index("/save") + len("/save")]
                pattern = re.escape(route)
                body_properties = {}
                for field, value in expected_body.items():
                    body_properties[field] = (
                        {"type": "number"} if field == "product[price]" else {"type": "string"}
                    )
                    evidence.append(
                        {
                            "field": (
                                f"/task_definition/completion_events/{index}/post_data/{field}"
                            ),
                            "artifact": "observation",
                            "location": f"/save_requests/{index}/post_data/{field}",
                            "modality": "text",
                            "derivation": "Actual Save POST parsed by the canonical HAR owner.",
                        }
                    )
                definition["completion_events"].append(
                    {
                        "url_template": (
                            "^__SHOPPING_ADMIN__"
                            + pattern
                            + "/id/"
                            + re.escape(path.split("/")[path.split("/").index("id") + 1])
                            + "(?:/.*)?$"
                            if args.mode
                            in {
                                "mutate-selected-product-price-options",
                                "mutate-orders-topk-derived-product-price",
                            }
                            else "^__SHOPPING_ADMIN__" + pattern + "(?:/.*)?$"
                        ),
                        "http_method": "POST",
                        "response_status": actual.request_status,
                        "post_data": expected_body,
                        "post_data_schema": {"type": "object", "properties": body_properties},
                        "last_event_only": False,
                    }
                )
            if args.mode == "mutate-product-name-price":
                more_expensive = 1 - cheaper
                other_editor = urlsplit(editors[more_expensive]["url"]).path.split("/")
                other_product_id = other_editor[other_editor.index("id") + 1]
                forbidden_index = len(definition["completion_events"])
                # should_not_exist filters by route/method, not post_data. Bind
                # the other observed product ID so the legal Save stays allowed.
                definition["completion_events"].append(
                    {
                        "url_template": (
                            "^__SHOPPING_ADMIN__"
                            + pattern
                            + "/id/"
                            + re.escape(other_product_id)
                            + "(?:/.*)?$"
                        ),
                        "http_method": "POST",
                        "post_data": {"product[sku]": initial_fields[more_expensive]["sku"]},
                        "post_data_schema": {
                            "type": "object",
                            "properties": {"product[sku]": {"type": "string"}},
                        },
                        "should_not_exist": True,
                        "last_event_only": False,
                    }
                )
                evidence.extend(
                    [
                        {
                            "field": (
                                f"/task_definition/completion_events/{forbidden_index}/url_template"
                            ),
                            "artifact": "observation",
                            "location": "/save_requests/0/url",
                            "modality": "text",
                            "derivation": (
                                "Use the actual Save route through /save, with its observed "
                                "id path argument bound to the more expensive initial editor. "
                                "The suffix stays unconstrained to allow ordinary Save variants."
                            ),
                        },
                        {
                            "field": (
                                f"/task_definition/completion_events/{forbidden_index}/url_template"
                            ),
                            "artifact": "observation",
                            "location": f"/before/{more_expensive}/url",
                            "modality": "text",
                            "derivation": (
                                "Read the other product's id argument from its formal initial "
                                "editor URL. Both formal initial Prices determine that it is "
                                "the more expensive product; no Save is performed on it."
                            ),
                        },
                        {
                            "field": (
                                f"/task_definition/completion_events/{forbidden_index}/"
                                "post_data/product[sku]"
                            ),
                            "artifact": "observation",
                            "location": (
                                f"/before/{more_expensive}/controls/"
                                f"{indices_by_editor[more_expensive]['product[sku]']}/value"
                            ),
                            "modality": "text",
                            "derivation": "Exact SKU of the more expensive formal initial editor.",
                        },
                    ]
                )
                for source_index, indices in enumerate(indices_by_editor):
                    evidence.append(
                        {
                            "field": (
                                f"/task_definition/completion_events/{forbidden_index}/url_template"
                            ),
                            "artifact": "observation",
                            "location": (
                                f"/before/{source_index}/controls/{indices['product[price]']}/value"
                            ),
                            "modality": "text",
                            "derivation": (
                                "Compare both formal initial regular Prices as decimals; "
                                "the larger Price identifies the editor that must not be saved."
                            ),
                        }
                    )
            elif args.mode in {
                "mutate-selected-product-attributes",
                "mutate-selected-product-price-options",
                "mutate-orders-topk-derived-product-price",
            }:
                for other_index, captured in enumerate(editors):
                    if other_index == selected_index:
                        continue
                    other_path = urlsplit(captured["url"]).path.split("/")
                    other_id = other_path[other_path.index("id") + 1]
                    forbidden_index = len(definition["completion_events"])
                    definition["completion_events"].append(
                        {
                            "url_template": (
                                "^__SHOPPING_ADMIN__"
                                + pattern
                                + "/id/"
                                + re.escape(other_id)
                                + "(?:/.*)?$"
                            ),
                            "http_method": "POST",
                            "should_not_exist": True,
                            "last_event_only": False,
                        }
                    )
                    evidence.append(
                        {
                            "field": (
                                f"/task_definition/completion_events/{forbidden_index}/url_template"
                            ),
                            "artifact": "observation",
                            "location": f"/before/{other_index}/url",
                            "modality": "text",
                            "derivation": (
                                "Bind the actual Save route to this unselected candidate's "
                                "observed editor ID; only the uniquely selected candidate "
                                "may be saved."
                            ),
                        }
                    )
        (output / "observation.json").write_text(
            json.dumps(archived_observation, ensure_ascii=False, indent=2) + "\n"
        )
        relative = output.relative_to(args.batch_root)
        draft = {
            "generated_task_id": args.generated_task_id,
            "task_id": config["task_id"],
            "intent_template_id": config["intent_template_id"],
            "site": "shopping_admin",
            "start_url": "__SHOPPING_ADMIN__",
            "revision": revision,
            "source_artifact": source_artifact,
            "source_record_id": source_record,
            "baseline": baseline,
            "task_definition": definition,
            "grounding": {
                "captured_at": datetime.now(UTC).isoformat(),
                "family": family,
                "artifacts": {
                    "har": str(relative / "network.har"),
                    "trace": str(relative / "trace.zip"),
                    "screenshot": str(relative / screenshot),
                    "observation": str(relative / "observation.json"),
                },
                "field_evidence": evidence,
                "final_state_checks": final_checks,
            },
        }
        (output / "draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "observed", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
