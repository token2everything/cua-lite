#!/usr/bin/env python3
"""Ground read-only shopping tasks on a coordinator-owned fixed image baseline.

The coordinator supplies the baseline receipt and authenticated storage state.
Each selected config opens a fresh browser, records actual UI facts, and emits
the existing author draft contract. This script does not prepare assets or call
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
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-url", default="http://localhost:7770")
    parser.add_argument("--task-configs", type=Path, required=True)
    parser.add_argument("--generated-task-id", required=True)
    parser.add_argument("--attempt", default="grounding-01")
    args = parser.parse_args()
    if not re.fullmatch(r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", args.generated_task_id):
        parser.error("generated task ID must use the generated batch namespace")
    if not args.attempt.replace("-", "").isalnum():
        parser.error("attempt must be a simple identifier")
    return args


def _record(page: Any, output: Path, name: str) -> dict[str, Any]:
    """Record rendered product cards, cart cells, controls, and product details."""
    observation = page.evaluate(
        """() => {
          const visible = e => {
            const r = e.getBoundingClientRect();
            return !!(r.width && r.height);
          };
          const text = e => e.innerText.trim();
          const product = document.querySelector('.product-info-main');
          const cards = Array.from(document.querySelectorAll('main .product-item-info'))
            .filter(visible).map(e => {
              const link = e.querySelector('.product-item-link');
              const rating = e.querySelector('.rating-result');
              const reviews = e.querySelector('.reviews-actions .action.view');
              return {
                name: text(link), url: link.href,
                price: text(e.querySelector('[data-price-type=finalPrice] .price')),
                rating: rating ? rating.getAttribute('title') : null,
                reviews: reviews ? text(reviews) : null
              };
            });
          return {
            url: location.href, title: document.title, body_text: document.body.innerText,
            products: cards,
            amounts: Array.from(document.querySelectorAll('.toolbar-amount'))
              .filter(visible).map(text),
            controls: Array.from(document.querySelectorAll('input, select, textarea, button'))
              .filter(visible).map(e => ({
                tag: e.tagName.toLowerCase(), type: e.type, name: e.name, id: e.id,
                text: text(e), title: e.title, value: e.value, disabled: e.disabled,
                aria_label: e.getAttribute('aria-label'),
                options: e.tagName === 'SELECT' ? Array.from(e.options).map(o => ({
                  label: o.textContent.trim(), value: o.value, selected: o.selected
                })) : undefined
              })),
            links: Array.from(document.querySelectorAll('a[href]')).filter(visible)
              .map(a => ({text: text(a), url: a.href, class_name: a.className})),
            tables: Array.from(document.querySelectorAll('table')).filter(visible).map(t => ({
              id: t.id, headers: Array.from(t.querySelectorAll('thead th')).map(text),
              rows: Array.from(t.querySelectorAll('tbody tr')).filter(visible)
                .map(tr => Array.from(tr.querySelectorAll('th, td')).map(text))
            })),
            cart_items: Array.from(document.querySelectorAll(
              '#shopping-cart-table tbody.cart.item tr.item-info')).filter(visible).map(tr => ({
                name: text(tr.querySelector('.product-item-name')),
                url: tr.querySelector('.product-item-name a').href,
                unit_price: text(tr.querySelector('.col.price .price')),
                quantity: tr.querySelector('input[data-role=cart-item-qty]').value,
                line_subtotal: text(tr.querySelector('.col.subtotal .price')),
                cells: Array.from(tr.querySelectorAll('td')).map(text)
            })),
            product_detail: product && visible(product) ? {
              name: text(document.querySelector('main .page-title-wrapper.product h1.page-title')),
              sku: text(product.querySelector('.sku .value')),
              price: text(product.querySelector('[data-price-type=finalPrice] .price')),
              body_text: text(product)
            } : null
          };
        }"""
    )
    page.screenshot(path=str(output / f"{name}.png"))
    (output / f"{name}.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
    )
    return observation


def _usd(value: str) -> Decimal:
    return Decimal(value.removeprefix("$").replace(",", ""))


def _collect_showcase(page: Any, output: Path, expected_count: int) -> list[dict[str, Any]]:
    """Cover the observed showcase pagination and prove distinct-product completeness."""
    pages = []
    seen = set()
    while True:
        page.locator("main .product-item-info:visible").first.wait_for(timeout=60_000)
        observation = _record(page, output, f"page_{len(pages):03}")
        pages.append(observation)
        for product in observation["products"]:
            if product["url"] in seen:
                raise ValueError("showcase pagination repeated a product")
            seen.add(product["url"])
        if len(pages) > 3:
            raise ValueError("showcase exceeded its observed bounded page scope")
        next_page = page.locator(".pages-item-next a:visible")
        if next_page.count() == 0:
            break
        next_page.first.click()
        page.wait_for_load_state("networkidle", timeout=60_000)
    if len(seen) != expected_count:
        raise ValueError("formal showcase coverage differs from the observed baseline total")
    return pages


def main() -> None:
    args = _parse_args()
    config = next(
        row
        for row in json.loads(args.task_configs.read_text())["tasks"]
        if row["generated_task_id"] == args.generated_task_id
    )
    baseline = json.loads(args.baseline_receipt.read_text())
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("authoring requires a coordinator-owned fixed image baseline")
    if baseline["site_url"].rstrip("/") != args.site_url.rstrip("/"):
        raise ValueError("author endpoint differs from the coordinator baseline receipt")
    output = args.batch_root / "author" / args.generated_task_id / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    parameters = dict(config["parameters"])
    evidence = []
    final_checks = []
    events = []
    task_type = "RETRIEVE"

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
            initial = _record(page, output, "initial")
            if config["family"] in {"showcase_filtered_rating_rank", "showcase_review_aggregate"}:
                pages = _collect_showcase(page, output, config["expected_product_count"])
                observation = {"initial": initial, "pages": pages}
                screenshot = "page_000.png"
                records = [
                    (product, page_index, index)
                    for page_index, captured in enumerate(pages)
                    for index, product in enumerate(captured["products"])
                ]
                if config["family"] == "showcase_filtered_rating_rank":
                    qualifying = [
                        record
                        for record in records
                        if record[0]["reviews"] is not None
                        and int(re.search(r"\d+", record[0]["reviews"])[0])
                        >= parameters["minimum_reviews"]
                        and _usd(record[0]["price"])
                        <= Decimal(str(parameters["maximum_price_usd"]))
                    ]
                    qualifying.sort(
                        key=lambda record: (
                            -Decimal(record[0]["rating"].removesuffix("%")),
                            _usd(record[0]["price"]),
                            record[0]["name"],
                        )
                    )
                    selected = qualifying[: parameters["limit"]]
                    if len(selected) != parameters["limit"]:
                        raise ValueError("formal showcase observations lack the requested top-k")
                    values = [
                        {
                            "product_name": product["name"],
                            "price_usd": float(_usd(product["price"])),
                            "rating_percent": float(Decimal(product["rating"].removesuffix("%"))),
                            "review_count": int(re.search(r"\d+", product["reviews"])[0]),
                        }
                        for product, _, _ in selected
                    ]
                    for index, (_, page_index, source_index) in enumerate(selected):
                        for field, source in {
                            "product_name": "name",
                            "price_usd": "price",
                            "rating_percent": "rating",
                            "review_count": "reviews",
                        }.items():
                            evidence.append(
                                {
                                    "field": f"/task_definition/answer/values/{index}/{field}",
                                    "artifact": "observation",
                                    "location": (
                                        f"/pages/{page_index}/products/{source_index}/{source}"
                                    ),
                                    "modality": "text",
                                    "derivation": (
                                        "Exact displayed name."
                                        if source == "name"
                                        else "Parse the displayed USD, percent, or review count."
                                    ),
                                }
                            )
                    goal = (
                        "Across every page of the homepage Product Showcases, consider products "
                        "priced at no more than {maximum_price_usd} USD with at least "
                        "{minimum_reviews} displayed reviews. Return the {limit} qualifying "
                        "products with the highest displayed rating percentage, sorting ratings "
                        "descending, then prices ascending, then exact product names "
                        "alphabetically ascending. For each report product_name, price_usd, "
                        "rating_percent, and review_count. Express the rating as the displayed "
                        "percentage number, without converting it to stars. Do not change the "
                        "cart, wishlist, or comparison list."
                    )
                    schema = {
                        "type": "object",
                        "properties": {
                            "product_name": {"type": "string"},
                            "price_usd": {"type": "number"},
                            "rating_percent": {"type": "number"},
                            "review_count": {"type": "integer"},
                        },
                        "required": list(values[0]),
                        "additionalProperties": False,
                    }
                else:
                    grouped = [
                        [record for record in records if record[0]["reviews"] is not None],
                        [record for record in records if record[0]["reviews"] is None],
                    ]
                    values = []
                    for index, group in enumerate(grouped):
                        values.append(
                            {
                                "has_displayed_reviews": index == 0,
                                "product_count": len(group),
                                "price_sum_usd": float(
                                    sum(
                                        (_usd(product["price"]) for product, _, _ in group),
                                        Decimal("0.00"),
                                    )
                                ),
                            }
                        )
                        for product, page_index, source_index in group:
                            for field, source, derivation in [
                                (
                                    "has_displayed_reviews",
                                    "reviews",
                                    "Classify by whether a review-count link is displayed.",
                                ),
                                (
                                    "product_count",
                                    "url",
                                    "Count distinct bound product URLs across every page.",
                                ),
                                (
                                    "price_sum_usd",
                                    "price",
                                    "Sum this class's bound USD prices as exact decimals.",
                                ),
                            ]:
                                evidence.append(
                                    {
                                        "field": f"/task_definition/answer/values/{index}/{field}",
                                        "artifact": "observation",
                                        "location": (
                                            f"/pages/{page_index}/products/{source_index}/{source}"
                                        ),
                                        "modality": "text",
                                        "derivation": derivation,
                                    }
                                )
                    if any(not group for group in grouped):
                        raise ValueError("both requested review-display classes must exist")
                    goal = (
                        "Audit all products across every page of the homepage Product "
                        "Showcases. Split them into products that display a review-count link "
                        "and products that display no review-count link. For each class report "
                        "has_displayed_reviews as a boolean, product_count as the number of "
                        "distinct products, and price_sum_usd as the sum of one displayed "
                        "product price per product in USD. Return the class with displayed "
                        "reviews first and the class without a displayed review link second. "
                        "This classification concerns the displayed link, without inferring "
                        "reviews from missing information. Do not change the cart, wishlist, "
                        "or comparison list."
                    )
                    schema = {
                        "type": "object",
                        "properties": {
                            "has_displayed_reviews": {"type": "boolean"},
                            "product_count": {"type": "integer"},
                            "price_sum_usd": {"type": "number"},
                        },
                        "required": list(values[0]),
                        "additionalProperties": False,
                    }
            elif config["family"] in {"cart_cost_share", "cart_constrained_product_navigation"}:
                page.goto(
                    args.site_url.rstrip("/") + "/checkout/cart/",
                    wait_until="networkidle",
                    timeout=120_000,
                )
                page.locator("#shopping-cart-table .item-info:visible").first.wait_for()
                cart = _record(page, output, "cart")
                observation = {"initial": initial, "cart": cart}
                items = cart["cart_items"]
                if len(items) != config["expected_cart_items"]:
                    raise ValueError("formal cart differs from the fixed-baseline item scope")
                if config["family"] == "cart_cost_share":
                    amounts = [
                        _usd(item["unit_price"]) * Decimal(item["quantity"]) for item in items
                    ]
                    if any(
                        amount != _usd(item["line_subtotal"])
                        for amount, item in zip(amounts, items)
                    ):
                        raise ValueError(
                            "cart quantity-price arithmetic differs from displayed subtotals"
                        )
                    total = sum(amounts, Decimal("0.00"))
                    ranked = sorted(
                        range(len(items)), key=lambda i: (-amounts[i], items[i]["name"])
                    )
                    values = [
                        {
                            "product_name": items[i]["name"],
                            "quantity": float(Decimal(items[i]["quantity"])),
                            "unit_price_usd": float(_usd(items[i]["unit_price"])),
                            "line_subtotal_usd": float(amounts[i]),
                            "subtotal_share_percent": float(
                                (amounts[i] / total * 100).quantize(
                                    Decimal("0.01"), rounding=ROUND_HALF_UP
                                )
                            ),
                        }
                        for i in ranked
                    ]
                    for index, source_index in enumerate(ranked):
                        for field, source in {
                            "product_name": "name",
                            "quantity": "quantity",
                            "unit_price_usd": "unit_price",
                            "line_subtotal_usd": "line_subtotal",
                        }.items():
                            evidence.append(
                                {
                                    "field": f"/task_definition/answer/values/{index}/{field}",
                                    "artifact": "observation",
                                    "location": f"/cart/cart_items/{source_index}/{source}",
                                    "modality": "text",
                                    "derivation": "Exact display; parse numeric fields as numbers.",
                                }
                            )
                        for item_index in range(len(items)):
                            evidence.append(
                                {
                                    "field": (
                                        f"/task_definition/answer/values/{index}/"
                                        "subtotal_share_percent"
                                    ),
                                    "artifact": "observation",
                                    "location": f"/cart/cart_items/{item_index}/line_subtotal",
                                    "modality": "text",
                                    "derivation": (
                                        f"Divide cart item {source_index}'s line subtotal by the "
                                        "sum of all bound line subtotals, multiply by 100, and "
                                        "round to two decimal places with half-up rounding."
                                    ),
                                }
                            )
                    screenshot = "cart.png"
                    goal = (
                        "Audit the current shopping cart without changing it. For every cart "
                        "line report product_name, quantity, unit_price_usd, line_subtotal_usd, "
                        "and subtotal_share_percent. Calculate each line subtotal as unit price "
                        "times its current quantity and confirm it matches the displayed line "
                        "subtotal. Calculate its percentage of the sum of all line subtotals, "
                        "rounding to two decimal places with half-up rounding. Sort by line "
                        "subtotal descending, then exact product name alphabetically ascending. "
                        "Use the cart's current prices and exclude shipping and tax. Do not "
                        "update quantities, remove items, or move any item to a wishlist."
                    )
                    schema = {
                        "type": "object",
                        "properties": {
                            "product_name": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price_usd": {"type": "number"},
                            "line_subtotal_usd": {"type": "number"},
                            "subtotal_share_percent": {"type": "number"},
                        },
                        "required": list(values[0]),
                        "additionalProperties": False,
                    }
                else:
                    candidates = []
                    for index, item in enumerate(items):
                        page.goto(item["url"], wait_until="networkidle", timeout=120_000)
                        captured = _record(page, output, f"product_{index:02}")
                        candidates.append(captured)
                    matches = [
                        index
                        for index, captured in enumerate(candidates)
                        if all(
                            word.casefold() in captured["product_detail"]["name"].casefold()
                            for word in [parameters["title_word_1"], parameters["title_word_2"]]
                        )
                        and _usd(captured["product_detail"]["price"])
                        > Decimal(str(parameters["minimum_price_usd"]))
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "formal cart-product constraints do not identify one target"
                        )
                    target_index = matches[0]
                    target = candidates[target_index]
                    parameters["product_name"] = target["product_detail"]["name"]
                    parameters["sku"] = target["product_detail"]["sku"]
                    parameters["product_path"] = re.escape(urlsplit(target["url"]).path.lstrip("/"))
                    observation["candidates"] = candidates
                    observation["matching_candidate_indices"] = matches
                    page.goto(target["url"], wait_until="networkidle", timeout=120_000)
                    observation["terminal"] = _record(page, output, "terminal")
                    evidence = [
                        {
                            "field": f"/task_definition/parameters/{field}",
                            "artifact": "observation",
                            "location": f"/candidates/{target_index}/product_detail/{source}",
                            "modality": "text",
                            "derivation": "Exact displayed product identity.",
                        }
                        for field, source in [("product_name", "name"), ("sku", "sku")]
                    ]
                    evidence.append(
                        {
                            "field": "/task_definition/parameters/product_path",
                            "artifact": "observation",
                            "location": f"/candidates/{target_index}/url",
                            "modality": "text",
                            "derivation": "Regex-escape the path of the visited product URL.",
                        }
                    )
                    final_checks = [
                        {
                            "kind": "visible_text_exact",
                            "parameter": "product_name",
                            "selector": "h1.page-title",
                        },
                        {
                            "kind": "visible_text_exact",
                            "parameter": "sku",
                            "selector": ".product-info-main .sku .value",
                        },
                    ]
                    events.append(
                        {
                            "url_template": "^__SHOPPING__/{product_path}(?:\\?.*)?$",
                            "http_method": "GET",
                            "response_status": 200,
                            "last_event_only": False,
                        }
                    )
                    task_type = "NAVIGATE"
                    values = None
                    schema = None
                    screenshot = "terminal.png"
                    goal = (
                        "Among the products currently present in the shopping cart, find the "
                        "one whose product-page title contains both {title_word_1} and "
                        "{title_word_2}, case-insensitively, and whose current product-page "
                        "price is greater than {minimum_price_usd} USD. Open that product's "
                        "detail page and leave it open. Verify its identity and price on the "
                        "product page. Keep every cart quantity and item unchanged, and do not "
                        "add it to a wishlist or comparison list."
                    )
            elif config["family"] == "advanced_search_price_rank":
                page.goto(
                    args.site_url.rstrip("/") + "/catalogsearch/advanced/",
                    wait_until="networkidle",
                    timeout=120_000,
                )
                before = _record(page, output, "advanced")
                form = page.locator("form#form-validate")
                form.locator("input[name='name']").fill(parameters["name_term"])
                form.locator("input[name='description']").fill(parameters["description_term"])
                form.locator("input[name='price[from]']").fill(str(parameters["minimum_price_usd"]))
                form.locator("input[name='price[to]']").fill(str(parameters["maximum_price_usd"]))
                form.locator("button[type=submit]:visible").click()
                page.wait_for_load_state("networkidle", timeout=60_000)
                page.locator("select[data-role=sorter]:visible").first.select_option("price")
                page.wait_for_load_state("networkidle", timeout=60_000)
                direction = page.locator("a[data-role=direction-switcher]:visible").first
                if direction.get_attribute("title") == "Set Ascending Direction":
                    direction.click()
                    page.wait_for_load_state("networkidle", timeout=60_000)
                pages = []
                selected = []
                seen = set()
                while len(selected) < parameters["limit"]:
                    captured = _record(page, output, f"page_{len(pages):03}")
                    pages.append(captured)
                    for index, product in enumerate(captured["products"]):
                        if product["url"] in seen:
                            raise ValueError("advanced search pagination repeated a product")
                        seen.add(product["url"])
                        price = _usd(product["price"])
                        if (
                            not Decimal(str(parameters["minimum_price_usd"]))
                            <= price
                            <= Decimal(str(parameters["maximum_price_usd"]))
                        ):
                            raise ValueError(
                                "formal search result violates the requested price range"
                            )
                        selected.append((product, len(pages) - 1, index))
                        if len(selected) == parameters["limit"]:
                            break
                    if len(selected) == parameters["limit"]:
                        break
                    if len(pages) == config["maximum_author_pages"]:
                        raise ValueError(
                            "requested result count exceeds the bounded author page scope"
                        )
                    next_page = page.locator(".pages-item-next a:visible")
                    if not next_page.count():
                        raise ValueError("formal search has fewer than the requested results")
                    next_page.first.click()
                    page.wait_for_load_state("networkidle", timeout=60_000)
                values = [
                    {"product_name": product["name"], "price_usd": float(_usd(product["price"]))}
                    for product, _, _ in selected
                ]
                if any(a["price_usd"] > b["price_usd"] for a, b in zip(values, values[1:])):
                    raise ValueError("formal result prices are not sorted ascending")
                for index, (_, page_index, source_index) in enumerate(selected):
                    for field, source in [("product_name", "name"), ("price_usd", "price")]:
                        evidence.append(
                            {
                                "field": f"/task_definition/answer/values/{index}/{field}",
                                "artifact": "observation",
                                "location": f"/pages/{page_index}/products/{source_index}/{source}",
                                "modality": "text",
                                "derivation": "Exact name, or parse the displayed USD price.",
                            }
                        )
                goal = (
                    "Use Catalog Advanced Search with Product Name set to {name_term}, "
                    "Description set to {description_term}, and Price from "
                    "{minimum_price_usd} through {maximum_price_usd} USD. Sort the results "
                    "by Price ascending and report the first {limit} products in the site's "
                    "displayed order, including product_name and price_usd for each. For "
                    "equal prices, preserve the site's displayed order. Return prices as "
                    "numbers without currency symbols. Do not change the cart, wishlist, "
                    "or comparison list."
                )
                schema = {
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string"},
                        "price_usd": {"type": "number"},
                    },
                    "required": ["product_name", "price_usd"],
                    "additionalProperties": False,
                }
                observation = {"initial": initial, "advanced": before, "pages": pages}
                screenshot = "page_000.png"
            else:
                raise ValueError(f"unsupported shopping task family: {config['family']}")
        finally:
            context.tracing.stop(path=str(output / "trace.zip"))
            context.close()
            browser.close()
    events.append(
        {
            "url_template": (
                "^__SHOPPING__/(?:checkout/cart/(?:add|updatePost|delete)|wishlist/index/add|"
                "catalog/product_compare/add|sales/order/reorder)/.*$"
            ),
            "http_method": "POST",
            "should_not_exist": True,
            "last_event_only": False,
        }
    )
    definition = {
        "task_type": task_type,
        "goal_template": goal,
        "parameters": parameters,
        "answer": {
            "status": "SUCCESS",
            "values": values,
            "item_schema": schema,
            "ordered": task_type == "RETRIEVE",
        },
        "completion_events": events,
    }
    (output / "observation.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
    )
    relative = output.relative_to(args.batch_root)
    draft = {
        "generated_task_id": args.generated_task_id,
        "task_id": config["task_id"],
        "intent_template_id": config["intent_template_id"],
        "site": "shopping",
        "start_url": "__SHOPPING__",
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
                "screenshot": str(relative / screenshot),
                "observation": str(relative / "observation.json"),
            },
            "field_evidence": evidence,
            "final_state_checks": final_checks,
        },
    }
    (output / "draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "grounded", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
