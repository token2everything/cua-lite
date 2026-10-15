#!/usr/bin/env python3
"""Ground a Shopping selection and one GitLab issue note on a fresh joint baseline.

Config supplies observed source navigation, selection rules and an existing issue.
Facts and completion events come from this execution's visible pages and HAR.
The coordinator compiles the emitted draft and owns all site resources.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from string import Formatter
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

from explore_cross_site import record_observation
from explore_single_site import ACTION_PATH


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-urls", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--generated-task-id", required=True)
    parser.add_argument("--attempt", default="grounding-01")
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--settle-ms", type=int, default=2000)
    args = parser.parse_args()
    if not re.fullmatch(r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", args.generated_task_id):
        parser.error("generated task ID must use the generated namespace")
    if not re.fullmatch(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", args.attempt):
        parser.error("attempt must be a simple identifier")
    return args


def select_entity(entities: list[dict], policy: dict) -> dict:
    """Select one visible entity with explicit filtering, rank and deterministic ties.

    A missing queried fact excludes that entity. Shares require every population
    amount, including rows excluded by the selection predicate.
    Original numeric facts retain their displayed strings.
    """
    from cross_site_evaluator import derive_value

    if policy["kind"] != "rank":
        raise ValueError("select_entity owns only explicit rank selection")
    if not entities or len({row["url"] for row in entities}) != len(entities):
        raise ValueError("Selection requires a nonempty, distinct visible entity population")
    rank_by = policy["rank_by"]
    fact_field = policy.get("fact_field", rank_by)
    numeric = {"price", "subtotal", "total", "quantity", "rating", "reviews", "subtotal_share"}
    product_share = fact_field in {"price_share", "eligible_price_share"}
    if rank_by not in numeric or fact_field not in numeric | {
        "status", "price_share", "eligible_price_share"
    }:
        raise ValueError("Unsupported selection rank or fact field")
    if product_share and rank_by != "price":
        raise ValueError("Product price shares require price ranking")
    if policy["direction"] not in {"asc", "desc"} or policy["tie_break"] != "title_ascending":
        raise ValueError("Selection needs an explicit direction and title_ascending tie rule")
    if type(policy["rank"]) is not int or policy["rank"] < 1:
        raise ValueError("Selection rank must be a positive integer")
    required = {rank_by, "price" if product_share else fact_field}
    for condition in policy.get("filters", []):
        field, op = condition["field"], condition["op"]
        if (field in numeric and op in {"lt", "lte", "eq", "gte", "gt"}) or (
            field == "status" and op == "eq"
        ):
            required.add(field)
        elif field != "title" or op != "contains":
            raise ValueError("Unsupported selection predicate")
    if "budget" in policy:
        required.add("price")
    raw_fields = required - {"subtotal_share"}
    if "subtotal_share" in required:
        raw_fields.add("subtotal")
    normalized = []
    for index, row in enumerate(entities):
        values = {}
        for field in raw_fields:
            if row.get(field) is None:
                continue
            if field == "status":
                values[field] = str(row[field])
                continue
            match = re.fullmatch(
                r"\s*\$?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:%|Reviews?|reviews?)?\s*",
                str(row[field]),
            )
            if not match:
                raise ValueError("Visible numeric field cannot be parsed: " + field)
            values[field] = Decimal(match[1].replace(",", ""))
        normalized.append({"index": index, "entity": row, "values": values})
    if product_share:
        if any("price" not in row["values"] for row in normalized):
            raise ValueError("Price share requires every visible population price")
        window_total = sum(row["values"]["price"] for row in normalized)
        if not window_total:
            raise ValueError("Price share requires a nonzero visible population total")
    if "subtotal_share" in required:
        if any("subtotal" not in row["values"] for row in normalized):
            raise ValueError("Subtotal share requires every visible population subtotal")
        total = sum(row["values"]["subtotal"] for row in normalized)
        if not total:
            raise ValueError("Subtotal share requires a nonzero visible population total")
        for row in normalized:
            row["values"]["subtotal_share"] = Decimal(
                derive_value(
                    {
                        "operation": "ratio_percent",
                        "parameters": ["subtotal", "total"],
                        "output_format": ".2f",
                    },
                    {"subtotal": row["values"]["subtotal"], "total": total},
                )
            )
    eligible = []
    for row in normalized:
        if not required <= row["values"].keys():
            continue
        matches = True
        for condition in policy.get("filters", []):
            field, op, expected = condition["field"], condition["op"], condition["value"]
            if field == "title" and op == "contains":
                match = str(expected).casefold() in row["entity"]["title"].casefold()
            elif field == "status":
                match = row["values"][field] == str(expected)
            else:
                actual, bound = row["values"][field], Decimal(str(expected))
                match = {
                    "lt": actual < bound,
                    "lte": actual <= bound,
                    "eq": actual == bound,
                    "gte": actual >= bound,
                    "gt": actual > bound,
                }[op]
            matches = matches and match
        if "budget" in policy:
            matches = matches and row["values"]["price"] <= Decimal(str(policy["budget"]))
        if matches:
            eligible.append(row)
    ordered = sorted(
        eligible,
        key=lambda row: (
            row["values"][rank_by] if policy["direction"] == "asc" else -row["values"][rank_by],
            row["entity"]["title"],
        ),
    )
    if len(ordered) < policy["rank"]:
        raise ValueError("The observed population has too few qualifying entities")
    selected = ordered[policy["rank"] - 1]
    if (
        sum(
            row["values"][rank_by] == selected["values"][rank_by]
            and row["entity"]["title"] == selected["entity"]["title"]
            for row in ordered
        )
        > 1
    ):
        raise ValueError("Public selection rules do not uniquely identify an entity")
    fact = (
        str(selected["values"][fact_field])
        if fact_field == "subtotal_share"
        else str(selected["entity"]["price" if product_share else fact_field])
    )
    result = {
        "index": selected["index"],
        "entity": selected["entity"],
        "fact": fact,
        "fact_field": fact_field,
        "eligible_indices": [row["index"] for row in ordered],
    }
    if fact_field == "subtotal_share":
        result["derivations"] = {
            "cart_total": {
                "operation": "sum",
                "parameters": [
                    "selected_subtotal"
                    if index == selected["index"]
                    else f"cart_item_{index}_subtotal"
                    for index in range(len(entities))
                ],
                "output_format": ".2f",
            },
            "selected_fact": {
                "operation": "ratio_percent",
                "parameters": ["selected_subtotal", "cart_total"],
                "output_format": ".2f",
            },
        }
    elif product_share:
        price_parameters = [
            "selected_price" if index == selected["index"] else f"product_{index}_price"
            for index in range(len(entities))
        ]
        result["derivations"] = {
            "window_total": {
                "operation": "sum", "parameters": price_parameters, "output_format": ".2f"
            }
        }
        if fact_field == "eligible_price_share":
            result["derivations"]["eligible_total"] = {
                "operation": "sum",
                "parameters": [price_parameters[row["index"]] for row in eligible],
                "output_format": ".2f",
            }
        result["derivations"]["selected_fact"] = {
            "operation": "ratio_percent",
            "parameters": [
                "eligible_total" if fact_field == "eligible_price_share" else "selected_price",
                "window_total",
            ],
            "output_format": ".2f",
        }
        numerator = (
            sum(row["values"]["price"] for row in eligible)
            if fact_field == "eligible_price_share"
            else selected["values"]["price"]
        )
        result["fact"] = derive_value(
            result["derivations"]["selected_fact"],
            {"selected_price": numerator, "eligible_total": numerator, "window_total": window_total},
        )
    return result


def ground_task(args) -> dict:
    """Read both declared sites, perform exactly one note Save and archive its proof."""
    from bs4 import BeautifulSoup
    from contract import sha256_file
    from cross_site_evaluator import derive_value, parse_fact_value, seal_recording
    from playwright.sync_api import sync_playwright
    from webarena_verified.core.utils.network_event_utils import load_har_trace

    config = json.loads(args.task_config.read_text())
    sites = json.loads(args.site_urls.read_text())
    baseline = json.loads(args.baseline_receipt.read_text())
    if config["generated_task_id"] != args.generated_task_id:
        raise ValueError("Selected task ID differs from the author config")
    if set(sites) != {"shopping", "gitlab"} or set(baseline["sites"]) != set(sites):
        raise ValueError("This author requires the declared Shopping and GitLab pair")
    if not baseline["fresh_from_image"] or not baseline["authentication_only_preparation"]:
        raise ValueError("Authoring requires a fresh authentication-only joint baseline")
    for site, url in sites.items():
        if baseline["sites"][site]["site_url"].rstrip("/") != url.rstrip("/"):
            raise ValueError("Author endpoints differ from the fresh joint baseline")
    source = config["source"]
    policy = config["selection"]
    if policy["kind"] not in {"rank", "budget_pair"}:
        raise ValueError("Selection kind must be rank or budget_pair")
    note_fields = {field for _, field, _, _ in Formatter().parse(config["note_template"]) if field}
    if note_fields != {"selected_entity_title", "selected_fact"}:
        raise ValueError("Note template must bind exactly the selected title and source fact")
    if source["kind"] not in {"products", "cart_items", "orders"}:
        raise ValueError("Source population must be visible product cards, cart lines or orders")
    if not 1 <= source["max_pages"] <= 10 or source["expected_count"] < 1:
        raise ValueError("Source capture needs a bounded page count and observed population size")
    if policy["kind"] == "budget_pair":
        if (
            set(policy) != {"kind", "budget", "tie_break"}
            or policy["tie_break"] != "sorted_title_tuple_ascending"
        ):
            raise ValueError(
                "Budget pair requires only its explicit budget and sorted title tuple tie rule"
            )
        if source["kind"] != "products" or source["max_pages"] != 2:
            raise ValueError("Budget pair requires two declared product pages")
        pair_budget = Decimal(str(policy["budget"]))
        if not pair_budget.is_finite() or pair_budget < 0:
            raise ValueError("Budget pair requires a finite nonnegative budget")
    elif (
        policy.get("fact_field") in {"price_share", "eligible_price_share"}
        and source["kind"] != "products"
    ):
        raise ValueError("Product price shares require a product population")
    os.umask(0o077)
    output = args.batch_root / "author" / args.generated_task_id / args.attempt
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    parameters = dict(config["parameters"])
    origins = {(urlsplit(url).scheme, urlsplit(url).netloc) for url in sites.values()}
    blocked, pages, saved = [], [], []
    evidence, source_final_checks = [], []
    save_phase = False
    target_project_path = None
    create_note_url = None
    submitted = False
    context_id = str(uuid4())
    started_at = datetime.now(UTC).isoformat()

    def restrict(route):
        nonlocal submitted
        request = route.request
        url = urlsplit(request.url)
        # Cart hydration uses this POST only to read shipping estimates.
        shipping_estimate = (
            request.method == "POST"
            and (url.scheme, url.netloc)
            == (urlsplit(sites["shopping"]).scheme, urlsplit(sites["shopping"]).netloc)
            and url.path == "/rest/default/V1/carts/mine/estimate-shipping-methods"
        )
        # GitLab hydrates its visible sidebar through single and batched queries.
        # The selected named operation must be the query at the document start.
        graphql_read = False
        if (
            request.method == "POST"
            and (url.scheme, url.netloc)
            == (urlsplit(sites["gitlab"]).scheme, urlsplit(sites["gitlab"]).netloc)
            and url.path == "/api/graphql"
        ):
            payload = request.post_data_json
            documents = payload if isinstance(payload, list) else [payload]
            graphql_read = bool(documents)
            for document in documents:
                operation = re.match(r"\s*query\s+([A-Za-z_][A-Za-z_0-9]*)\s*[(\{]", document["query"])
                if not operation or document.get("operationName", operation[1]) != operation[1]:
                    graphql_read = False
                    break
        read = (
            request.method in {"GET", "HEAD"} and not ACTION_PATH.search(url.path)
        ) or shipping_estimate or graphql_read
        target_origin = urlsplit(sites["gitlab"])
        write = (
            save_phase
            and not submitted
            and request.method == "POST"
            and (url.scheme, url.netloc) == (target_origin.scheme, target_origin.netloc)
            and request.url == create_note_url
        )
        if (url.scheme, url.netloc) in origins and (read or write):
            if write:
                submitted = True
            route.continue_()
        else:
            blocked.append({"method": request.method, "path": url.path})
            route.abort()

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
        context.set_default_timeout(args.timeout_ms)
        context.tracing.start(screenshots=True, snapshots=True, sources=False)
        try:
            page = context.new_page()
            page.goto(sites["shopping"].rstrip("/") + "/", wait_until="domcontentloaded")
            page.wait_for_timeout(args.settle_ms)
            current = record_observation(
                page,
                output,
                "source_home",
                "shopping",
                context_id=context_id,
                producer=Path(__file__),
            )
            cart_links = [
                link
                for link in current["links"]
                if link["title"].startswith("My Cart")
                and urlsplit(link["url"]).path.rstrip("/") == "/checkout/cart"
                and urlsplit(link["url"]).netloc == urlsplit(sites["shopping"]).netloc
            ]
            if len(cart_links) != 1:
                raise ValueError("Shopping homepage must expose its actual My Cart link")
            page.goto(cart_links[0]["url"], wait_until="domcontentloaded")
            page.wait_for_timeout(args.settle_ms)
            preservation_cart = record_observation(
                page,
                output,
                "preservation_cart",
                "shopping",
                context_id=context_id,
                producer=Path(__file__),
            )
            cart = preservation_cart["cart_items"]
            parameters.update(
                preservation_cart_url=preservation_cart["url"].replace(
                    sites["shopping"], "__SHOPPING__", 1
                ),
                preservation_cart_titles=[row["title"] for row in cart],
                preservation_added_count=0,
            )
            source_final_checks.append(
                {
                    "kind": "text_set_preserved",
                    "parameter": "preservation_cart_titles",
                    "selector": "#shopping-cart-table tbody.cart.item .product-item-name",
                    "added_count_parameter": "preservation_added_count",
                    "added_name_parameter": "note_body",
                    "site": "shopping",
                    "readback_url_parameter": "preservation_cart_url",
                }
            )
            for index, row in enumerate(cart):
                selector = (
                    "#shopping-cart-table tbody.cart.item tr.item-info:has("
                    ".product-item-name a[href$=" + json.dumps(urlsplit(row["url"]).path) + "])"
                )
                for field, kind in [("quantity", "input_value"), ("subtotal", "visible_text_exact")]:
                    parameter = f"preservation_cart_{index}_{field}"
                    parameters[parameter] = parse_fact_value(
                        row[field], row["readouts"][field]["value_type"]
                    )
                    source_final_checks.append(
                        {
                            "kind": kind,
                            "parameter": parameter,
                            "selector": selector + " " + row["readouts"][field]["selector"],
                            "site": "shopping",
                            "readback_url_parameter": "preservation_cart_url",
                        }
                    )
                    evidence.append(
                        {
                            "field": "/task_definition/parameters/" + parameter,
                            "artifact": "observation",
                            "location": f"/preservation_cart/cart_items/{index}/" + field,
                            "modality": "text",
                            "derivation": "Exact initial cart fact used only for independent preservation readback.",
                        }
                    )
            page.goto(sites["shopping"].rstrip("/") + "/", wait_until="domcontentloaded")
            page.wait_for_timeout(args.settle_ms)
            current = record_observation(
                page,
                output,
                "source_home_after_preservation",
                "shopping",
                context_id=context_id,
                producer=Path(__file__),
            )
            navigation = [current]
            for index, configured in enumerate(source["navigation"]):
                target = configured.replace("__SHOPPING__", sites["shopping"].rstrip("/"))
                parsed_target, source_origin = urlsplit(target), urlsplit(sites["shopping"])
                if (parsed_target.scheme, parsed_target.netloc) != (
                    source_origin.scheme,
                    source_origin.netloc,
                ) or ACTION_PATH.search(parsed_target.path):
                    raise ValueError(
                        "Source navigation must stay on read-only declared Shopping pages"
                    )
                links = [link for link in current["links"] if link["url"] == target]
                if not links:
                    raise ValueError(
                        "Configured source navigation is absent from current visible links"
                    )
                page.goto(target, wait_until="domcontentloaded")
                page.wait_for_timeout(args.settle_ms)
                current = record_observation(
                    page,
                    output,
                    f"source_navigation_{index:02d}",
                    "shopping",
                    context_id=context_id,
                    producer=Path(__file__),
                )
                navigation.append(current)
            source_url = page.url
            for index in range(source["max_pages"]):
                captured = record_observation(
                    page,
                    output,
                    f"source_{index:02d}",
                    "shopping",
                    context_id=context_id,
                    producer=Path(__file__),
                )
                pages.append(captured)
                # The public task declares this finite page window; an available
                # next page does not expand the selected population implicitly.
                if index + 1 == source["max_pages"]:
                    break
                following = page.locator(".pages-item-next a:visible")
                if not following.count():
                    break
                following.first.click()
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_timeout(args.settle_ms)
            bindings = [
                (page_index, row_index, entity)
                for page_index, captured in enumerate(pages)
                for row_index, entity in enumerate(captured[source["kind"]])
            ]
            entities = [entity for _, _, entity in bindings]
            if len(entities) != source["expected_count"]:
                raise ValueError("Fresh source population differs from observed configuration")
            if policy["kind"] == "rank":
                selected = select_entity(entities, policy)
            else:
                if len(pages) != 2 or any(not captured["products"] for captured in pages):
                    raise ValueError("Budget pair requires two nonempty actual product pages")
                if len({row["url"] for row in entities}) != len(entities):
                    raise ValueError("Selection requires distinct visible entity URLs")
                prices = []
                for row in entities:
                    match = re.fullmatch(
                        r"\s*\$?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*", str(row["price"])
                    )
                    if not match:
                        raise ValueError("Budget pair requires every visible population price")
                    prices.append(Decimal(match[1].replace(",", "")))
                if not sum(prices):
                    raise ValueError("Budget pair requires a nonzero visible population total")
                pairs = []
                # Enumerate only cross-page pairs; title tuples own deterministic ties.
                for first, (first_page, _, _) in enumerate(bindings):
                    for second in range(first + 1, len(bindings)):
                        second_page = bindings[second][0]
                        if first_page == second_page or prices[first] + prices[second] > pair_budget:
                            continue
                        indices = tuple(
                            sorted((first, second), key=lambda index: entities[index]["title"])
                        )
                        pairs.append((
                            prices[first] + prices[second],
                            tuple(entities[index]["title"] for index in indices),
                            indices,
                        ))
                if not pairs:
                    raise ValueError("The observed pages have no qualifying budget pair")
                ordered_pairs = sorted(pairs, key=lambda pair: (-pair[0], pair[1]))
                if sum(pair[:2] == ordered_pairs[0][:2] for pair in ordered_pairs) > 1:
                    raise ValueError("Public selection rules do not uniquely identify a budget pair")
                anchor, counterpart = ordered_pairs[0][2]
                price_parameters = [
                    "selected_price"
                    if index == anchor
                    else "counterpart_price"
                    if index == counterpart
                    else f"product_{index}_price"
                    for index in range(len(entities))
                ]
                selected = {
                    "index": anchor,
                    "entity": entities[anchor],
                    "counterpart_index": counterpart,
                    "fact_field": "pair_summary",
                    "derivations": {
                        "window_total": {
                            "operation": "sum", "parameters": price_parameters,
                            "output_format": ".2f",
                        },
                        "pair_total": {
                            "operation": "sum", "parameters": ["selected_price", "counterpart_price"],
                            "output_format": ".2f",
                        },
                        "pair_share": {
                            "operation": "ratio_percent", "parameters": ["pair_total", "window_total"],
                            "output_format": ".2f",
                        },
                        "selected_fact": {
                            "operation": "format",
                            "parameters": ["counterpart_title", "pair_total", "pair_share"],
                            "template": (
                                "Partner: {counterpart_title} | Pair total: ${pair_total}"
                                " | Window share: {pair_share}%"
                            ),
                        },
                    },
                }
            if source["kind"] == "products":
                for page_index, captured in enumerate(pages):
                    url_parameter = f"preservation_products_{page_index}_url"
                    parameters[url_parameter] = captured["url"].replace(
                        sites["shopping"], "__SHOPPING__", 1
                    )
                    for row_index, row in enumerate(captured["products"]):
                        selector = (
                            "main .product-item-info:has(.product-item-link[href$="
                            + json.dumps(urlsplit(row["url"]).path)
                            + "])"
                        )
                        for field in ["title", "price", "reviews"]:
                            if row[field] is None:
                                continue
                            parameter = f"preservation_product_{page_index}_{row_index}_{field}"
                            parameters[parameter] = row[field]
                            source_final_checks.append(
                                {
                                    "kind": "visible_text_exact",
                                    "parameter": parameter,
                                    "selector": selector + " " + row["readouts"][field]["selector"],
                                    "site": "shopping",
                                    "readback_url_parameter": url_parameter,
                                }
                            )
                            evidence.append(
                                {
                                    "field": "/task_definition/parameters/" + parameter,
                                    "artifact": "observation",
                                    "location": f"/source_pages/{page_index}/products/{row_index}/" + field,
                                    "modality": "text",
                                    "derivation": "Exact original displayed population fact used only for independent preservation readback.",
                                }
                            )
            if source["kind"] == "orders":
                for index, captured in enumerate(pages):
                    url_parameter = f"preservation_orders_{index}_url"
                    rows_parameter = f"preservation_orders_{index}_rows"
                    parameters[url_parameter] = captured["url"].replace(
                        sites["shopping"], "__SHOPPING__", 1
                    )
                    parameters[rows_parameter] = [row["visible_text"] for row in captured["orders"]]
                    source_final_checks.append(
                        {
                            "kind": "text_set_preserved",
                            "parameter": rows_parameter,
                            "selector": "#my-orders-table tbody tr",
                            "added_count_parameter": "preservation_added_count",
                            "added_name_parameter": "note_body",
                            "site": "shopping",
                            "readback_url_parameter": url_parameter,
                        }
                    )
            entity = selected["entity"]
            source_page_index, source_row_index, _ = bindings[selected["index"]]
            source_url = pages[source_page_index]["url"]
            entity_pointer = (
                f"/source_pages/{source_page_index}/{source['kind']}/{source_row_index}"
            )
            parameters.update(
                source_url=source_url.replace(sites["shopping"], "__SHOPPING__", 1),
                selected_entity_url=entity["url"].replace(sites["shopping"], "__SHOPPING__", 1),
                selected_entity_title=parse_fact_value(
                    entity["title"], entity["readouts"]["title"]["value_type"]
                ),
            )
            for parameter, location in [
                ("source_url", f"/source_pages/{source_page_index}/url"),
                ("selected_entity_url", entity_pointer + "/url"),
                ("selected_entity_title", entity_pointer + "/title"),
            ]:
                evidence.append(
                    {
                        "field": "/task_definition/parameters/" + parameter,
                        "artifact": "observation",
                        "location": location,
                        "modality": "text",
                        "derivation": "Exact visible identity; replace the declared site origin with its canonical placeholder.",
                    }
                )
            dependencies = []
            if selected["fact_field"] in {
                "subtotal_share", "price_share", "eligible_price_share", "pair_summary"
            }:
                amount_field = "subtotal" if selected["fact_field"] == "subtotal_share" else "price"
                if (amount_field == "subtotal" and source["kind"] != "cart_items") or (
                    amount_field == "price" and source["kind"] != "products"
                ):
                    raise ValueError("Share inputs must use their declared cart or product population")
                prefix = "cart_item" if amount_field == "subtotal" else "product"
                for index, (page_index, row_index, row) in enumerate(bindings):
                    pointer = f"/source_pages/{page_index}/{source['kind']}/{row_index}"
                    names = (
                        {
                            "url": "selected_entity_url",
                            "title": "selected_entity_title",
                            "source_url": "source_url",
                            amount_field: "selected_" + amount_field,
                        }
                        if index == selected["index"]
                        else {
                            field: (
                                "counterpart_" + field
                                if selected["fact_field"] == "pair_summary"
                                and index == selected["counterpart_index"]
                                else f"{prefix}_{index}_{field}"
                            )
                            for field in ["url", "title", "source_url", amount_field]
                        }
                    )
                    for field, name in names.items():
                        if index == selected["index"] and field != amount_field:
                            continue
                        value = pages[page_index]["url"] if field == "source_url" else row[field]
                        parameters[name] = (
                            value.replace(sites["shopping"], "__SHOPPING__", 1)
                            if field in {"url", "source_url"}
                            else parse_fact_value(value, row["readouts"][field]["value_type"])
                        )
                        evidence.append(
                            {
                                "field": "/task_definition/parameters/" + name,
                                "artifact": "observation",
                                "location": f"/source_pages/{page_index}/url"
                                if field == "source_url"
                                else pointer + "/" + field,
                                "modality": "text",
                                "derivation": "Exact original population amount and its actual visible entity.",
                            }
                        )
                    readouts = {names[amount_field]: row["readouts"][amount_field]}
                    if selected["fact_field"] == "pair_summary" and index == selected["counterpart_index"]:
                        readouts[names["title"]] = row["readouts"]["title"]
                    dependencies.append(
                        {
                            "name": f"{prefix}_{index}_to_share",
                            "source_site": "shopping",
                            "target_site": "gitlab",
                            "source_entity": {
                                "identity_parameter": names["url"],
                                "label_parameter": names["title"],
                            },
                            "source_url_parameter": names["source_url"],
                            "fact_readouts": readouts,
                            "target_event_index": 1,
                        }
                    )
                for name, rule in selected["derivations"].items():
                    parameters[name] = derive_value(rule, parameters)
                    evidence.append(
                        {
                            "field": "/task_definition/parameters/" + name,
                            "artifact": "observation",
                            "location": "/source_pages",
                            "modality": "text",
                            "rule": rule,
                            "inputs": [
                                "/task_definition/parameters/" + ref for ref in rule["parameters"]
                            ],
                            "derivation": "Apply the declared derivation to the original bound population inputs; Decimal arithmetic rounds ties to even.",
                        }
                    )
                selected["fact"] = parameters["selected_fact"]
            else:
                parameters["selected_fact"] = parse_fact_value(
                    selected["fact"], entity["readouts"][selected["fact_field"]]["value_type"]
                )
                evidence.append(
                    {
                        "field": "/task_definition/parameters/selected_fact",
                        "artifact": "observation",
                        "location": entity_pointer + "/" + selected["fact_field"],
                        "modality": "text",
                        "derivation": "Read the exact entity field selected by the archived policy; parse only its declared readout value type.",
                    }
                )
                dependencies.append(
                    {
                        "name": "selected_entity_to_issue_note",
                        "source_site": "shopping",
                        "target_site": "gitlab",
                        "source_entity": {
                            "identity_parameter": "selected_entity_url",
                            "label_parameter": "selected_entity_title",
                        },
                        "source_url_parameter": "source_url",
                        "fact_readouts": {
                            "selected_entity_title": entity["readouts"]["title"],
                            "selected_fact": entity["readouts"][selected["fact_field"]],
                        },
                        "target_event_index": 1,
                    }
                )
            parameters["note_body"] = config["note_template"].format_map(parameters)
            # Each dependency has an actual viewport image before its target use.
            entity_reads = []
            for index in (
                range(len(bindings))
                if selected["fact_field"] in {"subtotal_share", "price_share", "eligible_price_share", "pair_summary"}
                else [selected["index"]]
            ):
                page_index, row_index, _ = bindings[index]
                if page.url != pages[page_index]["url"]:
                    page.goto(pages[page_index]["url"], wait_until="domcontentloaded")
                    page.wait_for_timeout(args.settle_ms)
                selector = {
                    "products": "main .product-item-info:visible",
                    "cart_items": "#shopping-cart-table tbody.cart.item tr.item-info:visible",
                    "orders": "#my-orders-table tbody tr:visible",
                }[source["kind"]]
                page.locator(selector).nth(row_index).scroll_into_view_if_needed()
                entity_reads.append(
                    record_observation(
                        page,
                        output,
                        f"source_entity_{index:02d}",
                        "shopping",
                        context_id=context_id,
                        producer=Path(__file__),
                    )
                )
            issue_url = config["target"]["issue_url"].replace(
                "__GITLAB__", sites["gitlab"].rstrip("/")
            )
            parsed = urlsplit(issue_url)
            if (parsed.scheme, parsed.netloc) != (
                urlsplit(sites["gitlab"]).scheme,
                urlsplit(sites["gitlab"]).netloc,
            ) or not re.fullmatch(r"/(.+)/-/issues/\d+", parsed.path):
                raise ValueError("Target must be an observed existing issue on declared GitLab")
            target_response = page.goto(issue_url, wait_until="domcontentloaded")
            if target_response is None or not target_response.ok:
                raise ValueError("Existing target issue navigation did not return successful HTML")
            # Vue consumes this initialization data and removes it from the mounted DOM.
            target_html = BeautifulSoup(target_response.text(), "html.parser")
            target_metadata = target_html.select(
                '#js-vue-notes[data-noteable-type="Issue"][data-noteable-data]'
            )
            if len(target_metadata) != 1:
                raise ValueError("Target response must expose one actual Issue noteable identity")
            target_entity = json.loads(target_metadata[0]["data-noteable-data"])
            page.wait_for_timeout(args.settle_ms)
            before = record_observation(
                page,
                output,
                "target_before",
                "gitlab",
                context_id=context_id,
                producer=Path(__file__),
            )
            if (
                before["issue_state"]["title"] != config["target"]["title"]
                or before["url"] != issue_url
            ):
                raise ValueError("Fresh target does not match the observed existing issue")
            if (
                str(target_entity["iid"]) != before["issue_identity"]["issue_iid"]
                or target_entity["title"] != before["issue_state"]["title"]
                or urljoin(issue_url, target_entity["web_url"]) != before["url"]
            ):
                raise ValueError("Target initialization identity differs from the visible issue")
            create_note_url = urljoin(issue_url, target_entity["create_note_path"])
            create_note_parts = urlsplit(create_note_url)
            target_project_path = before["issue_identity"]["project_path"]
            if (
                (create_note_parts.scheme, create_note_parts.netloc) != (parsed.scheme, parsed.netloc)
                or create_note_parts.path != "/" + target_project_path + "/notes"
            ):
                raise ValueError("Target initialization Save must belong to its actual project")
            if any(note["text"] == parameters["note_body"] for note in before["notes"]):
                raise ValueError("Target already contains the requested note")
            textarea = page.get_by_test_id("comment-field")
            if textarea.count() != 1:
                raise ValueError("Existing issue must expose one visible comment editor")
            textarea.fill(parameters["note_body"])
            save_phase = True
            with page.expect_response(
                lambda response: (
                    response.request.method == "POST"
                    and response.url == create_note_url
                )
            ) as response_info:
                page.get_by_role("button", name="Comment", exact=True).click()
            save_phase = False
            response = response_info.value
            saved.append(
                {"url": response.url, "method": response.request.method, "status": response.status}
            )
            if not response.ok:
                raise ValueError("Actual note Save failed; preserve this attempt")
            page.goto(issue_url, wait_until="domcontentloaded")
            page.wait_for_timeout(args.settle_ms)
            after = record_observation(
                page,
                output,
                "target_after",
                "gitlab",
                context_id=context_id,
                producer=Path(__file__),
            )
            page.reload(wait_until="domcontentloaded")
            page.wait_for_timeout(args.settle_ms)
            terminal = record_observation(
                page, output, "terminal", "gitlab", context_id=context_id, producer=Path(__file__)
            )
            for captured in [after, terminal]:
                if captured["issue_state"] != before["issue_state"]:
                    raise ValueError("Adding the note changed another issue field")
                old = [row["text"] for row in before["notes"]]
                actual = [row["text"] for row in captured["notes"]]
                if sorted(actual) != sorted([*old, parameters["note_body"]]):
                    raise ValueError(
                        "Readback did not preserve existing notes and exactly one new note"
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
    events = load_har_trace(output / "network.har")
    source_events = [
        event
        for event in events
        if event.url == source_url and event.http_method == "GET" and event.request_status == 200
    ]
    saves = [
        event
        for event in events
        if event.url == saved[0]["url"]
        and event.http_method == "POST"
        and event.request_status == saved[0]["status"]
    ]
    if not source_events or len(saves) != 1 or not saves[0].is_request_success:
        raise ValueError("Completion proof requires the actual source read and one successful Save")
    pending, leaves = [("$", saves[0].post_data)], []
    while pending:
        pointer, value = pending.pop()
        if isinstance(value, dict):
            pending.extend(
                (pointer + "[" + json.dumps(key) + "]", child) for key, child in value.items()
            )
        elif isinstance(value, list):
            pending.extend((pointer + f"[{index}]", child) for index, child in enumerate(value))
        else:
            leaves.append((pointer, value))
    body_paths = [path for path, value in leaves if value == parameters["note_body"]]
    if len(body_paths) != 1:
        raise ValueError("Actual Save body must contain the derived note exactly once")
    critical = {body_paths[0]: "{note_body}"}
    for path, value in leaves:
        if any(
            path.endswith("[" + json.dumps(key) + "]")
            for key in [
                "noteable_id",
                "noteable_type",
                "target_id",
                "target_type",
                "issue_id",
                "iid",
                "internal",
            ]
        ):
            critical[path] = value
    noteable_ids = [value for path, value in leaves if path.endswith('["noteable_id"]')]
    noteable_types = [value for path, value in leaves if path.endswith('["noteable_type"]')]
    if noteable_ids != [target_entity["id"]] or noteable_types != ["Issue"]:
        raise ValueError(
            "Actual note Save does not bind the exact Issue entity observed before writing"
        )
    internal_flags = [value for path, value in leaves if path.endswith('["internal"]')]
    if len(internal_flags) != 1 or internal_flags[0] is not False:
        raise ValueError("Actual note Save must create an ordinary, non-internal comment")
    source_event = source_events[0]
    completion = [
        {
            "url_template": source_event.url.replace(sites["shopping"], "__SHOPPING__", 1),
            "http_method": "GET",
            "response_status": source_event.request_status,
            "last_event_only": False,
        },
        {
            "url_template": saves[0].url.replace(sites["gitlab"], "__GITLAB__", 1),
            "http_method": "POST",
            "response_status": saves[0].request_status,
            "post_data": critical,
            "post_data_schema": {
                "type": "object",
                "properties": {
                    path: {
                        "type": "boolean"
                        if isinstance(value, bool)
                        else "string" if isinstance(value, str) else "number"
                    }
                    for path, value in critical.items()
                },
            },
            "last_event_only": False,
        },
    ]
    parameters.update(
        issue_url=issue_url.replace(sites["gitlab"], "__GITLAB__", 1),
        preserved_notes=[row["text"] for row in before["notes"]],
        added_note_count=1,
        issue_title=before["issue_state"]["title"],
        issue_description=before["issue_state"]["description"],
        issue_assignee=before["issue_state"]["assignee"],
        issue_labels=before["issue_state"]["labels"],
        issue_state_control="Close issue" if before["issue_state"]["open"] else "Reopen issue",
        added_label_count=0,
    )
    final_checks = [
        {
            "kind": "visible_text_exact",
            "parameter": "note_body",
            "selector": "li.note-comment:not(.system-note) .note-text.md:has-text("
            + json.dumps(parameters["note_body"])
            + ")",
        },
        {
            "kind": "visible_text_exact",
            "parameter": "issue_state_control",
            "selector": 'form.js-main-target-form button[data-testid="close-reopen-button"]',
        },
        {
            "kind": "text_set_preserved",
            "parameter": "preserved_notes",
            "selector": "li.note-comment:not(.system-note) .note-text.md",
            "added_count_parameter": "added_note_count",
            "added_name_parameter": "note_body",
        },
        {
            "kind": "text_set_preserved",
            "parameter": "issue_labels",
            "selector": '[data-testid="sidebar-labels"] .gl-label-text',
            "added_count_parameter": "added_label_count",
            "added_name_parameter": "note_body",
        },
    ]
    for parameter, selector in [
        ("issue_title", "h1.title"),
        ("issue_description", "div.description"),
        ("issue_assignee", '[data-testid="assignee-block-container"]'),
    ]:
        if parameters[parameter] is not None:
            final_checks.append(
                {"kind": "visible_text_exact", "parameter": parameter, "selector": selector}
            )
    for index, note in enumerate(before["notes"]):
        parameter = f"preserved_note_{index}"
        parameters[parameter] = note["text"]
        final_checks.append(
            {
                "kind": "visible_text_exact",
                "parameter": parameter,
                "selector": "#" + note["id"] + " .note-text.md",
            }
        )
        evidence.append(
            {
                "field": "/task_definition/parameters/" + parameter,
                "artifact": "observation",
                "location": f"/target_before/notes/{index}/text",
                "modality": "text",
                "derivation": "Exact original note text bound to its actual existing DOM identity.",
            }
        )
    for index, item in enumerate(before["issue_state"]["checklist"]):
        parameter = f"preserved_checklist_{index}"
        parameters[parameter] = item["checked"]
        selector = "div.description li:has-text(" + json.dumps(item["label"]) + ")"
        final_checks.append(
            {
                "kind": "checked",
                "parameter": parameter,
                "selector": selector + " input[type=checkbox]",
                "label_selector": selector,
            }
        )
    for check in final_checks:
        check.update(site="gitlab", readback_url_parameter="issue_url")
    final_checks.extend(source_final_checks)
    archived = {
        "preservation_cart": preservation_cart,
        "source_navigation": navigation,
        "source_pages": pages,
        "source_entity_reads": entity_reads,
        "selection_policy": config["selection"],
        "selection": selected,
        "target_before": before,
        "target_after": after,
        "terminal": terminal,
        "target_entity": target_entity,
        "save_requests": saved,
        "completion_events": completion,
        "blocked_requests": blocked,
    }
    evidence.append(
        {
            "field": "/task_definition/parameters/issue_url",
            "artifact": "observation",
            "location": "/target_before/url",
            "modality": "text",
            "derivation": "Actual existing target issue URL.",
        }
    )
    evidence.append(
        {
            "field": "/task_definition/parameters/note_body",
            "artifact": "observation",
            "location": "/terminal/notes",
            "modality": "text",
            "derivation": "Render the declared note template from the selected source facts.",
            "rule": {
                "operation": "format",
                "template": config["note_template"],
                "parameters": ["selected_entity_title", "selected_fact"],
            },
            "inputs": [
                "/task_definition/parameters/selected_entity_title",
                "/task_definition/parameters/selected_fact",
            ],
        }
    )
    for index, event in enumerate(completion):
        evidence.append(
            {
                "field": f"/task_definition/completion_events/{index}",
                "artifact": "har",
                "location": event["url_template"],
                "modality": "text",
                "derivation": "Actual successful request from this context recording.",
            }
        )
    relative = output.relative_to(args.batch_root)
    draft = {
        "generated_task_id": args.generated_task_id,
        "task_id": config["task_id"],
        "intent_template_id": config["intent_template_id"],
        "revision": config["revision"],
        "sites": ["shopping", "gitlab"],
        "start_urls": ["__SHOPPING__", "__GITLAB__"],
        "source_artifact": config["source_artifact"],
        "source_record_id": config["source_record_id"],
        "baseline": baseline,
        "task_definition": {
            "task_type": "MUTATE",
            "goal_template": config["goal_template"],
            "parameters": parameters,
            "answer": {"status": "SUCCESS", "values": None, "item_schema": None, "ordered": False},
            "completion_events": completion,
            "dependencies": dependencies,
        },
        "grounding": {
            "captured_at": datetime.now(UTC).isoformat(),
            "family": config["family"],
            "artifacts": {
                "har": str(relative / "network.har"),
                "trace": str(relative / "trace.zip"),
                "screenshot": str(relative / "terminal.png"),
                "observation": str(relative / "observation.json"),
            },
            "field_evidence": evidence,
            "final_state_checks": final_checks,
            "episode_boundary_artifact": {
                "path": str(relative / "episode_boundary.json"),
                "sha256": sha256_file(output / "episode_boundary.json"),
            },
        },
    }
    (output / "observation.json").write_text(
        json.dumps(archived, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"draft": str(output / "draft.json"), "model_calls": 0}))
    return draft


if __name__ == "__main__":
    ground_task(_parse_args())
