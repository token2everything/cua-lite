#!/usr/bin/env python3
"""Ground bounded account-order joins on a new fixed-image Shopping baseline.

The coordinator supplies standard UI authentication only. Select one frozen
task with --task-configs and --generated-task-id. Read the actual Recent Orders
window and its five linked detail pages; emit a draft only after preserved UI
readback and a trace containing no Shopping business action. No model is called.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from datetime import UTC, datetime
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path
from urllib.parse import urlsplit

from author_shopping import _usd
from probe_shopping_next5_readonly_v1 import _record_readonly


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--task-configs", type=Path, required=True)
    parser.add_argument("--generated-task-id", required=True)
    parser.add_argument("--attempt", default="grounding-01")
    args = parser.parse_args()
    if not re.fullmatch(r"wavgen-v03-full300-03(?:4[6-9]|5[0-3])", args.generated_task_id):
        parser.error("task ID must belong to the frozen 0346–0353 order-line cohort")
    if not re.fullmatch(r"grounding-[0-9]{2}", args.attempt):
        parser.error("attempt must use the two-digit grounding namespace")
    origin = urlsplit(args.site_url)
    if (
        origin.scheme != "http"
        or not origin.netloc
        or any([origin.path, origin.query, origin.fragment, origin.username, origin.password])
    ):
        parser.error("site URL must be an explicit HTTP origin without credentials")
    return args


def parse_order_window(account: dict, details: list[dict], expected_count: int) -> list[dict]:
    """Reconcile each actual order header, line unit, and displayed monetary total."""
    tables = [table for table in account["tables"] if table["id"] == "my-orders-table"]
    if len(tables) != 1 or tables[0]["headers"] != [
        "Order #",
        "Date",
        "Ship To",
        "Order Total",
        "Status",
        "Action",
    ]:
        raise ValueError("account does not expose the observed Recent Orders table")
    rows = tables[0]["rows"]
    if len(rows) != expected_count or len(details) != expected_count:
        raise ValueError("the complete five-order detail scope is required")
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("Recent Orders window contains repeated order identities")
    parsed = []
    for index, (header, raw) in enumerate(zip(rows, details, strict=True)):
        table = [table for table in raw["tables"] if table["id"] == "my-orders-table"]
        if len(table) != 1 or table[0]["headers"] != [
            "Product Name",
            "SKU",
            "Price",
            "Qty",
            "Subtotal",
        ]:
            raise ValueError("order detail does not expose the observed line-unit table")
        heading = re.match(
            r"Order # ([0-9]+)\n([^\n]+)\nOrder Date: ([^\n]+)\n",
            raw["readonly_visible"]["main_text"][0],
        )
        if heading is None or heading[1] != header[0] or heading[2] != header[4]:
            raise ValueError("detail order identity/status differs from the account header")
        date = datetime.strptime(header[1], "%m/%d/%y").date()
        if datetime.strptime(heading[3], "%B %d, %Y").date() != date:
            raise ValueError("detail date differs from the account header")
        # Only these three monetary labels are observed. A missing tax/discount
        # label stays absent; it is never invented as a zero-valued field.
        footers = raw["readonly_visible"]["footer_rows"]
        if [row[0] for row in footers] != ["Subtotal", "Shipping & Handling", "Grand Total"]:
            raise ValueError("new footer labels need fresh explicit authoring before use")
        subtotal, shipping, grand = [_usd(row[1]) for row in footers]
        lines = []
        for line_index, cells in enumerate(table[0]["rows"]):
            if len(cells) != 5 or not cells[1]:
                raise ValueError("visible order line is incomplete")
            quantity = re.fullmatch(r"Ordered\s*([0-9]+)", cells[3])
            if quantity is None or int(quantity[1]) <= 0:
                raise ValueError("only actual positive Ordered quantities are supported")
            price, line_subtotal = _usd(cells[2]), _usd(cells[4])
            if price * int(quantity[1]) != line_subtotal:
                raise ValueError("visible price × Ordered quantity does not reconcile")
            lines.append(
                {
                    "product_name": cells[0].split("\n", 1)[0],
                    "displayed_name_and_options": cells[0],
                    "sku": cells[1],
                    "ordered_quantity": int(quantity[1]),
                    "unit_price": price,
                    "line_subtotal": line_subtotal,
                    "source": (
                        f"/details/{index}/tables/{raw['tables'].index(table[0])}"
                        f"/rows/{line_index}"
                    ),
                }
            )
        if not lines or len({line["sku"] for line in lines}) != len(lines):
            raise ValueError("detail requires a complete distinct-SKU line window")
        if sum(line["line_subtotal"] for line in lines) != subtotal:
            raise ValueError("all actual line subtotals do not equal the displayed subtotal")
        if subtotal + shipping != grand or grand != _usd(header[3]):
            raise ValueError("displayed subtotal, shipping, grand total and header disagree")
        parsed.append(
            {
                "order_number": header[0],
                "date": date.isoformat(),
                "status": header[4],
                "subtotal": subtotal,
                "shipping": shipping,
                "grand_total": grand,
                "lines": lines,
                "source": f"/details/{index}",
                "url": raw["url"],
            }
        )
    return parsed


def build_definition(config: dict, orders: list[dict]) -> tuple[dict, list[dict]]:
    """Apply the public joins, budget, ranking, or cent-allocation business policy."""
    parameters = dict(config["parameters"])
    family = config["family"]
    cent = Decimal("0.01")
    sources = []
    if family == "order_line_anchor_peer_price_join":
        candidates = [
            (order, line)
            for order in orders
            for line in order["lines"]
            if order["status"] == parameters["required_status"]
            and all(
                word.casefold() in line["product_name"].casefold()
                for word in [parameters["title_word_1"], parameters["title_word_2"]]
            )
        ]
        if len(candidates) != 1:
            raise ValueError("the public status/title predicate must identify one anchor line")
        anchor_order, anchor = candidates[0]
        peers = [
            (order, line)
            for order in orders
            for line in order["lines"]
            if order["status"] == parameters["required_status"]
            and order["date"] == anchor_order["date"]
            and order["order_number"] != anchor_order["order_number"]
            and line["unit_price"] > anchor["unit_price"]
        ]
        peers.sort(
            key=lambda pair: (-pair[1]["unit_price"], pair[0]["order_number"], pair[1]["sku"])
        )
        values = [
            {
                "order_number": order["order_number"],
                "order_date": order["date"],
                "order_status": order["status"],
                "product_name": line["product_name"],
                "sku": line["sku"],
                "ordered_quantity": line["ordered_quantity"],
                "unit_price_usd": float(line["unit_price"]),
                "line_subtotal_usd": float(line["line_subtotal"]),
                "anchor_order_number": anchor_order["order_number"],
                "anchor_sku": anchor["sku"],
                "anchor_unit_price_usd": float(anchor["unit_price"]),
                "unit_price_difference_usd": float(line["unit_price"] - anchor["unit_price"]),
            }
            for order, line in peers
        ]
        sources = [
            {
                "raw_sources": [order["source"], line["source"], anchor["source"]],
                "calculation": (
                    "Join status, exact date and line title; compare actual unit prices "
                    "and subtract anchor."
                ),
            }
            for order, line in peers
        ]
        goal = (
            "Use only the five orders displayed in My Account's initial Recent Orders window. "
            "Among orders with status {required_status}, find the unique order line whose "
            "product name contains both {title_word_1} and {title_word_2}, case-insensitively. "
            "Use that line as the anchor. From other {required_status} orders on that same "
            "date, report every line with a strictly higher unit price than the anchor. "
            "For each report order_number, order_date (YYYY-MM-DD), order_status, product_name "
            "(the product title without separate option labels), sku, ordered_quantity, "
            "unit_price_usd, line_subtotal_usd, anchor_order_number, anchor_sku, "
            "anchor_unit_price_usd and unit_price_difference_usd. Sort by unit price "
            "descending, then order number ascending, then SKU ascending."
        )
    elif family == "order_shipping_burden_reconciliation":
        parameters["allowed_status_text"] = ", ".join(config["allowed_statuses"])
        eligible = [order for order in orders if order["status"] in config["allowed_statuses"]]
        eligible.sort(
            key=lambda order: (-order["shipping"] / order["subtotal"], order["order_number"])
        )
        selected = eligible[: parameters["limit"]]
        if len(selected) != parameters["limit"]:
            raise ValueError("public eligible order scope is smaller than the requested rank")
        values = [
            {
                "order_number": order["order_number"],
                "order_date": order["date"],
                "order_status": order["status"],
                "distinct_sku_count": len(order["lines"]),
                "ordered_quantity_total": sum(line["ordered_quantity"] for line in order["lines"]),
                "line_subtotal_sum_usd": float(
                    sum(line["line_subtotal"] for line in order["lines"])
                ),
                "displayed_product_subtotal_usd": float(order["subtotal"]),
                "shipping_usd": float(order["shipping"]),
                "grand_total_usd": float(order["grand_total"]),
                "shipping_to_product_subtotal_percent": float(
                    (100 * order["shipping"] / order["subtotal"]).quantize(
                        cent, rounding=ROUND_HALF_UP
                    )
                ),
            }
            for order in selected
        ]
        sources = [
            {
                "raw_sources": [order["source"]],
                "calculation": (
                    "Complete actual SKU lines: count/sum Ordered and subtotals; "
                    "100 × shipping / product subtotal, half-up to 2 decimal places."
                ),
            }
            for order in selected
        ]
        goal = (
            "Audit the five orders displayed in My Account's initial Recent Orders window. "
            "Consider only orders with status {allowed_status_text}. Rank eligible orders by "
            "Shipping & Handling divided by the product Subtotal, descending, using the "
            "unrounded ratio for ranking; break ties by order number ascending. Report the "
            "first {limit}. Read their complete Items Ordered tables and for each report "
            "order_number, order_date (YYYY-MM-DD), order_status, distinct_sku_count, "
            "ordered_quantity_total, line_subtotal_sum_usd, displayed_product_subtotal_usd, "
            "shipping_usd, grand_total_usd and shipping_to_product_subtotal_percent. The "
            "percentage denominator is the product Subtotal. Round percentages half up "
            "to two decimal places and reconcile the line subtotals with the displayed totals."
        )
    elif family == "order_pair_merchandise_budget_optimization":
        eligible = [order for order in orders if order["status"] in {"Pending", "Complete"}]
        feasible = [
            pair
            for pair in itertools.combinations(eligible, 2)
            if sum(order["grand_total"] for order in pair) <= Decimal(str(parameters["budget_usd"]))
        ]
        if not feasible:
            raise ValueError("the public budget permits no pair of eligible orders")
        feasible.sort(
            key=lambda pair: (
                -sum(order["subtotal"] for order in pair),
                sum(order["shipping"] for order in pair),
                tuple(sorted(order["order_number"] for order in pair)),
            )
        )
        selected = sorted(feasible[0], key=lambda order: order["order_number"])
        answer = {}
        for index, order in enumerate(selected, 1):
            answer.update(
                {
                    f"order_{index}_number": order["order_number"],
                    f"order_{index}_distinct_sku_count": len(order["lines"]),
                    f"order_{index}_ordered_quantity_total": sum(
                        line["ordered_quantity"] for line in order["lines"]
                    ),
                    f"order_{index}_product_subtotal_usd": float(order["subtotal"]),
                    f"order_{index}_shipping_usd": float(order["shipping"]),
                    f"order_{index}_grand_total_usd": float(order["grand_total"]),
                }
            )
        pair_total = sum(order["grand_total"] for order in selected)
        answer.update(
            {
                "combined_product_subtotal_usd": float(
                    sum(order["subtotal"] for order in selected)
                ),
                "combined_shipping_usd": float(sum(order["shipping"] for order in selected)),
                "combined_grand_total_usd": float(pair_total),
                "unspent_budget_usd": float(Decimal(str(parameters["budget_usd"])) - pair_total),
            }
        )
        values = [answer]
        sources = [
            {
                "raw_sources": [order["source"] for order in orders],
                "calculation": (
                    "Enumerate all six pairs of actual Pending/Complete orders; reject "
                    "over-budget combined Grand Total; maximize combined product Subtotal, "
                    "then minimize shipping, then lexicographic sorted order numbers."
                ),
            }
        ]
        goal = (
            "Use only the five orders displayed in My Account's initial Recent Orders window. "
            "For a cost comparison, choose exactly two different orders with status Pending "
            "or Complete; exclude Canceled. Their combined displayed Grand Total must not "
            "exceed {budget_usd} USD. Maximize their combined product Subtotal; break ties by "
            "lower combined Shipping & Handling, then the lexicographically smaller pair "
            "of order numbers sorted ascending. Do not reorder or purchase anything. "
            "Read all candidate order details and reconcile each selected order's complete "
            "line table. Return one summary object with order_1_number and order_2_number "
            "in ascending order. For each prefix order_1 and order_2 also report "
            "distinct_sku_count, ordered_quantity_total, product_subtotal_usd, shipping_usd "
            "and grand_total_usd. Also report combined_product_subtotal_usd, "
            "combined_shipping_usd, combined_grand_total_usd and unspent_budget_usd."
        )
    elif family == "order_line_shipping_cent_allocation":
        eligible = [
            order for order in orders if len(order["lines"]) >= parameters["minimum_distinct_skus"]
        ]
        if config["selection"] == "largest_shipping_ratio":
            parameters["selection_text"] = (
                "the highest Shipping & Handling / product Subtotal ratio"
            )
            eligible.sort(
                key=lambda order: (-order["shipping"] / order["subtotal"], order["order_number"])
            )
        elif config["selection"] == "largest_product_subtotal":
            parameters["selection_text"] = "the highest product Subtotal"
            eligible.sort(key=lambda order: (-order["subtotal"], order["order_number"]))
        else:
            raise ValueError("unsupported public order selection")
        if not eligible:
            raise ValueError("no order meets the public distinct-SKU condition")
        order = eligible[0]
        shipping_cents = int(order["shipping"] * 100)
        exact = [
            Decimal(shipping_cents) * line["line_subtotal"] / order["subtotal"]
            for line in order["lines"]
        ]
        allocated = [int(value.to_integral_value(rounding=ROUND_DOWN)) for value in exact]
        residual_order = sorted(
            range(len(exact)), key=lambda i: (-(exact[i] - allocated[i]), order["lines"][i]["sku"])
        )
        for index in residual_order[: shipping_cents - sum(allocated)]:
            allocated[index] += 1
        values = [
            {
                "order_number": order["order_number"],
                "order_status": order["status"],
                "sku": line["sku"],
                "ordered_quantity": line["ordered_quantity"],
                "line_subtotal_usd": float(line["line_subtotal"]),
                "allocated_shipping_usd": float(Decimal(allocated[index]) / 100),
                "allocated_line_cost_usd": float(
                    line["line_subtotal"] + Decimal(allocated[index]) / 100
                ),
            }
            for index, line in enumerate(order["lines"])
        ]
        values.sort(key=lambda row: row["sku"])
        sources = [
            {
                "raw_sources": [order["source"]],
                "calculation": (
                    "Allocate actual shipping cents proportional to every actual line "
                    "subtotal; floor cents, distribute the remaining cents by descending "
                    "fractional remainder/SKU ascending; add line subtotal. Sum equals "
                    "displayed Grand Total."
                ),
            }
            for _ in values
        ]
        goal = (
            "Use only the five orders displayed in My Account's initial Recent Orders window, "
            "including Canceled. Among orders with at least {minimum_distinct_skus} distinct "
            "SKUs, select the order with {selection_text}; break a tie by order number "
            "ascending. Audit a proportional allocation of that order's displayed Shipping "
            "& Handling across its complete Items Ordered lines. Work in integer cents: "
            "multiply shipping cents by each line's Subtotal divided by the product "
            "Subtotal, floor each allocation, then distribute the remaining cents one per "
            "line by descending fractional remainder, breaking ties by SKU ascending. "
            "For each line report order_number, order_status, sku, ordered_quantity, "
            "line_subtotal_usd, allocated_shipping_usd and allocated_line_cost_usd "
            "(line subtotal plus allocated shipping). Sort the lines by SKU ascending. "
            "The allocations must sum to the displayed shipping and the allocated line "
            "costs must sum to the displayed Grand Total."
        )
    else:
        raise ValueError(f"unsupported order-line family: {family}")
    if not values:
        raise ValueError("successful retrieval requires a nonempty actual answer")
    fields = {}
    for name, value in values[0].items():
        fields[name] = {
            "type": "string"
            if isinstance(value, str)
            else "integer"
            if isinstance(value, int)
            else "number"
        }
    goal += (
        " Keep orders, cart, wishlist, comparison list and reviews unchanged. "
        "Return USD amounts as numbers without currency symbols."
    )
    events = [
        {
            "url_template": "__SHOPPING__" + urlsplit(order["url"]).path,
            "http_method": "GET",
            "response_status": 200,
            "last_event_only": False,
        }
        for order in orders
    ]
    # Bind observed business-write routes. A passive cart shipping-estimate POST
    # reads business data and must not become a hidden model failure condition.
    events.append(
        {
            "url_template": (
                "^__SHOPPING__/(?:checkout/cart/(?:add|addgroup|updatePost|delete)|"
                "wishlist/index/add|catalog/product_compare/add|sales/order/reorder|"
                "review/product/post)(?:/.*)?$"
            ),
            "http_method": "POST",
            "should_not_exist": True,
            "last_event_only": False,
        }
    )
    return (
        {
            "task_type": "RETRIEVE",
            "goal_template": goal,
            "parameters": parameters,
            "answer": {
                "status": "SUCCESS",
                "values": values,
                "ordered": True,
                "item_schema": {
                    "type": "object",
                    "properties": fields,
                    "required": list(fields),
                    "additionalProperties": False,
                },
            },
            "completion_events": events,
        },
        sources,
    )


def _read_window(page, output: Path, site_url: str, config: dict) -> tuple[dict, list[dict], dict]:
    def visit(path: str, name: str) -> dict:
        result = page.goto(site_url + path, wait_until="networkidle", timeout=120_000)
        captured = _record_readonly(page, output, name)
        if result is None or result.status != 200 or urlsplit(captured["url"]).path != path:
            raise ValueError("actual linked order read did not return its HTTP 200 page")
        return captured

    # Read the public task's bounded account window and its actual resource links.
    initial = visit("/customer/account/", "initial_account")
    tables = [table for table in initial["tables"] if table["id"] == "my-orders-table"]
    if len(tables) != 1 or len(tables[0]["rows"]) != config["expected_order_count"]:
        raise ValueError("account's complete Recent Orders window changed")
    links = [link for link in initial["links"] if link["text"] == "View Order"]
    by_order = {}
    for link in links:
        target = urlsplit(link["url"])
        matched = re.fullmatch(r"/sales/order/view/order_id/([0-9]+)/", target.path)
        if matched is None or target.netloc != urlsplit(site_url).netloc:
            raise ValueError("actual View Order link is outside the fixed owned order scope")
        by_order[matched[1].lstrip("0")] = target.path
    if len(by_order) != config["expected_order_count"]:
        raise ValueError("Recent Orders needs exactly one actual View Order link per order")
    # Record every linked detail page before deriving any answer.
    details = []
    for index, row in enumerate(tables[0]["rows"]):
        details.append(visit(by_order[row[0].lstrip("0")], f"order_{index:02}"))
    # Revisit the account to preserve the visible window and shopping lists.
    final = visit("/customer/account/", "final_account")
    if initial["tables"] != final["tables"] or any(
        message not in capture["body_text"]
        for message in ["You have no items to compare.", "You have no items in your wish list."]
        for capture in [initial, final]
    ):
        raise ValueError("initial/final actual order window or empty shopping lists changed")
    return initial, details, final


def main() -> None:
    args = _parse_args()
    plan = json.loads(args.task_configs.read_text())
    config = next(
        row for row in plan["tasks"] if row["generated_task_id"] == args.generated_task_id
    )
    baseline = json.loads(args.baseline_receipt.read_text())
    if (
        baseline["fresh_from_image"] is not True
        or baseline["authentication_only_preparation"] is not True
        or baseline["purpose"] != "author-grounding-only"
        or baseline["site_url"] != args.site_url
    ):
        raise ValueError("order-line author needs this fresh owned standard-auth-only baseline")
    output = args.batch_root / "author" / args.generated_task_id / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    blocked = []
    origin = urlsplit(args.site_url)

    def restrict(route):
        request = route.request
        target = urlsplit(request.url)
        same = (target.scheme, target.netloc) == (origin.scheme, origin.netloc)
        action = bool(
            re.search(
                r"/(?:reorder|cancel|pay|logout|add|remove|delete|submit|post|updatePost|addreview)(?:/|$)",
                target.path,
                re.IGNORECASE,
            )
        )
        if same and request.method in {"GET", "HEAD"} and not action:
            route.continue_()
        else:
            blocked.append({"method": request.method, "path": target.path, "same_origin": same})
            route.abort()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(
            storage_state=str(args.storage_state),
            viewport={"width": 1920, "height": 1080},
            service_workers="block",
            record_har_path=str(output / "network.har"),
            record_har_content="embed",
        )
        context.route("**/*", restrict)
        context.tracing.start(screenshots=True, snapshots=True, sources=False)
        try:
            initial, details, final = _read_window(
                context.new_page(), output, args.site_url, config
            )
            orders = parse_order_window(initial, details, config["expected_order_count"])
            definition, calculations = build_definition(config, orders)
            if any(row["same_origin"] for row in blocked):
                raise ValueError("order-only GUI unexpectedly attempted an owned business request")
        finally:
            (output / "blocked_requests.json").write_text(json.dumps(blocked, indent=2) + "\n")
            context.tracing.stop(path=str(output / "trace.zip"))
            context.close()
            browser.close()
    observation = {
        "initial_account": initial,
        "details": details,
        "final_account": final,
        "derived_answer": definition["answer"]["values"],
        "calculations": calculations,
        "visible_state_preserved": {"Recent Orders": True, "empty Wishlist/Compare": True},
        "scope": (
            "Five actual Recent Orders detail pages only; "
            "no full account-history or hidden-state claim."
        ),
    }
    (output / "observation.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
    )
    relative = output.relative_to(args.batch_root)
    evidence = [
        {
            "field": f"/task_definition/answer/values/{index}/{field}",
            "artifact": "observation",
            "location": f"/derived_answer/{index}/{field}",
            "modality": "text",
            "derivation": json.dumps(calculations[index], ensure_ascii=False),
        }
        for index, row in enumerate(definition["answer"]["values"])
        for field in row
    ]
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
                "screenshot": str(relative / "order_00.png"),
                "observation": str(relative / "observation.json"),
            },
            "field_evidence": evidence,
            "final_state_checks": [],
        },
    }
    (output / "draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({"status": "grounded", "output": str(output), "model_started": False}),
        flush=True,
    )


if __name__ == "__main__":
    main()
