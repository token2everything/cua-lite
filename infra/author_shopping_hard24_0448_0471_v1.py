#!/usr/bin/env python3
"""Capture one fixed-asset Shopping case from the frozen 0448–0471 batch.

The host supplies one fresh authentication-only baseline and a frozen config.
Selection uses the actual five order pages; catalog discovery and every business
action use normal GUI controls. This author emits original evidence, never GT,
teacher trajectories, model calls or canonical rows.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

from author_shopping_next5_order_lines_v1 import parse_order_window
from probe_shopping_review_v4 import _actual_link, _follow_visible_link, _record_ui, _review


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-configs", type=Path, required=True)
    parser.add_argument("--generated-task-id", required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--attempt", default="interaction-01")
    args = parser.parse_args()
    match = re.fullmatch(r"wavgen-v03-full300-([0-9]{4})", args.generated_task_id)
    if match is None or not 448 <= int(match[1]) <= 471:
        parser.error("choose one frozen 0448–0471 case")
    if not re.fullmatch(r"interaction-[0-9]{2}", args.attempt):
        parser.error("attempt must use the interaction two-digit namespace")
    origin = urlsplit(args.site_url)
    if (
        origin.scheme != "http"
        or not origin.netloc
        or any(
            [
                origin.path,
                origin.query,
                origin.fragment,
                origin.username,
                origin.password,
            ]
        )
    ):
        parser.error("site URL must be an explicit HTTP origin without credentials")
    return args


def select_case(policy: int, orders: list[dict], cart_skus: set[str]) -> dict:
    """Apply the 24 public policies to actual order facts using exact ratios."""
    lines = [
        {
            **line,
            "order_number": order["order_number"],
            "order_date": order["date"],
            "order_status": order["status"],
            "order_subtotal": order["subtotal"],
        }
        for order in orders
        for line in order["lines"]
    ]

    def key(line: dict) -> tuple:
        return line["unit_price"], int(line["order_number"]), line["sku"]

    def descending(line: dict) -> tuple:
        return -line["unit_price"], int(line["order_number"]), line["sku"]

    ascending = sorted(lines, key=key)
    pending = sorted(
        [order for order in orders if order["status"] == "Pending"],
        key=lambda order: (-int(order["date"].replace("-", "")), int(order["order_number"])),
    )
    canceled = sorted(
        [order for order in orders if order["status"] == "Canceled"],
        key=lambda order: (-int(order["date"].replace("-", "")), int(order["order_number"])),
    )[0]

    def highest(order: dict) -> dict:
        return sorted(
            [line for line in lines if line["order_number"] == order["order_number"]],
            key=descending,
        )[0]

    median_order = sorted(
        orders, key=lambda order: (order["grand_total"], int(order["order_number"]))
    )[2]
    outside_cart = sorted([line for line in lines if line["sku"] not in cart_skus], key=key)
    mean = sum(Fraction(line["unit_price"]) for line in lines) / len(lines)
    mean_ranked = sorted(
        lines,
        key=lambda line: (
            abs(Fraction(line["unit_price"]) - mean),
            int(line["order_number"]),
            line["sku"],
        ),
    )
    pivot = highest(canceled)
    pivot_index = ascending.index(pivot)
    review = removed = None
    shortlist = []
    if policy == 1:
        selected = [highest(order) for order in pending[:2]]
    elif policy == 2:
        selected = [
            highest(median_order),
            sorted(
                [line for line in lines if line["order_number"] == canceled["order_number"]],
                key=key,
            )[0],
        ]
    elif policy == 3:
        eligible = [line for line in lines if 10 <= line["unit_price"] <= 100]
        selected = [sorted(eligible, key=key)[0], sorted(eligible, key=descending)[0]]
    elif policy == 4:
        selected = [ascending[4], ascending[6]]
    elif policy == 5:
        selected = [ascending[pivot_index - 1], ascending[pivot_index + 1]]
    elif policy == 6:
        selected = sorted(outside_cart, key=descending)[:2]
    elif policy == 7:
        selected = [highest(pending[0])]
        review = selected[0]
    elif policy == 8:
        selected = [highest(median_order)]
        review = selected[0]
    elif policy == 9:
        selected = [
            sorted([line for line in lines if line["order_status"] == "Complete"], key=key)[2]
        ]
        review = selected[0]
    elif policy == 10:
        selected = [outside_cart[3]]
        review = selected[0]
    elif policy == 11:
        selected = [mean_ranked[0]]
        review = selected[0]
    elif policy == 12:
        selected = [
            sorted(
                lines,
                key=lambda line: (
                    -Fraction(line["line_subtotal"]) / Fraction(line["order_subtotal"]),
                    int(line["order_number"]),
                    line["sku"],
                ),
            )[0]
        ]
        review = selected[0]
    elif policy in {13, 14}:
        budget = 800 if policy == 13 else 1050
        pairs = [
            pair
            for pair in itertools.combinations(lines, 2)
            if pair[0]["order_number"] != pair[1]["order_number"]
            and sum(line["unit_price"] for line in pair) <= budget
        ]
        selected = list(
            sorted(
                pairs,
                key=lambda pair: (
                    -sum(line["unit_price"] for line in pair),
                    tuple(sorted(line["sku"] for line in pair)),
                ),
            )[0]
        )
        review = sorted(selected, key=key)[0 if policy == 13 else -1]
    elif policy == 15:
        ranked = sorted(
            orders,
            key=lambda order: (
                -Fraction(order["shipping"]) / Fraction(order["grand_total"]),
                int(order["order_number"]),
            ),
        )
        selected = [highest(order) for order in ranked[:2]]
        review = sorted(selected, key=key)[-1]
    elif policy == 16:
        selected = mean_ranked[:2]
        review = selected[0]
    elif policy == 17:
        eligible = [
            line
            for line in lines
            if line["order_status"] != "Canceled" and 10 <= line["unit_price"] <= 1000
        ]
        selected = [sorted(eligible, key=key)[0], sorted(eligible, key=descending)[0]]
        review = selected[1]
    elif policy == 18:
        selected = outside_cart[:2]
        review = selected[1]
    elif policy == 19:
        selected = [
            sorted(
                [line for line in lines if line["order_status"] == status],
                key=lambda line: (
                    Fraction(line["line_subtotal"]) / Fraction(line["order_subtotal"]),
                    int(line["order_number"]),
                    line["sku"],
                ),
            )[0]
            for status in ["Complete", "Canceled"]
        ]
        review = selected[1]
    elif policy == 20:
        nearest = sorted(
            [order for order in orders if order["status"] != "Canceled"],
            key=lambda order: (
                abs(order["grand_total"] - canceled["grand_total"]),
                int(order["order_number"]),
            ),
        )[0]
        selected = [highest(canceled), highest(nearest)]
        review = selected[1]
    elif policy == 21:
        shortlist = ascending[4:7]
        removed = sorted(shortlist, key=lambda line: (line["unit_price"], line["sku"]))[-1]
        selected = [line for line in shortlist if line["sku"] != removed["sku"]]
        review = ascending[5]
    elif policy == 22:
        shortlist = ascending[pivot_index - 1 : pivot_index + 2]
        removed = sorted(shortlist, key=lambda line: (line["unit_price"], line["sku"]))[-1]
        selected = [line for line in shortlist if line["sku"] != removed["sku"]]
        review = pivot
    elif policy == 23:
        shortlist = [
            sorted([line for line in lines if line["order_status"] == status], key=descending)[0]
            for status in ["Complete", "Canceled", "Pending"]
        ]
        removed = sorted(shortlist, key=lambda line: (line["unit_price"], line["sku"]))[-1]
        selected = [line for line in shortlist if line["sku"] != removed["sku"]]
        review = pivot
    elif policy == 24:
        shortlist = mean_ranked[:3]
        removed = sorted(
            shortlist,
            key=lambda line: (
                abs(Fraction(line["unit_price"]) - mean),
                int(line["order_number"]),
                line["sku"],
            ),
        )[-1]
        selected = [line for line in shortlist if line["sku"] != removed["sku"]]
        review = mean_ranked[0]
    else:
        raise ValueError("policy is outside the frozen 24 cases")
    return {
        "selected_lines": selected,
        "review_line": review,
        "shortlist_lines": shortlist,
        "removed_line": removed,
        "mean_price_exact": str(mean),
    }


def _discover_product(
    page, output: Path, line: dict, cart_products: dict, progress: dict, index: int
) -> dict:
    """Find the selected SKU through observed cart links or normal catalog search."""
    if line["sku"] in cart_products:
        return cart_products[line["sku"]]
    seen_products = set()
    for query_index, query in enumerate([line["sku"], line["product_name"]]):
        progress["stage"] = f"product_{index}_search_{query_index}"
        search = page.locator("#search:visible")
        search.fill(query)
        search.press("Enter")
        page.wait_for_load_state("networkidle", timeout=60_000)
        seen_pages = set()
        page_index = 0
        while True:
            record = _record_ui(page, output, f"search_{index}_{query_index}_{page_index}")
            if record["url"] in seen_pages:
                raise ValueError("normal catalog pagination repeated its actual page")
            seen_pages.add(record["url"])
            for candidate_index, card in enumerate(record["products"]):
                if card["url"] in seen_products:
                    continue
                if urlsplit(card["url"]).netloc != urlsplit(record["url"]).netloc:
                    raise ValueError("catalog card leaves the owned Shopping origin")
                seen_products.add(card["url"])
                progress["stage"] = (
                    f"product_{index}_candidate_{query_index}_{page_index}_{candidate_index}"
                )
                _follow_visible_link(page, card["url"])
                candidate = _record_ui(
                    page,
                    output,
                    f"product_{index}_candidate_{query_index}_{page_index}_{candidate_index}",
                )
                if (
                    candidate["product_detail"] is not None
                    and candidate["product_detail"]["sku"] == line["sku"]
                ):
                    return candidate
                page.goto(record["url"], wait_until="networkidle", timeout=120_000)
            following = page.locator(".pages-item-next a:visible")
            if following.count() == 0:
                break
            progress["stage"] = f"product_{index}_next_search_page"
            following.first.click()
            page.wait_for_load_state("networkidle", timeout=60_000)
            page_index += 1
    raise ValueError("selected actual order SKU was not found by normal catalog searches")


def _prepare_selection(page, output: Path, site_url: str, config: dict, progress: dict) -> dict:
    """Read the actual five-order baseline before any requested business action."""

    def visit(target: str, name: str) -> dict:
        progress["stage"] = "read_" + name
        response = page.goto(target, wait_until="networkidle", timeout=120_000)
        raw = _record_ui(page, output, name)
        if (
            response is None
            or response.status != 200
            or urlsplit(raw["url"]).path != urlsplit(target).path
        ):
            raise ValueError("actual visible resource did not load its HTTP200 page")
        return raw

    account = visit(site_url + config["observed_account_path"], "initial_account")
    table = [table for table in account["tables"] if table["id"] == "my-orders-table"]
    if len(table) != 1 or len(table[0]["rows"]) != 5:
        raise ValueError("complete initial five-order window differs from the fixed assets")
    details = []
    for index, row in enumerate(table[0]["rows"]):
        links = [
            link["url"]
            for link in account["links"]
            if link["text"] == "View Order"
            and urlsplit(link["url"]).path == f"/sales/order/view/order_id/{int(row[0])}/"
        ]
        if len(links) != 1:
            raise ValueError("each order requires its actual unique View Order link")
        details.append(visit(links[0], f"order_{index:02}"))
    orders = parse_order_window(account, details, 5)
    if len({line["sku"] for order in orders for line in order["lines"]}) != 11:
        raise ValueError("the original eleven-SKU order scope differs")
    resource_urls = {
        "cart": _actual_link(account, site_url, path=config["observed_cart_path"]),
        "wishlist": _actual_link(account, site_url, path=config["observed_wishlist_path"]),
        "reviews": _actual_link(account, site_url, text="My Product Reviews"),
    }
    initial = {name: visit(url, "initial_" + name) for name, url in resource_urls.items()}
    if len(initial["cart"]["cart_items"]) != 3:
        raise ValueError("original three-item cart differs")
    if "You have no items in your wish list." not in initial["wishlist"]["body_text"]:
        raise ValueError("initial Wishlist is not the original empty baseline")
    if "You have no items to compare." not in account["body_text"]:
        raise ValueError("initial Compare is not the original empty baseline")
    if "You have submitted no reviews." not in initial["reviews"]["body_text"]:
        raise ValueError("initial private Reviews are not the original empty baseline")
    cart_products = {}
    for index, item in enumerate(initial["cart"]["cart_items"]):
        product = visit(item["url"], f"cart_product_{index:02}")
        cart_products[product["product_detail"]["sku"]] = product
    if len(cart_products) != 3:
        raise ValueError("actual cart product SKUs do not form three distinct identities")
    selected = select_case(config["policy"], orders, set(cart_products))
    (output / "selection_policy_facts.json").write_text(
        json.dumps(
            {
                "orders": orders,
                "cart_SKUs": sorted(cart_products),
                "policy": config["policy"],
                "public_query": config["public_query"],
                "selection": selected,
            },
            default=str,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    all_needed = selected["shortlist_lines"] or selected["selected_lines"]
    products = {}
    for index, line in enumerate(all_needed):
        products[line["sku"]] = _discover_product(
            page, output, line, cart_products, progress, index
        )
    return {
        "account": account,
        "order_details": details,
        "orders": orders,
        **initial,
        "resource_urls": resource_urls,
        "selection": selected,
        "products": products,
    }


def _check_comparison(raw: dict, products: dict, expected_skus: list[str]) -> list[dict]:
    """Join actual SKU columns to their own current catalog names and prices."""
    tables = [
        table
        for table in raw["interaction_visible"]["table_details"]
        if table["id"] == "product-comparison"
    ]
    if len(tables) != 1:
        raise ValueError("actual Compare table is absent or repeated")
    sku_rows = [
        row
        for row in tables[0]["rows"]
        if row["cells"] and row["cells"][0]["tag"] == "th" and row["cells"][0]["text"] == "SKU"
    ]
    product_rows = [
        row
        for row in tables[0]["rows"]
        if row["cells"] and row["cells"][0]["tag"] == "th" and row["cells"][0]["text"] == "Product"
    ]
    if len(sku_rows) != 1 or len(product_rows) != 1:
        raise ValueError("actual Compare SKU/Product rows are not unique")
    sku_cells = sku_rows[0]["cells"][1:]
    product_cells = product_rows[0]["cells"][1:]
    actual = [cell["text"] for cell in sku_cells]
    if (
        len(actual) != len(expected_skus)
        or set(actual) != set(expected_skus)
        or len(product_cells) != len(actual)
    ):
        raise ValueError("actual Compare columns differ from the requested exact membership")
    columns = []
    for sku, cell in zip(actual, product_cells, strict=True):
        detail = products[sku]["product_detail"]
        target_path = urlsplit(products[sku]["url"]).path
        named_links = [
            link
            for link in cell["links"]
            if urlsplit(link["url"]).path == target_path and link["text"] == detail["name"]
        ]
        prices = [
            text
            for text in cell["text"].splitlines()
            if re.fullmatch(r"\$[0-9,]+\.[0-9]{2}", text.strip())
        ]
        if len(named_links) != 1 or detail["price"] not in prices:
            raise ValueError(
                "Compare SKU column does not retain its own current product name/price"
            )
        columns.append(
            {
                "sku": sku,
                "current_product_name": detail["name"],
                "current_product_price": detail["price"],
                "actual_product_link": named_links[0],
                "actual_product_cell": cell,
            }
        )
    return columns


def _open_comparison(
    page, output: Path, products: dict, expected_skus: list[str], progress: dict, name: str
) -> dict:
    """Follow a captured comparison view and record its reload without URL presets."""
    landing = _record_ui(page, output, name + "_entry")
    entries = []
    origin = urlsplit(landing["url"])
    for link in landing["links"]:
        target = urlsplit(link["url"])
        if (
            (target.scheme, target.netloc) == (origin.scheme, origin.netloc)
            and (
                target.path == "/catalog/product_compare/"
                or target.path.startswith("/catalog/product_compare/index/")
            )
            and "#" not in link["url"]
            and "delete" not in link["class_name"].split()
            and link["url"] not in entries
        ):
            entries.append(link["url"])
    if not entries:
        raise ValueError("actual comparison view link is absent")
    progress["stage"] = name + "_entry_click"
    _follow_visible_link(page, entries[0])
    initial = _record_ui(page, output, name)
    initial_columns = _check_comparison(initial, products, expected_skus)
    progress["stage"] = name + "_reload"
    page.reload(wait_until="networkidle", timeout=120_000)
    refreshed = _record_ui(page, output, name + "_refreshed")
    refreshed_columns = _check_comparison(refreshed, products, expected_skus)
    return {
        "status": "actual_comparison_and_reload_observed",
        "url": refreshed["url"],
        "initial_columns": initial_columns,
        "refreshed_columns": refreshed_columns,
        "column_order_constraint": False,
        "price_binding": "current product page, not historical order Price",
    }


def _add_compare(
    page, output: Path, product: dict, index: int, progress: dict, operation: dict
) -> None:
    progress["stage"] = f"read_product_for_compare_{index}"
    page.goto(product["url"], wait_until="networkidle", timeout=120_000)
    raw = _record_ui(page, output, f"before_compare_add_{index}")
    controls = raw["interaction_visible"]["compare_controls"]
    buttons = page.locator("a.action.tocompare:visible")
    if len(controls) != 1 or buttons.count() != 1:
        raise ValueError("one normal visible Compare control is required")
    posted = json.loads(controls[0]["data_post"])
    action = urlsplit(posted["action"])
    product_origin = urlsplit(product["url"])
    if (action.scheme, action.netloc) != (
        product_origin.scheme,
        product_origin.netloc,
    ) or action.path != "/catalog/product_compare/add/":
        raise ValueError("actual Compare form action differs")
    event = {
        "action": "click_visible_Add_to_Compare",
        "sku": product["product_detail"]["sku"],
        "actual_control": controls[0],
        "completion": "attempting_click",
    }
    progress["business_actions"].append(event)
    progress["stage"] = f"compare_add_{index}_click"
    operation.update({"active": f"compare_add_{index}", "permitted_path": action.path})
    try:
        buttons.click()
        event["completion"] = "click_returned"
        page.wait_for_load_state("networkidle", timeout=60_000)
        event["completion"] = "click_and_networkidle_returned"
    finally:
        operation.update({"active": None, "permitted_path": None})
    _record_ui(page, output, f"after_compare_add_{index}")


def _remove_compare(
    page, output: Path, sku: str, products: dict, progress: dict, operation: dict
) -> None:
    """Click the SKU column's actual Remove and its visible confirmation once."""
    raw = _record_ui(page, output, "before_compare_remove")
    table = next(
        table
        for table in raw["interaction_visible"]["table_details"]
        if table["id"] == "product-comparison"
    )
    sku_row = next(row for row in table["rows"] if row["cells"][0]["text"] == "SKU")
    column = [cell["text"] for cell in sku_row["cells"][1:]].index(sku)
    control = (
        page.locator("#product-comparison > tbody > tr")
        .filter(has=page.locator('th:text-is("Remove Product")'))
        .locator("td")
        .nth(column)
        .locator("a.action.delete:visible")
    )
    if control.count() != 1:
        raise ValueError("selected SKU has no unique normal Remove control")
    posted = json.loads(control.get_attribute("data-post"))
    action = urlsplit(posted["action"])
    origin = urlsplit(products[sku]["url"])
    if (action.scheme, action.netloc) != (
        origin.scheme,
        origin.netloc,
    ) or action.path != "/catalog/product_compare/remove/":
        raise ValueError("actual Remove action differs from the owned comparison")
    if posted["data"]["confirmation"] is not True:
        raise ValueError("actual Remove control does not expose its captured confirmation contract")
    event = {
        "action": "click_visible_Remove_Product_and_confirmation",
        "sku": sku,
        "actual_column": column,
        "actual_data_post": posted,
        "completion": "attempting_click",
    }
    progress["business_actions"].append(event)
    operation.update({"active": "compare_remove", "permitted_path": action.path})
    progress["stage"] = "compare_remove_control_click"
    try:
        control.click()
        event["completion"] = "Remove_control_clicked"
        _record_ui(page, output, "actual_remove_confirmation")
        dialog = page.locator(".modal-popup:visible")
        if dialog.count() != 1 or posted["data"]["confirmationMessage"] not in dialog.inner_text():
            raise ValueError("normal Remove confirmation differs; original capture remains pending")
        confirm = dialog.get_by_role("button", name="OK", exact=True)
        if confirm.count() != 1:
            raise ValueError("actual Remove confirmation has no unique visible OK button")
        progress["stage"] = "compare_remove_confirmation_click"
        confirm.click()
        event["completion"] = "normal_confirmation_clicked"
        page.wait_for_load_state("networkidle", timeout=60_000)
        event["completion"] = "click_and_networkidle_returned"
    finally:
        operation.update({"active": None, "permitted_path": None})
    _record_ui(page, output, "after_compare_remove")


def _capture(
    page, output: Path, site_url: str, config: dict, progress: dict, operation: dict
) -> dict:
    facts = _prepare_selection(page, output, site_url, config, progress)
    selection = facts["selection"]
    findings = {}
    if config["mode"] != "review":
        additions = selection["shortlist_lines"] or selection["selected_lines"]
        for index, line in enumerate(additions, 1):
            _add_compare(page, output, facts["products"][line["sku"]], index, progress, operation)
        findings["compare_before_remove"] = _open_comparison(
            page,
            output,
            facts["products"],
            [line["sku"] for line in additions],
            progress,
            "compare_table",
        )
        if selection["removed_line"] is not None:
            _remove_compare(
                page,
                output,
                selection["removed_line"]["sku"],
                facts["products"],
                progress,
                operation,
            )
            findings["compare_after_remove"] = _open_comparison(
                page,
                output,
                facts["products"],
                [line["sku"] for line in selection["selected_lines"]],
                progress,
                "compare_retained",
            )
    if selection["review_line"] is not None:
        line = selection["review_line"]
        review_config = {
            "review_stars": 4,
            "review_summary": config["review_summary"],
            "review_text": (
                f"Order {line['order_number']}; SKU {line['sku']}; "
                f"unit price ${line['unit_price']:.2f}. Four-star order note."
            ),
        }
        findings["review"] = _review(
            page,
            output,
            {"selected": [facts["products"][line["sku"]]], "resource_urls": facts["resource_urls"]},
            review_config,
            progress,
            operation,
        )
        (output / "review_requested_fields.json").write_text(
            json.dumps(review_config, ensure_ascii=False, indent=2) + "\n"
        )
        (output / "business_findings.json").write_text(
            json.dumps(findings, ensure_ascii=False, indent=2) + "\n"
        )
        if findings["review"]["status"] != "review_account_persistence_observed":
            raise ValueError(
                "original normal Review capture did not prove requested private persistence"
            )
    final = {}
    for name, url in facts["resource_urls"].items():
        progress["stage"] = "read_final_" + name
        page.goto(url, wait_until="networkidle", timeout=120_000)
        final[name] = _record_ui(page, output, "final_" + name)
    page.goto(site_url + config["observed_account_path"], wait_until="networkidle", timeout=120_000)
    final_account = _record_ui(page, output, "final_account")
    final_details = []
    for index, detail in enumerate(facts["order_details"]):
        page.goto(detail["url"], wait_until="networkidle", timeout=120_000)
        final_details.append(_record_ui(page, output, f"final_order_{index:02}"))
    preserved = {
        "cart_visible_lines": facts["cart"]["cart_items"] == final["cart"]["cart_items"],
        "Wishlist_visible_empty": "You have no items in your wish list."
        in final["wishlist"]["body_text"],
        "Recent_Orders_window": facts["account"]["tables"] == final_account["tables"],
        "all_observed_order_lines_and_totals": facts["orders"]
        == parse_order_window(final_account, final_details, 5),
    }
    if config["mode"] == "compare":
        preserved["private_Reviews_unchanged"] = (
            facts["reviews"]["tables"] == final["reviews"]["tables"]
        )
    else:
        review_tables = [
            table for table in final["reviews"]["tables"] if table["id"] == "my-reviews-table"
        ]
        preserved["exactly_one_private_Review_row"] = (
            len(review_tables) == 1 and len(review_tables[0]["rows"]) == 1
        )
    if config["mode"] == "review":
        preserved["Compare_unchanged_empty"] = (
            "You have no items to compare." in final_account["body_text"]
        )
    else:
        findings["final_compare"] = _open_comparison(
            page,
            output,
            facts["products"],
            [line["sku"] for line in selection["selected_lines"]],
            progress,
            "final_compare",
        )
    if not all(preserved.values()):
        raise ValueError("captured untargeted collections or requested persistence differ")
    if "review" in findings:
        page.goto(findings["review"]["detail_url"], wait_until="networkidle", timeout=120_000)
        page.reload(wait_until="networkidle", timeout=120_000)
    terminal = _record_ui(page, output, "terminal")
    if "review" in findings:
        requested = [
            review_config["review_summary"],
            review_config["review_text"],
            facts["products"][selection["review_line"]["sku"]]["product_detail"]["name"],
        ]
        rating_titles = [
            title
            for row in terminal["interaction_visible"]["ratings"]
            for title in row["visible_titles"]
        ]
        if (
            not all(value in terminal["body_text"] for value in requested)
            or "80%" not in rating_titles
        ):
            raise ValueError(
                "final requested review detail reload differs from the captured Review"
            )
    return {
        "status": "actual_author_capture_complete_pending_full_GT",
        "generated_task_id": config["generated_task_id"],
        "family": config["family"],
        "mode": config["mode"],
        "public_query": config["public_query"],
        "selection": selection,
        "selected_current_product_facts": {
            sku: raw["product_detail"] for sku, raw in facts["products"].items()
        },
        "findings": findings,
        "visible_state_preserved": preserved,
        "terminal_url": terminal["url"],
        "preservation_scope": (
            "Actual visible collections and original five-order details only, "
            "not a full DB invariant."
        ),
        "GT": 0,
        "drafts": 0,
        "model_calls": 0,
        "canonical_append": False,
    }


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
        or baseline["purpose"] != "author-interaction-probe"
        or baseline["site_url"] != args.site_url
        or baseline["model_started"] is not False
    ):
        raise ValueError("this capture requires its independent fresh authentication-only baseline")
    output = args.batch_root / "exploration/shopping" / args.generated_task_id / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (output / "frozen_public_query.txt").write_text(config["public_query"] + "\n")
    progress = {
        "status": "running",
        "stage": "prepare_selection",
        "mode": config["mode"],
        "business_actions": [],
        "gui_actions": [],
        "model_calls": 0,
        "drafts": 0,
        "GT": 0,
    }
    operation = {"active": None, "permitted_path": None}
    blocked, business_responses, passive_requests = [], [], []
    origin = urlsplit(args.site_url)

    def restrict(route):
        request = route.request
        target = urlsplit(request.url)
        same = (target.scheme, target.netloc) == (origin.scheme, origin.netloc)
        read = request.method in {"GET", "HEAD"}
        passive = False
        if request.method == "POST" and target.path in plan["passive_gui_posts"]:
            try:
                body = json.loads(request.post_data)
            except (json.JSONDecodeError, TypeError):
                body = None
            passive = (
                not target.query
                and not target.fragment
                and body == plan["passive_gui_posts"][target.path]
            )
            passive_requests.append(
                {
                    "path": target.path,
                    "raw_post_data": request.post_data,
                    "same_origin": same,
                    "exact_original_default_payload": passive,
                    "permitted": same and passive,
                    "phase": progress["stage"],
                    "active_gui_action": operation["active"],
                }
            )
        write = (
            request.method == "POST"
            and operation["active"] is not None
            and target.path == operation["permitted_path"]
        )
        action = bool(
            re.search(
                r"/(?:add|addgroup|post|submit|reorder|cancel|pay|logout|remove|clear|delete|updatePost)(?:/|$)",
                target.path,
                re.IGNORECASE,
            )
        )
        if same and ((read and not action) or passive or write):
            route.continue_()
        else:
            blocked.append(
                {
                    "method": request.method,
                    "path": target.path,
                    "same_origin": same,
                    "active_gui_action_at_request": operation["active"],
                    "phase_at_request": progress["stage"],
                    "raw_post_data": request.post_data,
                }
            )
            route.abort()

    def record_response(response):
        request = response.request
        if request.method not in {"GET", "HEAD"} and urlsplit(request.url).netloc == origin.netloc:
            business_responses.append(
                {
                    "url": request.url,
                    "method": request.method,
                    "http_status": response.status,
                    "raw_post_data": request.post_data,
                    "active_gui_action_at_response": operation["active"],
                    "phase_at_response": progress["stage"],
                }
            )

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
        context.set_default_timeout(60_000)
        context.route("**/*", restrict)
        context.on("response", record_response)
        trace_started = False
        try:
            context.tracing.start(screenshots=True, snapshots=True, sources=False)
            trace_started = True
            page = context.new_page()
            report = _capture(page, output, args.site_url, config, progress, operation)
            blocked_owned = [row for row in blocked if row["same_origin"]]
            report.update(
                {
                    "blocked_owned_requests": blocked_owned,
                    "observed_non_GET_responses": business_responses,
                    "passive_gui_calculation_requests": passive_requests,
                    "formal_admission": (
                        "Pending original complete compiler/replay/native/negative/contamination/"
                        "root gates."
                    ),
                }
            )
            if blocked_owned:
                report["status"] = "pending_blocked_owned_request"
            (output / "observation.json").write_text(
                json.dumps(report, default=str, ensure_ascii=False, indent=2) + "\n"
            )
            if blocked_owned:
                raise ValueError(
                    "original owned GUI request was blocked; preserve original evidence"
                )
            progress.update(
                {"status": "completed_with_actual_findings", "stage": "findings_recorded"}
            )
        except Exception as error:
            progress.update(
                {
                    "status": "original_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        finally:
            (output / "progress.json").write_text(
                json.dumps(progress, ensure_ascii=False, indent=2) + "\n"
            )
            (output / "blocked_requests.json").write_text(json.dumps(blocked, indent=2) + "\n")
            (output / "observed_non_GET_responses.json").write_text(
                json.dumps(business_responses, indent=2) + "\n"
            )
            (output / "passive_gui_calculation_requests.json").write_text(
                json.dumps(passive_requests, indent=2) + "\n"
            )
            try:
                if trace_started:
                    context.tracing.stop(path=str(output / "trace.zip"))
            except Exception as error:
                progress.update(
                    {
                        "status": "original_error",
                        "trace_finalization_error": {
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    }
                )
                (output / "progress.json").write_text(
                    json.dumps(progress, ensure_ascii=False, indent=2) + "\n"
                )
                raise
            finally:
                try:
                    context.close()
                finally:
                    browser.close()
    print(
        json.dumps({"status": progress["status"], "output": str(output), "model_started": False}),
        flush=True,
    )


if __name__ == "__main__":
    main()
