"""Entity selection and author projections use offline evidence without site or model calls."""

from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import author_cross_site
import pytest
from author_cross_site import ground_task, select_entity
from cross_site_evaluator import derive_value, parse_fact_value

pytest_plugins = ["test_episode_applicability"]


@pytest.fixture
def products():
    return [
        {
            "title": "Beta Bag",
            "url": "http://shopping.test/beta.html",
            "price": "$20.00",
            "quantity": "2",
            "subtotal": "$40.00",
            "rating": "80%",
            "reviews": "12 Reviews",
        },
        {
            "title": "Alpha Bag",
            "url": "http://shopping.test/alpha.html",
            "price": "$20.00",
            "quantity": "1",
            "subtotal": "$20.00",
            "rating": "90%",
            "reviews": "2 Reviews",
        },
        {
            "title": "Travel Bag",
            "url": "http://shopping.test/travel.html",
            "price": "$35.00",
            "quantity": "1",
            "subtotal": "$35.00",
            "rating": "95%",
            "reviews": "1 Review",
        },
    ]


@pytest.fixture
def observed_showcase():
    # Exact entity-field projection of v04explore20261012d/shopping_000.json.
    # Source SHA256: f2ec8d6519ba537ce26e41551ec803027543f87a856abd0d64ebf5e839833923.
    return [
        {
            "title": "Pre-baked Gingerbread House Kit Value Pack, 17 oz., Pack of 2, Total 34 oz.",
            "url": "http://shopping.localhost:7760/pre-baked-gingerbread-house-kit-value-pack-17-oz-pack-of-2-total-34-oz.html",
            "price": "$19.99",
            "rating": "20%",
            "reviews": "1 \xa0Review",
        },
        {
            "title": "V8 +Energy, Healthy Energy Drink, Steady Energy from Black and Green Tea, "
            "Pomegranate Blueberry, 8 Ounce Can ,Pack of 24",
            "url": "http://shopping.localhost:7760/v8-energy-healthy-energy-drink-steady-energy-from-black-and-green-tea-pomegranate-blueberry-8-ounce-can-pack-of-24.html",
            "price": "$14.47",
            "rating": "57%",
            "reviews": "12 \xa0Reviews",
        },
        {
            "title": "Elmwood Inn Fine Teas, Orange Vanilla Caffeine-free Fruit Infusion, 16-Ounce Pouch",
            "url": "http://shopping.localhost:7760/elmwood-inn-fine-teas-orange-vanilla-caffeine-free-fruit-infusion-16-ounce-pouch.html",
            "price": "$19.36",
            "rating": "95%",
            "reviews": "4 \xa0Reviews",
        },
        {
            "title": "Belle Of The Ball Princess Sprinkle Mix| Wedding Colorful Sprinkles| Cake Cupcake "
            "Cookie Sprinkles| Ice cream Candy Sprinkles| Yellow Gold Red Royal Red Rose Icing "
            "Flowers Decorating Sprinkles, 8OZ",
            "url": "http://shopping.localhost:7760/belle-of-the-ball-princess-sprinkle-mix-wedding-colorful-sprinkles-cake-cupcake-cookie-sprinkles-ice-cream-candy-sprinkles-yellow-gold-red-royal-red-rose-icing-flowers-decorating-sprinkles-8oz.html",
            "price": "$23.50",
            "rating": "63%",
            "reviews": "12 \xa0Reviews",
        },
        {
            "title": "So Delicious Dairy Free CocoWhip Light, Vegan, Non-GMO Project Verified, 9 oz. Tub",
            "url": "http://shopping.localhost:7760/so-delicious-dairy-free-cocowhip-light-vegan-non-gmo-project-verified-9-oz-tub.html",
            "price": "$15.62",
            "rating": "78%",
            "reviews": "12 \xa0Reviews",
        },
        {
            "title": "Cheongeun Sweet Potato Starch Powder 500g, 2ea(Sweet Potato 55%, Corn 45%)",
            "url": "http://shopping.localhost:7760/cheongeun-sweet-potato-starch-powder-500g-2ea-sweet-potato-55-corn-45.html",
            "price": "$34.00",
            "rating": None,
            "reviews": None,
        },
        {
            "title": "Q Mixers Premium Ginger Ale: Real Ingredients & Less Sweet, 6.7 Fl Oz (24 Bottles)",
            "url": "http://shopping.localhost:7760/q-mixers-premium-ginger-ale-real-ingredients-less-sweet-6-7-fl-oz-24-bottles.html",
            "price": "$68.50",
            "rating": "88%",
            "reviews": "12 \xa0Reviews",
        },
        {
            "title": "Stove Top Turkey Stuffing Mix (12 oz Boxes, Pack of 2)",
            "url": "http://shopping.localhost:7760/stove-top-turkey-stuffing-mix-12-oz-boxes-pack-of-2.html",
            "price": "$8.49",
            "rating": "85%",
            "reviews": "12 \xa0Reviews",
        },
        {
            "title": "Briess DME - Pilsen Light - 1 lb Bag",
            "url": "http://shopping.localhost:7760/briess-dme-pilsen-light-1-lb-bag.html",
            "price": "$12.99",
            "rating": None,
            "reviews": None,
        },
        {
            "title": "Tony Chachere's More Spice Creole Seasoning - 14 oz",
            "url": "http://shopping.localhost:7760/tony-chachere-s-more-spice-creole-seasoning-14-oz.html",
            "price": "$7.92",
            "rating": "75%",
            "reviews": "12 \xa0Reviews",
        },
        {
            "title": "Lobster Cobbler Pot Pie - Gourmet Frozen Seafood Appetizers (Set of 8 Trays)",
            "url": "http://shopping.localhost:7760/lobster-cobbler-pot-pie-gourmet-frozen-seafood-appetizers-set-of-8-trays.html",
            "price": "$411.76",
            "rating": None,
            "reviews": None,
        },
        {
            "title": "Crunchy Rice Rollers - Gluten Free - Vegan - 3.5 oz Individual Packs (4 Packs of 8 "
            "Rollers)",
            "url": "http://shopping.localhost:7760/crunchy-rice-rollers-gluten-free-vegan-3-5-oz-individual-packs-4-packs-of-8-rollers.html",
            "price": "$11.50",
            "rating": "83%",
            "reviews": "12 \xa0Reviews",
        },
    ]


@pytest.fixture
def observed_orders():
    # Exact visible Recent Orders projection from the initial Shopping account.
    return [
        {
            "title": "000000170",
            "url": "http://shopping.localhost:7760/sales/order/view/order_id/170/",
            "total": "$365.42",
            "status": "Canceled",
        },
        {
            "title": "000000189",
            "url": "http://shopping.localhost:7760/sales/order/view/order_id/189/",
            "total": "$754.99",
            "status": "Pending",
        },
        {
            "title": "000000188",
            "url": "http://shopping.localhost:7760/sales/order/view/order_id/188/",
            "total": "$2,004.99",
            "status": "Pending",
        },
        {
            "title": "000000187",
            "url": "http://shopping.localhost:7760/sales/order/view/order_id/187/",
            "total": "$1,004.99",
            "status": "Pending",
        },
        {
            "title": "000000180",
            "url": "http://shopping.localhost:7760/sales/order/view/order_id/180/",
            "total": "$65.32",
            "status": "Complete",
        },
    ]


@pytest.mark.parametrize(
    ("direction", "rank", "expected_title", "expected_total"),
    [
        ("asc", 1, "000000189", "$754.99"),
        ("desc", 1, "000000188", "$2,004.99"),
        ("desc", 2, "000000187", "$1,004.99"),
    ],
)
def test_recent_orders_filter_status_then_rank_exact_total(
    observed_orders, direction, rank, expected_title, expected_total
):
    selected = select_entity(
        observed_orders,
        {
            "kind": "rank", "rank_by": "total",
            "direction": direction,
            "rank": rank,
            "tie_break": "title_ascending",
            "filters": [{"field": "status", "op": "eq", "value": "Pending"}],
        },
    )
    assert selected["entity"]["title"] == expected_title
    assert selected["fact"] == expected_total
    assert set(selected["eligible_indices"]) == {1, 2, 3}
    assert selected["fact_field"] == "total"


def test_missing_queried_order_status_excludes_the_entity(observed_orders):
    population = [dict(row) for row in observed_orders]
    population[2]["status"] = None
    selected = select_entity(
        population,
        {
            "kind": "rank", "rank_by": "total",
            "direction": "desc",
            "rank": 1,
            "tie_break": "title_ascending",
            "filters": [{"field": "status", "op": "eq", "value": "Pending"}],
        },
    )
    assert selected["entity"]["title"] == "000000187"
    assert 2 not in selected["eligible_indices"]


def test_order_status_equality_uses_the_visible_string(observed_orders):
    with pytest.raises(ValueError, match="too few qualifying"):
        select_entity(
            observed_orders,
            {
                "kind": "rank", "rank_by": "total",
                "direction": "asc",
                "rank": 1,
                "tie_break": "title_ascending",
                "filters": [{"field": "status", "op": "eq", "value": "pending"}],
            },
        )


def test_order_status_fact_keeps_the_original_string_after_total_rank(observed_orders):
    selected = select_entity(
        observed_orders,
        {
            "kind": "rank", "rank_by": "total",
            "fact_field": "status",
            "direction": "desc",
            "rank": 1,
            "tie_break": "title_ascending",
            "filters": [{"field": "total", "op": "lte", "value": "500"}],
        },
    )
    assert selected["entity"]["title"] == "000000170"
    assert selected["fact"] == "Canceled"
    assert selected["fact_field"] == "status"
    population = [dict(row) for row in observed_orders]
    population[0]["status"] = None
    without_status = select_entity(
        population,
        {
            "kind": "rank", "rank_by": "total",
            "fact_field": "status",
            "direction": "desc",
            "rank": 1,
            "tie_break": "title_ascending",
            "filters": [{"field": "total", "op": "lte", "value": "500"}],
        },
    )
    assert without_status["entity"]["title"] == "000000180"
    assert without_status["fact"] == "Complete"


def test_order_total_does_not_supply_a_missing_price(observed_orders):
    with pytest.raises(ValueError, match="too few qualifying"):
        select_entity(
            observed_orders,
            {"kind": "rank", "rank_by": "price", "direction": "asc", "rank": 1, "tie_break": "title_ascending"},
        )


@pytest.mark.parametrize("internal_note", [False, True])
def test_author_reads_response_identity_and_emits_exact_preservation_checks(
    tmp_path, monkeypatch, runtime, observed_orders, internal_note
):
    """Exercise the producer across simulated browser/network boundaries, without sites."""
    from playwright import sync_api
    from webarena_verified.core.utils import network_event_utils

    sites = {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"}
    home_url = sites["shopping"] + "/"
    cart_url = sites["shopping"] + "/checkout/cart/"
    orders_url = sites["shopping"] + "/customer/account/?p=1"
    issue_url = sites["gitlab"] + "/byteblaze/a11y-webring.club/-/issues/2"
    create_note_url = sites["gitlab"] + (
        "/byteblaze/a11y-webring.club/notes?target_id=83755&target_type=issue"
    )
    note_body = "Order 000000187: Pending"
    identity = {
        "id": 83755,
        "iid": 2,
        "title": "Thoughts on line highlight?",
        "web_url": "/byteblaze/a11y-webring.club/-/issues/2",
        "create_note_path": create_note_url.removeprefix(sites["gitlab"]),
    }
    raw_metadata = json.dumps(identity).replace('"', "&quot;")
    target_response = MagicMock(ok=True)
    target_response.text.return_value = (
        '<div id="js-vue-notes" data-noteable-type="Issue" data-noteable-data="'
        + raw_metadata
        + '"></div>'
    )
    page, context, browser, manager = (MagicMock() for _ in range(4))
    manager.__enter__.return_value.chromium.launch.return_value = browser
    browser.new_context.return_value = context
    context.new_page.return_value = page
    page.locator.return_value.count.return_value = 0
    page.get_by_test_id.return_value.count.return_value = 1

    def goto(url, **kwargs):
        page.url = url
        return target_response if url == issue_url else MagicMock()

    page.goto.side_effect = goto
    save_response = SimpleNamespace(
        url=create_note_url,
        request=SimpleNamespace(method="POST"),
        status=200,
        ok=True,
    )
    page.expect_response.return_value.__enter__.return_value.value = save_response
    monkeypatch.setattr(sync_api, "sync_playwright", lambda: manager)
    readouts = {
        "title": {"kind": "text", "selector": "td.col.id", "value_type": "string"},
        "total": {"kind": "text", "selector": "td.col.total .price", "value_type": "string"},
        "status": {"kind": "text", "selector": "td.col.status", "value_type": "string"},
    }
    orders = [
        {
            **row,
            "url": row["url"].replace("http://shopping.localhost:7760", sites["shopping"]),
            "visible_text": row["title"] + "\n" + row["total"] + "\n" + row["status"],
            "readouts": readouts,
        }
        for row in observed_orders
    ]
    cart = [
        {
            "title": "Preserved cart item",
            "url": sites["shopping"] + "/cart-item.html",
            "quantity": "1",
            "subtotal": "$40.99",
            "readouts": {
                "quantity": {
                    "kind": "input_value",
                    "selector": "input[data-role=cart-item-qty]",
                    "value_type": "integer",
                },
                "subtotal": {
                    "kind": "text",
                    "selector": ".col.subtotal .price",
                    "value_type": "string",
                },
            },
        }
    ]
    old_notes = [
        {"text": "Love the idea. I'll see what I can do!", "id": "note_306775"},
        {"text": "Just added support!", "id": "note_306772"},
    ]
    issue_state = {
        "title": identity["title"],
        "description": "Original description",
        "assignee": None,
        "labels": [],
        "open": False,
        "checklist": [],
    }

    def observe(page, output, name, site, **kwargs):
        if site == "gitlab":
            return {
                "url": issue_url,
                "issue_state": deepcopy(issue_state),
                "issue_identity": {
                    "project_path": "byteblaze/a11y-webring.club",
                    "issue_iid": "2",
                },
                "notes": deepcopy(old_notes)
                + ([] if name == "target_before" else [{"text": note_body, "id": "note_999999"}]),
            }
        return {
            "url": page.url,
            "links": [
                {"title": "My Cart\n1 item", "url": cart_url},
                {"title": "My Account", "url": orders_url},
            ],
            "cart_items": deepcopy(cart) if name == "preservation_cart" else [],
            "orders": deepcopy(orders) if page.url == orders_url else [],
        }

    monkeypatch.setattr(author_cross_site, "record_observation", observe)

    def seal(output, *args):
        (output / "episode_boundary.json").write_text('{"scope":"offline-producer-test"}')

    monkeypatch.setattr("cross_site_evaluator.seal_recording", seal)
    source_event = SimpleNamespace(url=orders_url, http_method="GET", request_status=200)
    save_event = SimpleNamespace(
        url=create_note_url,
        http_method="POST",
        request_status=200,
        is_request_success=True,
        post_data={
            "note": {
                "note": note_body,
                "noteable_id": 83755,
                "noteable_type": "Issue",
                "internal": internal_note,
            }
        },
    )
    monkeypatch.setattr(network_event_utils, "load_har_trace", lambda path: [source_event, save_event])
    config = {
        "generated_task_id": "wavgen-author-orders-test",
        "task_id": 7000001,
        "intent_template_id": 90001,
        "revision": 1,
        "source_artifact": "abstract-family",
        "source_record_id": "offline-producer-test",
        "family": "order-status-to-issue-note",
        "source": {
            "kind": "orders",
            "navigation": ["__SHOPPING__/customer/account/?p=1"],
            "expected_count": 5,
            "max_pages": 1,
        },
        "selection": {
            "kind": "rank", "rank_by": "total",
            "fact_field": "status",
            "direction": "desc",
            "rank": 2,
            "tie_break": "title_ascending",
            "filters": [{"field": "status", "op": "eq", "value": "Pending"}],
        },
        "target": {"issue_url": issue_url, "title": identity["title"]},
        "parameters": {},
        "note_template": "Order {selected_entity_title}: {selected_fact}",
        "goal_template": "Select the order and add the required note; preserve existing state.",
    }
    baseline = {
        "fresh_from_image": True,
        "authentication_only_preparation": True,
        "sites": {site: {"site_url": url} for site, url in sites.items()},
    }
    for name, value in [("config.json", config), ("baseline.json", baseline), ("sites.json", sites)]:
        (tmp_path / name).write_text(json.dumps(value))
    args = SimpleNamespace(
        task_config=tmp_path / "config.json",
        site_urls=tmp_path / "sites.json",
        baseline_receipt=tmp_path / "baseline.json",
        storage_state=tmp_path / "storage.json",
        generated_task_id=config["generated_task_id"],
        batch_root=tmp_path,
        attempt="grounding-01",
        timeout_ms=100,
        settle_ms=0,
    )
    if internal_note:
        with pytest.raises(ValueError, match="ordinary, non-internal"):
            ground_task(args)
        return
    draft = ground_task(args)
    completion = draft["task_definition"]["completion_events"][1]
    assert completion["post_data"]['$["note"]["internal"]'] is False
    assert completion["post_data_schema"]["properties"]['$["note"]["internal"]'] == {
        "type": "boolean"
    }
    # Compile the actual producer output, then exercise the frozen official
    # evaluator. Hyphens, dots and query delimiters remain literal URL bytes.
    from author import _task_spec
    from webarena_verified.types import WebArenaSite, WebArenaVerifiedTask
    from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
    from webarena_verified.types.tracing import NetworkTrace

    compiled, reference_response = _task_spec(draft)
    task = WebArenaVerifiedTask.model_validate(compiled)
    evaluator_config = WebArenaVerifiedConfig(
        test_data_file=tmp_path / "private.jsonl",
        environments={WebArenaSite(site): EnvironmentConfig(urls=[url]) for site, url in sites.items()},
    )
    assert task.network_event_evaluator_cfgs[0].expected.url == (
        "__SHOPPING__/customer/account/?p=1"
    )
    assert task.network_event_evaluator_cfgs[1].expected.url == (
        "__GITLAB__/byteblaze/a11y-webring.club/notes?target_id=83755&target_type=issue"
    )
    entries = []
    for event in (source_event, save_event):
        request = {"method": event.http_method, "url": event.url, "headers": [], "queryString": []}
        if event.http_method == "POST":
            request["postData"] = {
                "mimeType": "application/json",
                "text": json.dumps(event.post_data),
            }
        entries.append({
            "startedDateTime": "2026-10-13T00:00:00+00:00",
            "time": 10,
            "timings": {"send": 1, "wait": 8, "receive": 1},
            "request": request,
            "response": {
                "status": 200,
                "headers": [],
                "cookies": [],
                "redirectURL": "",
                "content": {"mimeType": "application/json", "text": "{}"},
            },
        })
    positive = runtime.replay._evaluate(
        task, json.dumps(reference_response), NetworkTrace.from_content(entries), evaluator_config
    )
    assert [result["score"] for result in positive] == [1.0, 1.0, 1.0]
    for changed in ("project", "target_id", "target_type", "body_id", "body_note", "internal"):
        contrast = deepcopy(entries)
        request = contrast[1]["request"]
        if changed == "project":
            request["url"] = request["url"].replace("a11y-webring.club", "a11y-webringXclub")
        elif changed == "target_id":
            request["url"] = request["url"].replace("target_id=83755", "target_id=83756")
        elif changed == "target_type":
            request["url"] = request["url"].replace("target_type=issue", "target_type=merge_request")
        else:
            body = json.loads(request["postData"]["text"])
            field, value = {
                "body_id": ("noteable_id", 83756),
                "body_note": ("note", "Wrong order"),
                "internal": ("internal", True),
            }[changed]
            body["note"][field] = value
            request["postData"]["text"] = json.dumps(body)
        negative = runtime.replay._evaluate(
            task, json.dumps(reference_response), NetworkTrace.from_content(contrast), evaluator_config
        )
        assert [result["score"] for result in negative] == [1.0, 1.0, 0.0], changed
    target_response.text.assert_called_once()
    archived = json.loads(
        (tmp_path / "author" / config["generated_task_id"] / "grounding-01/observation.json").read_text()
    )
    assert archived["target_entity"] == identity
    assert "noteable_contexts" not in archived["target_before"]
    parameters = draft["task_definition"]["parameters"]
    assert parameters["selected_fact"] == "Pending"
    assert parameters["preservation_cart_0_quantity"] == 1
    assert parameters["preservation_cart_0_subtotal"] == "$40.99"
    checks = draft["grounding"]["final_state_checks"]
    by_parameter = {check["parameter"]: check for check in checks}
    for index, note in enumerate(old_notes):
        parameter = f"preserved_note_{index}"
        assert parameters[parameter] == note["text"]
        assert by_parameter[parameter] == {
            "kind": "visible_text_exact",
            "parameter": parameter,
            "selector": "#" + note["id"] + " .note-text.md",
            "site": "gitlab",
            "readback_url_parameter": "issue_url",
        }
    assert by_parameter["issue_state_control"]["selector"] == (
        'form.js-main-target-form button[data-testid="close-reopen-button"]'
    )
    cart_check = by_parameter["preservation_cart_0_quantity"]
    assert cart_check["kind"] == "input_value"
    assert '[href$="/cart-item.html"]' in cart_check["selector"]
    assert cart_check["site"] == "shopping"
    assert cart_check["readback_url_parameter"] == "preservation_cart_url"
    assert by_parameter["preservation_orders_0_rows"]["site"] == "shopping"
    assert parameters["preservation_added_count"] == 0
    assert not any(
        call.args == (".pages-item-next a:visible",) for call in page.locator.call_args_list
    )
    assert (page.goto.call_args_list[0].args[0], page.goto.call_args_list[1].args[0]) == (
        home_url,
        cart_url,
    )
    restrict = context.route.call_args.args[1]
    estimate = MagicMock(
        request=SimpleNamespace(
            method="POST",
            url=sites["shopping"] + "/rest/default/V1/carts/mine/estimate-shipping-methods",
        )
    )
    restrict(estimate)
    estimate.continue_.assert_called_once()
    estimate.abort.assert_not_called()
    change_cart = MagicMock(
        request=SimpleNamespace(method="POST", url=sites["shopping"] + "/checkout/cart/updatePost/")
    )
    restrict(change_cart)
    change_cart.abort.assert_called_once()
    change_cart.continue_.assert_not_called()
    label_query = {
        "operationName": "issueLabels",
        "query": "query issueLabels($fullPath: ID!, $iid: String) { project { issue { labels { nodes { title } } } } }",
    }
    assignee_query = {
        "operationName": "issueAssignees",
        "query": "query issueAssignees($fullPath: ID!, $iid: String!) { project { issue { assignees { nodes { name } } } } }",
    }
    mutation = {
        "operationName": "issueLabels",
        "query": "mutation issueLabels($iid: String!) { issueSetLabels { issue { id } } }",
    }
    selected_mutation = {
        "operationName": "issueSetLabels",
        "query": label_query["query"] + " mutation issueSetLabels { issueSetLabels { issue { id } } }",
    }
    for payload, allowed in [
        (label_query, True),
        ([label_query, assignee_query], True),
        (mutation, False),
        ([label_query, mutation], False),
        (selected_mutation, False),
    ]:
        graphql = MagicMock(
            request=SimpleNamespace(
                method="POST", url=sites["gitlab"] + "/api/graphql", post_data_json=payload
            )
        )
        restrict(graphql)
        if allowed:
            graphql.continue_.assert_called_once()
            graphql.abort.assert_not_called()
        else:
            graphql.abort.assert_called_once()
            graphql.continue_.assert_not_called()


def test_rank_uses_public_title_tie_break_and_retains_displayed_fact(products):
    selected = select_entity(
        products,
        {"kind": "rank", "rank_by": "price", "direction": "asc", "rank": 1, "tie_break": "title_ascending"},
    )
    assert selected["entity"]["title"] == "Alpha Bag"
    assert selected["fact"] == "$20.00"
    assert selected["eligible_indices"] == [1, 0, 2]


def test_budget_and_visible_review_threshold_precede_rating_rank(products):
    selected = select_entity(
        products,
        {
            "kind": "rank", "rank_by": "rating",
            "direction": "desc",
            "rank": 1,
            "tie_break": "title_ascending",
            "budget": "25",
            "fact_field": "price",
            "filters": [{"field": "reviews", "op": "gte", "value": 3}],
        },
    )
    assert selected["entity"]["title"] == "Beta Bag"
    assert selected["fact"] == "$20.00"


def test_subtotal_share_uses_whole_source_population_before_filter(products):
    selected = select_entity(
        products,
        {
            "kind": "rank", "rank_by": "subtotal",
            "direction": "desc",
            "rank": 1,
            "tie_break": "title_ascending",
            "fact_field": "subtotal_share",
            "filters": [{"field": "price", "op": "lte", "value": "20"}],
        },
    )
    assert selected["entity"]["title"] == "Beta Bag"
    assert selected["fact"] == "42.11"
    parameters = {
        "selected_subtotal": "$40.00",
        "cart_item_1_subtotal": "$20.00",
        "cart_item_2_subtotal": "$35.00",
    }
    for name, rule in selected["derivations"].items():
        parameters[name] = derive_value(rule, parameters)
    assert parameters["cart_total"] == "95.00"
    assert parameters["selected_fact"] == "42.11"
    assert set(selected["derivations"]["cart_total"]["parameters"]) == {
        "selected_subtotal",
        "cart_item_1_subtotal",
        "cart_item_2_subtotal",
    }
    parameters["cart_item_2_subtotal"] = "$40.00"
    for name, rule in selected["derivations"].items():
        parameters[name] = derive_value(rule, parameters)
    assert parameters["selected_fact"] == "40.00"


def test_selection_cannot_reuse_duplicate_entity_or_underfill_rank(products):
    policy = {"kind": "rank", "rank_by": "price", "direction": "asc", "rank": 4, "tie_break": "title_ascending"}
    with pytest.raises(ValueError, match="too few"):
        select_entity(products, policy)
    with pytest.raises(ValueError, match="distinct"):
        select_entity([products[0], products[0]], {**policy, "rank": 1})


def test_ambiguous_public_identity_fails_instead_of_private_url_tie_break(products):
    ambiguous = [products[0], {**products[0], "url": "http://shopping.test/other.html"}]
    with pytest.raises(ValueError, match="uniquely"):
        select_entity(
            ambiguous,
            {"kind": "rank", "rank_by": "price", "direction": "asc", "rank": 1, "tie_break": "title_ascending"},
        )


def test_currency_decimal_comparison_preserves_exact_cents():
    entities = [
        {"title": "Under", "url": "http://shopping.test/a", "price": "$19.99"},
        {"title": "Over", "url": "http://shopping.test/b", "price": "$20.01"},
    ]
    chosen = select_entity(
        entities,
        {
            "kind": "rank", "rank_by": "price",
            "direction": "desc",
            "rank": 1,
            "tie_break": "title_ascending",
            "budget": "20.00",
        },
    )
    assert chosen["entity"]["title"] == "Under"


def test_share_uses_the_same_ties_to_even_rule_as_dependency_scoring():
    entities = [
        {"title": "Small", "url": "http://shopping.test/a", "subtotal": "$1.00"},
        {"title": "Large", "url": "http://shopping.test/b", "subtotal": "$31.00"},
    ]
    chosen = select_entity(
        entities,
        {
            "kind": "rank", "rank_by": "subtotal",
            "direction": "asc",
            "rank": 1,
            "tie_break": "title_ascending",
            "fact_field": "subtotal_share",
        },
    )
    assert chosen["fact"] == "3.12"
    parameters = {"selected_subtotal": "$1.00", "cart_item_1_subtotal": "$31.00"}
    for name, rule in chosen["derivations"].items():
        parameters[name] = derive_value(rule, parameters)
    assert parameters["selected_fact"] == chosen["fact"]


def test_zero_total_is_not_represented_as_a_percentage():
    with pytest.raises(ValueError, match="nonzero"):
        select_entity(
            [{"title": "Free", "url": "http://shopping.test/free", "subtotal": "$0.00"}],
            {
                "kind": "rank", "rank_by": "subtotal",
                "direction": "asc",
                "rank": 1,
                "tie_break": "title_ascending",
                "fact_field": "subtotal_share",
            },
        )


def test_quantity_population_and_typed_readout_do_not_use_digits_from_price():
    entity = {
        "title": "Single item",
        "url": "http://shopping.test/item",
        "price": "$21.00",
        "quantity": "1",
    }
    policy = {
        "kind": "rank", "rank_by": "quantity",
        "direction": "desc",
        "rank": 1,
        "tie_break": "title_ascending",
        "filters": [{"field": "quantity", "op": "eq", "value": 2}],
    }
    with pytest.raises(ValueError, match="too few"):
        select_entity([entity], policy)
    selected = select_entity([{**entity, "quantity": "2"}], policy)
    assert parse_fact_value(selected["fact"], "integer") == 2
    assert type(parse_fact_value(selected["fact"], "integer")) is int


@pytest.mark.parametrize(
    ("policy", "expected_index", "expected_fact"),
    [
        (
            {
                "kind": "rank", "rank_by": "price",
                "direction": "asc",
                "filters": [{"field": "reviews", "op": "gte", "value": 10}],
            },
            9,
            "$7.92",
        ),
        ({"kind": "rank", "rank_by": "rating", "direction": "desc", "fact_field": "price"}, 2, "$19.36"),
        ({"kind": "rank", "rank_by": "reviews", "direction": "asc", "fact_field": "price"}, 0, "$19.99"),
        (
            {"kind": "rank", "rank_by": "price", "direction": "desc", "fact_field": "reviews"},
            6,
            "12 \u00a0Reviews",
        ),
    ],
)
def test_observed_showcase_excludes_missing_queried_fields(
    observed_showcase, policy, expected_index, expected_fact
):
    assert len(observed_showcase) == 12
    assert [index for index, row in enumerate(observed_showcase) if row["reviews"] is None] == [
        5,
        8,
        10,
    ]
    chosen = select_entity(observed_showcase, {"rank": 1, "tie_break": "title_ascending", **policy})
    assert chosen["index"] == expected_index
    assert chosen["entity"] == observed_showcase[expected_index]
    assert chosen["fact"] == expected_fact
    assert not {5, 8, 10}.intersection(chosen["eligible_indices"])


def test_observed_showcase_missing_reviews_are_not_zero_reviews(observed_showcase):
    with pytest.raises(ValueError, match="too few qualifying"):
        select_entity(
            observed_showcase,
            {
                "kind": "rank", "rank_by": "price",
                "direction": "asc",
                "rank": 1,
                "tie_break": "title_ascending",
                "filters": [{"field": "reviews", "op": "lte", "value": 0}],
            },
        )


def test_price_only_selection_keeps_unrated_observed_products(observed_showcase):
    chosen = select_entity(
        observed_showcase,
        {"kind": "rank", "rank_by": "price", "direction": "desc", "rank": 1, "tie_break": "title_ascending"},
    )
    assert chosen["index"] == 10
    assert chosen["fact"] == "$411.76"
    assert len(chosen["eligible_indices"]) == 12


def test_budget_requires_visible_price_even_when_ranking_other_fields(observed_showcase):
    population = [dict(row) for row in observed_showcase]
    population[2]["price"] = None
    chosen = select_entity(
        population,
        {
            "kind": "rank", "rank_by": "rating",
            "fact_field": "reviews",
            "direction": "desc",
            "rank": 1,
            "tie_break": "title_ascending",
            "budget": "25.00",
        },
    )
    assert chosen["index"] == 7
    assert 2 not in chosen["eligible_indices"]


@pytest.mark.parametrize("share_role", ["rank", "fact", "filter"])
def test_subtotal_share_never_drops_an_unqualified_row_from_incomplete_denominator(
    products, share_role
):
    population = [dict(row) for row in products]
    population[2]["subtotal"] = None
    policy = {
        "kind": "rank", "rank_by": "price",
        "fact_field": "price",
        "direction": "asc",
        "rank": 1,
        "tie_break": "title_ascending",
        "filters": [{"field": "price", "op": "lte", "value": 20}],
    }
    if share_role == "rank":
        policy["rank_by"] = "subtotal_share"
    elif share_role == "fact":
        policy["fact_field"] = "subtotal_share"
    else:
        policy["filters"].append({"field": "subtotal_share", "op": "gte", "value": 5})
    with pytest.raises(ValueError, match="every visible population subtotal"):
        select_entity(population, policy)


def test_rank_selection_requires_an_explicit_kind(products):
    with pytest.raises(KeyError, match="kind"):
        select_entity(
            products,
            {"rank_by": "price", "direction": "asc", "rank": 1, "tie_break": "title_ascending"},
        )


@pytest.mark.parametrize(
    ("fact_field", "expected"), [("price_share", "26.67"), ("eligible_price_share", "53.33")]
)
def test_product_price_shares_keep_all_denominator_and_eligible_inputs(products, fact_field, expected):
    selected = select_entity(
        products,
        {
            "kind": "rank",
            "rank_by": "price",
            "direction": "asc",
            "rank": 1,
            "tie_break": "title_ascending",
            "filters": [{"field": "price", "op": "lte", "value": "20.00"}],
            "fact_field": fact_field,
        },
    )
    assert selected["entity"]["title"] == "Alpha Bag"
    assert selected["fact"] == expected
    assert selected["eligible_indices"] == [1, 0]
    parameters = {"selected_price": "$20.00", "product_0_price": "$20.00", "product_2_price": "$35.00"}
    for name, rule in selected["derivations"].items():
        parameters[name] = derive_value(rule, parameters)
    assert parameters["window_total"] == "75.00"
    assert parameters["selected_fact"] == expected
    if fact_field == "eligible_price_share":
        assert parameters["eligible_total"] == "40.00"
    parameters["product_2_price"] = "$40.00"
    for name, rule in selected["derivations"].items():
        parameters[name] = derive_value(rule, parameters)
    assert parameters["selected_fact"] == ("25.00" if fact_field == "price_share" else "50.00")


@pytest.mark.parametrize("fact_field", ["price_share", "eligible_price_share"])
def test_product_share_cannot_drop_a_missing_excluded_price(products, fact_field):
    population = deepcopy(products)
    population[2]["price"] = None
    with pytest.raises(ValueError, match="every visible population price"):
        select_entity(
            population,
            {
                "kind": "rank", "rank_by": "price", "direction": "asc", "rank": 1,
                "tie_break": "title_ascending", "fact_field": fact_field,
                "filters": [{"field": "title", "op": "contains", "value": "Alpha"}],
            },
        )


@pytest.fixture
def product_window_author(tmp_path, monkeypatch, runtime):
    """Exercise real author projections with two simulated product pages and one Save."""
    from playwright import sync_api
    from webarena_verified.core.utils import network_event_utils

    def run(selection, first, second):
        sites = {"shopping": "http://shopping.test", "gitlab": "http://gitlab.test"}
        urls = [sites["shopping"] + "/", sites["shopping"] + "/?pbaocw=2"]
        cart_url = sites["shopping"] + "/checkout/cart/"
        issue_url = sites["gitlab"] + "/byteblaze/a11y-webring.club/-/issues/2"
        note_url = sites["gitlab"] + "/byteblaze/a11y-webring.club/notes"
        identity = {
            "id": 83755, "iid": 2, "title": "Comparison request",
            "web_url": issue_url, "create_note_path": note_url,
        }
        response = MagicMock(ok=True)
        response.text.return_value = (
            '<div id="js-vue-notes" data-noteable-type="Issue" data-noteable-data="'
            + json.dumps(identity).replace('"', "&quot;") + '"></div>'
        )
        page, context, browser, manager = (MagicMock() for _ in range(4))
        manager.__enter__.return_value.chromium.launch.return_value = browser
        browser.new_context.return_value = context
        context.new_page.return_value = page
        page.get_by_test_id.return_value.count.return_value = 1
        page.goto.side_effect = lambda url, **kwargs: setattr(page, "url", url) or response
        page.locator.return_value.count.return_value = 1
        page.locator.return_value.first.click.side_effect = lambda: setattr(page, "url", urls[1])
        saved = SimpleNamespace(
            url=note_url, request=SimpleNamespace(method="POST"), status=200, ok=True
        )
        page.expect_response.return_value.__enter__.return_value.value = saved
        monkeypatch.setattr(sync_api, "sync_playwright", lambda: manager)
        pages = []
        for raw_rows in (first, second):
            pages.append([
                {
                    "title": title,
                    "url": sites["shopping"] + "/" + title.lower().replace(" ", "-") + ".html",
                    "price": price,
                    "reviews": None,
                    "readouts": {
                        "title": {
                            "kind": "text", "selector": ".product-item-link", "value_type": "string"
                        },
                        "price": {"kind": "text", "selector": ".price", "value_type": "string"},
                    },
                }
                for title, price in raw_rows
            ])
        observations = []
        issue_state = {
            "title": identity["title"], "description": None, "assignee": None,
            "labels": [], "open": True, "checklist": [],
        }

        def observe(page, output, name, site, **kwargs):
            observations.append((name, page.url))
            if site == "gitlab":
                return {
                    "url": issue_url,
                    "issue_state": deepcopy(issue_state),
                    "issue_identity": {
                        "project_path": "byteblaze/a11y-webring.club", "issue_iid": "2"
                    },
                    "notes": [] if name == "target_before" else [{
                        "text": page.get_by_test_id.return_value.fill.call_args.args[0],
                        "id": "note_new",
                    }],
                }
            return {
                "url": page.url,
                "links": [{"title": "My Cart", "url": cart_url}],
                "cart_items": [],
                "products": deepcopy(pages[urls.index(page.url)]) if page.url in urls else [],
            }

        monkeypatch.setattr(author_cross_site, "record_observation", observe)
        monkeypatch.setattr(
            "cross_site_evaluator.seal_recording",
            lambda output, *args: (output / "episode_boundary.json").write_text("{}"),
        )

        def events(path):
            return [
                *(SimpleNamespace(url=url, http_method="GET", request_status=200) for url in urls),
                SimpleNamespace(
                    url=note_url, http_method="POST", request_status=200, is_request_success=True,
                    post_data={"note": {
                        "note": page.get_by_test_id.return_value.fill.call_args.args[0],
                        "noteable_id": 83755, "noteable_type": "Issue", "internal": False,
                    }},
                ),
            ]

        monkeypatch.setattr(network_event_utils, "load_har_trace", events)
        config = {
            "generated_task_id": "wavgen-product-window-test",
            "task_id": 7000099,
            "intent_template_id": 90099,
            "revision": 1,
            "source_artifact": "abstract-family",
            "source_record_id": "offline-window",
            "family": "product-window-note",
            "source": {
                "kind": "products", "navigation": [], "max_pages": 2,
                "expected_count": len(first) + len(second),
            },
            "selection": selection,
            "target": {"issue_url": issue_url, "title": identity["title"]},
            "parameters": {},
            "note_template": "Audit {selected_entity_title}: {selected_fact}",
            "goal_template": "Compare the two visible pages and record the result.",
        }
        baseline = {
            "fresh_from_image": True,
            "authentication_only_preparation": True,
            "sites": {site: {"site_url": url} for site, url in sites.items()},
        }
        for name, value in [("config.json", config), ("baseline.json", baseline), ("sites.json", sites)]:
            (tmp_path / name).write_text(json.dumps(value))
        args = SimpleNamespace(
            task_config=tmp_path / "config.json",
            site_urls=tmp_path / "sites.json",
            baseline_receipt=tmp_path / "baseline.json",
            storage_state=tmp_path / "storage.json",
            generated_task_id=config["generated_task_id"],
            batch_root=tmp_path,
            attempt="grounding-01",
            timeout_ms=100,
            settle_ms=0,
        )
        draft = ground_task(args)
        from author import _task_spec
        from cross_site_evaluator import validate_dependencies
        from webarena_verified.types import WebArenaVerifiedTask

        validate_dependencies(draft["task_definition"], draft["sites"])
        compiled, _ = _task_spec(draft)
        WebArenaVerifiedTask.model_validate(compiled)
        return draft, observations, page

    return run


@pytest.mark.parametrize(
    ("fact_field", "expected"), [("price_share", "10.00"), ("eligible_price_share", "30.00")]
)
def test_product_window_grounding_binds_every_denominator_entity(
    product_window_author, fact_field, expected
):
    draft, observations, page = product_window_author(
        {
            "kind": "rank", "rank_by": "price", "direction": "asc", "rank": 1,
            "tie_break": "title_ascending", "fact_field": fact_field,
            "filters": [{"field": "price", "op": "lte", "value": "20.00"}],
        },
        [("Alpha", "$10.00"), ("Beta", "$20.00")],
        [("Gamma", "$30.00"), ("Zeta", "$40.00")],
    )
    definition = draft["task_definition"]
    assert definition["parameters"]["selected_fact"] == expected
    assert len(definition["dependencies"]) == 4
    assert {
        definition["parameters"][dep["source_entity"]["label_parameter"]]
        for dep in definition["dependencies"]
    } == {"Alpha", "Beta", "Gamma", "Zeta"}
    assert len([name for name, _ in observations if name.startswith("source_entity_")]) == 4
    assert page.locator.return_value.nth.return_value.scroll_into_view_if_needed.call_count == 4
    parameters = deepcopy(definition["parameters"])
    for evidence in draft["grounding"]["field_evidence"]:
        if "rule" in evidence:
            name = evidence["field"].rsplit("/", 1)[1]
            assert derive_value(evidence["rule"], parameters) == parameters[name]
    rules = {
        e["field"].rsplit("/", 1)[1]: e["rule"]
        for e in draft["grounding"]["field_evidence"] if "rule" in e
    }
    pending, consumed = ["note_body"], set()
    while pending:
        name = pending.pop()
        if name not in consumed:
            consumed.add(name)
            pending.extend(rules[name]["parameters"] if name in rules else [])
    assert {ref for dep in definition["dependencies"] for ref in dep["fact_readouts"]} <= consumed


@pytest.mark.parametrize(
    ("first", "second", "budget", "anchor", "counterpart", "total"),
    [
        ([("Alpha", "$6.00"), ("Zeta", "$4.00")], [("Beta", "$3.00"), ("Gamma", "$2.00")], "10.00", "Alpha", "Beta", "9.00"),
        ([("Alpha", "$4.99"), ("Zeta", "$7.00")], [("Beta", "$5.01"), ("Gamma", "$4.99")], "10.00", "Alpha", "Beta", "10.00"),
        ([("Alpha", "$4.99"), ("Zeta", "$7.00")], [("Beta", "$5.01"), ("Gamma", "$4.99")], "9.99", "Alpha", "Gamma", "9.98"),
        ([("Zeta", "$7.00"), ("Alpha", "$5.00")], [("Beta", "$3.00"), ("Gamma", "$5.00")], "10.00", "Alpha", "Gamma", "10.00"),
    ],
)
def test_budget_pair_grounding_respects_pages_cents_and_public_ties(product_window_author, first, second, budget, anchor, counterpart, total):
    draft, observations, page = product_window_author(
        {"kind": "budget_pair", "budget": budget, "tie_break": "sorted_title_tuple_ascending"}, first, second,
    )
    definition = draft["task_definition"]
    parameters = definition["parameters"]
    assert parameters["selected_entity_title"] == anchor
    assert parameters["counterpart_title"] == counterpart
    assert parameters["pair_total"] == total
    window_total = sum((Decimal(price.removeprefix("$")) for _, price in first + second), Decimal(0))
    share = format(Decimal(total) / window_total * 100, ".2f")
    assert parameters["selected_fact"] == f"Partner: {counterpart} | Pair total: ${total} | Window share: {share}%"
    assert len(definition["dependencies"]) == 4
    assert len([name for name, _ in observations if name.startswith("source_entity_")]) == 4
    assert page.locator.return_value.nth.return_value.scroll_into_view_if_needed.call_count == 4
    rules = {e["field"].rsplit("/", 1)[1]: e["rule"] for e in draft["grounding"]["field_evidence"] if "rule" in e}
    for name, rule in rules.items():
        assert derive_value(rule, parameters) == parameters[name]
    pending, consumed = ["note_body"], set()
    while pending:
        name = pending.pop()
        if name not in consumed:
            consumed.add(name)
            pending.extend(rules[name]["parameters"] if name in rules else [])
    assert "counterpart_title" in consumed
    assert {ref for dep in definition["dependencies"] for ref in dep["fact_readouts"]} <= consumed
