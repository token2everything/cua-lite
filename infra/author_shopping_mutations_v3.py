#!/usr/bin/env python3
"""Ground shopping mutations through a real Save on a fresh fixed baseline.

The coordinator supplies an owned baseline and authenticated storage state.
Select one case with --generated-task-id and --task-configs. Only that task's
business edit is submitted; no asset preparation or model call is performed.
Drafts require successful HAR evidence and persisted UI readback. Synthetic
wrong-object and no-op contrasts exercise the upstream evaluator without writes.
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from author_shopping import _record, _usd


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


def _record_mutation(page: Any, output: Path, name: str, button: Any) -> dict[str, Any]:
    """Extend the existing shopping observation with the actual Save form boundary."""
    observation = _record(page, output, name)
    observation["mutation_form"] = button.evaluate(
        """b => {
          const f = b.form;
          const visible = e => {const r=e.getBoundingClientRect(); return !!(r.width&&r.height);};
          return {id:f.id,action:f.action,method:f.method,
            button:{name:b.name,value:b.value,text:b.innerText.trim()},
            fields:Array.from(f.elements).filter(e => visible(e) && e.name && !e.disabled &&
              !['hidden','password','submit','button'].includes(e.type)).map(e => ({
                name:e.name,id:e.id,type:e.type,value:e.value,
                checked:['checkbox','radio'].includes(e.type) ? e.checked : null}))};
        }"""
    )
    observation["cart_quantity_controls"] = page.locator(
        "#shopping-cart-table tbody.cart.item tr.item-info"
    ).evaluate_all(
        """rows => rows.map(tr => {
          const q=tr.querySelector('input[data-role=cart-item-qty]');
          return {url:tr.querySelector('.product-item-name a').href,
            name:q.name,id:q.id,value:q.value};
        })"""
    )
    (output / f"{name}.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
    )
    return observation


def _cart_targets(before: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    """Resolve the public price/initial-quantity conditions from the fresh cart UI."""
    parameters = config["parameters"]
    items = before["cart_items"]
    if len(items) != config["expected_cart_items"]:
        raise ValueError("fresh fixed cart differs from the observed selection scope")
    by_url = {item["url"]: item for item in items}
    controls = before["cart_quantity_controls"]
    if len(by_url) != len(items) or {c["url"] for c in controls} != set(by_url):
        raise ValueError("cart row identities and their observed quantity controls are ambiguous")
    qualifying = [
        item
        for item in items
        if Decimal(str(parameters["minimum_price_usd"]))
        <= _usd(item["unit_price"])
        <= Decimal(str(parameters["maximum_price_usd"]))
        and Decimal(item["quantity"]) == parameters["initial_quantity"]
    ]
    if len(qualifying) != 2:
        raise ValueError("public cart conditions do not select exactly two current entries")
    qualifying.sort(key=lambda item: _usd(item["unit_price"]))
    if _usd(qualifying[0]["unit_price"]) == _usd(qualifying[1]["unit_price"]):
        raise ValueError("the public higher/lower price comparison is tied")
    targets = {item["url"]: item["quantity"] for item in items}
    if config["family"] == "cart_condition_single_quantity":
        targets[qualifying[1]["url"]] = str(parameters["new_quantity"])
    elif config["family"] == "cart_condition_multiple_quantities":
        targets[qualifying[0]["url"]] = str(parameters["lower_price_quantity"])
        targets[qualifying[1]["url"]] = str(parameters["higher_price_quantity"])
    else:
        raise ValueError(f"unsupported cart family: {config['family']}")
    changed = [item for item in items if targets[item["url"]] != item["quantity"]]
    expected_changes = 1 if config["family"] == "cart_condition_single_quantity" else 2
    if len(changed) != expected_changes or any(
        not value.isdigit() or int(value) <= 0 for value in targets.values()
    ):
        raise ValueError("cart task must change the selected positive quantities and preserve rows")
    return targets


def _save_event_and_contrasts(
    output: Path,
    config: dict[str, Any],
    site_url: str,
    observation: dict[str, Any],
    expected_fields: dict[str, str],
    initial_fields: dict[str, str],
    numeric_fields: set[str],
) -> dict[str, Any]:
    """Bind actual HAR fields and audit no-op/wrong-object requests with the upstream owner."""
    from webarena_verified.core.evaluation.evaluators import EVALUATOR_REGISTRY
    from webarena_verified.types import WebArenaVerifiedTask
    from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
    from webarena_verified.types.eval import TaskEvalContext
    from webarena_verified.types.tracing import NetworkTrace

    trace = NetworkTrace.from_har(output / "network.har")
    receipt = observation["save_request"]
    matching = [
        event
        for event in trace.events
        if event.url == receipt["url"]
        and event.http_method == "POST"
        and event.request_status == receipt["response_status"]
    ]
    if len(matching) != 1 or not matching[0].is_request_success:
        raise ValueError("exactly one actual successful Save request is required")
    actual = matching[0]
    body = actual.post_data
    if any(
        (
            Decimal(str(body[field])) != Decimal(value)
            if field in numeric_fields
            else str(body[field]) != value
        )
        for field, value in expected_fields.items()
    ):
        raise ValueError("actual Save request differs from the fresh UI-derived mutation plan")
    critical = {field: body[field] for field in expected_fields}
    receipt["post_data"] = critical
    schema = {
        "type": "object",
        "properties": {
            field: {"type": "number" if field in numeric_fields else "string"} for field in critical
        },
    }
    path = urlsplit(actual.url).path.removeprefix(urlsplit(site_url).path.rstrip("/"))
    event = {
        "url_template": "^__SHOPPING__" + re.escape(path.rstrip("/")) + "/*$",
        "http_method": "POST",
        "response_status": actual.request_status,
        "post_data": critical,
        "post_data_schema": schema,
        "last_event_only": False,
    }
    # This task is an evaluator fixture, not an independently grounded task.
    response = {"task_type": "MUTATE", "status": "SUCCESS", "retrieved_data": None}
    task = WebArenaVerifiedTask.model_validate(
        {
            "sites": ["shopping"],
            "task_id": config["task_id"],
            "intent_template_id": config["intent_template_id"],
            "start_urls": ["__SHOPPING__"],
            "intent": "Audit the recorded Save request.",
            "intent_template": "Audit Save.",
            "instantiation_dict": {},
            "revision": 1,
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "ordered": False,
                    "results_schema": {"type": "null"},
                    "expected": response,
                },
                {
                    "evaluator": "NetworkEventEvaluator",
                    "last_event_only": False,
                    "post_data_schema": schema,
                    "expected": {
                        "url": event["url_template"],
                        "http_method": "POST",
                        "response_status": actual.request_status,
                        "post_data": critical,
                    },
                },
            ],
        }
    )
    evaluator = EVALUATOR_REGISTRY["NetworkEventEvaluator"]()
    fixture_path = output / "synthetic_evaluator_fixture.jsonl"
    fixture_path.write_text(task.model_dump_json() + "\n")
    eval_config = WebArenaVerifiedConfig(
        test_data_file=fixture_path,
        environments={"shopping": EnvironmentConfig(urls=[site_url])},
    )
    entry = deepcopy(actual.model_dump(mode="json")["data"])
    noop = dict(body)
    noop.update(initial_fields)
    wrong_object = dict(body)
    if numeric_fields:
        changed = [
            field
            for field in sorted(numeric_fields)
            if initial_fields[field] != expected_fields[field]
        ]
        other = next(field for field in sorted(numeric_fields) if field != changed[0])
        wrong_object[changed[0]], wrong_object[other] = body[other], body[changed[0]]
    else:
        # Editing a different address keeps the posted text correct but changes actual identity.
        parts = entry["request"]["url"].split("/")
        identity = parts.index("id") + 1
        parts[identity] = str(int(parts[identity]) + 1)
        wrong_object_url = "/".join(parts)
    equivalent = dict(body)
    for field in numeric_fields:
        equivalent[field] = str(float(Decimal(str(body[field]))))
    probes = []
    for name, probe_body, expected_score in (
        ("positive_actual_har", body, 1),
        ("no_op_initial_values", noop, 0),
        ("wrong_object", wrong_object, 0),
        (
            "equivalent_numeric_encoding" if numeric_fields else "preserved_string_encoding",
            equivalent,
            1,
        ),
    ):
        probe_entry = deepcopy(entry)
        if name == "wrong_object" and not numeric_fields:
            probe_entry["request"]["url"] = wrong_object_url
        probe_entry["request"]["postData"] = {
            "mimeType": "application/x-www-form-urlencoded",
            "text": urlencode(probe_body, doseq=True),
        }
        probe_trace = (
            trace if name == "positive_actual_har" else NetworkTrace.from_content([probe_entry])
        )
        context = TaskEvalContext(
            task=task,
            agent_response_raw=json.dumps(response | {"error_details": None}),
            network_trace=probe_trace,
            config=eval_config,
        )
        result = evaluator.evaluate(context=context, config=task.eval[1])
        passed = result.score == expected_score and result.status == (
            "success" if expected_score else "failure"
        )
        probes.append(
            {
                "probe": name,
                "artifact_kind": "actual_har"
                if name == "positive_actual_har"
                else "synthetic_contrast",
                "score": result.score,
                "status": result.status,
                "expected_score": expected_score,
                "passed": passed,
            }
        )
    observation["network_contrasts"] = probes
    (output / "observation.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
    )
    if not all(probe["passed"] for probe in probes):
        raise ValueError("the recorded Save's canonical wrong-object/no-op contrasts failed")
    return event


def main() -> None:
    args = _parse_args()
    config = next(
        row
        for row in json.loads(args.task_configs.read_text())["tasks"]
        if row["generated_task_id"] == args.generated_task_id
    )
    baseline = json.loads(args.baseline_receipt.read_text())
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("authoring requires a coordinator-owned fresh fixed image baseline")
    if baseline["site_url"].rstrip("/") != args.site_url.rstrip("/"):
        raise ValueError("author endpoint differs from the coordinator baseline receipt")
    output = args.batch_root / "author" / args.generated_task_id / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    parameters = dict(config["parameters"])
    observation: dict[str, Any] = {}
    final_checks = []
    evidence = []
    blocked = []
    allowed_save = None
    origin = urlsplit(args.site_url)
    cart_family = config["family"] in {
        "cart_condition_single_quantity",
        "cart_condition_multiple_quantities",
    }
    if not cart_family and config["family"] != "existing_address_two_field_update":
        raise ValueError(f"unsupported mutation family: {config['family']}")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            storage_state=str(args.storage_state),
            viewport={"width": 1920, "height": 1080},
            service_workers="block",
            record_har_path=str(output / "network.har"),
            record_har_content="embed",
        )

        def restrict(route: Any) -> None:
            request = route.request
            target = urlsplit(request.url)
            if (target.scheme, target.netloc) == (origin.scheme, origin.netloc) and (
                request.method in {"GET", "HEAD"}
                or (request.method == "POST" and request.url == allowed_save)
            ):
                route.continue_()
            else:
                blocked.append({"url": request.url, "method": request.method})
                route.abort()

        context.route("**/*", restrict)
        context.tracing.start(screenshots=True, snapshots=True, sources=False)
        page = context.new_page()
        try:
            site = args.site_url.rstrip("/")
            if cart_family:
                terminal_url = site + "/checkout/cart/"
                page.goto(terminal_url, wait_until="networkidle", timeout=120_000)
                button = page.locator('button[name="update_cart_action"][value="update_qty"]')
                before = _record_mutation(page, output, "before", button)
                observation["before"] = before
                targets = _cart_targets(before, config)
                initial_fields = {}
                expected_fields = {}
                for control in before["cart_quantity_controls"]:
                    field = control["name"]
                    initial_fields[field] = control["value"]
                    expected_fields[field] = targets[control["url"]]
                    page.locator(f"input[name={json.dumps(field)}]").fill(expected_fields[field])
                numeric_fields = set(expected_fields)
                goal = (
                    "In your existing shopping cart, consider entries whose current quantity "
                    "is {initial_quantity} and displayed unit price is between "
                    "{minimum_price_usd} and {maximum_price_usd} USD inclusive. "
                )
                if config["family"] == "cart_condition_single_quantity":
                    goal += (
                        "Of the two qualifying entries, set the higher-priced one's quantity "
                        "to {new_quantity}. "
                    )
                else:
                    goal += (
                        "Of the two qualifying entries, set the lower-priced one's quantity to "
                        "{lower_price_quantity} and the higher-priced one's quantity to "
                        "{higher_price_quantity}. "
                    )
                goal += (
                    "Save with Update Shopping Cart, then revisit or refresh to confirm. "
                    "Preserve every existing item and every other quantity. Do not remove or add "
                    "items, move items to a wishlist, apply coupons, or check out. "
                    "Finish on the cart."
                )
            else:
                page.goto(site + "/customer/address/", wait_until="networkidle", timeout=120_000)
                address_book = _record(page, output, "address_book")
                observation["address_book_before"] = address_book
                editor_urls = {
                    link["url"]
                    for link in address_book["links"]
                    if link["text"] in {"Change Billing Address", "Change Shipping Address"}
                }
                if len(editor_urls) != 1:
                    raise ValueError("task requires the same existing address to be both defaults")
                terminal_url = editor_urls.pop()
                page.goto(terminal_url, wait_until="networkidle", timeout=120_000)
                button = page.get_by_role("button", name="Save Address", exact=True)
                before = _record_mutation(page, output, "before", button)
                observation["before"] = before
                initial_fields = {
                    field["name"]: field["value"] for field in before["mutation_form"]["fields"]
                }
                if initial_fields["city"] != parameters["address_city"] or initial_fields[
                    "postcode"
                ] != str(parameters["address_postcode"]):
                    raise ValueError(
                        "fresh default address does not match the public city/postcode"
                    )
                default_labels = [
                    "It's a default billing address.",
                    "It's a default shipping address.",
                ]
                if any(label not in before["body_text"] for label in default_labels):
                    raise ValueError("selected existing address is not visibly both defaults")
                expected_fields = dict(initial_fields)
                expected_fields.update(
                    company=parameters["new_company"], telephone=parameters["new_telephone"]
                )
                if any(
                    initial_fields[field] == expected_fields[field]
                    for field in ("company", "telephone")
                ):
                    raise ValueError("both requested address fields must change")
                numeric_fields = set()
                for field in ("company", "telephone"):
                    page.locator(f"input[name={json.dumps(field)}]").fill(expected_fields[field])
                goal = (
                    "In Address Book, edit the existing address that is both your default "
                    "billing and shipping address and has City {address_city} and "
                    "Zip/Postal Code {address_postcode}. Set Company to {new_company} and "
                    "Phone Number to {new_telephone}. Preserve its first and last name, "
                    "street lines, city, state, postal code, country and both default roles. "
                    "Save Address, revisit its editor and refresh to verify the two saved values. "
                    "Do not add or delete addresses or change any shopping cart item. "
                    "Finish on that existing address's editor."
                )
            form = before["mutation_form"]
            save_target = urlsplit(form["action"])
            if form["method"].lower() != "post" or (save_target.scheme, save_target.netloc) != (
                origin.scheme,
                origin.netloc,
            ):
                raise ValueError("observed Save form is not a same-site POST")
            allowed_save = form["action"]
            # Cart Save redirects back to the same URL. Bind the new document
            # before clicking so the old document's load state cannot satisfy it.
            with page.expect_navigation(wait_until="networkidle", timeout=60_000):
                with page.expect_response(
                    lambda response: response.request.method == "POST"
                    and response.url == allowed_save
                ) as saved:
                    button.click()
                response = saved.value
                observation["save_request"] = {
                    "url": response.url,
                    "response_status": response.status,
                    "http_method": response.request.method,
                }
            observation["after_save"] = _record(page, output, "after_save")
            page.goto(terminal_url, wait_until="networkidle", timeout=120_000)
            observation["revisited"] = _record_mutation(page, output, "revisited", button)
            page.reload(wait_until="networkidle", timeout=120_000)
            terminal = _record_mutation(page, output, "terminal", button)
            observation["terminal"] = terminal
            if cart_family:
                current = {item["url"]: item for item in terminal["cart_items"]}
                if set(current) != set(targets) or any(
                    current[item["url"]]["quantity"] != targets[item["url"]]
                    or current[item["url"]]["name"] != item["name"]
                    or current[item["url"]]["unit_price"] != item["unit_price"]
                    for item in before["cart_items"]
                ):
                    raise ValueError(
                        "saved/refreshed cart changed row identity or missed planned quantities"
                    )
                parameters["initial_cart_names"] = [item["name"] for item in before["cart_items"]]
                final_checks.append(
                    {
                        "kind": "text_set_preserved",
                        "parameter": "initial_cart_names",
                        "selector": "#shopping-cart-table .product-item-name",
                    }
                )
                for index, control in enumerate(terminal["cart_quantity_controls"]):
                    parameter = f"cart_quantity_{index}"
                    parameters[parameter] = control["value"]
                    final_checks.append(
                        {
                            "kind": "input_value",
                            "parameter": parameter,
                            "selector": f"input[name={json.dumps(control['name'])}]",
                        }
                    )
                    evidence.append(
                        {
                            "field": f"/task_definition/parameters/{parameter}",
                            "artifact": "observation",
                            "location": f"/terminal/cart_quantity_controls/{index}/value",
                            "modality": "text",
                            "derivation": "Persisted refreshed quantity input.",
                        }
                    )
            else:
                actual_fields = {
                    field["name"]: field["value"] for field in terminal["mutation_form"]["fields"]
                }
                if actual_fields != expected_fields or any(
                    label not in terminal["body_text"] for label in default_labels
                ):
                    raise ValueError(
                        "saved/refreshed address differs from the two-field edit or lost defaults"
                    )
                for index, field in enumerate(terminal["mutation_form"]["fields"]):
                    parameter = f"address_field_{index}"
                    parameters[parameter] = field["value"]
                    final_checks.append(
                        {
                            "kind": "input_value",
                            "parameter": parameter,
                            "selector": f"[name={json.dumps(field['name'])}]",
                        }
                    )
                    evidence.append(
                        {
                            "field": f"/task_definition/parameters/{parameter}",
                            "artifact": "observation",
                            "location": f"/terminal/mutation_form/fields/{index}/value",
                            "modality": "text",
                            "derivation": "Persisted refreshed address field.",
                        }
                    )
                for index, label in enumerate(default_labels):
                    parameter = f"preserved_default_{index}"
                    parameters[parameter] = label
                    final_checks.append(
                        {"kind": "visible_text", "parameter": parameter, "selector": "main"}
                    )
            parameters["terminal_url"] = terminal_url.replace(site, "__SHOPPING__", 1)
            final_checks.insert(
                0, {"kind": "final_url", "parameter": "terminal_url", "selector": ""}
            )
        finally:
            (output / "observation.json").write_text(
                json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
            )
            (output / "blocked_requests.json").write_text(json.dumps(blocked, indent=2) + "\n")
            context.tracing.stop(path=str(output / "trace.zip"))
            context.close()
            browser.close()
    event = _save_event_and_contrasts(
        output, config, args.site_url, observation, expected_fields, initial_fields, numeric_fields
    )
    for field in event["post_data"]:
        evidence.append(
            {
                "field": f"/task_definition/completion_events/0/post_data/{field}",
                "artifact": "observation",
                "location": f"/save_request/post_data/{field}",
                "modality": "text",
                "derivation": "Actual Save POST parsed by the canonical HAR owner.",
            }
        )
    relative = output.relative_to(args.batch_root)
    prohibited_cart_operations = (
        "add|delete|couponPost" if cart_family else "add|updatePost|delete|couponPost"
    )
    forbidden_event = {
        "url_template": (
            "^__SHOPPING__/(checkout/cart/(" + prohibited_cart_operations + ")|"
            "wishlist/index/(add|remove)|catalog/product_compare/add|sales/order/reorder)(/.*)*$"
        ),
        "http_method": "POST",
        "should_not_exist": True,
        "last_event_only": False,
    }
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
        "task_definition": {
            "task_type": "MUTATE",
            "goal_template": goal,
            "parameters": parameters,
            "answer": {"status": "SUCCESS", "values": None, "item_schema": None, "ordered": False},
            "completion_events": [event, forbidden_event],
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
        },
    }
    (output / "draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "grounded", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
