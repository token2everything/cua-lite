"""Own cross-site entity, visible fact, target-use and recording-order evidence.

This gate evaluates one original GUI episode. Native WebArena evaluators still
own response and request correctness; this owner adds the declared dependency.
"""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from contract import ContractError, sha256_file, sha256_json


def derive_value(rule: dict[str, Any], parameters: dict[str, Any]) -> Any:
    """Apply an explicit parameter derivation to grounded inputs, without guessing facts."""
    values = [parameters[name] for name in rule["parameters"]]
    operation = rule["operation"]
    if operation == "format":
        return rule["template"].format_map({name: parameters[name] for name in rule["parameters"]})
    numbers = [Decimal(str(value).replace("$", "").replace(",", "")) for value in values]
    if operation == "sum":
        result = sum(numbers, Decimal(0))
    elif operation == "multiply":
        result = math.prod(numbers)
    elif operation == "ratio_percent" and len(numbers) == 2:
        result = numbers[0] / numbers[1] * 100
    else:
        raise ContractError("unsupported explicit arithmetic derivation")
    return format(result, rule["output_format"]) if "output_format" in rule else float(result)


def parse_fact_value(value: Any, value_type: str) -> str | int | float:
    """Parse a declared DOM fact type without inferring numbers from surrounding text."""
    if value_type == "string":
        if not isinstance(value, str):
            raise ValueError("string facts require an actual DOM string")
        return " ".join(value.split())
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("numeric facts require a complete numeric DOM value") from error
    if not number.is_finite():
        raise ValueError("numeric facts must be finite")
    if value_type == "integer" and number == number.to_integral_value():
        return int(number)
    if value_type == "decimal":
        return float(number)
    raise ValueError("fact value does not satisfy its declared type")


def validate_dependencies(definition: dict[str, Any], sites: list[str]) -> None:
    """Require named facts and cross-site edges in the private task definition."""
    dependencies = definition["dependencies"]
    if not isinstance(dependencies, list) or bool(dependencies) != (len(sites) == 2):
        raise ContractError("two-site tasks require dependencies; single-site tasks use []")
    names = set()
    parameters = definition["parameters"]
    for dependency in dependencies:
        if set(dependency) != {
            "name",
            "source_site",
            "target_site",
            "source_entity",
            "source_url_parameter",
            "fact_readouts",
            "target_event_index",
        }:
            raise ContractError("dependency fields must use the canonical cross-site contract")
        if not dependency["name"] or dependency["name"] in names:
            raise ContractError("dependency names must be nonempty and unique")
        names.add(dependency["name"])
        source, target = dependency["source_site"], dependency["target_site"]
        if source == target or source not in sites or target not in sites:
            raise ContractError("dependency must connect two declared sites")
        entity = dependency["source_entity"]
        if set(entity) != {"identity_parameter", "label_parameter"}:
            raise ContractError("source entity must name its canonical identity and visible label")
        refs = [
            dependency["source_url_parameter"],
            *entity.values(),
            *dependency["fact_readouts"],
        ]
        if not dependency["fact_readouts"] or any(ref not in parameters for ref in refs):
            raise ContractError("dependency facts and entity must reference semantic parameters")
        for parameter, readout in dependency["fact_readouts"].items():
            if (
                set(readout) != {"kind", "selector", "value_type"}
                or readout["kind"] not in {"text", "input_value"}
                or readout["value_type"] not in {"string", "integer", "decimal"}
                or not isinstance(readout["selector"], str)
                or not readout["selector"]
            ):
                raise ContractError(
                    "facts require an explicit DOM readout kind, selector and value type"
                )
            value = parameters[parameter]
            if (
                (readout["value_type"] == "string" and type(value) is not str)
                or (readout["value_type"] == "integer" and type(value) is not int)
                or (readout["value_type"] == "decimal" and type(value) not in {int, float})
            ):
                raise ContractError("semantic fact values must use their declared canonical type")
            parse_fact_value(value, readout["value_type"])
        source_placeholder = f"__{source.upper()}__"
        if any(
            not isinstance(parameters[name], str) or source_placeholder not in parameters[name]
            for name in (dependency["source_url_parameter"], entity["identity_parameter"])
        ):
            raise ContractError("dependency source URLs must use their declared site's placeholder")
        index = dependency["target_event_index"]
        if type(index) is not int or not 0 <= index < len(definition["completion_events"]):
            raise ContractError("dependency target must reference a completion event")
        event = definition["completion_events"][index]
        if f"__{target.upper()}__" not in event["url_template"]:
            raise ContractError("dependency target event must belong to its declared site")


_GUI_OBSERVATION = """() => {
        // Navigation can expose a document before its body exists. Keep that
        // delivered frame as empty evidence instead of waiting for another page.
        if (document.body === null) return null;
        const layout = [];
        const visible = (node, rects) => {
            const parents = [];
            for (let e = node.nodeType === 1 ? node : node.parentElement; e; e = e.parentElement) {
                const style = getComputedStyle(e);
                if (style.display === 'none' || style.visibility !== 'visible' || Number(style.opacity) === 0) return false;
                parents.push([e, style]);
            }
            return rects.length > 0 && rects.every(r => r.width > 0 && r.height > 0 &&
                r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight &&
                parents.every(([e, style]) => {
                    const box = e.getBoundingClientRect();
                    return (style.overflowX === 'visible' || (r.left >= box.left && r.right <= box.right)) &&
                           (style.overflowY === 'visible' || (r.top >= box.top && r.bottom <= box.bottom));
                }));
        };
        const visibleText = root => {
            const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
            const texts = [];
            for (let node = walker.nextNode(); node; node = walker.nextNode()) {
                if (!node.textContent.trim()) continue;
                const range = document.createRange(); range.selectNodeContents(node);
                const rects = [...range.getClientRects()];
                if (visible(node, rects)) {
                    texts.push(node.textContent);
                    layout.push(rects.map(r => [r.x, r.y, r.width, r.height]));
                }
            }
            return texts.join(' ');
        };
        const visibleHTML = node => {
            if (node.nodeType === Node.TEXT_NODE) {
                const range = document.createRange(); range.selectNodeContents(node);
                return node.textContent.trim() && visible(node, [...range.getClientRects()])
                    ? document.createTextNode(node.textContent) : null;
            }
            if (node.nodeType !== Node.ELEMENT_NODE) return null;
            const nodeVisible = visible(node, [...node.getClientRects()]);
            if (node.tagName === 'INPUT' && !nodeVisible) return null;
            const copy = node.cloneNode(false);
            if (node.tagName === 'INPUT') {
                copy.setAttribute('value', node.value);
                layout.push([...node.getClientRects()].map(r => [r.x, r.y, r.width, r.height]));
            }
            for (const child of node.childNodes) {
                const projected = visibleHTML(child);
                if (projected !== null) copy.appendChild(projected);
            }
            // Preserve selector ancestry for visible facts; hidden templates
            // and comments cannot change the screenshot's evidence projection.
            return node === document.body || copy.hasChildNodes() ||
                (nodeVisible && (node.tagName === 'INPUT' || node.tagName === 'IMG' ||
                                 (node.tagName === 'A' && node.hasAttribute('href'))))
                ? copy : null;
        };
        const observation = {page_url: location.href, body_text: visibleText(document.body), visible_html: visibleHTML(document.body).outerHTML,
            links: [...document.querySelectorAll('a[href]')]
            .filter(a => visible(a, [...a.getClientRects()]))
            .map(a => ({href: a.href, text: visibleText(a), body_text: visibleText(
                a.closest('tr,li,article,div.product-item-info,div.product-item,div.product-card,div.product,div.item') || a
            ), visible_html: visibleHTML(
                a.closest('tr,li,article,div.product-item-info,div.product-item,div.product-card,div.product,div.item') || a
            ).outerHTML}))};
        return {...observation, layout, viewport: [innerWidth, innerHeight, scrollX, scrollY],
            visibility: document.visibilityState, focused: document.hasFocus()};
    }"""


def _write_gui_frame(
    output: Path,
    observation: dict[str, Any] | None,
    screenshot: str,
    context_id: str,
    scope: str,
    producer: Path,
) -> None:
    frame = {
        "at_utc": datetime.now(UTC).isoformat(),
        "scope": scope,
        "context_id": context_id,
        "page_url": observation["page_url"] if observation is not None else "",
        "screenshot": screenshot,
        "screenshot_sha256": sha256_file(output / screenshot),
        "producer_sha256": sha256_file(producer),
        "body_text": observation["body_text"] if observation is not None else "",
        "visible_html": observation["visible_html"] if observation is not None else "",
        "links": observation["links"] if observation is not None else [],
    }
    with (output / "cross_site_gui_frames.jsonl").open("a") as stream:
        stream.write(json.dumps(frame, ensure_ascii=False) + "\n")


def record_gui_frame(
    output: Path, page: Any | None, screenshot: str, context_id: str, scope: str, producer: Path
) -> None:
    """Record browser-owned screenshot evidence from the current document."""
    observation = page.evaluate(_GUI_OBSERVATION) if page is not None else None
    _write_gui_frame(output, observation, screenshot, context_id, scope, producer)


def capture_gui_frame(
    output: Path,
    active_page: Callable[[], Any | None],
    capture: Callable[[], bytes],
    screenshot: str,
    context_id: str,
    scope: str,
    producer: Path,
) -> bytes:
    """Capture once; only unchanged document and visible state can supply DOM facts."""
    from playwright.sync_api import Error as PlaywrightError

    document = None
    observation = None
    before = None
    try:
        try:
            page = active_page()
            if page is not None:
                document = page.evaluate_handle("document")
                before = document.evaluate(
                    "doc => doc === document ? (" + _GUI_OBSERVATION + ")() : null"
                )
        except PlaywrightError as error:
            if "Execution context was destroyed, most likely because of a navigation" not in str(
                error
            ):
                raise
        finally:
            # Capture even if the pre-capture document disappears. This is the
            # only screenshot for this turn and is never replaced by a later DOM.
            image = capture()
            (output / screenshot).write_bytes(image)
        try:
            if before is not None and active_page() is page:
                after = document.evaluate(
                    "doc => doc === document ? (" + _GUI_OBSERVATION + ")() : null"
                )
                if before == after:
                    observation = before
        except PlaywrightError as error:
            if "Execution context was destroyed, most likely because of a navigation" not in str(
                error
            ):
                raise
    finally:
        if document is not None:
            document.dispose()
    _write_gui_frame(output, observation, screenshot, context_id, scope, producer)
    return image


def seal_recording(
    output: Path,
    context_id: str,
    scope: str,
    started_at: str,
    ended_at: str,
    har: Path,
    trace: Path,
    producer: Path,
) -> None:
    """Seal author recordings after trace stop and context close have flushed both files."""
    boundary = {
        "scope": scope,
        "recording_context_id": context_id,
        "har_context_id": context_id,
        "playwright_context_id": context_id,
        "audit_context_ids": [],
        "started_at": started_at,
        "ended_at": ended_at,
        "artifact_sha256": {
            "har": sha256_file(har),
            "trace": sha256_file(trace),
            "gui_frames": sha256_file(output / "cross_site_gui_frames.jsonl"),
        },
    }
    with (output / "episode_boundary.json").open("x") as stream:
        json.dump(boundary, stream, indent=2)
    (output / "episode_recording_lifecycle.json").write_text(
        json.dumps(
            {
                "producer_sha256": sha256_file(producer),
                "boundary_sha256": sha256_json(boundary),
                "recording_context_id": context_id,
                "recording_started_at": started_at,
                "context_close_and_HAR_flush_completed_at": ended_at,
            },
            indent=2,
        )
    )


def _resolve(value: str, site_urls: dict[str, str]) -> str:
    for site, endpoint in site_urls.items():
        value = value.replace(f"__{site.upper()}__", endpoint.rstrip("/"))
    return value


def _url_identity(value: str) -> tuple[str, str, str]:
    parsed = urlsplit(value)
    return parsed.scheme, parsed.netloc, parsed.path.rstrip("/")


def _contains(text: str, value: Any) -> bool:
    return " ".join(str(value).split()) in " ".join(text.split())


def _readout_matches(block: Any, readouts: dict[str, Any], parameters: dict[str, Any]) -> bool:
    for parameter, readout in readouts.items():
        elements = block.select(readout["selector"])
        if len(elements) != 1:
            return False
        element = elements[0]
        value = (
            element.get("value")
            if readout["kind"] == "input_value"
            else element.get_text(" ", strip=True)
        )
        if value is None:
            return False
        try:
            actual = parse_fact_value(value, readout["value_type"])
        except ValueError:
            return False
        if actual != parse_fact_value(parameters[parameter], readout["value_type"]):
            return False
    return True


def _source_facts(
    event: Any, identity: str, label: str, readouts: dict[str, Any], parameters: dict[str, Any]
) -> bool:
    content = event.data["response"].get("content", {})
    text = content.get("text")
    if not isinstance(text, str):
        return False
    if content.get("encoding") == "base64":
        text = base64.b64decode(text).decode("utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    if _url_identity(event.url) == _url_identity(identity):
        body = soup.get_text(" ", strip=True)
        return _contains(body, label) and _readout_matches(soup, readouts, parameters)
    # Lists, carts and search results can expose the same entity. Its link and
    # facts must share a local row/card, rather than matching another item's price.
    for anchor in soup.select("a[href]"):
        if _url_identity(urljoin(event.url, anchor["href"])) != _url_identity(identity):
            continue
        block = anchor.find_parent(
            lambda tag: (
                tag.name in {"tr", "li", "article"}
                or (
                    tag.name == "div"
                    and any(
                        cls
                        in {"product-item-info", "product-item", "product-card", "product", "item"}
                        for cls in tag.get("class", [])
                    )
                )
            )
        )
        if block is not None:
            body = block.get_text(" ", strip=True)
            if _contains(body, label) and _readout_matches(block, readouts, parameters):
                return True
    return False


def _window(event: Any) -> tuple[datetime, datetime] | None:
    raw = event.data
    started, duration = raw.get("startedDateTime"), raw.get("time")
    if not isinstance(started, str) or type(duration) not in {int, float}:
        return None
    timings = raw.get("timings", {})
    if (
        not math.isfinite(duration)
        or duration < 0
        or any(
            type(timings.get(key)) not in {int, float}
            or not math.isfinite(timings[key])
            or timings[key] < 0
            for key in ("send", "wait", "receive")
        )
    ):
        return None
    start = datetime.fromisoformat(started)
    if start.tzinfo is None:
        return None
    return start, start + timedelta(milliseconds=duration)


def evaluate_dependencies(
    candidate: dict[str, Any],
    trace: Any,
    site_urls: dict[str, str],
    frames: list[dict[str, Any]],
    boundary: dict[str, Any],
    expected_scope: str,
) -> dict[str, Any]:
    """Bind source facts to a real target request using one original episode."""
    from webarena_verified.core.evaluation.evaluators import EVALUATOR_REGISTRY
    from webarena_verified.types import WebArenaSite, WebArenaVerifiedTask
    from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
    from webarena_verified.types.eval import TaskEvalContext

    definition = candidate["task_definition"]
    validate_dependencies(definition, candidate["task_spec"]["sites"])
    if not definition["dependencies"]:
        return {"status": "pass", "matches": {}, "reason": "single-site task"}
    context_id = boundary["recording_context_id"]
    if (
        boundary["scope"] != expected_scope
        or boundary["har_context_id"] != context_id
        or boundary["playwright_context_id"] != context_id
        or context_id in boundary["audit_context_ids"]
    ):
        return {"status": "hold", "matches": {}, "reason": "original recording scope unproven"}
    start, end = (datetime.fromisoformat(boundary[key]) for key in ("started_at", "ended_at"))
    task = WebArenaVerifiedTask.model_validate(candidate["task_spec"])
    config = WebArenaVerifiedConfig(
        test_data_file=Path("private.jsonl"),
        environments={
            WebArenaSite(site): EnvironmentConfig(urls=[url]) for site, url in site_urls.items()
        },
    )
    network_evals = task.network_event_evaluator_cfgs
    matches = {}
    for dependency in definition["dependencies"]:
        parameters = definition["parameters"]
        entity = dependency["source_entity"]
        identity = _resolve(str(parameters[entity["identity_parameter"]]), site_urls)
        label = str(parameters[entity["label_parameter"]])
        readouts = dependency["fact_readouts"]
        facts = {name: parameters[name] for name in readouts}
        source_origin = _url_identity(site_urls[dependency["source_site"]])[:2]
        source_reads = [
            (index, event)
            for index, event in enumerate(trace.events)
            if event.http_method == "GET"
            and 200 <= event.request_status < 300
            and _url_identity(event.url)[:2] == source_origin
            and "text/html" in event.data["response"].get("content", {}).get("mimeType", "")
        ]
        source_events = [
            (index, event)
            for index, event in source_reads
            if _source_facts(event, identity, label, readouts, parameters)
        ]
        target_config = network_evals[dependency["target_event_index"]]
        event_template = json.dumps(
            definition["completion_events"][dependency["target_event_index"]]
        )
        pending_parameters = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", event_template)
        used_parameters = set()
        rules = {
            record["field"].removeprefix("/task_definition/parameters/"): record["rule"]
            for record in candidate["grounding"]["field_evidence"]
            if record["field"].startswith("/task_definition/parameters/") and "rule" in record
        }
        while pending_parameters:
            parameter = pending_parameters.pop()
            if parameter in used_parameters:
                continue
            used_parameters.add(parameter)
            if parameter in rules:
                pending_parameters.extend(rules[parameter]["parameters"])
        if not set(readouts) <= used_parameters:
            return {
                "status": "fail",
                "matches": matches,
                "reason": "source inputs do not derive target values",
            }
        target_events = []
        evaluator = EVALUATOR_REGISTRY["NetworkEventEvaluator"]()
        for index, event in enumerate(trace.events):
            one_event = trace.model_copy(update={"events": (event,)})
            result = evaluator.evaluate(
                context=TaskEvalContext(
                    task=task,
                    agent_response_raw=json.dumps(candidate["reference_response"]),
                    network_trace=one_event,
                    config=config,
                ),
                config=target_config,
            )
            if result.score == 1 and result.status == "success":
                target_events.append((index, event))
        source_documents = {
            _url_identity(_resolve(str(parameters[dependency["source_url_parameter"]]), site_urls)),
            _url_identity(identity),
        }
        source_documents.update(
            _url_identity(frame["page_url"])
            for frame in frames
            if frame["scope"] == expected_scope
            and frame["context_id"] == context_id
            and _url_identity(frame["page_url"])[:2] == source_origin
        )
        if not source_events and any(
            _url_identity(event.url) in source_documents
            and not isinstance(event.data["response"].get("content", {}).get("text"), str)
            for _, event in source_reads
        ):
            return {
                "status": "hold",
                "matches": matches,
                "reason": "original source response body unavailable",
            }
        if not source_events or not target_events:
            return {
                "status": "fail",
                "matches": matches,
                "reason": "source entity/facts or target missing",
            }
        incomplete_timing = False
        selected = None
        for source_index, source in source_events:
            source_window = _window(source)
            for target_index, target in target_events:
                target_window = _window(target)
                if source_window is None or target_window is None:
                    incomplete_timing = True
                    continue
                if (
                    not start
                    <= source_window[0]
                    <= source_window[1]
                    <= target_window[0]
                    <= target_window[1]
                    <= end
                ):
                    continue
                for frame in frames:
                    frame_time = datetime.fromisoformat(frame["at_utc"])
                    if (
                        frame["context_id"] == context_id
                        and frame["scope"] == expected_scope
                        and _url_identity(frame["page_url"]) == _url_identity(source.url)
                        and source_window[1] <= frame_time <= target_window[0]
                        and _contains(frame["body_text"], label)
                        and (
                            (
                                _url_identity(frame["page_url"]) == _url_identity(identity)
                                and _readout_matches(
                                    BeautifulSoup(frame["visible_html"], "html.parser"),
                                    readouts,
                                    parameters,
                                )
                            )
                            or any(
                                _url_identity(link["href"]) == _url_identity(identity)
                                and _contains(link["body_text"], label)
                                and _readout_matches(
                                    BeautifulSoup(link["visible_html"], "html.parser"),
                                    readouts,
                                    parameters,
                                )
                                for link in frame["links"]
                            )
                        )
                    ):
                        selected = {
                            "source_event_index": source_index,
                            "target_event_index": target_index,
                            "gui_screenshot_sha256": frame["screenshot_sha256"],
                            "source_entity": identity,
                            "facts": facts,
                        }
                        break
                if selected:
                    break
            if selected:
                break
        if selected is None:
            return {
                "status": "hold" if incomplete_timing else "fail",
                "matches": matches,
                "reason": "source-to-target timing or original visible evidence absent",
            }
        matches[dependency["name"]] = selected
    return {
        "status": "pass",
        "matches": matches,
        "reason": "original visible source entities and facts precede their target use",
    }


def evaluate_recording(
    candidate: dict[str, Any],
    traces: dict[str, Any],
    site_urls: dict[str, str],
    *,
    artifact_root: Path,
    har_path: Path,
    trace_path: Path,
    expected_scope: str,
) -> dict[str, Any]:
    """Verify recording hashes, then require the same dependency conclusion in both traces."""
    if not candidate["task_definition"]["dependencies"]:
        return {
            "cross_site_dependency_status": "pass",
            "cross_site_results": {},
            "checks": {"cross_site_dependencies_pass": True},
        }
    if expected_scope == "author_context_only":
        proof = candidate["grounding"]["episode_boundary_artifact"]
        boundary_path = (artifact_root / proof["path"]).resolve()
        if (
            not boundary_path.is_relative_to(artifact_root.resolve())
            or not boundary_path.is_file()
            or sha256_file(boundary_path) != proof["sha256"]
        ):
            return {
                "cross_site_dependency_status": "hold",
                "cross_site_results": {},
                "checks": {"cross_site_dependencies_pass": False},
            }
    else:
        boundary_path = artifact_root / "episode_boundary.json"
    recording_root = boundary_path.parent
    frames_path = recording_root / "cross_site_gui_frames.jsonl"
    if not boundary_path.is_file() or not frames_path.is_file():
        return {
            "cross_site_dependency_status": "hold",
            "cross_site_results": {},
            "checks": {"cross_site_dependencies_pass": False},
        }
    boundary = json.loads(boundary_path.read_text())
    hashes = {
        "har": sha256_file(har_path),
        "trace": sha256_file(trace_path),
        "gui_frames": sha256_file(frames_path),
    }
    if boundary["artifact_sha256"] != hashes:
        return {
            "cross_site_dependency_status": "hold",
            "cross_site_results": {},
            "checks": {"cross_site_dependencies_pass": False},
        }
    frames = [json.loads(line) for line in frames_path.read_text().splitlines() if line.strip()]
    for frame in frames:
        path = (recording_root / frame["screenshot"]).resolve()
        if (
            not path.is_relative_to(recording_root.resolve())
            or not path.is_file()
            or sha256_file(path) != frame["screenshot_sha256"]
        ):
            return {
                "cross_site_dependency_status": "hold",
                "cross_site_results": {},
                "checks": {"cross_site_dependencies_pass": False},
            }
    results = {
        name: evaluate_dependencies(candidate, trace, site_urls, frames, boundary, expected_scope)
        for name, trace in traces.items()
    }
    statuses = [result["status"] for result in results.values()]
    status = "hold" if "hold" in statuses or len(set(statuses)) != 1 else statuses[0]
    return {
        "cross_site_dependency_status": status,
        "cross_site_results": results,
        "cross_site_contract_sha256": sha256_json(candidate["task_definition"]["dependencies"]),
        "cross_site_recording_boundary_sha256": sha256_file(boundary_path),
        "checks": {"cross_site_dependencies_pass": status == "pass"},
    }
