#!/usr/bin/env python3
"""Ground cross-article table comparisons in the fixed offline Wikipedia book.

Coordinator runtime example (requires Playwright and installed Chromium):
  /opt/venv/bin/python -B /outputs/infra/author_wikipedia.py \
    --task-config /outputs/plans/wikipedia_task_configs_0058.json \
    --baseline-receipt /outputs/private/wikipedia_author_baseline.json \
    --site-url http://localhost:8888 --batch-root /outputs

The coordinator restores the baseline. This read-only author entry follows the
visible book link and article links, captures rendered tables, and emits the
existing author.compile_candidate draft. Concrete answers come from this new
run; source hashing and evaluator construction remain with the author owner.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--storage-state", type=Path)
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--attempt", default="grounding-01")
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    args = parser.parse_args(argv)
    site = urlsplit(args.site_url)
    if (
        site.scheme not in {"http", "https"}
        or not site.netloc
        or site.username
        or site.password
        or site.query
        or site.fragment
        or site.path not in {"", "/"}
    ):
        parser.error("site URL must be an explicit HTTP(S) origin without credentials")
    if not re.fullmatch(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", args.attempt):
        parser.error("attempt must be a simple identifier")
    if args.timeout_ms <= 0 or args.settle_seconds < 0:
        parser.error("timeout must be positive and settle seconds nonnegative")
    return args


def _record(page: Any, output: Path, name: str) -> dict[str, Any]:
    """Archive rendered article titles, table cells, and visible linked labels."""
    observation = page.evaluate(
        """() => {
          const visible = e => {
            const r = e.getBoundingClientRect();
            return !!(r.width && r.height) && getComputedStyle(e).visibility !== 'hidden';
          };
          const text = e => e.innerText.trim();
          const links = root => Array.from(root.querySelectorAll('a[href]'))
            .filter(visible).map(a => ({text: text(a), url: a.href}));
          const headings = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6'))
            .filter(visible);
          return {
            url: location.href, title: document.title, body_text: document.body.innerText,
            article_title: headings.find(h => h.tagName === 'H1')?.innerText.trim() || null,
            links: links(document),
            tables: Array.from(document.querySelectorAll('table')).filter(visible).map(t => ({
              text: text(t),
              section: headings.filter(h => h.compareDocumentPosition(t) &
                Node.DOCUMENT_POSITION_FOLLOWING).at(-1)?.innerText.trim() || null,
              groups: Array.from(t.querySelectorAll('.sidebar-list')).filter(visible)
                .map(group => ({
                  label: group.querySelector('.sidebar-list-title')?.innerText.trim() || null,
                  links: links(group)
                })),
              rows: Array.from(t.rows).filter(row => row.closest('table') === t)
                .filter(visible).map(row => ({
                  cells: Array.from(row.cells).filter(visible)
                    .map(cell => ({text: text(cell), links: links(cell)}))
                }))
            }))
          };
        }"""
    )
    page.screenshot(path=str(output / f"{name}.png"))
    (output / f"{name}.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
    )
    return observation


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = json.loads(args.task_config.read_text())
    if config["mode"] not in {"compare-sidebar-branches", "compare-filtered-table-topics"}:
        raise ValueError("config must select an evidenced cross-article table family")
    if not re.fullmatch(r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", config["generated_task_id"]):
        raise ValueError("task ID must use the generated namespace")
    baseline = json.loads(args.baseline_receipt.read_text())
    site_url = args.site_url.rstrip("/")
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("authoring requires a coordinator-owned fixed image baseline")
    if baseline["site_url"].rstrip("/") != site_url:
        raise ValueError("author endpoint differs from the coordinator baseline receipt")
    os.umask(0o077)
    output = args.batch_root / "author" / config["generated_task_id"] / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    parameters = dict(config["parameters"])
    site = urlsplit(site_url)
    blocked = []

    def restrict(route):
        requested = urlsplit(route.request.url)
        if (requested.scheme, requested.netloc) != (
            site.scheme,
            site.netloc,
        ) or route.request.method not in {"GET", "HEAD"}:
            blocked.append({"method": route.request.method, "path": requested.path})
            route.abort()
        else:
            route.continue_()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = None
        trace_started = False
        try:
            context = browser.new_context(
                storage_state=str(args.storage_state) if args.storage_state else None,
                viewport={"width": 1920, "height": 1080},
                service_workers="block",
                accept_downloads=False,
                record_har_path=str(output / "network.har"),
                record_har_content="embed",
            )
            context.route("**/*", restrict)
            context.set_default_timeout(args.timeout_ms)
            context.tracing.start(screenshots=True, snapshots=True, sources=False)
            trace_started = True
            page = context.new_page()
            page.goto(site_url + "/", wait_until="domcontentloaded", timeout=args.timeout_ms)
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            initial = _record(page, output, "initial")
            book_url = config["book_link"].replace("__WIKIPEDIA__", site_url)
            if not any(link["url"] == book_url for link in initial["links"]):
                raise ValueError("the configured offline book is absent from the visible library")
            page.goto(book_url, wait_until="domcontentloaded", timeout=args.timeout_ms)
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            landing = _record(page, output, "landing")
            articles = []
            collected = []
            for index, target in enumerate(config["article_links"]):
                url = target["url"].replace("__WIKIPEDIA__", site_url)
                article = parameters[f"article_{index + 1}"]
                if not any(
                    link["url"] == url and link["text"] == article for link in landing["links"]
                ):
                    raise ValueError("article title and URL are absent from the fresh landing UI")
                page.goto(url, wait_until="domcontentloaded", timeout=args.timeout_ms)
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                observation = _record(page, output, f"article_{index:02}")
                if observation["article_title"] != article:
                    raise ValueError("article navigation reached a different visible title")
                if config["mode"] == "compare-sidebar-branches":
                    tables = [
                        (number, table)
                        for number, table in enumerate(observation["tables"])
                        if table["text"].startswith("Part of a series on")
                        and any(
                            group["label"] == parameters["group_label"] for group in table["groups"]
                        )
                    ]
                    if len(tables) != 1:
                        raise ValueError("the article's series sidebar is missing or ambiguous")
                    table_index, table = tables[0]
                    groups = [
                        (number, group)
                        for number, group in enumerate(table["groups"])
                        if group["label"] == parameters["group_label"]
                    ]
                    if len(groups) != 1:
                        raise ValueError("the visible branch group is missing or ambiguous")
                    group_index, group = groups[0]
                    links = [
                        (
                            link,
                            f"/articles/{index}/tables/{table_index}/groups/{group_index}/links/{number}/text",
                        )
                        for number, link in enumerate(group["links"])
                        if link["text"]
                    ]
                    count_field, names_field = "branch_count", "branch_names"
                else:
                    scope = config["table_scopes"][index]
                    if "headers" in scope:
                        tables = [
                            (number, table)
                            for number, table in enumerate(observation["tables"])
                            if table["rows"]
                            and [cell["text"] for cell in table["rows"][0]["cells"]]
                            == scope["headers"]
                        ]
                    else:
                        tables = [
                            (number, table)
                            for number, table in enumerate(observation["tables"])
                            if table["section"] == scope["section"]
                        ]
                    if len(tables) != 1:
                        raise ValueError(
                            "the requested visible article table is missing or ambiguous"
                        )
                    table_index, table = tables[0]
                    links = []
                    for row_index, row in enumerate(table["rows"]):
                        if "row_label" in scope and (
                            not row["cells"] or row["cells"][0]["text"] != scope["row_label"]
                        ):
                            continue
                        for cell_index, cell in enumerate(row["cells"]):
                            if "column" in scope and cell_index != scope["column"]:
                                continue
                            for link_index, link in enumerate(cell["links"]):
                                if (
                                    link["text"]
                                    and parameters["label_contains"].casefold()
                                    in link["text"].casefold()
                                ):
                                    links.append(
                                        (
                                            link,
                                            f"/articles/{index}/tables/{table_index}/rows/{row_index}/cells/{cell_index}/links/{link_index}/text",
                                        )
                                    )
                    count_field, names_field = "topic_count", "topic_names"
                # Duplicate visible labels count once, as specified in the query.
                distinct = {}
                for link, location in links:
                    distinct.setdefault(link["text"], location)
                names = sorted(distinct, key=str.casefold)
                if not names:
                    raise ValueError("the requested table group contains no grounded linked topics")
                collected.append(
                    (
                        index,
                        {"article": article, count_field: len(names), names_field: names},
                        distinct,
                    )
                )
                # Capture the selected table in the author viewport as well as
                # archiving all rendered raw cells for every answer leaf.
                focus = page.locator("table:visible").nth(table_index)
                if config["mode"] == "compare-filtered-table-topics" and "row_label" in scope:
                    focus = focus.locator("tr:visible").filter(
                        has_text=re.compile(r"^\s*" + re.escape(scope["row_label"]) + r"(?:\s|$)")
                    )
                focus.scroll_into_view_if_needed()
                _record(page, output, f"article_{index:02}_focus")
                articles.append(observation)
            ranked = sorted(collected, key=lambda item: (-item[1][count_field], item[1]["article"]))
            values = [value for _, value, _ in ranked]
            if config["mode"] == "compare-sidebar-branches":
                goal = (
                    "Compare the {article_1} and {article_2} articles in the offline "
                    "Wikipedia book. "
                    "For each, use only its 'Part of a series on' sidebar and the {group_label} "
                    "group. Report article, branch_count, and branch_names. Count each distinct "
                    "linked branch label once and list the exact displayed labels alphabetically, "
                    "ignoring case. Order articles by branch_count descending, then article title "
                    "ascending. Exclude the separate Modern physics box and bottom "
                    "navigation boxes."
                )
            else:
                goal = (
                    "Compare linked topic labels containing '{label_contains}' (ignore case) in "
                    "two offline Wikipedia article tables: in {article_1}, use only the Subfields "
                    "cell of the {field_row} row in the table headed Field, Subfields, Major "
                    "theories, Concepts; in {article_2}, use the topic table immediately under "
                    "{section_heading}. Report article, topic_count, and topic_names. Count each "
                    "distinct matching linked label once and list its exact displayed label "
                    "alphabetically, ignoring case. Order articles by topic_count descending, "
                    "then article title ascending. Exclude prose links outside those tables."
                )
            definition = {
                "task_type": "RETRIEVE",
                "goal_template": goal,
                "parameters": parameters,
                "answer": {
                    "status": "SUCCESS",
                    "values": values,
                    "ordered": True,
                    "item_schema": {
                        "type": "object",
                        "properties": {
                            "article": {"type": "string"},
                            count_field: {"type": "integer"},
                            names_field: {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["article", count_field, names_field],
                        "additionalProperties": False,
                    },
                },
                "completion_events": [],
            }
            evidence = []
            for answer_index, (article_index, value, locations) in enumerate(ranked):
                evidence.extend(
                    [
                        {
                            "field": f"/task_definition/answer/values/{answer_index}/article",
                            "artifact": "observation",
                            "location": f"/articles/{article_index}/article_title",
                            "modality": "text",
                        },
                        {
                            "field": f"/task_definition/answer/values/{answer_index}/{count_field}",
                            "artifact": "observation",
                            "location": f"/articles/{article_index}/tables",
                            "modality": "text",
                            "derivation": (
                                "Count distinct visible linked labels in the requested "
                                "group/table scope."
                            ),
                        },
                    ]
                )
                evidence.extend(
                    {
                        "field": (
                            f"/task_definition/answer/values/{answer_index}/{names_field}/{number}"
                        ),
                        "artifact": "observation",
                        "location": locations[name],
                        "modality": "text",
                    }
                    for number, name in enumerate(value[names_field])
                )
            archived = {
                "initial": initial,
                "landing": landing,
                "articles": articles,
                "blocked_requests": blocked,
            }
        finally:
            try:
                if context is not None and trace_started:
                    context.tracing.stop(path=str(output / "trace.zip"))
            finally:
                try:
                    if context is not None:
                        context.close()
                finally:
                    browser.close()
    (output / "observation.json").write_text(
        json.dumps(archived, ensure_ascii=False, indent=2) + "\n"
    )
    relative = output.relative_to(args.batch_root)
    draft = {
        "generated_task_id": config["generated_task_id"],
        "task_id": config["task_id"],
        "intent_template_id": config["intent_template_id"],
        "site": "wikipedia",
        "start_url": "__WIKIPEDIA__",
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
                "screenshot": str(relative / "article_00_focus.png"),
                "observation": str(relative / "observation.json"),
            },
            "field_evidence": evidence,
            "final_state_checks": [],
        },
    }
    (output / "draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "observed",
                "generated_task_id": config["generated_task_id"],
                "draft": str(output / "draft.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1) from None
