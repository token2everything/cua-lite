#!/usr/bin/env python3
"""Ground three read-only GitLab families from fresh visible baseline UI.

Run inside the coordinator-owned prepared runtime with Playwright/Chromium:
  /opt/venv/bin/python -B /outputs/infra/author_gitlab.py \
    --task-config /outputs/plans/gitlab_task_configs_0017.json \
    --baseline-receipt /outputs/private/gitlab_author_baseline.json \
    --storage-state /outputs/private/gitlab_storage_state.json \
    --site-url http://localhost:8023 --batch-root /outputs

The coordinator creates/restores the fixed baseline and supplies private auth.
This entry starts only its own browser, follows actual visible links, captures
HAR/trace/screenshots, and emits the existing author.compile_candidate draft.
It never changes business assets, provisions a site, or calls a model. Source
hashes and upstream evaluator construction remain with the existing author owner.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--attempt", default="grounding-01")
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
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
    """Archive only rendered page facts and actual visible navigation targets."""
    observation = page.evaluate(
        r"""() => {
          const visible = e => {
            const r = e.getBoundingClientRect();
            return !!(r.width && r.height) && getComputedStyle(e).visibility !== 'hidden';
          };
          const text = e => e.innerText.trim();
          const links = root => Array.from(root.querySelectorAll('a[href]'))
            .filter(visible).map(a => ({text: text(a), url: a.href}));
          const heading = Array.from(document.querySelectorAll('h1.home-panel-title'))
            .find(visible);
          return {
            url: location.href, title: document.title, body_text: document.body.innerText,
            project_heading: heading ? text(heading) : null,
            headings: Array.from(document.querySelectorAll('h1,h2,h3'))
              .filter(visible).map(text),
            links: links(document),
            controls: Array.from(document.querySelectorAll('input,select,textarea,button'))
              .filter(visible).map(e => ({
                tag: e.tagName.toLowerCase(), type: e.type, name: e.name,
                text: text(e), aria_label: e.getAttribute('aria-label'),
                placeholder: e.getAttribute('placeholder'),
                value: e.type === 'password' ? undefined : e.value
              })),
            issues: Array.from(document.querySelectorAll('li.issue')).filter(visible)
              .map(row => {
                const rowLinks = links(row);
                const title = rowLinks.find(link => {
                  const url = new URL(link.url);
                  return /\/-\/issues\/\d+$/.test(url.pathname) && !url.hash;
                });
                const reference = row.querySelector('.issuable-reference');
                const author = row.querySelector('.author');
                const comments = rowLinks.find(link => new URL(link.url).hash === '#notes');
                return {
                  title: title ? title.text : null, url: title ? title.url : null,
                  reference: reference && visible(reference) ? text(reference) : null,
                  author: author && visible(author) ? text(author) : null,
                  labels: Array.from(row.querySelectorAll('.gl-label-text'))
                    .filter(visible).map(text),
                  comments_text: comments ? comments.text : null,
                  body_text: text(row), links: rowLinks
                };
              })
          };
        }"""
    )
    page.screenshot(path=str(output / f"{name}.png"))
    (output / f"{name}.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
    )
    return observation


def _project_fields(observation: dict[str, Any]) -> dict[str, Any]:
    """Parse the explicitly labelled visible project-home statistics."""
    body = observation["body_text"]
    result = {
        "project_path": urlsplit(observation["url"]).path.rstrip("/"),
        "project_name": observation["project_heading"],
    }
    if not result["project_name"]:
        raise ValueError("the project homepage has no visible project title")
    for field, label in {
        "commits": r"Commits?",
        "branches": r"Branch(?:es)?",
        "tags": r"Tags?",
    }.items():
        matches = re.findall(r"(?<![\d,.])([\d,]+)\s+" + label + r"\b", body)
        if len(matches) != 1:
            raise ValueError("visible project statistics are missing or ambiguous")
        result[field] = int(matches[0].replace(",", ""))
    storage = re.findall(r"([\d.,]+\s+(?:B|KB|MB|GB|TB))\s+Project Storage\b", body)
    if len(storage) != 1:
        raise ValueError("visible project storage is missing or ambiguous")
    result["project_storage"] = storage[0]
    return result


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = json.loads(args.task_config.read_text())
    modes = {"retrieve-project-metadata", "navigate-project-metadata", "retrieve-assigned-issues"}
    if config["mode"] not in modes:
        raise ValueError("task config must select one of the three grounded read-only families")
    if not re.fullmatch(r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", config["generated_task_id"]):
        raise ValueError("generated task ID must use the generated namespace")
    if config["task_id"] < 2_000_000 or config["intent_template_id"] < 2_000_000:
        raise ValueError("task and template IDs must use the generated range")
    baseline = json.loads(args.baseline_receipt.read_text())
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("authoring requires a coordinator-owned fixed image baseline")
    site_url = args.site_url.rstrip("/")
    if baseline["site_url"].rstrip("/") != site_url:
        raise ValueError("author endpoint differs from the coordinator baseline receipt")
    os.umask(0o077)
    output = args.batch_root / "author" / config["generated_task_id"] / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    parameters = dict(config["parameters"])
    blocked_requests = []
    site = urlsplit(site_url)

    def restrict(route):
        requested = urlsplit(route.request.url)
        if (requested.scheme, requested.netloc) != (
            site.scheme,
            site.netloc,
        ) or route.request.method not in {"GET", "HEAD"}:
            blocked_requests.append({"method": route.request.method, "path": requested.path})
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
            trace_started = True
            page = context.new_page()
            page.goto(site_url + "/", wait_until="domcontentloaded", timeout=args.timeout_ms)
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            initial = _record(page, output, "initial")
            pages = []
            if config["mode"] in {"retrieve-project-metadata", "navigate-project-metadata"}:
                projects = []
                for index, target in enumerate(config["project_links"]):
                    expected_url = target["url"].replace("__GITLAB__", site_url)
                    # Initial UI must expose both this target URL and its visible
                    # project label; configuration alone never authorizes a guess.
                    matches = [
                        link
                        for link in initial["links"]
                        if link["url"] == expected_url
                        and " ".join(link["text"].split()) == parameters[f"project_{index + 1}"]
                    ]
                    if not matches:
                        raise ValueError(
                            "configured project is absent from the fresh visible dashboard"
                        )
                    page.goto(
                        matches[0]["url"], wait_until="domcontentloaded", timeout=args.timeout_ms
                    )
                    page.wait_for_timeout(int(args.settle_seconds * 1000))
                    observation = _record(page, output, f"project_{index:02}")
                    if urlsplit(observation["url"]).path.rstrip("/") != urlsplit(
                        expected_url
                    ).path.rstrip("/"):
                        raise ValueError("project navigation did not reach the observed homepage")
                    pages.append(observation)
                    projects.append(_project_fields(observation))
                archived = {"initial": initial, "projects": pages}
                final_checks = []
                if config["mode"] == "retrieve-project-metadata":
                    ranked = sorted(
                        enumerate(projects),
                        key=lambda item: (-item[1]["commits"], item[1]["project_path"]),
                    )
                    values = [project for _, project in ranked]
                    definition = {
                        "task_type": "RETRIEVE",
                        "goal_template": (
                            "Compare the existing projects {project_1}, {project_2}, "
                            "and {project_3}. From each project's homepage, report project_path "
                            "(the path beginning with /), project_name, commits, branches, "
                            "tags, and project_storage (the exact displayed value and unit). "
                            "Use the homepage totals, not the file-tree rows. Order the "
                            "projects by commits descending; for equal commits, use "
                            "project_path ascending. "
                            "Do not change or create project data."
                        ),
                        "parameters": parameters,
                        "answer": {
                            "status": "SUCCESS",
                            "values": values,
                            "ordered": True,
                            "item_schema": {
                                "type": "object",
                                "properties": {
                                    field: {
                                        "type": "integer"
                                        if field in {"commits", "branches", "tags"}
                                        else "string"
                                    }
                                    for field in values[0]
                                },
                                "required": list(values[0]),
                                "additionalProperties": False,
                            },
                        },
                        "completion_events": [],
                    }
                    evidence = [
                        {
                            "field": f"/task_definition/answer/values/{answer_index}/{field}",
                            "artifact": "observation",
                            "modality": "text",
                            "location": f"/projects/{page_index}/"
                            + (
                                "url"
                                if field == "project_path"
                                else "project_heading"
                                if field == "project_name"
                                else "body_text"
                            ),
                            "derivation": "Path of actual observed URL."
                            if field == "project_path"
                            else "Exact visible title."
                            if field == "project_name"
                            else (
                                "Parse the named visible homepage statistic; integer counts "
                                "omit commas and storage retains its displayed unit."
                            ),
                        }
                        for answer_index, (page_index, project) in enumerate(ranked)
                        for field in project
                    ]
                    screenshot = "project_00.png"
                else:
                    matches = [
                        index
                        for index, project in enumerate(projects)
                        if project["branches"] >= parameters["minimum_branches"]
                        and project["tags"] >= parameters["minimum_tags"]
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "the fresh visible metadata does not identify exactly one project"
                        )
                    selected = matches[0]
                    page.goto(
                        pages[selected]["url"],
                        wait_until="domcontentloaded",
                        timeout=args.timeout_ms,
                    )
                    page.wait_for_timeout(int(args.settle_seconds * 1000))
                    terminal = _record(page, output, "terminal")
                    if _project_fields(terminal) != projects[selected]:
                        raise ValueError("terminal navigation did not retain the selected project")
                    parameters["project_url"] = "__GITLAB__" + urlsplit(terminal["url"]).path
                    parameters["project_name"] = terminal["project_heading"]
                    definition = {
                        "task_type": "NAVIGATE",
                        "goal_template": (
                            "Among {project_1}, {project_2}, and {project_3}, find the "
                            "project whose "
                            "homepage shows at least {minimum_branches} branches and at least "
                            "{minimum_tags} tags. Open that project's homepage and leave it open. "
                            "Use the homepage totals and do not change or create project data."
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
                                "url_template": "^__GITLAB__"
                                + re.escape(urlsplit(terminal["url"]).path.rstrip("/"))
                                + "/?(?:\\?.*)?$",
                                "http_method": "GET",
                                "response_status": 200,
                                "last_event_only": False,
                            }
                        ],
                    }
                    evidence = [
                        {
                            "field": "/task_definition/parameters/project_url",
                            "artifact": "observation",
                            "location": "/terminal/url",
                            "modality": "text",
                            "derivation": (
                                "Actual terminal URL with its site origin projected to __GITLAB__."
                            ),
                        },
                        {
                            "field": "/task_definition/parameters/project_name",
                            "artifact": "observation",
                            "location": "/terminal/project_heading",
                            "modality": "text",
                        },
                        {
                            "field": "/task_definition/completion_events/0/url_template",
                            "artifact": "observation",
                            "location": "/terminal/url",
                            "modality": "text",
                            "derivation": (
                                "Regex-escape the actual terminal project path and project "
                                "its site origin to __GITLAB__."
                            ),
                        },
                    ]
                    final_checks = [
                        {"kind": "final_url", "parameter": "project_url"},
                        {
                            "kind": "visible_text_exact",
                            "parameter": "project_name",
                            "selector": "h1.home-panel-title:visible",
                        },
                    ]
                    archived.update(matching_project_indices=matches, terminal=terminal)
                    screenshot = "terminal.png"
            else:
                target = config["assigned_issues_link"].replace("__GITLAB__", site_url)
                if not any(link["url"] == target for link in initial["links"]):
                    raise ValueError(
                        "assigned issue URL is absent from the fresh visible dashboard"
                    )
                page.goto(target, wait_until="domcontentloaded", timeout=args.timeout_ms)
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                assigned = _record(page, output, "assigned")
                open_links = [
                    link for link in assigned["links"] if re.fullmatch(r"Open\s+\d+", link["text"])
                ]
                if len(open_links) != 1:
                    raise ValueError("the visible open-issue tab is missing or ambiguous")
                page.goto(
                    open_links[0]["url"], wait_until="domcontentloaded", timeout=args.timeout_ms
                )
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                issues = _record(page, output, "issues")
                query = parse_qs(urlsplit(issues["url"]).query)
                if query.get("state") != ["opened"] or query.get("assignee_username") != [
                    parameters["assignee_username"]
                ]:
                    raise ValueError(
                        "the observed issue list does not have the requested open assignee scope"
                    )
                if not re.search(
                    r"Assignee\s*=\s*" + re.escape(parameters["assignee"]) + r"\b",
                    issues["body_text"],
                ):
                    raise ValueError("the requested assignee is absent from the visible filter")
                expected_count = int(open_links[0]["text"].split()[-1])
                rows = issues["issues"]
                if (
                    len(rows) != expected_count
                    or len({row["reference"] for row in rows}) != expected_count
                ):
                    raise ValueError(
                        "the visible issue list is incomplete; this family requires "
                        "its complete single-page scope"
                    )
                selected = [
                    (index, row)
                    for index, row in enumerate(rows)
                    if {parameters["label_1"], parameters["label_2"]} <= set(row["labels"])
                    and row["author"] == parameters["author"]
                ]
                selected.sort(
                    key=lambda item: (
                        item[1]["reference"].split("#")[0],
                        int(item[1]["reference"].split("#")[1]),
                    )
                )
                if not selected:
                    raise ValueError("no fresh visible issue satisfies the requested conditions")
                values = [
                    {
                        "issue_reference": row["reference"],
                        "title": row["title"],
                        "author": row["author"],
                        "comment_count": int(row["comments_text"]),
                    }
                    for _, row in selected
                ]
                definition = {
                    "task_type": "RETRIEVE",
                    "goal_template": (
                        "Find every currently open GitLab issue assigned to {assignee} "
                        "that has both "
                        "the {label_1} and {label_2} labels and was created by {author}. Report "
                        "issue_reference (the exact project-path#number displayed in the "
                        "issue list), title, author, and comment_count. Order by project "
                        "path ascending and then "
                        "numeric issue number ascending. Include every matching issue; do not "
                        "change or create project data."
                    ),
                    "parameters": parameters,
                    "answer": {
                        "status": "SUCCESS",
                        "values": values,
                        "ordered": True,
                        "item_schema": {
                            "type": "object",
                            "properties": {
                                field: {"type": "integer" if field == "comment_count" else "string"}
                                for field in values[0]
                            },
                            "required": list(values[0]),
                            "additionalProperties": False,
                        },
                    },
                    "completion_events": [],
                }
                evidence = [
                    {
                        "field": f"/task_definition/answer/values/{answer_index}/{field}",
                        "artifact": "observation",
                        "modality": "text",
                        "location": f"/issues/issues/{row_index}/"
                        + {
                            "issue_reference": "reference",
                            "title": "title",
                            "author": "author",
                            "comment_count": "comments_text",
                        }[field],
                        "derivation": (
                            "Exact visible issue-list field; comment count is parsed as an integer."
                        ),
                    }
                    for answer_index, (row_index, _) in enumerate(selected)
                    for field in values[answer_index]
                ]
                archived = {
                    "initial": initial,
                    "assigned": assigned,
                    "issues": issues,
                    "displayed_open_count": expected_count,
                    "complete_single_page_scope": True,
                }
                final_checks = []
                screenshot = "issues.png"
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
    archived["blocked_requests"] = blocked_requests
    (output / "observation.json").write_text(
        json.dumps(archived, ensure_ascii=False, indent=2) + "\n"
    )
    relative = output.relative_to(args.batch_root)
    draft = {
        "generated_task_id": config["generated_task_id"],
        "task_id": config["task_id"],
        "intent_template_id": config["intent_template_id"],
        "site": "gitlab",
        "start_url": "__GITLAB__",
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
        # Authentication state and provider call logs remain private, including
        # on failure. Failed grounding emits no candidate or fabricated GT.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1) from None
