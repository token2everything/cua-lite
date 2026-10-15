#!/usr/bin/env python3
"""Inspect GitLab editors or ground one mutation on a fresh fixed baseline.

Coordinator-owned runtime (Playwright/Chromium; sibling author_gitlab.py required):
  /opt/venv/bin/python -B /outputs/infra/author_gitlab_mutations.py \
    --mode prepare --task-config /outputs/plans/gitlab_task_configs_mutate.json \
    --baseline-receipt /outputs/private/gitlab_author_baseline.json \
    --storage-state /outputs/private/gitlab_storage_state.json \
    --site-url http://localhost:8023 --batch-root /outputs

For a formal case, use --mode formal --generated-task-id wavgen-v03-full300-0020
and --attempt grounding-01 with that case's new coordinator-owned baseline.
Formal authoring uses real UI Save, then revisit and refresh, and emits the
existing author.compile_candidate draft only after persisted state and the
canonical HAR request body agree. It never calls an API to write assets.

Prepare follows only links exposed by the fresh dashboard/project/issue UI. It
records editor forms and opens actual dropdown buttons without selecting items
or saving. The same-origin UI may POST only frozen, observed GraphQL query
documents; writes and unknown documents are blocked and recorded. It produces no
draft, candidate, GT, or claim of mutation readiness. Formal Save/readback modes
require the new editor evidence and a separate fresh baseline for every case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from author_gitlab import _project_fields, _record


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "formal"], required=True)
    parser.add_argument("--generated-task-id")
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--attempt", default="mutate-readiness-01")
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--dropdown-limit", type=int, default=4)
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
    if args.dropdown_limit not in range(5):
        parser.error("dropdown limit must be between zero and four")
    if args.mode == "formal" and not args.generated_task_id:
        parser.error("formal mode requires one generated task ID and one fresh baseline")
    return args


def _record_editor(page: Any, output: Path, name: str) -> dict[str, Any]:
    """Extend the existing GitLab observation owner with visible editor metadata."""
    observation = _record(page, output, name)
    observation.update(
        page.evaluate(
            """() => {
              const visible = e => {
                const r = e.getBoundingClientRect();
                return !!(r.width && r.height) && getComputedStyle(e).visibility !== 'hidden';
              };
              const text = e => e.innerText.trim();
              const form = e => e ? {id: e.id, action: e.action, method: e.method} : null;
              const one = selector => {
                const elements = Array.from(document.querySelectorAll(selector)).filter(visible);
                return elements.length === 1 ? elements[0].innerText : null;
              };
              return {
                issue_state: {
                  title: one('h1.title'),
                  description_text: one('div.description'),
                  assignee_text: one('[data-testid="assignee-block-container"]'),
                  labels: Array.from(document.querySelectorAll(
                    '[data-testid="sidebar-labels"] .gl-label-text')).filter(visible).map(text),
                  open: Array.from(document.querySelectorAll('button')).filter(visible)
                    .some(e => text(e) === 'Close issue'),
                  comments: Array.from(document.querySelectorAll('.note-text.md'))
                    .filter(visible).map(text),
                  checklist: Array.from(document.querySelectorAll(
                    'div.description input[type=checkbox]')).map(e => ({
                      checked: e.checked, label: text(e.closest('li'))
                    }))
                },
                forms: Array.from(document.forms).filter(visible).map(e => ({
                  ...form(e),
                  field_names: Array.from(e.elements).map(c => ({name: c.name, type: c.type}))
                })),
                editor_controls: Array.from(document.querySelectorAll(
                  'input,select,textarea,button,[contenteditable=true]'))
                  .filter(visible).map(e => ({
                    tag: e.tagName.toLowerCase(), type: e.type, name: e.name, id: e.id,
                    text: text(e), title: e.title, class_name: e.className,
                    aria_label: e.getAttribute('aria-label'),
                    aria_haspopup: e.getAttribute('aria-haspopup'),
                    data_toggle: e.getAttribute('data-toggle'),
                    dropdown_title: e.getAttribute('data-dropdown-title'),
                    field_name: e.getAttribute('data-field-name'),
                    placeholder: e.getAttribute('placeholder'),
                    value: e.type === 'password' ? undefined : e.value,
                    disabled: !!e.disabled,
                    labels: e.labels ? Array.from(e.labels).filter(visible).map(text) : [],
                    form: form(e.form),
                    options: e.tagName === 'SELECT' ? Array.from(e.options)
                      .map(o => ({label: o.textContent.trim(), value: o.value,
                        selected: o.selected})) : undefined
                  })),
                menu_items: Array.from(document.querySelectorAll(
                  '[role=option],[role=menuitem],[role=menuitemcheckbox]'))
                  .filter(visible).map(e => ({text: text(e), role: e.getAttribute('role'),
                    aria_selected: e.getAttribute('aria-selected'),
                    aria_checked: e.getAttribute('aria-checked')}))
              };
            }"""
        )
    )
    (output / f"{name}.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + "\n"
    )
    return observation


def ground_mutation(args: argparse.Namespace) -> None:
    """Produce a single case from real UI actions, persisted facts and native HAR."""
    config = json.loads(args.task_config.read_text())
    cases = [
        case for case in config["cases"] if case["generated_task_id"] == args.generated_task_id
    ]
    if len(cases) != 1:
        raise ValueError("formal task ID must identify one configured case")
    case = cases[0]
    family = case["family"]
    if family not in {
        "project_predicate_issue_creation",
        "filtered_issue_edit_with_preservation",
        "ranked_issue_comment_append_with_preservation",
    }:
        raise ValueError("this family lacks a canonical critical Save-body contract")
    baseline = json.loads(args.baseline_receipt.read_text())
    site_url = args.site_url.rstrip("/")
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("formal authoring requires a coordinator-owned fixed image baseline")
    if baseline["site_url"].rstrip("/") != site_url:
        raise ValueError("formal endpoint differs from the coordinator baseline receipt")
    os.umask(0o077)
    output = args.batch_root / "author" / case["generated_task_id"] / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    site = urlsplit(site_url)
    read_documents = set(config["observed_graphql_read_documents"])
    blocked = []
    save_phase = False

    def restrict(route):
        requested = urlsplit(route.request.url)
        same_origin = (requested.scheme, requested.netloc) == (site.scheme, site.netloc)
        read = route.request.method in {"GET", "HEAD"}
        if same_origin and route.request.method == "POST" and requested.path == "/api/graphql":
            payload = json.loads(route.request.post_data)
            items = payload if isinstance(payload, list) else [payload]
            read = bool(items) and all(
                isinstance(item, dict)
                and isinstance(item.get("query"), str)
                and hashlib.sha256(item["query"].encode()).hexdigest() in read_documents
                for item in items
            )
        if same_origin and (read or save_phase):
            route.continue_()
        else:
            blocked.append({"method": route.request.method, "path": requested.path})
            route.abort()

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
            page.goto(site_url + "/", wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            initial = _record_editor(page, output, "initial")
            archived = {"initial": initial, "blocked_requests": blocked}
            parameters = {}
            if family == "project_predicate_issue_creation":
                projects = []
                project_pages = []
                for index, target in enumerate(config["project_links"]):
                    url = target["url"].replace("__GITLAB__", site_url)
                    matches = [
                        link
                        for link in initial["links"]
                        if link["url"] == url and " ".join(link["text"].split()) == target["label"]
                    ]
                    if not matches:
                        raise ValueError("a configured project is absent from the fresh dashboard")
                    page.goto(matches[0]["url"], wait_until="domcontentloaded")
                    page.wait_for_timeout(int(args.settle_seconds * 1000))
                    observed = _record_editor(page, output, f"project_{index:02}")
                    project_pages.append(observed)
                    projects.append(_project_fields(observed))
                    parameters[f"project_{index + 1}"] = target["label"]
                selected = [
                    index
                    for index, project in enumerate(projects)
                    if project["branches"] >= config["project_selection"]["minimum_branches"]
                    and project["tags"] >= config["project_selection"]["minimum_tags"]
                ]
                if len(selected) != 1:
                    raise ValueError("fresh project predicate is not unique")
                selected_project = projects[selected[0]]
                archived.update(
                    projects=project_pages, project_facts=projects, selected_index=selected[0]
                )
                issue_urls = {
                    link["url"]
                    for link in project_pages[selected[0]]["links"]
                    if re.fullmatch(r"Issues(?:\s+\d+)?", link["text"])
                    and urlsplit(link["url"]).path.endswith("/-/issues")
                }
                if len(issue_urls) != 1:
                    raise ValueError("the actual project Issues link is missing or ambiguous")
                page.goto(next(iter(issue_urls)), wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                listing = _record_editor(page, output, "project_issues")
                new_urls = {link["url"] for link in listing["links"] if link["text"] == "New issue"}
                if len(new_urls) != 1:
                    raise ValueError("actual New issue URL is missing or ambiguous")
                page.goto(next(iter(new_urls)), wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                archived["editor_before"] = _record_editor(page, output, "editor_before")
                title = case["new_issue_title"]
                description = case["new_issue_description_template"].format_map(selected_project)
                page.get_by_label("Title (required)", exact=True).fill(title)
                page.get_by_label("Description", exact=True).fill(description)
                page.get_by_role("button", name="Unassigned", exact=True).click()
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                archived["assignee_options"] = _record_editor(page, output, "assignee_options")
                page.locator("a:visible").filter(
                    has_text=re.compile(r"^\s*Byte Blaze\s+@byteblaze\s*$")
                ).click()
                page.get_by_role("button", name="Labels", exact=True).click()
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                archived["label_options"] = _record_editor(page, output, "label_options")
                for label in case["existing_label_candidates"]:
                    page.get_by_role("link", name=label, exact=True).click()
                page.keyboard.press("Escape")
                save = page.get_by_role("button", name="Create issue", exact=True)
                parameters.update(
                    issue_title=title,
                    issue_description=description,
                    description_pattern=case["new_issue_description_template"],
                    minimum_branches=config["project_selection"]["minimum_branches"],
                    minimum_tags=config["project_selection"]["minimum_tags"],
                    label_1=case["existing_label_candidates"][0],
                    label_2=case["existing_label_candidates"][1],
                    assignee=case["assignee_candidate"],
                )
                goal = (
                    "Compare {project_1}, {project_2}, and {project_3} using their homepage branch "
                    "and tag totals. In the unique project with at least {minimum_branches} "
                    "branches and {minimum_tags} tags, create one issue titled {issue_title}. "
                    "Fill this description pattern using that project's actual totals: "
                    "{description_pattern} Assign {assignee} and apply "
                    "only the existing {label_1} and {label_2} labels. Keep the other defaults "
                    "unchanged and leave the new issue open on its detail page."
                )
                critical_values = [title, description]
            else:
                assigned_url = config["assigned_issues_link"].replace("__GITLAB__", site_url)
                if not any(link["url"] == assigned_url for link in initial["links"]):
                    raise ValueError("assigned issues are absent from the actual dashboard")
                page.goto(assigned_url, wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                dashboard = _record_editor(page, output, "assigned_dashboard")
                open_urls = {
                    link["url"]
                    for link in dashboard["links"]
                    if re.fullmatch(r"Open\s+\d+", link["text"])
                }
                if len(open_urls) != 1:
                    raise ValueError("actual Open issue tab is missing or ambiguous")
                page.goto(next(iter(open_urls)), wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                listing = _record_editor(page, output, "assigned_issues")
                counts = [
                    int(link["text"].split()[-1])
                    for link in listing["links"]
                    if re.fullmatch(r"Open\s+\d+", link["text"])
                ]
                if len(counts) != 1 or counts[0] != len(listing["issues"]):
                    raise ValueError(
                        "formal issue selection requires the complete visible open scope"
                    )
                matching = [
                    row
                    for row in listing["issues"]
                    if row["reference"].split("#")[0] == case["project_path"]
                    and set(case["required_labels"]) <= set(row["labels"])
                    and not set(case["excluded_labels"]) & set(row["labels"])
                    and (case["author"] is None or row["author"] == case["author"])
                ]
                if family == "ranked_issue_comment_append_with_preservation":
                    matching.sort(
                        key=lambda row: int(row["reference"].split("#")[1]),
                        reverse=case["rank_order"] == "descending",
                    )
                    if not matching:
                        raise ValueError("no actual issue satisfies the requested selection")
                    target = matching[0]
                else:
                    if len(matching) != 1:
                        raise ValueError("fresh issue-edit predicate is not unique")
                    target = matching[0]
                archived.update(
                    assigned_dashboard=dashboard,
                    assigned_issues=listing,
                    matching_issues=matching,
                    target=target,
                )
                page.goto(target["url"], wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                before = _record_editor(page, output, "before")
                archived["before"] = before
                if (
                    before["issue_state"]["title"] != target["title"]
                    or not before["issue_state"]["open"]
                ):
                    raise ValueError("the selected detail is not the fresh open issue")
                parameters.update(
                    project_path=case["project_path"],
                    assignee="Byte Blaze",
                    label_1=case["required_labels"][0],
                )
                if family == "ranked_issue_comment_append_with_preservation":
                    comment = case["comment_text"]
                    if comment in before["issue_state"]["comments"]:
                        raise ValueError(
                            "the supplied comment already exists in the fresh baseline"
                        )
                    page.get_by_test_id("comment-field").fill(comment)
                    save = page.get_by_role("button", name="Comment", exact=True)
                    parameters.update(
                        label_2=case["required_labels"][1],
                        comment=comment,
                        rank_order=case["rank_order"],
                    )
                    goal = (
                        "Among open issues assigned to {assignee} in {project_path} with both "
                        "{label_1} and {label_2} labels, order by numeric issue number "
                        "{rank_order} and choose the first. Append this exact comment once: "
                        "{comment} Preserve the title, description, labels, assignee, state, "
                        "checklist, and all existing comments. Leave that issue's detail page open."
                    )
                    critical_values = [comment]
                else:
                    page.get_by_role(
                        "button", name="Edit title and description", exact=True
                    ).click()
                    page.wait_for_timeout(int(args.settle_seconds * 1000))
                    archived["editor_before"] = _record_editor(page, output, "editor_before")
                    title = before["issue_state"]["title"] + case["title_suffix"]
                    description_before = page.get_by_label("Description", exact=True).input_value()
                    description = description_before + "\n\n" + case["description_append_text"]
                    page.get_by_label("Title", exact=True).fill(title)
                    page.get_by_label("Description", exact=True).fill(description)
                    save = page.get_by_role("button", name="Save changes", exact=True)
                    parameters.update(
                        title_suffix=case["title_suffix"],
                        description_append_text=case["description_append_text"],
                        author=case["author"],
                        excluded_label=case["excluded_labels"][0],
                    )
                    goal = (
                        "Among open issues assigned to {assignee} in {project_path}, select "
                        "the unique issue created by {author} with {label_1} but without "
                        "{excluded_label}. Append this suffix to its current title: {title_suffix} "
                        "Append this paragraph to the existing description: "
                        "{description_append_text} Preserve the original description, checklist, "
                        "labels, assignee, open state, "
                        "and existing comments; leave that issue's detail page open."
                    )
                    critical_values = [title, description]

            archived["unsaved"] = _record_editor(page, output, "unsaved")
            # Exactly one real GUI Save is issued. A transport failure leaves
            # artifacts and raises; authoring never retries a business write.
            save_phase = True
            with page.expect_response(
                lambda response: (
                    response.request.method in {"POST", "PUT", "PATCH"}
                    and urlsplit(response.url).netloc == site.netloc
                    and urlsplit(response.url).path != "/api/graphql"
                ),
                timeout=args.timeout_ms,
            ) as saved:
                save.click()
            save_phase = False
            response = saved.value
            saved_request = {
                "url": response.url,
                "method": response.request.method,
                "status": response.status,
            }
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            saved_page = _record_editor(page, output, "saved")
            terminal_url = (
                saved_page["url"] if family == "project_predicate_issue_creation" else target["url"]
            )
            page.goto(terminal_url, wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            archived["revisited"] = _record_editor(page, output, "revisited")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            if family == "filtered_issue_edit_with_preservation":
                page.get_by_role("button", name="Edit title and description", exact=True).click()
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                archived["editor_after"] = _record_editor(page, output, "editor_after")
                if (
                    page.get_by_label("Title", exact=True).input_value() != title
                    or page.get_by_label("Description", exact=True).input_value() != description
                ):
                    raise ValueError("refreshed editable fields differ from the exact saved goal")
                page.reload(wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
            after = _record_editor(page, output, "after")
            archived.update(saved=saved_page, after=after, save_requests=[saved_request])
            state = after["issue_state"]
            if (
                not state["open"]
                or not state["title"]
                or state != archived["revisited"]["issue_state"]
            ):
                raise ValueError("persisted issue state is missing or differs after refresh")
            if family == "project_predicate_issue_creation":
                if (
                    state["title"] != title
                    or state["description_text"].strip() != description
                    or sorted(state["labels"]) != sorted(case["existing_label_candidates"])
                ):
                    raise ValueError("created issue persisted values differ from the supplied goal")
                if "Byte Blaze" not in state["assignee_text"]:
                    raise ValueError("new issue did not preserve the selected assignee")
            else:
                preserved = ["labels", "assignee_text", "open", "checklist"]
                if family == "ranked_issue_comment_append_with_preservation":
                    preserved += ["title", "description_text"]
                    old_comments = before["issue_state"]["comments"]
                    if sorted(state["comments"]) != sorted(old_comments + [comment]):
                        raise ValueError(
                            "comment was not appended exactly once with old comments preserved"
                        )
                else:
                    if (
                        state["title"] != title
                        or state["comments"] != before["issue_state"]["comments"]
                    ):
                        raise ValueError("edited fields or existing comments differ from the goal")
                if any(state[field] != before["issue_state"][field] for field in preserved):
                    raise ValueError("an unspecified issue field changed")
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

    from webarena_verified.core.utils.network_event_utils import load_har_trace

    events = load_har_trace(output / "network.har")
    actual = [
        event
        for event in events
        if event.url == saved_request["url"]
        and event.http_method == saved_request["method"]
        and event.request_status == saved_request["status"]
    ]
    if len(actual) != 1 or not actual[0].is_request_success or actual[0].post_data is None:
        raise ValueError("the real Save requires one successful canonical HAR request body")
    body = actual[0].post_data
    leaves = []
    pending = [("$", body)]
    while pending:
        path, value = pending.pop()
        if isinstance(value, dict):
            pending.extend(
                (path + "[" + json.dumps(key) + "]", child) for key, child in value.items()
            )
        elif isinstance(value, list):
            pending.extend((path + f"[{index}]", child) for index, child in enumerate(value))
        else:
            leaves.append((path, value))
    critical = {}
    for expected in critical_values:
        matches = [(path, value) for path, value in leaves if value == expected]
        if len(matches) != 1:
            raise ValueError(
                "critical supplied content is absent or ambiguous in the real Save body"
            )
        critical.update(matches)
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
            ]
        ):
            critical[path] = value
    saved_request["post_data"] = critical
    parameters.update(
        final_url=terminal_url.replace(site_url, "__GITLAB__", 1),
        final_title=state["title"],
        final_description=state["description_text"],
        final_assignee_text=state["assignee_text"],
        final_labels=state["labels"],
        unexpected_label_count=0,
        preserved_comments=before["issue_state"]["comments"]
        if family != "project_predicate_issue_creation"
        else [],
        added_comment_count=1 if family == "ranked_issue_comment_append_with_preservation" else 0,
        added_comment_name=case["comment_text"]
        if family == "ranked_issue_comment_append_with_preservation"
        else state["title"],
        close_issue_label="Close issue",
    )
    final_checks = [
        {"kind": "final_url", "parameter": "final_url"},
        {
            "kind": "visible_text_exact",
            "parameter": "close_issue_label",
            "selector": "button.js-note-target-close",
        },
        {"kind": "visible_text_exact", "parameter": "final_title", "selector": "h1.title"},
        {
            "kind": "visible_text_exact",
            "parameter": "final_description",
            "selector": "div.description",
        },
        {
            "kind": "visible_text_exact",
            "parameter": "final_assignee_text",
            "selector": '[data-testid="assignee-block-container"]',
        },
        {
            "kind": "text_set_preserved",
            "parameter": "final_labels",
            "selector": '[data-testid="sidebar-labels"] .gl-label-text',
            "added_count_parameter": "unexpected_label_count",
            "added_name_parameter": "label_1",
        },
        {
            "kind": "text_set_preserved",
            "parameter": "preserved_comments",
            "selector": ".note-text.md",
            "added_count_parameter": "added_comment_count",
            "added_name_parameter": "added_comment_name",
        },
    ]
    if family != "project_predicate_issue_creation":
        for index, item in enumerate(state["checklist"]):
            parameters[f"checklist_{index}"] = item["checked"]
            row = "div.description li:has-text(" + json.dumps(item["label"]) + ")"
            final_checks.append(
                {
                    "kind": "checked",
                    "parameter": f"checklist_{index}",
                    "selector": row + " input[type=checkbox]",
                    "label_selector": row,
                }
            )
    event = {
        "url_template": "^__GITLAB__" + re.escape(urlsplit(actual[0].url).path) + r"(?:\?.*)?$",
        "http_method": actual[0].http_method,
        "response_status": actual[0].request_status,
        "post_data": critical,
        "post_data_schema": {
            "type": "object",
            "properties": {
                path: {"type": "number" if isinstance(value, (int, float)) else "string"}
                for path, value in critical.items()
            },
        },
        "last_event_only": False,
    }
    definition = {
        "task_type": "MUTATE",
        "goal_template": goal,
        "parameters": parameters,
        "answer": {"status": "SUCCESS", "values": None, "item_schema": None, "ordered": False},
        "completion_events": [event],
    }
    archived["terminal_state"] = state
    (output / "observation.json").write_text(
        json.dumps(archived, ensure_ascii=False, indent=2) + "\n"
    )
    relative = output.relative_to(args.batch_root)
    evidence = [
        {
            "field": "/task_definition/parameters/" + field,
            "artifact": "observation",
            "modality": "text",
            "location": "/terminal_state/" + source,
            "derivation": "Persisted visible issue state agrees after revisit and refresh.",
        }
        for field, source in [
            ("final_title", "title"),
            ("final_description", "description_text"),
            ("final_assignee_text", "assignee_text"),
            ("final_labels", "labels"),
        ]
    ]
    evidence += [
        {
            "field": "/task_definition/completion_events/0/post_data/"
            + path.replace("~", "~0").replace("/", "~1"),
            "artifact": "observation",
            "modality": "text",
            "location": "/save_requests/0/post_data/" + path.replace("~", "~0").replace("/", "~1"),
            "derivation": (
                "Actual successful UI Save parsed by the canonical HAR owner; "
                "no synthesized request."
            ),
        }
        for path in critical
    ]
    draft = {
        "generated_task_id": case["generated_task_id"],
        "task_id": case["task_id"],
        "intent_template_id": case["intent_template_id"],
        "site": "gitlab",
        "start_url": "__GITLAB__",
        "revision": 1,
        "source_artifact": case["source_artifact"],
        "source_record_id": case["source_record_id"],
        "baseline": baseline,
        "task_definition": definition,
        "grounding": {
            "captured_at": datetime.now(UTC).isoformat(),
            "family": family,
            "artifacts": {
                "har": str(relative / "network.har"),
                "trace": str(relative / "trace.zip"),
                "screenshot": str(relative / "after.png"),
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
                "generated_task_id": case["generated_task_id"],
                "draft": str(output / "draft.json"),
            }
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.mode == "formal":
        ground_mutation(args)
        return
    config = json.loads(args.task_config.read_text())
    baseline = json.loads(args.baseline_receipt.read_text())
    site_url = args.site_url.rstrip("/")
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("prepare requires a coordinator-owned fixed image baseline")
    if baseline["site_url"].rstrip("/") != site_url:
        raise ValueError("prepare endpoint differs from the coordinator baseline receipt")
    os.umask(0o077)
    output = args.batch_root / "exploration" / "gitlab" / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    site = urlsplit(site_url)
    blocked = []
    allowed_read_requests = []
    pages = []
    findings = []
    read_documents = set(config["observed_graphql_read_documents"])

    def restrict(route):
        requested = urlsplit(route.request.url)
        same_origin = (requested.scheme, requested.netloc) == (
            site.scheme,
            site.netloc,
        )
        if same_origin and route.request.method in {"GET", "HEAD"}:
            route.continue_()
            return
        documents = []
        if same_origin and route.request.method == "POST" and requested.path == "/api/graphql":
            try:
                payload = json.loads(route.request.post_data or "")
            except json.JSONDecodeError:
                payload = None
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("query"), str):
                    break
                documents.append(
                    {
                        "operation_name": item.get("operationName"),
                        "query_sha256": hashlib.sha256(item["query"].encode()).hexdigest(),
                    }
                )
            if (
                items
                and len(documents) == len(items)
                and all(document["query_sha256"] in read_documents for document in documents)
            ):
                allowed_read_requests.extend(documents)
                route.continue_()
                return
        blocked.append(
            {"method": route.request.method, "path": requested.path, "graphql": documents}
        )
        route.abort()

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
            initial = _record_editor(page, output, "initial")
            pages.append(initial)
            projects = []
            for index, target in enumerate(config["project_links"]):
                url = target["url"].replace("__GITLAB__", site_url)
                matches = [
                    link
                    for link in initial["links"]
                    if link["url"] == url and " ".join(link["text"].split()) == target["label"]
                ]
                if not matches:
                    raise ValueError("configured project is absent from the fresh dashboard")
                page.goto(matches[0]["url"], wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                project = _record_editor(page, output, f"project_{index:02}")
                pages.append(project)
                projects.append(_project_fields(project))

            selected = [
                index
                for index, project in enumerate(projects)
                if project["branches"] >= config["project_selection"]["minimum_branches"]
                and project["tags"] >= config["project_selection"]["minimum_tags"]
            ]
            if len(selected) != 1:
                raise ValueError("the fresh project-stat predicate is not unique")
            selected_project = pages[1 + selected[0]]
            issue_lists = [
                link
                for link in selected_project["links"]
                if re.fullmatch(r"Issues(?:\s+\d+)?", link["text"])
                and urlsplit(link["url"]).path.endswith("/-/issues")
            ]
            if len(issue_lists) != 1:
                raise ValueError("the selected project's visible Issues link is ambiguous")
            page.goto(issue_lists[0]["url"], wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            issue_list = _record_editor(page, output, "selected_project_issues")
            pages.append(issue_list)
            new_urls = {link["url"] for link in issue_list["links"] if link["text"] == "New issue"}
            new_editor = None
            if len(new_urls) == 1:
                page.goto(next(iter(new_urls)), wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                new_editor = _record_editor(page, output, "new_issue_editor")
                pages.append(new_editor)
                findings.append({"phase": "new_issue_editor", "status": "visible_form_recorded"})
            else:
                findings.append(
                    {"phase": "new_issue_editor", "status": "visible_link_missing_or_ambiguous"}
                )

            assigned_url = config["assigned_issues_link"].replace("__GITLAB__", site_url)
            if not any(link["url"] == assigned_url for link in initial["links"]):
                raise ValueError("assigned issues are absent from the fresh visible dashboard")
            page.goto(assigned_url, wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            assigned_dashboard = _record_editor(page, output, "assigned_dashboard")
            pages.append(assigned_dashboard)
            open_links = [
                link
                for link in assigned_dashboard["links"]
                if re.fullmatch(r"Open\s+\d+", link["text"])
            ]
            if len(open_links) != 1:
                raise ValueError("the actual open-issue tab is missing or ambiguous")
            page.goto(open_links[0]["url"], wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            assigned = _record_editor(page, output, "assigned_issues")
            pages.append(assigned)
            issue_targets = []
            for probe in config["existing_issue_probes"]:
                matches = [
                    row
                    for row in assigned["issues"]
                    if row["reference"] == probe["reference"]
                    and row["title"] == probe["title"]
                    and set(probe["required_labels"]) <= set(row["labels"])
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "configured issue is absent or ambiguous in fresh visible scope"
                    )
                issue_targets.append((probe, matches[0]["url"]))
            for index, (probe, url) in enumerate(issue_targets):
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                issue = _record_editor(page, output, f"issue_{index:02}")
                pages.append(issue)
                edit = page.get_by_role("button", name="Edit title and description", exact=True)
                edit_count = edit.count()
                findings.append(
                    {
                        "phase": "existing_issue",
                        "reference": probe["reference"],
                        "inline_edit_button_count": edit_count,
                    }
                )
                if edit_count == 1:
                    edit.click()
                    page.wait_for_timeout(int(args.settle_seconds * 1000))
                    pages.append(_record_editor(page, output, f"issue_{index:02}_editor"))

                # These selectors come from the archived actual issue frame.
                # Reload each issue before opening a field menu; no selection
                # or form submission changes the initial business assets.
                for field, widget, role in [
                    ("labels", "sidebar-labels", "button"),
                    ("assignee", "assignee-block-container", "link"),
                ]:
                    page.goto(url, wait_until="domcontentloaded")
                    page.wait_for_timeout(int(args.settle_seconds * 1000))
                    menu = page.get_by_test_id(widget).get_by_role(role, name="Edit", exact=True)
                    count = menu.count()
                    findings.append(
                        {"phase": "issue_field_menu", "field": field, "control_count": count}
                    )
                    if count == 1:
                        menu.click()
                        page.wait_for_timeout(int(args.settle_seconds * 1000))
                        pages.append(
                            _record_editor(page, output, f"issue_{index:02}_{field}_options")
                        )

            # Inspect field dropdowns on the actual new issue page. Opening a
            # menu is separate from selecting a value; no save/submit is issued.
            editors = [new_editor] if new_editor is not None else []
            for editor_index, editor in enumerate(editors):
                page.goto(editor["url"], wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                for dropdown_index, control in enumerate(
                    [
                        control
                        for control in editor["editor_controls"]
                        if control["tag"] == "button"
                        and control["type"] == "button"
                        and not control["disabled"]
                        and (
                            control["aria_haspopup"] in {"true", "menu", "listbox"}
                            or control["data_toggle"] == "dropdown"
                        )
                        and re.search(
                            r"label|assign",
                            " ".join(
                                control.get(field) or ""
                                for field in [
                                    "text",
                                    "title",
                                    "aria_label",
                                    "dropdown_title",
                                    "field_name",
                                ]
                            ),
                            re.IGNORECASE,
                        )
                    ][: args.dropdown_limit]
                ):
                    label = control["aria_label"] or control["text"] or control["title"]
                    if control["id"]:
                        locator = page.locator(f"button[id={json.dumps(control['id'])}]:visible")
                    elif label:
                        locator = page.get_by_role("button", name=label, exact=True)
                    else:
                        findings.append(
                            {"phase": "dropdown", "status": "unlabelled_control_requires_evidence"}
                        )
                        continue
                    if locator.count() != 1:
                        findings.append(
                            {"phase": "dropdown", "label": label, "status": "control_not_unique"}
                        )
                        continue
                    locator.click()
                    page.wait_for_timeout(int(args.settle_seconds * 1000))
                    pages.append(
                        _record_editor(
                            page, output, f"dropdown_{editor_index:02}_{dropdown_index:02}"
                        )
                    )
                    page.keyboard.press("Escape")
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

    archived = {
        "initial": initial,
        "projects": projects,
        "selected_project_index": selected[0],
        "pages": pages,
        "findings": findings,
        "blocked_requests": blocked,
        "allowed_graphql_read_requests": allowed_read_requests,
    }
    (output / "observation.json").write_text(
        json.dumps(archived, ensure_ascii=False, indent=2) + "\n"
    )
    receipt = {
        "status": "actual_ui_recorded_formal_mutation_grounding_pending",
        "captured_at": datetime.now(UTC).isoformat(),
        "site": "gitlab",
        "baseline": baseline,
        "account_policy": "coordinator-provided private storage state; no credentials printed",
        "network_policy": (
            "same_origin_GET_HEAD_and_frozen_observed_graphql_query_documents_only; "
            "no submit/save or option selection"
        ),
        "blocked_requests": blocked,
        "allowed_graphql_read_requests": allowed_read_requests,
        "findings": findings,
        "formal_authoring_ready": False,
        "generated_candidates": 0,
        "observed_page_count": len(pages),
        "pending": [
            "actual label/assignee options and exact editor controls",
            "per-case fresh baseline and real UI Save",
            "POST/PUT evidence and persisted refresh/revisit readback",
            "owner final_state_checks and normal author.compile_candidate draft",
        ],
        "artifacts": {
            "har": "network.har",
            "trace": "trace.zip",
            "observation": "observation.json",
            "screenshot": "new_issue_editor.png"
            if new_editor is not None
            else "selected_project_issues.png",
        },
    }
    (output / "readiness_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(output / "readiness_receipt.json"),
                "generated_candidates": 0,
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
