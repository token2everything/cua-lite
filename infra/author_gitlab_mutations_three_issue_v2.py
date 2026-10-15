#!/usr/bin/env python3
"""Inspect GitLab editors or ground one mutation on a fresh fixed baseline.

Coordinator-owned runtime (Playwright/Chromium; sibling author_gitlab.py required):
  /opt/venv/bin/python -B /infra/author_gitlab_mutations_three_issue_v2.py \
    --mode formal-three-issue --generated-task-id wavgen-v03-full300-0337 \
    --task-config /infra/wavgen-v03-full300-0337.json --site-url http://localhost:8090 \
    --baseline-receipt FRESH_BASELINE --storage-state STANDARD_AUTH_STATE \
    --batch-root /outputs --attempt grounding-01

The sibling handoff_v4 source owns raw versus rendered note projection. All
three initial editors/discussions are read before the first business write.
Existing prepare/single-Issue modes retain their original implementation:
  /opt/venv/bin/python -B /outputs/infra/author_gitlab_mutations.py \
    --mode prepare --task-config /outputs/plans/gitlab_task_configs_mutate.json \
    --baseline-receipt /outputs/private/gitlab_author_baseline.json \
    --storage-state /outputs/private/gitlab_storage_state.json \
    --site-url http://localhost:8023 --batch-root /outputs

For a formal case, use --mode formal --generated-task-id wavgen-v03-full300-0020
and --attempt grounding-01 with that case's new coordinator-owned baseline.
For immutable successful closure captures rejected by the original parser, use
--mode compile-recorded --recorded-directory ORIGINAL --recorded-owner FROZEN
with the original config/baseline and a new attempt. This reads local evidence
only and requires the separately versioned JSON-array evaluator package.
Formal authoring uses real UI Save, then revisit and refresh, and emits the
existing author.compile_candidate draft only after persisted state and the
canonical HAR request body agree. It never calls an API to write assets.

The new --mode formal-three-issue reads all three initial editors/discussions
before any business write, then independently hands off, saves and conditionally
closes each Issue. It follows the selected Issue's visible project List link and
clicks All, then binds all three visible title/reference/label/state/comment-count
rows. Raw Markdown and discussion GETs are per-entity persisted snapshots; the
terminal reader independently sees only the final project summary. The new mode
uses the sibling handoff_v4 raw-note/rendered-note owner. Existing modes remain
unchanged.

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
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import author_gitlab_mutations_handoff_v4
from author_gitlab import _project_fields, _record


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["prepare", "formal", "formal-three-issue", "compile-recorded"],
        required=True,
    )
    parser.add_argument("--generated-task-id")
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path)
    parser.add_argument("--recorded-directory", type=Path)
    parser.add_argument("--recorded-owner", type=Path)
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
    if (
        args.mode in {"formal", "formal-three-issue", "compile-recorded"}
        and not args.generated_task_id
    ):
        parser.error("formal mode requires one generated task ID and one fresh baseline")
    if args.mode != "compile-recorded" and not args.storage_state:
        parser.error("live capture requires a private storage state")
    if args.mode == "compile-recorded" and (not args.recorded_directory or not args.recorded_owner):
        parser.error("compile-recorded requires original capture directory and frozen owner")
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
        "filtered_issue_checklist_choice_and_close",
        "conditional_issue_discussion_handoff_and_resolution",
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
                matching, target = _select_existing_issue(case, listing)
                archived.update(
                    assigned_dashboard=dashboard,
                    assigned_issues=listing,
                    matching_issues=matching,
                    target=target,
                )
                page.goto(target["url"], wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                if family == "conditional_issue_discussion_handoff_and_resolution":
                    page.locator("li.note-skeleton:visible").first.wait_for(state="hidden")
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
                if family == "conditional_issue_discussion_handoff_and_resolution":
                    user_notes = "li.note-comment:not(.system-note) .note-text.md:visible"
                    archived["initial_user_comments"] = [
                        text.strip() for text in page.locator(user_notes).all_inner_texts()
                    ]
                    page.get_by_role(
                        "button", name="Edit title and description", exact=True
                    ).click()
                    archived["editor_before"] = _record_editor(page, output, "editor_before")
                    description_before = page.get_by_label("Description", exact=True).input_value()
                    title, description, comment, parameters, goal = _handoff_goal(
                        case, before, description_before, archived["initial_user_comments"], target
                    )
                    archived["handoff_description"] = description
                    archived["handoff_comment"] = comment
                    archived["close_required"] = not archived["initial_user_comments"] and all(
                        case["responsibility_choices"]
                    )
                    page.get_by_role("button", name="Cancel", exact=True).click()
                    page.get_by_test_id("comment-field").fill(comment)
                    save = page.get_by_role("button", name="Comment", exact=True)
                    critical_values = [comment]
                elif family == "filtered_issue_checklist_choice_and_close":
                    page.get_by_role(
                        "button", name="Edit title and description", exact=True
                    ).click()
                    page.wait_for_timeout(int(args.settle_seconds * 1000))
                    archived["editor_before"] = _record_editor(page, output, "editor_before")
                    description_before = page.get_by_label("Description", exact=True).input_value()
                    title, description, parameters, goal = _checklist_goal(
                        case, before, description_before
                    )
                    page.get_by_label("Description", exact=True).fill(description)
                    save = page.get_by_role("button", name="Save changes", exact=True)
                    critical_values = [description]
                elif family == "ranked_issue_comment_append_with_preservation":
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
            # One real editor Save is issued. The closure family then closes the
            # same issue once; neither business action is replayed on failure.
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
            if family in {
                "filtered_issue_checklist_choice_and_close",
                "conditional_issue_discussion_handoff_and_resolution",
            }:
                saved_request["request_body_sha256"] = hashlib.sha256(
                    response.request.post_data.encode()
                ).hexdigest()
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            saved_page = _record_editor(page, output, "saved")
            if family == "conditional_issue_discussion_handoff_and_resolution":
                archived["comment_saved"] = saved_page
                archived["save_requests"] = [saved_request]
                page.get_by_role("button", name="Edit title and description", exact=True).click()
                page.get_by_label("Description", exact=True).fill(description)
                archived["unsaved_description"] = _record_editor(
                    page, output, "unsaved_description"
                )
                save_phase = True
                try:
                    with page.expect_response(
                        lambda response: (
                            response.request.method in {"POST", "PUT", "PATCH"}
                            and urlsplit(response.url).netloc == site.netloc
                            and urlsplit(response.url).path != "/api/graphql"
                        ),
                        timeout=args.timeout_ms,
                    ) as changed:
                        page.get_by_role("button", name="Save changes", exact=True).click()
                finally:
                    save_phase = False
                response = changed.value
                archived["save_requests"].append(
                    {
                        "url": response.url,
                        "method": response.request.method,
                        "status": response.status,
                        "request_body_sha256": hashlib.sha256(
                            response.request.post_data.encode()
                        ).hexdigest(),
                    }
                )
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                saved_page = _record_editor(page, output, "checklist_saved")
                archived["checklist_saved"] = saved_page
            if family == "filtered_issue_checklist_choice_and_close" or (
                family == "conditional_issue_discussion_handoff_and_resolution"
                and archived["close_required"]
            ):
                # Collect the actual Close response without assuming its method,
                # endpoint, body keys, or a Reopen control. HAR owns parsing below.
                close_responses = []

                def record_close_response(response):
                    requested = urlsplit(response.url)
                    if (requested.scheme, requested.netloc) == (
                        site.scheme,
                        site.netloc,
                    ) and response.request.method not in {"GET", "HEAD"}:
                        close_responses.append(
                            {
                                "url": response.url,
                                "method": response.request.method,
                                "status": response.status,
                            }
                        )

                page.on("response", record_close_response)
                save_phase = True
                try:
                    close_button = page.locator("button.js-note-target-close:visible").filter(
                        has_text=re.compile(r"^\s*Close issue\s*$")
                    )
                    close_button.click()
                    close_button.wait_for(state="hidden")
                    page.wait_for_timeout(int(args.settle_seconds * 1000))
                finally:
                    save_phase = False
                    page.remove_listener("response", record_close_response)
                archived["checklist_saved"] = saved_page
                archived["close_responses"] = close_responses
                saved_page = _record_editor(page, output, "closed")
            terminal_url = (
                saved_page["url"] if family == "project_predicate_issue_creation" else target["url"]
            )
            page.goto(terminal_url, wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            archived["revisited"] = _record_editor(page, output, "revisited")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            if family in {
                "filtered_issue_edit_with_preservation",
                "filtered_issue_checklist_choice_and_close",
                "conditional_issue_discussion_handoff_and_resolution",
            }:
                page.get_by_role("button", name="Edit title and description", exact=True).click()
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                archived["editor_after"] = _record_editor(page, output, "editor_after")
                if family == "filtered_issue_edit_with_preservation" and (
                    page.get_by_label("Title", exact=True).input_value() != title
                    or page.get_by_label("Description", exact=True).input_value() != description
                ):
                    raise ValueError("refreshed editable fields differ from the exact saved goal")
                page.reload(wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
            after = _record_editor(page, output, "after")
            if family == "conditional_issue_discussion_handoff_and_resolution":
                archived.update(saved=saved_page, after=after)
                archived["final_user_comments"] = [
                    text.strip() for text in page.locator(user_notes).all_inner_texts()
                ]
            else:
                archived.update(saved=saved_page, after=after, save_requests=[saved_request])
            state = after["issue_state"]
            if family not in {
                "filtered_issue_checklist_choice_and_close",
                "conditional_issue_discussion_handoff_and_resolution",
            } and (
                state["open"] is not True
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
            elif family in {
                "filtered_issue_checklist_choice_and_close",
                "conditional_issue_discussion_handoff_and_resolution",
            }:
                pass  # The shared GT owner validates captured closure state below.
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

    _emit_grounded_mutation(
        args, case, baseline, output, archived, parameters, goal, critical_values
    )


def _select_existing_issue(case: dict[str, Any], listing: dict[str, Any]):
    """Select from the complete, visible Open scope for both author inputs."""
    family = case["family"]
    counts = [
        int(link["text"].split()[-1])
        for link in listing["links"]
        if re.fullmatch(r"Open\s+\d+", link["text"])
    ]
    if len(counts) != 1 or counts[0] != len(listing["issues"]):
        raise ValueError("formal issue selection requires the complete visible open scope")
    matching = [
        row
        for row in listing["issues"]
        if row["reference"].split("#")[0] == case["project_path"]
        and set(case["required_labels"]) <= set(row["labels"])
        and not set(case["excluded_labels"]) & set(row["labels"])
        and (case["author"] is None or row["author"] == case["author"])
    ]
    if (
        family == "conditional_issue_discussion_handoff_and_resolution"
        and case["rank_order"] == "unique"
    ):
        if len(matching) != 1:
            raise ValueError("fresh handoff predicate must select one unique open issue")
        target = matching[0]
    elif family in {
        "ranked_issue_comment_append_with_preservation",
        "filtered_issue_checklist_choice_and_close",
        "conditional_issue_discussion_handoff_and_resolution",
    }:
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
    return matching, target


def _checklist_goal(case: dict[str, Any], before: dict[str, Any], description_before: str):
    """Derive exact Markdown choices and the public closure goal from observed state."""
    checklist = before["issue_state"]["checklist"]
    if [item["label"] for item in checklist] != case["checklist_labels"] or checklist[2][
        "checked"
    ] is not True:
        raise ValueError("the visible responsibility/conduct checklist differs")
    title = before["issue_state"]["title"]
    description = description_before
    for index, item in enumerate(checklist):
        pattern = re.compile(
            r"(?m)^([ \t]*[-+*] \[)([ xX])(\] +" + re.escape(item["label"]) + r"[ \t]*)(\r?$)"
        )
        matches = list(pattern.finditer(description))
        if len(matches) != 1 or (matches[0][2].lower() == "x") != item["checked"]:
            raise ValueError("visible checklist and editable Markdown disagree")
        if index < 2:
            marker = "x" if case["responsibility_choices"][index] else " "
            description = pattern.sub(
                lambda match: match[1] + marker + match[3] + match[4],
                description,
            )
    if description == description_before:
        raise ValueError("the requested checklist would make no persisted change")
    parameters = dict(
        label_2=case["required_labels"][1],
        rank_order=case["rank_order"],
        responsibility_1=case["checklist_labels"][0],
        responsibility_2=case["checklist_labels"][1],
        choice_1="checked" if case["responsibility_choices"][0] else "unchecked",
        choice_2="checked" if case["responsibility_choices"][1] else "unchecked",
        conduct_label=case["checklist_labels"][2],
    )
    goal = (
        "Among open issues assigned to {assignee} in {project_path} with both "
        "{label_1} and {label_2} labels, order by numeric issue number "
        "{rank_order} and choose the first. Set the description's responsibility "
        "checkbox '{responsibility_1}' to {choice_1} and '{responsibility_2}' "
        "to {choice_2}. Keep '{conduct_label}' checked. Close that same issue. "
        "Preserve the title, all other description text, labels, assignee and "
        "existing comments; add no comment. Leave its closed detail page open."
    )
    parameters = {
        "project_path": case["project_path"],
        "assignee": "Byte Blaze",
        "label_1": case["required_labels"][0],
        **parameters,
    }
    return title, description, parameters, goal


def _handoff_goal(case, before, description_before, initial_user_comments, target):
    """Freeze the initial discussion branch and edit only two Markdown markers."""
    choices = case["responsibility_choices"]
    if len(choices) != 2 or any(type(choice) is not bool for choice in choices):
        raise ValueError("handoff requires two boolean responsibility choices")
    checklist = before["issue_state"]["checklist"]
    if (
        len(checklist) != 3
        or [item["label"] for item in checklist] != case["checklist_labels"]
        or checklist[2]["checked"] is not True
    ):
        raise ValueError("handoff requires the actual three responsibility/conduct items")
    description = description_before
    for index, item in enumerate(checklist):
        pattern = re.compile(
            r"(?m)^([ \t]*[-+*] \[)([ xX])(\] +" + re.escape(item["label"]) + r"[ \t]*)(\r?$)"
        )
        matches = list(pattern.finditer(description))
        if len(matches) != 1 or (matches[0][2].lower() == "x") != item["checked"]:
            raise ValueError("initial visible checklist and complete Markdown disagree")
        if index < 2:
            description = pattern.sub(
                lambda match: match[1] + ("x" if choices[index] else " ") + match[3] + match[4],
                description,
            )
    filled = {
        "issue_reference": target["reference"],
        "issue_title": before["issue_state"]["title"],
        "initial_user_comment_count": len(initial_user_comments),
        "choice_1": "checked" if choices[0] else "unchecked",
        "choice_2": "checked" if choices[1] else "unchecked",
    }
    comment = case["handoff_comment_pattern"].format_map(filled)
    if comment in initial_user_comments:
        raise ValueError("the requested public handoff already exists initially")
    parameters = {
        "project_path": case["project_path"],
        "assignee": "Byte Blaze",
        "selection_rule": case["selection_rule"],
        "handoff_pattern": case["handoff_comment_pattern"],
        "responsibility_1": case["checklist_labels"][0],
        "responsibility_2": case["checklist_labels"][1],
        "choice_1": filled["choice_1"],
        "choice_2": filled["choice_2"],
        "conduct_label": case["checklist_labels"][2],
    }
    goal = (
        "From the complete Open issues assigned to {assignee} in {project_path}, "
        "{selection_rule} Before making any change, read that issue's full description "
        "and count its existing user discussion comments; exclude system activity notes. "
        "Append exactly one ordinary public comment by filling this handoff pattern "
        "from its original reference, title and initial user-comment count: "
        "{handoff_pattern} Use checked/unchecked for the supplied responsibility choices. "
        "Then use Edit title and description to set the original Markdown checkbox "
        "'{responsibility_1}' to {choice_1} and '{responsibility_2}' to {choice_2}. "
        "Change only these two checkbox markers and save the full Description; preserve "
        "every other Markdown character and keep '{conduct_label}' checked. Close the "
        "same issue only if it originally had zero user discussion comments AND both "
        "supplied responsibility choices are checked. Otherwise keep it Open and do "
        "not Close it. Use the initial discussion count before your new handoff comment. "
        "Preserve title, labels, assignee and all existing user comments. Revisit or "
        "refresh to confirm the comment, checklist and requested state persisted; "
        "finish on that issue's detail page."
    )
    return before["issue_state"]["title"], description, comment, parameters, goal


def _build_handoff_reference(args, case, output, archived):
    """Require actual comment, full Markdown Save and the original-discussion branch."""
    from webarena_verified.core.utils.network_event_utils import load_har_trace

    site_url = args.site_url.rstrip("/")
    before, after, target = archived["before"], archived["after"], archived["target"]
    initial_users = archived["initial_user_comments"]
    markdown_controls = [
        c for c in archived["editor_before"]["editor_controls"] if "Description" in c["labels"]
    ]
    if len(markdown_controls) != 1:
        raise ValueError("initial complete Markdown field is absent or ambiguous")
    title, description, comment, parameters, goal = _handoff_goal(
        case, before, markdown_controls[0]["value"], initial_users, target
    )
    close = not initial_users and all(case["responsibility_choices"])
    state = after["issue_state"]
    expected_checklist = [dict(item) for item in before["issue_state"]["checklist"]]
    for index, choice in enumerate(case["responsibility_choices"]):
        expected_checklist[index]["checked"] = choice
    if (
        archived["close_required"] is not close
        or state["open"] is not (not close)
        or state != archived["revisited"]["issue_state"]
        or state["checklist"] != expected_checklist
        or sorted(archived["final_user_comments"]) != sorted(initial_users + [comment])
        or any(
            state[field] != before["issue_state"][field]
            for field in ["title", "description_text", "labels", "assignee_text"]
        )
        or archived["target"]["url"] != after["url"]
    ):
        raise ValueError("handoff branch, persisted checklist/comments or preserved fields differ")
    refreshed = {
        label: [
            c["value"] for c in archived["editor_after"]["editor_controls"] if label in c["labels"]
        ]
        for label in ["Title", "Description"]
    }
    if refreshed != {"Title": [title], "Description": [description]}:
        raise ValueError("refreshed complete Markdown/title differ from the exact requested Save")
    events = load_har_trace(output / "network.har")
    if len(archived["save_requests"]) != 2:
        raise ValueError("handoff requires one ordinary comment then one full Description Save")
    actual = []
    for receipt in archived["save_requests"]:
        matches = [
            event
            for event in events
            if event.url == receipt["url"]
            and event.http_method == receipt["method"]
            and event.request_status == receipt["status"]
            and hashlib.sha256(event.data["request"]["postData"]["text"].encode()).hexdigest()
            == receipt["request_body_sha256"]
        ]
        if len(matches) != 1 or not matches[0].is_request_success:
            raise ValueError(
                "each actual handoff action requires one successful canonical HAR body"
            )
        actual.append(matches[0])
    note, editor = actual
    number = str(int(target["reference"].split("#")[1]))
    entity = editor.response_content
    if (
        urlsplit(editor.url).path != urlsplit(target["url"]).path + ".json"
        or str(entity["iid"]) != number
        or entity["title"] != title
        or editor.post_data["title"] != title
        or editor.post_data["description"] != description
        or note.post_data["note"]
        != {
            "noteable_type": "Issue",
            "noteable_id": entity["id"],
            "internal": False,
            "note": comment,
        }
        or note.data["startedDateTime"] >= editor.data["startedDateTime"]
    ):
        raise ValueError(
            "actual public comment/editor body, selected issue identity or order differ"
        )
    parameters.update(
        expected_comment=comment,
        expected_saved_description=description,
        initial_user_comment_count=len(initial_users),
        final_url=target["url"].replace(site_url, "__GITLAB__", 1),
        final_title=title,
        final_description=state["description_text"],
        final_assignee_text=state["assignee_text"],
        final_labels=state["labels"],
        unexpected_label_count=0,
        preserved_comments=initial_users,
        added_comment_count=1,
        added_comment_name=comment,
    )
    bodies = [
        {
            '$["note"]["noteable_type"]': "Issue",
            '$["note"]["noteable_id"]': entity["id"],
            '$["note"]["internal"]': False,
            '$["note"]["note"]': "{expected_comment}",
        },
        {'$["title"]': "{final_title}", '$["description"]': "{expected_saved_description}"},
    ]
    completion = []
    for event, body in zip(actual, bodies):
        completion.append(
            {
                "url_template": event.url.replace(site_url, "__GITLAB__", 1),
                "http_method": event.http_method,
                "response_status": event.request_status,
                "post_data": body,
                "post_data_schema": {
                    "type": "object",
                    "properties": {
                        path: {
                            "type": "boolean"
                            if type(value) is bool
                            else "integer"
                            if type(value) is int
                            else "string"
                        }
                        for path, value in body.items()
                    },
                },
                "last_event_only": False,
            }
        )
    completion[1]["response_content"] = {'$["id"]': entity["id"], '$["iid"]': entity["iid"]}
    close_body = {
        "$[0].operationName": "updateIssue",
        "$[0].variables.input.iid": number,
        "$[0].variables.input.projectPath": case["project_path"],
        "$[0].variables.input.stateEvent": "CLOSE",
    }
    closure = []
    for event in events:
        if urlsplit(event.url).path != "/api/graphql" or event.http_method != "POST":
            continue
        post = event.post_data
        if not isinstance(post, list) or len(post) != 1:
            continue
        operation = post[0]
        if operation.get("operationName") != "updateIssue":
            continue
        inputs = operation["variables"]["input"]
        if all(
            inputs[key] == value
            for key, value in {
                "iid": number,
                "projectPath": case["project_path"],
                "stateEvent": "CLOSE",
            }.items()
        ):
            closure.append(event)
    if close:
        if len(closure) != 1 or not closure[0].is_request_success:
            raise ValueError(
                "the initial-discussion Close branch needs one actual successful Close"
            )
        event = closure[0]
        closed = event.response_content[0]["data"]["updateIssue"]["issuable"]
        if (
            closed["id"] != "gid://gitlab/Issue/" + str(entity["id"])
            or closed["state"] != "closed"
            or event.data["startedDateTime"] <= editor.data["startedDateTime"]
        ):
            raise ValueError(
                "actual Close response or action order differs from the selected issue"
            )
        completion.append(
            {
                "url_template": event.url.replace(site_url, "__GITLAB__", 1),
                "http_method": event.http_method,
                "response_status": event.request_status,
                "post_data": close_body,
                "response_content": {
                    "$[0].data.updateIssue.issuable.id": closed["id"],
                    "$[0].data.updateIssue.issuable.state": "closed",
                },
                "last_event_only": False,
            }
        )
    else:
        if closure:
            raise ValueError("the original-discussion Open branch forbids Close")
        completion.append(
            {
                "url_template": "__GITLAB__/api/graphql",
                "http_method": "POST",
                "post_data": close_body,
                "should_not_exist": True,
                "last_event_only": False,
            }
        )
    controls = [
        c
        for c in after["editor_controls"]
        if c["tag"] == "button"
        and any(t.startswith("js-note-target-") for t in c["class_name"].split())
    ]
    if len(controls) != 1 or controls[0]["disabled"] or not controls[0]["text"]:
        raise ValueError("persisted Issue state has no unique enabled visible state control")
    if (controls[0]["text"] == "Close issue") is close:
        raise ValueError("visible Issue state control disagrees with the conditional branch")
    classes = [t for t in controls[0]["class_name"].split() if t.startswith("js-note-target-")]
    if len(classes) != 1 or not re.fullmatch(r"[a-zA-Z0-9_-]+", classes[0]):
        raise ValueError("actual state control has no unambiguous stable selector")
    parameters["state_control_label"] = controls[0]["text"]
    checks = [
        {"kind": "final_url", "parameter": "final_url"},
        {
            "kind": "visible_text_exact",
            "parameter": "state_control_label",
            "selector": "button." + classes[0],
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
            "added_name_parameter": "project_path",
        },
        {
            "kind": "text_set_preserved",
            "parameter": "preserved_comments",
            "selector": "li.note-comment:not(.system-note) .note-text.md",
            "added_count_parameter": "added_comment_count",
            "added_name_parameter": "added_comment_name",
        },
    ]
    for index, item in enumerate(state["checklist"]):
        parameters[f"checklist_{index}"] = item["checked"]
        row = "div.description li:has-text(" + json.dumps(item["label"]) + ")"
        checks.append(
            {
                "kind": "checked",
                "parameter": f"checklist_{index}",
                "selector": row + " input[type=checkbox]",
                "label_selector": row,
            }
        )
    for index, text in enumerate(initial_users):
        parameters[f"preserved_user_comment_{index}"] = text
        checks.append(
            {
                "kind": "visible_text_exact",
                "parameter": f"preserved_user_comment_{index}",
                "selector": "li.note-comment:not(.system-note) .note-text.md:has-text("
                + json.dumps(" ".join(text.split()))
                + ")",
            }
        )
    archived["handoff_completion_events"] = completion
    evidence = [
        {
            "field": "/task_definition/parameters/" + name,
            "artifact": "observation",
            "modality": "text",
            "location": location,
            "derivation": derivation,
        }
        for name, location, derivation in [
            (
                "expected_comment",
                "/handoff_comment",
                "Public pattern filled once from initial Issue reference/title, "
                "original user discussion count and supplied choices.",
            ),
            (
                "expected_saved_description",
                "/handoff_description",
                "Complete initial Markdown changes only two observed responsibility markers; "
                "exact canonical Save and refreshed editor agree.",
            ),
            (
                "initial_user_comment_count",
                "/initial_user_comments",
                "Count of initial visible user-note bodies; "
                "system notes excluded before any mutation.",
            ),
            (
                "final_title",
                "/after/issue_state/title",
                "Original title preserved after revisit and refresh.",
            ),
            (
                "final_description",
                "/after/issue_state/description_text",
                "All original rendered text preserved after revisit and refresh.",
            ),
            (
                "final_assignee_text",
                "/after/issue_state/assignee_text",
                "Original assignee preserved.",
            ),
            ("final_labels", "/after/issue_state/labels", "Complete original label set preserved."),
        ]
    ]
    for index, event in enumerate(completion):
        for content in ["post_data", "response_content"]:
            for path in event.get(content, {}):
                pointer = path.replace("~", "~0").replace("/", "~1")
                evidence.append(
                    {
                        "field": f"/task_definition/completion_events/{index}/{content}/{pointer}",
                        "artifact": "observation",
                        "modality": "text",
                        "location": f"/handoff_completion_events/{index}/{content}/{pointer}",
                        "derivation": (
                            "Actual canonical successful comment/editor/Close evidence; "
                            "Open negative event expresses the original-discussion rule."
                        ),
                    }
                )
    evidence.extend(
        {
            "field": f"/task_definition/parameters/preserved_user_comment_{index}",
            "artifact": "observation",
            "modality": "text",
            "location": f"/initial_user_comments/{index}",
            "derivation": "The complete initial visible user note must remain exactly once.",
        }
        for index in range(len(initial_users))
    )
    return (
        {
            "task_type": "MUTATE",
            "goal_template": goal,
            "parameters": parameters,
            "answer": {"status": "SUCCESS", "values": None, "item_schema": None, "ordered": False},
            "completion_events": completion,
        },
        checks,
        evidence,
    )


def _emit_grounded_mutation(
    args, case, baseline, output, archived, parameters, goal, critical_values
):
    """Own semantic checks and GT emission for live capture and recorded ingestion."""
    family = case["family"]
    if family == "conditional_issue_discussion_handoff_and_resolution":
        definition, final_checks, evidence = _build_handoff_reference(args, case, output, archived)
        state = archived["after"]["issue_state"]
        archived["terminal_state"] = state
        (output / "observation.json").write_text(
            json.dumps(archived, ensure_ascii=False, indent=2) + "\n"
        )
        relative = output.relative_to(args.batch_root)
    else:
        site_url = args.site_url.rstrip("/")
        after = archived["after"]
        state = after["issue_state"]
        saved_request = archived["save_requests"][0]
        terminal_url = (
            archived["saved"]["url"]
            if family == "project_predicate_issue_creation"
            else archived["target"]["url"]
        )
        if family != "project_predicate_issue_creation":
            before = archived["before"]
            target = archived["target"]
        if family == "filtered_issue_checklist_choice_and_close":
            if (
                state["open"] is not False
                or not state["title"]
                or state != archived["revisited"]["issue_state"]
            ):
                raise ValueError("persisted issue state is missing or differs after refresh")
            title = before["issue_state"]["title"]
            description = critical_values[0]
            fields = {}
            for label in ["Title", "Description"]:
                values = [
                    c["value"]
                    for c in archived["editor_after"]["editor_controls"]
                    if label in c["labels"]
                ]
                if len(values) != 1:
                    raise ValueError("refreshed editable field is absent or ambiguous")
                fields[label] = values[0]
            if fields != {"Title": title, "Description": description}:
                raise ValueError("refreshed editable fields differ from the exact saved goal")
            closed_controls = [
                control
                for control in after["editor_controls"]
                if control["tag"] == "button"
                and any(
                    token.startswith("js-note-target-") for token in control["class_name"].split()
                )
            ]
            if (
                len(closed_controls) != 1
                or not closed_controls[0]["text"]
                or closed_controls[0]["text"] == "Close issue"
                or closed_controls[0]["disabled"]
            ):
                raise ValueError("closure lacks one changed visible state control")
            archived["closed_state_control"] = closed_controls[0]
            state_classes = [
                token
                for token in closed_controls[0]["class_name"].split()
                if token.startswith("js-note-target-") and re.fullmatch(r"[a-zA-Z0-9_-]+", token)
            ]
            if len(state_classes) != 1:
                raise ValueError("actual closed state control has no unique stable selector")
            archived["closed_state_selector"] = "button." + state_classes[0]
            expected_checklist = [dict(item) for item in before["issue_state"]["checklist"]]
            for index, choice in enumerate(case["responsibility_choices"]):
                expected_checklist[index]["checked"] = choice
            if state["checklist"] != expected_checklist:
                raise ValueError("the responsibility choices did not persist exactly")
            if any(
                state[field] != before["issue_state"][field]
                for field in [
                    "title",
                    "description_text",
                    "labels",
                    "assignee_text",
                    "comments",
                ]
            ):
                raise ValueError("an unspecified issue field changed during closure")

        from webarena_verified.core.utils.network_event_utils import load_har_trace

        events = load_har_trace(output / "network.har")
        actual = [
            event
            for event in events
            if event.url == saved_request["url"]
            and event.http_method == saved_request["method"]
            and event.request_status == saved_request["status"]
            and (
                family != "filtered_issue_checklist_choice_and_close"
                or hashlib.sha256(event.data["request"]["postData"]["text"].encode()).hexdigest()
                == saved_request["request_body_sha256"]
            )
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
            added_comment_count=1
            if family == "ranked_issue_comment_append_with_preservation"
            else 0,
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
            "url_template": actual[0].url.replace(site_url, "__GITLAB__", 1),
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
        completion_events = [event]
        if family == "filtered_issue_checklist_choice_and_close":
            # A legal model may click rendered checkboxes instead of using this
            # author's editor. Only Close evidence and semantic terminal checks gate
            # the task; the editor Save remains private author provenance.
            close_events = []
            issue_number = str(int(target["reference"].split("#")[1]))
            save_entity = actual[0].response_content
            if (
                not isinstance(save_entity, dict)
                or str(save_entity["iid"]) != issue_number
                or save_entity["title"] != state["title"]
            ):
                raise ValueError("actual editor Save does not identify the selected issue")
            issue_gid = "gid://gitlab/Issue/" + str(save_entity["id"])
            for closed_event in events:
                if (
                    not any(
                        closed_event.url == observed["url"]
                        and closed_event.http_method == observed["method"]
                        and closed_event.request_status == observed["status"]
                        for observed in archived["close_responses"]
                    )
                    or not closed_event.is_request_success
                ):
                    continue
                post, response = closed_event.post_data, closed_event.response_content
                if (
                    not isinstance(post, list)
                    or len(post) != 1
                    or not isinstance(response, list)
                    or len(response) != 1
                ):
                    continue
                operation = post[0]
                if operation.get("operationName") != "updateIssue":
                    continue
                input_value = operation["variables"]["input"]
                if any(
                    input_value[key] != value
                    for key, value in {
                        "iid": issue_number,
                        "projectPath": case["project_path"],
                        "stateEvent": "CLOSE",
                    }.items()
                ):
                    continue
                entity = response[0]["data"]["updateIssue"]["issuable"]
                if entity["id"] != issue_gid or entity["state"] != "closed":
                    continue
                if closed_event.data["startedDateTime"] <= actual[0].data["startedDateTime"]:
                    raise ValueError("actual closure does not follow the recorded checklist Save")
                request_proof = {
                    "$[0].operationName": operation["operationName"],
                    "$[0].variables.input.iid": input_value["iid"],
                    "$[0].variables.input.projectPath": input_value["projectPath"],
                    "$[0].variables.input.stateEvent": input_value["stateEvent"],
                }
                response_proof = {
                    "$[0].data.updateIssue.issuable.id": entity["id"],
                    "$[0].data.updateIssue.issuable.state": entity["state"],
                }
                close_events.append(
                    {
                        "url_template": closed_event.url.replace(site_url, "__GITLAB__", 1),
                        "http_method": closed_event.http_method,
                        "response_status": closed_event.request_status,
                        "post_data": request_proof,
                        "post_data_schema": {
                            "type": "object",
                            "properties": {path: {"type": "string"} for path in request_proof},
                        },
                        "response_content": response_proof,
                        "last_event_only": False,
                    }
                )
            if len(close_events) != 1:
                raise ValueError(
                    "Close lacks one successful canonical response proving closed state"
                )
            archived["close_completion_event"] = close_events[0]
            completion_events = close_events
            del parameters["close_issue_label"]
            parameters["closed_state_control_label"] = archived["closed_state_control"]["text"]
            final_checks[1] = {
                "kind": "visible_text_exact",
                "parameter": "closed_state_control_label",
                "selector": archived["closed_state_selector"],
            }
        definition = {
            "task_type": "MUTATE",
            "goal_template": goal,
            "parameters": parameters,
            "answer": {"status": "SUCCESS", "values": None, "item_schema": None, "ordered": False},
            "completion_events": completion_events,
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
                "location": "/save_requests/0/post_data/"
                + path.replace("~", "~0").replace("/", "~1"),
                "derivation": (
                    "Actual successful UI Save parsed by the canonical HAR owner; "
                    "no synthesized request."
                ),
            }
            for path in critical
        ]
        if family == "filtered_issue_checklist_choice_and_close":
            evidence = evidence[:4] + [
                {
                    "field": "/task_definition/completion_events/0/"
                    + content
                    + "/"
                    + path.replace("~", "~0").replace("/", "~1"),
                    "artifact": "observation",
                    "modality": "text",
                    "location": "/close_completion_event/"
                    + content
                    + "/"
                    + path.replace("~", "~0").replace("/", "~1"),
                    "derivation": (
                        "Actual matched updateIssue request and response identify "
                        "the same closed issue."
                    ),
                }
                for content in ["post_data", "response_content"]
                for path in completion_events[0][content]
            ]
    draft = {
        "generated_task_id": case["generated_task_id"],
        "task_id": case["task_id"],
        "intent_template_id": case["intent_template_id"],
        "site": "gitlab",
        "start_url": "__GITLAB__",
        "revision": 2 if family == "filtered_issue_checklist_choice_and_close" else 1,
        "source_artifact": case["source_artifact"],
        "source_record_id": case["source_record_id"],
        "baseline": baseline,
        "task_definition": definition,
        "grounding": {
            "captured_at": (
                archived["recorded_provenance"]["captured_at"]
                if "recorded_provenance" in archived
                else datetime.now(UTC).isoformat()
            ),
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


def compile_recorded(args: argparse.Namespace) -> None:
    """Ingest immutable closure evidence; issue no browser, HTTP or model calls."""
    from webarena_verified.core.utils.checksum import compute_evaluator_checksum
    from webarena_verified.core.utils.network_event_utils import load_har_trace

    config = json.loads(args.task_config.read_text())
    cases = [
        case for case in config["cases"] if case["generated_task_id"] == args.generated_task_id
    ]
    if len(cases) != 1 or cases[0]["family"] != "filtered_issue_checklist_choice_and_close":
        raise ValueError("recorded ingestion requires one configured checklist-closure case")
    case = cases[0]
    source = args.recorded_directory
    baseline = json.loads(args.baseline_receipt.read_text())
    site_url = args.site_url.rstrip("/")
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("recorded capture requires its original fresh owned baseline")
    if baseline["site_url"].rstrip("/") != site_url or baseline != json.loads(
        (source / "baseline.json").read_text()
    ):
        raise ValueError("recorded endpoint/baseline differ from the immutable capture")
    names = [
        "initial",
        "assigned_dashboard",
        "assigned_issues",
        "before",
        "editor_before",
        "unsaved",
        "saved",
        "closed",
        "revisited",
        "editor_after",
        "after",
    ]
    archived = {name: json.loads((source / f"{name}.json").read_text()) for name in names}
    assigned_url = config["assigned_issues_link"].replace("__GITLAB__", site_url)
    if (
        assigned_url not in {link["url"] for link in archived["initial"]["links"]}
        or archived["assigned_dashboard"]["url"] != assigned_url
    ):
        raise ValueError("recorded assigned scope does not follow the visible dashboard link")
    open_urls = {
        link["url"]
        for link in archived["assigned_dashboard"]["links"]
        if re.fullmatch(r"Open\s+\d+", link["text"])
    }
    if open_urls != {archived["assigned_issues"]["url"]}:
        raise ValueError("recorded Open listing does not follow its visible tab")
    matching, target = _select_existing_issue(case, archived["assigned_issues"])
    before = archived["before"]
    if (
        before["url"] != target["url"]
        or before["issue_state"]["title"] != target["title"]
        or before["issue_state"]["open"] is not True
    ):
        raise ValueError("recorded detail differs from the uniquely selected open issue")
    if any(archived[name]["url"] != target["url"] for name in names[4:]):
        raise ValueError("recorded mutation stages do not belong to the selected detail")
    descriptions = [
        control["value"]
        for control in archived["editor_before"]["editor_controls"]
        if "Description" in control["labels"]
    ]
    if len(descriptions) != 1:
        raise ValueError("recorded original editable Markdown is absent or ambiguous")
    title, description, parameters, goal = _checklist_goal(case, before, descriptions[0])
    events = load_har_trace(source / "network.har")
    issue_number = str(int(target["reference"].split("#")[1]))
    saves = [
        event
        for event in events
        if event.is_request_success
        and urlsplit(event.url).netloc == urlsplit(site_url).netloc
        and event.http_method not in {"GET", "HEAD"}
        and isinstance(event.post_data, dict)
        and event.post_data.get("description") == description
        and event.post_data.get("title") == title
        and isinstance(event.response_content, dict)
        and str(event.response_content.get("iid")) == issue_number
    ]
    if len(saves) != 1:
        raise ValueError("recorded editor Save requires one actual successful matching entity/body")
    save = saves[0]
    archived.update(
        matching_issues=matching,
        target=target,
        checklist_saved=archived["saved"],
        saved=archived["closed"],
        save_requests=[
            {
                "url": save.url,
                "method": save.http_method,
                "status": save.request_status,
                "request_body_sha256": hashlib.sha256(
                    save.data["request"]["postData"]["text"].encode()
                ).hexdigest(),
            }
        ],
        close_responses=[
            {"url": event.url, "method": event.http_method, "status": event.request_status}
            for event in events
            if event.is_request_success
            and urlsplit(event.url).netloc == urlsplit(site_url).netloc
            and event.http_method not in {"GET", "HEAD"}
            and event.data["startedDateTime"] > save.data["startedDateTime"]
        ],
    )
    original_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.iterdir())
        if path.is_file()
    }
    archived["recorded_provenance"] = {
        "original_directory": str(source.resolve()),
        "original_owner": str(args.recorded_owner.resolve()),
        "original_owner_sha256": hashlib.sha256(args.recorded_owner.read_bytes()).hexdigest(),
        "original_artifact_sha256": original_hashes,
        "original_config_sha256": hashlib.sha256(args.task_config.read_bytes()).hexdigest(),
        "original_baseline_sha256": hashlib.sha256(args.baseline_receipt.read_bytes()).hexdigest(),
        "derivation_owner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluator_checksum": compute_evaluator_checksum(),
        "captured_at": max(event.data["startedDateTime"] for event in events),
        "derived_at": datetime.now(UTC).isoformat(),
        "gui_actions": 0,
        "model_calls": 0,
        "historical_failure_preserved": True,
        "ingestion": "immutable-local-evidence",
    }
    os.umask(0o077)
    output = (
        args.batch_root
        / "reports"
        / "new_gitlab_recorded"
        / case["generated_task_id"]
        / args.attempt
    )
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in original_hashes:
        shutil.copy2(source / name, output / name)
    _emit_grounded_mutation(args, case, baseline, output, archived, parameters, goal, [description])


def _select_three_issues(case, listing):
    """Select three existing Issues from the complete visible Assigned Open scope."""
    counts = [
        int(link["text"].split()[-1])
        for link in listing["links"]
        if re.fullmatch(r"Open\s+\d+", link["text"])
    ]
    if counts != [len(listing["issues"])]:
        raise ValueError("three-Issue selection requires the entire visible Open scope")
    if case["selection_count"] != 3 or case["selection_window"] not in {"lowest", "highest"}:
        raise ValueError("three-Issue selection requires lowest/highest three")
    plan = case["responsibility_plan"]
    if len(plan) != 3 or any(
        len(choices) != 2 or any(type(choice) is not bool for choice in choices) for choices in plan
    ):
        raise ValueError("each ascending-rank Issue requires two boolean choices")
    matching = sorted(
        [
            row
            for row in listing["issues"]
            if row["reference"].split("#")[0] == case["project_path"]
            and set(case["required_labels"]) <= set(row["labels"])
            and not set(case["excluded_labels"]) & set(row["labels"])
            and (case["author"] is None or row["author"] == case["author"])
        ],
        key=lambda row: int(row["reference"].split("#")[1]),
    )
    if len(matching) < 3 or len({row["url"] for row in matching}) != len(matching):
        raise ValueError("fewer than three distinct matching visible Issues")
    return matching[:3] if case["selection_window"] == "lowest" else matching[-3:]


def _three_issue_goal(case):
    """Expose selection, original-discussion branching and rank-specific duties."""
    parameters = {
        "project_path": case["project_path"],
        "assignee": "Byte Blaze",
        "selection_rule": case["selection_rule"],
        "handoff_pattern": case["handoff_comment_pattern"],
        "responsibility_1": case["checklist_labels"][0],
        "responsibility_2": case["checklist_labels"][1],
        "conduct_label": case["checklist_labels"][2],
    }
    for index, choices in enumerate(case["responsibility_plan"]):
        for member, choice in enumerate(choices):
            parameters[f"rank_{index + 1}_choice_{member + 1}"] = (
                "checked" if choice else "unchecked"
            )
    goal = (
        "From the complete Open issues assigned to {assignee} in {project_path}, "
        "{selection_rule}. Before changing any of the three Issues, read all three full "
        "Descriptions and count each Issue's initial user discussion comments, excluding "
        "system activity notes. Number the selected Issues 1–3 in ascending numeric Issue "
        "order. For rank 1 use {rank_1_choice_1}/{rank_1_choice_2}, for rank 2 use "
        "{rank_2_choice_1}/{rank_2_choice_2}, and for rank 3 use "
        "{rank_3_choice_1}/{rank_3_choice_2}, respectively for '{responsibility_1}' and "
        "'{responsibility_2}'. On each Issue append exactly one ordinary public comment "
        "by filling this pattern with its original reference, title, initial user-comment "
        "count and its supplied checked/unchecked choices: {handoff_pattern} "
        "Then use Edit title and description to change only these two original Markdown "
        "checkbox markers and save the entire Description; preserve every other Markdown "
        "character, the original title, labels, assignee and all existing user comments, "
        "and keep '{conduct_label}' checked. Close an Issue only when it originally had "
        "zero user comments AND both supplied responsibility choices are checked; otherwise "
        "keep it Open and do not Close it. Use the frozen initial discussion counts, "
        "before any new comment. After finishing all three, revisit and refresh each to "
        "confirm its comment, checklist and state persisted. Change no other Issue. "
        "Finish on this project's Issue List with the All tab active, showing the three "
        "selected Issues together with their final states."
    )
    return parameters, goal


def _build_three_issue_reference(args, case, output, archived):
    """Bind three complete independent handoffs and the final discussion snapshots."""
    from webarena_verified.core.evaluation.data_types import NormalizedString
    from webarena_verified.core.utils.network_event_utils import load_har_trace

    events = load_har_trace(output / "network.har")
    selected = _select_three_issues(case, archived["assigned_issues"])
    stages = archived["issues"]
    if len(stages) != 3 or [stage["target"] for stage in stages] != selected:
        raise ValueError("captured Issue identities differ from the initial selection")
    frozen = datetime.fromisoformat(archived["initial_all_read_completed_at"])
    writes = [
        event
        for event in events
        if event.http_method in {"POST", "PUT", "PATCH", "DELETE"}
        and (
            urlsplit(event.url).path != "/api/graphql"
            or isinstance(event.post_data, list)
            and any(item.get("operationName") == "updateIssue" for item in event.post_data)
        )
    ]
    if any(datetime.fromisoformat(event.data["startedDateTime"]) <= frozen for event in writes):
        raise ValueError("business write preceded completion of all three initial GUI reads")
    parameters, goal = _three_issue_goal(case)
    completion, evidence = [], []
    for index, stage in enumerate(stages):
        stage_case = {**case, "responsibility_choices": case["responsibility_plan"][index]}
        definition, _, bindings = author_gitlab_mutations_handoff_v4._build_handoff_reference(
            args, stage_case, output, stage
        )
        # The raw POST and final GET prove the same actual note at authoring time.
        # A new note ID depends on legal cross-Issue execution order, so scoring
        # binds its entity/content rather than a generated sequence number.
        del definition["completion_events"][0]["response_content"]['$["id"]']
        bindings = [
            binding
            for binding in bindings
            if binding["field"]
            != '/task_definition/completion_events/0/response_content/$["id"]'
        ]
        target = stage["target"]
        target_path = urlsplit(target["url"]).path
        put = [
            event
            for event in events
            if event.http_method == "PUT" and urlsplit(event.url).path == target_path + ".json"
        ]
        if len(put) != 1:
            raise ValueError("formal capture must contain exactly one full Save per Issue")
        editor = put[0]
        body = editor.response_content
        if (
            body["description"] != definition["parameters"]["expected_saved_description"]
            or body["title"] != definition["parameters"]["final_title"]
        ):
            raise ValueError("complete raw Save response Markdown/title differ")
        saved_labels = [label["title"] for label in body["labels"]]
        saved_assignees = [assignee["id"] for assignee in body["assignees"]]
        if sorted(saved_labels) != sorted(stage["before"]["issue_state"]["labels"]):
            raise ValueError("raw Save response must preserve the original visible label set")
        stage["raw_saved_labels"] = saved_labels
        stage["raw_saved_assignee_ids"] = saved_assignees
        for label_index, label in enumerate(saved_labels):
            definition["parameters"][f"raw_saved_label_{label_index}"] = label
        definition["completion_events"][1].update(
            last_event_only=True,
            response_content={
                '$["id"]': body["id"],
                '$["iid"]': body["iid"],
                '$["title"]': "{final_title}",
                '$["description"]': "{expected_saved_description}",
                '$["labels"][?(@.title != null)].title': [
                    f"{{raw_saved_label_{label_index}}}" for label_index in range(len(saved_labels))
                ],
                '$["assignees"][?(@.id != null)].id': saved_assignees,
            },
        )
        discussions = [
            event
            for event in events
            if event.http_method == "GET"
            and urlsplit(event.url).path == target_path + "/discussions.json"
        ]
        initial = [
            event
            for event in discussions
            if datetime.fromisoformat(event.data["startedDateTime"]) <= frozen
        ]
        final = [
            event
            for event in discussions
            if event.data["startedDateTime"] > editor.data["startedDateTime"]
        ]
        if not initial or not final or not final[-1].is_request_success:
            raise ValueError("each Issue needs actual initial and final GUI discussions GETs")
        raw_initial = [
            note["note"]
            for group in initial[-1].response_content
            for note in group["notes"]
            if note["system"] is False
        ]
        raw_final = [
            note["note"]
            for group in final[-1].response_content
            for note in group["notes"]
            if note["system"] is False
        ]
        if len(raw_initial) != len(stage["initial_user_comments"]) or sorted(raw_final) != sorted(
            raw_initial + [stage["handoff_comment"]]
        ):
            raise ValueError("complete final raw user discussion must preserve initial notes once")
        stage["initial_raw_user_notes"] = raw_initial
        stage["final_raw_user_notes"] = raw_final
        definition["parameters"]["final_raw_user_notes"] = raw_final
        for note_index, note_text in enumerate(raw_final):
            definition["parameters"][f"final_raw_user_note_{note_index}"] = note_text
        definition["completion_events"].append(
            {
                "url_template": final[-1].url.replace(args.site_url.rstrip("/"), "__GITLAB__", 1),
                "http_method": "GET",
                "response_status": final[-1].request_status,
                "response_content": {
                    "$[*].notes[?(@.system == false)].note": [
                        f"{{final_raw_user_note_{note_index}}}"
                        for note_index in range(len(raw_final))
                    ]
                },
                "last_event_only": True,
            }
        )
        number = str(int(target["reference"].split("#")[1]))
        definition["completion_events"].append(
            {
                "url_template": "__GITLAB__/api/graphql",
                "http_method": "POST",
                "post_data": {
                    "$[0].operationName": "updateIssue",
                    "$[0].variables.input.iid": number,
                    "$[0].variables.input.projectPath": case["project_path"],
                    "$[0].variables.input.stateEvent": "REOPEN",
                },
                "should_not_exist": True,
                "last_event_only": False,
            }
        )
        prefix = f"issue_{index}_"
        names = set(definition["parameters"])

        def namespace(value):
            if isinstance(value, str):
                return re.sub(
                    r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}",
                    lambda match: "{" + prefix + match[1] + "}" if match[1] in names else match[0],
                    value,
                )
            if isinstance(value, list):
                return [namespace(item) for item in value]
            if isinstance(value, dict):
                return {key: namespace(item) for key, item in value.items()}
            return value

        parameters.update({prefix + key: value for key, value in definition["parameters"].items()})
        offset = len(completion)
        completion.extend(namespace(definition["completion_events"]))
        for binding in bindings:
            field = binding["field"]
            if field.startswith("/task_definition/parameters/"):
                field = field.replace(
                    "/task_definition/parameters/", "/task_definition/parameters/" + prefix, 1
                )
            else:
                field = re.sub(
                    r"/completion_events/(\d+)/",
                    lambda match: f"/completion_events/{int(match[1]) + offset}/",
                    field,
                )
            evidence.append(
                {**binding, "field": field, "location": f"/issues/{index}" + binding["location"]}
            )
        for name, location in [("final_raw_user_notes", "/final_raw_user_notes")]:
            evidence.append(
                {
                    "field": "/task_definition/parameters/" + prefix + name,
                    "artifact": "observation",
                    "modality": "text",
                    "location": f"/issues/{index}" + location,
                    "derivation": "Complete actual GUI-triggered final discussions GET; "
                    "raw original user-note multiset plus one actual public handoff.",
                }
            )
        stage["three_issue_completion_events"] = namespace(definition["completion_events"])
        evidence.extend(
            {
                "field": "/task_definition/parameters/"
                + prefix
                + f"final_raw_user_note_{note_index}",
                "artifact": "observation",
                "modality": "text",
                "location": f"/issues/{index}/final_raw_user_notes/{note_index}",
                "derivation": "One full raw user note from the final GUI discussion snapshot; "
                "list members render once, preserving literal Markdown braces.",
            }
            for note_index in range(len(raw_final))
        )
        evidence.extend(
            {
                "field": "/task_definition/parameters/" + prefix + f"raw_saved_label_{label_index}",
                "artifact": "observation",
                "modality": "text",
                "location": f"/issues/{index}/raw_saved_labels/{label_index}",
                "derivation": "Actual unique successful Issue PUT raw label title, "
                "equal to the initial and refreshed visible label set.",
            }
            for label_index in range(len(saved_labels))
        )
        for local_index, event in enumerate(definition["completion_events"]):
            for content in ["post_data", "response_content"]:
                for path in event.get(content, {}):
                    field = f"/task_definition/completion_events/{offset + local_index}/{content}/"
                    field += path.replace("~", "~0").replace("/", "~1")
                    evidence = [item for item in evidence if item["field"] != field]
                    evidence.append(
                        {
                            "field": field,
                            "artifact": "observation",
                            "modality": "text",
                            "location": f"/issues/{index}/three_issue_completion_events/"
                            f"{local_index}/{content}/"
                            + path.replace("~", "~0").replace("/", "~1"),
                            "derivation": "Entity-bound actual request/raw response or explicit "
                            "no-Reopen/Open policy; shared GraphQL uses all events.",
                        }
                    )
    selected_save_paths = "|".join(
        re.escape(urlsplit(row["url"]).path.lstrip("/")) + r"\.json(?:\?.*)?$" for row in selected
    )
    # Actual Issue editor PUT paths own this prohibition, including unlisted Issues.
    completion.append(
        {
            "url_template": "^__GITLAB__/(?!(?:"
            + selected_save_paths
            + r")).+/-/issues/[0-9]+\.json(?:\?.*)?$",
            "http_method": "PUT",
            "should_not_exist": True,
            "last_event_only": False,
        }
    )
    selected_entities = "|".join(
        str(stage["three_issue_completion_events"][1]["response_content"]['$["id"]'])
        for stage in stages
    )
    selected_numbers = "|".join(row["reference"].split("#")[1] for row in selected)
    # Comments carry global Issue IDs. GraphQL state writes carry project + IID;
    # both components bind the prohibition, including overlapping IIDs elsewhere.
    completion.extend(
        [
            {
                "url_template": r"^__GITLAB__/.+/notes$",
                "http_method": "POST",
                "query_params": {"target_id": ["^[0-9]+$"], "target_type": ["issue"]},
                "post_data": {
                    '$["note"]["noteable_type"]': "Issue",
                    '$["note"]["noteable_id"]': "^(?!(?:" + selected_entities + r")$)[0-9]+$",
                },
                "post_data_schema": {
                    "type": "object",
                    "properties": {
                        '$["note"]["noteable_type"]': {"type": "string"},
                        '$["note"]["noteable_id"]': {"type": "string"},
                    },
                },
                "should_not_exist": True,
                "last_event_only": False,
            },
            {
                "url_template": "__GITLAB__/api/graphql",
                "http_method": "POST",
                "post_data": {
                    "$[0].operationName": "updateIssue",
                    "$[0].variables.input.stateEvent": "^(?:CLOSE|REOPEN)$",
                    "$[0].variables.input.projectPath": case["project_path"],
                    "$[0].variables.input.iid": "^(?!(?:" + selected_numbers + r")$)[0-9]+$",
                },
                "should_not_exist": True,
                "last_event_only": False,
            },
            {
                "url_template": "__GITLAB__/api/graphql",
                "http_method": "POST",
                "post_data": {
                    "$[0].operationName": "updateIssue",
                    "$[0].variables.input.stateEvent": "^(?:CLOSE|REOPEN)$",
                    "$[0].variables.input.projectPath": "^(?!"
                    + re.escape(NormalizedString(case["project_path"]).normalized)
                    + r"$).+$",
                },
                "should_not_exist": True,
                "last_event_only": False,
            },
        ]
    )
    archived["three_issue_completion_events"] = completion
    for index in range(len(completion) - 3, len(completion)):
        for content in ["post_data", "response_content"]:
            for path in completion[index].get(content, {}):
                escaped = path.replace("~", "~0").replace("/", "~1")
                evidence.append(
                    {
                        "field": f"/task_definition/completion_events/{index}/{content}/{escaped}",
                        "artifact": "observation",
                        "modality": "text",
                        "location": f"/three_issue_completion_events/{index}/{content}/{escaped}",
                        "derivation": "No-other-Issue policy over actual global editor IDs "
                        "and project/reference identities; "
                        "observed comment and state writers only.",
                    }
                )
    return {
        "task_type": "MUTATE",
        "goal_template": goal,
        "parameters": parameters,
        "answer": {"status": "SUCCESS", "values": None, "item_schema": None, "ordered": False},
        "completion_events": completion,
    }, evidence


def _three_issue_all_terminal(args, case, archived, definition):
    """Use the actual project All page's stable visible fields, never avatars or dates."""
    summary, activation = archived["all_summary"], archived["all_tab_activation"]
    rows = summary["issues"]
    if (
        activation["aria_selected"] != "true"
        or activation["row_count"] != activation["total"]
        or len(rows) != activation["total"]
        or urlsplit(summary["url"]).path.rstrip("/") != "/" + case["project_path"] + "/-/issues"
        or len({row["reference"] for row in rows}) != len(rows)
    ):
        raise ValueError("terminal project All tab must be active with the complete visible list")
    parameters = definition["parameters"]
    parameters.update(
        final_summary_url=summary["url"].replace(args.site_url.rstrip("/"), "__GITLAB__", 1),
        final_all_tab_text=activation["text"],
        final_all_references=[row["reference"] for row in rows],
        unexpected_summary_rows=0,
    )
    checks = [
        {"kind": "final_url", "parameter": "final_summary_url"},
        {
            "kind": "visible_text_exact",
            "parameter": "final_all_tab_text",
            "selector": 'a.nav-link.active[aria-selected="true"]:has-text("All")',
        },
        {
            "kind": "text_set_preserved",
            "parameter": "final_all_references",
            "selector": "li.issue .issuable-reference",
            "added_count_parameter": "unexpected_summary_rows",
            "added_name_parameter": "project_path",
        },
    ]
    for index, stage in enumerate(archived["issues"]):
        matches = [row for row in rows if row["url"] == stage["target"]["url"]]
        if len(matches) != 1:
            raise ValueError("each selected Issue must appear once on the final All page")
        row = matches[0]
        closed = "CLOSED" in row["body_text"].splitlines()
        state = stage["after"]["issue_state"]
        if (
            closed is state["open"]
            or row["title"] != state["title"]
            or set(row["labels"]) != set(state["labels"])
            or row["reference"] != "#" + stage["target"]["reference"].split("#")[1]
            or int(row["comments_text"]) != len(stage["final_user_comments"])
        ):
            raise ValueError(
                "All-page title/state/reference/labels/comment count differ from persisted Issues"
            )
        prefix = f"summary_{index}_"
        parameters.update(
            {
                prefix + "title": row["title"],
                prefix + "reference": row["reference"],
                prefix + "labels": row["labels"],
                prefix + "comments": row["comments_text"],
            }
        )
        path = urlsplit(row["url"]).path
        selector = "li.issue:has(a.issue-title-text[href$=" + json.dumps(path) + "])"
        selector += (
            ':has(li.issuable-status:has-text("CLOSED"))'
            if closed
            else ':not(:has(li.issuable-status:has-text("CLOSED")))'
        )
        checks.extend(
            [
                {
                    "kind": "visible_text_exact",
                    "parameter": prefix + "title",
                    "selector": selector + " a.issue-title-text",
                },
                {
                    "kind": "visible_text_exact",
                    "parameter": prefix + "reference",
                    "selector": selector + " .issuable-reference",
                },
                {
                    "kind": "text_set_preserved",
                    "parameter": prefix + "labels",
                    "selector": selector + " .gl-label-text",
                    "added_count_parameter": "unexpected_summary_rows",
                    "added_name_parameter": "project_path",
                },
                {
                    "kind": "visible_text_exact",
                    "parameter": prefix + "comments",
                    "selector": selector + " a[href$=" + json.dumps(path + "#notes") + "]",
                },
            ]
        )
    sources = {
        "final_summary_url": (
            "/all_summary/url",
            "Actual terminal project All URL after the observed All tab click.",
        ),
        "final_all_tab_text": (
            "/all_tab_activation/text",
            "Actual active All tab visible text and count.",
        ),
        "final_all_references": (
            "/all_summary/issues",
            "Complete actual visible reference collection, not hidden resource inventory.",
        ),
        "unexpected_summary_rows": (
            "/all_summary/issues",
            "No added reference or label outside the formal visible collection is allowed.",
        ),
    }
    for index, stage in enumerate(archived["issues"]):
        row_index = next(i for i, row in enumerate(rows) if row["url"] == stage["target"]["url"])
        for field, source in [
            ("title", "title"),
            ("reference", "reference"),
            ("labels", "labels"),
            ("comments", "comments_text"),
        ]:
            sources[f"summary_{index}_{field}"] = (
                f"/all_summary/issues/{row_index}/{source}",
                "Actual same-page visible field matches this entity's formal persisted readback; "
                "state is bound by presence or absence of its literal CLOSED status.",
            )
    archived["all_terminal_field_evidence"] = [
        {
            "field": "/task_definition/parameters/" + name,
            "artifact": "observation",
            "modality": "text",
            "location": source,
            "derivation": derivation,
        }
        for name, (source, derivation) in sources.items()
    ]
    return checks


def ground_three_issues(args):
    """Read all initial objects before writes, then persist and reread three handoffs."""
    config = json.loads(args.task_config.read_text())
    cases = [
        case for case in config["cases"] if case["generated_task_id"] == args.generated_task_id
    ]
    if (
        len(cases) != 1
        or cases[0]["family"] != "three_issue_initial_discussion_handoff_and_resolution"
    ):
        raise ValueError("three-Issue mode requires exactly one configured case of its family")
    case = cases[0]
    baseline = json.loads(args.baseline_receipt.read_text())
    site_url = args.site_url.rstrip("/")
    if (
        baseline["fresh_from_image"] is not True
        or not baseline["owned_run"]
        or baseline["site_url"].rstrip("/") != site_url
    ):
        raise ValueError("three-Issue capture requires its fresh owned fixed-image endpoint")
    os.umask(0o077)
    output = args.batch_root / "author" / case["generated_task_id"] / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    site, blocked, save_phase = urlsplit(site_url), [], False
    read_documents = set(config["observed_graphql_read_documents"])

    def restrict(route):
        requested = urlsplit(route.request.url)
        same = (requested.scheme, requested.netloc) == (site.scheme, site.netloc)
        read = route.request.method in {"GET", "HEAD"}
        if same and route.request.method == "POST" and requested.path == "/api/graphql":
            payload = json.loads(route.request.post_data)
            items = payload if isinstance(payload, list) else [payload]
            read = bool(items) and all(
                isinstance(item, dict)
                and isinstance(item.get("query"), str)
                and hashlib.sha256(item["query"].encode()).hexdigest() in read_documents
                for item in items
            )
        if same and (read or save_phase):
            route.continue_()
        else:
            blocked.append({"method": route.request.method, "path": requested.path})
            route.abort()

    from playwright.sync_api import sync_playwright

    archived = {"issues": [], "blocked_requests": blocked}
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = None
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
            page = context.new_page()
            page.goto(site_url + "/", wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            initial = _record_editor(page, output, "initial")
            assigned = config["assigned_issues_link"].replace("__GITLAB__", site_url)
            if not any(link["url"] == assigned for link in initial["links"]):
                raise ValueError("fresh dashboard does not expose the configured Assigned link")
            page.goto(assigned, wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            dashboard = _record_editor(page, output, "assigned_dashboard")
            open_urls = {
                link["url"]
                for link in dashboard["links"]
                if re.fullmatch(r"Open\s+\d+", link["text"])
            }
            if len(open_urls) != 1:
                raise ValueError("fresh Assigned view lacks one observed Open link")
            page.goto(open_urls.pop(), wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            listing = _record_editor(page, output, "assigned_issues")
            archived.update(initial=initial, assigned_dashboard=dashboard, assigned_issues=listing)
            targets = _select_three_issues(case, listing)
            user_notes = "li.note-comment:not(.system-note) .note-text.md:visible"
            # Freeze complete initial Markdown and user discussions for all three first.
            for index, target in enumerate(targets):
                page.goto(target["url"], wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                page.locator("li.note-skeleton:visible").first.wait_for(state="hidden")
                before = _record_editor(page, output, f"issue_{index}_before")
                if (
                    before["issue_state"]["title"] != target["title"]
                    or not before["issue_state"]["open"]
                ):
                    raise ValueError("selected detail differs from the initial Open issue identity")
                users = [text.strip() for text in page.locator(user_notes).all_inner_texts()]
                page.get_by_role("button", name="Edit title and description", exact=True).click()
                editor = _record_editor(page, output, f"issue_{index}_editor_before")
                single = {**case, "responsibility_choices": case["responsibility_plan"][index]}
                _, description, comment, _, _ = _handoff_goal(
                    single,
                    before,
                    page.get_by_label("Description", exact=True).input_value(),
                    users,
                    target,
                )
                page.get_by_role("button", name="Cancel", exact=True).click()
                cancel = _record_editor(page, output, f"issue_{index}_cancelled")
                if cancel["issue_state"] != before["issue_state"]:
                    raise ValueError("reading the full initial editor changed the Issue")
                archived["issues"].append(
                    {
                        "target": target,
                        "before": before,
                        "editor_before": editor,
                        "initial_user_comments": users,
                        "handoff_description": description,
                        "handoff_comment": comment,
                        "close_required": not users and all(single["responsibility_choices"]),
                        "save_requests": [],
                    }
                )
            archived["initial_all_read_completed_at"] = datetime.now(UTC).isoformat()
            # Each Issue gets one ordinary comment and one complete Description Save.
            for index, stage in enumerate(archived["issues"]):
                page.goto(stage["target"]["url"], wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                for action in ["Comment", "Save changes"]:
                    if action == "Comment":
                        page.get_by_test_id("comment-field").fill(stage["handoff_comment"])
                    else:
                        page.get_by_role(
                            "button", name="Edit title and description", exact=True
                        ).click()
                        page.get_by_label("Description", exact=True).fill(
                            stage["handoff_description"]
                        )
                    _record_editor(
                        page, output, f"issue_{index}_unsaved_{len(stage['save_requests'])}"
                    )
                    save_phase = True
                    try:
                        with page.expect_response(
                            lambda response: (
                                response.request.method in {"POST", "PUT", "PATCH"}
                                and urlsplit(response.url).netloc == site.netloc
                                and urlsplit(response.url).path != "/api/graphql"
                            ),
                            timeout=args.timeout_ms,
                        ) as saved:
                            page.get_by_role("button", name=action, exact=True).click()
                    finally:
                        save_phase = False
                    response = saved.value
                    stage["save_requests"].append(
                        {
                            "url": response.url,
                            "method": response.request.method,
                            "status": response.status,
                            "request_body_sha256": hashlib.sha256(
                                response.request.post_data.encode()
                            ).hexdigest(),
                        }
                    )
                    page.wait_for_timeout(int(args.settle_seconds * 1000))
                    _record_editor(
                        page, output, f"issue_{index}_saved_{len(stage['save_requests'])}"
                    )
                if stage["close_required"]:
                    save_phase = True
                    try:
                        close = page.locator("button.js-note-target-close:visible").filter(
                            has_text=re.compile(r"^\s*Close issue\s*$")
                        )
                        close.click()
                        close.wait_for(state="hidden")
                        page.wait_for_timeout(int(args.settle_seconds * 1000))
                    finally:
                        save_phase = False
                    _record_editor(page, output, f"issue_{index}_closed")
            # Read all three persisted entities only after all business writes finish.
            for index, stage in enumerate(archived["issues"]):
                page.goto(stage["target"]["url"], wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                page.locator("li.note-skeleton:visible").first.wait_for(state="hidden")
                stage["revisited"] = _record_editor(page, output, f"issue_{index}_revisited")
                page.reload(wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                page.get_by_role("button", name="Edit title and description", exact=True).click()
                stage["editor_after"] = _record_editor(page, output, f"issue_{index}_editor_after")
                page.get_by_role("button", name="Cancel", exact=True).click()
                page.reload(wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                page.locator("li.note-skeleton:visible").first.wait_for(state="hidden")
                stage["after"] = _record_editor(page, output, f"issue_{index}_after")
                stage["final_user_comments"] = [
                    text.strip() for text in page.locator(user_notes).all_inner_texts()
                ]
            lists = {
                link["url"]
                for link in archived["issues"][-1]["after"]["links"]
                if link["text"] == "List"
            }
            if len(lists) != 1:
                raise ValueError("selected Issue does not expose one actual project List link")
            page.goto(lists.pop(), wait_until="networkidle")
            archived["project_issues"] = _record_editor(page, output, "project_issues")
            tabs = [
                link
                for link in archived["project_issues"]["links"]
                if re.fullmatch(r"All\s+\d+", link["text"])
            ]
            if len(tabs) != 1:
                raise ValueError("actual project List lacks one visible All tab")
            total = int(tabs[0]["text"].split()[-1])
            all_tab = page.locator("a:visible").filter(has_text=re.compile(r"^\s*All\s+\d+\s*$"))
            if all_tab.count() != 1:
                raise ValueError("actual All tab is not a unique visible link")
            all_tab.click()
            page.wait_for_load_state("networkidle")
            page.wait_for_function(
                "n => [...document.querySelectorAll('li.issue')].filter(e => {"
                "const r=e.getBoundingClientRect();return r.width&&r.height&&"
                "getComputedStyle(e).visibility!=='hidden';}).length===n",
                arg=total,
            )
            archived["all_tab_activation"] = all_tab.evaluate(
                "e => ({text:e.innerText.trim(),aria_selected:e.getAttribute('aria-selected')})"
            )
            archived["all_tab_activation"].update(
                total=total, row_count=page.locator("li.issue:visible").count()
            )
            archived["all_summary"] = _record_editor(page, output, "after")
        finally:
            if context is not None:
                try:
                    context.tracing.stop(path=str(output / "trace.zip"))
                finally:
                    context.close()
            browser.close()
            (output / "capture.json").write_text(
                json.dumps(archived, ensure_ascii=False, indent=2) + "\n"
            )
    definition, evidence = _build_three_issue_reference(args, case, output, archived)
    checks = _three_issue_all_terminal(args, case, archived, definition)
    evidence.extend(archived["all_terminal_field_evidence"])
    (output / "observation.json").write_text(
        json.dumps(archived, ensure_ascii=False, indent=2) + "\n"
    )
    relative = output.relative_to(args.batch_root)
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
            "family": case["family"],
            "artifacts": {
                "har": str(relative / "network.har"),
                "trace": str(relative / "trace.zip"),
                "screenshot": str(relative / "after.png"),
                "observation": str(relative / "observation.json"),
            },
            "field_evidence": evidence,
            "final_state_checks": checks,
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
    if args.mode == "formal-three-issue":
        ground_three_issues(args)
        return
    if args.mode == "formal":
        ground_mutation(args)
        return
    if args.mode == "compile-recorded":
        compile_recorded(args)
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
