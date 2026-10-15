#!/usr/bin/env python3
"""Capture 48 fixed-image single-site cases from frozen initial GUI evidence.

Run with one reviewed case config and a fresh authentication-only baseline.
The ordered plan binds partial Markdown edits, inter-Issue evidence and repeated
edits of the same newly-created ordinary comment. It emits original HAR/PW and
refreshed observations for later native GT review; it calls no model.
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

from author_gitlab import _project_fields
from author_gitlab_hard12_v2 import change_responsibilities, feature_paragraph
from author_gitlab_mutations_handoff_v5 import _record_editor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-task-id", required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--attempt", default="grounding-01")
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    return parser.parse_args()


def markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    """Represent full cell content using the exact publicly stated escaping."""

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def source_table(
    issues: list[dict[str, Any]], *, ordinary: bool = False, paragraph: bool = True
) -> str:
    """Join original source identity, title, full paragraph and ordinary comments."""
    columns = ["Reference", "Title", "Initial ordinary comments"]
    if paragraph:
        columns.append("First feature paragraph")
    if ordinary:
        columns.append("Original ordinary comment text")
    rows = []
    for issue in sorted(issues, key=lambda item: item["iid"]):
        row = [issue["reference"], issue["title"], len(issue["notes"])]
        if paragraph:
            row.append(feature_paragraph(issue["markdown"]))
        if ordinary:
            row.append("\n\n".join(note["text"] for note in issue["notes"]) or "[]")
        rows.append(row)
    return markdown_table(columns, rows)


AGGREGATION_COLUMNS = [
    ["Project", "Branches", "Tags", "Assigned open issues", "Initial ordinary comments"],
    [
        "Project",
        "Branches",
        "Tags",
        "Assigned open issues authored by Byte Blaze",
        "Initial ordinary comments",
    ],
    [
        "Project",
        "Branches",
        "Tags",
        "Assigned open issues with ordinary comments",
        "Initial ordinary comments",
    ],
    [
        "Project",
        "Branches",
        "Tags",
        "Assigned open issues",
        "Initial ordinary comments",
        "References",
    ],
    ["Project", "Branches", "Tags", "Assigned open issues without ordinary comments", "References"],
    [
        "Project",
        "Branches",
        "Tags",
        "Feature assigned open issues",
        "Other assigned open issues",
        "Feature initial ordinary comments",
        "Other initial ordinary comments",
    ],
    ["Project", "Branches minus tags", "Assigned open issues", "Reference and title pairs"],
    [
        "Project",
        "Branches",
        "Tags",
        "All assigned open issues",
        "All references",
        "All initial ordinary comments",
        "Zero-comment assigned open issues",
        "Zero-comment references",
        "Zero-comment initial ordinary comments",
    ],
]


def plan_actions(
    case: dict[str, Any], issues: list[dict[str, Any]], projects: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute all 48 ordered operations solely from complete original GUI facts."""
    family, variant = case["family"], case["variant"]
    actions: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    plan: dict[str, Any] = {"actions": actions, "steps": steps}
    if family == "Assigned_and_project_metadata_aggregation":
        rows, members_by_project = [], {}
        for project in sorted(projects, key=lambda item: item["project_path"]):
            path = project["project_path"].strip("/")
            members = sorted(
                [item for item in issues if item["project_path"] == path],
                key=lambda item: item["iid"],
            )
            members_by_project[path] = members
            selected = members
            if variant == 1:
                selected = [item for item in members if item["author"] == "Byte Blaze"]
            elif variant == 2:
                selected = [item for item in members if item["notes"]]
            elif variant == 4:
                selected = [item for item in members if not item["notes"]]
            base = [path, project["branches"], project["tags"]]
            if variant in {0, 1, 2, 3}:
                row = base + [len(selected), sum(len(item["notes"]) for item in selected)]
                if variant == 3:
                    row.append(", ".join(item["reference"] for item in selected) or "[]")
            elif variant == 4:
                row = base + [
                    len(selected),
                    ", ".join(item["reference"] for item in selected) or "[]",
                ]
            elif variant == 5:
                feature = [item for item in members if "feature" in item["labels"]]
                other = [item for item in members if "feature" not in item["labels"]]
                row = base + [
                    len(feature),
                    len(other),
                    sum(len(i["notes"]) for i in feature),
                    sum(len(i["notes"]) for i in other),
                ]
            elif variant == 6:
                row = [
                    path,
                    project["branches"] - project["tags"],
                    len(members),
                    "; ".join(item["reference"] + ": " + item["title"] for item in members) or "[]",
                ]
            else:
                zero = [item for item in members if not item["notes"]]
                row = base + [
                    len(members),
                    ", ".join(i["reference"] for i in members) or "[]",
                    sum(len(i["notes"]) for i in members),
                    len(zero),
                    ", ".join(i["reference"] for i in zero) or "[]",
                    0,
                ]
            rows.append(row)
        scores = {}
        for project in projects:
            path = project["project_path"].strip("/")
            if variant in {0, 1, 5}:
                scores[path] = project["tags"]
            elif variant in {2, 7}:
                scores[path] = project["branches"]
            elif variant in {3, 4}:
                scores[path] = sum(len(i["notes"]) for i in members_by_project[path])
            else:
                scores[path] = project["branches"] - project["tags"]
        choose = min if variant in {1, 2, 4, 7} else max
        extremum = choose(scores.values())
        selected_projects = [
            p for p in projects if scores[p["project_path"].strip("/")] == extremum
        ]
        if len(selected_projects) != 1:
            raise ValueError("the requested project extremum must be unique in initial GUI facts")
        plan["create"] = {
            "project": selected_projects[0],
            "title": case["new_issue_title"],
            "description": "### Assigned evidence audit\n\n"
            + markdown_table(AGGREGATION_COLUMNS[variant], rows),
        }
        return plan
    features = sorted(
        [item for item in issues if "feature" in item["labels"]], key=lambda item: item["iid"]
    )
    zero = [item for item in features if item["author"] == "Byte Blaze" and not item["notes"]]
    discussed = [item for item in features if item["notes"]]
    help_wanted = [item for item in features if "help wanted" in item["labels"]]
    form = [item for item in zero if "help wanted" not in item["labels"]]
    if len(features) != 4 or len(discussed) != 1 or len(help_wanted) != 2 or len(form) != 1:
        raise ValueError("complete initial GUI cohort differs from observed four-member structure")
    anchor, form_target = discussed[0], form[0]

    def add_comment(issue: dict[str, Any], body: str) -> None:
        actions.setdefault(issue["reference"], {})["note"] = body
        steps.append({"kind": "create_note", "reference": issue["reference"], "body": body})

    def revise_comment(issue: dict[str, Any], body: str) -> None:
        actions[issue["reference"]]["edit_note"] = body
        steps.append({"kind": "edit_own_note", "reference": issue["reference"], "body": body})

    def save_description(issue: dict[str, Any], body: str) -> None:
        actions.setdefault(issue["reference"], {})["description"] = body
        steps.append({"kind": "save_description", "reference": issue["reference"], "body": body})

    def reread(issue: dict[str, Any]) -> None:
        steps.append(
            {
                "kind": "reread",
                "reference": issue["reference"],
                "expected_ordinary_count": len(issue["notes"])
                + int("note" in actions.get(issue["reference"], {})),
            }
        )

    def count_table(sources: list[dict[str, Any]]) -> str:
        return markdown_table(
            ["Reference", "Title", "Initial ordinary comments", "Refreshed ordinary comments"],
            [
                [
                    item["reference"],
                    item["title"],
                    len(item["notes"]),
                    len(item["notes"]) + int("note" in actions.get(item["reference"], {})),
                ]
                for item in sorted(sources, key=lambda i: i["iid"])
            ],
        )

    if family == "dependent_partial_Markdown_and_review":
        recipients = help_wanted if variant in {0, 3, 5, 6} else zero
        if variant == 2:
            recipients = [zero[0], zero[-1]]
        elif variant == 5:
            recipients = [
                i
                for i in zero
                if any(
                    text in feature_paragraph(i["markdown"])
                    for text in ["Netlify", "https://mastodon.social/"]
                )
            ]
        cohort = ", ".join(item["reference"] for item in recipients)
        for index, issue in enumerate(recipients):
            if variant == 0:
                choices = [index == 0, index != 0]
            elif variant == 1:
                choices = ["help wanted" in issue["labels"], "help wanted" not in issue["labels"]]
            elif variant == 2:
                choices = [index == 0, index != 0]
            elif variant == 3:
                choices = [index != 0, index == 0]
            elif variant == 4:
                choices = (
                    [True, True]
                    if "help wanted" not in issue["labels"]
                    else [
                        bool(
                            re.search(
                                r"(?m)^- \[[xX]\] " + re.escape(label) + r"$", issue["markdown"]
                            )
                        )
                        for label in [
                            "I would like to work on this feature",
                            "I would like the site maintainers to work on this feature",
                        ]
                    ]
                )
            elif variant == 5:
                choices = [
                    "https://mastodon.social/" in issue["markdown"],
                    "Netlify" in issue["markdown"],
                ]
            elif variant == 6:
                choices = [index != 0, False]
            else:
                choices = [index != len(recipients) - 1, index != 0]
            if not (variant == 4 and "help wanted" in issue["labels"]):
                save_description(issue, change_responsibilities(issue["markdown"], choices))
            body = (
                f"Review {issue['reference']}: {issue['title']}; initial ordinary comments: "
                f"{len(issue['notes'])}; eligible cohort: {cohort}; contributor: "
                f"{'checked' if choices[0] else 'unchecked'}; maintainer: "
                f"{'checked' if choices[1] else 'unchecked'}."
            )
            if variant == 2:
                peer = next(item for item in recipients if item["reference"] != issue["reference"])
                body += f" Peer: {peer['reference']}."
            if variant == 3:
                body += (
                    f" Anchor: {anchor['reference']}; anchor initial ordinary comments: "
                    f"{len(anchor['notes'])}."
                )
            add_comment(issue, body)
        if variant == 6:
            add_comment(anchor, "### Anchor summary\n\n" + source_table(help_wanted))
    elif family == "cross_object_evidence_transfer":
        if variant == 0:
            add_comment(anchor, "### Source handoff\n\n" + source_table(help_wanted))
        elif variant == 1:
            add_comment(
                form_target,
                "### Source handoff\n\n" + source_table([i for i in features if i != form_target]),
            )
        elif variant == 2:
            for target in help_wanted:
                add_comment(
                    target,
                    "### Source handoff\n\n"
                    + source_table([anchor], ordinary=True, paragraph=False),
                )
        elif variant == 3:
            add_comment(anchor, "### Source handoff\n\n" + source_table(help_wanted))
            add_comment(
                form_target,
                "### Source handoff\n\n" + source_table([i for i in zero if i != form_target]),
            )
        elif variant == 4:
            selected = [
                i
                for i in zero
                if any(
                    text in feature_paragraph(i["markdown"])
                    for text in ["Netlify", "https://mastodon.social/"]
                )
            ]
            remaining = [i for i in zero if i not in selected]
            if len(remaining) != 1:
                raise ValueError("content sources must leave exactly one zero-comment destination")
            add_comment(remaining[0], "### Source handoff\n\n" + source_table(selected))
        elif variant == 5:
            add_comment(help_wanted[0], "### Source handoff\n\n" + source_table([help_wanted[1]]))
            add_comment(help_wanted[1], "### Source handoff\n\n" + source_table([help_wanted[0]]))
            add_comment(anchor, "### Source handoff\n\n" + source_table(help_wanted))
        elif variant == 6:
            add_comment(
                form_target,
                "### Source handoff\n\n"
                + source_table([i for i in features if i != form_target], ordinary=True),
            )
        else:
            for target in help_wanted:
                add_comment(
                    target,
                    "### Source handoff\n\n" + source_table([i for i in features if i != target]),
                )
            for target in help_wanted:
                reread(target)
            add_comment(anchor, "### Refreshed handoff summary\n\n" + count_table(help_wanted))
    elif family == "frozen_predicate_mixed_writes":
        if variant in {5, 6}:
            for issue in help_wanted:
                add_comment(
                    issue,
                    f"Content review {issue['reference']}: {issue['title']}; "
                    f"initial ordinary comments: {len(issue['notes'])}.",
                )
            if variant == 6:
                for issue in help_wanted:
                    reread(issue)
            table = count_table(help_wanted) if variant == 6 else source_table(help_wanted)
            save_description(anchor, anchor["markdown"] + "\n\n### Content evidence\n\n" + table)
        else:
            for issue in features:
                body, selected = issue["markdown"], {}
                if variant in {0, 1}:
                    if "Netlify" in body:
                        selected["note"] = True
                    elif (variant == 0 and "https://mastodon.social/" in body) or (
                        variant == 1 and "theme-h" in body
                    ):
                        selected["matrix"] = True
                    elif "WAI" in body and issue["notes"]:
                        selected = {"note": True, "matrix": True}
                elif variant == 2:
                    if issue["notes"]:
                        selected["matrix"] = True
                    elif "Netlify" in body:
                        selected["note"] = True
                    elif "https://mastodon.social/" in body:
                        selected = {"note": True, "choices": [True, False]}
                elif variant == 3:
                    if "help wanted" in issue["labels"]:
                        selected["note"] = True
                    elif "WAI" in body and issue["notes"]:
                        selected = {"note": True, "matrix": True}
                    elif "theme-h" in body:
                        selected["matrix"] = True
                elif variant == 4 and issue in zero:
                    if "Netlify" in body:
                        selected["matrix"] = True
                    elif "https://mastodon.social/" in body:
                        selected["note"] = True
                    else:
                        selected = {"note": True, "choices": [True, False]}
                elif variant == 7:
                    if "Netlify" in body:
                        selected = {"note": True, "choices": [False, True]}
                    elif "https://mastodon.social/" in body:
                        selected = {"note": True, "choices": [True, False]}
                    elif "WAI" in body and issue["notes"]:
                        selected = {"matrix": True, "choices": [True, True]}
                if "choices" in selected:
                    body = change_responsibilities(body, selected["choices"])
                if selected.get("matrix"):
                    body += "\n\n### Content evidence\n\n" + source_table(
                        [i for i in features if i != issue]
                    )
                if body != issue["markdown"]:
                    save_description(issue, body)
                if selected.get("note"):
                    add_comment(
                        issue,
                        f"Content review {issue['reference']}: {issue['title']}; "
                        f"initial ordinary comments: {len(issue['notes'])}.",
                    )
    elif family == "same_new_comment_revision":
        targets = (
            [anchor]
            if variant in {0, 4, 7}
            else [form_target]
            if variant in {1, 3}
            else help_wanted
            if variant == 2
            else [help_wanted[0]]
            if variant == 5
            else [anchor, form_target]
        )
        for target in targets:
            initial_sources = (
                [anchor]
                if variant == 3
                else [zero[0]]
                if variant in {4, 7}
                else [help_wanted[1] if target == help_wanted[0] else help_wanted[0]]
                if variant == 2
                else [help_wanted[1]]
                if variant == 5
                else [help_wanted[0]]
            )
            draft = source_table(initial_sources, paragraph=variant != 3)
            add_comment(target, "### Draft handoff\n\n" + draft)
        later_sources = (
            [anchor]
            if variant in {2, 3}
            else zero[1:]
            if variant in {4, 7}
            else [help_wanted[1], anchor]
            if variant == 5
            else [help_wanted[1]]
        )
        for index, source in enumerate(later_sources):
            reread(source)
            if variant == 7 and index == 0:
                revise_comment(anchor, "### Intermediate handoff\n\n" + source_table(zero[:2]))
        for target in targets:
            sources = (
                [anchor]
                if variant == 3
                else zero
                if variant in {4, 7}
                else [help_wanted[1] if target == help_wanted[0] else help_wanted[0], anchor]
                if variant == 2
                else [help_wanted[1], anchor]
                if variant == 5
                else help_wanted
            )
            table = source_table(sources, ordinary=variant == 3, paragraph=variant != 3)
            revise_comment(target, "### Final handoff\n\n" + table)
        if variant == 5:
            reread(help_wanted[0])
            add_comment(
                form_target,
                "### Handoff completion\n\n"
                + f"Destination: {help_wanted[0]['reference']}\n\n"
                + actions[help_wanted[0]["reference"]]["edit_note"],
            )
    elif family == "complete_Markdown_preservation_matrix":
        if variant in {0, 1, 2, 5}:
            targets = (
                [anchor]
                if variant == 0
                else help_wanted
                if variant == 1
                else [form_target]
                if variant == 2
                else [
                    i
                    for i in zero
                    if any(
                        text in feature_paragraph(i["markdown"])
                        for text in ["Netlify", "https://mastodon.social/"]
                    )
                ]
            )
            for target in targets:
                header = "Original discussion matrix" if variant == 2 else "Preservation matrix"
                save_description(
                    target,
                    target["markdown"]
                    + "\n\n### "
                    + header
                    + "\n\n"
                    + source_table([i for i in features if i != target], ordinary=variant == 2),
                )
        elif variant == 3:
            section = "### Preservation matrix\n\n" + source_table(help_wanted)
            save_description(anchor, anchor["markdown"] + "\n\n" + section)
            add_comment(
                form_target,
                "### Matrix handoff\n\n" + f"Destination: {anchor['reference']}\n\n" + section,
            )
        elif variant == 4:
            rows = []
            for index, target in enumerate(help_wanted):
                original = [
                    bool(
                        re.search(r"(?m)^- \[[xX]\] " + re.escape(label) + r"$", target["markdown"])
                    )
                    for label in [
                        "I would like to work on this feature",
                        "I would like the site maintainers to work on this feature",
                    ]
                ]
                final = [index == 0, index != 0]
                save_description(target, change_responsibilities(target["markdown"], final))
                rows.append(
                    [
                        target["reference"],
                        target["title"],
                        *["checked" if value else "unchecked" for value in original + final],
                    ]
                )
            table = markdown_table(
                [
                    "Reference",
                    "Title",
                    "Original contributor",
                    "Original maintainer",
                    "Final contributor",
                    "Final maintainer",
                ],
                rows,
            )
            save_description(
                anchor, anchor["markdown"] + "\n\n### Responsibility matrix\n\n" + table
            )
        elif variant == 6:
            for target in help_wanted:
                add_comment(
                    target,
                    f"Content review {target['reference']}: {target['title']}; "
                    f"initial ordinary comments: {len(target['notes'])}.",
                )
            for target in help_wanted:
                reread(target)
            save_description(
                form_target,
                form_target["markdown"]
                + "\n\n### Refreshed count matrix\n\n"
                + count_table([i for i in features if i != form_target]),
            )
        else:
            for target in help_wanted:
                save_description(
                    target,
                    target["markdown"]
                    + "\n\n### Preservation matrix\n\n"
                    + source_table([i for i in features if i != target]),
                )
            for target in help_wanted:
                reread(target)
            rows = [
                [
                    i["reference"],
                    i["title"],
                    feature_paragraph(i["markdown"]),
                    feature_paragraph(actions[i["reference"]]["description"]),
                    len(i["notes"]),
                    len(i["notes"]),
                ]
                for i in help_wanted
            ]
            table = markdown_table(
                [
                    "Reference",
                    "Title",
                    "Original first feature paragraph",
                    "Final first feature paragraph",
                    "Initial ordinary comments",
                    "Refreshed ordinary comments",
                ],
                rows,
            )
            save_description(
                anchor, anchor["markdown"] + "\n\n### Original/final matrix\n\n" + table
            )
    else:
        raise ValueError("case has no authorized hard48 structure")
    return plan


def capture(args: argparse.Namespace) -> None:
    """Read all initial evidence before writes; preserve raw partial capture on failure."""
    config = json.loads(args.task_config.read_text())
    case = config["case"]
    if case["generated_task_id"] != args.generated_task_id:
        raise ValueError("one author call must bind exactly one frozen task config")
    baseline = json.loads(args.baseline_receipt.read_text())
    origin = args.site_url.rstrip("/")
    site = urlsplit(origin)
    if (
        not baseline["fresh_from_image"]
        or not baseline["owned_run"]
        or baseline["site_url"].rstrip("/") != origin
    ):
        raise ValueError("capture requires a new coordinator-owned fixed-image baseline")
    os.umask(0o077)
    output = args.batch_root / "author" / args.generated_task_id / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (output / "public_query.txt").write_text(case["intent"] + "\n")
    state: dict[str, Any] = {
        "write_phase": None,
        "initial_all_read_completed_at": None,
        "issues": [],
        "projects": [],
        "business_responses": [],
        "discussion_responses": [],
        "blocked_requests": [],
        "case": case,
        "status": "capture_started_no_GT",
    }

    def restrict(route):
        request = route.request
        url = urlsplit(request.url)
        same = (url.scheme, url.netloc) == (site.scheme, site.netloc)
        read = request.method in {"GET", "HEAD"}
        if same and request.method == "POST" and url.path == "/api/graphql":
            payload = json.loads(request.post_data)
            items = payload if isinstance(payload, list) else [payload]
            read = bool(items) and all(
                isinstance(item, dict)
                and isinstance(item.get("query"), str)
                and hashlib.sha256(item["query"].encode()).hexdigest()
                in config["observed_graphql_read_documents"]
                for item in items
            )
        if same and (read or (state["write_phase"] is not None and url.path != "/api/graphql")):
            route.continue_()
        else:
            state["blocked_requests"].append(
                {"method": request.method, "path": url.path, "phase": state["write_phase"]}
            )
            route.abort()

    def record_response(response):
        request = response.request
        url = urlsplit(response.url)
        if url.netloc != site.netloc or url.path == "/api/graphql":
            return
        kind = (
            "discussion_responses"
            if request.method == "GET" and url.path.endswith("/discussions.json")
            else "business_responses"
            if request.method not in {"GET", "HEAD"}
            else None
        )
        if kind is not None:
            state[kind].append(
                {
                    "url": response.url,
                    "path": url.path,
                    "method": request.method,
                    "status": response.status,
                    "request_body": request.post_data,
                    "response_body": "" if response.status == 302 else response.text(),
                    "phase": state["write_phase"],
                    "observed_at": datetime.now(UTC).isoformat(),
                }
            )

    from playwright.sync_api import sync_playwright

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
        context.on("response", record_response)
        context.set_default_timeout(args.timeout_ms)
        context.tracing.start(screenshots=True, snapshots=True, sources=False)
        page = context.new_page()
        counter = 0

        def visit(url: str, name: str) -> dict[str, Any]:
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            return _record_editor(page, output, name)

        def read_issue(target: dict[str, Any], name: str) -> dict[str, Any]:
            before = visit(target["url"], name)
            page.locator("li.note-skeleton:visible").first.wait_for(state="hidden")
            note_rows = page.locator("li.note-comment:not(.system-note):visible").evaluate_all(
                "rows => rows.map(e => ({id:e.id,text:e.querySelector('.note-text.md').innerText}))"
            )
            page.get_by_role("button", name="Edit title and description", exact=True).click()
            editor = _record_editor(page, output, name + "_editor")
            markdown = page.get_by_label("Description", exact=True).input_value()
            page.get_by_role("button", name="Cancel", exact=True).click()
            cancelled = _record_editor(page, output, name + "_cancelled")
            if before["issue_state"] != cancelled["issue_state"]:
                raise ValueError("opening and cancelling an initial editor changed business state")
            return {
                **target,
                "markdown": markdown,
                "notes": note_rows,
                "before": before,
                "editor": editor,
            }

        def save(button, phase: str) -> dict[str, Any]:
            nonlocal counter
            counter += 1
            state["write_phase"] = phase
            try:
                with page.expect_response(
                    lambda response: (
                        response.request.method in {"POST", "PUT", "PATCH"}
                        and urlsplit(response.url).netloc == site.netloc
                        and urlsplit(response.url).path != "/api/graphql"
                    )
                ) as pending:
                    button.click()
                response = pending.value
                body = "" if phase == "create_audit_issue" else response.text()
                result = {
                    "url": response.url,
                    "method": response.request.method,
                    "request_body": response.request.post_data,
                    "response_body": body,
                    "status": response.status,
                    "phase": phase,
                }
                success = 200 <= response.status < 300
                if phase == "create_audit_issue":
                    success = response.status == 302
                if not success:
                    raise ValueError("actual GUI Save did not return a successful native response")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                _record_editor(page, output, f"write_{counter:02d}")
                return result
            finally:
                state["write_phase"] = None

        try:
            initial = visit(origin + "/", "initial")
            assigned = config["assigned_issues_link"].replace("__GITLAB__", origin)
            if not any(link["url"] == assigned for link in initial["links"]):
                raise ValueError("initial dashboard lacks the actual Assigned link")
            if case["family"] == "Assigned_and_project_metadata_aggregation":
                for index, configured in enumerate(config["project_links"]):
                    url = configured["url"].replace("__GITLAB__", origin)
                    if not any(link["url"] == url for link in initial["links"]):
                        raise ValueError(
                            "one configured project lacks a fresh visible dashboard link"
                        )
                    observation = visit(url, f"project_{index:02d}")
                    state["projects"].append(
                        {**_project_fields(observation), "url": url, "observation": observation}
                    )
            dashboard = visit(assigned, "assigned_dashboard")
            open_tabs = [
                link for link in dashboard["links"] if re.fullmatch(r"Open\s+\d+", link["text"])
            ]
            if len(open_tabs) != 1:
                raise ValueError("Assigned view lacks a unique visible Open tab")
            listing = visit(open_tabs[0]["url"], "assigned_issues")
            advertised = int(open_tabs[0]["text"].split()[-1])
            if len(listing["issues"]) != advertised:
                raise ValueError(
                    "Assigned Open scope is paginated or incomplete; do not derive partial GT"
                )
            state["assigned_scope"] = listing
            all_targets = []
            for row in listing["issues"]:
                project = urlsplit(row["url"]).path.split("/-/issues/")[0].strip("/")
                iid = int(row["url"].rsplit("/", 1)[1])
                all_targets.append(
                    {
                        **row,
                        "project_path": project,
                        "iid": iid,
                        "reference": project + "#" + str(iid),
                    }
                )
            targets = (
                all_targets
                if case["family"] == "Assigned_and_project_metadata_aggregation"
                else [
                    row
                    for row in all_targets
                    if row["project_path"] == config["feature_project"]
                    and "feature" in row["labels"]
                ]
            )
            for index, target in enumerate(targets):
                state["issues"].append(read_issue(target, f"initial_issue_{index:02d}"))
            state["initial_all_read_completed_at"] = datetime.now(UTC).isoformat()
            plan = plan_actions(case, state["issues"], state["projects"])
            state["plan"] = plan
            by_reference = {issue["reference"]: issue for issue in state["issues"]}
            state["performed_actions"] = {}
            created_notes = {}
            for step_index, step in enumerate(plan["steps"]):
                reference, kind = step["reference"], step["kind"]
                issue = by_reference[reference]
                if kind == "reread":
                    later = read_issue(issue, f"reread_{step_index:02d}")
                    if len(later["notes"]) != step["expected_ordinary_count"]:
                        raise ValueError(
                            "actual refreshed ordinary count differs from the ordered plan"
                        )
                    expected_markdown = (
                        plan["actions"].get(reference, {}).get("description", issue["markdown"])
                    )
                    if later["markdown"] != expected_markdown:
                        raise ValueError("actual refreshed complete Markdown differs from the plan")
                    state.setdefault("ordered_rereads", []).append(
                        {"step_index": step_index, "actual": later}
                    )
                    continue
                visit(issue["url"], f"step_{step_index:02d}_before_write")
                performed = state["performed_actions"].setdefault(reference, [])
                if kind == "create_note":
                    page.get_by_test_id("comment-field").fill(step["body"])
                    receipt = save(
                        page.get_by_role("button", name="Comment", exact=True),
                        "create_note:" + reference,
                    )
                    created = json.loads(receipt["response_body"])
                    if created["note"] != step["body"] or created["system"] or created["internal"]:
                        raise ValueError(
                            "actual new ordinary comment differs from the planned full body"
                        )
                    if reference in created_notes:
                        raise ValueError(
                            "one destination may receive only one new ordinary comment"
                        )
                    created_notes[reference] = created
                    receipt["note_id"] = created["id"]
                elif kind == "edit_own_note":
                    created = created_notes[reference]
                    note = page.locator(
                        "li.note-comment#note_" + str(created["id"]) + ":not(.system-note):visible"
                    )
                    if note.count() != 1:
                        raise ValueError(
                            "new note identity does not uniquely bind one ordinary DOM note"
                        )
                    note.get_by_role("button", name="Edit comment", exact=True).click()
                    _record_editor(page, output, f"own_note_editor_{step_index:02d}")
                    note.locator("textarea:visible").fill(step["body"])
                    receipt = save(
                        note.get_by_role("button", name="Save comment", exact=True),
                        "edit_own_note:" + reference + ":" + str(step_index),
                    )
                    actual = json.loads(receipt["response_body"])
                    if (
                        actual["id"] != created["id"]
                        or actual["note"] != step["body"]
                        or actual["system"]
                        or actual["internal"]
                    ):
                        raise ValueError(
                            "actual revision did not persist the same new ordinary "
                            "note identity/body"
                        )
                    receipt["note_id"] = actual["id"]
                elif kind == "save_description":
                    page.get_by_role(
                        "button", name="Edit title and description", exact=True
                    ).click()
                    page.get_by_label("Description", exact=True).fill(step["body"])
                    receipt = save(
                        page.get_by_role("button", name="Save changes", exact=True),
                        "save_description:" + reference,
                    )
                    saved = json.loads(receipt["response_body"])
                    if saved["description"] != step["body"] or saved["title"] != issue["title"]:
                        raise ValueError(
                            "actual native full Description/title differs from the plan"
                        )
                else:
                    raise ValueError("unrecognized ordered author step")
                receipt["step_index"] = step_index
                performed.append(receipt)
            if "create" in plan:
                create = plan["create"]
                observation = visit(create["project"]["url"], "selected_project")
                links = [
                    link
                    for link in observation["links"]
                    if re.fullmatch(r"Issues(?:\s+\d+)?", link["text"])
                    and urlsplit(link["url"]).path.endswith("/-/issues")
                ]
                if len({link["url"] for link in links}) != 1:
                    raise ValueError("selected project lacks its actual Issues link")
                listing = visit(links[0]["url"], "selected_project_issues")
                links = [link for link in listing["links"] if link["text"] == "New issue"]
                if len({link["url"] for link in links}) != 1:
                    raise ValueError("selected project lacks its actual New issue link")
                visit(links[0]["url"], "new_issue_editor")
                page.get_by_label("Title (required)", exact=True).fill(create["title"])
                page.get_by_label("Description", exact=True).fill(create["description"])
                receipt = save(
                    page.get_by_role("button", name="Create issue", exact=True),
                    "create_audit_issue",
                )
                state["created_issue"] = {
                    "url": page.url,
                    "receipt": receipt,
                    "after": _record_editor(page, output, "created_issue"),
                }
            # Reread selected and excluded source objects only after every write.
            state["final_issues"] = []
            for index, issue in enumerate(state["issues"]):
                final = read_issue(issue, f"final_issue_{index:02d}")
                action = plan["actions"].get(issue["reference"], {})
                if final["markdown"] != action.get("description", issue["markdown"]):
                    raise ValueError(
                        "refreshed complete Markdown differs from its authorized persistent value"
                    )
                expected_count = len(issue["notes"]) + int("note" in action)
                if len(final["notes"]) != expected_count:
                    raise ValueError(
                        "final ordinary-note count differs from the single authorized create"
                    )
                original = {note["id"]: note["text"] for note in issue["notes"]}
                final_notes = {note["id"]: note["text"] for note in final["notes"]}
                if any(final_notes.get(key) != value for key, value in original.items()):
                    raise ValueError("an original ordinary discussion note changed or disappeared")
                old = issue["before"]["issue_state"]
                new = final["before"]["issue_state"]
                if any(
                    old[key] != new[key] for key in ["title", "labels", "assignee_text", "open"]
                ):
                    raise ValueError("a preserved original field changed")
                if "note" in action:
                    responses = [
                        response
                        for response in state["discussion_responses"]
                        if response["path"] == urlsplit(issue["url"]).path + "/discussions.json"
                    ]
                    if not responses:
                        raise ValueError(
                            "final ordinary notes lack a native persistent discussion response"
                        )
                    groups = json.loads(responses[-1]["response_body"])
                    notes = [
                        note
                        for group in groups
                        for note in group["notes"]
                        if not note["system"] and not note["internal"]
                    ]
                    created_id = created_notes[issue["reference"]]["id"]
                    matches = [note for note in notes if str(note["id"]) == str(created_id)]
                    if len(matches) != 1 or matches[0]["note"] != action.get(
                        "edit_note", action["note"]
                    ):
                        raise ValueError(
                            "late native GET does not preserve the authorized body "
                            "on its same new note ID"
                        )
                    final["persisted_new_note"] = matches[0]
                state["final_issues"].append(final)
            if "create" in plan:
                refreshed = visit(state["created_issue"]["url"], "created_issue_refreshed")
                if refreshed["issue_state"]["title"] != plan["create"]["title"]:
                    raise ValueError(
                        "refreshed new Issue title differs from the requested audit title"
                    )
                page.get_by_role("button", name="Edit title and description", exact=True).click()
                state["created_issue"]["editor_after"] = _record_editor(
                    page, output, "created_issue_editor_after"
                )
                if (
                    page.get_by_label("Description", exact=True).input_value()
                    != plan["create"]["description"]
                ):
                    raise ValueError(
                        "refreshed new Issue full Markdown differs from the joined audit"
                    )
                page.get_by_role("button", name="Cancel", exact=True).click()
                state["terminal"] = visit(state["created_issue"]["url"], "after")
            else:
                last = visit(state["issues"][-1]["url"], "terminal_project_link_source")
                links = {link["url"] for link in last["links"] if link["text"] == "List"}
                if len(links) != 1:
                    raise ValueError("feature Issue lacks its actual project List link")
                observation = visit(links.pop(), "project_issues")
                all_tabs = [
                    link
                    for link in observation["links"]
                    if re.fullmatch(r"All\s+\d+", link["text"])
                ]
                if len(all_tabs) != 1:
                    raise ValueError("project List lacks its unique visible All tab")
                all_tab = page.locator("a:visible").filter(
                    has_text=re.compile(r"^\s*All\s+\d+\s*$")
                )
                all_tab.click()
                page.wait_for_load_state("networkidle")
                state["all_tab_activation"] = all_tab.evaluate(
                    "e => ({text:e.innerText,aria_selected:e.getAttribute('aria-selected')})"
                )
                if state["all_tab_activation"]["aria_selected"] != "true":
                    raise ValueError("the actual final All tab is not active")
                state["terminal"] = _record_editor(page, output, "after")
            state["status"] = (
                "actual_GUI_capture_persisted_pending_native_GT_compile_and_root_review"
            )
        finally:
            try:
                context.tracing.stop(path=str(output / "trace.zip"))
            finally:
                context.close()
                browser.close()
                (output / "observation.json").write_text(
                    json.dumps(state, ensure_ascii=False, indent=2) + "\n"
                )
    print(
        json.dumps(
            {
                "status": state["status"],
                "generated_task_id": args.generated_task_id,
                "capture": str(output),
                "canonical_candidate": False,
                "model_calls": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    capture(_parse_args())
