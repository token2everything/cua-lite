#!/usr/bin/env python3
"""Capture twelve prospective GitLab cases on separate fixed-image baselines.

Run under the existing author runtime with the frozen handoff-v5 observation
owner. This entry performs real GUI reference actions and records complete HAR,
Playwright trace, original Markdown, ordinary notes and refreshed persistence.
It emits an actual capture packet, not a canonical candidate or teacher sample.
The coordinator must build and validate private GT from the resulting actual
native events before admitting a case or calling either model.
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


def feature_paragraph(markdown: str) -> str:
    """Read the complete first paragraph of the original Feature suggestion."""
    section = markdown.split("### Feature suggestion\n\n", 1)[1]
    return section.split("\n\n", 1)[0]


def evidence_table(issues: list[dict[str, Any]]) -> str:
    """Keep rows in numeric order and preserve source paragraph Markdown."""
    rows = [
        "| Reference | Title | Initial ordinary comments | First feature paragraph |",
        "| --- | --- | --- | --- |",
    ]
    for issue in sorted(issues, key=lambda item: item["iid"]):
        values = [
            issue["reference"],
            issue["title"],
            str(len(issue["notes"])),
            feature_paragraph(issue["markdown"]),
        ]
        rows.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(rows)


def change_responsibilities(markdown: str, choices: list[bool]) -> str:
    """Change exactly two existing markers, retaining conduct and all other bytes."""
    labels = [
        "I would like to work on this feature",
        "I would like the site maintainers to work on this feature",
    ]
    result = markdown
    for label, selected in zip(labels, choices, strict=True):
        pattern = r"(?m)^(- \[)[ xX](\] " + re.escape(label) + r")$"
        result, matches = re.subn(
            pattern, lambda match: match[1] + ("x" if selected else " ") + match[2], result
        )
        if matches != 1:
            raise ValueError("each original responsibility must have one complete Markdown line")
    return result


def plan_actions(
    case: dict[str, Any], issues: list[dict[str, Any]], projects: list[dict[str, Any]]
) -> dict[str, Any]:
    """Derive authorized actions solely from frozen initial GUI observations."""
    family, variant = case["family"], case["variant"]
    features = sorted(
        [item for item in issues if "feature" in item["labels"]], key=lambda item: item["iid"]
    )
    actions: dict[str, Any] = {}
    result: dict[str, Any] = {"actions": actions, "read_after_first_comment": []}
    if family == "project_and_issue_scope_joined_audit_creation":
        rows = [
            "| Project | Branches | Tags | Assigned open issues | Initial ordinary comments |",
            "| --- | --- | --- | --- | --- |",
        ]
        for project in sorted(projects, key=lambda item: item["project_path"]):
            members = [
                item
                for item in issues
                if item["project_path"] == project["project_path"].strip("/")
            ]
            rows.append(
                f"| {project['project_path'].strip('/')} | {project['branches']} | "
                f"{project['tags']} | {len(members)} | "
                f"{sum(len(item['notes']) for item in members)} |"
            )
        ordered = sorted(projects, key=lambda item: item["tags"])
        target = ordered[-1] if variant == 0 else ordered[0]
        if sum(project["tags"] == target["tags"] for project in projects) != 1:
            raise ValueError("project tag extremum is not unique in fresh GUI facts")
        result["create"] = {
            "project": target,
            "title": case["new_issue_title"],
            "description": "### Assigned issue audit\n\n" + "\n".join(rows),
        }
        return result
    if len(features) != 4:
        raise ValueError("the fresh assigned Open feature cohort must contain four actual rows")
    zero = [item for item in features if item["author"] == "Byte Blaze" and not item["notes"]]
    discussed = [item for item in features if item["notes"]]
    if len(discussed) != 1:
        raise ValueError("the initial discussion predicate must select one actual Issue")
    help_wanted = [item for item in features if "help wanted" in item["labels"]]
    form = [item for item in zero if "help wanted" not in item["labels"]]
    if len(help_wanted) != 2 or len(form) != 1:
        raise ValueError("the fresh label/author predicate differs from its pinned source evidence")
    if family == "cross_issue_evidence_transfer":
        target = discussed[0] if variant == 0 else form[0]
        actions[target["reference"]] = {
            "note": "### Source handoff\n\n" + evidence_table(help_wanted)
        }
    elif family == "discussion_derived_selective_cohort":
        selected = help_wanted if variant == 0 else zero
        cohort = ", ".join(item["reference"] for item in selected)
        for index, item in enumerate(selected):
            choices = (
                [index == 0, index != 0]
                if variant == 0
                else ["help wanted" in item["labels"], "help wanted" not in item["labels"]]
            )
            actions[item["reference"]] = {
                "description": change_responsibilities(item["markdown"], choices),
                "note": (
                    f"Review {item['reference']}: {item['title']}; "
                    f"initial ordinary comments: {len(item['notes'])}; eligible cohort: {cohort}; "
                    f"contributor: {'checked' if choices[0] else 'unchecked'}; "
                    f"maintainer: {'checked' if choices[1] else 'unchecked'}."
                ),
            }
    elif family == "cross_issue_preservation_matrix_in_Markdown":
        target = discussed[0] if variant == 0 else form[0]
        sources = [item for item in features if item["reference"] != target["reference"]]
        actions[target["reference"]] = {
            "description": target["markdown"]
            + "\n\n### Evidence matrix\n\n"
            + evidence_table(sources)
        }
    elif family == "own_new_comment_revision_with_stable_identity":
        target = discussed[0] if variant == 0 else form[0]
        sources = help_wanted
        first = "### Draft handoff\n\n" + evidence_table(sources[:1])
        final = "### Final handoff\n\n" + evidence_table(sources)
        actions[target["reference"]] = {"note": first, "edit_note": final}
        result["read_after_first_comment"] = [sources[-1]["reference"]]
    elif family == "content_predicate_mixed_action_cohort":
        for item in features:
            body = item["markdown"]
            note = (
                f"Content audit {item['reference']}: {item['title']}; "
                f"initial ordinary comments: {len(item['notes'])}."
            )
            suffix = "\n\n### Content audit\n\n" + evidence_table(
                [source for source in features if source["reference"] != item["reference"]]
            )
            action = {}
            if "Netlify" in body:
                action["note"] = note
            elif (variant == 0 and "https://mastodon.social/" in body) or (
                variant == 1 and "theme-h" in body
            ):
                action["description"] = body + suffix
            elif "WAI" in body and item["notes"]:
                action = {"note": note, "description": body + suffix}
            if action:
                actions[item["reference"]] = action
    else:
        raise ValueError("case has no authorized hard12 operation structure")
    return result


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
            if case["family"] == "project_and_issue_scope_joined_audit_creation":
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
                if case["family"] == "project_and_issue_scope_joined_audit_creation"
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
            for reference, action in plan["actions"].items():
                issue = by_reference[reference]
                visit(issue["url"], f"target_{issue['iid']}_before_write")
                performed = state["performed_actions"][reference] = []
                if "note" in action:
                    page.get_by_test_id("comment-field").fill(action["note"])
                    receipt = save(
                        page.get_by_role("button", name="Comment", exact=True),
                        "create_note:" + reference,
                    )
                    created = json.loads(receipt["response_body"])
                    if (
                        created["note"] != action["note"]
                        or created["system"]
                        or created["internal"]
                    ):
                        raise ValueError(
                            "created ordinary public note differs from the authorized body"
                        )
                    receipt["note_id"] = created["id"]
                    performed.append(receipt)
                    if "edit_note" in action:
                        for source in plan["read_after_first_comment"]:
                            later = read_issue(
                                by_reference[source],
                                "second_read_" + str(by_reference[source]["iid"]),
                            )
                            if (
                                later["markdown"] != by_reference[source]["markdown"]
                                or later["notes"] != by_reference[source]["notes"]
                            ):
                                raise ValueError(
                                    "the required second source read differs from frozen "
                                    "source state"
                                )
                        visit(issue["url"], "own_note_before_edit")
                        note = page.locator(
                            "li.note-comment#note_"
                            + str(created["id"])
                            + ":not(.system-note):visible"
                        )
                        if note.count() != 1:
                            raise ValueError(
                                "new note native DOM identity does not uniquely bind the created ID"
                            )
                        note.get_by_role("button", name="Edit comment", exact=True).click()
                        _record_editor(page, output, "own_note_editor")
                        note.locator("textarea:visible").fill(action["edit_note"])
                        edited = save(
                            note.locator("button[type=submit]:visible"),
                            "edit_own_note:" + reference,
                        )
                        actual = json.loads(edited["response_body"])
                        if actual["id"] != created["id"] or actual["note"] != action["edit_note"]:
                            raise ValueError(
                                "actual editor did not update the same newly-created note"
                            )
                        edited["note_id"] = actual["id"]
                        performed.append(edited)
                if "description" in action:
                    page.get_by_role(
                        "button", name="Edit title and description", exact=True
                    ).click()
                    page.get_by_label("Description", exact=True).fill(action["description"])
                    receipt = save(
                        page.get_by_role("button", name="Save changes", exact=True),
                        "save_description:" + reference,
                    )
                    saved = json.loads(receipt["response_body"])
                    if (
                        saved["description"] != action["description"]
                        or saved["title"] != issue["title"]
                    ):
                        raise ValueError(
                            "actual full Description Save did not preserve its title "
                            "and expected Markdown"
                        )
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
                    created_id = state["performed_actions"][issue["reference"]][0]["note_id"]
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
