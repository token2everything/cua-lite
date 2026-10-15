#!/usr/bin/env python3
"""Ground new GitLab retrieval, navigation and focused edits from fresh visible UI.

The coordinator supplies an authentication-only fixed-image baseline. Readiness
captures rendered lists, their actual pagination and detail forms without GT.
Formal mode recomputes each task from a separate fresh baseline and emits the
standard author draft, HAR, Playwright trace and field-level observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

from author_gitlab import _record


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["explore-readiness", "formal"], default="formal")
    parser.add_argument("--generated-task-id")
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--attempt", default="grounding-01")
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    args = parser.parse_args(argv)
    origin = urlsplit(args.site_url)
    if (origin.scheme not in {"http", "https"} or not origin.netloc or origin.username
            or origin.password or origin.query or origin.fragment or origin.path not in {"", "/"}):
        parser.error("site URL must be an explicit HTTP(S) origin without credentials")
    if not re.fullmatch(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", args.attempt):
        parser.error("attempt must be a simple identifier")
    if args.timeout_ms <= 0 or args.settle_seconds < 0:
        parser.error("timeout must be positive and settle seconds nonnegative")
    if args.mode == "formal" and not args.generated_task_id:
        parser.error("formal mode requires a generated task ID")
    return args


def _record_extension(page: Any, output: Path, name: str) -> dict[str, Any]:
    """Add rendered MR, starrer and to-do rows to the existing GitLab owner."""
    observed = _record(page, output, name)
    del observed["controls"]
    del observed["headings"]
    observed.update(page.evaluate(r"""() => {
      const visible = e => { const r=e.getBoundingClientRect();
        return !!(r.width && r.height) && getComputedStyle(e).visibility!=='hidden'; };
      const text = e => e ? e.innerText.trim() : null;
      const links = root => Array.from(root.querySelectorAll('a[href]')).filter(visible)
        .map(a => ({text:text(a),url:a.href,title:a.title,rel:a.rel}));
      const issuables = selector => Array.from(document.querySelectorAll(selector))
        .filter(visible).map(row => {
          const targets = Array.from(row.querySelectorAll('a[href]')).filter(visible);
          const title = targets.find(a => /\/-\/(?:issues|merge_requests)\/\d+$/.test(
            new URL(a.href).pathname) && !new URL(a.href).hash && a.innerText.trim());
          const reference=row.querySelector('.issuable-reference');
          const author=row.querySelector('.author');
          const comments=row.querySelector('[data-testid="issuable-comments"]');
          const approvalBadges=Array.from(row.querySelectorAll(
            'li:has(> svg[data-testid="approval-solid-icon"])')).filter(visible);
          if (approvalBadges.length > 1) {
            throw new Error('rendered merge request has ambiguous approval badges');
          }
          return {title:text(title),url:title?title.href:null,reference:text(reference),
            author:text(author),labels:Array.from(row.querySelectorAll('.gl-label-text'))
              .filter(visible).map(text),comments_text:text(comments),body_text:text(row),
            approval_status:approvalBadges.length ? text(approvalBadges[0]) : null,
            links:links(row)};
        });
      return {
        pagination_links:Array.from(document.querySelectorAll('li.page-item a[href]')).filter(visible)
          .map(a => ({text:text(a),url:a.href,disabled:!!a.closest('li.disabled'),rel:a.rel})),
        merge_requests:issuables('li.merge-request'),
        starrers:Array.from(document.querySelectorAll('div.user-info')).filter(visible)
          .map(row => { const user=row.querySelector('a.user');
            return {name:text(user),username:text(row.querySelector('span.cgray')),
              url:user?user.href:null,body_text:text(row)}; }),
        todos:Array.from(document.querySelectorAll('li.todo')).filter(visible).map(row => {
          const target=row.querySelector('a.todo-target-link');
          return {id:row.id,identity_text:text(row.querySelector('.todo-title')),title:text(row.querySelector('.todo-target-title')),
            reference:text(target),url:target?target.href:null,
            action:text(row.querySelector('.todo-note')),body_text:text(row),links:links(row)};
        }),
        visible_nodes:Array.from(document.querySelectorAll('h1,h2,h3,form,input,select,textarea,button,[role]'))
          .filter(e => e instanceof HTMLElement).filter(visible).map(e=>({tag:e.tagName.toLowerCase(),id:e.id,class_name:e.className,
            role:e.getAttribute('role'),name:e.name,type:e.type,text:text(e),
            aria_label:e.getAttribute('aria-label'),placeholder:e.getAttribute('placeholder'),
            data_testid:e.getAttribute('data-testid'),value:e.type==='password'?undefined:e.value,
            form:e.form?{id:e.form.id,action:e.form.action,method:e.form.method}:null}))
      };
    }"""))
    (output / f"{name}.json").write_text(json.dumps(observed, ensure_ascii=False, indent=2) + "\n")
    (output / f"{name}.html").write_text(page.content())
    return observed


def _read_scope(page: Any, output: Path, url: str, name: str, args: argparse.Namespace,
                row_key: str) -> dict[str, Any]:
    """Cover a real visible paginated window, following only its Next link."""
    pages = []
    seen = set()
    while True:
        if url in seen:
            raise ValueError("visible pagination repeated a page")
        seen.add(url)
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        observation = _record_extension(page, output, f"{name}_{len(pages):02}")
        pages.append(observation)
        next_urls = {link["url"] for link in observation["pagination_links"]
                     if link["text"].strip() == "Next" and not link["disabled"] and link["url"] != observation["url"]
                     and urlsplit(link["url"]).path == urlsplit(observation["url"]).path}
        if not next_urls:
            break
        if len(next_urls) != 1 or len(pages) >= 10:
            raise ValueError("readiness window is ambiguous or exceeds its ten-page bound")
        url = next_urls.pop()
    rows = [dict(row, page_index=index, row_index=row_index)
            for index, observed in enumerate(pages)
            for row_index, row in enumerate(observed[row_key])]
    identity_field = "id" if row_key == "todos" else "url"
    if len({row[identity_field] for row in rows}) != len(rows):
        raise ValueError("the visible pagination repeated an entity")
    return {"pages": pages, "rows": rows, "complete_window": True}


def _collect_scope(page: Any, output: Path, initial: dict, descriptor: dict,
                   name: str, args: argparse.Namespace) -> dict:
    """Resolve the declared scope through links exposed by this fresh UI."""
    site = args.site_url.rstrip("/")
    root_url = descriptor["root_url"].replace("__GITLAB__", site)
    if not any(link["url"] == root_url for link in initial["links"]):
        raise ValueError("declared scope root is absent from the fresh dashboard")
    page.goto(root_url, wait_until="domcontentloaded")
    page.wait_for_timeout(int(args.settle_seconds * 1000))
    root = _record_extension(page, output, name + "_root")
    landing = root
    target = root_url
    if "child_suffix" in descriptor:
        urls = {link["url"] for link in root["links"]
                if urlsplit(link["url"]).path.endswith(descriptor["child_suffix"])}
        if len(urls) != 1:
            raise ValueError("scope child is absent or ambiguous on the observed project")
        target = urls.pop()
        page.goto(target, wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        landing = _record_extension(page, output, name + "_landing")
    if "tab" in descriptor:
        matches = [link for link in landing["links"]
                   if re.fullmatch(re.escape(descriptor["tab"]) + r"\s+\d+", link["text"])]
        if len({link["url"] for link in matches}) != 1:
            raise ValueError("scope tab is absent or ambiguous in the fresh UI")
        target = matches[0]["url"]
        expected_count = int(matches[0]["text"].split()[-1])
    else:
        totals = re.findall(r"(\d+) starrers?:", landing["body_text"])
        if len(totals) != 1:
            raise ValueError("starrer scope lacks its labelled visible total")
        expected_count = int(totals[0])
    scope = _read_scope(page, output, target, name, args, descriptor["row_key"])
    if len(scope["rows"]) != expected_count:
        raise ValueError("the visible scope does not cover its displayed complete count")
    scope.update(root=root, landing=landing, displayed_count=expected_count,
                 state=descriptor.get("tab"), row_key=descriptor["row_key"])
    for row in scope["rows"]:
        if descriptor["row_key"] in {"issues", "merge_requests", "todos"}:
            path = urlsplit(row["url"]).path
            project, object_kind, number = re.fullmatch(r"(.+)/-/(issues|merge_requests)/(\d+)", path).groups()
            row["project_path"] = project.lstrip("/")
            row["number"] = int(number)
            row["reference"] = row["project_path"] + ("#" if object_kind == "issues" else "!") + number
        if descriptor["row_key"] in {"issues", "merge_requests"}:
            row["comment_count"] = int(row["comments_text"])
    return scope


def _filter_rows(rows: list[dict], case: dict) -> list[dict]:
    """Apply the public conjunction and deterministic ranking to visible rows."""
    parameters = case["parameters"]
    matched = []
    for row in rows:
        if "project_path" in parameters and row.get("project_path") != parameters["project_path"]:
            continue
        if "author" in parameters and row.get("author") != parameters["author"]:
            continue
        if "approval_status" in parameters and row["approval_status"] != parameters["approval_status"]:
            continue
        labels = set(row.get("labels", []))
        if not set(parameters.get("required_labels", [])) <= labels:
            continue
        if set(parameters.get("excluded_labels", [])) & labels:
            continue
        if row.get("comment_count", 0) < parameters.get("minimum_comments", 0):
            continue
        if "maximum_comments" in parameters and row.get("comment_count", 0) > parameters["maximum_comments"]:
            continue
        if "title_contains" in parameters and parameters["title_contains"].casefold() not in row["title"].casefold():
            continue
        if "todo_action_contains" in parameters and "action" in row and parameters["todo_action_contains"] not in row["action"]:
            continue
        matched.append(row)
    order = case.get("order", "reference")
    if order == "comments_desc":
        matched.sort(key=lambda row: (-row["comment_count"], row["reference"]))
    elif order == "number_desc":
        matched.sort(key=lambda row: (-row["number"], row["project_path"]))
    elif order == "number_asc":
        matched.sort(key=lambda row: (row["number"], row["project_path"]))
    else:
        matched.sort(key=lambda row: (row.get("project_path", ""), row.get("number", 0), row.get("username", "")))
    return matched


def _build_definition(case: dict, archive: dict) -> tuple[dict, list[dict], list[dict], dict | None]:
    """Compute answers and relationship joins solely from the formal window."""
    parameters = dict(case["parameters"])
    scopes = archive["scopes"]
    family = case["family"]
    source = scopes[case["scopes"][0]]["rows"]
    selected = _filter_rows(source, case)
    derived = []
    target = None
    if family == "issue_state_balance":
        open_rows = scopes[case["scopes"][0]]["rows"]
        closed_rows = scopes[case["scopes"][1]]["rows"]
        projects = sorted({row["project_path"] for row in open_rows + closed_rows})
        derived = [{"project_path": project,
                    "open_count": sum(row["project_path"] == project for row in open_rows),
                    "closed_count": sum(row["project_path"] == project for row in closed_rows)}
                   for project in projects]
        for row in derived:
            row["closed_minus_open"] = row["closed_count"] - row["open_count"]
        derived.sort(key=lambda row: (-row["closed_minus_open"], row["project_path"]))
    elif family in {"issue_filtered_list", "mr_filtered_rank"}:
        derived = [{key: row[key] for key in ["reference", "title", "author", "comment_count"]}
                   for row in selected]
        if "top_count" in parameters:
            derived = derived[:parameters["top_count"]]
    elif family in {"issue_project_aggregate", "issue_author_aggregate", "mr_label_aggregate"}:
        if family == "issue_project_aggregate":
            keys = sorted({row["project_path"] for row in selected})
            groups = [(key, [row for row in selected if row["project_path"] == key]) for key in keys]
            name = "project_path"
        elif family == "issue_author_aggregate":
            keys = sorted({row["author"] for row in selected})
            groups = [(key, [row for row in selected if row["author"] == key]) for key in keys]
            name = "author"
        else:
            groups = [(label, [row for row in selected if label in row["labels"]])
                      for label in parameters["group_labels"]]
            name = "label"
        derived = [{name: key, "item_count": len(rows),
                    "comment_total": sum(row["comment_count"] for row in rows),
                    "zero_comment_count": sum(row["comment_count"] == 0 for row in rows)}
                   for key, rows in groups]
        derived.sort(key=lambda row: (-row["comment_total"], row[name]))
    elif family == "mr_starrer_relationship":
        starrers = {row["name"]: row for row in scopes[case["scopes"][1]]["rows"]}
        selected = [row for row in selected if row["author"] in starrers]
        derived = [{"reference": row["reference"], "title": row["title"], "author": row["author"],
                    "username": starrers[row["author"]]["username"], "comment_count": row["comment_count"]}
                   for row in selected]
    elif family == "assigned_mr_creator_pending_balance":
        request_scope, todo_scope = case["scopes"]
        groups = {}
        for index, row in enumerate(source):
            creator_links = [(link_index, link) for link_index, link in enumerate(row["links"])
                             if link["text"] == row["author"]
                             and re.fullmatch(r"/[^/]+", urlsplit(link["url"]).path)]
            if len({link["url"] for _, link in creator_links}) != 1:
                raise ValueError("assigned request creator lacks one visible profile link")
            link_index, creator = creator_links[0]
            profile_path = urlsplit(creator["url"]).path
            request_source = f"/scopes/{request_scope}/rows/{index}"
            group = groups.setdefault(profile_path, {
                "creator_profile_path": profile_path,
                "creator_username": "@" + profile_path.lstrip("/"),
                "source_windows": {"requests": f"/scopes/{request_scope}",
                                   "pending": f"/scopes/{todo_scope}"},
                "requests": []})
            group["requests"].append({
                "request_url": row["url"], "request_source": request_source,
                "creator_profile_source": request_source + f"/links/{link_index}/url",
                "comment_count": row["comment_count"],
                "comment_badge_source": request_source + "/comments_text",
                "pending_sources": [f"/scopes/{todo_scope}/rows/{todo_index}/url"
                                    for todo_index, todo in enumerate(scopes[todo_scope]["rows"])
                                    if todo["url"] == row["url"]]})
        for group in groups.values():
            requests = group["requests"]
            pending_count = len({row["request_url"] for row in requests if row["pending_sources"]})
            derived.append({"creator_username": group["creator_username"],
                            "assigned_mr_count": len({row["request_url"] for row in requests}),
                            "pending_mr_count": pending_count,
                            "without_pending_mr_count": len({row["request_url"] for row in requests}) - pending_count,
                            "comment_total": sum(row["comment_count"] for row in requests)})
        order = sorted(range(len(derived)), key=lambda index: (
            -derived[index]["without_pending_mr_count"], -derived[index]["comment_total"],
            derived[index]["creator_username"]))
        archive["creator_pending_groups"] = [list(groups.values())[index] for index in order]
        derived = [derived[index] for index in order]
    elif family == "project_starrer_assigned_mr_todo_aggregate":
        request_scope, todo_scope = case["scopes"][:2]
        pending = scopes[todo_scope]["rows"]
        groups = []
        # Each project uses its own complete starrer window. Profile paths
        # identify accounts; matching display names alone never creates a join.
        for project, starrer_scope in zip(parameters["project_paths"], case["scopes"][2:], strict=True):
            starrers = scopes[starrer_scope]
            actual_project = urlsplit(starrers["root"]["url"]).path.strip("/")
            if project != actual_project:
                raise ValueError("public project group differs from its fresh visible project root")
            matches = []
            for index, row in enumerate(source):
                if row["project_path"] != project:
                    continue
                author_links = [(link_index, link) for link_index, link in enumerate(row["links"])
                                if link["text"] == row["author"]
                                and re.fullmatch(r"/[^/]+", urlsplit(link["url"]).path)]
                if len({link["url"] for _, link in author_links}) != 1:
                    raise ValueError("merge request creator lacks one visible profile link")
                link_index, author_link = author_links[0]
                profile_path = urlsplit(author_link["url"]).path
                starrer_sources = [f"/scopes/{starrer_scope}/rows/{starrer_index}/url"
                                   for starrer_index, starrer in enumerate(starrers["rows"])
                                   if urlsplit(starrer["url"]).path == profile_path]
                if not starrer_sources:
                    continue
                request_source = f"/scopes/{request_scope}/rows/{index}"
                matches.append({"request_url": row["url"], "request_source": request_source,
                                "author_profile_path": profile_path,
                                "author_profile_source": request_source + f"/links/{link_index}/url",
                                "starrer_sources": starrer_sources,
                                "comment_count": row["comment_count"],
                                "comment_badge_source": request_source + "/comments_text",
                                "pending_sources": [f"/scopes/{todo_scope}/rows/{todo_index}/url"
                                                    for todo_index, todo in enumerate(pending)
                                                    if todo["url"] == row["url"]]})
            group = {"project_path": actual_project,
                     "project_root_source": f"/scopes/{starrer_scope}/root/url",
                     "source_windows": {"requests": f"/scopes/{request_scope}",
                                        "starrers": f"/scopes/{starrer_scope}",
                                        "pending": f"/scopes/{todo_scope}"},
                     "matched_requests": matches}
            groups.append(group)
            derived.append({"project_path": actual_project,
                            "matched_creator_count": len({row["author_profile_path"] for row in matches}),
                            "matched_mr_count": len({row["request_url"] for row in matches}),
                            "comment_total": sum(row["comment_count"] for row in matches),
                            "pending_mr_count": len({row["request_url"] for row in matches
                                                     if row["pending_sources"]})})
        order = sorted(range(len(derived)), key=lambda index: (
            -derived[index]["pending_mr_count"], -derived[index]["comment_total"],
            derived[index]["project_path"]))
        archive["relationship_groups"] = [groups[index] for index in order]
        derived = [derived[index] for index in order]
    elif family == "starrer_membership":
        membership = {}
        for scope_name in case["scopes"]:
            for row in scopes[scope_name]["rows"]:
                member = membership.setdefault(row["username"], {"name": row["name"], "username": row["username"], "project_count": 0})
                member["project_count"] += 1
        derived = [row for row in membership.values() if row["project_count"] >= parameters["minimum_projects"]]
        derived.sort(key=lambda row: (-row["project_count"], row["username"]))
    elif family == "issue_note_authors":
        detail = archive["details"][0]
        notes = detail["detail_state"]["notes"]
        selected = [detail["source_row"]]
        authors = sorted({note["author"] for note in notes})
        derived = [{"author": author, "ordinary_comment_count": sum(note["author"] == author for note in notes),
                    "creator_comment_count": sum(note["author"] == author and author == selected[0]["author"] for note in notes)}
                   for author in authors]
        derived.sort(key=lambda row: (-row["ordinary_comment_count"], row["author"]))
    elif family == "issue_checklist_states":
        selected = [detail["source_row"] for detail in archive["details"]]
        derived = [{"reference": detail["source_row"]["reference"], "title": detail["source_row"]["title"],
                    "checked_count": sum(item["checked"] for item in detail["detail_state"]["checklist"]),
                    "checklist_count": len(detail["detail_state"]["checklist"])} for detail in archive["details"]]
    elif family == "mr_without_pending_todo":
        pending = {row["url"] for row in scopes[case["scopes"][1]]["rows"]}
        selected = [row for row in selected if row["url"] not in pending]
        derived = [{key: row[key] for key in ["reference", "title", "author", "comment_count"]}
                   for row in selected]
    elif family == "todo_mr_relationship":
        merge_requests = {row["url"]: row for row in scopes[case["scopes"][1]]["rows"]}
        selected = [row for row in selected if row["url"] in merge_requests]
        derived = [{"reference": row["reference"], "title": merge_requests[row["url"]]["title"],
                    "todo_action": row["action"], "author": merge_requests[row["url"]]["author"],
                    "comment_count": merge_requests[row["url"]]["comment_count"]} for row in selected]
    elif case["task_type"] == "MUTATE":
        if family == "project_star_from_participants":
            selected = []
            for request_name, starrer_name in zip(case["scopes"][:2], case["scopes"][2:], strict=True):
                requests = scopes[request_name]["rows"]
                starrers = scopes[starrer_name]
                if len(requests) == 1 and requests[0]["author"] == parameters["mr_author"] and len(starrers["rows"]) == parameters["starrer_count"]:
                    selected.append({"url": starrers["root"]["url"], "title": starrers["root"]["project_heading"],
                                     "starrer_scope": starrer_name})
        elif family == "todo_complete_mr_relation":
            requests = {row["url"] for row in scopes[case["scopes"][1]]["rows"]
                        if row["author"] == parameters["mr_author"] and row["comment_count"] == parameters["related_mr_comments"]}
            selected = [row for row in selected if row["url"] in requests]
        elif family == "mr_description_todo_action":
            pending = {row["url"] for row in scopes[case["scopes"][1]]["rows"]
                       if parameters["todo_action_contains"] in row["action"]}
            selected = [row for row in selected if row["url"] in pending]
        elif family == "mr_title_prefix_unassigned_without_pending":
            request_scope, assigned_scope, todo_scope, starrer_scope = case["scopes"]
            actual_project = urlsplit(scopes[starrer_scope]["root"]["url"]).path.strip("/")
            if actual_project != parameters["project_path"]:
                raise ValueError("mutation project differs from its fresh visible starrer root")
            eligible = []
            excluded = []
            for index, row in enumerate(source):
                request_source = f"/scopes/{request_scope}/rows/{index}"
                proof = {"request_source": request_source, "request_url": row["url"],
                         "number": row["number"], "number_source": request_source + "/number",
                         "assigned_sources": [f"/scopes/{assigned_scope}/rows/{assigned_index}/url"
                                              for assigned_index, assigned in enumerate(scopes[assigned_scope]["rows"])
                                              if assigned["url"] == row["url"]],
                         "pending_sources": [f"/scopes/{todo_scope}/rows/{todo_index}/url"
                                             for todo_index, todo in enumerate(scopes[todo_scope]["rows"])
                                             if todo["url"] == row["url"]]}
                if row["project_path"] != actual_project:
                    excluded.append({**proof, "reason": "outside the public project"})
                    continue
                if proof["assigned_sources"] or proof["pending_sources"]:
                    excluded.append({**proof, "reason": "assigned or pending target"})
                    continue
                creator_links = [(link_index, link) for link_index, link in enumerate(row["links"])
                                 if link["text"] == row["author"]
                                 and re.fullmatch(r"/[^/]+", urlsplit(link["url"]).path)]
                if len({link["url"] for _, link in creator_links}) != 1:
                    raise ValueError("mutation creator lacks one visible profile link")
                link_index, creator = creator_links[0]
                creator_path = urlsplit(creator["url"]).path
                starrer_sources = [f"/scopes/{starrer_scope}/rows/{starrer_index}/url"
                                   for starrer_index, starrer in enumerate(scopes[starrer_scope]["rows"])
                                   if urlsplit(starrer["url"]).path == creator_path]
                proof.update(creator_profile_source=request_source + f"/links/{link_index}/url",
                             creator_profile_path=creator_path, starrer_sources=starrer_sources)
                if not starrer_sources:
                    excluded.append({**proof, "reason": "creator is not a same-project starrer"})
                    continue
                eligible.append(proof)
            eligible.sort(key=lambda item: item["number"])
            archive["mutation_relation"] = {
                "project_root_source": f"/scopes/{starrer_scope}/root/url",
                "source_windows": {"requests": f"/scopes/{request_scope}",
                                   "assigned": f"/scopes/{assigned_scope}",
                                   "pending": f"/scopes/{todo_scope}",
                                   "starrers": f"/scopes/{starrer_scope}"},
                "ranked_eligible_requests": eligible, "excluded_requests": excluded,
                "selection": "smallest project-local merge-request number"}
            selected = [row for row in source if eligible and row["url"] == eligible[0]["request_url"]]
        if len(selected) != 1:
            raise ValueError("formal mutation predicate does not identify one fresh target")
        target = selected[0]
    elif family.startswith("navigate_"):
        selection = case["selection"]
        if family == "navigate_closed_issue_unqueued_mr_creator":
            request_scope, issue_scope, todo_scope, starrer_scope = case["scopes"]
            actual_project = urlsplit(scopes[starrer_scope]["root"]["url"]).path.strip("/")
            if actual_project != parameters["project_path"]:
                raise ValueError("navigation project differs from its fresh visible starrer root")
            creators = {}
            outside_requests = []
            for index, row in enumerate(source):
                request_source = f"/scopes/{request_scope}/rows/{index}"
                if row["project_path"] != actual_project:
                    outside_requests.append(request_source)
                    continue
                creator_links = [(link_index, link) for link_index, link in enumerate(row["links"])
                                 if link["text"] == row["author"]
                                 and re.fullmatch(r"/[^/]+", urlsplit(link["url"]).path)]
                if len({link["url"] for _, link in creator_links}) != 1:
                    raise ValueError("navigation MR creator lacks one visible profile link")
                link_index, creator = creator_links[0]
                profile_path = urlsplit(creator["url"]).path
                group = creators.setdefault(profile_path, {
                    "creator_profile_path": profile_path,
                    "creator_username": "@" + profile_path.lstrip("/"),
                    "requests": [], "closed_issues": [],
                    "starrer_sources": [f"/scopes/{starrer_scope}/rows/{starrer_index}/url"
                                        for starrer_index, starrer in enumerate(scopes[starrer_scope]["rows"])
                                        if urlsplit(starrer["url"]).path == profile_path]})
                group["requests"].append({
                    "request_url": row["url"], "request_source": request_source,
                    "creator_profile_source": request_source + f"/links/{link_index}/url",
                    "pending_sources": [f"/scopes/{todo_scope}/rows/{todo_index}/url"
                                        for todo_index, todo in enumerate(scopes[todo_scope]["rows"])
                                        if todo["url"] == row["url"]]})
            excluded_issues = []
            for index, row in enumerate(scopes[issue_scope]["rows"]):
                issue_source = f"/scopes/{issue_scope}/rows/{index}"
                if row["project_path"] != actual_project:
                    excluded_issues.append({"issue_source": issue_source, "reason": "outside the public project"})
                    continue
                creator_links = [(link_index, link) for link_index, link in enumerate(row["links"])
                                 if link["text"] == row["author"]
                                 and re.fullmatch(r"/[^/]+", urlsplit(link["url"]).path)]
                if len({link["url"] for _, link in creator_links}) != 1:
                    raise ValueError("navigation issue creator lacks one visible profile link")
                link_index, creator = creator_links[0]
                profile_path = urlsplit(creator["url"]).path
                proof = {"issue_source": issue_source, "issue_url": row["url"],
                         "creator_profile_source": issue_source + f"/links/{link_index}/url",
                         "creator_profile_path": profile_path,
                         "comment_count": row["comment_count"], "number": row["number"],
                         "comment_badge_source": issue_source + "/comments_text",
                         "number_source": issue_source + "/number"}
                if profile_path not in creators:
                    excluded_issues.append({**proof, "reason": "creator has no same-project open MR"})
                    continue
                creators[profile_path]["closed_issues"].append(proof)
            eligible = []
            excluded = []
            for group in creators.values():
                if not group["starrer_sources"]:
                    excluded.append({**group, "reason": "creator is not a same-project starrer"})
                elif any(row["pending_sources"] for row in group["requests"]):
                    excluded.append({**group, "reason": "at least one same-project open MR has a pending To-Do"})
                elif not group["closed_issues"]:
                    excluded.append({**group, "reason": "creator has no same-project assigned closed issue"})
                else:
                    group["closed_issue_count"] = len(group["closed_issues"])
                    group["closed_issue_comment_total"] = sum(row["comment_count"] for row in group["closed_issues"])
                    group["closed_issues"].sort(key=lambda row: (-row["comment_count"], -row["number"]))
                    eligible.append(group)
            eligible.sort(key=lambda group: (-group["closed_issue_comment_total"],
                                             -group["closed_issue_count"], group["creator_username"]))
            archive["navigation_relation"] = {
                "project_root_source": f"/scopes/{starrer_scope}/root/url",
                "source_windows": {"requests": f"/scopes/{request_scope}",
                                   "closed_issues": f"/scopes/{issue_scope}",
                                   "pending": f"/scopes/{todo_scope}",
                                   "starrers": f"/scopes/{starrer_scope}"},
                "ranked_creators": eligible, "excluded_creators": excluded,
                "outside_project_request_sources": outside_requests,
                "excluded_issues": excluded_issues}
            chosen_url = eligible[0]["closed_issues"][0]["issue_url"] if eligible else None
            selected = [row for row in scopes[issue_scope]["rows"] if row["url"] == chosen_url]
        elif family == "navigate_project_ranked_pending_mr_assigner_difference":
            request_scope, todo_scope = case["scopes"]
            projects = sorted({row["project_path"] for row in source})
            project_counts = [{"project_path": project,
                               "request_sources": [f"/scopes/{request_scope}/rows/{index}"
                                                   for index, row in enumerate(source)
                                                   if row["project_path"] == project]}
                              for project in projects]
            for group in project_counts:
                group["assigned_mr_count"] = len(group["request_sources"])
            project_counts.sort(key=lambda group: (-group["assigned_mr_count"], group["project_path"]))
            if not project_counts:
                raise ValueError("formal navigation predicate has no assigned project")
            chosen_project = project_counts[0]["project_path"]
            requests = {row["url"]: (index, row) for index, row in enumerate(source)}
            matched = {}
            excluded = []
            for index, todo in enumerate(scopes[todo_scope]["rows"]):
                todo_source = f"/scopes/{todo_scope}/rows/{index}"
                proof = {"todo_source": todo_source, "target_source": todo_source + "/url"}
                if todo["url"] not in requests:
                    excluded.append({**proof, "reason": "issue or outside assigned All requests"})
                    continue
                request_index, row = requests[todo["url"]]
                request_source = f"/scopes/{request_scope}/rows/{request_index}"
                proof["request_source"] = request_source
                if row["project_path"] != chosen_project:
                    excluded.append({**proof, "reason": "outside the project-count winner"})
                    continue
                action = re.fullmatch(r"(.+) assigned you\.", todo["action"])
                proof["action_source"] = todo_source + "/action"
                if action is None:
                    excluded.append({**proof, "reason": "not a named-person assigned-you action"})
                    continue
                creator_links = [(link_index, link) for link_index, link in enumerate(row["links"])
                                 if link["text"] == row["author"]
                                 and re.fullmatch(r"/[^/]+", urlsplit(link["url"]).path)]
                assigner_links = [(link_index, link) for link_index, link in enumerate(todo["links"])
                                  if link["text"] == action.group(1)
                                  and re.fullmatch(r"/[^/]+", urlsplit(link["url"]).path)]
                if len({link["url"] for _, link in creator_links}) != 1 or len({link["url"] for _, link in assigner_links}) != 1:
                    raise ValueError("navigation relationship lacks one visible creator/assigner profile link")
                creator_index, creator = creator_links[0]
                assigner_index, assigner = assigner_links[0]
                creator_path = urlsplit(creator["url"]).path
                assigner_path = urlsplit(assigner["url"]).path
                proof.update(creator_profile_source=request_source + f"/links/{creator_index}/url",
                             assigner_profile_source=todo_source + f"/links/{assigner_index}/url",
                             creator_profile_path=creator_path, assigner_profile_path=assigner_path)
                if creator_path == assigner_path:
                    excluded.append({**proof, "reason": "creator and assigning person are the same profile"})
                    continue
                relation = matched.setdefault(row["url"], {
                    "request_url": row["url"], "request_source": request_source,
                    "project_source": request_source + "/project_path",
                    "comment_badge_source": request_source + "/comments_text",
                    "number_source": request_source + "/number",
                    "comment_count": row["comment_count"], "number": row["number"],
                    "pending_matches": []})
                relation["pending_matches"].append(proof)
            ranked = sorted(matched.values(), key=lambda row: (-row["comment_count"], -row["number"]))
            archive["navigation_relation"] = {
                "source_windows": {"requests": f"/scopes/{request_scope}", "pending": f"/scopes/{todo_scope}"},
                "project_counts": project_counts, "selected_project": chosen_project,
                "ranked_requests": ranked, "excluded_pending_rows": excluded,
                "unselected_assigned_sources": [f"/scopes/{request_scope}/rows/{index}"
                                                for index, row in enumerate(source)
                                                if row["url"] not in matched]}
            selected = [requests[ranked[0]["request_url"]][1]] if ranked else []
        elif selection == "last_username_page":
            row = sorted(source, key=lambda row: (row["username"].casefold(), row["username"]))[-1]
            source_page = scopes[case["scopes"][0]]["pages"][row["page_index"]]
            profile_path = urlsplit(row["url"]).path
            selected = [{**row, "url": source_page["url"], "title": row["username"],
                         "terminal_selector": 'div.user-info:has(a.user[href=' + json.dumps(profile_path) + ']) span.cgray:visible'}]
        elif selection == "participant_overlap":
            requests = scopes[case["scopes"][0]]["rows"]
            issues = _filter_rows(scopes[case["scopes"][1]]["rows"], case)
            if len(requests) != 1 or len(issues) != 1:
                raise ValueError("participant relation requires the stated unique source objects")
            creators = {requests[0]["author"], issues[0]["author"]}
            selected = []
            for name in case["scopes"][2:]:
                scope = scopes[name]
                if creators <= {row["name"] for row in scope["rows"]}:
                    selected.append({"url": scope["root"]["url"], "title": scope["root"]["project_heading"],
                                     "terminal_selector": "h1.home-panel-title:visible"})
        elif selection == "starrer_author":
            participants = {row["name"] for row in scopes[case["scopes"][1]]["rows"]}
            selected = [row for row in selected if row["author"] in participants]
        elif selection == "todo_join":
            todos = {row["url"] for row in scopes[case["scopes"][1]]["rows"]
                     if parameters.get("todo_action_contains", "") in row["action"]}
            selected = [row for row in selected if row["url"] in todos]
        elif selection == "issue_mr_author":
            authors = {row["author"] for row in scopes[case["scopes"][1]]["rows"]
                       if row["comment_count"] == parameters["related_mr_comments"]}
            selected = [row for row in selected if row["author"] in authors]
        if case.get("choose") == "first":
            selected = selected[:1]
        if len(selected) != 1:
            raise ValueError("formal navigation predicate does not identify one visible target")
        target = selected[0]
    else:
        raise ValueError("family still awaits current detail readiness")
    archive["derived"] = derived
    archive["selected_rows"] = selected
    evidence = [{"field": "/task_definition/parameters/" + key, "artifact": "observation",
                 "location": "/scopes", "modality": "text",
                 "derivation": "Public rule applied to the complete rendered window; retain all source values and pagination."}
                for key in parameters]
    if family == "navigate_approved_merged_request":
        # The separate approval badge is not inferred from a workflow label,
        # body text or tooltip. Bind each public filter to its selected UI row.
        parameter_sources = {"project_path": "/selected_rows/0/project_path",
                             "required_labels": "/selected_rows/0/labels",
                             "approval_status": "/selected_rows/0/approval_status"}
        for record in evidence:
            key = record["field"].rsplit("/", 1)[1]
            record["location"] = parameter_sources[key]
            record["derivation"] = "Exact rendered selected-row field, checked within the complete assigned Merged window; the approval_status is the text of the distinct visible approval-icon badge, without reading tooltip identities."
        evidence.extend({"field": f"/task_definition/parameters/required_labels/{index}",
                         "artifact": "observation",
                         "location": f"/selected_rows/0/labels/{target['labels'].index(label)}",
                         "modality": "text", "derivation": "Exact visible workflow label in the selected row; it is separate from the approval badge and the Merged lifecycle tab."}
                        for index, label in enumerate(parameters["required_labels"]))
    if family == "project_starrer_assigned_mr_todo_aggregate":
        evidence.extend({"field": f"/task_definition/parameters/project_paths/{index}",
                         "artifact": "observation",
                         "location": f"/scopes/{case['scopes'][index + 2]}/root/url",
                         "modality": "text",
                         "derivation": "Exact public project path equals the path of its actually visited project-root link, whose complete starrer window is retained."}
                        for index in range(len(parameters["project_paths"])))
    if family == "navigate_project_ranked_pending_mr_assigner_difference":
        evidence.append({"field": "/task_definition/goal_template", "artifact": "observation",
                         "location": "/navigation_relation", "modality": "text",
                         "derivation": "Apply the public project-count ranking to every assigned All row, join each pending target by its exact actual request URL, and compare the two visible profile paths named by the MR creator and assigned-you action. Retain all project-count sources, eligible and excluded To-Do source pointers and numeric ranking inputs. Deduplicate pending matches by request URL, then choose the greatest comment badge and project-local number within the project-count winner. The target's actual GET response, final URL and visible title are separately bound after navigation."})
    if family == "navigate_closed_issue_unqueued_mr_creator":
        for record in evidence:
            if record["field"] == "/task_definition/parameters/project_path":
                record["location"] = archive["navigation_relation"]["project_root_source"]
                record["derivation"] = "Parse the actual visited same-project starrer root URL path and require equality to the public project path."
        evidence.append({"field": "/task_definition/goal_template", "artifact": "observation",
                         "location": "/navigation_relation", "modality": "text",
                         "derivation": "Join complete same-project Open MR creators, assigned Closed issue creators and starrers by exact visible profile paths. Exclude the entire creator group if any of that creator's same-project Open MR URLs matches any pending To-Do target URL. Retain all included and excluded groups, every scope and numeric source pointer; rank eligible creators by closed-issue comment sum descending, closed-issue count descending then exact @username ascending. Select that creator's assigned closed issue by comment count descending and numeric issue number descending, then bind actual GET 200, final URL and visible title after navigation."})
    if family == "mr_title_prefix_unassigned_without_pending":
        evidence.append({"field": "/task_definition/goal_template", "artifact": "observation",
                         "location": "/mutation_relation", "modality": "text",
                         "derivation": "Retain complete Open, assigned All, pending To-Do and same-project starrer windows. Anti-join exact actual MR target URLs against both exclusion windows, join creator and starrer visible profile paths, and rank all eligible rows by project-local number ascending. Keep excluded rows and every identity/ranking source pointer. Execute the new focused title edit only after this fresh selection."})
    direct_families = {"issue_filtered_list", "mr_filtered_rank", "mr_without_pending_todo"}
    for index, row in enumerate(derived):
        for field in row:
            location = f"/selected_rows/{index}/{field}" if family in direct_families else "/details" if family in {"issue_note_authors", "issue_checklist_states"} else "/scopes"
            derivation = case["derivation"]
            if family == "project_starrer_assigned_mr_todo_aggregate":
                group = archive["relationship_groups"][index]
                location = group["project_root_source"] if field == "project_path" else f"/relationship_groups/{index}"
                derivation += (
                    " Project path comes from the actually visited project root. Count distinct "
                    "author_profile_path for matched_creator_count and request_url for matched_mr_count; "
                    "sum each retained request's parsed comment badge for comment_total; count distinct "
                    "request_url with nonempty pending_sources for pending_mr_count. Every retained "
                    "request binds its actual MR row, author profile link, matching starrer profile "
                    "and pending target links by source pointers. Complete source_windows prove zero "
                    "groups and exclude issue/unassigned targets; sort by pending count, comment sum "
                    "and project path as stated."
                )
            if family == "assigned_mr_creator_pending_balance":
                location = f"/creator_pending_groups/{index}"
                derivation += (
                    " Group by the exact visible creator profile path, preserving its case and "
                    "prefixing @ for creator_username. Count distinct assigned request URLs; count "
                    "distinct request URLs with any pending_sources once for pending_mr_count; "
                    "subtract from the assigned count for without_pending_mr_count and sum each "
                    "original integer comment badge once for comment_total. Complete source windows "
                    "and all profile/comment/pending pointers prove both positive membership and "
                    "absence; do not merge same display names or count To-Do rows as requests."
                )
            if family in direct_families:
                derivation += " This leaf is the retained original rendered row field; comment_count parses comments_text as an integer, and project-local references are joined to the path of the actual visible object URL."
            evidence.append({"field": f"/task_definition/answer/values/{index}/{field}",
                             "artifact": "observation", "location": location, "modality": "text",
                             "derivation": derivation})
    task_type = case["task_type"]
    if task_type == "RETRIEVE" and not derived:
        raise ValueError("formal retrieval rule has no nonempty grounded answer")
    item_schema = None
    if task_type == "RETRIEVE":
        item_schema = {"type": "object", "properties": {key: {"type": "integer" if type(value) is int else "string"}
                       for key, value in derived[0].items()}, "required": list(derived[0]), "additionalProperties": False}
    definition = {"task_type": task_type, "goal_template": case["goal_template"], "parameters": parameters,
                  "answer": {"status": "SUCCESS", "values": derived if task_type == "RETRIEVE" else None,
                             "item_schema": item_schema, "ordered": task_type == "RETRIEVE"},
                  "completion_events": []}
    return definition, evidence, [], target


def _record_detail(page: Any, output: Path, name: str, selectors: dict,
                   expected_comments: int, timeout_ms: int) -> dict:
    """Capture detail fields after the visible ordinary-comment window is complete.

    The source list's comment badge counts ordinary notes, including each
    suggestion note once and excluding system events. The observed MR UI loads
    its first twenty discussions before automatically fetching the remaining
    cursor page; a disappeared initial skeleton alone is not a complete window.
    """
    from playwright.sync_api import expect

    expect(page.locator(
        'li.note-skeleton[data-qa-selector="skeleton_note_placeholder"]:visible'
    )).to_have_count(0, timeout=timeout_ms)
    expect(page.locator(selectors["notes"])).to_have_count(expected_comments, timeout=timeout_ms)
    observation = _record_extension(page, output, name)
    state = {}
    for field, selector in selectors["fields"].items():
        texts = page.locator(selector).all_inner_texts()
        if len(texts) != 1:
            raise ValueError(f"fresh detail field {field} is absent or ambiguous")
        state[field] = texts[0]
    state["labels"] = [text.strip() for text in page.locator(selectors["labels"]).all_inner_texts()]
    notes = []
    for note in page.locator(selectors["notes"]).all():
        bodies = note.locator(selectors["note_body"]).all_inner_texts()
        authors = note.locator(selectors["note_author"]).all_inner_texts()
        if len(bodies) != 1 or len(authors) != 1:
            raise ValueError("ordinary note lacks its one rendered body and author")
        notes.append({"id": note.get_attribute("id"), "body": bodies[0], "author": authors[0].strip()})
    state["notes"] = notes
    checklist = []
    if "checklist" in selectors:
        for control in page.locator(selectors["checklist"]).all():
            label = control.locator(selectors["checklist_label"]).inner_text()
            checklist.append({"checked": control.is_checked(), "label": label.strip()})
    state["checklist"] = checklist
    observation["detail_state"] = state
    (output / f"{name}.json").write_text(json.dumps(observation, ensure_ascii=False, indent=2) + "\n")
    return observation


def _save_action(page: Any, action: Any, capture: dict, args: argparse.Namespace) -> None:
    """Issue one actual GUI action and retain its native business response."""
    capture["writing"] = True
    try:
        with page.expect_response(lambda response: response.request.method in {"POST", "PUT", "PATCH", "DELETE"}
                                 and urlsplit(response.url).netloc == urlsplit(args.site_url).netloc
                                 and not _is_read_graphql(response.request, capture["read_documents"])):
            action.click()
        page.wait_for_timeout(int(args.settle_seconds * 1000))
    finally:
        capture["writing"] = False


def _is_read_graphql(request: Any, read_documents: set[str]) -> bool:
    """Identify only the already observed query documents, without response facts."""
    if request.method != "POST" or urlsplit(request.url).path != "/api/graphql":
        return False
    payload = json.loads(request.post_data)
    items = payload if isinstance(payload, list) else [payload]
    return bool(items) and all(hashlib.sha256(item["query"].encode()).hexdigest() in read_documents for item in items)


def _ground_mutation(page: Any, output: Path, config: dict, case: dict, args: argparse.Namespace,
                     archive: dict, definition: dict, target: dict, capture: dict) -> tuple[list, list]:
    """Own focused edits, persisted readbacks and exact non-target preservation."""
    family = case["family"]
    parameters = definition["parameters"]
    evidence = []
    if family == "project_star_from_participants":
        page.goto(target["url"], wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        before = _record_extension(page, output, "before")
        if not any(node["tag"] == "button" and node["text"] == "Star" for node in before["visible_nodes"]):
            raise ValueError("selected project is not available to star in this fresh baseline")
        _save_action(page, page.get_by_role("button", name="Star", exact=True), capture, args)
        after_scope = _collect_scope(page, output, archive["initial"], config["scopes"][target["starrer_scope"]], "after_starrers", args)
        original = archive["scopes"][target["starrer_scope"]]["rows"]
        if not {row["username"] for row in original} <= {row["username"] for row in after_scope["rows"]}:
            raise ValueError("starring removed an original visible member")
        added = [row for row in after_scope["rows"] if row["username"] not in {row["username"] for row in original}]
        if len(added) != 1 or added[0]["username"] != "@byteblaze":
            raise ValueError("starring did not add exactly the current Byte Blaze account")
        archive.update(before=before, after_scope=after_scope, added_starrer=added[0])
        parameters.update(final_url=page.url.replace(args.site_url.rstrip("/"), "__GITLAB__", 1),
                          preserved_names=[row["name"] for row in original], added_member_count=1,
                          added_member_name=added[0]["name"], current_username=added[0]["username"])
        checks = [{"kind": "final_url", "parameter": "final_url"},
                  {"kind": "visible_text", "parameter": "current_username"},
                  {"kind": "text_set_preserved", "parameter": "preserved_names", "selector": "div.user-info a.user:visible",
                   "added_count_parameter": "added_member_count", "added_name_parameter": "added_member_name"}]
    elif family == "todo_complete_mr_relation":
        scope = archive["scopes"][case["scopes"][0]]
        original_pending = scope["rows"]
        original_done = archive["scopes"][case["scopes"][2]]["rows"]
        page.goto(scope["pages"][target["page_index"]]["url"], wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        before = _record_extension(page, output, "before")
        target_path = urlsplit(target["url"]).path
        action = page.locator('li.todo:has(a.todo-target-link[href=' + json.dumps(target_path) + ']) a[title="Mark as done"]:visible')
        _save_action(page, action, capture, args)
        pending = _collect_scope(page, output, archive["initial"], config["scopes"][case["scopes"][0]], "after_pending", args)
        done = _collect_scope(page, output, archive["initial"], config["scopes"][case["scopes"][2]], "after_done", args)
        if {row["id"] for row in pending["rows"]} != {row["id"] for row in original_pending if row["id"] != target["id"]}:
            raise ValueError("completing the To-Do changed another pending item")
        if {row["id"] for row in done["rows"]} != {row["id"] for row in original_done} | {target["id"]}:
            raise ValueError("Done does not preserve exactly the old items plus the target")
        archive.update(before=before, after_pending=pending, after_done=done)
        parameters.update(final_url=page.url.replace(args.site_url.rstrip("/"), "__GITLAB__", 1),
                          preserved_done_titles=[row["identity_text"] for row in original_done],
                          added_done_count=1, added_done_name=target["identity_text"])
        checks = [{"kind": "final_url", "parameter": "final_url"},
                  {"kind": "text_set_preserved", "parameter": "preserved_done_titles", "selector": ".todo-title:visible",
                   "added_count_parameter": "added_done_count", "added_name_parameter": "added_done_name"}]
    else:
        object_kind = "issue" if family.startswith("issue_") else "merge_request"
        selectors = config["detail_selectors"][object_kind]
        page.goto(target["url"], wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        before = _record_detail(page, output, "before", selectors, target["comment_count"], args.timeout_ms)
        if selectors["edit_kind"] == "button":
            page.get_by_role("button", name=selectors["edit_text"], exact=True).click()
        else:
            links = {link["url"] for link in before["links"] if link["text"] == selectors["edit_text"]
                     and urlsplit(link["url"]).path == urlsplit(target["url"]).path + "/edit"}
            if len(links) != 1:
                raise ValueError("actual edit link is absent or ambiguous")
            page.goto(links.pop(), wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        editor = _record_extension(page, output, "editor_before")
        title = page.locator(selectors["title_input"]).input_value()
        description = page.locator(selectors["description_input"]).input_value()
        new_title, new_description = title, description
        if family == "issue_title_append":
            new_title = title + parameters["title_suffix"]
        elif family in {"mr_title_prefix", "mr_title_prefix_unassigned_without_pending"}:
            new_title = parameters["title_prefix"] + title
        elif family == "issue_description_mr_summary":
            rows = [row for row in archive["scopes"][case["scopes"][1]]["rows"]
                    if {"javascript", "markup"} <= set(row["labels"]) and "styling" not in row["labels"]]
            rows.sort(key=lambda row: row["number"])
            if not rows:
                raise ValueError("fresh related request predicate has no source rows")
            archive["related_requests"] = rows
            new_description = description + "\n\n" + parameters["summary_heading"] + "\n" + "\n".join(row["reference"] + " — " + row["title"] for row in rows)
        elif family == "mr_description_todo_action":
            todos = [row for row in archive["scopes"][case["scopes"][1]]["rows"] if row["url"] == target["url"]]
            if len(todos) != 1:
                raise ValueError("related To-Do action is not uniquely visible")
            archive["related_todo"] = todos[0]
            new_description = description + "\n\n" + parameters["summary_prefix"] + todos[0]["action"]
        # The observed issue title-save path removes exactly one terminal LF
        # from unchanged Markdown. Keep its full submitted/request value while
        # requiring the separately projected persisted editor representation.
        persisted_description_expected = (
            new_description.removesuffix("\n") if family == "issue_title_append" else new_description
        )
        persistence_projection = "remove_one_terminal_lf" if family == "issue_title_append" else "identity"
        page.locator(selectors["title_input"]).fill(new_title)
        page.locator(selectors["description_input"]).fill(new_description)
        pending = _record_extension(page, output, "editor_pending")
        _save_action(page, page.get_by_role("button", name=selectors["save_text"], exact=True), capture, args)
        page.goto(target["url"], wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        page.reload(wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        after = _record_detail(page, output, "after", selectors, target["comment_count"], args.timeout_ms)
        # Reopen the real editor to verify exact persisted Markdown, then return
        # to the same detail page without another business write.
        if selectors["edit_kind"] == "button":
            page.get_by_role("button", name=selectors["edit_text"], exact=True).click()
        else:
            links = {link["url"] for link in after["links"] if link["text"] == selectors["edit_text"]
                     and urlsplit(link["url"]).path == urlsplit(target["url"]).path + "/edit"}
            page.goto(links.pop(), wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        readback = _record_extension(page, output, "editor_readback")
        persisted_description = page.locator(selectors["description_input"]).input_value()
        if page.locator(selectors["title_input"]).input_value() != new_title or persisted_description != persisted_description_expected:
            raise ValueError("the exact submitted title and Markdown did not persist")
        page.goto(target["url"], wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        terminal = _record_detail(page, output, "terminal", selectors, target["comment_count"], args.timeout_ms)
        changed_field = "title" if family in {"issue_title_append", "mr_title_prefix", "mr_title_prefix_unassigned_without_pending"} else "description"
        for field in before["detail_state"]:
            if field != changed_field and before["detail_state"][field] != terminal["detail_state"][field]:
                raise ValueError(f"unspecified detail field {field} changed during the focused edit")
        archive.update(before=before, editor_before=editor, editor_pending=pending, editor_readback=readback,
                       after=after, terminal=terminal, submitted_title=new_title, submitted_description=new_description,
                       persisted_description=persisted_description,
                       description_persistence_projection={"algorithm": persistence_projection,
                           "source": "/submitted_description", "expected": persisted_description_expected,
                           "actual": "/persisted_description"})
        archive["critical_write_values"] = {"title": new_title, "description": new_description}
        parameters.update(submitted_title=new_title, submitted_description=new_description,
                          persisted_description=persisted_description)
        parameters["final_url"] = page.url.replace(args.site_url.rstrip("/"), "__GITLAB__", 1)
        checks = [{"kind": "final_url", "parameter": "final_url"}]
        for field, selector in selectors["fields"].items():
            key = "final_" + field
            parameters[key] = terminal["detail_state"][field]
            checks.append({"kind": "visible_text_exact", "parameter": key, "selector": selector})
        parameters.update(preserved_labels=before["detail_state"]["labels"],
                          preserved_notes=[note["body"].strip() for note in before["detail_state"]["notes"]],
                          added_label_count=0, added_note_count=0, unchanged_name=new_title)
        checks.extend([
            {"kind": "text_set_preserved", "parameter": "preserved_labels", "selector": selectors["labels"],
             "added_count_parameter": "added_label_count", "added_name_parameter": "unchanged_name"},
            {"kind": "text_set_preserved", "parameter": "preserved_notes", "selector": selectors["notes"] + " " + selectors["note_body"],
             "added_count_parameter": "added_note_count", "added_name_parameter": "unchanged_name"}])
    archive["terminal"] = _record_extension(page, output, "terminal") if "terminal" not in archive else archive["terminal"]
    parameter_sources = {
        "final_url": "/terminal/url", "preserved_labels": "/before/detail_state/labels",
        "preserved_notes": "/before/detail_state/notes", "unchanged_name": "/submitted_title",
        "preserved_names": "/scopes/" + case["scopes"][-1] + "/rows",
        "added_member_name": "/added_starrer/name", "current_username": "/added_starrer/username",
        "added_member_count": "/after_scope/rows", "added_done_count": "/after_done/rows",
        "preserved_done_titles": "/scopes/" + case["scopes"][-1] + "/rows",
        "added_done_name": "/selected_rows/0/identity_text", "title_suffix": "/submitted_title",
        "title_prefix": "/submitted_title", "submitted_title": "/submitted_title", "submitted_description": "/submitted_description", "summary_heading": "/submitted_description",
        "persisted_description": "/persisted_description",
        "summary_prefix": "/submitted_description", "added_label_count": "/before/detail_state/labels",
        "added_note_count": "/before/detail_state/notes"}
    if family == "mr_title_prefix_unassigned_without_pending":
        parameter_sources["project_path"] = "/scopes/" + case["scopes"][3] + "/root/url"
    for key in definition["parameters"]:
        location = parameter_sources.get(key, "/scopes")
        if key.startswith("final_") and key != "final_url":
            location = "/terminal/detail_state/" + key.removeprefix("final_")
        derivation = "Use the exact persisted/source UI field; counts derive the difference between complete before/after sets, zero additions implement the public preservation rule, and supplied suffix/prefix/heading literals appear in the exact submitted editor value."
        if family == "mr_title_prefix_unassigned_without_pending" and key == "project_path":
            derivation = "Parse the actual visited same-project starrer root URL path and remove its leading slash; require equality to the public project path before creator membership or target ranking."
        evidence.append({"field": "/task_definition/parameters/" + key, "artifact": "observation",
                         "location": location, "modality": "text",
                         "derivation": derivation})
    return evidence, checks


def _bind_mutation_events(output: Path, archive: dict, definition: dict) -> list[dict]:
    """Project actual canonical HAR writes into private completion expectations."""
    from webarena_verified.core.utils.network_event_utils import load_har_trace
    events = load_har_trace(output / "network.har")
    writes = archive["mutation_responses"]
    selected = [event for event in events if any(event.url == write["url"] and event.http_method == write["method"]
                and event.request_status == write["status"]
                and event.data["request"].get("postData", {}).get("text") == write["raw_post_data"]
                for write in writes)]
    if len(writes) != 1 or len(selected) != 1 or not selected[0].is_request_success:
        raise ValueError("focused edit needs one actual successful canonical business write")
    event = selected[0]
    expected = {"url_template": "^__GITLAB__" + re.escape(urlsplit(event.url).path) + r"(?:\?.*)?$",
                "http_method": event.http_method, "response_status": event.request_status, "last_event_only": False}
    critical_values = archive.get("critical_write_values", {})
    if critical_values:
        form_encoded = event.data["request"]["postData"]["mimeType"].split(";", 1)[0].strip() == "application/x-www-form-urlencoded"
        leaves = []
        pending = [("$", None, event.post_data)]
        while pending:
            path, leaf_name, value = pending.pop()
            if isinstance(value, dict):
                pending.extend((path + "[" + json.dumps(key) + "]", key, child) for key, child in value.items())
            elif isinstance(value, list):
                pending.extend((path + f"[{index}]", leaf_name, child) for index, child in enumerate(value))
            else:
                leaves.append((path, leaf_name, value))
        critical = {}
        request_values = {}
        for semantic_name, value in critical_values.items():
            # Native HTML textarea form submission serializes line endings as
            # CRLF. JSON submissions retain the exact editor LF representation.
            serialized = re.sub(r"\r\n|\r|\n", "\r\n", value) if form_encoded else value
            matches = [(path, child) for path, leaf_name, child in leaves if child == serialized and isinstance(leaf_name, str)
                       and re.search(r"(?:^|\[)" + re.escape(semantic_name) + r"\]?$", leaf_name)]
            if len(matches) != 1:
                raise ValueError(f"submitted {semantic_name} lacks one exact canonical request-body field")
            path, child = matches[0]
            request_values[semantic_name] = child
            parameter = "request_" + semantic_name
            definition["parameters"][parameter] = child
            critical[path] = "{" + parameter + "}"
        for path, leaf_name, value in leaves:
            if any(path.endswith("[" + json.dumps(key) + "]") for key in ["id", "iid", "issue_id", "merge_request_id", "fullPath"]):
                critical[path] = value
        expected["post_data"] = critical
        archive["request_field_values"] = request_values
    definition["completion_events"] = [expected]
    archive["canonical_write_event"] = {"url": event.url, "http_method": event.http_method,
                                         "response_status": event.request_status, "post_data": event.post_data}
    evidence = [{"field": "/task_definition/completion_events/0/" + field, "artifact": "observation",
             "location": "/canonical_write_event", "modality": "text",
             "derivation": "Use the method/status and endpoint from the actual canonical HAR write; regex-escape its path and retain exact submitted content and target identity leaves."}
            for field in expected if field != "last_event_only"]
    evidence.extend({"field": "/task_definition/parameters/request_" + name, "artifact": "observation",
                     "location": "/request_field_values/" + name, "modality": "text",
                     "derivation": "Exact canonical request-body field, checked against the editor text with only the observed native HTML form CRLF projection; JSON retains editor line endings."}
                    for name in critical_values)
    return evidence


def _formal(page: Any, output: Path, config: dict, case: dict, args: argparse.Namespace, capture: dict) -> tuple[dict, dict, list, list]:
    """Capture every source window, compute GT, then prove any terminal target."""
    # Resolve actual startup responses before interpreting dashboard links.
    # Only the observed transient 502/503 errors get two bounded read retries.
    initial_loads = []
    deadline = monotonic() + args.timeout_ms / 1000
    for attempt in range(1, 4):
        remaining_ms = max(1, int((deadline - monotonic()) * 1000))
        response = page.goto(args.site_url.rstrip("/") + "/", wait_until="domcontentloaded",
                             timeout=remaining_ms)
        page.wait_for_timeout(min(int(args.settle_seconds * 1000),
                                  max(0, int((deadline - monotonic()) * 1000))))
        name = f"initial_load_{attempt:02}"
        observed = _record_extension(page, output, name)
        initial_loads.append({"attempt": attempt, "http_status": response.status,
                              "title": observed["title"], "url": observed["url"],
                              "observation": name + ".json", "screenshot": name + ".png"})
        (output / "initial_load_attempts.json").write_text(json.dumps(initial_loads, indent=2) + "\n")
        if response.status == 200 and observed["title"] == "Projects · Dashboard · GitLab":
            initial = _record_extension(page, output, "initial")
            break
        if response.status not in {502, 503} or attempt == 3 or monotonic() >= deadline:
            raise RuntimeError(f"fresh dashboard is not ready: HTTP {response.status}, title {observed['title']!r}; initial snapshots retained")
        page.wait_for_timeout(min(1000, max(0, int((deadline - monotonic()) * 1000))))
    archive = {"initial": initial, "initial_load_attempts": initial_loads, "scopes": {}}
    # Prepare input: retain every page in each declared visible source window.
    for name in case["scopes"]:
        archive["scopes"][name] = _collect_scope(page, output, initial, config["scopes"][name], name, args)
    if case["family"] in {"issue_note_authors", "issue_checklist_states"}:
        rows = _filter_rows(archive["scopes"][case["scopes"][0]]["rows"], case)
        if case.get("choose") == "first":
            rows = rows[:1]
        if not rows:
            raise ValueError("fresh detail predicate has no source issue")
        archive["details"] = []
        for index, row in enumerate(rows):
            page.goto(row["url"], wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            detail = _record_detail(page, output, f"detail_{index:02}", config["detail_selectors"]["issue"],
                                    row["comment_count"], args.timeout_ms)
            detail["source_row"] = row
            archive["details"].append(detail)
    # Compute the answer or unique target from the newly captured observations.
    definition, evidence, checks, target = _build_definition(case, archive)
    if case["family"] in {"navigate_project_ranked_pending_mr_assigner_difference",
                          "navigate_closed_issue_unqueued_mr_creator"}:
        (output / "navigation_relation.json").write_text(
            json.dumps(archive["navigation_relation"], ensure_ascii=False, indent=2) + "\n")
    if case["family"] == "mr_title_prefix_unassigned_without_pending":
        (output / "mutation_relation.json").write_text(
            json.dumps(archive["mutation_relation"], ensure_ascii=False, indent=2) + "\n")
    # Execute the objective, then keep persistence and terminal checks visible.
    if case["task_type"] == "MUTATE":
        mutation_evidence, checks = _ground_mutation(page, output, config, case, args, archive, definition, target, capture)
        replaced_fields = {record["field"] for record in mutation_evidence}
        evidence = [record for record in evidence if record["field"] not in replaced_fields] + mutation_evidence
    elif target is not None:
        terminal_response = page.goto(target["url"], wait_until="domcontentloaded")
        if case["family"] in {"navigate_project_ranked_pending_mr_assigner_difference",
                              "navigate_closed_issue_unqueued_mr_creator"}:
            archive["terminal_navigation_response"] = {
                "url": terminal_response.url, "http_method": terminal_response.request.method,
                "response_status": terminal_response.status}
            (output / "terminal_navigation_response.json").write_text(
                json.dumps(archive["terminal_navigation_response"], ensure_ascii=False, indent=2) + "\n")
            if terminal_response.status != 200 or terminal_response.request.method != "GET":
                raise RuntimeError("formal navigation target requires an actual successful GET 200")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        terminal = _record_extension(page, output, "terminal")
        parameters = definition["parameters"]
        parameters["final_url"] = terminal["url"].replace(args.site_url.rstrip("/"), "__GITLAB__", 1)
        parameters["final_title"] = target["title"]
        definition["completion_events"] = [{"url_template": "^__GITLAB__" + re.escape(urlsplit(terminal["url"]).path) + r"/?(?:\?.*)?$",
                                            "http_method": "GET", "response_status": 200, "last_event_only": False}]
        checks = [{"kind": "final_url", "parameter": "final_url"},
                  {"kind": "visible_text_exact", "parameter": "final_title", "selector": target.get("terminal_selector", case.get("terminal_title_selector"))}]
        actual = page.locator(target.get("terminal_selector", case.get("terminal_title_selector"))).all_inner_texts()
        if len(actual) != 1 or actual[0].strip() != parameters["final_title"]:
            raise ValueError("formal terminal title differs from the selected fresh list item")
        parameters["final_title"] = actual[0]
        archive["terminal_title_text"] = actual[0]
        archive["terminal"] = terminal
        evidence.extend([{"field": "/task_definition/parameters/final_url", "artifact": "observation",
                          "location": "/terminal/url", "modality": "text", "derivation": "Project the actual terminal origin to __GITLAB__."},
                         {"field": "/task_definition/parameters/final_title", "artifact": "observation",
                          "location": "/terminal_title_text", "modality": "text", "derivation": "Exact terminal element text, with its trimmed content rechecked against the fresh list title."},
                         {"field": "/task_definition/completion_events/0/url_template", "artifact": "observation",
                          "location": "/terminal/url", "modality": "text", "derivation": "Regex-escape the actual terminal path and project its origin."}])
        if case["family"] in {"navigate_project_ranked_pending_mr_assigner_difference",
                              "navigate_closed_issue_unqueued_mr_creator"}:
            evidence.extend({"field": "/task_definition/completion_events/0/" + field,
                             "artifact": "observation", "location": "/terminal_navigation_response/" + field,
                             "modality": "text", "derivation": "Exact method/status of the actual successful selected-target navigation response; the later gate separately checks its canonical HAR/trace completion event."}
                            for field in ["http_method", "response_status"])
    else:
        archive["terminal"] = _record_extension(page, output, "terminal")
    return archive, definition, evidence, checks


def _explore_readiness(page: Any, output: Path, config: dict, args: argparse.Namespace) -> dict:
    """Explore new bounded scopes and editors; never save or construct answers."""
    site = args.site_url.rstrip("/")
    page.goto(site + "/", wait_until="domcontentloaded")
    page.wait_for_timeout(int(args.settle_seconds * 1000))
    initial = _record_extension(page, output, "initial")
    archive = {"initial": initial, "scopes": {}, "details": [], "projects": []}
    for entry in config["readiness"]["dashboards"]:
        url = entry["url"].replace("__GITLAB__", site)
        if not any(link["url"] == url for link in initial["links"]):
            raise ValueError("readiness dashboard is absent from the actual initial UI")
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        dashboard = _record_extension(page, output, entry["name"])
        for tab in entry["tabs"]:
            matches = [link for link in dashboard["links"]
                       if re.fullmatch(re.escape(tab) + r"\s+\d+", link["text"])]
            if len({link["url"] for link in matches}) != 1:
                raise ValueError("readiness tab is not uniquely visible")
            name = entry["name"] + "_" + tab.lower().replace(" ", "_")
            archive["scopes"][name] = _read_scope(
                page, output, matches[0]["url"], name, args, entry["row_key"])
    for target in config["readiness"]["projects"]:
        url = target["url"].replace("__GITLAB__", site)
        if not any(link["url"] == url and " ".join(link["text"].split()) == target["name"]
                   for link in initial["links"]):
            raise ValueError("readiness project is absent from the actual dashboard")
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        project = _record_extension(page, output, "project_" + target["key"])
        archive["projects"].append(project)
        for child in target["children"]:
            matches = {link["url"] for link in project["links"]
                       if urlsplit(link["url"]).path.endswith(child["suffix"])}
            if len(matches) != 1:
                raise ValueError("readiness child link is not uniquely exposed by the project")
            archive["scopes"][target["key"] + "_" + child["key"]] = _read_scope(
                page, output, matches.pop(), target["key"] + "_" + child["key"],
                args, child["row_key"])
    known = {row["url"]: row for scope in archive["scopes"].values() for row in scope["rows"]}
    for index, expected in enumerate(config["readiness"]["detail_urls"]):
        url = expected.replace("__GITLAB__", site)
        if url not in known:
            raise ValueError("readiness detail is absent from the freshly covered visible list")
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(int(args.settle_seconds * 1000))
        detail = _record_extension(page, output, f"detail_{index:02}")
        # Only the actual visible edit button is eligible; opening/cancelling is
        # observational and all business writes remain blocked by the context.
        controls = [node for node in detail["visible_nodes"] if node["tag"] == "button"
                    and (node["aria_label"] == "Edit title and description"
                         or node["text"] == "Edit title and description")]
        if controls:
            page.get_by_role("button", name="Edit title and description", exact=True).click()
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            detail["editor"] = _record_extension(page, output, f"detail_{index:02}_editor")
        else:
            editors = {link["url"] for link in detail["links"] if link["text"] == "Edit"
                       and urlsplit(link["url"]).path == urlsplit(url).path + "/edit"}
            if len(editors) == 1:
                page.goto(editors.pop(), wait_until="domcontentloaded")
                page.wait_for_timeout(int(args.settle_seconds * 1000))
                detail["editor"] = _record_extension(page, output, f"detail_{index:02}_editor")
        archive["details"].append(detail)
    archive["policy"] = {"model_calls": 0, "business_writes": False, "gt_emitted": False}
    return archive


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = json.loads(args.task_config.read_text())
    baseline = json.loads(args.baseline_receipt.read_text())
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("authoring requires a coordinator-owned fresh fixed-image baseline")
    site_url = args.site_url.rstrip("/")
    if baseline["site_url"].rstrip("/") != site_url:
        raise ValueError("endpoint differs from the supplied baseline receipt")
    os.umask(0o077)
    if args.mode == "formal":
        cases = [case for case in config["cases"] if case["generated_task_id"] == args.generated_task_id]
        if len(cases) != 1:
            raise ValueError("formal task ID must identify exactly one configured case")
        case = cases[0]
        output = args.batch_root / "author" / case["generated_task_id"] / args.attempt
    else:
        output = args.batch_root / "exploration" / "gitlab-extension" / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    site = urlsplit(site_url)
    read_documents = set(config["observed_graphql_read_documents"])
    blocked = []
    graphql_documents = {}
    capture = {"writing": False, "responses": [], "read_documents": read_documents}

    def record_response(response):
        request = response.request
        if capture["writing"] and urlsplit(response.url).netloc == site.netloc and request.method in {"POST", "PUT", "PATCH", "DELETE"} and not _is_read_graphql(request, read_documents):
            capture["responses"].append({"url": response.url, "method": request.method, "status": response.status,
                                          "raw_post_data": request.post_data})
            (output / "mutation_responses.json").write_text(
                json.dumps(capture["responses"], ensure_ascii=False, indent=2) + "\n")

    def restrict(route):
        requested = urlsplit(route.request.url)
        same_origin = (requested.scheme, requested.netloc) == (site.scheme, site.netloc)
        allowed = route.request.method in {"GET", "HEAD"}
        if same_origin and route.request.method == "POST" and requested.path == "/api/graphql":
            payload = json.loads(route.request.post_data)
            items = payload if isinstance(payload, list) else [payload]
            hashes = []
            for item in items:
                query = item.get("query", "")
                digest = hashlib.sha256(query.encode()).hexdigest()
                graphql_documents[digest] = {"operation_name": item.get("operationName"), "query": query}
                hashes.append(digest)
            allowed = bool(hashes) and all(digest in read_documents for digest in hashes)
        if same_origin and (allowed or capture["writing"]):
            route.continue_()
        else:
            blocked.append({"method": route.request.method, "path": requested.path})
            route.abort()

    from playwright.sync_api import sync_playwright
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(storage_state=str(args.storage_state),
            viewport={"width": 1920, "height": 1080}, service_workers="block", accept_downloads=False,
            record_har_path=str(output / "network.har"), record_har_content="embed")
        context.route("**/*", restrict)
        context.on("response", record_response)
        context.set_default_timeout(args.timeout_ms)
        context.tracing.start(screenshots=True, snapshots=True, sources=False)
        try:
            page = context.new_page()
            if args.mode == "explore-readiness":
                archive = _explore_readiness(page, output, config, args)
            else:
                archive, definition, evidence, checks = _formal(page, output, config, case, args, capture)
        finally:
            context.tracing.stop(path=str(output / "trace.zip"))
            context.close()
            browser.close()
    archive["blocked_requests"] = blocked
    archive["graphql_documents"] = graphql_documents
    if args.mode == "formal" and case["task_type"] == "MUTATE":
        archive["mutation_responses"] = capture["responses"]
        # Keep real observations and response-window identity even when binding
        # fails. Draft construction remains after a successful canonical binding.
        (output / "mutation_responses.json").write_text(
            json.dumps(capture["responses"], ensure_ascii=False, indent=2) + "\n")
        (output / "observation.json").write_text(json.dumps(archive, ensure_ascii=False, indent=2) + "\n")
        evidence.extend(_bind_mutation_events(output, archive, definition))
    (output / "observation.json").write_text(json.dumps(archive, ensure_ascii=False, indent=2) + "\n")
    if args.mode == "formal":
        relative = output.relative_to(args.batch_root)
        draft = {"generated_task_id": case["generated_task_id"], "task_id": case["task_id"],
                 "intent_template_id": case["intent_template_id"], "site": "gitlab", "start_url": "__GITLAB__",
                 "revision": 1, "source_artifact": case["source_artifact"], "source_record_id": case["source_record_id"],
                 "baseline": baseline, "task_definition": definition,
                 "grounding": {"captured_at": datetime.now(UTC).isoformat(), "family": case["family"],
                               "artifacts": {"har": str(relative / "network.har"), "trace": str(relative / "trace.zip"),
                                             "screenshot": str(relative / "terminal.png"), "observation": str(relative / "observation.json")},
                               "field_evidence": evidence, "final_state_checks": checks}}
        (output / "draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "observed", "mode": args.mode, "directory": str(output),
                      "draft_emitted": args.mode == "formal"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)}), flush=True)
        traceback.print_exc()
        raise SystemExit(1) from None
