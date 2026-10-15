#!/usr/bin/env python3
"""Record fresh, read-only Reddit UI evidence and ground five task families.

The coordinator supplies a new fixed-baseline receipt and authenticated state.
Run in its prepared Playwright runtime, for example:
  python -B author_reddit.py --batch-root /outputs \
    --baseline-receipt /outputs/private/reddit-baseline.json \
    --storage-state /outputs/private/reddit-storage.json \
    --task-config /outputs/plans/reddit_task_configs.json \
    --generated-task-id wavgen-v03-full300-0045

Exploration selects existing resource links only. Answers, derived navigation
targets, and field evidence come exclusively from this newly recorded run.
This entry creates no site resources, calls no models, and reads no catalog.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-url", default="http://localhost:9999")
    parser.add_argument("--generated-task-id", required=True)
    parser.add_argument("--attempt", default="attempt-01")
    parser.add_argument("--task-config", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", args.generated_task_id):
        parser.error("task ID must use the generated batch namespace")
    if not args.attempt.replace("-", "").isalnum():
        parser.error("attempt must be a simple identifier")
    return args


def _record(page: Any, output: Path, name: str) -> dict[str, Any]:
    """Archive rendered card text and actual navigation links, without form secrets."""
    observed = page.evaluate(
        """() => {
          const visible = e => {
            if (!e) return false;
            const box = e.getBoundingClientRect();
            return !!(box.width && box.height);
          };
          const text = e => visible(e) ? e.innerText.trim() : null;
          const link = e => visible(e) ? e.href : null;
          return {
            url: location.href, title: document.title, body_text: document.body.innerText,
            sort_labels: Array.from(document.querySelectorAll('button[aria-label]'))
              .filter(visible).map(e => e.getAttribute('aria-label'))
              .filter(label => label.startsWith('Sort by:')),
            submissions: Array.from(document.querySelectorAll('article.submission'))
              .filter(visible).map(e => {
                const discussion = Array.from(e.querySelectorAll('.submission__nav a[href]'))
                  .find(a => visible(a) && /^[0-9][0-9,]* comments?$/.test(a.innerText.trim()));
                return {
                  title: text(e.querySelector('.submission__title')),
                  domain: text(e.querySelector('.submission__host')),
                  author: text(e.querySelector('.submission__submitter')),
                  comment_count_text: text(discussion), discussion_url: link(discussion),
                  score_text: text(e.querySelector('.submission__vote .vote__net-score'))
                };
              }),
            comments: Array.from(document.querySelectorAll('article.comment'))
              .filter(visible).map(e => {
                const row = e.querySelector(':scope > .comment__row');
                return {
                  dom_id: e.id,
                  author: text(row.querySelector('.comment__info a[href*="/user/"]')),
                  reddit_tag: text(Array.from(
                    row.querySelectorAll('.comment__info small.user-flag'))
                    .find(flag => visible(flag) && /^t1_/.test(flag.innerText.trim()))),
                  body_text: text(row.querySelector('.comment__body')),
                  score_text: text(row.querySelector('.comment__vote .vote__net-score')),
                  permalink_url: link(row.querySelector('a.comment__permalink')),
                  parent_url: link(row.querySelector('a.comment__parent-link'))
                };
              }),
            links: Array.from(document.querySelectorAll('a[href]')).filter(visible)
              .map(e => ({text: text(e), url: e.href}))
          };
        }"""
    )
    page.screenshot(path=str(output / f"{name}.png"), full_page=True)
    (output / f"{name}.json").write_text(json.dumps(observed, ensure_ascii=False, indent=2) + "\n")
    return observed


def build_task_definition(config: dict, observation: dict) -> tuple[dict, list, list]:
    """Derive a task solely from the formal run's recorded visible observations."""
    parameters = dict(config["parameters"])
    evidence = []
    final_checks = []
    pages = observation["pages"]
    mode = config["mode"]
    if mode in {
        "retrieve-forum-comparison",
        "navigate-ranked-discussion",
        "retrieve-forum-source-presence-comparison",
        "navigate-source-presence-discussion",
    }:
        windows = []
        for page_index, page in enumerate(pages):
            if "Sort by: Hot" not in page["sort_labels"]:
                raise ValueError("formal forum window is not the requested Hot view")
            if len(page["submissions"]) != parameters["window_size"]:
                raise ValueError(
                    "formal first-page window does not cover the requested submissions"
                )
            rows = []
            for row_index, item in enumerate(page["submissions"]):
                row = dict(item)
                row["comment_count"] = int(
                    re.fullmatch(r"([0-9][0-9,]*) comments?", row["comment_count_text"])[1].replace(
                        ",", ""
                    )
                )
                row["score"] = int(row["score_text"].replace(",", "").replace("−", "-"))
                rows.append((row, page_index, row_index))
            windows.append(rows)
        if mode in {"retrieve-forum-comparison", "retrieve-forum-source-presence-comparison"}:
            if mode == "retrieve-forum-source-presence-comparison" and len(windows) != 2:
                raise ValueError("source-presence comparison requires two formal forum windows")
            groups = []
            for forum_number, rows in enumerate(windows, start=1):
                if mode == "retrieve-forum-source-presence-comparison":
                    qualifying = [
                        record
                        for record in rows
                        if bool(record[0]["domain"]) == (parameters["source_presence"] == "present")
                        and record[0]["comment_count"] >= parameters["minimum_comments"]
                        and record[0]["score"] >= parameters["minimum_score"]
                    ]
                else:
                    qualifying = [
                        record
                        for record in rows
                        if record[0]["domain"] in {parameters["domain_1"], parameters["domain_2"]}
                        and record[0]["comment_count"] >= parameters["minimum_comments"]
                    ]
                if len(qualifying) < 2:
                    raise ValueError("forum comparison requires multiple qualifying submissions")
                leading = sorted(
                    qualifying,
                    key=lambda record: (
                        -record[0]["score"],
                        record[0]["title"].casefold(),
                        record[0]["title"],
                    ),
                )[0]
                groups.append(
                    (
                        {
                            "forum": parameters[f"forum_{forum_number}"],
                            "qualifying_posts_count": len(qualifying),
                            "total_comments": sum(
                                record[0]["comment_count"] for record in qualifying
                            ),
                            "total_score": sum(record[0]["score"] for record in qualifying),
                            "leading_post_title": leading[0]["title"],
                            "leading_post_author": leading[0]["author"],
                            "leading_post_score": leading[0]["score"],
                        },
                        rows,
                        qualifying,
                        leading,
                    )
                )
            groups.sort(key=lambda group: (-group[0]["total_comments"], group[0]["forum"]))
            for index, (answer, rows, qualifying, leading) in enumerate(groups):
                answer_pointer = f"/task_definition/answer/values/{index}"
                evidence.append(
                    {
                        "field": f"{answer_pointer}/forum",
                        "artifact": "observation",
                        "location": f"/pages/{rows[0][1]}/url",
                        "modality": "text",
                        "derivation": "Forum name in the freshly observed URL.",
                    }
                )
                for row, page_index, row_index in rows:
                    for source in (
                        ("domain", "comment_count_text", "score_text")
                        if mode == "retrieve-forum-source-presence-comparison"
                        else ("domain", "comment_count_text")
                    ):
                        evidence.append(
                            {
                                "field": f"{answer_pointer}/qualifying_posts_count",
                                "artifact": "observation",
                                "location": f"/pages/{page_index}/submissions/{row_index}/{source}",
                                "modality": "text",
                                "derivation": (
                                    "Inspect all first-page records, classify by visible "
                                    "source-domain presence plus minimum comments and score, "
                                    "then count qualifying records."
                                    if mode == "retrieve-forum-source-presence-comparison"
                                    else "Inspect all first-page records, classify by the "
                                    "requested "
                                    "domains and minimum comment count, then count "
                                    "qualifying records."
                                ),
                            }
                        )
                for field, source in {
                    "total_comments": "comment_count_text",
                    "total_score": "score_text",
                }.items():
                    for _, page_index, row_index in qualifying:
                        evidence.append(
                            {
                                "field": f"{answer_pointer}/{field}",
                                "artifact": "observation",
                                "location": f"/pages/{page_index}/submissions/{row_index}/{source}",
                                "modality": "text",
                                "derivation": (
                                    "Parse and sum displayed integers of qualifying records. "
                                    "Classification is bound to qualifying_posts_count."
                                ),
                            }
                        )
                for field, source in {
                    "leading_post_title": "title",
                    "leading_post_author": "author",
                    "leading_post_score": "score_text",
                }.items():
                    evidence.append(
                        {
                            "field": f"{answer_pointer}/{field}",
                            "artifact": "observation",
                            "location": f"/pages/{leading[1]}/submissions/{leading[2]}/{source}",
                            "modality": "text",
                            "derivation": (
                                "Choose highest score among qualifying records; tie-break "
                                "by title alphabetically, case-insensitive then exact title."
                            ),
                        }
                    )
            answer_values = [group[0] for group in groups]
            if mode == "retrieve-forum-source-presence-comparison":
                goal = (
                    "Compare the first Hot submissions page of /f/{forum_1} with that of "
                    "/f/{forum_2}. Each window consists of its first {window_size} listed "
                    "submissions; do not include later pages. In each window retain submissions "
                    "whose source-domain visibility is {source_presence}: present means a source "
                    "domain label is visibly displayed on the card, and absent means no source "
                    "domain label is visibly displayed. Also require at least {minimum_comments} "
                    "displayed comments and a displayed score of at least {minimum_score}. "
                    "For each forum return forum, qualifying_posts_count, total_comments, "
                    "total_score, leading_post_title, leading_post_author, and leading_post_score. "
                    "Sum comments and scores over only qualifying posts. The leading post has "
                    "the highest score; for ties use title alphabetically, ignoring case then "
                    "comparing the exact title. Order forums by total_comments descending, "
                    "then forum name ascending. Use exact displayed titles and usernames and "
                    "numeric counts/scores without commas. Do not vote, subscribe, "
                    "or modify content."
                )
            else:
                goal = (
                    "Compare the first Hot submissions page of /f/{forum_1} with that of "
                    "/f/{forum_2}. Each window consists of its first {window_size} "
                    "listed submissions; "
                    "do not include later pages. In each window retain submissions whose displayed "
                    "source domain is {domain_1} or {domain_2} and which have at least "
                    "{minimum_comments} displayed comments. For each forum return forum, "
                    "qualifying_posts_count, total_comments, total_score, leading_post_title, "
                    "leading_post_author, and leading_post_score. Sum the displayed comment counts "
                    "and scores over only qualifying posts. The leading post has the "
                    "highest score; "
                    "for ties use title alphabetically, ignoring case then comparing "
                    "the exact title. "
                    "Order forums by total_comments descending, then forum name ascending. "
                    "Use exact displayed titles and usernames; return numeric counts and scores "
                    "without commas. Do not vote, subscribe, or modify any content."
                )
        else:
            if len(windows) != 1:
                raise ValueError("ranked discussion navigation requires one formal forum window")
            if mode == "navigate-source-presence-discussion":
                qualifying = [
                    record
                    for record in windows[0]
                    if bool(record[0]["domain"]) == (parameters["source_presence"] == "present")
                    and record[0]["comment_count"] >= parameters["minimum_comments"]
                    and record[0]["score"] >= parameters["minimum_score"]
                ]
            else:
                qualifying = [
                    record
                    for record in windows[0]
                    if record[0]["domain"] == parameters["domain"]
                    and record[0]["comment_count"] >= parameters["minimum_comments"]
                    and record[0]["score"] >= parameters["minimum_score"]
                ]
            if len(qualifying) < 2:
                raise ValueError("ranked navigation requires multiple qualifying submissions")
            selected = sorted(
                qualifying,
                key=lambda record: (
                    -record[0]["comment_count"],
                    -record[0]["score"],
                    record[0]["title"].casefold(),
                    record[0]["title"],
                ),
            )[0]
            path = urlsplit(selected[0]["discussion_url"]).path
            parameters.update(
                target_url="__REDDIT__" + path,
                target_path_regex=re.escape(path),
                selected_title=selected[0]["title"],
            )
            for field, source in {
                "selected_title": "title",
                "target_url": "discussion_url",
                "target_path_regex": "discussion_url",
            }.items():
                evidence.append(
                    {
                        "field": f"/task_definition/parameters/{field}",
                        "artifact": "observation",
                        "location": f"/pages/{selected[1]}/submissions/{selected[2]}/{source}",
                        "modality": "text",
                        "derivation": (
                            "Rank qualifying records by comments descending, score descending, "
                            "then title alphabetically (case-insensitive then exact). Retain "
                            "the observed discussion link and title. Replace only the site "
                            "origin with __REDDIT__."
                        ),
                    }
                )
            final_checks = [
                {"kind": "final_url", "parameter": "target_url"},
                {
                    "kind": "visible_text_exact",
                    "parameter": "selected_title",
                    "selector": "article.submission .submission__title",
                },
            ]
            return (
                {
                    "task_type": "NAVIGATE",
                    "goal_template": (
                        "In /f/{forum}, inspect the first {window_size} submissions on its Hot "
                        "page; do not include later pages. Find submissions whose source-domain "
                        "visibility is {source_presence}: present means a source domain label "
                        "is visibly displayed on the card, and absent means no source domain "
                        "label is visibly displayed. Also require at least {minimum_comments} "
                        "comments and a displayed score of at least {minimum_score}. Open the "
                        "local discussion with the largest comment count and leave it open. "
                        "For tied comment counts choose the higher score; if still tied use "
                        "title alphabetically, ignoring case then comparing the exact title. "
                        "Do not follow an external source link, vote, subscribe, or modify content."
                    )
                    if mode == "navigate-source-presence-discussion"
                    else (
                        "In /f/{forum}, inspect the first {window_size} submissions on its Hot "
                        "page. Find submissions whose displayed source domain is {domain}, whose "
                        "comment count is at least {minimum_comments}, and whose displayed score "
                        "is at least {minimum_score}. Open the discussion with the largest comment "
                        "count and leave that discussion page open. For tied comment counts choose "
                        "the higher score; if still tied use title alphabetically, ignoring case "
                        "then comparing the exact title. Open the local discussion rather than "
                        "the external news article. Do not vote, subscribe, or modify content."
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
                            "url_template": "^__REDDIT__{target_path_regex}(?:[?#].*)?$",
                            "http_method": "GET",
                            "response_status": 200,
                            "last_event_only": False,
                        }
                    ],
                },
                evidence,
                final_checks,
            )
    elif mode == "retrieve-comment-chains":
        answer_values = []
        for index, chain in enumerate(observation["chains"]):
            steps = chain["steps"]
            if len(steps) < 2 or steps[-1]["comment"]["parent_url"] is not None:
                raise ValueError("comment aggregation requires a complete multi-level Parent chain")
            leaf = steps[0]["comment"]
            if (
                leaf["author"] != parameters[f"leaf_author_{index + 1}"]
                or leaf["reddit_tag"] != parameters[f"leaf_tag_{index + 1}"]
            ):
                raise ValueError("freshly observed leaf differs from the selected visible resource")
            first_page = pages[steps[0]["page_index"]]
            if (
                len(first_page["submissions"]) != 1
                or first_page["submissions"][0]["title"] != parameters[f"thread_title_{index + 1}"]
            ):
                raise ValueError("fresh comment discussion differs from the selected visible title")
            for field, location in {
                f"leaf_author_{index + 1}": f"/chains/{index}/steps/0/comment/author",
                f"leaf_tag_{index + 1}": f"/chains/{index}/steps/0/comment/reddit_tag",
                f"thread_title_{index + 1}": f"/pages/{steps[0]['page_index']}/submissions/0/title",
            }.items():
                evidence.append(
                    {
                        "field": f"/task_definition/parameters/{field}",
                        "artifact": "observation",
                        "location": location,
                        "modality": "text",
                        "derivation": "Exact freshly recorded UI text.",
                    }
                )
            scores = [
                int(step["comment"]["score_text"].replace(",", "").replace("−", "-"))
                for step in steps
            ]
            winner = max(range(len(steps)), key=lambda number: scores[number])
            answer_values.append(
                {
                    "leaf_author": leaf["author"],
                    "chain_length": len(steps),
                    "distinct_author_count": len({step["comment"]["author"] for step in steps}),
                    "chain_score": sum(scores),
                    "highest_score_author": steps[winner]["comment"]["author"],
                }
            )
            for field, source, step_numbers, derivation in (
                ("leaf_author", "author", [0], "Exact visible leaf author."),
                (
                    "chain_length",
                    "permalink_url",
                    range(len(steps)),
                    "Count leaf plus all observed Parent ancestors, ending without a Parent link.",
                ),
                (
                    "distinct_author_count",
                    "author",
                    range(len(steps)),
                    "Count distinct exact displayed usernames along the complete recorded chain.",
                ),
                (
                    "chain_score",
                    "score_text",
                    range(len(steps)),
                    "Parse and sum every displayed signed integer score in this complete chain.",
                ),
                (
                    "highest_score_author",
                    "score_text",
                    range(len(steps)),
                    "Compare all scores; a tied maximum chooses the comment closest to the leaf.",
                ),
                (
                    "highest_score_author",
                    "author",
                    [winner],
                    "Exact author of the selected maximum-score comment.",
                ),
            ):
                for step_number in step_numbers:
                    evidence.append(
                        {
                            "field": f"/task_definition/answer/values/{index}/{field}",
                            "artifact": "observation",
                            "location": f"/chains/{index}/steps/{step_number}/comment/{source}",
                            "modality": "text",
                            "derivation": derivation,
                        }
                    )
        goal = (
            "Compare these two comment chains: the comment by {leaf_author_1}, displayed tag "
            "{leaf_tag_1}, in '{thread_title_1}', and the comment by {leaf_author_2}, displayed "
            "tag {leaf_tag_2}, in '{thread_title_2}'. From each leaf comment follow Parent "
            "repeatedly until its top-level ancestor has no Parent link. Include the leaf and "
            "every ancestor, excluding siblings and descendants. For each chain return "
            "leaf_author, chain_length, distinct_author_count, chain_score (sum of displayed "
            "signed scores), and highest_score_author. If highest scores tie choose the comment "
            "nearest the starting leaf. Count distinct exact displayed usernames, treating "
            "[deleted] as one displayed name. Return chains in the order named above, using "
            "exact usernames and numeric counts/scores without commas. Do not reply, vote, "
            "or modify any content."
        )
    else:
        raise ValueError(f"unsupported explicit Reddit author mode: {mode}")
    schema = {
        "type": "object",
        "properties": {
            name: {"type": "integer" if type(value) is int else "string"}
            for name, value in answer_values[0].items()
        },
        "required": list(answer_values[0]),
        "additionalProperties": False,
    }
    return (
        {
            "task_type": "RETRIEVE",
            "goal_template": goal,
            "parameters": parameters,
            "answer": {
                "status": "SUCCESS",
                "values": answer_values,
                "item_schema": schema,
                "ordered": True,
            },
            "completion_events": [],
        },
        evidence,
        final_checks,
    )


def main() -> None:
    args = _parse_args()
    configs = json.loads(args.task_config.read_text())
    matches = [
        config
        for config in configs["tasks"]
        if config["generated_task_id"] == args.generated_task_id
    ]
    if len(matches) != 1:
        raise ValueError("select exactly one configured generated Reddit task")
    config = matches[0]
    if config["mode"] in {
        "retrieve-forum-source-presence-comparison",
        "navigate-source-presence-discussion",
    } and config["parameters"]["source_presence"] not in {"present", "absent"}:
        raise ValueError("source_presence must be present or absent")
    start_url = "__REDDIT__"
    if config["mode"] == "retrieve-forum-source-presence-comparison":
        start_url += "/f/" + config["parameters"]["forum_1"]
    elif config["mode"] == "navigate-source-presence-discussion":
        start_url += "/f/" + config["parameters"]["forum"]
    baseline = json.loads(args.baseline_receipt.read_text())
    if baseline["fresh_from_image"] is not True or not baseline["owned_run"]:
        raise ValueError("authoring requires a coordinator-owned fixed image baseline")
    if baseline["site_url"].rstrip("/") != args.site_url.rstrip("/"):
        raise ValueError("author endpoint differs from the coordinator baseline receipt")
    output = args.batch_root / "author" / args.generated_task_id / args.attempt
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    observed = {"parameters": config["parameters"], "pages": [], "chains": []}
    blocked = []
    origin = urlsplit(args.site_url)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(
            storage_state=str(args.storage_state),
            viewport={"width": 1920, "height": 1080},
            accept_downloads=False,
            service_workers="block",
            record_har_path=str(output / "network.har"),
            record_har_content="embed",
        )

        def restrict(route: Any) -> None:
            requested = urlsplit(route.request.url)
            if (requested.scheme, requested.netloc) != (origin.scheme, origin.netloc):
                blocked.append({"path": requested.path, "reason": "foreign_origin"})
                route.abort()
            elif route.request.method not in {"GET", "HEAD"}:
                blocked.append({"path": requested.path, "reason": "non_read_method"})
                route.abort()
            else:
                route.continue_()

        context.route("**/*", restrict)
        context.tracing.start(screenshots=True, snapshots=True, sources=False)
        page = context.new_page()
        try:
            page.goto(
                start_url.replace("__REDDIT__", args.site_url.rstrip("/"))
                + ("/" if start_url == "__REDDIT__" else ""),
                wait_until="networkidle",
                timeout=120_000,
            )
            _record(page, output, "initial")
            if config["mode"] == "retrieve-comment-chains":
                for leaf_number in (1, 2):
                    chain = {"steps": []}
                    target = config["resource_links"][f"leaf_{leaf_number}"]
                    for _ in range(config["maximum_chain_length"]):
                        target = target.replace("__REDDIT__", args.site_url.rstrip("/"))
                        page.goto(target, wait_until="networkidle", timeout=120_000)
                        snapshot = _record(page, output, f"page_{len(observed['pages']):03}")
                        observed["pages"].append(snapshot)
                        comment_id = re.search(r"/comment/(\d+)", target)[1]
                        matching = [
                            item
                            for item in snapshot["comments"]
                            if item["dom_id"] == f"comment_{comment_id}"
                        ]
                        if len(matching) != 1:
                            raise ValueError("formal permalink does not expose one target comment")
                        comment = matching[0]
                        if not all(
                            comment[field] is not None
                            for field in (
                                "author",
                                "reddit_tag",
                                "body_text",
                                "score_text",
                                "permalink_url",
                            )
                        ):
                            raise ValueError("target comment is not fully rendered and visible")
                        chain["steps"].append(
                            {"page_index": len(observed["pages"]) - 1, "comment": comment}
                        )
                        if comment["parent_url"] is None:
                            break
                        target = comment["parent_url"]
                    else:
                        raise ValueError("Parent chain exceeds the bounded formal recording budget")
                    observed["chains"].append(chain)
            else:
                for link in config["resource_links"].values():
                    page.goto(
                        link.replace("__REDDIT__", args.site_url.rstrip("/")),
                        wait_until="networkidle",
                        timeout=120_000,
                    )
                    observed["pages"].append(
                        _record(page, output, f"page_{len(observed['pages']):03}")
                    )
                definition, _, _ = build_task_definition(config, observed)
                if config["mode"] in {
                    "navigate-ranked-discussion",
                    "navigate-source-presence-discussion",
                }:
                    target = definition["parameters"]["target_url"].replace(
                        "__REDDIT__", args.site_url.rstrip("/")
                    )
                    page.goto(target, wait_until="networkidle", timeout=120_000)
                    observed["terminal"] = _record(page, output, "terminal")
                    if (
                        len(observed["terminal"]["submissions"]) != 1
                        or observed["terminal"]["submissions"][0]["title"]
                        != definition["parameters"]["selected_title"]
                    ):
                        raise ValueError(
                            "observed terminal discussion does not match ranked selection"
                        )
        finally:
            context.tracing.stop(path=str(output / "trace.zip"))
            context.close()
            browser.close()
            (output / "blocked_requests.json").write_text(json.dumps(blocked, indent=2))
            (output / "observation.json").write_text(
                json.dumps(observed, ensure_ascii=False, indent=2)
            )
    definition, evidence, final_checks = build_task_definition(config, observed)
    relative = output.relative_to(args.batch_root)
    draft = {
        "generated_task_id": args.generated_task_id,
        "task_id": config["task_id"],
        "intent_template_id": config["intent_template_id"],
        "site": "reddit",
        "start_url": start_url,
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
                "screenshot": str(relative / ("terminal.png" if final_checks else "page_000.png")),
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
                "status": "grounded",
                "generated_task_id": args.generated_task_id,
                "output": str(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
