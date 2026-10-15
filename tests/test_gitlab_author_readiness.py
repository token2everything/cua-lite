"""Offline UI-boundary regressions, never formal task or browser success evidence."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode

import author_gitlab_extension as owner
import pytest

SELECTORS = {
    "fields": {
        "title": "h1.title:visible", "description": "div.description:visible",
        "assignee": '[data-testid="assignee-block-container"]:visible',
        "reviewers": '[data-qa-selector="reviewers_block_container"]:visible',
        "state": "span.issuable-status-badge:visible",
    },
    "labels": '[data-testid="sidebar-labels"] .gl-label-text:visible',
    "notes": "li.note-comment:not(.system-note):visible",
    "note_body": ".note-body:visible", "note_author": ".note-header-author-name:visible",
    "edit_kind": "link", "edit_text": "Edit",
    "title_input": "input#merge_request_title:visible",
    "description_input": "textarea#merge_request_description:visible", "save_text": "Save changes",
}
SKELETON = 'li.note-skeleton[data-qa-selector="skeleton_note_placeholder"]:visible'
URL = "http://fixture.test/group/project/-/merge_requests/7"


class Texts:
    def __init__(self, values):
        self.values = values

    def all_inner_texts(self):
        return self.values


class Note:
    def __init__(self, number, *, system=False, suggestion=False):
        self.id = f"note_{number}"
        self.system = system
        self.body = "Suggested change\n-old\n+new" if suggestion else f"Comment {number}"

    def get_attribute(self, name):
        assert name == "id"
        return self.id

    def locator(self, selector):
        if selector == SELECTORS["note_body"]:
            return Texts([self.body])
        assert selector == SELECTORS["note_author"]
        return Texts(["Fixture author"])


class DetailPage:
    """Advance external UI through skeleton, first page and cursor page."""
    def __init__(self, counts=(0, 20, 26), skeletons=(10, 0, 0)):
        self.frames = list(zip(counts, skeletons, strict=True))
        self.frame = 0
        self.url = URL

    def advance(self):
        if self.frame + 1 == len(self.frames):
            raise AssertionError("UI did not finish the declared comment window")
        self.frame += 1

    def locator(self, selector):
        if selector in {SKELETON, SELECTORS["notes"]}:
            page = self
            class Rows:
                def count(self):
                    return page.frames[page.frame][1 if selector == SKELETON else 0]

                def all(self):
                    # System events are excluded; nested suggestions are notes,
                    # not extra rows or merely their inner text widgets.
                    notes = [Note(n, suggestion=n < 11) for n in range(page.frames[page.frame][0])]
                    notes += [Note(100, system=True), Note(101, system=True)]
                    return [note for note in notes if not note.system]
            return Rows()
        return Texts(["fixture-label"] if selector == SELECTORS["labels"] else ["Fixture field"])


@pytest.fixture
def count_expectation(monkeypatch):
    page = DetailPage()
    def expect(locator):
        class Expectation:
            def to_have_count(self, expected, *, timeout):
                assert timeout == 120_000
                while locator.count() != expected:
                    page.advance()
        return Expectation()
    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace(expect=expect))
    return page


def test_capture_waits_for_cursor_page_after_initial_skeleton_disappears(tmp_path, monkeypatch, count_expectation):
    page = count_expectation
    captured = []
    def record(*args):
        captured.append(page.frames[page.frame])
        return {"url": URL}
    monkeypatch.setattr(owner, "_record_extension", record)
    detail = owner._record_detail(page, tmp_path, "before", SELECTORS, 26, 120_000)
    notes = detail["detail_state"]["notes"]
    assert captured == [(26, 0)] and len(notes) == 26
    assert len({note["id"] for note in notes}) == 26
    assert sum("Suggested change" in note["body"] for note in notes) == 11


def test_partial_window_never_becomes_an_observation_or_gt(tmp_path, monkeypatch, count_expectation):
    page = count_expectation
    page.frames = [(0, 10), (20, 0)]
    captured = []
    monkeypatch.setattr(owner, "_record_extension", lambda *args: captured.append(args))
    with pytest.raises(AssertionError, match="did not finish"):
        owner._record_detail(page, tmp_path, "before", SELECTORS, 26, 120_000)
    assert captured == [] and not (tmp_path / "before.json").exists()


def test_zero_ordinary_comments_does_not_count_system_events(tmp_path, monkeypatch, count_expectation):
    page = count_expectation
    page.frames = [(0, 0)]
    monkeypatch.setattr(owner, "_record_extension", lambda *args: {"url": URL})
    detail = owner._record_detail(page, tmp_path, "before", SELECTORS, 0, 120_000)
    assert detail["detail_state"]["notes"] == []


def edit_fixture(monkeypatch, changed_field=None, *, issue=False,
                 description="Preserve this\nMarkdown.", readback_description=None):
    selectors = deepcopy(SELECTORS)
    if issue:
        selectors["fields"].pop("reviewers")
        selectors.update(edit_kind="button", edit_text="Edit title and description",
                         title_input="input#issuable-title:visible",
                         description_input="textarea#issue-description:visible")
    target_url = URL.replace("merge_requests", "issues") if issue else URL
    state = {
        "title": "Fixture request", "description": description.removesuffix("\n"),
        "assignee": "0 Assignees", "reviewers": "Fixture reviewer", "state": "Open",
        "labels": ["fixture-label"], "notes": [
            {"id": f"note_{n}", "body": f"Original comment {n}", "author": "Fixture author"}
            for n in range(26)
        ], "checklist": [],
    }
    if issue:
        state.pop("reviewers")
    values = {selectors["title_input"]: state["title"], selectors["description_input"]: description}
    class Page:
        url = target_url
        def goto(self, url, **kwargs):
            self.url = url
        def reload(self, **kwargs):
            pass
        def wait_for_timeout(self, milliseconds):
            pass
        def get_by_role(self, *args, **kwargs):
            return SimpleNamespace(click=lambda: None)
        def locator(self, selector):
            return SimpleNamespace(input_value=lambda: values[selector],
                                   fill=lambda value: values.__setitem__(selector, value))
    def record_detail(_page, _output, name, _selectors, expected_comments, timeout_ms):
        assert expected_comments == 26 and timeout_ms == 120_000
        captured = deepcopy(state)
        if name != "before":
            captured["title"] = values[selectors["title_input"]]
            if changed_field == "notes":
                captured["notes"][0]["body"] = "Unauthorized replacement"
            elif changed_field:
                captured[changed_field] = "Unauthorized replacement"
        return {"url": target_url, "detail_state": captured,
                "links": [{"text": "Edit", "url": target_url + "/edit"}]}
    def save(*args):
        if readback_description is not None:
            values[selectors["description_input"]] = readback_description
        elif issue:
            values[selectors["description_input"]] = values[selectors["description_input"]].removesuffix("\n")
    monkeypatch.setattr(owner, "_record_detail", record_detail)
    monkeypatch.setattr(owner, "_record_extension", lambda *args: {"url": target_url})
    monkeypatch.setattr(owner, "_save_action", save)
    Page.selectors = selectors
    return Page()


def test_focused_title_edit_preserves_complete_ordinary_notes(tmp_path, monkeypatch):
    page = edit_fixture(monkeypatch)
    definition = {"parameters": {"title_prefix": "Draft: "}}
    archive = {"scopes": {"requests": {}}, "initial": {}}
    owner._ground_mutation(
        page, tmp_path, {"detail_selectors": {"merge_request": SELECTORS}},
        {"family": "mr_title_prefix", "scopes": ["requests"]},
        SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000),
        archive, definition, {"url": URL, "comment_count": 26}, {},
    )
    assert archive["before"]["detail_state"]["notes"] == archive["terminal"]["detail_state"]["notes"]
    assert len(definition["parameters"]["preserved_notes"]) == 26
    assert definition["parameters"]["submitted_title"] == "Draft: Fixture request"


def unassigned_title_fixture():
    """Model visible accounts, complete anti-join windows and numeric ordering."""
    project = "group/project"
    case = {"task_type": "MUTATE", "family": "mr_title_prefix_unassigned_without_pending",
            "scopes": ["project_open", "assigned_all", "pending", "starrers"],
            "parameters": {"project_path": project, "title_prefix": "Review queue: "},
            "goal_template": "Choose the oldest unassigned nonpending starrer-created request and prefix {title_prefix}."}
    rows = []
    for number, person in [(20, "creator"), (2, "creator"), (3, "creator"),
                           (5, "homonym"), (10, "creator")]:
        rows.append({"url": f"http://fixture.test/{project}/-/merge_requests/{number}",
                     "title": "Repeated request title", "project_path": project, "number": number,
                     "reference": f"{project}!{number}", "comment_count": number,
                     "comments_text": str(number), "author": "Same display name",
                     "links": [{"text": "Same display name", "url": "http://fixture.test/" + person}]})
    assigned = [{"url": rows[1]["url"], "title": "An independently changed title", "state": "Merged"}]
    pending = [{"id": "todo_1", "url": rows[2]["url"]},
               {"id": "todo_2", "url": rows[2]["url"]},
               {"id": "issue", "url": "http://fixture.test/group/project/-/issues/10"},
               {"id": "other_project", "url": "http://fixture.test/group/other/-/merge_requests/10"}]
    starrers = [{"name": "Same display name", "url": "http://fixture.test/creator", "username": "@creator"}]
    scopes = {"project_open": {"rows": rows, "pages": [{"merge_requests": rows}], "state": "Open"},
              "assigned_all": {"rows": assigned, "pages": [{"merge_requests": assigned}], "state": "All"},
              "pending": {"rows": pending, "pages": [{"todos": pending}], "state": "To Do"},
              "starrers": {"rows": starrers, "pages": [{"starrers": starrers}],
                           "root": {"url": "http://fixture.test/" + project}}}
    for scope in scopes.values():
        scope.update(displayed_count=len(scope["rows"]), complete_window=True)
    return case, {"scopes": scopes}


def test_unassigned_title_selection_joins_profile_accounts_and_excludes_exact_targets():
    case, archive = unassigned_title_fixture()
    definition, _, _, target = owner._build_definition(case, archive)
    assert target["number"] == 10
    assert definition["answer"]["values"] is definition["answer"]["item_schema"] is None
    relation = archive["mutation_relation"]
    assert [row["number"] for row in relation["ranked_eligible_requests"]] == [10, 20]
    excluded = {row["number"]: row for row in relation["excluded_requests"]}
    assert excluded[2]["assigned_sources"] and not excluded[2]["pending_sources"]
    assert len(excluded[3]["pending_sources"]) == 2
    assert excluded[5]["reason"] == "creator is not a same-project starrer"
    # Issue numbers, other projects and display-name collisions cannot join.
    assert relation["ranked_eligible_requests"][0]["pending_sources"] == []


def test_unassigned_title_target_recomputed_from_changed_formal_assignment():
    case, archive = unassigned_title_fixture()
    archive["scopes"]["assigned_all"]["rows"].append(
        {"url": "http://fixture.test/group/project/-/merge_requests/10"})
    _, _, _, target = owner._build_definition(case, archive)
    assert target["number"] == 20


def test_unassigned_title_target_ignores_same_title_on_different_request():
    case, archive = unassigned_title_fixture()
    archive["scopes"]["assigned_all"]["rows"][0]["title"] = "Repeated request title"
    _, _, _, target = owner._build_definition(case, archive)
    assert target["number"] == 10


@pytest.mark.parametrize("identity", ["missing", "ambiguous"])
def test_unassigned_title_creator_requires_actual_unique_profile(identity):
    case, archive = unassigned_title_fixture()
    links = archive["scopes"]["project_open"]["rows"][0]["links"]
    if identity == "missing":
        links.clear()
    else:
        links.append({"text": "Same display name", "url": "http://fixture.test/another"})
    with pytest.raises(ValueError, match="one visible profile link"):
        owner._build_definition(case, archive)


def test_unassigned_title_uses_same_project_starrer_root():
    case, archive = unassigned_title_fixture()
    archive["scopes"]["starrers"]["root"]["url"] = "http://fixture.test/group/other"
    with pytest.raises(ValueError, match="fresh visible starrer root"):
        owner._build_definition(case, archive)


def test_unassigned_title_no_eligible_target_is_author_failure():
    case, archive = unassigned_title_fixture()
    archive["scopes"]["starrers"]["rows"].clear()
    with pytest.raises(ValueError, match="one fresh target"):
        owner._build_definition(case, archive)


def test_unassigned_title_relation_retains_complete_windows_and_resolvable_pointers():
    case, archive = unassigned_title_fixture()
    _, evidence, _, _ = owner._build_definition(case, archive)
    def resolve(pointer):
        value = archive
        for part in pointer.lstrip("/").split("/"):
            value = value[int(part)] if isinstance(value, list) else value[part]
        return value
    relation = archive["mutation_relation"]
    assert all(resolve(pointer)["complete_window"] for pointer in relation["source_windows"].values())
    assert resolve(relation["project_root_source"]) == "http://fixture.test/group/project"
    for proof in relation["ranked_eligible_requests"] + relation["excluded_requests"]:
        assert resolve(proof["request_source"])["url"] == proof["request_url"]
        assert resolve(proof["number_source"]) == proof["number"]
        for pointer in proof["assigned_sources"] + proof["pending_sources"]:
            assert resolve(pointer) == proof["request_url"]
        if "creator_profile_source" in proof:
            assert resolve(proof["creator_profile_source"]).endswith(proof["creator_profile_path"])
            assert all(resolve(pointer).endswith(proof["creator_profile_path"])
                       for pointer in proof["starrer_sources"])
    proof = next(record for record in evidence if record["field"] == "/task_definition/goal_template")
    assert resolve(proof["location"]) == relation


def test_unassigned_title_reuses_persisted_title_edit_and_complete_preservation(tmp_path, monkeypatch):
    page = edit_fixture(monkeypatch)
    case, archive = unassigned_title_fixture()
    archive["initial"] = {}
    definition = {"parameters": dict(case["parameters"])}
    evidence, checks = owner._ground_mutation(
        page, tmp_path, {"detail_selectors": {"merge_request": SELECTORS}}, case,
        SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000),
        archive, definition, {"url": URL, "comment_count": 26}, {})
    assert archive["submitted_title"] == "Review queue: Fixture request"
    assert archive["critical_write_values"]["title"] == archive["submitted_title"]
    assert archive["submitted_description"] == archive["persisted_description"] == "Preserve this\nMarkdown."
    assert archive["before"]["detail_state"]["notes"] == archive["terminal"]["detail_state"]["notes"]
    assert len(definition["parameters"]["preserved_notes"]) == 26
    assert any(check["kind"] == "visible_text_exact" and check["parameter"] == "final_title" for check in checks)
    project = next(record for record in evidence if record["field"] == "/task_definition/parameters/project_path")
    assert project["location"] == "/scopes/starrers/root/url"


@pytest.mark.parametrize("changed_field", ["notes", "labels", "reviewers", "description", "assignee", "state"])
def test_unassigned_title_rejects_every_observed_non_target_change(tmp_path, monkeypatch, changed_field):
    page = edit_fixture(monkeypatch, changed_field)
    case, archive = unassigned_title_fixture()
    archive["initial"] = {}
    with pytest.raises(ValueError, match=f"detail field {changed_field} changed"):
        owner._ground_mutation(
            page, tmp_path, {"detail_selectors": {"merge_request": SELECTORS}}, case,
            SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000),
            archive, {"parameters": dict(case["parameters"])}, {"url": URL, "comment_count": 26}, {})


@pytest.mark.parametrize("changed_field", ["notes", "labels", "reviewers", "description"])
def test_focused_edit_still_rejects_non_target_changes(tmp_path, monkeypatch, changed_field):
    page = edit_fixture(monkeypatch, changed_field)
    with pytest.raises(ValueError, match=f"detail field {changed_field} changed"):
        owner._ground_mutation(
            page, tmp_path, {"detail_selectors": {"merge_request": SELECTORS}},
            {"family": "mr_title_prefix", "scopes": ["requests"]},
            SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000),
            {"scopes": {"requests": {}}, "initial": {}}, {"parameters": {"title_prefix": "Draft: "}},
            {"url": URL, "comment_count": 26}, {},
        )


def test_issue_title_save_projects_one_terminal_lf_and_preserves_full_submission(tmp_path, monkeypatch):
    original = "Preserve\nall [links](/target).\n"
    page = edit_fixture(monkeypatch, issue=True, description=original)
    archive = {"scopes": {"requests": {}}, "initial": {}}
    definition = {"parameters": {"title_suffix": " [follow-up]"}}
    owner._ground_mutation(
        page, tmp_path, {"detail_selectors": {"issue": page.selectors}},
        {"family": "issue_title_append", "scopes": ["requests"]},
        SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000),
        archive, definition, {"url": page.url, "comment_count": 26}, {},
    )
    assert archive["submitted_description"] == archive["critical_write_values"]["description"] == original
    assert archive["persisted_description"] == original.removesuffix("\n")
    assert archive["description_persistence_projection"]["algorithm"] == "remove_one_terminal_lf"
    assert archive["before"]["detail_state"]["description"] == archive["terminal"]["detail_state"]["description"]
    assert archive["before"]["detail_state"]["notes"] == archive["terminal"]["detail_state"]["notes"]


@pytest.mark.parametrize("original,actual", [
    ("First\nMiddle\nLast\n", "First\nChanged\nLast"),
    ("Preserve\n\n", "Preserve"),
    ("Preserve\n", "Preserve\n"),
])
def test_issue_readback_rejects_content_changes_or_multiple_or_missing_lf_projection(tmp_path, monkeypatch, original, actual):
    page = edit_fixture(monkeypatch, issue=True, description=original, readback_description=actual)
    with pytest.raises(ValueError, match="Markdown did not persist"):
        owner._ground_mutation(
            page, tmp_path, {"detail_selectors": {"issue": page.selectors}},
            {"family": "issue_title_append", "scopes": ["requests"]},
            SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000),
            {"scopes": {"requests": {}}, "initial": {}}, {"parameters": {"title_suffix": " [follow-up]"}},
            {"url": page.url, "comment_count": 26}, {},
        )


def test_issue_title_projection_does_not_allow_other_field_changes(tmp_path, monkeypatch):
    page = edit_fixture(monkeypatch, "labels", issue=True, description="Preserve\n")
    with pytest.raises(ValueError, match="detail field labels changed"):
        owner._ground_mutation(
            page, tmp_path, {"detail_selectors": {"issue": page.selectors}},
            {"family": "issue_title_append", "scopes": ["requests"]},
            SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000),
            {"scopes": {"requests": {}}, "initial": {}}, {"parameters": {"title_suffix": " [follow-up]"}},
            {"url": page.url, "comment_count": 26}, {},
        )


def test_mr_title_readback_still_requires_identity_description(tmp_path, monkeypatch):
    page = edit_fixture(monkeypatch, description="Preserve\n", readback_description="Preserve")
    with pytest.raises(ValueError, match="Markdown did not persist"):
        owner._ground_mutation(
            page, tmp_path, {"detail_selectors": {"merge_request": SELECTORS}},
            {"family": "mr_title_prefix", "scopes": ["requests"]},
            SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000),
            {"scopes": {"requests": {}}, "initial": {}}, {"parameters": {"title_prefix": "Draft: "}},
            {"url": URL, "comment_count": 26}, {},
        )


def startup_fixture(tmp_path, monkeypatch, responses):
    class Page:
        calls = 0
        url = "http://fixture.test/"
        def goto(self, url, *, wait_until, timeout):
            assert url == self.url and wait_until == "domcontentloaded" and 0 < timeout <= 120_000
            self.status, self.title = responses[self.calls]
            self.calls += 1
            return SimpleNamespace(status=self.status)
        def wait_for_timeout(self, milliseconds):
            pass
    page = Page()
    fetched = []
    def record(_page, output, name):
        observed = {"url": page.url, "title": page.title, "links": []}
        (output / (name + ".json")).write_text(json.dumps(observed))
        (output / (name + ".png")).write_bytes(b"offline screenshot placeholder")
        return observed
    def scope(_page, _output, initial, _descriptor, name, _args):
        assert page.status == 200 and initial["title"] == "Projects · Dashboard · GitLab"
        fetched.append(name)
        return {}
    monkeypatch.setattr(owner, "_record_extension", record)
    monkeypatch.setattr(owner, "_collect_scope", scope)
    monkeypatch.setattr(owner, "_build_definition", lambda *args: ({}, [], [], None))
    return page, fetched


@pytest.mark.parametrize("responses,expected_calls", [
    ([(502, "GitLab is not responding (502)"), (200, "Projects · Dashboard · GitLab")], 2),
    ([(503, "Unavailable"), (502, "GitLab is not responding (502)"),
      (200, "Projects · Dashboard · GitLab")], 3),
])
def test_formal_startup_waits_for_actual_dashboard_before_fetching_scopes(tmp_path, monkeypatch, responses, expected_calls):
    page, fetched = startup_fixture(tmp_path, monkeypatch, responses)
    archive, *_ = owner._formal(
        page, tmp_path, {"scopes": {"fixture": {}}},
        {"scopes": ["fixture"], "family": "offline", "task_type": "RETRIEVE"},
        SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000), {},
    )
    assert page.calls == expected_calls and fetched == ["fixture"]
    records = json.loads((tmp_path / "initial_load_attempts.json").read_text())
    assert [r["http_status"] for r in records] == [status for status, _ in responses]
    assert archive["initial_load_attempts"] == records
    assert all((tmp_path / r["observation"]).exists() and (tmp_path / r["screenshot"]).exists() for r in records)


@pytest.mark.parametrize("responses,expected_calls", [
    ([(502, "GitLab is not responding (502)")] * 3, 3),
    ([(200, "Sign in · GitLab")], 1),
    ([(500, "Unknown server error")], 1),
])
def test_nonready_startup_preserves_snapshots_and_never_interprets_scope_absence(tmp_path, monkeypatch, responses, expected_calls):
    page, fetched = startup_fixture(tmp_path, monkeypatch, responses)
    with pytest.raises(RuntimeError, match="fresh dashboard is not ready"):
        owner._formal(
            page, tmp_path, {"scopes": {"fixture": {}}},
            {"scopes": ["fixture"], "family": "offline", "task_type": "RETRIEVE"},
            SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000), {},
        )
    assert page.calls == expected_calls and fetched == []
    assert not (tmp_path / "initial.json").exists()
    assert len(json.loads((tmp_path / "initial_load_attempts.json").read_text())) == expected_calls


def bind_fixture(monkeypatch, *, form=False, batch=False, different_description=False,
                 description="Line one {literal}\n\nLine two"):
    title = "Fixture request"
    wire = description.replace("\n", "\r\n") if form else description
    if different_description:
        wire += "Unauthorized addition"
    if form:
        encoded = urlencode({"merge_request[title]": title, "merge_request[description]": wire,
                             "merge_request[target_branch]": "", "unrelated": ""})
        payload = {k: v[0] for k, v in parse_qs(encoded, keep_blank_values=True).items()}
        raw_post_data = encoded
        mime = "application/x-www-form-urlencoded"
    else:
        payload = {"variables": {"input": {"title": title, "description": wire,
                                            "id": "gid://gitlab/MergeRequest/7"}}}
        if batch:
            payload = [payload]
        raw_post_data = json.dumps(payload)
        mime = "application/json"
    event = SimpleNamespace(url=URL, http_method="POST", request_status=302, is_request_success=True,
                            post_data=payload, data={"request": {"postData": {"mimeType": mime, "text": raw_post_data}}})
    monkeypatch.setitem(sys.modules, "webarena_verified.core.utils.network_event_utils",
                        SimpleNamespace(load_har_trace=lambda path: [event]))
    definition = {"parameters": {"submitted_title": title, "submitted_description": description}}
    archive = {"mutation_responses": [{"url": URL, "method": "POST", "status": 302,
                                       "raw_post_data": raw_post_data}],
               "critical_write_values": {"title": title, "description": description}}
    return definition, archive, wire


@pytest.mark.parametrize("form,batch", [(False, False), (False, True), (True, False)])
def test_exact_body_binding_projects_only_native_form_line_endings(tmp_path, monkeypatch, form, batch):
    definition, archive, wire = bind_fixture(monkeypatch, form=form, batch=batch)
    evidence = owner._bind_mutation_events(tmp_path, archive, definition)
    event = definition["completion_events"][0]
    assert event["http_method"] == "POST" and event["response_status"] == 302
    assert definition["parameters"]["request_description"] == wire
    assert definition["parameters"]["submitted_description"] == "Line one {literal}\n\nLine two"
    assert any(e["field"] == "/task_definition/parameters/request_description" for e in evidence)
    rendered = {p: v.format_map(definition["parameters"]) for p, v in event["post_data"].items()}
    assert wire in rendered.values()
    assert any("title" in p and v == "Fixture request" for p, v in rendered.items())


def test_empty_description_is_bound_to_its_field_not_other_empty_form_values(tmp_path, monkeypatch):
    definition, archive, _ = bind_fixture(monkeypatch, form=True, description="")
    owner._bind_mutation_events(tmp_path, archive, definition)
    assert definition["completion_events"][0]["post_data"]['$["merge_request[description]"]'] == "{request_description}"


def test_json_line_endings_are_not_normalized_like_native_forms(tmp_path, monkeypatch):
    definition, archive, _ = bind_fixture(monkeypatch)
    module = sys.modules["webarena_verified.core.utils.network_event_utils"]
    event = module.load_har_trace(tmp_path / "network.har")[0]
    event.post_data["variables"]["input"]["description"] = definition["parameters"]["submitted_description"].replace("\n", "\r\n")
    with pytest.raises(ValueError, match="submitted description lacks one exact"):
        owner._bind_mutation_events(tmp_path, archive, definition)


@pytest.mark.parametrize("form", [False, True])
def test_changed_submission_is_rejected_without_relaxing_event_matcher(tmp_path, monkeypatch, form):
    definition, archive, _ = bind_fixture(monkeypatch, form=form, different_description=True)
    with pytest.raises(ValueError, match="submitted description lacks one exact"):
        owner._bind_mutation_events(tmp_path, archive, definition)


@pytest.mark.parametrize("graphql_write", [False, True])
def test_many_read_graphql_200_events_do_not_replace_one_captured_write(
    tmp_path, monkeypatch, graphql_write,
):
    definition, archive, _ = bind_fixture(monkeypatch, form=not graphql_write)
    module = sys.modules["webarena_verified.core.utils.network_event_utils"]
    write = module.load_har_trace(tmp_path / "network.har")[0]
    graphql_url = "http://fixture.test/api/graphql"
    if graphql_write:
        write.url = graphql_url
        write.request_status = 200
        archive["mutation_responses"][0].update(url=graphql_url, status=200)
    reads = []
    for n in range(29):
        body = json.dumps({"query": "query permission { project { id } }", "variables": {"n": n}})
        reads.append(SimpleNamespace(url=graphql_url, http_method="POST", request_status=200,
                                     is_request_success=True, post_data=json.loads(body),
                                     data={"request": {"postData": {"mimeType": "application/json", "text": body}}}))
    monkeypatch.setitem(sys.modules, "webarena_verified.core.utils.network_event_utils",
                        SimpleNamespace(load_har_trace=lambda path: reads + [write]))
    owner._bind_mutation_events(tmp_path, archive, definition)
    assert len(definition["completion_events"]) == 1
    assert definition["completion_events"][0]["response_status"] == (200 if graphql_write else 302)
    assert definition["parameters"]["request_title"] == "Fixture request"


def test_unknown_additional_actual_write_remains_a_rejection(tmp_path, monkeypatch):
    definition, archive, _ = bind_fixture(monkeypatch, form=True)
    module = sys.modules["webarena_verified.core.utils.network_event_utils"]
    original = module.load_har_trace(tmp_path / "network.har")[0]
    url = "http://fixture.test/api/graphql"
    body = json.dumps({"query": "mutation extraWrite { changeState }"})
    extra = SimpleNamespace(url=url, http_method="POST", request_status=200, is_request_success=True,
                            post_data=json.loads(body), data={"request": {"postData": {
                                "mimeType": "application/json", "text": body}}})
    archive["mutation_responses"].append({"url": url, "method": "POST", "status": 200,
                                           "raw_post_data": body})
    monkeypatch.setitem(sys.modules, "webarena_verified.core.utils.network_event_utils",
                        SimpleNamespace(load_har_trace=lambda path: [original, extra]))
    with pytest.raises(ValueError, match="one actual successful canonical business write"):
        owner._bind_mutation_events(tmp_path, archive, definition)


def test_captured_body_must_equal_the_canonical_request_body(tmp_path, monkeypatch):
    definition, archive, _ = bind_fixture(monkeypatch, form=True)
    archive["mutation_responses"][0]["raw_post_data"] += "&different_request=1"
    with pytest.raises(ValueError, match="one actual successful canonical business write"):
        owner._bind_mutation_events(tmp_path, archive, definition)


def test_absent_body_has_one_identity_for_actual_star_and_todo_requests(tmp_path, monkeypatch):
    event = SimpleNamespace(url=URL, http_method="DELETE", request_status=200,
                            is_request_success=True, post_data=None, data={"request": {}})
    monkeypatch.setitem(sys.modules, "webarena_verified.core.utils.network_event_utils",
                        SimpleNamespace(load_har_trace=lambda path: [event]))
    definition = {"parameters": {}}
    archive = {"mutation_responses": [{"url": URL, "method": "DELETE", "status": 200,
                                       "raw_post_data": None}]}
    owner._bind_mutation_events(tmp_path, archive, definition)
    assert definition["completion_events"][0]["http_method"] == "DELETE"


def test_allowlisted_permission_query_field_is_not_a_mutation():
    import hashlib
    query = """query mrUserPermission($projectPath: ID!, $iid: String!) {
  project(fullPath: $projectPath) {
    id
    mergeRequest(iid: $iid) {
      id
      userPermissions {
        updateMergeRequest
        __typename
      }
      __typename
    }
    __typename
  }
}
"""
    digest = hashlib.sha256(query.encode()).hexdigest()
    assert digest == "329750bb683332b36065224f912318a7cdb1d104d7fc0329c5083cebb041a6de"
    request = SimpleNamespace(method="POST", url="http://fixture.test/api/graphql",
                              post_data=json.dumps({"query": query, "operationName": "mrUserPermission"}))
    assert owner._is_read_graphql(request, {digest})
    request.post_data = json.dumps({"query": "mutation updateMergeRequest { updateMergeRequest }"})
    assert not owner._is_read_graphql(request, {digest})


def test_main_preserves_capture_and_observation_before_failed_binding_without_draft(
    tmp_path, monkeypatch,
):
    task = "wavgen-offline-6105110"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"cases": [{"generated_task_id": task, "task_type": "MUTATE"}],
                                 "observed_graphql_read_documents": []}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"fresh_from_image": True, "owned_run": "offline",
                                   "site_url": "http://fixture.test"}))
    args = SimpleNamespace(mode="formal", task_config=config, baseline_receipt=baseline,
                           storage_state=tmp_path / "auth.json", site_url="http://fixture.test",
                           batch_root=tmp_path / "output", generated_task_id=task, attempt="grounding-01",
                           timeout_ms=120_000)
    monkeypatch.setattr(owner, "_parse_args", lambda argv: args)
    callbacks = {}
    class Context:
        tracing = SimpleNamespace(start=lambda **kw: None, stop=lambda **kw: None)
        def route(self, *args):
            pass
        def on(self, name, callback):
            callbacks[name] = callback
        def set_default_timeout(self, timeout):
            pass
        def new_page(self):
            return None
        def close(self):
            pass
    browser = SimpleNamespace(new_context=lambda **kw: Context(), close=lambda: None)
    class Runtime:
        chromium = SimpleNamespace(launch=lambda **kw: browser)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=Runtime))
    body = "merge_request%5Btitle%5D=Fixture"
    def formal(_page, output, _config, _case, _args, capture):
        capture["writing"] = True
        callbacks["response"](SimpleNamespace(url=URL, status=302,
            request=SimpleNamespace(method="POST", url=URL, post_data=body)))
        assert json.loads((output / "mutation_responses.json").read_text())[0]["raw_post_data"] == body
        capture["writing"] = False
        return {"terminal": {"url": URL}}, {}, [], []
    def fail_binding(*args):
        raise ValueError("offline canonical binding rejected")
    monkeypatch.setattr(owner, "_formal", formal)
    monkeypatch.setattr(owner, "_bind_mutation_events", fail_binding)
    with pytest.raises(ValueError, match="canonical binding rejected"):
        owner.main([])
    directory = args.batch_root / "author" / task / "grounding-01"
    observation = json.loads((directory / "observation.json").read_text())
    assert observation["mutation_responses"][0]["raw_post_data"] == body
    assert not (directory / "draft.json").exists()


def approval_navigation_fixture():
    case = {"task_type": "NAVIGATE", "family": "navigate_approved_merged_request",
            "selection": "unique", "scopes": ["assigned_mrs_merged"],
            "parameters": {"project_path": "group/project", "required_labels": ["in review"],
                           "approval_status": "Approved"},
            "goal_template": "Open the merged request marked {approval_status} carrying {required_labels}."}
    rows = [{"url": URL, "project_path": "group/project", "number": 7,
             "reference": "group/project!7", "title": "Fixture approved request",
             "labels": ["in review", "resource"], "approval_status": "Approved",
             "body_text": "MERGED\nApproved\nin review"},
            {"url": URL.replace("/7", "/8"), "project_path": "group/project", "number": 8,
             "reference": "group/project!8", "title": "Approved in title alone",
             "labels": ["in review", "Approved"], "approval_status": None,
             "body_text": "MERGED\nApproved\nin review"},
            {"url": URL.replace("/7", "/9"), "project_path": "group/project", "number": 9,
             "reference": "group/project!9", "title": "Fixture pending request",
             "labels": ["in review"], "approval_status": "Pending",
             "body_text": "MERGED\nPending\nin review"}]
    archive = {"scopes": {"assigned_mrs_merged": {"rows": rows, "state": "Merged",
                                               "displayed_count": 3, "complete_window": True}}}
    return case, archive


def test_approval_navigation_uses_distinct_badge_instead_of_label_or_body():
    case, archive = approval_navigation_fixture()
    definition, _, _, target = owner._build_definition(case, archive)
    assert target["url"] == URL
    assert archive["selected_rows"] == [archive["scopes"]["assigned_mrs_merged"]["rows"][0]]
    assert definition["answer"]["values"] is None
    assert definition["answer"]["item_schema"] is None
    # The workflow label can remain even though lifecycle state is Merged;
    # neither the label nor the word Approved in title/body proves approval.
    archive["scopes"]["assigned_mrs_merged"]["rows"][0]["approval_status"] = None
    with pytest.raises(ValueError, match="one visible target"):
        owner._build_definition(case, archive)


@pytest.mark.parametrize("status", [None, "", "Pending", "approved", "Approved by someone", "Unknown"])
def test_unknown_or_nonexact_approval_badges_do_not_default_to_approved(status):
    case, archive = approval_navigation_fixture()
    row = archive["scopes"]["assigned_mrs_merged"]["rows"][0]
    row["approval_status"] = status
    assert owner._filter_rows([row], case) == []


@pytest.mark.parametrize("changed", ["project_path", "labels"])
def test_approved_badge_does_not_override_other_public_filters(changed):
    case, archive = approval_navigation_fixture()
    row = archive["scopes"]["assigned_mrs_merged"]["rows"][0]
    row[changed] = "other/project" if changed == "project_path" else ["resource"]
    assert owner._filter_rows([row], case) == []


def test_two_approved_matches_remain_an_ambiguous_navigation_failure():
    case, archive = approval_navigation_fixture()
    rows = archive["scopes"]["assigned_mrs_merged"]["rows"]
    rows[1]["approval_status"] = "Approved"
    with pytest.raises(ValueError, match="one visible target"):
        owner._build_definition(case, archive)


def test_missing_structured_approval_field_fails_instead_of_reading_old_body_text():
    case, archive = approval_navigation_fixture()
    row = archive["scopes"]["assigned_mrs_merged"]["rows"][0]
    del row["approval_status"]
    with pytest.raises(KeyError, match="approval_status"):
        owner._filter_rows([row], case)


def test_approval_and_workflow_label_evidence_resolve_to_distinct_rendered_fields():
    case, archive = approval_navigation_fixture()
    definition, evidence, _, target = owner._build_definition(case, archive)
    fields = {item["field"]: item for item in evidence}
    for field, value in [("project_path", "group/project"), ("approval_status", "Approved")]:
        record = fields["/task_definition/parameters/" + field]
        assert record["location"] == "/selected_rows/0/" + field
        assert target[field] == value == definition["parameters"][field]
    label = fields["/task_definition/parameters/required_labels/0"]
    assert label["location"] == "/selected_rows/0/labels/0"
    assert target["labels"][0] == definition["parameters"]["required_labels"][0] == "in review"
    assert fields["/task_definition/parameters/approval_status"]["modality"] == "text"


def test_approval_capture_does_not_change_existing_family_selection():
    _, archive = approval_navigation_fixture()
    rows = archive["scopes"]["assigned_mrs_merged"]["rows"]
    case = {"parameters": {"project_path": "group/project", "required_labels": ["in review"]}}
    assert owner._filter_rows(rows, case) == rows


def project_relationship_fixture():
    """Provide distinct accounts and complete UI windows for the new join."""
    projects = ["group/alpha", "group/beta", "group/empty", "group/gamma"]
    case = {"task_type": "RETRIEVE", "family": "project_starrer_assigned_mr_todo_aggregate",
            "scopes": ["assigned_all", "pending", "alpha", "beta", "empty", "gamma"],
            "parameters": {"project_paths": projects},
            "goal_template": "Compare {project_paths} using assigned All requests and pending To-Dos.",
            "derivation": "Join creators by actual profile path within each project; keep zero groups."}
    people = {"alpha": ["alice"], "beta": ["bob"], "empty": ["other"], "gamma": ["alice"]}
    scopes = {}
    for project, scope_name in zip(projects, case["scopes"][2:], strict=True):
        rows = [{"name": "Same display name", "username": person,
                 "url": "http://fixture.test/" + person, "page_index": 0, "row_index": index}
                for index, person in enumerate(people[scope_name])]
        scopes[scope_name] = {"root": {"url": "http://fixture.test/" + project},
                              "rows": rows, "pages": [{"starrers": rows}],
                              "displayed_count": len(rows), "complete_window": True}
    requests = []
    for project, number, person, comments, lifecycle in [
        ("group/alpha", 1, "alice", 3, "Open"),
        ("group/alpha", 2, "alice", 5, "Merged"),
        ("group/alpha", 3, "homonym", 99, "Open"),
        ("group/beta", 4, "bob", 8, "Open"),
        ("group/beta", 5, "alice", 99, "Open"),
        ("group/gamma", 6, "alice", 9, "Merged"),
        ("outside/project", 7, "alice", 999, "Open"),
    ]:
        index = len(requests)
        requests.append({"project_path": project, "number": number,
                         "url": f"http://fixture.test/{project}/-/merge_requests/{number}",
                         "reference": f"{project}!{number}", "title": f"Request {number}",
                         "author": "Same display name", "comment_count": comments,
                         "comments_text": str(comments), "state": lifecycle,
                         "page_index": 0, "row_index": index,
                         "links": [{"text": "unrelated", "url": "http://fixture.test/unrelated"},
                                   {"text": "Same display name", "url": "http://fixture.test/" + person}]})
    pending = [{"url": row["url"], "id": f"todo_{index}"}
               for index, row in enumerate([requests[0], requests[0], requests[3], requests[5]])]
    pending.extend([
        {"url": "http://fixture.test/group/alpha/-/issues/1", "id": "todo_issue"},
        {"url": "http://fixture.test/group/alpha/-/merge_requests/99", "id": "todo_unassigned"},
    ])
    scopes["assigned_all"] = {"rows": requests, "pages": [{"merge_requests": requests}],
                              "state": "All", "displayed_count": len(requests), "complete_window": True}
    scopes["pending"] = {"rows": pending, "pages": [{"todos": pending}],
                         "state": "To Do", "displayed_count": len(pending), "complete_window": True}
    return case, {"scopes": scopes}


def test_project_join_counts_distinct_creators_and_pending_requests_and_keeps_zero_group():
    case, archive = project_relationship_fixture()
    definition, _, _, target = owner._build_definition(case, archive)
    assert target is None
    assert definition["answer"]["values"] == [
        {"project_path": "group/gamma", "matched_creator_count": 1, "matched_mr_count": 1,
         "comment_total": 9, "pending_mr_count": 1},
        {"project_path": "group/alpha", "matched_creator_count": 1, "matched_mr_count": 2,
         "comment_total": 8, "pending_mr_count": 1},
        {"project_path": "group/beta", "matched_creator_count": 1, "matched_mr_count": 1,
         "comment_total": 8, "pending_mr_count": 1},
        {"project_path": "group/empty", "matched_creator_count": 0, "matched_mr_count": 0,
         "comment_total": 0, "pending_mr_count": 0},
    ]
    alpha = next(group for group in archive["relationship_groups"] if group["project_path"] == "group/alpha")
    assert len(alpha["matched_requests"][0]["pending_sources"]) == 2
    assert alpha["matched_requests"][1]["pending_sources"] == []
    assert definition["answer"]["ordered"] is True
    assert definition["answer"]["item_schema"]["properties"]["comment_total"] == {"type": "integer"}


def test_profile_identity_excludes_homonym_and_own_project_starrer_join_excludes_other_project():
    case, archive = project_relationship_fixture()
    owner._build_definition(case, archive)
    matched = {row["request_url"] for group in archive["relationship_groups"]
               for row in group["matched_requests"]}
    requests = archive["scopes"]["assigned_all"]["rows"]
    assert requests[2]["url"] not in matched  # same displayed author, different profile
    assert requests[4]["url"] not in matched  # Alice stars Alpha, not Beta
    assert requests[6]["url"] not in matched  # outside the public project groups
    assert requests[1]["url"] in matched     # Merged remains in the assigned All window


def test_pending_rank_precedes_comments_and_only_exact_retained_mr_targets_count():
    case, archive = project_relationship_fixture()
    pending = archive["scopes"]["pending"]["rows"]
    pending.append({"url": archive["scopes"]["assigned_all"]["rows"][1]["url"], "id": "todo_second"})
    definition, _, _, _ = owner._build_definition(case, archive)
    rows = definition["answer"]["values"]
    assert rows[0]["project_path"] == "group/alpha"
    assert rows[0]["pending_mr_count"] == 2
    assert rows[0]["comment_total"] == 8 < rows[1]["comment_total"]
    assert sum(row["pending_mr_count"] for row in rows) == 4


@pytest.mark.parametrize("author", [None, "", "Profile link is absent"])
def test_missing_visible_creator_identity_fails_formal_join(author):
    case, archive = project_relationship_fixture()
    archive["scopes"]["assigned_all"]["rows"][0]["author"] = author
    with pytest.raises(ValueError, match="one visible profile link"):
        owner._build_definition(case, archive)


def test_two_different_visible_creator_profile_links_remain_ambiguous():
    case, archive = project_relationship_fixture()
    archive["scopes"]["assigned_all"]["rows"][0]["links"].append(
        {"text": "Same display name", "url": "http://fixture.test/other"})
    with pytest.raises(ValueError, match="one visible profile link"):
        owner._build_definition(case, archive)


def test_public_project_group_must_equal_formal_project_root():
    case, archive = project_relationship_fixture()
    archive["scopes"]["alpha"]["root"]["url"] = "http://fixture.test/group/unrequested"
    with pytest.raises(ValueError, match="fresh visible project root"):
        owner._build_definition(case, archive)


def test_relationship_evidence_resolves_each_answer_leaf_and_original_source_pointer():
    case, archive = project_relationship_fixture()
    definition, evidence, _, _ = owner._build_definition(case, archive)
    def resolve(pointer):
        value = archive
        for part in pointer.lstrip("/").split("/"):
            value = value[int(part)] if isinstance(value, list) else value[part]
        return value
    fields = {item["field"]: item for item in evidence}
    for index, group in enumerate(archive["relationship_groups"]):
        row = definition["answer"]["values"][index]
        assert resolve(group["project_root_source"]).endswith("/" + row["project_path"])
        for field in row:
            proof = fields[f"/task_definition/answer/values/{index}/{field}"]
            value = resolve(proof["location"])
            assert value.endswith("/" + row[field]) if field == "project_path" else value == group
        for source in group["source_windows"].values():
            assert resolve(source)["complete_window"] is True
        for request in group["matched_requests"]:
            original = resolve(request["request_source"])
            assert original["url"] == request["request_url"]
            assert resolve(request["author_profile_source"]).endswith(request["author_profile_path"])
            assert int(resolve(request["comment_badge_source"])) == request["comment_count"]
            assert all(resolve(source).endswith(request["author_profile_path"])
                       for source in request["starrer_sources"])
            assert all(resolve(source) == request["request_url"] for source in request["pending_sources"])
    for index, project in enumerate(definition["parameters"]["project_paths"]):
        proof = fields[f"/task_definition/parameters/project_paths/{index}"]
        assert resolve(proof["location"]).endswith("/" + project)


def test_relation_is_recomputed_from_changed_formal_badge_not_exploratory_totals():
    case, archive = project_relationship_fixture()
    row = archive["scopes"]["assigned_all"]["rows"][1]
    row.update(comments_text="13", comment_count=13)
    definition, _, _, _ = owner._build_definition(case, archive)
    alpha = next(item for item in definition["answer"]["values"] if item["project_path"] == "group/alpha")
    assert alpha["comment_total"] == 16
    assert definition["answer"]["values"][0]["project_path"] == "group/alpha"


def pending_navigation_fixture():
    """Exercise project ranking and creator/assigner identities across UI windows."""
    case = {"task_type": "NAVIGATE", "family": "navigate_project_ranked_pending_mr_assigner_difference",
            "selection": "unique", "scopes": ["assigned_all", "pending"], "parameters": {},
            "goal_template": "Open the ranked assigned pending request with a different creator and assigning person.",
            "derivation": "Count complete assigned-All projects and compare actual visible profile paths.",
            "terminal_title_selector": "h1.title:visible"}
    requests = []
    for project, number, person, comments, lifecycle in [
        ("group/alpha", 2, "alice", 9, "Open"),
        ("group/alpha", 10, "alice", 9, "Merged"),
        ("group/alpha", 11, "bob", 100, "Open"),
        ("group/alpha", 12, "alice", 90, "Open"),
        ("group/alpha", 13, "alice", 80, "Open"),
        ("group/beta", 20, "alice", 999, "Open"),
    ]:
        requests.append({"url": f"http://fixture.test/{project}/-/merge_requests/{number}",
                         "project_path": project, "number": number,
                         "reference": f"{project}!{number}", "title": f"Request {number}",
                         "author": "Alex", "comment_count": comments, "comments_text": str(comments),
                         "state": lifecycle, "links": [{"text": "Alex", "url": "http://fixture.test/" + person}]})
    pending = []
    for index, (request, actor, action) in enumerate([
        (requests[0], "roshan", "Alex assigned you."),
        (requests[1], "roshan", "Alex assigned you."),
        (requests[1], "roshan", "Alex assigned you."),
        (requests[2], "bob", "Alex assigned you."),
        (requests[3], "byteblaze", "You assigned to yourself."),
        (requests[4], None, "Could not merge."),
        (requests[5], "roshan", "Alex assigned you."),
    ]):
        pending.append({"id": f"todo_{index}", "url": request["url"], "action": action,
                        "links": [{"text": "Alex", "url": "http://fixture.test/" + actor}] if actor else []})
    pending.extend([
        {"id": "todo_issue", "url": "http://fixture.test/group/alpha/-/issues/2", "action": "Alex assigned you.", "links": []},
        {"id": "todo_unassigned", "url": "http://fixture.test/group/alpha/-/merge_requests/999", "action": "Alex assigned you.", "links": []},
    ])
    return case, {"scopes": {
        "assigned_all": {"rows": requests, "pages": [{"merge_requests": requests}],
                         "state": "All", "displayed_count": len(requests), "complete_window": True},
        "pending": {"rows": pending, "pages": [{"todos": pending}],
                    "state": "To Do", "displayed_count": len(pending), "complete_window": True}}}


def test_pending_navigation_uses_all_project_count_profile_difference_and_numeric_tie_break():
    case, archive = pending_navigation_fixture()
    definition, evidence, _, target = owner._build_definition(case, archive)
    assert target["number"] == 10 and target["state"] == "Merged"
    assert definition["answer"]["values"] is None and definition["answer"]["item_schema"] is None
    assert len(archive["navigation_relation"]["project_counts"][0]["request_sources"]) == 5
    assert archive["navigation_relation"]["selected_project"] == "group/alpha"
    ranked = archive["navigation_relation"]["ranked_requests"]
    assert [row["number"] for row in ranked] == [10, 2]
    assert len(ranked[0]["pending_matches"]) == 2  # two To-Dos, one target request
    assert ranked[0]["pending_matches"][0]["creator_profile_path"] == "/alice"
    assert ranked[0]["pending_matches"][0]["assigner_profile_path"] == "/roshan"
    assert any(item["location"] == "/navigation_relation" for item in evidence)


def test_project_count_tie_uses_project_path_before_comparing_target_comments():
    case, archive = pending_navigation_fixture()
    source = archive["scopes"]["assigned_all"]["rows"]
    beta = deepcopy(source[-1])
    beta.update(url=beta["url"].replace("/20", "/21"), number=21)
    archive["scopes"]["assigned_all"]["rows"] = [source[0], source[1], source[-1], beta]
    _, _, _, target = owner._build_definition(case, archive)
    assert target["project_path"] == "group/alpha" and target["number"] == 10
    groups = archive["navigation_relation"]["project_counts"]
    assert [len(group["request_sources"]) for group in groups] == [2, 2]
    assert [group["project_path"] for group in groups] == ["group/alpha", "group/beta"]


def test_comment_number_ranking_dominates_request_number_and_is_recomputed():
    case, archive = pending_navigation_fixture()
    archive["scopes"]["assigned_all"]["rows"][0].update(comment_count=11, comments_text="11")
    _, _, _, target = owner._build_definition(case, archive)
    assert target["number"] == 2
    assert [row["comment_count"] for row in archive["navigation_relation"]["ranked_requests"]] == [11, 9]


def test_pending_navigation_preserves_all_exclusions_without_treating_self_assignment_as_named_actor():
    case, archive = pending_navigation_fixture()
    owner._build_definition(case, archive)
    excluded = archive["navigation_relation"]["excluded_pending_rows"]
    assert [row["reason"] for row in excluded] == [
        "creator and assigning person are the same profile",
        "not a named-person assigned-you action",
        "not a named-person assigned-you action",
        "outside the project-count winner",
        "issue or outside assigned All requests",
        "issue or outside assigned All requests",
    ]
    assert excluded[0]["creator_profile_path"] == excluded[0]["assigner_profile_path"] == "/bob"
    assert excluded[-2]["target_source"] == "/scopes/pending/rows/7/url"


def test_no_eligible_pending_navigation_target_is_a_formal_failure():
    case, archive = pending_navigation_fixture()
    for row in archive["scopes"]["pending"]["rows"][:3]:
        row["links"][0]["url"] = "http://fixture.test/alice"
    with pytest.raises(ValueError, match="one visible target"):
        owner._build_definition(case, archive)
    assert archive["navigation_relation"]["ranked_requests"] == []


@pytest.mark.parametrize("side", ["creator", "assigner"])
def test_missing_or_ambiguous_visible_profile_does_not_silently_select_a_navigation_target(side):
    case, archive = pending_navigation_fixture()
    row = archive["scopes"]["assigned_all" if side == "creator" else "pending"]["rows"][0]
    row["links"].append({"text": "Alex", "url": "http://fixture.test/different"})
    with pytest.raises(ValueError, match="one visible creator/assigner profile link"):
        owner._build_definition(case, archive)


def test_pending_navigation_private_proof_resolves_every_project_join_exclusion_and_rank_source():
    case, archive = pending_navigation_fixture()
    owner._build_definition(case, archive)
    def resolve(pointer):
        value = archive
        for token in pointer.lstrip("/").split("/"):
            value = value[int(token)] if isinstance(value, list) else value[token]
        return value
    relation = archive["navigation_relation"]
    for pointer in relation["source_windows"].values():
        assert resolve(pointer)["complete_window"] is True
    for group in relation["project_counts"]:
        assert all(resolve(pointer)["project_path"] == group["project_path"] for pointer in group["request_sources"])
        assert group["assigned_mr_count"] == len(group["request_sources"])
    for ranked in relation["ranked_requests"]:
        assert resolve(ranked["request_source"])["url"] == ranked["request_url"]
        assert int(resolve(ranked["comment_badge_source"])) == ranked["comment_count"]
        assert resolve(ranked["number_source"]) == ranked["number"]
        for match in ranked["pending_matches"]:
            assert resolve(match["target_source"]) == ranked["request_url"]
            assert resolve(match["action_source"]) == "Alex assigned you."
            assert resolve(match["creator_profile_source"]).endswith(match["creator_profile_path"])
            assert resolve(match["assigner_profile_source"]).endswith(match["assigner_profile_path"])
    for excluded in relation["excluded_pending_rows"]:
        assert resolve(excluded["todo_source"])["url"] == resolve(excluded["target_source"])
    retained_urls = {row["request_url"] for row in relation["ranked_requests"]}
    assert all(resolve(pointer)["url"] not in retained_urls for pointer in relation["unselected_assigned_sources"])


def formal_pending_navigation_fixture(monkeypatch, status=200, terminal_title="Request 10"):
    case, archived = pending_navigation_fixture()
    class Page:
        url = "http://fixture.test/"
        def goto(self, url, **kwargs):
            self.url = url
            return SimpleNamespace(url=url, status=200 if url.endswith(".test/") else status,
                                   request=SimpleNamespace(method="GET"))
        def wait_for_timeout(self, milliseconds):
            pass
        def locator(self, selector):
            assert selector == "h1.title:visible"
            return SimpleNamespace(all_inner_texts=lambda: [terminal_title])
    monkeypatch.setattr(owner, "_record_extension", lambda page, output, name: {
        "title": "Projects · Dashboard · GitLab" if page.url.endswith(".test/") else terminal_title,
        "url": page.url, "links": []})
    monkeypatch.setattr(owner, "_collect_scope", lambda page, output, initial, descriptor, name, args: deepcopy(archived["scopes"][name]))
    return case, Page()


def test_formal_pending_navigation_binds_actual_get_200_final_url_and_exact_visible_title(tmp_path, monkeypatch):
    case, page = formal_pending_navigation_fixture(monkeypatch)
    archive, definition, evidence, checks = owner._formal(
        page, tmp_path, {"scopes": {"assigned_all": {}, "pending": {}}}, case,
        SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000), {})
    assert archive["terminal_navigation_response"]["response_status"] == 200
    assert definition["parameters"]["final_url"] == "__GITLAB__/group/alpha/-/merge_requests/10"
    assert definition["parameters"]["final_title"] == "Request 10"
    assert definition["completion_events"][0]["http_method"] == "GET"
    assert definition["completion_events"][0]["response_status"] == 200
    assert checks == [{"kind": "final_url", "parameter": "final_url"},
                      {"kind": "visible_text_exact", "parameter": "final_title", "selector": "h1.title:visible"}]
    assert {item["location"] for item in evidence} >= {
        "/terminal_navigation_response/http_method", "/terminal_navigation_response/response_status",
        "/terminal/url", "/terminal_title_text", "/navigation_relation"}


@pytest.mark.parametrize("status", [404, 503])
def test_formal_pending_navigation_preserves_actual_failed_response_and_relationship_proof(tmp_path, monkeypatch, status):
    case, page = formal_pending_navigation_fixture(monkeypatch, status=status)
    with pytest.raises(RuntimeError, match="actual successful GET 200"):
        owner._formal(page, tmp_path, {"scopes": {"assigned_all": {}, "pending": {}}}, case,
                      SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000), {})
    response = json.loads((tmp_path / "terminal_navigation_response.json").read_text())
    relation = json.loads((tmp_path / "navigation_relation.json").read_text())
    assert response["response_status"] == status
    assert relation["ranked_requests"][0]["request_url"] == response["url"]
    assert not (tmp_path / "draft.json").exists()


def test_formal_pending_navigation_keeps_exact_title_check_even_after_real_get_200(tmp_path, monkeypatch):
    case, page = formal_pending_navigation_fixture(monkeypatch, terminal_title="Wrong target title")
    with pytest.raises(ValueError, match="terminal title differs"):
        owner._formal(page, tmp_path, {"scopes": {"assigned_all": {}, "pending": {}}}, case,
                      SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000), {})
    assert json.loads((tmp_path / "terminal_navigation_response.json").read_text())["response_status"] == 200


def creator_balance_fixture():
    case = {"task_type": "RETRIEVE", "family": "assigned_mr_creator_pending_balance",
            "scopes": ["assigned_all", "pending"], "parameters": {},
            "goal_template": "Report creator account workload and its pending balance.",
            "derivation": "Complete All requests joined to pending targets by exact URL."}
    requests = []
    for number, creator, comments, state in [(1, "alice", 3, "Open"),
                                             (2, "alice", 5, "Merged"),
                                             (3, "bob", 12, "Merged"),
                                             (4, "charlie", 0, "Open")]:
        requests.append({"url": f"http://fixture.test/group/project/-/merge_requests/{number}",
                         "project_path": "group/project", "number": number,
                         "reference": f"group/project!{number}", "title": f"Request {number}",
                         "author": "Same displayed name", "comment_count": comments,
                         "comments_text": str(comments), "state": state,
                         "links": [{"text": "Same displayed name", "url": "http://fixture.test/" + creator}]})
    pending = [{"id": "todo_1", "url": requests[0]["url"]},
               {"id": "todo_2", "url": requests[0]["url"]},
               {"id": "todo_issue", "url": "http://fixture.test/group/project/-/issues/3"},
               {"id": "todo_unassigned", "url": "http://fixture.test/group/project/-/merge_requests/99"}]
    return case, {"scopes": {
        "assigned_all": {"rows": requests, "complete_window": True, "state": "All"},
        "pending": {"rows": pending, "complete_window": True, "state": "To Do"}}}


def test_creator_workload_counts_requests_once_and_separates_homonymous_accounts():
    case, archive = creator_balance_fixture()
    definition, _, _, target = owner._build_definition(case, archive)
    assert target is None
    assert definition["answer"]["values"] == [
        {"creator_username": "@bob", "assigned_mr_count": 1, "pending_mr_count": 0,
         "without_pending_mr_count": 1, "comment_total": 12},
        {"creator_username": "@alice", "assigned_mr_count": 2, "pending_mr_count": 1,
         "without_pending_mr_count": 1, "comment_total": 8},
        {"creator_username": "@charlie", "assigned_mr_count": 1, "pending_mr_count": 0,
         "without_pending_mr_count": 1, "comment_total": 0},
    ]
    schema = definition["answer"]["item_schema"]
    assert schema["properties"]["creator_username"] == {"type": "string"}
    assert all(schema["properties"][key] == {"type": "integer"} for key in schema["required"]
               if key != "creator_username")
    assert definition["answer"]["ordered"] is True


def test_creator_workload_order_prioritizes_unqueued_count_then_comments_then_username():
    case, archive = creator_balance_fixture()
    requests = archive["scopes"]["assigned_all"]["rows"]
    requests[0].update(comment_count=7, comments_text="7")
    requests.append({**deepcopy(requests[3]), "url": "http://fixture.test/group/project/-/merge_requests/5"})
    archive["scopes"]["pending"]["rows"].append({"id": "todo_bob", "url": requests[2]["url"]})
    definition, _, _, _ = owner._build_definition(case, archive)
    assert [row["creator_username"] for row in definition["answer"]["values"]] == ["@charlie", "@alice", "@bob"]
    archive["scopes"]["pending"]["rows"] = []
    requests[0].update(comment_count=0, comments_text="0")
    requests[1].update(comment_count=0, comments_text="0")
    definition, _, _, _ = owner._build_definition(case, archive)
    assert [row["creator_username"] for row in definition["answer"]["values"]][:2] == ["@alice", "@charlie"]


@pytest.mark.parametrize("ambiguous", [False, True])
def test_creator_workload_rejects_missing_or_ambiguous_visible_creator_identity(ambiguous):
    case, archive = creator_balance_fixture()
    row = archive["scopes"]["assigned_all"]["rows"][0]
    row["links"] = row["links"] + [{"text": row["author"], "url": "http://fixture.test/other"}] if ambiguous else []
    with pytest.raises(ValueError, match="one visible profile link"):
        owner._build_definition(case, archive)


def test_creator_workload_answer_leaves_bind_complete_windows_and_each_original_source():
    case, archive = creator_balance_fixture()
    definition, evidence, _, _ = owner._build_definition(case, archive)
    def resolve(pointer):
        value = archive
        for token in pointer.lstrip("/").split("/"):
            value = value[int(token)] if isinstance(value, list) else value[token]
        return value
    fields = {item["field"]: item for item in evidence}
    for index, group in enumerate(archive["creator_pending_groups"]):
        assert all(resolve(pointer)["complete_window"] for pointer in group["source_windows"].values())
        for field in definition["answer"]["values"][index]:
            assert resolve(fields[f"/task_definition/answer/values/{index}/{field}"]["location"]) == group
        for request in group["requests"]:
            assert resolve(request["request_source"])["url"] == request["request_url"]
            assert resolve(request["creator_profile_source"]).endswith(group["creator_profile_path"])
            assert int(resolve(request["comment_badge_source"])) == request["comment_count"]
            assert all(resolve(pointer) == request["request_url"] for pointer in request["pending_sources"])


def closed_issue_creator_fixture():
    case = {"task_type": "NAVIGATE", "family": "navigate_closed_issue_unqueued_mr_creator",
            "selection": "unique", "scopes": ["project_open", "assigned_closed", "pending", "starrers"],
            "parameters": {"project_path": "group/alpha"},
            "goal_template": "Open the ranked closed issue for a starrer-created unqueued MR creator in {project_path}.",
            "derivation": "Whole-creator pending anti-join and closed-issue comment/count ranking.",
            "terminal_title_selector": "h1.title:visible"}
    requests = []
    for project, number, person in [("group/alpha", 1, "alice"), ("group/alpha", 2, "alice"),
                                    ("group/alpha", 3, "bob"), ("group/alpha", 4, "charlie"),
                                    ("group/alpha", 5, "dave"), ("group/alpha", 6, "eve"),
                                    ("group/beta", 7, "bob")]:
        requests.append({"url": f"http://fixture.test/{project}/-/merge_requests/{number}",
                         "project_path": project, "number": number, "title": f"Request {number}",
                         "reference": f"{project}!{number}", "author": "Same displayed name",
                         "comment_count": 0, "comments_text": "0",
                         "links": [{"text": "Same displayed name", "url": "http://fixture.test/" + person}]})
    issues = []
    for project, number, person, comments in [("group/alpha", 11, "alice", 99),
                                              ("group/alpha", 20, "bob", 3),
                                              ("group/alpha", 21, "bob", 6),
                                              ("group/alpha", 30, "charlie", 9),
                                              ("group/alpha", 40, "dave", 500),
                                              ("group/alpha", 50, "outsider", 1000),
                                              ("group/beta", 99, "bob", 1000)]:
        issues.append({"url": f"http://fixture.test/{project}/-/issues/{number}",
                       "project_path": project, "number": number, "title": f"Issue {number}",
                       "reference": f"{project}#{number}", "author": "Same displayed name",
                       "comment_count": comments, "comments_text": str(comments),
                       "links": [{"text": "Same displayed name", "url": "http://fixture.test/" + person}]})
    pending = [{"id": "todo_1", "url": requests[1]["url"]},
               {"id": "todo_2", "url": requests[1]["url"]},
               {"id": "todo_other_project", "url": requests[6]["url"]},
               {"id": "todo_issue", "url": issues[2]["url"]},
               {"id": "todo_unknown_mr", "url": "http://fixture.test/group/alpha/-/merge_requests/99"}]
    return case, {"scopes": {
        "project_open": {"rows": requests, "state": "Open", "complete_window": True},
        "assigned_closed": {"rows": issues, "state": "Closed", "complete_window": True,
                            "pages": [{"issues": issues[:2]}, {"issues": issues[2:4]}, {"issues": issues[4:]}]},
        "pending": {"rows": pending, "state": "To Do", "complete_window": True},
        "starrers": {"root": {"url": "http://fixture.test/group/alpha"}, "complete_window": True,
                     "rows": [{"url": "http://fixture.test/" + person} for person in ["alice", "bob", "charlie", "eve"]]}}}


def test_closed_issue_navigation_excludes_entire_pending_creator_and_keeps_only_same_project_relations():
    case, archive = closed_issue_creator_fixture()
    definition, _, _, target = owner._build_definition(case, archive)
    assert target["url"].endswith("/group/alpha/-/issues/21")
    assert definition["answer"]["values"] is None
    relation = archive["navigation_relation"]
    assert [group["creator_username"] for group in relation["ranked_creators"]] == ["@bob", "@charlie"]
    assert relation["ranked_creators"][0]["closed_issue_comment_total"] == 9
    alice = next(group for group in relation["excluded_creators"] if group["creator_username"] == "@alice")
    assert len(alice["requests"]) == 2 and alice["requests"][0]["pending_sources"] == []
    assert alice["reason"] == "at least one same-project open MR has a pending To-Do"
    assert any(group["reason"] == "creator is not a same-project starrer" for group in relation["excluded_creators"])
    assert any(group["reason"] == "creator has no same-project assigned closed issue" for group in relation["excluded_creators"])
    assert len(relation["outside_project_request_sources"]) == 1


def test_closed_issue_navigation_recomputes_creator_rank_and_numeric_issue_tie():
    case, archive = closed_issue_creator_fixture()
    issues = archive["scopes"]["assigned_closed"]["rows"]
    issues[1].update(comment_count=4, comments_text="4")
    issues[2].update(comment_count=4, comments_text="4")
    issues[3].update(comment_count=8, comments_text="8")
    _, _, _, target = owner._build_definition(case, archive)
    assert target["number"] == 21  # creator count breaks sum tie, numeric issue number breaks its badge tie
    issues.append({**deepcopy(issues[3]), "number": 31, "url": "http://fixture.test/group/alpha/-/issues/31",
                   "comment_count": 0, "comments_text": "0"})
    _, _, _, target = owner._build_definition(case, archive)
    assert target["number"] == 21  # equal sum/count, exact @username ascending
    issues[3].update(comment_count=9, comments_text="9")
    _, _, _, target = owner._build_definition(case, archive)
    assert target["number"] == 30  # newly recorded badges change the formal target


@pytest.mark.parametrize("scope", ["project_open", "assigned_closed"])
def test_closed_issue_navigation_does_not_join_ambiguous_creator_profile(scope):
    case, archive = closed_issue_creator_fixture()
    row = archive["scopes"][scope]["rows"][0]
    row["links"].append({"text": row["author"], "url": "http://fixture.test/other"})
    with pytest.raises(ValueError, match="one visible profile link"):
        owner._build_definition(case, archive)


def test_closed_issue_navigation_proves_full_three_page_window_and_resolves_rank_and_exclusion_sources():
    case, archive = closed_issue_creator_fixture()
    _, evidence, _, _ = owner._build_definition(case, archive)
    def resolve(pointer):
        value = archive
        for token in pointer.lstrip("/").split("/"):
            value = value[int(token)] if isinstance(value, list) else value[token]
        return value
    relation = archive["navigation_relation"]
    assert len(resolve(relation["source_windows"]["closed_issues"])["pages"]) == 3
    assert all(resolve(pointer)["complete_window"] for pointer in relation["source_windows"].values())
    assert any(item["location"] == "/navigation_relation" for item in evidence)
    for group in relation["ranked_creators"] + relation["excluded_creators"]:
        for request in group["requests"]:
            assert resolve(request["request_source"])["url"] == request["request_url"]
            assert resolve(request["creator_profile_source"]).endswith(group["creator_profile_path"])
            assert all(resolve(pointer) == request["request_url"] for pointer in request["pending_sources"])
        for issue in group["closed_issues"]:
            assert resolve(issue["issue_source"])["url"] == issue["issue_url"]
            assert int(resolve(issue["comment_badge_source"])) == issue["comment_count"]
            assert resolve(issue["number_source"]) == issue["number"]
            assert resolve(issue["creator_profile_source"]).endswith(group["creator_profile_path"])
    assert all(resolve(item["issue_source"])["project_path"] == "group/beta"
               if item["reason"] == "outside the public project" else True for item in relation["excluded_issues"])


def formal_closed_issue_fixture(monkeypatch, *, status=200, method="GET", terminal_title="Issue 21"):
    case, archived = closed_issue_creator_fixture()
    class Page:
        url = "http://fixture.test/"
        def goto(self, url, **kwargs):
            self.url = url
            return SimpleNamespace(url=url, status=200 if url.endswith(".test/") else status,
                                   request=SimpleNamespace(method=method))
        def wait_for_timeout(self, milliseconds):
            pass
        def locator(self, selector):
            assert selector == "h1.title:visible"
            return SimpleNamespace(all_inner_texts=lambda: [terminal_title])
    monkeypatch.setattr(owner, "_record_extension", lambda page, output, name: {
        "title": "Projects · Dashboard · GitLab" if page.url.endswith(".test/") else terminal_title,
        "url": page.url, "links": []})
    monkeypatch.setattr(owner, "_collect_scope", lambda page, output, initial, descriptor, name, args: deepcopy(archived["scopes"][name]))
    return case, Page()


def test_formal_closed_issue_navigation_binds_actual_get_200_and_recomputed_terminal_issue(tmp_path, monkeypatch):
    case, page = formal_closed_issue_fixture(monkeypatch)
    archive, definition, evidence, checks = owner._formal(
        page, tmp_path, {"scopes": {name: {} for name in case["scopes"]}}, case,
        SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000), {})
    assert archive["terminal_navigation_response"]["response_status"] == 200
    assert definition["parameters"]["final_url"] == "__GITLAB__/group/alpha/-/issues/21"
    assert definition["parameters"]["final_title"] == "Issue 21"
    assert definition["completion_events"][0]["http_method"] == "GET"
    assert checks[1] == {"kind": "visible_text_exact", "parameter": "final_title", "selector": "h1.title:visible"}
    assert {item["location"] for item in evidence} >= {
        "/terminal_navigation_response/http_method", "/terminal_navigation_response/response_status"}


@pytest.mark.parametrize("status,method", [(404, "GET"), (503, "GET"), (200, "POST")])
def test_formal_closed_issue_navigation_retains_actual_failed_response_without_emitting_gt(tmp_path, monkeypatch, status, method):
    case, page = formal_closed_issue_fixture(monkeypatch, status=status, method=method)
    with pytest.raises(RuntimeError, match="actual successful GET 200"):
        owner._formal(page, tmp_path, {"scopes": {name: {} for name in case["scopes"]}}, case,
                      SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000), {})
    response = json.loads((tmp_path / "terminal_navigation_response.json").read_text())
    assert response["response_status"] == status and response["http_method"] == method
    assert json.loads((tmp_path / "navigation_relation.json").read_text())["ranked_creators"][0]["creator_username"] == "@bob"
    assert not (tmp_path / "draft.json").exists()


def test_formal_closed_issue_navigation_requires_one_exact_terminal_title(tmp_path, monkeypatch):
    case, page = formal_closed_issue_fixture(monkeypatch, terminal_title="Wrong issue title")
    with pytest.raises(ValueError, match="terminal title differs"):
        owner._formal(page, tmp_path, {"scopes": {name: {} for name in case["scopes"]}}, case,
                      SimpleNamespace(site_url="http://fixture.test", settle_seconds=0, timeout_ms=120_000), {})
