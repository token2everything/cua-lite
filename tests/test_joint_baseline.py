from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import author_baseline
import bootstrap_execution
import pytest
import requests
import worker
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError


class AuthenticationContext:
    def __init__(self, declared_sites, page_outcomes, rejected_logins, login_outcomes):
        self.declared_sites = declared_sites
        self.page_outcomes = page_outcomes
        self.rejected_logins = rejected_logins
        self.login_outcomes = login_outcomes
        self.authenticated = set()
        self.confirmed_landings = set()
        self.login_contexts = []
        self.login_sites = []
        self.landings = []
        self.state_snapshots = []
        self.waits = []
        self.closed = False
        self.pages = []
        self.page_listeners = []
        self.login_pages = []

    def login(self, site, page):
        self.login_sites.append(site)
        self.login_contexts.append(id(self))
        self.login_pages.append(page)
        outcomes = self.login_outcomes.get(site, [])
        if outcomes:
            responses = outcomes.pop(0)
            if responses is not None:
                for navigation in responses:
                    page.emit_response(**navigation)
                raise PlaywrightTimeoutError("Username locator timed out")
        page.emit_response(
            200,
            url=("http://gitlab.localhost:8033/users/sign_in" if site == "gitlab"
                 else "http://shopping.localhost:7760/customer/account/login/"),
        )
        if site not in self.rejected_logins:
            self.authenticated.add(site)

    def set_default_timeout(self, timeout):
        pass

    def route(self, pattern, callback):
        self.restrict = callback

    def on(self, event, callback):
        assert event == "page"
        self.page_listeners.append(callback)

    def remove_listener(self, event, callback):
        assert event == "page"
        self.page_listeners.remove(callback)

    def new_page(self):
        context = self

        class Page:
            url = "about:blank"

            def __init__(self):
                self.main_frame = object()
                self.response_listener = None
                self.closed = False

            def on(self, event, callback):
                assert event == "response"
                self.response_listener = callback

            def emit_response(self, status, url="http://gitlab.localhost:8033/users/sign_in",
                              navigation=True, main_frame=True):
                if navigation and main_frame:
                    self.url = url
                request = SimpleNamespace(
                    method="GET",
                    is_navigation_request=lambda: navigation,
                    frame=self.main_frame if main_frame else object(),
                )
                if self.response_listener:
                    self.response_listener(SimpleNamespace(url=url, status=status, request=request))

            def goto(self, url, **kwargs):
                self.url = url
                endpoint = urlsplit(url)
                site = endpoint.hostname.split(".", 1)[0]
                status = 200
                if endpoint.path == "/":
                    outcomes = context.page_outcomes.get(site, [])
                    if outcomes:
                        status, path = outcomes.pop(0)
                    elif site not in context.authenticated:
                        path = "/users/sign_in" if site == "gitlab" else "/customer/account/login/"
                    else:
                        path = "/"
                    self.url = f"{endpoint.scheme}://{endpoint.netloc}{path}"
                    if path in {"/users/sign_in", "/customer/account/login/"}:
                        context.authenticated.discard(site)
                    if status == 200 and path == "/" and site in context.authenticated:
                        context.confirmed_landings.add(site)
                    context.landings.append((url, set(context.authenticated)))
                return SimpleNamespace(status=status)

            def wait_for_timeout(self, duration):
                context.waits.append(duration)

            def title(self):
                return (
                    "Sign in · GitLab"
                    if self.url.endswith("/users/sign_in")
                    else "Projects · Dashboard · GitLab"
                )

            def get_by_label(self, *args, **kwargs):
                return SimpleNamespace(count=lambda: 1, is_visible=lambda: True)

            def get_by_role(self, *args, **kwargs):
                return SimpleNamespace(is_visible=lambda: "shopping" in context.authenticated)

            def close(self):
                self.closed = True

            def is_closed(self):
                return self.closed

        page = Page()
        page.context = context
        self.pages.append(page)
        for callback in self.page_listeners:
            callback(page)
        return page

    def storage_state(self):
        assert self.authenticated == self.confirmed_landings == self.declared_sites
        self.state_snapshots.append(set(self.confirmed_landings))
        return {
            "cookies": [
                {"name": "session", "value": "test", "domain": f"{site}.localhost"}
                for site in sorted(self.authenticated)
            ],
            "origins": [],
        }

    def close(self):
        self.closed = True


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    sites = ["shopping", "gitlab"]
    manifest = {"sites": {}}
    prepared = {"sites": {}}
    for index, site in enumerate(sites):
        site_port, health_port, inside_port = bootstrap_execution.SITE_PORTS[site]
        image = f"registry.test/{site}@sha256:{str(index + 1) * 64}"
        image_id = f"sha256:{str(index + 3) * 64}"
        prepared["sites"][site] = {"image": image, "image_id": image_id, "repo_digests": [image]}
        manifest["sites"][site] = {
            "inspection": {
                "container_id": str(index + 5) * 64,
                "container_name": f"/wav-v04-{site}-author-wavgen-joint-0001-01",
                "created_at": f"2026-10-12T00:00:0{index}Z",
                "image": image,
                "image_id": image_id,
                "run_id": "run-test",
                "session_nonce": "nonce-test",
                "site": site,
                "generated_task_id": "wavgen-joint-0001",
                "model_profile": "author",
                "attempt": "01",
                "running": True,
                "ports": {
                    f"{inside_port}/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(site_port)}],
                    "8877/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(health_port)}],
                },
            },
            "site_url": f"http://{site}.localhost:{site_port}",
            "site_port": site_port,
            "health_url": f"http://localhost:{health_port}/status",
            "health_port": health_port,
        }
    manifest_path = tmp_path / "manifest.json"
    prepared_path = tmp_path / "prepared-assets.json"
    manifest_path.write_text(json.dumps(manifest))
    prepared_path.write_text(json.dumps(prepared))
    launcher = tmp_path / "run_case.sh"
    launcher.write_bytes(Path(bootstrap_execution.__file__).with_name("run_case.sh").read_bytes())
    producer = tmp_path / "bootstrap_execution.py"
    producer.write_bytes(Path(bootstrap_execution.__file__).read_bytes())
    helper = tmp_path / "helpers.py"
    helper.write_text("# Frozen desktop helper fixture\n")
    entrypoint = tmp_path / "run_desktop_reference.py"
    entrypoint.write_text("# Frozen entrypoint fixture\n")
    (tmp_path / "image-lock.env").write_text("# Frozen image lock fixture\n")
    login = tmp_path / "login_owner.py"
    login.write_text(
        "class WebArenaInstance:\n"
        "    def ui_login(self, site, page):\n"
        "        page = page.context.new_page()\n"
        "        page.context.login(site, page)\n"
        "        page.close()\n"
    )
    monkeypatch.setattr(bootstrap_execution, "__file__", str(producer))
    monkeypatch.setattr(bootstrap_execution, "DESKTOP_HELPERS", helper)
    monkeypatch.setattr(bootstrap_execution, "DESKTOP_ENTRYPOINT", entrypoint)
    monkeypatch.setattr(bootstrap_execution.os, "umask", lambda value: 0)
    for name in ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
        monkeypatch.setenv(f"WA_{name}", "http://fixture.test")
    contexts = []
    health_calls = []
    page_outcomes = {}
    rejected_logins = set()
    login_outcomes = {}

    class Browser:
        def new_context(self, **kwargs):
            declared_sites = set(json.loads(manifest_path.read_text())["sites"])
            context = AuthenticationContext(
                declared_sites, page_outcomes, rejected_logins, login_outcomes
            )
            contexts.append(context)
            return context

        def close(self):
            pass

    class Runtime:
        chromium = SimpleNamespace(launch=lambda **kwargs: Browser())

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Health:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"success":true}'

    def urlopen(url, **kwargs):
        health_calls.append(url)
        return Health()

    monkeypatch.setattr("playwright.sync_api.sync_playwright", Runtime)
    monkeypatch.setattr(bootstrap_execution, "urlopen", urlopen)
    baseline = tmp_path / "baselines/task/author-01.json"
    storage = tmp_path / "private/task/author-01.json"
    site_urls = tmp_path / "baselines/task/author-01-site-urls.json"
    args = [
        "--manifest",
        str(manifest_path),
        "--prepared-assets",
        str(prepared_path),
        "--baseline-receipt",
        str(baseline),
        "--storage-state",
        str(storage),
        "--site-urls",
        str(site_urls),
        "--run-id",
        "run-test",
        "--nonce",
        "nonce-test",
        "--sandbox-id",
        "sandbox-test",
        "--generated-task-id",
        "wavgen-joint-0001",
        "--model-profile",
        "author",
        "--attempt",
        "01",
        "--launcher-sha256",
        bootstrap_execution.hashlib.sha256(launcher.read_bytes()).hexdigest(),
        "--desktop-helpers-sha256",
        bootstrap_execution.hashlib.sha256(helper.read_bytes()).hexdigest(),
        "--login-owner",
        str(login),
    ]
    return SimpleNamespace(
        args=args,
        manifest=manifest,
        manifest_path=manifest_path,
        prepared=prepared,
        prepared_path=prepared_path,
        baseline=baseline,
        storage=storage,
        site_urls=site_urls,
        contexts=contexts,
        health_calls=health_calls,
        page_outcomes=page_outcomes,
        rejected_logins=rejected_logins,
        login_outcomes=login_outcomes,
    )


@pytest.mark.parametrize("sites", [["shopping"], ["shopping", "gitlab"]])
def test_joint_authentication_binds_every_fresh_site_in_one_context(preparation, sites, tmp_path):
    case = preparation
    case.manifest_path.write_text(
        json.dumps({"sites": {site: case.manifest["sites"][site] for site in sites}})
    )
    case.prepared_path.write_text(
        json.dumps({"sites": {site: case.prepared["sites"][site] for site in sites}})
    )
    bootstrap_execution.main(case.args)
    receipt = json.loads(case.baseline.read_text())
    assert set(receipt["sites"]) == set(sites)
    assert len(case.contexts) == 1
    context = case.contexts[0]
    assert context.closed
    assert context.login_sites == sites
    assert len([page for page in context.pages if page.url == "about:blank"]) == len(sites)
    assert all(page.is_closed() for page in context.pages)
    assert not context.page_listeners
    probes = json.loads(case.baseline.with_suffix(".auth_probes.json").read_text())["probes"]
    navigations = [probe for probe in probes if probe["phase"] == "login_navigation"]
    assert [probe["site"] for probe in navigations] == sites
    assert all(probe["http_status"] == 200 for probe in navigations)
    assert not any(probe["phase"] == "authentication_retry" for probe in probes)
    assert len(set(context.login_contexts)) == 1
    assert all(authenticated == set(sites) for _, authenticated in context.landings)
    assert set(json.loads(case.site_urls.read_text())) == set(sites)
    assert {cookie["domain"] for cookie in json.loads(case.storage.read_text())["cookies"]} == {
        f"{site}.localhost" for site in sites
    }
    assert len(case.health_calls) == len(sites)
    author = tmp_path / "author-baseline.json"
    author_baseline.main(["--execution-baseline", str(case.baseline), "--output", str(author)])
    projection = json.loads(author.read_text())
    assert projection["fresh_from_image"] is True
    assert projection["snapshot"] == "webarena-verified-1.2.3-seed"
    assert projection["sites"] == receipt["sites"]
    assert "container_id" not in projection
    assert "image_digest" not in projection
    with pytest.raises(FileExistsError, match="earlier attempt"):
        bootstrap_execution.main(case.args)
    assert len(case.contexts) == 1


@pytest.mark.parametrize(
    "corruption",
    ["foreign_nonce", "wrong_image", "shared_cid", "wrong_hostname", "missing_prepared_site"],
)
def test_joint_baseline_rejects_scope_or_asset_mismatch_before_auth(preparation, corruption):
    case = preparation
    declared = case.manifest["sites"]["gitlab"]
    if corruption == "foreign_nonce":
        declared["inspection"]["session_nonce"] = "co-tenant"
    elif corruption == "wrong_image":
        declared["inspection"]["image_id"] = "sha256:" + "9" * 64
    elif corruption == "shared_cid":
        declared["inspection"]["container_id"] = case.manifest["sites"]["shopping"]["inspection"][
            "container_id"
        ]
    elif corruption == "wrong_hostname":
        declared["site_url"] = "http://shopping.localhost:8033"
    elif corruption == "missing_prepared_site":
        del case.prepared["sites"]["gitlab"]
    case.manifest_path.write_text(json.dumps(case.manifest))
    case.prepared_path.write_text(json.dumps(case.prepared))
    with pytest.raises(ValueError):
        bootstrap_execution.main(case.args)
    assert not case.contexts
    assert not case.health_calls
    assert not case.baseline.exists()


def test_joint_authentication_routes_only_declared_site_origins(preparation):
    bootstrap_execution.main(preparation.args)
    decisions = []
    for url in (
        "http://shopping.localhost:7760/customer/account/login/",
        "http://gitlab.localhost:8033/users/sign_in",
        "http://gitlab.localhost:7760/",
        "http://unrelated.test/",
    ):
        route = SimpleNamespace(
            request=SimpleNamespace(url=url, method="POST"),
            continue_=lambda: decisions.append("allowed"),
            abort=lambda: decisions.append("blocked"),
        )
        preparation.contexts[0].restrict(route)
    assert decisions == ["allowed", "allowed", "blocked", "blocked"]


@pytest.mark.parametrize("gateway_status", [502, 503])
def test_gateway_failure_then_logged_out_allows_one_standard_authentication_retry(
    preparation, gateway_status
):
    case = preparation
    case.page_outcomes["gitlab"] = [
        (gateway_status, "/"),
        (gateway_status, "/"),
        (200, "/users/sign_in"),
        (200, "/"),
    ]
    bootstrap_execution.main(case.args)
    assert len(case.contexts) == 1
    context = case.contexts[0]
    assert context.login_sites == ["shopping", "gitlab", "gitlab"]
    assert len(set(context.login_contexts)) == 1
    assert context.state_snapshots == [{"shopping", "gitlab"}]
    assert context.waits == [2000, 2000]
    probes = json.loads(case.baseline.with_suffix(".auth_probes.json").read_text())
    retries = [probe for probe in probes["probes"] if probe["phase"] == "authentication_retry"]
    assert len(retries) == 1
    assert retries[0]["reason"] == "gateway_5xx_then_logged_out"
    assert retries[0]["model_started"] is False
    assert retries[0]["business_assets_changed"] is False
    confirmed = {
        probe["site"] for probe in probes["probes"] if probe.get("authentication_confirmed")
    }
    assert confirmed == {"shopping", "gitlab"}
    state = json.loads(case.storage.read_text())
    assert {cookie["domain"] for cookie in state["cookies"]} == {
        "shopping.localhost",
        "gitlab.localhost",
    }
    assert not case.baseline.with_suffix(".auth_failure.json").exists()


def test_incorrect_credentials_without_gateway_failure_are_not_retried(preparation):
    case = preparation
    case.rejected_logins.add("gitlab")
    with pytest.raises(RuntimeError, match="standard gitlab authentication"):
        bootstrap_execution.main(case.args)
    assert len(case.contexts) == 1
    context = case.contexts[0]
    assert context.login_sites == ["shopping", "gitlab"]
    assert context.state_snapshots == [] and context.closed
    assert not case.storage.exists()
    assert not case.baseline.exists()
    assert not case.site_urls.exists()
    failure = json.loads(case.baseline.with_suffix(".auth_failure.json").read_text())
    assert all(probe["phase"] != "authentication_retry" for probe in failure["probes"])
    assert all(probe.get("http_status") not in {502, 503} for probe in failure["probes"])
    assert failure["model_started"] is False and failure["business_assets_changed"] is False


def test_gateway_recovery_to_authenticated_landing_does_not_repeat_login(preparation):
    case = preparation
    case.page_outcomes["gitlab"] = [(502, "/"), (200, "/")]
    bootstrap_execution.main(case.args)
    context = case.contexts[0]
    assert context.login_sites == ["shopping", "gitlab"]
    assert context.state_snapshots == [{"shopping", "gitlab"}]
    probes = json.loads(case.baseline.with_suffix(".auth_probes.json").read_text())
    assert all(probe["phase"] != "authentication_retry" for probe in probes["probes"])


def test_standard_authentication_retry_is_bounded_to_two_attempts(preparation):
    case = preparation
    case.page_outcomes["gitlab"] = [
        (502, "/"),
        (200, "/users/sign_in"),
        (503, "/"),
        (200, "/users/sign_in"),
    ]
    with pytest.raises(RuntimeError, match="standard gitlab authentication"):
        bootstrap_execution.main(case.args)
    context = case.contexts[0]
    assert context.login_sites.count("gitlab") == 2
    assert len(set(context.login_contexts)) == 1
    assert context.state_snapshots == []
    assert not case.storage.exists() and not case.baseline.exists()
    failure = json.loads(case.baseline.with_suffix(".auth_failure.json").read_text())
    retries = [probe for probe in failure["probes"] if probe["phase"] == "authentication_retry"]
    assert len(retries) == 1


@pytest.mark.parametrize("gateway_status", [502, 503])
def test_login_navigation_gateway_timeout_recovers_with_failure_evidence(preparation, gateway_status):
    case = preparation
    case.login_outcomes["gitlab"] = [[{"status": gateway_status}]]
    bootstrap_execution.main(case.args)
    context = case.contexts[0]
    assert context.login_sites == ["shopping", "gitlab", "gitlab"]
    assert len(set(context.login_contexts)) == 1
    assert context.state_snapshots == [{"shopping", "gitlab"}]
    probes = json.loads(case.baseline.with_suffix(".auth_probes.json").read_text())["probes"]
    readiness = [probe for probe in probes if probe["phase"] == "login_readiness"]
    assert len(readiness) == 2
    assert all(probe["http_status"] == 200 for probe in readiness)
    assert all(probe["title"] == "Sign in · GitLab" for probe in readiness)
    navigation = next(
        probe for probe in probes
        if probe["phase"] == "login_navigation" and probe["site"] == "gitlab"
    )
    assert navigation["http_status"] == gateway_status
    assert all(page.is_closed() for page in context.login_pages)
    assert not context.page_listeners
    assert navigation["same_origin"] is True
    failures = [probe for probe in probes if probe["phase"] == "login_failure"]
    assert len(failures) == 1 and failures[0]["retry"] is True
    assert failures[0]["error_type"] == "TimeoutError"
    assert failures[0]["authentication_attempt"] == 1
    retries = [probe for probe in probes if probe["phase"] == "authentication_retry"]
    assert [probe["reason"] for probe in retries] == ["login_navigation_5xx_then_timeout"]


@pytest.mark.parametrize("responses", [
    [],
    [{"status": 200}],
    [{"status": 401}],
    [{"status": 403}],
    [{"status": 502, "navigation": False}],
    [{"status": 503, "main_frame": False}],
    [{"status": 502, "url": "http://shopping.localhost:7760/customer/account/login/"}],
    [{"status": 502}, {"status": 200}],
    [{"status": 502}, {"status": 200, "url": "http://foreign.test/users/sign_in"}],
])
def test_login_timeout_without_current_same_origin_gateway_does_not_retry(preparation, responses):
    case = preparation
    case.login_outcomes["gitlab"] = [responses]
    with pytest.raises(PlaywrightTimeoutError):
        bootstrap_execution.main(case.args)
    assert case.contexts[0].login_sites == ["shopping", "gitlab"]
    assert not case.storage.exists() and not case.baseline.exists() and not case.site_urls.exists()
    failure = json.loads(case.baseline.with_suffix(".auth_failure.json").read_text())
    assert not any(probe["phase"] == "authentication_retry" for probe in failure["probes"])
    assert failure["probes"][-1]["retry"] is False
    assert all(page.is_closed() for page in case.contexts[0].login_pages)
    assert not case.contexts[0].page_listeners


def test_second_login_gateway_timeout_preserves_both_failures(preparation):
    case = preparation
    case.login_outcomes["gitlab"] = [[{"status": 502}], [{"status": 503}]]
    with pytest.raises(PlaywrightTimeoutError):
        bootstrap_execution.main(case.args)
    assert case.contexts[0].login_sites == ["shopping", "gitlab", "gitlab"]
    assert not case.storage.exists() and not case.baseline.exists() and not case.site_urls.exists()
    probes = json.loads(case.baseline.with_suffix(".auth_failure.json").read_text())["probes"]
    failures = [probe for probe in probes if probe["phase"] == "login_failure"]
    assert [probe["authentication_attempt"] for probe in failures] == [1, 2]
    assert [probe["retry"] for probe in failures] == [True, False]


def test_login_and_landing_recovery_share_two_call_budget(preparation):
    case = preparation
    case.login_outcomes["gitlab"] = [[{"status": 502}]]
    case.page_outcomes["gitlab"] = [(503, "/"), (200, "/users/sign_in")]
    with pytest.raises(RuntimeError, match="standard gitlab authentication"):
        bootstrap_execution.main(case.args)
    assert case.contexts[0].login_sites == ["shopping", "gitlab", "gitlab"]
    assert not case.storage.exists() and not case.baseline.exists()
    probes = json.loads(case.baseline.with_suffix(".auth_failure.json").read_text())["probes"]
    assert len([probe for probe in probes if probe["phase"] == "authentication_retry"]) == 1


def test_landing_retry_login_timeout_cannot_restart_budget(preparation):
    case = preparation
    case.page_outcomes["gitlab"] = [(502, "/"), (200, "/users/sign_in")]
    case.login_outcomes["gitlab"] = [None, [{"status": 503}]]
    with pytest.raises(PlaywrightTimeoutError):
        bootstrap_execution.main(case.args)
    assert case.contexts[0].login_sites == ["shopping", "gitlab", "gitlab"]
    assert not case.storage.exists() and not case.baseline.exists()


def test_worker_sites_cli_rejects_duplicates_before_reading_credentials(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["worker.py", "--root", "/unused", "--sites", '["shopping", "shopping"]', "create"],
    )
    monkeypatch.setattr(
        worker.module, "_read_env", lambda *args: pytest.fail("credentials were read")
    )
    with pytest.raises(SystemExit):
        worker.main()


@pytest.fixture
def worker_execution(tmp_path, monkeypatch):
    root = tmp_path / "worker-run"
    (root / "plans").mkdir(parents=True)
    (root / "plans/pipeline.xml").write_text("<pipeline/>\n")
    monkeypatch.setattr(
        "sys.argv",
        ["worker.py", "--root", str(root), "--sites", '["shopping", "gitlab"]', "create"],
    )
    monkeypatch.setattr(
        worker.module,
        "_read_env",
        lambda *args: {"OPEN_SANDBOX_SECRET_KEY": "fake-secret", "OPENAI_API_KEY": "fake-key"},
    )
    return root


def test_worker_projects_declared_sites_into_string_metadata(worker_execution, monkeypatch, capsys):
    submissions = []

    class Client:
        project = "test-project"

        def __init__(self, *args, **kwargs):
            pass

        def create(self, payload):
            assert all(isinstance(value, str) for value in payload["metadata"].values())
            submissions.append(payload)
            return {"id": "owned-worker"}

        def detail(self, sandbox_id):
            assert sandbox_id == "owned-worker"
            return {"metadata": submissions[0]["metadata"], "state": "Running"}

    monkeypatch.setattr(worker.module, "SandboxClient", Client)
    worker.main()
    assert json.loads(submissions[0]["metadata"]["sites"]) == ["shopping", "gitlab"]
    status = json.loads(capsys.readouterr().out)
    assert status["sites"] == ["shopping", "gitlab"]
    record = worker_execution / "infra/workers/execution.json"
    assert json.loads(record.read_text())["id"] == "owned-worker"
    intent = worker_execution / "infra/workers/execution.intent.json"
    assert json.loads(intent.read_text())["submission_status"] == "accepted"


@pytest.mark.parametrize("http_status", [422, 503, None])
def test_worker_archives_creation_errors_without_a_second_post(
    worker_execution, monkeypatch, http_status
):
    submissions = []
    listings = []

    class Client:
        project = "test-project"

        def __init__(self, *args, **kwargs):
            pass

        def create(self, payload):
            submissions.append(payload)
            if http_status is None:
                raise requests.Timeout("response lost")
            response = requests.Response()
            response.status_code = http_status
            response._content = b'{"error":"metadata must be strings; fake-secret fake-key"}'
            raise requests.HTTPError("creation failed: fake-secret", response=response)

        def list_sandboxes(self):
            listings.append(True)
            return []

    monkeypatch.setattr(worker.module, "SandboxClient", Client)
    with pytest.raises(RuntimeError, match="inspect execution.create-response.json"):
        worker.main()
    state = worker_execution / "infra/workers"
    response = json.loads((state / "execution.create-response.json").read_text())
    expected = "rejected" if http_status == 422 else "unknown"
    assert response["submission_status"] == expected
    assert response["http_status"] == http_status
    assert "fake-secret" not in json.dumps(response)
    assert "fake-key" not in json.dumps(response)
    if http_status is not None:
        assert "metadata must be strings" in response["response_body"]
    with pytest.raises(RuntimeError, match="another POST"):
        worker.main()
    assert len(submissions) == 1
    assert len(listings) == (0 if expected == "rejected" else 1)
    assert not (state / "execution.json").exists()


def test_worker_reconciles_an_accepted_creation_with_a_lost_response(worker_execution, monkeypatch):
    submissions = []

    class Client:
        project = "test-project"

        def __init__(self, *args, **kwargs):
            pass

        def create(self, payload):
            submissions.append(payload)
            raise requests.Timeout("accepted but response lost")

        def list_sandboxes(self):
            return [{"id": "original-worker", "metadata": submissions[0]["metadata"]}]

        def detail(self, sandbox_id):
            assert sandbox_id == "original-worker"
            return {"metadata": submissions[0]["metadata"], "state": "Running"}

    monkeypatch.setattr(worker.module, "SandboxClient", Client)
    with pytest.raises(RuntimeError, match="creation unknown"):
        worker.main()
    worker.main()
    assert len(submissions) == 1
    record = worker_execution / "infra/workers/execution.json"
    assert json.loads(record.read_text())["id"] == "original-worker"


@pytest.mark.parametrize("entry", ["prepare.sh", "run_case.sh"])
@pytest.mark.parametrize(
    ("sites_json", "sites"),
    [
        ('["shopping"]', ["shopping"]),
        (' [ "gitlab", "shopping" ] ', ["gitlab", "shopping"]),
        ('[\n"shopping",\t"gitlab"\n]', ["shopping", "gitlab"]),
    ],
)
def test_entrypoint_parses_declared_sites_without_host_executables(
    tmp_path, entry, sites_json, sites
):
    source = Path(bootstrap_execution.__file__).with_name(entry).read_text()
    # Exercise the actual executable ingress before its resource lifecycle begins.
    ingress = source.split("\nROOT=", 1)[0] + '\nprintf "%s\\n" "${SITES[@]}"\n'
    script = tmp_path / entry
    script.write_text(ingress)
    args = [sites_json]
    if entry == "run_case.sh":
        args.extend(
            ["author", "wavgen-test-0001", "run-test", "nonce-test", "sandbox-test", "01", "a" * 64]
        )
    result = subprocess.run(
        ["/bin/bash", str(script), *args],
        env={"PATH": str(tmp_path / "no-executables")},
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == sites


@pytest.mark.parametrize("entry", ["prepare.sh", "run_case.sh"])
@pytest.mark.parametrize(
    "sites_json",
    [
        "shopping",
        "[]",
        '["shopping","shopping"]',
        '["shopp ing"]',
        '["shopping", "gitlab", "reddit"]',
        '["shopping", 1]',
        '{"sites":["shopping","gitlab"]}',
        '["shopping"]; echo unsafe',
    ],
)
def test_entrypoint_rejects_noncanonical_sites_before_host_actions(tmp_path, entry, sites_json):
    source = Path(bootstrap_execution.__file__).with_name(entry).read_text()
    script = tmp_path / entry
    script.write_text(source.split("\nROOT=", 1)[0])
    args = [sites_json]
    if entry == "run_case.sh":
        args.extend(
            ["author", "wavgen-test-0001", "run-test", "nonce-test", "sandbox-test", "01", "a" * 64]
        )
    result = subprocess.run(
        ["/bin/bash", str(script), *args],
        env={"PATH": str(tmp_path / "no-executables")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Declare" in result.stderr
    assert "command not found" not in result.stderr


@pytest.mark.parametrize("profile", ["teacher", "student", "author"])
@pytest.mark.parametrize("student_model", ["z-ai/glm-5.3-flash", "configured-student"])
def test_case_uses_owned_runtime_profile_model_without_exposing_credentials(
    tmp_path, profile, student_model
):
    from lite.agents.extensions.api_config import load_profile

    root = Path(__file__).resolve().parents[1]
    models = json.loads((root / "runtime/models.example.json").read_text())
    assert models["models"]["student"]["model"] == "z-ai/glm-5.3-flash"
    assert models["models"]["student"]["expected_returned_model"] == "glm-5.3-flash"
    models["models"]["student"]["model"] = student_model
    for spec in models["models"].values():
        spec["api_key"] = "secret-must-not-print"
        spec["base_url"] = "https://synthetic.test"
    config = tmp_path / "models.json"
    config.write_text(json.dumps(models))
    (tmp_path / "image-lock.env").write_text("")
    source = (root / "infra/run_case.sh").read_text()
    assert '--model-id "$MODEL_ID"' in source
    assert "glm-5.3-flash" not in source and "qwen3.8-max" not in source
    script = tmp_path / "run_case.sh"
    script.write_text(
        source.split('\nEXECUTION_DIR=', 1)[0].replace('/mnt/results/v04', str(tmp_path))
        + '\nprintf "%s\\n%s\\n" "${MODEL_ID-unread}" "$OUTPUT_ROOT"\n'
    )
    commands = tmp_path / "bin"
    commands.mkdir()
    docker = commands / "docker"
    docker.write_text(
        "#!" + sys.executable + "\n"
        "import os, subprocess, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['FAKE_DOCKER_LOG'], 'a') as log:\n"
        "    log.write(sys.argv[1] + '\\n')\n"
        "if sys.argv[1] == 'inspect':\n"
        "    field = sys.argv[3]\n"
        "    print('nonce-test' if 'session-nonce' in field else "
        "'run-test' if 'run-id' in field else 'true')\n"
        "else:\n"
        "    assert sys.argv[1:5] == "
        "['exec', 'wav-v04-runtime', '/opt/venv/bin/python', '-B']\n"
        "    assert sys.argv[7] == '/private/models_qwen_v2.json'\n"
        "    subprocess.run([sys.executable, *sys.argv[4:7], "
        "os.environ['FAKE_MODELS'], sys.argv[8]], check=True)\n"
    )
    docker.chmod(0o700)
    log = tmp_path / "docker.log"
    if profile == "author":
        config.unlink()
    result = subprocess.run(
        ["/bin/bash", str(script), '["shopping","gitlab"]', profile,
         "wavgen-test-0001", "run-test", "nonce-test", "sandbox-test", "01", "a" * 64],
        env={**os.environ, "PATH": str(commands), "FAKE_DOCKER_LOG": str(log),
             "FAKE_MODELS": str(config), "PYTHONDONTWRITEBYTECODE": "1"},
        check=True, capture_output=True, text=True,
    )
    expected = "unread" if profile == "author" else load_profile(config, profile)["model"]
    assert result.stdout.splitlines()[0] == expected
    assert "secret-must-not-print" not in result.stdout + result.stderr
    assert log.read_text().splitlines() == ["inspect"] * 3 + ([] if profile == "author" else ["exec"])


def test_prepared_assets_are_built_with_owned_runtime_python_only(tmp_path):
    root = tmp_path / "infra"
    host = tmp_path / "host"
    commands = tmp_path / "bin"
    root.mkdir()
    commands.mkdir()
    for executable in ("mkdir", "sha256sum", "cat", "tar", "gzip"):
        (commands / executable).symlink_to(shutil.which(executable))
    images = {
        site: f"registry.test/{site}@sha256:{str(index + 1) * 64}"
        for index, site in enumerate(["shopping", "gitlab"])
    }
    runtime_image_id = "sha256:" + "7" * 64
    (root / "image-lock.env").write_text(
        "\n".join(f'{site.upper()}_IMAGE="{image}"' for site, image in images.items())
        + '\nQWENRUNNER_RUNTIME_IMAGE="registry.test/runtime@sha256:'
        + "8" * 64
        + '"\n'
    )
    (root / "runtime-image-id.txt").write_text(runtime_image_id)
    (root / "prepared-sites.json").write_text(json.dumps(list(images), separators=(",", ":")))
    for index, (site, image) in enumerate(images.items()):
        (root / f"{site}-image-inspection.json").write_text(
            json.dumps({"Id": "sha256:" + str(index + 3) * 64, "RepoDigests": [image]})
        )
    archive = root / "source-runtime.tar.gz"
    with tarfile.open(archive, "w:gz"):
        pass
    script = root / "start_model_runtime.sh"
    source = Path(bootstrap_execution.__file__).with_name("start_model_runtime.sh").read_text()
    script.write_text(
        source.replace("/mnt/results/v04", str(root)).replace("/var/lib/wav-v04", str(host))
    )
    docker = commands / "docker"
    docker.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
  image) printf '%s\\n' "$FAKE_RUNTIME_IMAGE_ID" ;;
  inspect)
    if [[ "$2" = --format && -f "$FAKE_CREATED" ]]; then printf '{}\\n'; else exit 1; fi ;;
  run) : > "$FAKE_CREATED"; printf 'owned-runtime-cid\\n' ;;
  exec)
    shift 2
    while [[ "$1" = -e ]]; do export "$2"; shift 2; done
    [[ "$1" = owned-runtime-cid && "$2" = /opt/venv/bin/python && "$3" = -B ]]
    shift 2
    argv=()
    for argument in "$@"; do
      if [[ "$argument" = /infra ]]; then argv+=("$FAKE_INFRA"); else argv+=("$argument"); fi
    done
    exec "$FAKE_CONTAINER_PYTHON" "${argv[@]}" ;;
  *) exit 1 ;;
esac
"""
    )
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    subprocess.run(
        [
            "/bin/bash",
            str(script),
            "run-test",
            "nonce-test",
            str(archive),
            hashlib.sha256(archive.read_bytes()).hexdigest(),
        ],
        env={
            **os.environ,
            "PATH": str(commands),
            "FAKE_DOCKER_LOG": str(log),
            "FAKE_RUNTIME_IMAGE_ID": runtime_image_id,
            "FAKE_CREATED": str(tmp_path / "created"),
            "FAKE_INFRA": str(root),
            "FAKE_CONTAINER_PYTHON": sys.executable,
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assets = json.loads((root / "prepared-assets.json").read_text())["sites"]
    assert set(assets) == set(images)
    assert {site: asset["image"] for site, asset in assets.items()} == images
    assert "exec -i -e SHOPPING_IMAGE=" in log.read_text()
    assert "owned-runtime-cid /opt/venv/bin/python -B - /infra" in log.read_text()
    assert "--add-host shopping.localhost:127.0.0.1" in log.read_text()
    assert "--add-host gitlab.localhost:127.0.0.1" in log.read_text()
    assert not (commands / "python3").exists()
    assert not (commands / "jq").exists()
