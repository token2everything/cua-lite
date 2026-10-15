from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import bootstrap_single_site_execution as bootstrap
import pytest
from playwright import sync_api

# The deployed owner's GitLab branch, including its independent page/navigation.
# Only fixture credentials differ; no private login API participates in these tests.
LOGIN_OWNER = '''
class WebArenaInstance:
    def ui_login(self, site, page):
        assert site == "gitlab"
        page = page.context.new_page()
        page.goto("http://localhost:8033/users/sign_in")
        page.get_by_label("Username or email").fill("fixture-user")
        page.get_by_label("Password").fill("fixture-password")
        page.get_by_role("button", name="Sign in").click()
        page.close()
'''


class FakeLocator:
    def __init__(self, page, kind):
        self.page = page
        self.kind = kind

    def count(self):
        return int(self.page.view == "signin")

    def is_visible(self):
        return bool(self.count())

    def fill(self, value):
        if not self.is_visible():
            raise sync_api.TimeoutError(f"get_by_label({self.kind!r}).fill: 120000ms timeout")
        self.page.filled.append(self.kind)

    def click(self):
        assert self.kind == "Sign in"
        assert self.page.filled == ["Username or email", "Password"]
        self.page.context.request("POST", "/users/sign_in")
        effect = self.page.context.login_effects.popleft()
        self.page.context.authenticated = effect in {"authenticated", "authenticated_timeout"}
        if effect == "authenticated_timeout":
            raise sync_api.TimeoutError("login click timed out after authentication")


class FakePage:
    def __init__(self, context):
        self.context = context
        self.url = "about:blank"
        self.view = "blank"
        self.status = None
        self.filled = []
        self.closed = False

    def goto(self, url, **kwargs):
        path = urlsplit(url).path
        self.context.request("GET", path)
        if self.context.navigations:
            expected, status, view = self.context.navigations.popleft()
            assert path == expected
        else:
            assert self.context.unavailable_path == path
            status, view = 502, "bad"
        if isinstance(view, Exception):
            raise view
        if view == "auto":
            view = "dashboard" if self.context.authenticated else "signin"
        self.view = view
        self.status = status
        self.url = "http://localhost:8033" + ("/users/sign_in" if view == "signin" else path)
        return SimpleNamespace(status=status)

    def title(self):
        return {
            "blank": "",
            "signin": "Sign in · GitLab",
            "dashboard": "Projects · Dashboard · GitLab",
            "bad": "GitLab is not responding (502)",
            "unknown": "Unexpected public page",
        }[self.view]

    def get_by_label(self, name, **kwargs):
        assert name in {"Username or email", "Password"}
        return FakeLocator(self, name)

    def get_by_role(self, role, name, **kwargs):
        assert role == "button" and name == "Sign in"
        return FakeLocator(self, name)

    def wait_for_timeout(self, milliseconds):
        self.context.clock += milliseconds / 1000

    def close(self):
        self.closed = True

    def screenshot(self, path):
        Path(path).write_bytes(b"fixture-readiness-screenshot")


class FakeContext:
    def __init__(self, navigations, login_effects=(), unavailable_path=None):
        self.navigations = deque(navigations)
        self.login_effects = deque(login_effects)
        self.unavailable_path = unavailable_path
        self.authenticated = False
        self.clock = 0
        self.requests = []
        self.pages = []
        self.closed = False

    def set_default_timeout(self, timeout):
        assert timeout == 120_000

    def route(self, pattern, callback):
        assert pattern == "**/*"
        self.restrict = callback

    def request(self, method, path):
        self.requests.append((method, path))
        route = SimpleNamespace(
            request=SimpleNamespace(method=method, url="http://localhost:8033" + path),
            continue_=lambda: None,
            abort=lambda: pytest.fail("the real restriction aborted a fixture auth/read request"),
        )
        self.restrict(route)

    def new_page(self):
        page = FakePage(self)
        self.pages.append(page)
        return page

    def storage_state(self):
        assert self.authenticated
        return {"cookies": [{"name": "fixture-auth", "value": "authenticated"}], "origins": []}

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self, context):
        self.context = context
        self.chromium = SimpleNamespace(launch=lambda **kwargs: self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def new_context(self, **kwargs):
        return self.context

    def close(self):
        pass


@pytest.fixture
def execute(tmp_path, monkeypatch):
    initial_umask = os.umask(0o077)
    os.umask(initial_umask)
    for site in ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
        monkeypatch.setenv(f"WA_{site}", "fixture-prior-env")
    task = "wavgen-auth-gitlab-0001"
    baseline = tmp_path / "baselines" / task / "author-01.json"
    login_owner = tmp_path / "instance.py"
    login_owner.write_text(LOGIN_OWNER)
    cid = "a" * 64
    image_id = "sha256:" + "b" * 64
    image = "fixture-gitlab@" + image_id
    identity = {
        "container_id": cid,
        "container_name": f"/wav-v03-gitlab-author-{task}-01",
        "image": image,
        "image_id": image_id,
        "run_id": "auth_fixture",
        "session_nonce": "auth_nonce",
        "site": "gitlab",
        "generated_task_id": task,
        "model_profile": "author",
        "attempt": "01",
    }
    inspection = tmp_path / "inspection.json"
    inspection.write_text(json.dumps({
        **identity,
        "created_at": "fixture-time",
        "running": True,
        "mounts": [],
        "ports": {
            "8033/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8033"}],
            "8877/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8877"}],
        },
    }))
    author = tmp_path / "author-baseline.json"
    author.write_text(json.dumps({
        "image_id": image_id, "image_digest": image_id, "snapshot": "fixture-fixed-seed",
    }))
    desktop = Path("/root/cua-lite/scripts/webarena_verified_qwen.py")
    launcher = Path(bootstrap.__file__).with_name("run_single_site_case.sh")
    args = SimpleNamespace(
        **{key: value for key, value in identity.items() if key != "container_name"},
        inspection=inspection,
        author_baseline=author,
        baseline_receipt=baseline,
        storage_state=tmp_path / "private" / "state.json",
        site_urls=tmp_path / "site_urls.json",
        site_url="http://localhost:8033",
        site_port=8033,
        health_url="http://localhost:8877/status",
        health_port=8877,
        nonce="auth_nonce",
        sandbox_id="auth_fixture_sandbox",
        login_owner=login_owner,
        launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
        desktop_helpers_sha256=hashlib.sha256(desktop.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(bootstrap, "_parse_args", lambda: args)

    class Health:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"success"

    monkeypatch.setattr(bootstrap, "urlopen", lambda *args, **kwargs: Health())

    def run(context):
        monkeypatch.setattr(sync_api, "sync_playwright", lambda: FakeRuntime(context))
        monkeypatch.setattr(bootstrap, "time", SimpleNamespace(monotonic=lambda: context.clock))
        bootstrap.main()

    yield run, args
    os.umask(initial_umask)


@pytest.mark.parametrize("matching_pair", [True, False])
def test_real_bootstrap_checks_its_actual_adjacent_epoch_launcher(execute, tmp_path, matching_pair):
    _, args = execute
    pair = tmp_path / "api-epoch"
    pair.mkdir()
    source = Path(bootstrap.__file__)
    copied = pair / source.name
    shutil.copyfile(source, copied)
    launcher = pair / "run_single_site_case.sh"
    shutil.copyfile(source.with_name(launcher.name), launcher)
    if not matching_pair:
        launcher.write_text(launcher.read_text() + "\n# Different paired launcher bytes\n")
    spec = importlib.util.spec_from_file_location("actual_paired_bootstrap", copied)
    paired = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paired)
    paired._parse_args = lambda: args

    class ReachedScopedHealth(Exception):
        pass

    def reached_health(*_args, **_kwargs):
        raise ReachedScopedHealth

    paired.urlopen = reached_health
    expected = ReachedScopedHealth if matching_pair else ValueError
    match = None if matching_pair else "runtime launcher differs"
    with pytest.raises(expected, match=match):
        paired.main()
    bindings = args.baseline_receipt.parent.parent / "container_bindings" / (args.container_id + ".json")
    assert bindings.exists() is matching_pair
    assert not args.baseline_receipt.exists()
    assert copied.read_bytes() == source.read_bytes()


def read_auth(args, suffix):
    return json.loads(args.baseline_receipt.with_suffix(suffix).read_text())


class FakeMapPage(FakePage):
    def goto(self, url, **kwargs):
        path = urlsplit(url).path
        self.context.request("GET", path)
        self.context.navigation_options.append(kwargs)
        if self.context.navigations:
            expected, status, view = self.context.navigations.popleft()
            assert path == expected == "/"
        else:
            status, view = 503, "bad"
        if isinstance(view, Exception):
            if isinstance(view, sync_api.TimeoutError):
                self.context.clock += kwargs["timeout"] / 1000
            raise view
        self.view, self.status = view, status
        self.url = url + "login" if view == "redirected_map" else url
        if kwargs["wait_until"] == "networkidle" and view == "map":
            raise sync_api.TimeoutError("Map tiles never reached networkidle")
        return SimpleNamespace(status=status)

    def title(self):
        return {"blank": "", "map": "OpenStreetMap", "bad": "503 Service Unavailable",
                "unknown": "Unexpected public page", "hidden_log_in": "OpenStreetMap",
                "duplicate_log_in": "OpenStreetMap", "redirected_map": "OpenStreetMap"}[self.view]

    def locator(self, selector):
        assert selector == "body"
        return SimpleNamespace(inner_text=lambda: "OpenStreetMap Log In" if self.view in {
            "map", "hidden_log_in", "duplicate_log_in", "redirected_map"
        } else self.title())

    def get_by_role(self, role, name, **kwargs):
        assert role == "link" and name == "Log In" and kwargs == {"exact": True}
        return SimpleNamespace(
            count=lambda: 2 if self.view == "duplicate_log_in" else int(self.view in {
                "map", "hidden_log_in", "redirected_map"
            }),
            is_visible=lambda: self.view in {"map", "duplicate_log_in", "redirected_map"},
        )


class FakeMapContext(FakeContext):
    def __init__(self, navigations):
        super().__init__(navigations)
        self.navigation_options = []

    def request(self, method, path):
        assert method == "GET" and path == "/"
        self.requests.append((method, path))
        self.restrict(SimpleNamespace(
            request=SimpleNamespace(method=method, url="http://localhost:8080" + path),
            continue_=lambda: None,
            abort=lambda: pytest.fail("anonymous public Map GET was rejected"),
        ))

    def new_page(self):
        page = FakeMapPage(self)
        self.pages.append(page)
        return page

    def storage_state(self):
        return {"cookies": [], "origins": []}


@pytest.fixture
def execute_map(execute):
    run, args = execute
    args.site = "map"
    args.site_url, args.site_port = "http://localhost:8080", 8080
    args.health_url, args.health_port = "http://localhost:8081/status", 8081
    inspection = json.loads(args.inspection.read_text())
    inspection.update(site="map", container_name=f"/wav-v03-map-author-{args.generated_task_id}-01",
                      ports={"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8080"}],
                             "8877/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8081"}]})
    root = f"/var/lib/wav-v03/map-executions/{args.generated_task_id}/author-01"
    inspection["mounts"] = [
        {"Destination": target, "Source": root + "/" + name, "RW": True}
        for target, name in {"/data/database": "tile-db", "/data/nominatim/postgres": "nominatim-db",
                             "/var/lib/postgresql/14/main": "website-db", "/data/tiles": "tiles",
                             "/data/style": "style"}.items()
    ] + [{"Destination": target, "Source": "/immutable" + target, "RW": False}
         for target in ("/entrypoint.sh", "/patched-entrypoint.sh", "/mnt/map-source/nominatim-flatnode",
                        "/mnt/map-source/routing-car", "/mnt/map-source/routing-bike",
                        "/mnt/map-source/routing-foot")]
    args.inspection.write_text(json.dumps(inspection))
    return run, args


def test_map_controller_ready_before_rails_preserves_503_and_waits_only_public_get(execute_map):
    run, args = execute_map
    context = FakeMapContext([("/", 503, "bad"), ("/", 502, "bad"), ("/", 200, "map")])
    run(context)
    proof = read_auth(args, ".auth_probes.json")
    assert [row["http_status"] for row in proof["probes"]] == [503, 502, 200]
    assert proof["policy"] == "public_anonymous" and proof["model_started"] is False
    assert context.requests == [("GET", "/")] * 3
    assert args.baseline_receipt.exists()
    assert json.loads(args.storage_state.read_text()) == {"cookies": [], "origins": []}
    assert all(row["wait_until"] == "domcontentloaded" for row in context.navigation_options)


def test_map_loaded_document_does_not_wait_for_outstanding_tiles(execute_map):
    run, args = execute_map
    context = FakeMapContext([("/", 200, "map")])
    run(context)
    assert read_auth(args, ".auth_probes.json")["probes"][0]["authentication_confirmed"]
    assert context.navigation_options == [{"wait_until": "domcontentloaded", "timeout": 15_000}]
    assert context.requests == [("GET", "/")]


def test_map_navigation_timeout_is_recorded_before_a_new_actual_response(execute_map):
    run, args = execute_map
    context = FakeMapContext([
        ("/", None, sync_api.TimeoutError("document did not load")),
        ("/", 503, "bad"), ("/", 200, "map"),
    ])
    run(context)
    probes = read_auth(args, ".auth_probes.json")["probes"]
    assert [row["http_status"] for row in probes] == [None, 503, 200]
    assert [row["authentication_confirmed"] for row in probes] == [False, False, True]
    assert probes[0]["navigation_error"] == "document did not load"
    assert context.requests == [("GET", "/")] * 3


def test_map_navigation_timeouts_are_bounded_and_never_invent_http_success(execute_map):
    run, args = execute_map
    context = FakeMapContext([
        ("/", None, sync_api.TimeoutError("document did not load")) for _ in range(8)
    ])
    with pytest.raises(RuntimeError, match="public landing"):
        run(context)
    failure = read_auth(args, ".auth_failure.json")
    assert context.clock == 120
    assert all(row["http_status"] is None for row in failure["probes"])
    assert all(not row["authentication_confirmed"] for row in failure["probes"])
    assert context.navigation_options[-1]["timeout"] < 15_000
    assert not args.baseline_receipt.exists() and not args.storage_state.exists()


@pytest.mark.parametrize("view", ["hidden_log_in", "duplicate_log_in", "redirected_map"])
def test_map_http_200_title_and_body_do_not_replace_visible_anonymous_landing(execute_map, view):
    run, args = execute_map
    context = FakeMapContext([("/", 200, view)])
    with pytest.raises(RuntimeError, match="public landing"):
        run(context)
    assert context.requests == [("GET", "/")]
    assert not args.baseline_receipt.exists() and not args.storage_state.exists()


@pytest.mark.parametrize("view", ["bad", "unknown"])
def test_map_unavailable_or_wrong_public_ui_never_creates_baseline(execute_map, view):
    run, args = execute_map
    context = FakeMapContext([("/", 503 if view == "bad" else 200, view)])
    with pytest.raises(RuntimeError, match="public landing"):
        run(context)
    failure = read_auth(args, ".auth_failure.json")
    assert failure["model_started"] is False and failure["business_assets_changed"] is False
    assert not args.baseline_receipt.exists() and not args.storage_state.exists()
    assert all(method == "GET" and path == "/" for method, path in context.requests)
    assert len(context.requests) == (61 if view == "bad" else 1)


def test_map_unknown_navigation_failure_is_not_retried(execute_map):
    run, args = execute_map
    original = sync_api.Error("unexpected Map transport failure")
    context = FakeMapContext([("/", 200, original)])
    with pytest.raises(sync_api.Error) as raised:
        run(context)
    assert raised.value is original and context.requests == [("GET", "/")]
    assert not args.baseline_receipt.exists()


def assert_failed_pre_model(args, context, attempts):
    failure = read_auth(args, ".auth_failure.json")
    assert failure["model_started"] is False
    assert failure["business_assets_changed"] is False
    assert len(failure["login_attempts"]) == attempts
    assert not args.baseline_receipt.exists()
    assert not args.storage_state.exists()
    assert not args.site_urls.exists()
    assert context.closed
    assert {path for method, path in context.requests if method == "POST"} <= {"/users/sign_in"}
    return failure


def test_post_login_502_then_signin_requires_new_ready_and_root_before_second_login(execute):
    run, args = execute
    context = FakeContext([
        ("/users/sign_in", 200, "signin"), ("/users/sign_in", 200, "signin"),
        ("/", 502, "bad"), ("/", 503, "bad"), ("/", 200, "auto"),
        ("/users/sign_in", 200, "signin"), ("/", 200, "auto"),
        ("/users/sign_in", 200, "signin"), ("/", 200, "auto"),
    ], ["unauthenticated", "authenticated"])
    run(context)
    report = read_auth(args, ".auth_probes.json")
    assert [row["http_status"] for row in report["probes"]] == [502, 503, 200, 200, 200]
    assert report["probes"][3]["phase"] == "before_retry"
    assert report["probes"][3]["unauthenticated_sign_in_confirmed"]
    assert [row["login_attempt"] for row in report["readiness_probes"]] == [1, 2]
    assert [row["authentication_confirmed"] for row in report["login_attempts"]] == [False, True]
    assert args.baseline_receipt.exists() and args.storage_state.exists()
    assert context.requests.count(("POST", "/users/sign_in")) == 2


def test_owner_navigation_race_timeout_without_post_can_retry_standard_owner(execute):
    run, args = execute
    context = FakeContext([
        ("/users/sign_in", 200, "signin"), ("/users/sign_in", 503, "bad"),
        ("/", 200, "auto"), ("/users/sign_in", 200, "signin"),
        ("/", 200, "auto"), ("/users/sign_in", 200, "signin"), ("/", 200, "auto"),
    ], ["authenticated"])
    run(context)
    attempts = read_auth(args, ".auth_probes.json")["login_attempts"]
    assert attempts[0]["outcome"] == "timeout"
    assert attempts[0]["error_type"] == "TimeoutError"
    assert attempts[0]["requests"] == [{"method": "GET", "path": "/users/sign_in", "allowed": True}]
    assert attempts[1]["authentication_confirmed"]
    assert context.requests.count(("POST", "/users/sign_in")) == 1


@pytest.mark.parametrize("effect", ["authenticated", "authenticated_timeout"])
def test_confirmed_authenticated_landing_never_repeats_login(execute, effect):
    run, args = execute
    context = FakeContext([
        ("/users/sign_in", 200, "signin"), ("/users/sign_in", 200, "signin"),
        ("/", 502, "bad"), ("/", 200, "auto"),
    ], [effect])
    run(context)
    report = read_auth(args, ".auth_probes.json")
    assert len(report["login_attempts"]) == 1
    assert report["login_attempts"][0]["authentication_confirmed"]
    assert len(report["readiness_probes"]) == 1
    assert context.requests.count(("POST", "/users/sign_in")) == 1


@pytest.mark.parametrize("second_owner_status", [200, 503])
def test_second_failure_retains_both_attempts_and_no_model_state(execute, second_owner_status):
    run, args = execute
    context = FakeContext([
        ("/users/sign_in", 200, "signin"), ("/users/sign_in", 503, "bad"),
        ("/", 200, "auto"), ("/users/sign_in", 200, "signin"), ("/", 200, "auto"),
        ("/users/sign_in", second_owner_status, "signin" if second_owner_status == 200 else "bad"),
        ("/", 200, "auto"),
    ], ["unauthenticated"] if second_owner_status == 200 else [])
    expected = RuntimeError if second_owner_status == 200 else sync_api.TimeoutError
    with pytest.raises(expected):
        run(context)
    failure = assert_failed_pre_model(args, context, 2)
    assert failure["login_attempts"][0]["error_type"] == "TimeoutError"
    if second_owner_status == 503:
        assert failure["error_type"] == "TimeoutError"
        assert failure["error"] == failure["login_attempts"][1]["error"]
    assert not context.navigations


def test_signin_never_ready_preserves_readonly_probes_and_skips_login(execute):
    run, args = execute
    context = FakeContext([], unavailable_path="/users/sign_in")
    with pytest.raises(RuntimeError, match="did not render"):
        run(context)
    failure = assert_failed_pre_model(args, context, 0)
    assert failure["readiness_probes"]
    assert all(row["http_status"] == 502 for row in failure["readiness_probes"])
    assert all(method == "GET" for method, path in context.requests)
    assert args.baseline_receipt.with_suffix(".readiness_failure.png").exists()


def test_unavailable_root_does_not_authorize_second_login(execute):
    run, args = execute
    context = FakeContext([
        ("/users/sign_in", 200, "signin"), ("/users/sign_in", 200, "signin"),
    ], ["unauthenticated"], unavailable_path="/")
    with pytest.raises(RuntimeError, match="did not match"):
        run(context)
    failure = assert_failed_pre_model(args, context, 1)
    assert all(row["http_status"] == 502 for row in failure["probes"])
    assert context.requests.count(("POST", "/users/sign_in")) == 1


@pytest.mark.parametrize("where", ["readiness", "owner"])
def test_unknown_playwright_exception_is_preserved_without_retry(execute, where):
    run, args = execute
    error = sync_api.Error("unexpected browser protocol failure")
    navigations = [("/users/sign_in", 200, "signin")] if where == "owner" else []
    navigations.append(("/users/sign_in", None, error))
    context = FakeContext(navigations)
    with pytest.raises(sync_api.Error) as raised:
        run(context)
    assert raised.value is error
    failure = assert_failed_pre_model(args, context, int(where == "owner"))
    assert failure["error"] == str(error)
    assert not context.navigations
    if where == "owner":
        assert failure["login_attempts"][0]["outcome"] == "error"
    else:
        assert failure["readiness_probes"][0]["navigation_error"] == str(error)


def test_unknown_public_landing_error_is_recorded_and_never_replayed(execute):
    run, args = execute
    error = sync_api.Error("unexpected landing navigation failure")
    context = FakeContext([
        ("/users/sign_in", 200, "signin"), ("/users/sign_in", 200, "signin"),
        ("/", None, error),
    ], ["unauthenticated"])
    with pytest.raises(sync_api.Error) as raised:
        run(context)
    assert raised.value is error
    failure = assert_failed_pre_model(args, context, 1)
    assert failure["probes"][0]["probe_error"] == str(error)
    assert context.requests.count(("POST", "/users/sign_in")) == 1


def test_second_ready_form_without_unauthenticated_public_root_cannot_repeat_login(execute):
    run, args = execute
    context = FakeContext([
        ("/users/sign_in", 200, "signin"), ("/users/sign_in", 200, "signin"),
        ("/", 200, "auto"), ("/users/sign_in", 200, "signin"), ("/", 200, "unknown"),
    ], ["unauthenticated"])
    with pytest.raises(RuntimeError, match="did not prove"):
        run(context)
    failure = assert_failed_pre_model(args, context, 1)
    assert failure["probes"][-1]["phase"] == "before_retry"
    assert not failure["probes"][-1]["unauthenticated_sign_in_confirmed"]
    assert context.requests.count(("POST", "/users/sign_in")) == 1
