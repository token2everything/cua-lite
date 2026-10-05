"""Unit tests for BrowserGymServices.health() — the per-benchmark readiness probe
that makes `GET /envs/browsergym.<bench>` report the truth.
Mocks the HTTP probes; no docker.

Run: uv run pytest tests/gym/envs/browsergym/test_browsergym_health.py -q
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")
from lite.gym.envs.browsergym import main as bg
from lite.gym.errors import EnvDepsMissingError


@pytest.fixture
def svc():
    return bg.BrowserGymServices()


def test_health_passes_when_stack_up(svc, monkeypatch):
    monkeypatch.setattr(bg, "_service_up", lambda b: True)
    monkeypatch.setattr(bg, "_homepage_up", lambda: True)
    monkeypatch.setattr(bg, "_classifieds_up", lambda: True)
    svc.health("browsergym.webarena")  # no raise
    svc.health("browsergym.visualwebarena")
    svc.health("browsergym.miniwob")


def test_health_raises_when_stack_cold(svc, monkeypatch):
    monkeypatch.setattr(bg, "_service_up", lambda b: False)
    with pytest.raises(EnvDepsMissingError, match="webarena"):
        svc.health("browsergym.webarena")


def test_health_vwa_needs_classifieds(svc, monkeypatch):
    # WA stack up but classifieds down ⇒ visualwebarena not healthy, webarena is.
    monkeypatch.setattr(bg, "_service_up", lambda b: True)
    monkeypatch.setattr(bg, "_homepage_up", lambda: True)
    monkeypatch.setattr(bg, "_classifieds_up", lambda: False)
    svc.health("browsergym.webarena")  # WA fine
    with pytest.raises(EnvDepsMissingError, match="visualwebarena"):
        svc.health("browsergym.visualwebarena")


def test_health_per_leaf(svc, monkeypatch):
    # webarena up, visualwebarena down (its classifieds missing) — only VWA fails.
    monkeypatch.setattr(bg, "_service_up", lambda b: True)
    monkeypatch.setattr(bg, "_homepage_up", lambda: True)
    monkeypatch.setattr(bg, "_classifieds_up", lambda: False)
    svc.health("browsergym.webarena")
    with pytest.raises(EnvDepsMissingError):
        svc.health("browsergym.visualwebarena")


def test_health_raises_when_homepage_down(svc, monkeypatch):
    # I3's needle probe is a health conjunct of its own: services can be "up"
    # (ports listening) while the WA homepage serves an error page.
    monkeypatch.setattr(bg, "_service_up", lambda b: True)
    monkeypatch.setattr(bg, "_classifieds_up", lambda: True)
    monkeypatch.setattr(bg, "_homepage_up", lambda: False)
    with pytest.raises(EnvDepsMissingError, match="webarena"):
        svc.health("browsergym.webarena")


def test_health_bare_umbrella_is_noop(svc, monkeypatch):
    # Bare umbrella has no single backend to probe — must not raise even if cold.
    monkeypatch.setattr(bg, "_service_up", lambda b: False)
    svc.health("browsergym")  # no raise


def test_health_never_boots(svc, monkeypatch):
    # health must be probe-only — assert it never calls the boot path.
    monkeypatch.setattr(bg, "_service_up", lambda b: True)
    monkeypatch.setattr(bg, "_homepage_up", lambda: True)
    monkeypatch.setattr(bg, "_classifieds_up", lambda: True)
    def _boom(*a, **k):
        raise AssertionError("health() must not boot (_ensure_services called)")
    monkeypatch.setattr(bg, "_ensure_services", _boom)
    svc.health("browsergym.webarena")


# --------------------------------------------------------------------------- #
# all-sites readiness: _service_up(WA/VWA) gates on EVERY local site, not just
# shopping — so available:true can't lie while gitlab is still booting.
# --------------------------------------------------------------------------- #

def _only_gitlab_down(url, needle, ok_codes):
    # Resolve the gitlab port the SAME way _wa_stack_up does (env override → default)
    # so the match holds even when GITLAB_PORT is exported in the runner's env.
    import os
    gitlab_port = os.environ.get("GITLAB_PORT", str(bg._GITLAB_PORT_DEFAULT))
    return f":{gitlab_port}/" not in url  # every site up except gitlab


def test_service_up_false_when_only_gitlab_down(monkeypatch):
    monkeypatch.setattr(bg, "_wa_site_up", _only_gitlab_down)
    assert bg._service_up("webarena") is False  # the whole stack is "not ready"


def test_service_up_true_when_all_sites_up(monkeypatch):
    monkeypatch.setattr(bg, "_wa_site_up", lambda url, needle, ok: True)
    assert bg._service_up("webarena") is True


def test_down_sites_names_gitlab(monkeypatch):
    monkeypatch.setattr(bg, "_wa_site_up", _only_gitlab_down)
    assert bg._wa_down_sites("webarena") == ["gitlab"]


def test_vwa_down_sites_includes_classifieds(monkeypatch):
    monkeypatch.setattr(bg, "_wa_site_up", lambda url, needle, ok: True)  # shared sites up
    monkeypatch.setattr(bg, "_classifieds_up", lambda: False)            # classifieds down
    assert bg._wa_down_sites("visualwebarena") == ["classifieds"]
    # webarena (no classifieds dependency) is fully up in the same state
    assert bg._wa_down_sites("webarena") == []


def test_verified_requires_operator_urls_without_legacy_defaults():
    required = set(bg._REQUIRED_ENV_VARS["webarena_verified"])
    assert bg._DEFAULT_ENV_VARS["webarena_verified"] == {}
    assert required == {
        "WA_SHOPPING",
        "WA_SHOPPING_ADMIN",
        "WA_REDDIT",
        "WA_GITLAB",
        "WA_WIKIPEDIA",
        "WA_MAP",
        "WA_HOMEPAGE",
    }
    assert "OPENAI_API_KEY" not in required


def test_verified_missing_urls_fail_without_populating_legacy_defaults(monkeypatch):
    for name in bg._REQUIRED_ENV_VARS["webarena_verified"]:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(OSError, match="WA_SHOPPING"):
        bg._check_env_vars("webarena_verified")

    assert "WA_SHOPPING" not in os.environ


def test_verified_probe_uses_all_operator_urls(monkeypatch):
    expected_urls = set()
    for site, env_var, _needle in bg._WEBARENA_VERIFIED_SITE_VARS:
        url = f"https://verified.example/{site}"
        monkeypatch.setenv(env_var, url)
        expected_urls.add(url)
    seen = []
    monkeypatch.setattr(
        bg,
        "_wa_site_up",
        lambda url, needle, ok_codes, headers: seen.append((url, headers)) or True,
    )

    assert bg._webarena_verified_down_sites() == []
    assert {url for url, _headers in seen} == expected_urls


def test_verified_probe_forwards_playwright_headers(tmp_path, monkeypatch):
    headers_path = tmp_path / "headers.json"
    headers_path.write_text('{"x-site-token": "secret"}')
    monkeypatch.setenv("PW_EXTRA_HEADERS", str(headers_path))
    for site, env_var, _needle in bg._WEBARENA_VERIFIED_SITE_VARS:
        monkeypatch.setenv(env_var, f"https://verified.example/{site}")
    seen = []
    monkeypatch.setattr(
        bg,
        "_wa_site_up",
        lambda url, needle, ok_codes, headers: seen.append(headers) or True,
    )

    assert bg._webarena_verified_down_sites() == []
    assert seen and all(headers == {"x-site-token": "secret"} for headers in seen)


def test_verified_ensure_never_invokes_legacy_start_script(monkeypatch):
    for site, env_var, _needle in bg._WEBARENA_VERIFIED_SITE_VARS:
        monkeypatch.setenv(env_var, f"https://verified.example/{site}")
    monkeypatch.setattr(bg, "_webarena_verified_down_sites", lambda: [])
    monkeypatch.setattr(
        bg.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Verified must not invoke start.sh")
        ),
    )
    bg._services_started.discard("webarena_verified")

    bg._ensure_services("browsergym.webarena_verified")

    assert "webarena_verified" in bg._services_started
    bg._services_started.discard("webarena_verified")
