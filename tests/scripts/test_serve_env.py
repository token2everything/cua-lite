from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from lite.gym.remote import serve
from lite.gym.remote.alive import SERVER_KEEP_ALIVE_TIMEOUT_SEC

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "serve_env.py"


@pytest.mark.parametrize(
    "entrypoint",
    [(str(SCRIPT),), ("-m", "lite.gym.remote.serve")],
    ids=["script", "module"],
)
def test_serve_env_help_documents_operator_flags(entrypoint) -> None:
    proc = subprocess.run(
        [sys.executable, "-B", *entrypoint, "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--max-live-envs" in proc.stdout
    assert "--warm-singleton" in proc.stdout
    assert "--timeout-keep-alive" in proc.stdout
    assert "--reset-concurrency" not in proc.stdout


def test_serve_env_cli_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["serve_env"])
    assert vars(serve._parse_args()) == {
        "host": "0.0.0.0",
        "port": 30100,
        "max_live_envs": None,
        "idle_ttl_sec": 600.0,
        "token": None,
        "admin_token": None,
        "env_ids": None,
        "warm_singleton": False,
        "timeout_keep_alive": None,
    }


@pytest.mark.parametrize(
    ("explicit_timeout", "expected_timeout"),
    [(None, SERVER_KEEP_ALIVE_TIMEOUT_SEC), (123.0, 123.0)],
)
def test_serve_env_builds_real_state_without_default_blocks(
    monkeypatch, explicit_timeout, expected_timeout
) -> None:
    captured: dict = {}

    monkeypatch.setattr(serve, "cached_host_capacity", lambda: object())
    monkeypatch.setattr(serve, "derive_admission_config", lambda _host, **_kwargs: object())
    monkeypatch.setattr(serve, "AdmissionGate", lambda cfg: ("gate", cfg))
    monkeypatch.setattr(serve, "log_admission_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(serve, "log_keep_alive_timeout", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("CUA_LITE_RESET_CONCURRENCY", "4")

    def fake_make_app(state, **kwargs):
        captured["state"] = state
        captured["auth"] = kwargs
        return object()

    def fake_run(_app, **kwargs):
        captured["server"] = kwargs

    monkeypatch.setattr(serve, "make_app", fake_make_app)
    monkeypatch.setattr(serve.uvicorn, "run", fake_run)

    serve.main(
        argparse.Namespace(
            host="127.0.0.1",
            port=30100,
            max_live_envs=None,
            idle_ttl_sec=600.0,
            token="token",
            admin_token=None,
            env_ids=None,
            warm_singleton=False,
            timeout_keep_alive=explicit_timeout,
        )
    )

    state = captured["state"]
    assert isinstance(state, serve.State)
    assert state.blocked_env_ids == frozenset()
    assert state.allowed_env_ids is None
    assert state.reset_concurrency == 4
    assert captured["auth"] == {"token": "token", "admin_token": None, "port": 30100}
    assert captured["server"]["host"] == "127.0.0.1"
    assert captured["server"]["port"] == 30100
    assert captured["server"]["timeout_keep_alive"] == expected_timeout
