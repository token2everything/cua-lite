"""Sandbox management transport tests with no platform or Feishu access."""

from __future__ import annotations

import io
import json
import stat
import subprocess
import urllib.error
from unittest.mock import Mock

import pytest

from lite.platforms.sandbox import client as sandbox_client


@pytest.fixture
def client():
    return sandbox_client.SandboxClient(
        base_url="http://manager.example",
        project="test-project",
        afs_url="http://afs.example:28081",
        credentials={"access_key": "test-access", "secret_key": "test-secret"},
    )


@pytest.mark.parametrize(
    "contents",
    [
        "AK=test-access\nSK=test-secret",
        '{"access_key":"test-access","secret_key":"test-secret"}',
        "test-access\ntest-secret",
    ],
)
def test_credentials_parse_without_shell(tmp_path, contents):
    path = tmp_path / "credentials"
    path.write_text(contents)
    assert sandbox_client.load_credentials(path) == {
        "access_key": "test-access",
        "secret_key": "test-secret",
    }


def test_credential_file_cannot_execute_code(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "credentials"
    path.write_text(f"touch {marker}")
    with pytest.raises(ValueError, match="AK and SK"):
        sandbox_client.credentials_from_file(path)
    assert not marker.exists()


def test_write_private_json_replaces_permissive_file(tmp_path):
    path = tmp_path / "connection.json"
    path.write_text("old")
    path.chmod(0o644)
    sandbox_client.write_json(path, {"token": "private-session"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {"token": "private-session"}
    assert list(tmp_path.iterdir()) == [path]


def test_api_auth_and_empty_delete_response(client):
    client.opener = Mock()
    client.opener.open.return_value = io.BytesIO(b"")
    assert client.request("DELETE", "/v1/sandboxes/owned") == {}
    request = client.opener.open.call_args.args[0]
    assert request.full_url == "http://manager.example/v1/sandboxes/owned"
    assert request.get_header("Open-sandbox-api-key") == "test-access:test-secret"
    assert sandbox_client.NoRedirect().redirect_request(None, None, 302, None, None, None) is None


def test_api_error_redacts_credentials_without_retry(client):
    client.opener = Mock()
    client.opener.open.side_effect = urllib.error.HTTPError(
        "http://manager.example", 403, "denied", {}, io.BytesIO(b"test-access:test-secret")
    )
    with pytest.raises(RuntimeError, match="HTTP 403") as caught:
        client.request("POST", "/v1/sandboxes", {})
    assert isinstance(caught.value, sandbox_client.SandboxAPIError)
    assert caught.value.status_code == 403
    assert "test-access" not in str(caught.value)
    assert "test-secret" not in str(caught.value)
    assert client.opener.open.call_count == 1


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_network_timeout_keeps_operation_context_without_retrying(client, method):
    client.opener = Mock()
    client.opener.open.side_effect = TimeoutError("timed out test-secret")
    with pytest.raises(sandbox_client.SandboxTransportError) as caught:
        client.request(method, "/v1/sandboxes/owned")
    assert str(caught.value) == f"{method} /v1/sandboxes/owned: timed out [redacted]"
    assert client.opener.open.call_count == 1


@pytest.mark.parametrize("code", [3, 7])
def test_afs_failure_retains_exit_code_without_leaking_secrets(client, monkeypatch, code):
    monkeypatch.setattr(sandbox_client.subprocess, "run", Mock(side_effect=[
        Mock(stdout="obscured-value\n"),
        subprocess.CalledProcessError(
            code, ["rclone", "copy"], stderr="test-secret obscured-value"
        ),
    ]))
    with pytest.raises(sandbox_client.AFSCommandError) as caught:
        client.afs("copy", "afs:missing", "output")
    assert caught.value.returncode == code
    assert str(caught.value) == "[redacted] [redacted]"


def test_proxy_endpoint_preserves_path_or_uses_manager(client, monkeypatch):
    request = Mock(return_value={"endpoint": "http://10.0.0.2:10100/v1/sandboxes/sbx/proxy/8800"})
    monkeypatch.setattr(client, "request", request)
    assert client.endpoint("sbx", 8800) == "http://manager.example/v1/sandboxes/sbx/proxy/8800"
    request.return_value = {"endpoint": "http://manager.example/custom/proxy/8800/"}
    assert client.endpoint("sbx", 8800) == "http://manager.example/custom/proxy/8800"


def test_afs_configuration_stays_in_child_environment(client, monkeypatch):
    run = Mock(side_effect=[Mock(stdout="obscured-value\n"), Mock(stdout="file data")])
    monkeypatch.setattr(sandbox_client.subprocess, "run", run)
    assert client.afs("cat", "afs:results/control.json") == "file data"
    obscure, transfer = run.call_args_list
    assert obscure.kwargs["input"] == "test-secret"
    assert transfer.args[0] == [
        "rclone",
        "--config",
        "/dev/null",
        "cat",
        "afs:results/control.json",
    ]
    assert transfer.kwargs["env"]["RCLONE_CONFIG_AFS_USER"] == "ak_test-access@test-project"
    assert transfer.kwargs["env"]["RCLONE_CONFIG_AFS_PASS"] == "obscured-value"
    assert "afs.example" in transfer.kwargs["env"]["NO_PROXY"]
    assert "test-secret" not in " ".join(transfer.args[0])


def test_afs_error_is_redacted(client, monkeypatch):
    run = Mock(
        side_effect=[
            Mock(stdout="obscured-value\n"),
            subprocess.CalledProcessError(
                1, ["rclone"], stderr="denied test-access test-secret obscured-value"
            ),
        ]
    )
    monkeypatch.setattr(sandbox_client.subprocess, "run", run)
    with pytest.raises(RuntimeError) as caught:
        client.afs("cat", "afs:results/control.json")
    assert "test-secret" not in str(caught.value)
    assert "test-access" not in str(caught.value)
    assert "obscured-value" not in str(caught.value)


def test_feishu_uses_only_explicit_file_or_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("SANDBOX_FEISHU_WEBHOOK", raising=False)
    monkeypatch.chdir(tmp_path)
    state = tmp_path / ".sandbox"
    state.mkdir()
    hook = "https://open.feishu.cn/open-apis/bot/v2/hook/test-hook"
    private = state / "feishu-webhook.json"
    private.write_text(json.dumps({"webhook_url": hook}))
    assert sandbox_client.feishu_webhook() is None
    monkeypatch.setenv("SANDBOX_FEISHU_WEBHOOK", hook + "-env")
    assert sandbox_client.feishu_webhook() == hook + "-env"
    assert sandbox_client.feishu_webhook(private) == hook
    assert sandbox_client.feishu_webhook(private, disabled=True) is None


@pytest.mark.parametrize(
    "url",
    [
        "http://open.feishu.cn/open-apis/bot/v2/hook/private-hook",
        "https://open.feishu.cn.evil.example/open-apis/bot/v2/hook/private-hook",
        "https://user:password@open.feishu.cn/open-apis/bot/v2/hook/private-hook",
    ],
)
def test_feishu_invalid_destination_never_echoes_secret(url, monkeypatch):
    monkeypatch.setenv("SANDBOX_FEISHU_WEBHOOK", url)
    with pytest.raises(ValueError) as caught:
        sandbox_client.feishu_webhook()
    assert "private-hook" not in str(caught.value)


def test_feishu_message_contains_no_management_credentials(monkeypatch):
    hook = "https://open.feishu.cn/open-apis/bot/v2/hook/test-hook"
    opener = Mock()
    opener.open.return_value = io.BytesIO(b'{"code":0}')
    monkeypatch.setattr(sandbox_client.urllib.request, "build_opener", Mock(return_value=opener))
    assert sandbox_client.feishu_test(hook)["code"] == 0
    request = opener.open.call_args.args[0]
    assert request.get_header("Open-sandbox-api-key") is None
    assert json.loads(request.data)["msg_type"] == "text"
    opener.open.return_value = io.BytesIO(b'{"code":19024,"msg":"rejected test-hook"}')
    with pytest.raises(RuntimeError, match="19024") as caught:
        sandbox_client.feishu_test(hook)
    assert "test-hook" not in str(caught.value)


def test_afs_monitor_timeout_is_bounded_and_does_not_expose_partial_output(client, monkeypatch):
    run = Mock(side_effect=[Mock(stdout="obscured-value\n"), subprocess.TimeoutExpired(
        ["rclone", "cat"], 10, stderr=b"test-secret obscured-value",
    )])
    monkeypatch.setattr(sandbox_client.subprocess, "run", run)
    with pytest.raises(sandbox_client.AFSCommandError, match="timed out") as caught:
        client.afs("cat", "afs:results/progress", timeout=10)
    assert caught.value.returncode is None
    assert 0 < run.call_args.kwargs["timeout"] <= 10
    assert "test-secret" not in str(caught.value)


def test_card_webhook_validates_destination_and_size_before_send(monkeypatch):
    opener = Mock()
    monkeypatch.setattr(sandbox_client.urllib.request, "build_opener", opener)
    with pytest.raises(ValueError, match="HTTPS Feishu"):
        sandbox_client.send_feishu_message("https://example.org/private", {})
    with pytest.raises(ValueError, match="20 KiB"):
        sandbox_client.send_feishu_message(
            "https://open.feishu.cn/open-apis/bot/v2/hook/test-only", {"text": "中" * 20000},
        )
    opener.assert_not_called()
