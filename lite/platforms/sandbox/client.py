"""Sandbox System management transport, private state, and optional Feishu hooks.

Credentials come from an explicitly supplied file or OPEN_SANDBOX_ACCESS_KEY /
OPEN_SANDBOX_SECRET_KEY. This module has no benchmark or local dataset dependency.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward management or Worker credentials to a redirect destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class SandboxAPIError(RuntimeError):
    """An HTTP failure whose status can distinguish absence from access/transport errors."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class AFSCommandError(RuntimeError):
    """A redacted rclone failure retaining its exit code for missing-artifact handling."""

    def __init__(self, message: str, returncode: int | None):
        super().__init__(message)
        self.returncode = returncode


def safe_component(value: str) -> str:
    """Validate IDs before putting them in an API or local state path."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Expected a simple identifier, not a path")
    return value


def redact(value: Any, secrets=()) -> Any:
    """Remove credential fields and known private values from public output."""
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            sensitive = normalized in {
                "ak",
                "sk",
                "headers",
                "authorization",
                "credentials",
                "cookie",
                "webhookurl",
                "webhookfeishu",
            } or normalized.endswith(
                ("apikey", "accesskey", "secretkey", "password", "token", "secret")
            )
            output[key] = "[redacted]" if sensitive else redact(item, secrets)
        return output
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
    return value


def write_json(path: Path, value: Any) -> None:
    """Atomically publish private JSON; temporary and final files use mode 0600."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix="." + path.name + "-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def credentials_from_file(path: Path) -> dict[str, str]:
    """Read JSON, AK/SK assignments, or two raw lines without evaluating shell text."""
    text = Path(path).expanduser().read_text().strip()
    aliases = {
        "ak": "access_key",
        "accesskey": "access_key",
        "accesskeyid": "access_key",
        "opensandboxaccesskey": "access_key",
        "sk": "secret_key",
        "secretkey": "secret_key",
        "secretaccesskey": "secret_key",
        "opensandboxsecretkey": "secret_key",
    }
    values = {}
    try:
        source = json.loads(text)
    except json.JSONDecodeError:
        source = None
    if isinstance(source, dict):
        for key, value in source.items():
            name = aliases.get(re.sub(r"[^a-z]", "", str(key).lower()))
            if name and isinstance(value, str):
                values[name] = value.strip()
    else:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for line in lines:
            match = re.fullmatch(r"(?:export\s+)?([A-Za-z_ -]+)\s*[:=：]\s*(.+)", line)
            if match:
                name = aliases.get(re.sub(r"[^a-z]", "", match[1].lower()))
                if name:
                    values[name] = match[2].strip().strip("\"'")
        if (
            not values
            and len(lines) == 2
            and all(re.fullmatch(r"[A-Za-z0-9_./+=-]+", line) for line in lines)
        ):
            values = dict(zip(("access_key", "secret_key"), lines))
    if set(values) != {"access_key", "secret_key"} or not all(values.values()):
        raise ValueError("Credential file must contain AK and SK (JSON, assignments, or two lines)")
    return values


def load_credentials(path: Path | None = None) -> dict[str, str]:
    """Resolve explicit credentials without searching other repositories or state directories."""
    if path is not None:
        return credentials_from_file(path)
    values = {
        key: os.environ.get(env, "").strip()
        for key, env in (
            ("access_key", "OPEN_SANDBOX_ACCESS_KEY"),
            ("secret_key", "OPEN_SANDBOX_SECRET_KEY"),
        )
    }
    if not all(values.values()):
        raise ValueError(
            "Pass --credentials or set OPEN_SANDBOX_ACCESS_KEY and OPEN_SANDBOX_SECRET_KEY"
        )
    return values


def state_of(info: dict) -> str:
    """Normalize the platform's top-level or nested Sandbox status."""
    status = info.get("status")
    state = info.get("state") or (status.get("state") if isinstance(status, dict) else status)
    return str(state or "unknown").lower()


class SandboxClient:
    """Small synchronous management/AFS client; it never creates resources on construction."""

    def __init__(
        self, *, base_url: str, project: str, afs_url: str, credentials: dict[str, str], secrets=()
    ):
        origin = urllib.parse.urlsplit(base_url)
        if (
            origin.scheme not in ("http", "https")
            or not origin.netloc
            or origin.username
            or origin.password
            or origin.path not in ("", "/")
            or origin.query
            or origin.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) API origin without /v1")
        afs = urllib.parse.urlsplit(afs_url)
        if (
            afs.scheme not in ("http", "https")
            or not afs.netloc
            or afs.username
            or afs.password
            or afs.query
            or afs.fragment
        ):
            raise ValueError("afs_url must be an HTTP(S) WebDAV URL without credentials")
        self.base_url, self.afs_url = base_url.rstrip("/"), afs_url.rstrip("/")
        self.project = safe_component(project)
        self.credentials = credentials
        self.secrets = [*credentials.values(), *secrets]
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, method: str, path: str, payload=None) -> dict:
        """Call the internal API without proxying credentials or retrying mutations."""
        headers = {
            "OPEN-SANDBOX-API-KEY": (
                f"{self.credentials['access_key']}:{self.credentials['secret_key']}"
            ),
            "Accept": "application/json",
        }
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data, headers, method=method)
        try:
            with self.opener.open(request, timeout=60) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as error:
            detail = redact(error.read(2048).decode(errors="replace"), self.secrets)
            raise SandboxAPIError(
                f"{method} {path}: HTTP {error.code}: {detail}", error.code
            ) from None
        except (OSError, ValueError) as error:
            raise RuntimeError(redact(str(error), self.secrets)) from None

    def afs(self, *command: str) -> str:
        """Run rclone with an in-memory WebDAV configuration, never an on-disk secret."""
        obscured = None
        try:
            obscured = subprocess.run(
                ["rclone", "obscure", "-"],
                input=self.credentials["secret_key"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            env = os.environ.copy()
            no_proxy = ",".join(
                filter(
                    None,
                    [
                        env.get("NO_PROXY"),
                        env.get("no_proxy"),
                        urllib.parse.urlsplit(self.afs_url).hostname,
                        "127.0.0.1",
                        "localhost",
                    ],
                )
            )
            env.update(
                RCLONE_CONFIG_AFS_TYPE="webdav",
                RCLONE_CONFIG_AFS_URL=self.afs_url,
                RCLONE_CONFIG_AFS_VENDOR="other",
                RCLONE_CONFIG_AFS_USER=f"ak_{self.credentials['access_key']}@{self.project}",
                RCLONE_CONFIG_AFS_PASS=obscured,
                NO_PROXY=no_proxy,
                no_proxy=no_proxy,
            )
            for attempt in range(2):
                try:
                    return subprocess.run(
                        ["rclone", "--config", "/dev/null", *command],
                        env=env,
                        check=True,
                        text=True,
                        capture_output=True,
                    ).stdout
                except subprocess.CalledProcessError as error:
                    if attempt or "401 Unauthorized" not in (error.stderr or ""):
                        raise
                    time.sleep(1)
        except (OSError, subprocess.CalledProcessError) as error:
            message = getattr(error, "stderr", None) or str(error)
            raise AFSCommandError(
                redact(message[-3000:], [*self.secrets, obscured]),
                getattr(error, "returncode", None),
            ) from None

    def endpoint(self, sandbox_id: str, port: int) -> str:
        """Use the full proxy endpoint, mapping advertised internal hosts to the manager."""
        safe_component(sandbox_id)
        response = self.request("GET", f"/v1/sandboxes/{sandbox_id}/endpoints/{port}")
        endpoint = response["endpoint"].rstrip("/")
        url = urllib.parse.urlsplit(endpoint)
        if (
            url.scheme not in ("http", "https")
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Platform returned an invalid proxy endpoint")
        if (url.scheme, url.netloc) == (
            urllib.parse.urlsplit(self.base_url).scheme,
            urllib.parse.urlsplit(self.base_url).netloc,
        ):
            return endpoint
        return self.base_url + f"/v1/sandboxes/{sandbox_id}/proxy/{port}"


def feishu_webhook(config_file: Path | None = None, *, disabled: bool = False) -> str | None:
    """Resolve only an explicit file or SANDBOX_FEISHU_WEBHOOK; never search private state."""
    if disabled:
        return None
    if config_file is not None:
        config = json.loads(Path(config_file).expanduser().read_text())
        if not isinstance(config, dict) or not isinstance(config.get("webhook_url"), str):
            raise ValueError("Feishu config must contain a string webhook_url")
        value = config["webhook_url"]
    else:
        value = os.environ.get("SANDBOX_FEISHU_WEBHOOK", "")
    value = value.strip()
    if not value:
        return None
    url = urllib.parse.urlsplit(value)
    if (
        url.scheme != "https"
        or url.netloc != "open.feishu.cn"
        or url.query
        or url.fragment
        or not re.fullmatch(r"/open-apis/bot/v2/hook/[A-Za-z0-9_-]+", url.path)
    ):
        raise ValueError("Expected an HTTPS Feishu bot v2 webhook URL")
    return value


def feishu_test(webhook: str) -> dict:
    """Send one explicitly requested connectivity message, without platform credentials."""
    payload = {
        "msg_type": "text",
        "content": {
            "text": (
                "Sandbox 飞书机器人连接测试：正在验证 Webhook 连通性。"
                "这是一条测试消息，不是沙箱故障告警。"
            )
        },
    }
    request = urllib.request.Request(
        webhook,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=10) as response:
            result = json.load(response)
        if not isinstance(result, dict) or result.get("code") != 0:
            raise RuntimeError(
                "Feishu rejected the test: " + json.dumps(result, ensure_ascii=False)
            )
    except (OSError, RuntimeError, ValueError) as error:
        message = redact(str(error), [webhook, webhook.rsplit("/", 1)[-1]])
        raise RuntimeError("Feishu connection test failed: " + message) from None
    return {"feishu": "test_message_accepted", "code": 0}
