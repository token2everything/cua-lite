"""Sandbox transport shipped with the v0.3 pipeline; no campaign dependency."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import re
import time
import uuid
from pathlib import Path
from typing import Any
import requests

IMAGE = "registry2-cloud.d.pjlab.org.cn/ccr-sandbox/sandbox-dind:dind-202608311205"

ASSET_SUBPATH = "eval/benchmark/webarena-verified/1.2.3"

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")
    os.replace(temporary, path)

def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name.strip()] = value
    return values

class SandboxClient:
    """Open Sandbox transport for one authenticated project."""

    def __init__(self, env: dict[str, str], *, api_key_env: str, receipts: Path) -> None:
        required = (
            "BASE",
            "PROJECT",
            "OPEN_SANDBOX_ACCESS_KEY",
            "OPEN_SANDBOX_SECRET_KEY",
            api_key_env,
            "OPENAI_BASE_URL",
        )
        missing = [name for name in required if not env.get(name)]
        if missing:
            raise ValueError(f"missing environment values: {missing}")
        self.base = env["BASE"].rstrip("/")
        self.receipts = receipts
        self.project = env["PROJECT"]
        self.headers = {
            "OPEN-SANDBOX-API-KEY": (
                env["OPEN_SANDBOX_ACCESS_KEY"] + ":" + env["OPEN_SANDBOX_SECRET_KEY"]
            )
        }
        self.model_env = (
            f"OPENAI_API_KEY={env[api_key_env]}\nOPENAI_BASE_URL={env['OPENAI_BASE_URL']}\n"
        ).encode()

    def _get_resource(self, path: str, *, timeout: int, params: dict | None = None) -> dict[str, Any]:
        """Read resource JSON, retrying transient GET failures at most four times."""
        for attempt in range(1, 5):
            try:
                response = requests.get(
                    self.base + path, headers=self.headers, params=params, timeout=timeout
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 4:
                    raise
            else:
                transient = response.status_code == 429 or response.status_code >= 500
                if not transient or attempt == 4:
                    response.raise_for_status()
                    return response.json()
            time.sleep(min(10, 2**attempt))
        raise AssertionError("unreachable")

    def detail(self, sandbox_id: str) -> dict[str, Any]:
        return self._get_resource(f"/v1/sandboxes/{sandbox_id}", timeout=30)

    def list_sandboxes(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self._get_resource(
                "/v1/sandboxes", timeout=45, params={"page": page, "pageSize": 1000}
            )
            items.extend(payload["items"])
            if not payload["pagination"]["hasNextPage"]:
                return items
            page += 1

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Creation is deliberately one-shot. An ambiguous response is reconciled
        # against the intent nonce before another POST is allowed.
        response = requests.post(
            f"{self.base}/v1/sandboxes",
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        return response.json()

    def command(
        self,
        sandbox_id: str,
        command: str,
        *,
        label: str,
        background: bool,
        timeout_ms: int | None = None,
    ) -> str:
        body: dict[str, Any] = {"command": command, "background": background}
        if timeout_ms is not None:
            body["timeout"] = timeout_ms
        intent_path = self.receipts / (label + '.intent.json')
        handle_path = self.receipts / (label + '.handle.json')
        intent = {'sandbox_id': sandbox_id, 'command': command, 'body': body}
        if intent_path.exists():
            if json.loads(intent_path.read_text()) != intent:
                raise ValueError('Original command intent differs')
            if not handle_path.exists():
                raise RuntimeError('Original submission outcome unknown; no second POST')
            return json.loads(handle_path.read_text())['command_id']
        _atomic_json(intent_path, intent)
        response = requests.post(
            f"{self.base}/v1/sandboxes/{sandbox_id}/proxy/44772/command",
            json=body,
            timeout=60 if background else max(60, (timeout_ms or 30_000) / 1000 + 15),
        )
        response.raise_for_status()
        for line in response.text.splitlines():
            if line.strip():
                event = json.loads(line)
                if event.get("type") == "init":
                    _atomic_json(handle_path, {'command_id': event['text']})
                    return event["text"]
        raise RuntimeError(f"execd did not return a command ID for {sandbox_id}")

    def command_status(self, sandbox_id: str, command_id: str) -> dict[str, Any]:
        url = f"{self.base}/v1/sandboxes/{sandbox_id}/proxy/44772/command/status/{command_id}"
        for attempt in range(1, 5):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 429 and response.status_code < 500:
                    response.raise_for_status()
                    status = response.json()
                    # Execd omits the terminal code while a command is running.
                    return {**status, 'exit_code': None if status['running'] else status['exit_code']}
                if attempt == 4:
                    response.raise_for_status()
            except requests.RequestException:
                if attempt == 4:
                    raise
            time.sleep(min(10, 2**attempt))
        raise AssertionError("unreachable")

    def checked_command(self, sandbox_id: str, command: str, *, timeout_ms: int = 30_000) -> None:
        label = 'transport-' + hashlib.sha256((sandbox_id+'\n'+command).encode()).hexdigest()
        command_id = self.command(sandbox_id, command, label=label,
                                  background=False, timeout_ms=timeout_ms)
        status = self.command_status(sandbox_id, command_id)
        if status.get("running") or status.get("exit_code") != 0:
            raise RuntimeError(
                f"remote command failed in {sandbox_id}: exit_code={status.get('exit_code')}"
            )

    def upload(self, sandbox_id: str, remote_path: str, data: bytes, *, mode: int = 600) -> None:
        if len(data) > 1024 * 1024:
            # The sandbox file endpoint can stall on a 16 MiB multipart POST.
            # Publish small parts first, then atomically replace the target
            # only after the assembled payload matches its local digest.
            part_paths = []
            for offset in range(0, len(data), 1024 * 1024):
                part_path = f"{remote_path}.part-{offset // (1024 * 1024):04d}"
                self.upload(sandbox_id, part_path, data[offset : offset + 1024 * 1024])
                part_paths.append(part_path)
            assembled = f"{remote_path}.assembled"
            digest = hashlib.sha256(data).hexdigest()
            self.checked_command(
                sandbox_id,
                f"cat {' '.join(map(shlex.quote, part_paths))} > {shlex.quote(assembled)} && "
                f"sha256sum {shlex.quote(assembled)} | grep -q '^{digest} ' && "
                f"chmod {mode} {shlex.quote(assembled)} && "
                f"mv {shlex.quote(assembled)} {shlex.quote(remote_path)} && "
                f"rm -f {' '.join(map(shlex.quote, part_paths))}",
                timeout_ms=300_000,
            )
            return
        metadata = {"path": remote_path, "owner": "root", "group": "root", "mode": mode}
        for attempt in range(1, 4):
            try:
                response = requests.post(
                    f"{self.base}/v1/sandboxes/{sandbox_id}/proxy/44772/files/upload",
                    files=[
                        (
                            "metadata",
                            ("metadata.json", json.dumps(metadata).encode(), "application/json"),
                        ),
                        (
                            "file",
                            (Path(remote_path).name, io.BytesIO(data), "application/octet-stream"),
                        ),
                    ],
                    timeout=(15, 30),
                )
            except requests.RequestException:
                if attempt == 3:
                    raise
            else:
                if response.status_code not in {429} and response.status_code < 500:
                    response.raise_for_status()
                    return
                if attempt == 3:
                    response.raise_for_status()
            time.sleep(min(10, 2**attempt))

    def download(self, sandbox_id: str, remote_path: str) -> bytes | None:
        url = f"{self.base}/v1/sandboxes/{sandbox_id}/proxy/44772/files/download"
        for attempt in range(1, 4):
            try:
                response = requests.get(url, params={"path": remote_path}, timeout=(15, 60))
                if response.status_code == 404:
                    return None
                if response.status_code != 429 and response.status_code < 500:
                    response.raise_for_status()
                    return response.content
                if attempt == 3:
                    response.raise_for_status()
            except requests.RequestException:
                if attempt == 3:
                    raise
            time.sleep(min(10, 2**attempt))
        raise AssertionError("unreachable")

    def download_large_file(
        self, sandbox_id: str, remote_path: str, local_path: Path, expected_sha256: str
    ) -> None:
        if local_path.exists() and _sha(local_path) == expected_sha256:
            return
        transfer = hashlib.sha256((sandbox_id+'\n'+remote_path+'\n'+expected_sha256).encode()).hexdigest()
        export_dir = f"/tmp/wav-v03-export-{transfer}"
        self.checked_command(
            sandbox_id,
            f"mkdir -m 700 {export_dir} && "
            f"split -b 4m -a 4 {shlex.quote(remote_path)} {export_dir}/part- && "
            f"find {export_dir} -maxdepth 1 -name 'part-*' -type f "
            f"| sort | sed 's#.*/##' > {export_dir}/parts && "
            f"sha256sum {shlex.quote(remote_path)} | grep -q '^{expected_sha256} '",
            timeout_ms=300_000,
        )
        part_manifest = self.download(sandbox_id, f"{export_dir}/parts")
        if part_manifest is None:
            raise RuntimeError(f"archive split manifest missing: {remote_path}")
        part_names = part_manifest.decode().splitlines()
        if not part_names or any(not re.fullmatch(r"part-[a-z]{4}", name) for name in part_names):
            raise RuntimeError(f"invalid archive split manifest: {remote_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_path.with_name(f".{local_path.name}.{uuid.uuid4().hex}.download")
        with temporary.open("xb") as output:
            for name in part_names:
                data = self.download(sandbox_id, f"{export_dir}/{name}")
                if data is None:
                    raise RuntimeError(f"archive split part missing: {name}")
                output.write(data)
        if _sha(temporary) != expected_sha256:
            raise RuntimeError(f"archive hash mismatch: {remote_path}")
        os.replace(temporary, local_path)
        self.checked_command(sandbox_id, f"rm -rf {export_dir}")
