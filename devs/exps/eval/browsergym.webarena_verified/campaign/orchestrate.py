#!/usr/bin/env python3
"""Provision and run a GLM WebArena-Verified campaign on owned sandboxes.

Usage: uv run python devs/exps/eval/browsergym.webarena_verified/campaign/orchestrate.py
    --root /path/to/results/<run> create

The result root must contain a generated plan, campaign.env, pinned source
archive, image lock, and the three worker scripts. Credentials come only from
the local .env and are uploaded directly to owned sandboxes, never to the
result root. Run `create -> wait-ready -> stage -> prepare -> startup -> full`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import shlex
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

IMAGE = "registry2-cloud.d.pjlab.org.cn/ccr-sandbox/sandbox-dind:dind-202608311205"
ASSET_SUBPATH = "eval/benchmark/webarena-verified/1.2.3"
BUNDLE_FILES = (
    "EXPERIMENT_CONFIG.json",
    "campaign.env",
    "PLAN_MANIFEST.sha256",
    "worker-plan.json",
    "image-lock.env",
    "cua-lite-source.tar.gz",
    "prepare-worker.sh",
    "run-case.sh",
    "run-worker.sh",
)
INDEX_FILES = (
    "completed.json",
    "status.json",
    "evaluation.json",
    "run_metadata.json",
    "agent_response.json",
    "instruction.json",
    "baseline_receipt.json",
    "startup_receipt.json",
    "archive.sha256",
    "sites.txt",
    "worker.txt",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="campaign result root")
    parser.add_argument("--env", type=Path, default=Path("/root/cua-lite/.env"))
    parser.add_argument("--worker", action="append", help="limit phase to a planned worker ID")
    parser.add_argument(
        "phase",
        choices=("create", "wait-ready", "stage", "prepare", "startup", "full", "sync", "status"),
    )
    return parser.parse_args()


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

    def __init__(self, env: dict[str, str]) -> None:
        required = (
            "BASE",
            "PROJECT",
            "OPEN_SANDBOX_ACCESS_KEY",
            "OPEN_SANDBOX_SECRET_KEY",
            "GLM_TEST_API_KEY",
            "OPENAI_BASE_URL",
        )
        missing = [name for name in required if not env.get(name)]
        if missing:
            raise ValueError(f"missing environment values: {missing}")
        self.base = env["BASE"].rstrip("/")
        self.project = env["PROJECT"]
        self.headers = {
            "OPEN-SANDBOX-API-KEY": (
                env["OPEN_SANDBOX_ACCESS_KEY"] + ":" + env["OPEN_SANDBOX_SECRET_KEY"]
            )
        }
        self.model_env = (
            f"OPENAI_API_KEY={env['GLM_TEST_API_KEY']}\n"
            f"OPENAI_BASE_URL={env['OPENAI_BASE_URL']}\n"
        ).encode()

    def detail(self, sandbox_id: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.base}/v1/sandboxes/{sandbox_id}", headers=self.headers, timeout=30
        )
        response.raise_for_status()
        return response.json()

    def list_sandboxes(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            response = requests.get(
                f"{self.base}/v1/sandboxes",
                headers=self.headers,
                params={"page": page, "pageSize": 1000},
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
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
        background: bool,
        timeout_ms: int | None = None,
    ) -> str:
        body: dict[str, Any] = {"command": command, "background": background}
        if timeout_ms is not None:
            body["timeout"] = timeout_ms
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
                    return event["text"]
        raise RuntimeError(f"execd did not return a command ID for {sandbox_id}")

    def command_status(self, sandbox_id: str, command_id: str) -> dict[str, Any]:
        url = f"{self.base}/v1/sandboxes/{sandbox_id}/proxy/44772/command/status/{command_id}"
        for attempt in range(1, 5):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 429 and response.status_code < 500:
                    response.raise_for_status()
                    return response.json()
                if attempt == 4:
                    response.raise_for_status()
            except requests.RequestException:
                if attempt == 4:
                    raise
            time.sleep(min(10, 2**attempt))
        raise AssertionError("unreachable")

    def checked_command(
        self, sandbox_id: str, command: str, *, timeout_ms: int = 30_000
    ) -> None:
        command_id = self.command(
            sandbox_id, command, background=False, timeout_ms=timeout_ms
        )
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
        response = requests.post(
            f"{self.base}/v1/sandboxes/{sandbox_id}/proxy/44772/files/upload",
            files=[
                ("metadata", ("metadata.json", json.dumps(metadata).encode(), "application/json")),
                ("file", (Path(remote_path).name, io.BytesIO(data), "application/octet-stream")),
            ],
            timeout=120,
        )
        response.raise_for_status()

    def download(self, sandbox_id: str, remote_path: str) -> bytes | None:
        url = f"{self.base}/v1/sandboxes/{sandbox_id}/proxy/44772/files/download"
        for attempt in range(1, 4):
            try:
                response = requests.get(
                    url, params={"path": remote_path}, timeout=(15, 60)
                )
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
        export_dir = f"/tmp/wav-glm53-export-{uuid.uuid4().hex}"
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
        if not part_names or any(
            not re.fullmatch(r"part-[a-z]{4}", name) for name in part_names
        ):
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


class Campaign:
    """Bind one result root, immutable plan, and session-owned worker set."""

    def __init__(self, root: Path, env_path: Path) -> None:
        self.root = root.resolve()
        self.settings = _read_env(self.root / "campaign.env")
        required = ("RUN_ID", "RUNNER_SHA256", "NAME_PREFIX", "MODEL_PROFILE")
        if any(not self.settings.get(name) for name in required):
            raise ValueError("campaign.env is incomplete")
        self.run_id = self.settings["RUN_ID"]
        self.runner_sha = self.settings["RUNNER_SHA256"]
        if not re.fullmatch(r"wav-glm53flash-infra-v[0-9]+-[a-z0-9-]+-[0-9]{8}", self.run_id):
            raise ValueError("invalid GLM run ID")
        if not re.fullmatch(r"[0-9a-f]{64}", self.runner_sha):
            raise ValueError("invalid runner SHA")
        if self.settings["MODEL_PROFILE"] != "glm53flash":
            raise ValueError("this coordinator serves only GLM campaigns")
        self.client = SandboxClient(_read_env(env_path))
        self.image_sha = _sha(self.root / "image-lock.env")
        self.plan_sha = _sha(self.root / "PLAN_MANIFEST.sha256")
        self.plan = self._load_plan()
        config = json.loads((self.root / "EXPERIMENT_CONFIG.json").read_text())
        if config["run_id"] != self.run_id or config["model"] != "glm-5.3-flash":
            raise ValueError("experiment config differs from campaign.env")
        with tarfile.open(self.root / "cua-lite-source.tar.gz", "r:gz") as source:
            runner = source.extractfile("./scripts/webarena_verified_qwen.py")
            if runner is None or hashlib.sha256(runner.read()).hexdigest() != self.runner_sha:
                raise ValueError("source archive runner SHA mismatch")
            if source.extractfile("./scripts/webarena_verified_case_receipt.py") is None:
                raise ValueError("source archive lacks case receipt validator")

    def _load_plan(self) -> dict[str, Any]:
        manifest: dict[str, str] = {}
        for line in (self.root / "PLAN_MANIFEST.sha256").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            path = (self.root / relative).resolve()
            if self.root not in path.parents or relative in manifest or _sha(path) != expected:
                raise ValueError(f"invalid plan manifest entry: {relative}")
            manifest[relative] = expected
        plan = json.loads((self.root / "worker-plan.json").read_text())
        worker_ids = list(plan["workers"])
        if worker_ids != [f"{index:02d}" for index in range(1, len(worker_ids) + 1)]:
            raise ValueError("worker IDs must be contiguous")
        expected_files = {"worker-plan.json"}
        task_ids: list[int] = []
        dataset_path = self.root.parents[1] / "dataset/webarena-verified.json"
        dataset = {
            int(row["task_id"]): sorted(row["sites"])
            for row in json.loads(dataset_path.read_text())
        }
        for worker in worker_ids:
            expected_files.update(
                {f"workers/{worker}/tasks.tsv", f"workers/{worker}/capabilities.txt"}
            )
            rows = [
                line.split("\t", 1)
                for line in (self.root / "workers" / worker / "tasks.tsv").read_text().splitlines()
            ]
            if any(len(row) != 2 for row in rows):
                raise ValueError(f"malformed task file for worker {worker}")
            current = [int(row[0]) for row in rows]
            if not current or len(current) != plan["workers"][worker]["tasks"]:
                raise ValueError(f"task count mismatch for worker {worker}")
            if current[0] != plan["workers"][worker]["startup_task"]:
                raise ValueError(f"startup task mismatch for worker {worker}")
            if any(row[1].split(",") != dataset[int(row[0])] for row in rows):
                raise ValueError(f"task site mismatch for worker {worker}")
            sites = sorted({site for task_id in current for site in dataset[task_id]})
            if sites != plan["workers"][worker]["capabilities"]:
                raise ValueError(f"capability mismatch for worker {worker}")
            task_ids.extend(current)
        if set(manifest) != expected_files or len(task_ids) != len(set(task_ids)):
            raise ValueError("plan manifest or task assignment is incomplete")
        if len(task_ids) != plan["task_count"]:
            raise ValueError("plan task count mismatch")
        if plan["selection"] == "full" and set(task_ids) != set(range(812)):
            raise ValueError("full plan must cover exactly 0..811")
        return plan

    def workers(self, requested: list[str] | None) -> list[str]:
        if not requested:
            return list(self.plan["workers"])
        normalized = [f"{int(worker):02d}" for worker in requested]
        if len(normalized) != len(set(normalized)) or any(
            worker not in self.plan["workers"] for worker in normalized
        ):
            raise ValueError("requested workers must be unique planned IDs")
        return normalized

    def _cluster(self) -> dict[str, dict[str, str]]:
        path = self.root / "sandbox-cluster.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _owned_detail(self, worker: str) -> dict[str, Any]:
        record = self._cluster().get(worker)
        if record is None:
            raise RuntimeError(f"worker {worker} has not been created")
        detail = self.client.detail(record["id"])
        metadata = detail.get("metadata") or {}
        expected = {
            "run_id": self.run_id,
            "worker": worker,
            "plan_sha256": self.plan_sha,
            "session_nonce": record["nonce"],
        }
        if any(metadata.get(name) != value for name, value in expected.items()):
            raise RuntimeError(f"sandbox ownership mismatch: {record['id']}")
        return detail

    def create(self, workers: list[str]) -> None:
        cluster = self._cluster()
        for worker in workers:
            if worker in cluster:
                detail = self._owned_detail(worker)
                print(f"{worker}: {detail['id']} {detail['state']}", flush=True)
                continue
            intent_path = self.root / "worker-intents" / f"{worker}.json"
            if intent_path.exists():
                intent = json.loads(intent_path.read_text())
                if intent["payload"]["metadata"]["run_id"] != self.run_id:
                    raise RuntimeError(f"worker {worker} intent belongs to another run")
            else:
                capabilities = self.plan["workers"][worker]["capabilities"]
                nonce = uuid.uuid4().hex
                extensions = {
                    "project": self.client.project,
                    "runtimeClassName": "sysbox-runc",
                }
                if "map" in capabilities:
                    extensions["fuse"] = "true"
                intent = {
                    "nonce": nonce,
                    "post_attempted": False,
                    "payload": {
                        "image": {"uri": IMAGE},
                        "entrypoint": ["sleep", "inf"],
                        "timeout": 259200,
                        "resourceRequests": {"cpu": "4", "memory": "20Gi"},
                        "volumes": [
                            {
                                "mountPath": "/mnt/webarena-verified",
                                "subPath": ASSET_SUBPATH,
                                "readOnly": True,
                            }
                        ],
                        "metadata": {
                            "run_id": self.run_id,
                            "worker": worker,
                            "plan_sha256": self.plan_sha,
                            "selection": self.plan["selection"],
                            "task_count": str(self.plan["workers"][worker]["tasks"]),
                            "model": "glm-5.3-flash",
                            "benchmark_version": "1.2.3",
                            "owner": "codex-session",
                            "purpose": "versioned-infra-regression",
                            "session_nonce": nonce,
                        },
                        "ports": [
                            {
                                "containerPort": 44772,
                                "name": "execd",
                                "protocol": "tcp",
                                "purpose": "system service",
                            }
                        ],
                        "extensions": extensions,
                    },
                }
                _atomic_json(intent_path, intent)

            # Reconcile the nonce first; an unknown POST result is not authority
            # to create a second sandbox for the same worker.
            matches = [
                item
                for item in self.client.list_sandboxes()
                if (item.get("metadata") or {}).get("session_nonce") == intent["nonce"]
            ]
            if len(matches) > 1:
                raise RuntimeError(f"duplicate sandbox nonce for worker {worker}")
            if matches:
                created = matches[0]
            elif intent["post_attempted"]:
                raise RuntimeError(
                    f"worker {worker} has an unresolved create attempt; reconcile before retry"
                )
            else:
                intent["post_attempted"] = True
                _atomic_json(intent_path, intent)
                created = self.client.create(intent["payload"])
            if (created.get("metadata") or {}).get("session_nonce") != intent["nonce"]:
                raise RuntimeError(f"created sandbox nonce mismatch for worker {worker}")
            cluster[worker] = {"id": created["id"], "nonce": intent["nonce"]}
            _atomic_json(self.root / "sandbox-cluster.json", cluster)
            print(f"{worker}: created {created['id']} {created['state']}", flush=True)

    def wait_ready(self, workers: list[str]) -> None:
        deadline = time.monotonic() + 3600
        while True:
            states = {worker: self._owned_detail(worker)["state"] for worker in workers}
            print(f"sandbox states: {states}", flush=True)
            if any(state == "failed" for state in states.values()):
                raise RuntimeError("an owned sandbox failed during startup")
            if all(state == "running" for state in states.values()):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("sandboxes did not become ready in one hour")
            time.sleep(10)

    def _bundle(self) -> bytes:
        archive = io.BytesIO()
        # A stable gzip header lets a timed-out stage retry match receipts
        # already written by other workers for these same source files.
        with gzip.GzipFile(fileobj=archive, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as bundle:
                for name in BUNDLE_FILES:
                    bundle.add(self.root / name, arcname=name, recursive=False)
                for worker in self.plan["workers"]:
                    for name in ("tasks.tsv", "capabilities.txt"):
                        relative = f"workers/{worker}/{name}"
                        bundle.add(self.root / relative, arcname=relative, recursive=False)
        return archive.getvalue()

    def stage(self, workers: list[str]) -> None:
        bundle = self._bundle()
        digest = hashlib.sha256(bundle).hexdigest()
        for worker in workers:
            detail = self._owned_detail(worker)
            if detail["state"] != "running":
                raise RuntimeError(f"worker {worker} sandbox is not running")
            marker_path = self.root / "worker-intents" / f"{worker}-stage.json"
            if marker_path.exists():
                marker = json.loads(marker_path.read_text())
                if marker["bundle_sha256"] != digest:
                    raise RuntimeError(f"worker {worker} already staged a different bundle")
                self.client.checked_command(
                    detail["id"], "test -f /mnt/results/run-worker.sh"
                )
                continue
            remote = "/tmp/wav-glm53-campaign-bundle.tar.gz"
            self.client.upload(detail["id"], remote, bundle)
            self.client.checked_command(
                detail["id"],
                f"sha256sum {remote} | grep -q '^{digest} ' && "
                "install -d -m 700 /mnt/results && "
                f"tar -xzf {remote} -C /mnt/results && "
                "chmod -R u+rwX /mnt/results && "
                "test -f /mnt/results/prepare-worker.sh && "
                f"rm -f {remote}",
            )
            self.client.checked_command(
                detail["id"], "install -d -m 700 /var/lib/wav-qwen/private"
            )
            self.client.upload(
                detail["id"], "/var/lib/wav-qwen/private/model.env", self.client.model_env
            )
            self.client.checked_command(
                detail["id"],
                "test -s /var/lib/wav-qwen/private/model.env && "
                'test "$(stat -c %a /var/lib/wav-qwen/private/model.env)" = 600',
            )
            _atomic_json(marker_path, {"sandbox_id": detail["id"], "bundle_sha256": digest})
            print(f"{worker}: staged {digest}", flush=True)

    def ensure_dockerd(self, workers: list[str]) -> None:
        for worker in workers:
            detail = self._owned_detail(worker)
            if detail["state"] != "running":
                raise RuntimeError(f"worker {worker} sandbox is not running")
            path = self.root / "worker-intents" / f"{worker}-dockerd.json"
            record = json.loads(path.read_text()) if path.exists() else None
            if record is not None:
                status = self.client.command_status(detail["id"], record["command_id"])
                if status.get("running"):
                    self.client.checked_command(detail["id"], "docker info >/dev/null 2>&1")
                    continue
            command = (
                "export NO_PROXY=registry-v2.h.pjlab.org.cn,10.102.254.2,.pjlab.org.cn; "
                'export no_proxy="$NO_PROXY"; '
                "exec dockerd >/tmp/wav-glm53-dockerd.log 2>&1"
            )
            command_id = self.client.command(detail["id"], command, background=True)
            _atomic_json(path, {"sandbox_id": detail["id"], "command_id": command_id})
            self.client.checked_command(
                detail["id"],
                "for n in $(seq 1 180); do docker info >/dev/null 2>&1 && exit 0; "
                "sleep 1; done; exit 1",
                timeout_ms=240_000,
            )
            status = self.client.command_status(detail["id"], command_id)
            if not status.get("running"):
                raise RuntimeError(f"worker {worker} dockerd did not persist")
            print(f"{worker}: dockerd ready", flush=True)

    def _prepared(self, worker: str, sandbox_id: str) -> bool:
        data = self.client.download(
            sandbox_id, f"/mnt/results/workers/{worker}/prepared.json"
        )
        if data is None:
            return False
        marker = json.loads(data)
        expected = {
            "run_id": self.run_id,
            "worker": worker,
            "sandbox_id": sandbox_id,
            "capabilities": ",".join(self.plan["workers"][worker]["capabilities"]),
            "source_sha256": _sha(self.root / "cua-lite-source.tar.gz"),
            "image_lock_sha256": self.image_sha,
        }
        if any(marker.get(name) != value for name, value in expected.items()):
            raise RuntimeError(f"worker {worker} prepared marker differs from campaign")
        return True

    def prepare(self, workers: list[str]) -> None:
        pending = []
        active: dict[str, dict[str, str]] = {}
        for worker in workers:
            detail = self._owned_detail(worker)
            stage_path = self.root / "worker-intents" / f"{worker}-stage.json"
            if not stage_path.is_file():
                raise RuntimeError(f"worker {worker} has not been staged")
            if self._prepared(worker, detail["id"]):
                print(f"{worker}: already prepared", flush=True)
                continue
            command_path = self.root / "worker-intents" / f"{worker}-prepare.json"
            if command_path.exists():
                record = json.loads(command_path.read_text())
                if record["sandbox_id"] != detail["id"]:
                    raise RuntimeError(f"worker {worker} prepare command sandbox mismatch")
                status = self.client.command_status(detail["id"], record["command_id"])
                if status.get("running"):
                    active[worker] = record
                    continue
                raise RuntimeError(
                    f"worker {worker} prepare stopped without a marker: {status.get('exit_code')}"
                )
            pending.append(worker)

        deadline = time.monotonic() + 10_800
        while pending or active:
            # Observe completion before allocating more image pulls or database
            # copies; each worker's Docker daemon remains a separate command.
            for worker, record in list(active.items()):
                status = self.client.command_status(record["sandbox_id"], record["command_id"])
                if status.get("running"):
                    continue
                if status.get("exit_code") != 0 or not self._prepared(
                    worker, record["sandbox_id"]
                ):
                    raise RuntimeError(
                        f"worker {worker} prepare failed: {status.get('exit_code')}"
                    )
                del active[worker]
                print(f"{worker}: prepared", flush=True)

            # Keep map copies serial and image pulls bounded on the shared
            # cluster. They remain fully parallel for case execution later.
            for worker in list(pending):
                if len(active) >= 8:
                    break
                capabilities = self.plan["workers"][worker]["capabilities"]
                if "map" in capabilities and any(
                    "map" in self.plan["workers"][current]["capabilities"]
                    for current in active
                ):
                    continue
                detail = self._owned_detail(worker)
                command = shlex.join(
                    [
                        "bash",
                        "/mnt/results/prepare-worker.sh",
                        worker,
                        ",".join(capabilities),
                        detail["id"],
                    ]
                )
                command_id = self.client.command(detail["id"], command, background=True)
                record = {"sandbox_id": detail["id"], "command_id": command_id}
                _atomic_json(self.root / "worker-intents" / f"{worker}-prepare.json", record)
                active[worker] = record
                pending.remove(worker)
                print(f"{worker}: preparation started", flush=True)

            if pending or active:
                if time.monotonic() >= deadline:
                    raise TimeoutError("worker preparation did not finish in three hours")
                time.sleep(15)

    def _validated_receipt(
        self, receipt: dict[str, Any], worker: str, sandbox_id: str, task_id: int
    ) -> None:
        expected = {
            "run_id": self.run_id,
            "plan_sha256": self.plan_sha,
            "worker": worker,
            "sandbox_id": sandbox_id,
            "task_id": task_id,
            "runner_sha256": self.runner_sha,
            "image_lock_sha256": self.image_sha,
        }
        if any(receipt.get(name) != value for name, value in expected.items()):
            raise RuntimeError(f"task {task_id} receipt belongs to another campaign")
        terminal = receipt.get("terminal_status")
        score = receipt.get("score")
        if terminal == "evaluator_error":
            if score is not None:
                raise RuntimeError(f"task {task_id} evaluator error was scored")
        elif terminal in {"completed", "model_protocol_failed"}:
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise RuntimeError(f"task {task_id} has invalid official score")
        else:
            raise RuntimeError(f"task {task_id} has nonterminal receipt")
        if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("archive_sha256", ""))):
            raise RuntimeError(f"task {task_id} receipt lacks archive hash")

    def _sync_one(self, worker: str, sandbox_id: str, task_id: int) -> str:
        case = f"task-{task_id:04d}"
        remote_index = f"/mnt/results/index/{case}"
        data = self.client.download(sandbox_id, f"{remote_index}/completed.json")
        if data is None:
            return "pending"
        receipt = json.loads(data)
        self._validated_receipt(receipt, worker, sandbox_id, task_id)
        index = self.root / "index" / case
        archive = self.root / "archives" / f"{case}.tar.gz"
        expected_sha = receipt["archive_sha256"]
        if index.exists():
            local = json.loads((index / "completed.json").read_text())
            if local != receipt or not archive.is_file() or _sha(archive) != expected_sha:
                raise RuntimeError(f"task {task_id} local terminal evidence differs from remote")
            return "already_synced"

        # Publish the archive only after its hash is verified. Then publish the
        # index atomically; a partial transfer never counts as a finished case.
        if archive.exists():
            if _sha(archive) != expected_sha:
                raise RuntimeError(f"task {task_id} local archive hash mismatch")
        else:
            self.client.download_large_file(
                sandbox_id, f"/mnt/results/archives/{case}.tar.gz", archive, expected_sha
            )
        index.parent.mkdir(parents=True, exist_ok=True)
        temporary = index.with_name(f".{case}.{uuid.uuid4().hex}.sync")
        temporary.mkdir()
        for name in INDEX_FILES:
            content = self.client.download(sandbox_id, f"{remote_index}/{name}")
            if content is not None:
                (temporary / name).write_bytes(content)
        required = {
            "completed.json",
            "status.json",
            "evaluation.json",
            "run_metadata.json",
            "baseline_receipt.json",
            "startup_receipt.json",
            "archive.sha256",
        }
        if not required <= {path.name for path in temporary.iterdir()}:
            raise RuntimeError(f"task {task_id} remote index is incomplete")
        if json.loads((temporary / "completed.json").read_text()) != receipt:
            raise RuntimeError(f"task {task_id} remote receipt changed during sync")
        if (temporary / "archive.sha256").read_text().strip() != expected_sha:
            raise RuntimeError(f"task {task_id} remote archive checksum differs")
        os.replace(temporary, index)
        return "synced"

    def sync(self, workers: list[str]) -> None:
        totals = {"pending": 0, "already_synced": 0, "synced": 0}
        for worker in workers:
            detail = self._owned_detail(worker)
            if detail["state"] != "running":
                raise RuntimeError(f"worker {worker} sandbox is not running")
            for line in (self.root / "workers" / worker / "tasks.tsv").read_text().splitlines():
                task_id = int(line.split("\t", 1)[0])
                result = self._sync_one(worker, detail["id"], task_id)
                totals[result] += 1
                if result == "synced":
                    print(f"{worker} task {task_id}: synced", flush=True)
        print(json.dumps(totals, indent=2), flush=True)

    def _launch(self, worker: str, mode: str, attempt: int) -> dict[str, str]:
        detail = self._owned_detail(worker)
        if detail["state"] != "running" or not self._prepared(worker, detail["id"]):
            raise RuntimeError(f"worker {worker} is not ready for {mode}")
        path = self.root / "worker-intents" / f"{worker}-{mode}-{attempt}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["sandbox_id"] != detail["id"]:
                raise RuntimeError(f"worker {worker} {mode} launch sandbox mismatch")
        else:
            launch_id = uuid.uuid4().hex
            record = {
                "sandbox_id": detail["id"],
                "launch_id": launch_id,
                "command": shlex.join(
                    [
                        "bash",
                        "/mnt/results/run-worker.sh",
                        worker,
                        detail["id"],
                        mode,
                        launch_id,
                    ]
                ),
            }
            _atomic_json(path, record)
        if "command_id" not in record:
            # Reusing the launch ID is safe after an ambiguous POST: run-worker
            # serializes on its owned worker lock and records that ID's exit.
            record["command_id"] = self.client.command(
                detail["id"], record["command"], background=True
            )
            _atomic_json(path, record)
        return record

    def _wait_launches(
        self, records: dict[str, dict[str, str]], *, mode: str
    ) -> dict[str, int]:
        deadline = time.monotonic() + (10_800 if mode == "startup" else 86_400)
        active = dict(records)
        completed: dict[str, int] = {}
        while active:
            # Poll command receipts without changing site or model state.
            for worker, record in list(active.items()):
                status = self.client.command_status(record["sandbox_id"], record["command_id"])
                if status.get("running"):
                    continue
                code = status.get("exit_code")
                if not isinstance(code, int):
                    raise RuntimeError(f"worker {worker} {mode} has no exit code")
                completed[worker] = code
                del active[worker]
                print(f"{worker}: {mode} exited {code}", flush=True)
            if active:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{mode} commands did not finish before deadline")
                time.sleep(30)
        return completed

    def _local_receipt(self, worker: str, sandbox_id: str, task_id: int) -> bool:
        receipt_path = self.root / "index" / f"task-{task_id:04d}" / "completed.json"
        archive = self.root / "archives" / f"task-{task_id:04d}.tar.gz"
        if not receipt_path.is_file() or not archive.is_file():
            return False
        receipt = json.loads(receipt_path.read_text())
        self._validated_receipt(receipt, worker, sandbox_id, task_id)
        if _sha(archive) != receipt["archive_sha256"]:
            raise RuntimeError(f"task {task_id} local archive changed")
        return True

    def _worker_receipts_complete(self, worker: str, sandbox_id: str) -> bool:
        return all(
            self._local_receipt(worker, sandbox_id, int(line.split("\t", 1)[0]))
            for line in (self.root / "workers" / worker / "tasks.tsv").read_text().splitlines()
        )

    def startup(self, workers: list[str]) -> None:
        remaining = [
            worker
            for worker in workers
            if not self._local_receipt(
                worker,
                self._owned_detail(worker)["id"],
                self.plan["workers"][worker]["startup_task"],
            )
        ]
        for attempt in range(1, 4):
            if not remaining:
                break
            records = {worker: self._launch(worker, "startup", attempt) for worker in remaining}
            exit_codes = self._wait_launches(records, mode="startup")
            next_remaining = []
            for worker in remaining:
                sandbox_id = records[worker]["sandbox_id"]
                task_id = self.plan["workers"][worker]["startup_task"]
                outcome = self._sync_one(worker, sandbox_id, task_id)
                if exit_codes[worker] == 0 and outcome != "pending":
                    print(f"{worker}: startup receipt passed", flush=True)
                else:
                    next_remaining.append(worker)
                    print(f"{worker}: startup retry needed ({outcome})", flush=True)
            remaining = next_remaining
        if remaining:
            raise RuntimeError(f"startup gate did not pass for workers: {remaining}")

        # Publish the global gate only when every planned worker has a bound,
        # locally archived startup receipt, including earlier partial batches.
        cluster = self._cluster()
        receipts = {}
        for worker, item in cluster.items():
            if worker not in self.plan["workers"]:
                raise RuntimeError(f"unplanned worker in cluster: {worker}")
            task_id = self.plan["workers"][worker]["startup_task"]
            path = self.root / "index" / f"task-{task_id:04d}" / "completed.json"
            if path.is_file():
                receipt = json.loads(path.read_text())
                self._validated_receipt(receipt, worker, item["id"], task_id)
                receipts[worker] = receipt["archive_sha256"]
        if set(receipts) == set(self.plan["workers"]):
            _atomic_json(
                self.root / "startup-gate.json",
                {
                    "run_id": self.run_id,
                    "plan_sha256": self.plan_sha,
                    "runner_sha256": self.runner_sha,
                    "image_lock_sha256": self.image_sha,
                    "startup_archives": receipts,
                },
            )
            print(f"startup gate passed: {len(receipts)} workers", flush=True)
        else:
            print(f"startup gate partial: {len(receipts)}/{len(self.plan['workers'])}", flush=True)

    def _verify_startup_gate(self) -> None:
        path = self.root / "startup-gate.json"
        if not path.is_file():
            raise RuntimeError("global startup gate has not passed")
        gate = json.loads(path.read_text())
        expected = {
            "run_id": self.run_id,
            "plan_sha256": self.plan_sha,
            "runner_sha256": self.runner_sha,
            "image_lock_sha256": self.image_sha,
        }
        if any(gate.get(name) != value for name, value in expected.items()):
            raise RuntimeError("startup gate is for another campaign version")
        if set(gate["startup_archives"]) != set(self.plan["workers"]):
            raise RuntimeError("startup gate does not cover every worker")
        cluster = self._cluster()
        for worker, archive_sha in gate["startup_archives"].items():
            task_id = self.plan["workers"][worker]["startup_task"]
            path = self.root / "index" / f"task-{task_id:04d}" / "completed.json"
            receipt = json.loads(path.read_text())
            self._validated_receipt(receipt, worker, cluster[worker]["id"], task_id)
            if receipt["archive_sha256"] != archive_sha:
                raise RuntimeError(f"worker {worker} startup archive changed")

    def full(self, workers: list[str]) -> None:
        self._verify_startup_gate()
        remaining = [
            worker
            for worker in workers
            if not self._worker_receipts_complete(worker, self._owned_detail(worker)["id"])
        ]
        for attempt in range(1, 4):
            if not remaining:
                break
            records = {worker: self._launch(worker, "full", attempt) for worker in remaining}
            exit_codes = self._wait_launches(records, mode="full")
            next_remaining = []
            for worker in remaining:
                sandbox_id = records[worker]["sandbox_id"]
                self.sync([worker])
                if exit_codes[worker] == 0 and self._worker_receipts_complete(worker, sandbox_id):
                    print(f"{worker}: full shard complete", flush=True)
                else:
                    next_remaining.append(worker)
                    print(f"{worker}: full shard retry needed", flush=True)
            remaining = next_remaining
        if remaining:
            raise RuntimeError(f"full campaign has incomplete workers: {remaining}")

        # Official evaluator errors remain terminal but unscored. Never present
        # an 812-denominator accuracy unless all 812 official scores exist.
        if set(workers) == set(self.plan["workers"]):
            receipts = [
                json.loads(path.read_text())
                for path in sorted((self.root / "index").glob("task-*/completed.json"))
            ]
            scored = [item for item in receipts if item["terminal_status"] != "evaluator_error"]
            summary = {
                "run_id": self.run_id,
                "planned": self.plan["task_count"],
                "terminal": len(receipts),
                "scored": len(scored),
                "evaluator_errors": len(receipts) - len(scored),
                "successes": sum(item["score"] == 1 for item in scored),
                "official_accuracy": (
                    sum(item["score"] == 1 for item in scored) / len(receipts)
                    if len(scored) == len(receipts) == self.plan["task_count"]
                    else None
                ),
            }
            _atomic_json(self.root / "campaign-summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)

    def status(self, workers: list[str]) -> None:
        counts: dict[str, int] = {}
        scored = 0
        successes = 0
        for worker in workers:
            detail = self._owned_detail(worker)
            counts[detail["state"]] = counts.get(detail["state"], 0) + 1
            for line in (self.root / "workers" / worker / "tasks.tsv").read_text().splitlines():
                task_id = int(line.split("\t", 1)[0])
                receipt_path = self.root / "index" / f"task-{task_id:04d}" / "completed.json"
                if receipt_path.is_file():
                    receipt = json.loads(receipt_path.read_text())
                    if receipt.get("terminal_status") != "evaluator_error":
                        scored += 1
                        successes += receipt.get("score") == 1
        print(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "sandboxes": counts,
                    "planned_tasks": sum(self.plan["workers"][w]["tasks"] for w in workers),
                    "scored": scored,
                    "successes": successes,
                },
                indent=2,
            )
        )


def main() -> None:
    args = _parse_args()
    campaign = Campaign(args.root, args.env)
    workers = campaign.workers(args.worker)
    if args.phase == "create":
        campaign.create(workers)
    elif args.phase == "wait-ready":
        campaign.wait_ready(workers)
    elif args.phase == "stage":
        campaign.stage(workers)
    elif args.phase == "prepare":
        campaign.ensure_dockerd(workers)
        campaign.prepare(workers)
    elif args.phase == "status":
        campaign.status(workers)
    elif args.phase == "sync":
        campaign.sync(workers)
    elif args.phase == "startup":
        campaign.startup(workers)
    elif args.phase == "full":
        campaign.full(workers)
    else:
        raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
