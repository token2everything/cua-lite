"""Owned Sandbox lifecycle shared by benchmark submission tools and the batch queue.

Call create only after preparing the platform payload and journaling any job
inputs. Ownership is persisted before callers verify Feishu or start monitoring.
The existing worker-<id>.json / worker_run record contract is retained.
"""

from __future__ import annotations

import json
from pathlib import Path

from lite.platforms.sandbox.client import SandboxClient, safe_component, state_of, write_json


def create(
    state_dir: Path,
    client: SandboxClient,
    payload: dict,
    *,
    owner: str,
    details: dict,
    webhook: str | None,
) -> dict:
    """Create once and save ownership immediately, including opaque benchmark details."""
    if webhook:
        client.secrets.extend((webhook, webhook.rsplit("/", 1)[-1]))
    created = client.request("POST", "/v1/sandboxes", payload)
    sandbox_id = safe_component(created["id"])
    record = {
        **details,
        "sandbox_id": sandbox_id,
        "environment_id": payload["extensions"]["environmentId"],
        "project": payload["extensions"]["project"],
        "base_url": client.base_url,
        "worker_run": payload["metadata"]["worker_run"],
        "state": state_of(created),
        "managed_by": owner,
        "feishu_webhook_configured": bool(webhook),
    }
    path = state_dir / f"worker-{sandbox_id}.json"
    try:
        write_json(path, record)
    except OSError as error:
        raise RuntimeError(
            f"Sandbox {sandbox_id} was created but its ownership record could not be saved: {error}"
        ) from None
    return {**record, "ownership_file": str(path)}


def verify_feishu(client: SandboxClient, record: dict, webhook: str) -> dict:
    """Verify the effective webhook only after the creation receipt exists."""
    record = dict(record)
    path = Path(record.pop("ownership_file"))
    try:
        actual = client.request("GET", "/v1/sandboxes/" + record["sandbox_id"])
    except RuntimeError as error:
        raise RuntimeError(
            f"Sandbox {record['sandbox_id']} created; could not verify Feishu: {error}. "
            f"Ownership record: {path}; inspect it before resubmitting"
        ) from None
    record["feishu_webhook_verified"] = (
        actual.get("extensions", {}).get("webhook-feishu") == webhook
    )
    write_json(path, record)
    if not record["feishu_webhook_verified"]:
        raise RuntimeError(
            f"Sandbox {record['sandbox_id']} created but effective Feishu verification failed; "
            f"inspect it before resubmitting. Ownership record: {path}"
        )
    return {**record, "ownership_file": str(path)}


def read_record(
    state_dir: Path,
    sandbox_id: str,
    client: SandboxClient,
    *,
    project: str,
    owner: str,
) -> dict:
    """Require a creation receipt belonging to this deployment and benchmark owner."""
    path = state_dir / f"worker-{safe_component(sandbox_id)}.json"
    if not path.is_file():
        raise ValueError(
            "No local create record for this Sandbox; attach does not grant deletion ownership"
        )
    record = json.loads(path.read_text())
    if (
        record.get("sandbox_id") != sandbox_id
        or record.get("project") != project
        or record.get("base_url") != client.base_url
        or record.get("managed_by") != owner
    ):
        raise ValueError("Sandbox ownership record does not match this deployment")
    return record


def check_remote(record: dict, current: dict) -> None:
    """Reject recycled IDs or foreign resources before monitoring or deletion."""
    if (
        not record.get("worker_run")
        or current.get("metadata", {}).get("worker_run") != record["worker_run"]
    ):
        raise ValueError("Remote Sandbox ownership does not match its local create record")


def delete(
    state_dir: Path,
    sandbox_id: str,
    client: SandboxClient,
    *,
    project: str,
    owner: str,
) -> dict:
    """Request deletion only after checking both local ownership and the remote nonce."""
    record = read_record(state_dir, sandbox_id, client, project=project, owner=owner)
    current = client.request("GET", "/v1/sandboxes/" + sandbox_id)
    check_remote(record, current)
    client.request("DELETE", "/v1/sandboxes/" + sandbox_id)
    record["state"] = "deletion_requested"
    write_json(state_dir / f"worker-{sandbox_id}.json", record)
    (state_dir / f"connection-{sandbox_id}.json").unlink(missing_ok=True)
    return {"sandbox_id": sandbox_id, "state": "deletion_requested"}
