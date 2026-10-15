"""Manage this execution's worker for its declared fixed-image sites.

Reuse the frozen SandboxClient transport without constructing its official
Campaign. Persist creation intent before POST and reconcile unknown outcomes.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import uuid
from pathlib import Path

import requests
import sandbox as module

TRANSPORT = Path(module.__file__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--sites", type=json.loads, required=True, help='JSON list, e.g. ["shopping", "gitlab"]'
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--file", type=Path)
    parser.add_argument("--remote")
    parser.add_argument("--command")
    parser.add_argument(
        "phase",
        choices=("create", "status", "dockerd", "upload", "command", "command-status", "download"),
    )
    args = parser.parse_args()
    if (
        not isinstance(args.sites, list)
        or len(args.sites) not in {1, 2}
        or any(
            not isinstance(site, str) or site not in {"shopping", "gitlab"} for site in args.sites
        )
        or len(set(args.sites)) != len(args.sites)
    ):
        parser.error("sites must name one or two distinct supported sites: shopping, gitlab")
    if args.phase in {"upload", "download"} and (not args.file or not args.remote):
        parser.error(f"{args.phase} requires --file and --remote")
    if args.phase == "command" and (not args.command or not args.remote):
        parser.error("command requires --command and --remote as unique command label")
    if args.phase == "command-status" and not args.remote:
        parser.error("command-status requires --remote as the original command label")
    return args


def main() -> None:
    args = _parse_args()
    env = module._read_env(args.env)
    root = args.root.resolve()
    client = module.SandboxClient(env, api_key_env="OPENAI_API_KEY", receipts=root / "commands")
    state = root / "infra" / "workers"
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (state / "execution.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    record_path = state / "execution.json"
    intent_path = state / "execution.intent.json"
    run_id = root.name
    plan_sha = hashlib.sha256((root / "plans/pipeline.xml").read_bytes()).hexdigest()
    sites_metadata = json.dumps(args.sites, separators=(",", ":"))

    def save(path: Path, data: dict) -> None:
        module._atomic_json(path, data)
        path.chmod(0o600)

    if args.phase == "create" and not record_path.exists():
        if intent_path.exists():
            intent = json.loads(intent_path.read_text())
        else:
            nonce = uuid.uuid4().hex
            extensions = {"project": client.project, "runtimeClassName": "sysbox-runc"}
            intent = {
                "submission_status": "not_submitted",
                "payload": {
                    "image": {"uri": module.IMAGE},
                    "entrypoint": ["sleep", "inf"],
                    "timeout": 259200,
                    "resourceRequests": {"cpu": "4", "memory": "20Gi"},
                    "volumes": [
                        {
                            "mountPath": "/mnt/webarena-verified",
                            "subPath": env.get("ASSET_SUBPATH", module.ASSET_SUBPATH),
                            "readOnly": True,
                        }
                    ],
                    "metadata": {
                        "run_id": run_id,
                        "sites": sites_metadata,
                        "plan_sha256": plan_sha,
                        "session_nonce": nonce,
                        "owner": "codex-session",
                        "purpose": "generated-fixed-assets-v04",
                    },
                    "ports": [
                        {
                            "containerPort": 44772,
                            "name": "execd",
                            "protocol": "tcp",
                            "purpose": "system-service",
                        }
                    ],
                    "extensions": extensions,
                },
            }
            save(intent_path, intent)
        payload = intent["payload"]
        if (
            payload["metadata"]["run_id"] != run_id
            or payload["metadata"]["plan_sha256"] != plan_sha
            or payload["metadata"]["sites"] != sites_metadata
        ):
            raise ValueError("Frozen worker intent differs from this plan")
        if intent["submission_status"] == "rejected":
            raise RuntimeError(
                "Creation was rejected; inspect execution.create-response.json. "
                "Do not submit another POST for this intent"
            )
        if intent["submission_status"] in {"unknown", "accepted"}:
            matches = [
                row
                for row in client.list_sandboxes()
                if row.get("metadata", {}).get("session_nonce")
                == payload["metadata"]["session_nonce"]
            ]
            if len(matches) != 1:
                raise RuntimeError("Creation outcome unresolved; do not submit another POST")
            created = matches[0]
        else:
            # An interrupted request is uncertain until its original nonce is
            # reconciled. A known HTTP rejection remains a terminal attempt.
            intent["submission_status"] = "unknown"
            save(intent_path, intent)
            try:
                created = client.create(payload)
            except requests.RequestException as error:
                response = error.response
                status = response.status_code if response is not None else None
                rejected = status is not None and 400 <= status < 500 and status not in {408, 499}
                outcome = "rejected" if rejected else "unknown"
                body = response.text if response is not None else ""
                message = str(error)
                for name in (
                    "OPEN_SANDBOX_ACCESS_KEY",
                    "OPEN_SANDBOX_SECRET_KEY",
                    "OPENAI_API_KEY",
                ):
                    secret = env.get(name)
                    if secret:
                        body = body.replace(secret, "[REDACTED]")
                        message = message.replace(secret, "[REDACTED]")
                save(
                    state / "execution.create-response.json",
                    {
                        "submission_status": outcome,
                        "http_status": status,
                        "response_body": body,
                        "request_error": message,
                        "run_id": run_id,
                        "session_nonce": payload["metadata"]["session_nonce"],
                    },
                )
                intent["submission_status"] = outcome
                save(intent_path, intent)
                raise RuntimeError(
                    f"Sandbox creation {outcome} (HTTP {status}); "
                    "inspect execution.create-response.json; no second POST was sent"
                ) from error
        intent["submission_status"] = "accepted"
        save(intent_path, intent)
        save(
            record_path,
            {
                "id": created["id"],
                "metadata": payload["metadata"],
                "transport_sha256": hashlib.sha256(TRANSPORT.read_bytes()).hexdigest(),
                "commands": {},
            },
        )
    if not record_path.exists():
        raise ValueError("No session-owned worker record")
    record = json.loads(record_path.read_text())
    if record["metadata"]["sites"] != sites_metadata or record["metadata"]["run_id"] != run_id:
        raise ValueError("Worker record differs from this execution's declared sites")
    detail = client.detail(record["id"])
    if any(
        detail.get("metadata", {}).get(key) != value for key, value in record["metadata"].items()
    ):
        raise ValueError("Remote resource ownership differs from the creation receipt")
    if args.phase in {"create", "status"}:
        print(
            json.dumps(
                {
                    "sandbox_id": record["id"],
                    "sites": args.sites,
                    "state": detail.get("state"),
                    "commands": record["commands"],
                }
            )
        )
    elif args.phase == "dockerd":
        if "dockerd" not in record["commands"]:
            record["commands"]["dockerd"] = client.command(
                record["id"],
                "mkdir -p /mnt/results/v04 && exec dockerd > /mnt/results/v04/dockerd.log 2>&1",
                label="worker-dockerd",
                background=True,
                timeout_ms=259200000,
            )
            save(record_path, record)
        print(json.dumps(client.command_status(record["id"], record["commands"]["dockerd"])))
    elif args.phase == "upload":
        client.upload(record["id"], args.remote, args.file.read_bytes())
        print(
            json.dumps(
                {
                    "uploaded": args.remote,
                    "bytes": args.file.stat().st_size,
                    "sha256": module._sha(args.file),
                }
            )
        )
    elif args.phase == "command":
        if args.remote in record["commands"]:
            raise ValueError("Command label already submitted; inspect its original handle")
        command_id = client.command(
            record["id"], args.command, label=args.remote, background=True, timeout_ms=259200000
        )
        record["commands"][args.remote] = command_id
        save(record_path, record)
        print(json.dumps({"command_id": command_id, "label": args.remote}))
    elif args.phase == "command-status":
        print(json.dumps(client.command_status(record["id"], record["commands"][args.remote])))
    elif args.phase == "download":
        content = client.download(record["id"], args.remote)
        if content is None:
            raise FileNotFoundError(args.remote)
        args.file.parent.mkdir(parents=True, exist_ok=True)
        args.file.write_bytes(content)
        args.file.chmod(0o600)
        print(
            json.dumps(
                {
                    "downloaded": args.remote,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        )


if __name__ == "__main__":
    main()
