"""Manage only this generated batch's newly created single-site workers.

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

import sandbox as module

TRANSPORT = Path(module.__file__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--site", choices=("shopping_admin", "gitlab", "shopping", "reddit", "map", "wikipedia"), required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--file", type=Path)
    parser.add_argument("--remote")
    parser.add_argument("--command")
    parser.add_argument("phase", choices=("create", "status", "dockerd", "upload", "command", "command-status", "download"))
    args = parser.parse_args()
    env = module._read_env(args.env)
    root = args.root.resolve()
    client = module.SandboxClient(env, api_key_env="OPENAI_API_KEY", receipts=root/'commands')
    state = root / "infra" / "workers"
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (state / f"{args.site}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    record_path = state / f"{args.site}.json"
    intent_path = state / f"{args.site}.intent.json"
    run_id = root.name
    plan_sha = hashlib.sha256((root / "plans/v03_pipeline.xml").read_bytes()).hexdigest()

    def save(path: Path, data: dict) -> None:
        module._atomic_json(path, data)
        path.chmod(0o600)

    if args.phase == "create" and not record_path.exists():
        if intent_path.exists():
            intent = json.loads(intent_path.read_text())
        else:
            nonce = uuid.uuid4().hex
            extensions = {"project": client.project, "runtimeClassName": "sysbox-runc"}
            if args.site == "map":
                extensions["fuse"] = "true"
            intent = {"post_attempted": False, "payload": {
                "image": {"uri": module.IMAGE}, "entrypoint": ["sleep", "inf"],
                "timeout": 259200, "resourceRequests": {"cpu": "4", "memory": "20Gi"},
                "volumes": [{"mountPath": "/mnt/webarena-verified", "subPath": env.get("ASSET_SUBPATH", module.ASSET_SUBPATH), "readOnly": True}],
                "metadata": {"run_id": run_id, "worker": args.site, "plan_sha256": plan_sha,
                             "session_nonce": nonce, "owner": "codex-session", "purpose": "generated-fixed-assets-single-site-v03"},
                "ports": [{"containerPort": 44772, "name": "execd", "protocol": "tcp", "purpose": "system-service"}],
                "extensions": extensions,
            }}
            save(intent_path, intent)
        payload = intent["payload"]
        if payload["metadata"]["run_id"] != run_id or payload["metadata"]["plan_sha256"] != plan_sha:
            raise ValueError("Frozen worker intent differs from this plan")
        if intent["post_attempted"]:
            matches = [row for row in client.list_sandboxes() if row.get("metadata", {}).get("session_nonce") == payload["metadata"]["session_nonce"]]
            if len(matches) != 1:
                raise RuntimeError("Creation outcome unresolved; do not submit another POST")
            created = matches[0]
        else:
            intent["post_attempted"] = True
            save(intent_path, intent)
            created = client.create(payload)
        save(record_path, {"id": created["id"], "metadata": payload["metadata"],
                           "transport_sha256": hashlib.sha256(TRANSPORT.read_bytes()).hexdigest(),
                           "commands": {}})
    if not record_path.exists():
        raise ValueError("No session-owned worker record")
    record = json.loads(record_path.read_text())
    detail = client.detail(record["id"])
    if any(detail.get("metadata", {}).get(key) != value for key, value in record["metadata"].items()):
        raise ValueError("Remote resource ownership differs from the creation receipt")
    if args.phase in {"create", "status"}:
        print(json.dumps({"sandbox_id": record["id"], "site": args.site, "state": detail.get("state"), "commands": record["commands"]}))
    elif args.phase == "dockerd":
        if "dockerd" not in record["commands"]:
            record["commands"]["dockerd"] = client.command(record["id"], "mkdir -p /mnt/results/v03 && exec dockerd > /mnt/results/v03/dockerd.log 2>&1", label='worker-dockerd', background=True, timeout_ms=259200000)
            save(record_path, record)
        print(json.dumps(client.command_status(record["id"], record["commands"]["dockerd"])))
    elif args.phase == "upload":
        if not args.file or not args.remote:
            parser.error("upload requires --file and --remote")
        client.upload(record["id"], args.remote, args.file.read_bytes())
        print(json.dumps({"uploaded": args.remote, "bytes": args.file.stat().st_size, "sha256": module._sha(args.file)}))
    elif args.phase == "command":
        if not args.command or not args.remote:
            parser.error("command requires --command and --remote as unique command label")
        if args.remote in record["commands"]:
            raise ValueError("Command label already submitted; inspect its original handle")
        command_id = client.command(record["id"], args.command, label=args.remote, background=True, timeout_ms=259200000)
        record["commands"][args.remote] = command_id
        save(record_path, record)
        print(json.dumps({"command_id": command_id, "label": args.remote}))
    elif args.phase == "command-status":
        print(json.dumps(client.command_status(record["id"], record["commands"][args.remote])))
    elif args.phase == "download":
        if not args.file or not args.remote:
            parser.error("download requires --file and --remote")
        content = client.download(record["id"], args.remote)
        if content is None:
            raise FileNotFoundError(args.remote)
        args.file.parent.mkdir(parents=True, exist_ok=True)
        args.file.write_bytes(content)
        args.file.chmod(0o600)
        print(json.dumps({"downloaded": args.remote, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}))


if __name__ == "__main__":
    main()
