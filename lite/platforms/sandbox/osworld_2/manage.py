"""Submit OSWorld Agent jobs and manage release-matched Sandbox Workers.

Run: python -m lite.platforms.sandbox.osworld_2.manage --help
Copy configs/run.yaml to .sandbox/run.yaml for all commands, including create.
YAML paths are relative to that file; CLI path overrides are relative to the
current directory. No private keys belong in the YAML.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

import yaml

from lite.gym.envs.osworld_2.catalog import DEFAULT_RELEASE, load_release
from lite.platforms.sandbox import lifecycle
from lite.platforms.sandbox.client import (
    NoRedirect,
    SandboxClient,
    feishu_test,
    feishu_webhook,
    load_credentials,
    redact,
    safe_component,
    state_of,
    write_json,
)
from lite.platforms.sandbox.osworld_2.job import Job

# Stable provisioning identity; renaming the Gym entry must not invalidate ownership.
_OWNER = "cua-lite:sandbox.osworld_v2"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".sandbox/run.yaml"))
    parser.add_argument("--release")
    parser.add_argument("--project")
    parser.add_argument("--base-url")
    parser.add_argument("--afs-url")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--credentials", type=Path)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("plan", help="Preview the configured job without credentials or network")
    subs.add_parser("submit", help="Stage the configured job on AFS and start it in Sandbox")
    batch = subs.add_parser(
        "batch", help="Split tasks into owned Sandboxes with bounded concurrency"
    )
    batch.add_argument("--concurrency", type=int, help="Maximum live Sandboxes (default: 4)")
    batch.add_argument("--dry-run", action="store_true", help="Preview without credentials/network")
    batch.add_argument("--resume", type=safe_component, metavar="BATCH_ID")
    batch.add_argument("--poll-interval", type=int, help="Seconds between polls (default: 30)")
    batch.add_argument(
        "--task-timeout", type=int, help="Seconds per task, including boot (default: 14400)"
    )
    create = subs.add_parser("create", help="Create a new Worker from an existing Environment")
    create.add_argument("environment_id", nargs="?", type=safe_component)
    create.add_argument("--dry-run", action="store_true")
    create.add_argument(
        "--agent-job", help="AFS private job path; run it with the Environment's Agent image"
    )
    create.add_argument("--agent-results-root", default="/results/agent-results")
    hook = create.add_mutually_exclusive_group()
    hook.add_argument("--feishu-config", type=Path)
    hook.add_argument("--no-feishu", action="store_true")
    test = subs.add_parser("feishu-test", help="Send one explicitly requested test message")
    test.add_argument("--feishu-config", type=Path)
    for command in ("attach", "status", "delete", "renew", "logs", "sync"):
        sub = subs.add_parser(command)
        sub.add_argument("sandbox_id", type=safe_component)
        if command == "renew":
            sub.add_argument("--seconds", type=int, default=7200)
        elif command == "logs":
            sub.add_argument("--lines", type=int, default=80)
    results = subs.add_parser("results", help="Download one run's persisted Worker artifacts")
    results.add_argument("run_id", type=safe_component)
    results.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def prepare_create(args, config: dict, client: SandboxClient | None, release: str) -> tuple:
    """Resolve the Environment before either direct create or private job staging."""
    environment_id = safe_component(args.environment_id or config["environment_id"])
    if args.agent_job is not None:
        path = PurePosixPath(args.agent_job)
        if path.is_absolute() or ".." in path.parts or str(path) in (".", ""):
            raise ValueError("agent-job must be a relative AFS project path")
        path = PurePosixPath(args.agent_results_root)
        if not path.is_absolute() or ".." in path.parts or ":" in str(path) or str(path) == "/":
            raise ValueError("agent-results-root must be an absolute directory without '..' or ':'")
        if args.dry_run:
            raise ValueError("Agent create needs the live Environment to merge its mounts and env")
    worker_run = uuid.uuid4().hex
    webhook = feishu_webhook(args.feishu_config, disabled=args.no_feishu)
    inherited_webhook = None
    if not args.dry_run:
        template = client.request("GET", "/v1/sandbox-environments/" + environment_id)
        if template.get("env", {}).get("OSWORLD_BENCHMARK_RELEASE") != release:
            raise ValueError(
                f"Environment must declare OSWORLD_BENCHMARK_RELEASE={release}; "
                "publish its matching Worker image and data before creating a Sandbox"
            )
        if args.agent_job:
            image = template.get("env", {}).get("CUA_AGENT_IMAGE")
            if not isinstance(image, str) or not re.fullmatch(
                r"[^\s@]+@sha256:[0-9a-f]{64}", image
            ):
                raise ValueError(
                    "Environment CUA_AGENT_IMAGE must be an OCI image pinned by sha256 digest"
                )
        if not args.no_feishu:
            inherited_webhook = template.get("extensions", {}).get("webhook-feishu")
        if args.agent_job and any(
            PurePosixPath(args.agent_job).is_relative_to(volume["subPath"])
            for volume in template.get("volumes", [])
        ):
            raise ValueError("agent-job must be outside all inherited Worker volume paths")
    payload = {
        "extensions": {"environmentId": environment_id, "project": config["project"]},
        "metadata": {
            "name": f"{release}-worker-{worker_run[:8]}",
            "purpose": "osworld-worker",
            "worker_run": worker_run,
            "benchmark_release": release,
        },
    }
    if webhook:
        payload["extensions"]["webhook-feishu"] = webhook
    elif args.no_feishu:
        payload["extensions"]["webhook-feishu"] = ""
    if args.agent_job:
        payload.update(
            entrypoint=["/bin/bash", "/opt/osworld-build/input/launch_rollout.sh"],
            env={**template.get("env", {}), "OSWORLD_AGENT_RESULTS_ROOT": args.agent_results_root},
            volumes=[
                *template.get("volumes", []),
                {"subPath": args.agent_job, "mountPath": "/opt/osworld-job", "readOnly": True},
            ],
        )
    effective_webhook = webhook or inherited_webhook
    if effective_webhook and client is not None:
        client.secrets.extend((effective_webhook, effective_webhook.rsplit("/", 1)[-1]))
    return payload, effective_webhook


def create(
    args, config: dict, client: SandboxClient | None, release: str, *,
    prepared: tuple | None = None, submission: dict | None = None,
) -> dict:
    """Create once and immediately record the returned resource's local ownership."""
    payload, effective_webhook = prepared or prepare_create(args, config, client, release)
    if args.dry_run:
        return redact(payload, [effective_webhook] if effective_webhook else [])
    details = {
        "release": release,
        "agent_in_sandbox": bool(args.agent_job),
    }
    if args.agent_job:
        details["agent_image"] = payload["env"]["CUA_AGENT_IMAGE"]
    if submission is not None:
        details["submission"] = submission
        details["feishu_webhook_verified"] = False
    return lifecycle.create(
        args.state_dir, client, payload, owner=_OWNER, details=details, webhook=effective_webhook,
    )


def submit(
    args, config: dict, client: SandboxClient, release: str, job: Job,
    *, batch_run: str | None = None,
) -> dict:
    """Validate, stage an immutable private job, create once, and verify Feishu."""
    from lite.agents.extensions.api_config import load_profile

    if job.selection["unsupported"] and not (
        job.selection["skipped"] or job.selection["verification_mode"]
    ):
        raise ValueError(
            "Unsupported tasks in plan; select supported tasks or set skip_unsupported"
        )
    if not job.selection["scheduled_task_ids"]:
        raise ValueError("No tasks remain to submit")
    model = load_profile(job.model_config, job.model_profile)
    client.secrets.append(model["api_key"])
    if model["api_type"] != "openai_chat" and any(
        flag in job.rollout_args for flag in ("--history-turns", "--full-history")
    ):
        raise ValueError("history_turns/full_history apply to the Chat gateway only")
    args.environment_id = None
    args.dry_run = False
    args.no_feishu = False
    args.agent_job = job.afs_path
    args.agent_results_root = str(job.results_root)
    prepared = prepare_create(args, config, client, release)
    payload, webhook = prepared
    if not webhook:
        raise ValueError("submit requires template Feishu notifications or feishu_config")
    output_mounts = [v for v in payload["volumes"] if v["mountPath"] == "/results"]
    if (
        len(output_mounts) != 1 or output_mounts[0]["subPath"] != config["results_prefix"]
        or output_mounts[0].get("readOnly", False)
    ):
        raise ValueError("results_prefix must match the Environment's writable /results mount")

    submission = {
        **job.summary(), "afs_url": client.afs_url, "results_prefix": config["results_prefix"],
        "agent_image": payload["env"]["CUA_AGENT_IMAGE"],
    }
    if batch_run is not None:
        submission["batch_run"] = batch_run
    # Keep the nonce even if the create response is lost; never retry POST blindly.
    directory = args.state_dir / "jobs" / job.experiment
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(directory / "submission.json", {
        **submission, "project": config["project"], "base_url": client.base_url,
        "worker_run": payload["metadata"]["worker_run"],
    })
    write_json(directory / "job.json", {
        "rollout_args": job.rollout_args,
        "model": {"model_id": model["model"], "api_type": model["api_type"]},
    })
    private_model = model if job.model_profile is None else {
        "default_model": job.model_profile, "models": {job.model_profile: model},
    }
    write_json(directory / "model-api.json", private_model)
    client.afs(
        "copy", str(directory), f"afs:{job.afs_path}", "--immutable", "--ignore-times",
        "--include", "/job.json", "--include", "/model-api.json",
    )
    try:
        record = create(args, config, client, release, prepared=prepared, submission=submission)
    except RuntimeError as error:
        raise RuntimeError(
            f"{error}; do not resubmit blindly. Check worker_run="
            f"{payload['metadata']['worker_run']} in Portal; staging record: {directory}"
        ) from None
    return lifecycle.verify_feishu(client, record, webhook)


def worker_health(sandbox_id: str, client: SandboxClient) -> tuple[str, dict]:
    """Discover the Worker identity through its Sandbox-scoped, read-only health endpoint."""
    endpoint = client.endpoint(sandbox_id, 8800)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(endpoint + "/healthz", timeout=30) as stream:
        health = json.load(stream)
    return endpoint, health


def attach(args, config: dict, client: SandboxClient, release: str) -> dict:
    """Read this Worker's AFS control record and write a private connection file."""
    endpoint, health = worker_health(args.sandbox_id, client)
    if health["release"] != release:
        raise ValueError("Worker release does not match the packaged OSWorld release manifest")
    worker_id = safe_component(health["worker_id"])
    control = json.loads(
        client.afs("cat", f"afs:{config['results_prefix']}/_workers/{worker_id}/control.json")
    )
    if control["worker_id"] != worker_id or control["release"] != release:
        raise ValueError("Worker control record does not match its live identity and release")
    if not isinstance(control.get("token"), str) or not control["token"]:
        raise ValueError("Worker control record has no session token")
    path = args.state_dir / f"connection-{args.sandbox_id}.json"
    write_json(
        path,
        {
            "endpoint": endpoint,
            "token": control["token"],
            "worker_id": worker_id,
            "sandbox_id": args.sandbox_id,
            "release": release,
            "results_prefix": config["results_prefix"],
        },
    )
    return {
        "connection_file": str(path),
        "worker_id": worker_id,
        "state": health["state"],
        "release": release,
    }


def sync(record: dict, client: SandboxClient) -> dict:
    """Mirror a submission's persisted outputs, including failures before env initialization."""
    submission = record["submission"]
    experiment = safe_component(submission["experiment"])
    output = Path(submission["results_root"])
    client.afs(
        "copy", f"afs:{submission['results_prefix']}/agent-results", str(output),
        "--include", f"/{experiment}/**", "--include", f"/_environment/{experiment}/**",
        "--ignore-times", "--ignore-size", "--multi-thread-streams", "0", "--transfers", "4",
    )
    return {"sandbox_id": record["sandbox_id"], "results": str(output / experiment),
            "environment_results": str(output / "_environment" / experiment)}


def main(argv=None) -> None:
    """Resolve declared deployment settings, run one command, and redact public output."""
    args = _parse_args(argv)
    client = None
    secrets = []
    try:
        if args.command == "feishu-test":
            webhook = feishu_webhook(args.feishu_config)
            if webhook is None:
                raise ValueError("Pass --feishu-config or set SANDBOX_FEISHU_WEBHOOK")
            print(json.dumps(feishu_test(webhook), ensure_ascii=False, indent=2))
            return
        args.config = args.config.expanduser().resolve()
        config = yaml.safe_load(args.config.read_text())
        if not isinstance(config, dict):
            raise ValueError("Deployment config must be a mapping")
        for name in ("project", "base_url", "afs_url"):
            if getattr(args, name) is not None:
                config[name] = getattr(args, name)
        required = {"project", "base_url", "afs_url", "environment_id", "results_prefix"}
        optional = {"release", "credentials", "state_dir", "feishu_config", "job"}
        if not required <= set(config) or set(config) - required - optional or not all(
            isinstance(config.get(name), str) and config[name] for name in required
        ):
            raise ValueError(
                "Config requires " + ", ".join(sorted(required))
                + "; optional keys: " + ", ".join(sorted(optional))
            )
        for name in ("credentials", "state_dir", "feishu_config"):
            cli_path = getattr(args, name, None)
            if cli_path is not None:
                path = cli_path.expanduser().resolve()
            elif name in config:
                if not isinstance(config[name], str) or not config[name].strip():
                    raise ValueError(f"{name} must be a file/directory path string")
                path = (args.config.parent / Path(config[name]).expanduser()).resolve()
            else:
                path = None
            setattr(args, name, path)
        args.state_dir = args.state_dir or Path(".sandbox").resolve()
        safe_component(config["project"])
        prefix = PurePosixPath(config["results_prefix"])
        if prefix.is_absolute() or ".." in prefix.parts or str(prefix) in (".", ""):
            raise ValueError("results_prefix must be a relative AFS project path")
        selected = load_release(args.release or config.get("release", DEFAULT_RELEASE))
        release = selected.name
        if args.command in ("plan", "submit"):
            job = Job.from_config(config.get("job"), args.config.parent, selected)
            if args.command == "plan":
                print(json.dumps({
                    "project": config["project"], "environment_id": config["environment_id"],
                    "feishu": "config_file" if args.feishu_config else "inherit_template",
                    "state_dir": str(args.state_dir), **job.summary(),
                }, ensure_ascii=False, indent=2))
                return
        if args.command == "batch":
            from lite.platforms.sandbox import batch
            from lite.platforms.sandbox.osworld_2.batch import OSWorldBatch, prepare

            batch_state = prepare(args, config, selected)
            if args.dry_run:
                print(json.dumps(batch_state, ensure_ascii=False, indent=2))
                return
        if args.command == "create" and args.dry_run:
            print(json.dumps(create(args, config, None, release), ensure_ascii=False, indent=2))
            return
        args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        args.state_dir.chmod(0o700)
        if args.command == "sync":
            path = args.state_dir / f"worker-{args.sandbox_id}.json"
            record = json.loads(path.read_text())
            if record.get("managed_by") != _OWNER or record.get("sandbox_id") != args.sandbox_id:
                raise ValueError("Not an owned Sandbox submission record")
            submission = record.get("submission")
            if not submission:
                raise ValueError(
                    "sync requires a Sandbox created by submit; use results for raw Workers"
                )
            # Later edits to run.yaml must not redirect downloads for an existing job.
            config.update(
                project=record["project"], base_url=record["base_url"],
                afs_url=submission["afs_url"], results_prefix=submission["results_prefix"],
            )
        credentials = load_credentials(args.credentials)
        secrets.extend(credentials.values())
        client = SandboxClient(
            base_url=config["base_url"],
            project=config["project"],
            afs_url=config["afs_url"],
            credentials=credentials,
        )
        if args.command == "submit":
            result = submit(args, config, client, release, job)
        elif args.command == "batch":
            result = batch.run(
                args, client, batch_state, OSWorldBatch(args, config, client, batch_state),
            )
            result["skipped"] = batch_state["selection"]["skipped"]
        elif args.command == "create":
            result = create(args, config, client, release)
        elif args.command == "sync":
            result = sync(record, client)
        elif args.command == "attach":
            result = attach(args, config, client, release)
        elif args.command == "delete":
            result = lifecycle.delete(
                args.state_dir, args.sandbox_id, client, project=config["project"], owner=_OWNER,
            )
        elif args.command == "results":
            output = args.output.expanduser().resolve()
            source = f"afs:{config['results_prefix']}/runs/{args.run_id}"
            mutable_artifacts = (
                "/{episode.json,result.json,result.txt,trajectory.jsonl,runtime.log}"
            )
            # Keep per-step artifacts incremental with normal transfer checks.
            client.afs(
                "copy", source, str(output), "--filter", f"- {mutable_artifacts}",
                "--transfers", "4",
            )
            # Mutable files can change after WebDAV listing. Fetch one complete
            # response, retaining HTTP error checks and atomic local replacement.
            client.afs(
                "copy", source, str(output), "--filter", f"+ {mutable_artifacts}",
                "--filter", "- **", "--ignore-times", "--ignore-size",
                "--multi-thread-streams", "0", "--transfers", "4",
            )
            result = {"results": str(output)}
        elif args.command == "logs":
            if not 1 <= args.lines <= 2000:
                raise ValueError("lines must be between 1 and 2000")
            response = client.request(
                "GET", f"/v1/logs/sandboxes/{args.sandbox_id}?since=1h&limit={args.lines}"
            )
            result = {
                "sandbox_id": args.sandbox_id,
                "items": [
                    {key: item[key] for key in ("time", "container", "message") if key in item}
                    for item in response.get("items", [])
                ],
                "has_more": response.get("hasMore"),
            }
        else:
            path = "/v1/sandboxes/" + args.sandbox_id
            if args.command == "renew":
                if args.seconds <= 0:
                    raise ValueError("renewal seconds must be positive")
                value = client.request(
                    "POST", path + "/renew-expiration", {"timeout": args.seconds}
                )
            else:
                value = client.request("GET", path)
            result = {
                "sandbox_id": args.sandbox_id,
                "state": state_of(value),
                "expiresAt": value.get("expiresAt"),
            }
        print(json.dumps(redact(result, client.secrets), ensure_ascii=False, indent=2))
        if args.command == "batch" and result["failed"]:
            raise SystemExit(1)
    except (OSError, ValueError, RuntimeError, KeyError, yaml.YAMLError) as error:
        raise SystemExit(redact(str(error), client.secrets if client else secrets)) from None


if __name__ == "__main__":
    main()
