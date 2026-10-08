#!/usr/bin/env python3
"""Preflight or release only this campaign's exact, locally archived sandboxes.

Usage: uv run python devs/exps/eval/browsergym.webarena_verified/campaign/cleanup.py
    --root /path/to/results/run --env .env --output-dir /private/cleanup-preflight

The default performs read-only platform observations. Add --delete explicitly
to release the created subset, including a one-worker smoke run. Running cases,
unknown command states, changed ownership, or unsynced evidence refuse release.
Only the recorded owned dockerd command may remain running. No model, evaluator,
remote shell command, or retry is started; all local evidence is retained.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import re
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx

from lite.platforms.sandbox.client import safe_component, write_json

_SPEC = importlib.util.spec_from_file_location(
    "webarena_cleanup_campaign_owner", Path(__file__).with_name("orchestrate.py")
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Cannot load the campaign ownership contract")
orchestrator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(orchestrator)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new private artifact directory"
    )
    parser.add_argument(
        "--delete", action="store_true", help="explicitly release exact owned targets"
    )
    return parser.parse_args(argv)


def _journal_snapshot(root: Path) -> dict[str, str]:
    """Bind recorded ownership, launches, and terminal evidence against new work."""
    paths = [
        *root.glob("worker-intents/*.json"),
        *root.glob("case-intents/*.json"),
        *root.glob("attempts/task-*/*.json"),
        *root.glob("index/task-*/completed.json"),
        *root.glob("index/task-*/archive.sha256"),
    ]
    return {path.relative_to(root).as_posix(): orchestrator._sha(path) for path in sorted(paths)}


def ownership_matches(detail: dict[str, Any], target: dict[str, Any]) -> bool:
    """Require the exact ID, project, and every original creation metadata field."""
    return (
        detail.get("id") == target["sandbox_id"]
        and detail.get("projectID") == target["project"]
        and all(
            (detail.get("metadata") or {}).get(key) == value
            for key, value in target["metadata"].items()
        )
    )


def local_targets(campaign: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve created workers and preserve all terminal, hash-bound local evidence.

    Campaign construction owns immutable source/config/plan validation. Cleanup
    owns the destructive boundary: command identity and retained case evidence.
    An explicitly null no-archive ledger is retained with usage unknown, never
    interpreted as zero model requests or cost.
    """
    root = campaign.root
    evidence: dict[str, str] = {}

    def read(path: Path) -> dict[str, Any]:
        data = path.read_bytes()
        evidence[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    for name in (
        "EXPERIMENT_CONFIG.json",
        "VERSION_MANIFEST.json",
        "worker-plan.json",
        "ASSET_MANIFEST.json",
    ):
        read(root / name)
    cluster = read(root / "sandbox-cluster.json")
    if not set(cluster) <= set(campaign.plan["workers"]):
        raise ValueError("cluster contains an unplanned worker")
    identity = {"run_id": campaign.run_id, "plan_sha256": campaign.plan_sha}
    expected = {
        **identity,
        "runner_sha256": campaign.runner_sha,
        "source_sha256": campaign.version["source_sha256"],
        "image_lock_sha256": campaign.image_sha,
    }
    by_worker = {}
    for worker, record in cluster.items():
        creation = read(root / "worker-intents" / f"{worker}.json")
        metadata = {
            "owner": campaign.config["sandbox"]["owner"],
            **identity,
            "worker": worker,
            "session_nonce": record["nonce"],
            "model": campaign.config["model"],
        }
        if (
            creation["nonce"] != record["nonce"]
            or creation["payload"]["extensions"]["project"] != campaign.client.project
            or any(
                creation["payload"]["metadata"].get(key) != value for key, value in metadata.items()
            )
        ):
            raise ValueError("worker creation intent differs from exact campaign ownership")
        by_worker[worker] = {
            "sandbox_id": safe_component(record["id"]),
            "project": campaign.client.project,
            "metadata": creation["payload"]["metadata"],
            "campaign_root": str(root),
            "commands": [],
        }
    if len({row["sandbox_id"] for row in by_worker.values()}) != len(by_worker):
        raise ValueError("workers overlap sandbox targets")
    if len({row["metadata"]["session_nonce"] for row in by_worker.values()}) != len(by_worker):
        raise ValueError("workers overlap creation nonces")

    ledgers: dict[tuple[int, str], dict[str, Any]] = {}
    ledger_files: dict[tuple[int, str], str] = {}
    no_archive = []
    for path in sorted(root.glob("attempts/task-*/*.json")):
        row = read(path)
        task, worker = row["task_id"], row["worker"]
        if (
            type(task) is not int
            or str(task) not in campaign.plan["tasks"]
            or path.parent.name != f"task-{task:04d}"
            or worker not in by_worker
        ):
            raise ValueError("attempt ledger belongs to an unplanned task or worker")
        bound = {
            **expected,
            "task_id": task,
            "worker": worker,
            "sandbox_id": by_worker[worker]["sandbox_id"],
        }
        if any(row.get(key) != value for key, value in bound.items()) or (
            type(row.get("case_exit_code")) is not int
            or type(row.get("runner_exit_code")) is not int
            or type(row.get("model_started")) is not bool
            or not isinstance(row.get("finished_at"), str)
            or not row["finished_at"]
            or row.get("terminal_status")
            not in {
                "completed",
                "model_protocol_failed",
                "evaluator_error",
                "infrastructure_failed",
            }
            or not re.fullmatch(r"[0-9a-f]{32}", str(row.get("launch_id", "")))
        ):
            raise ValueError("attempt ledger is not complete terminal campaign evidence")
        key = task, row["launch_id"]
        if key in ledgers:
            raise ValueError("attempt launch has duplicate local ledger owners")
        ledgers[key], ledger_files[key] = row, path.relative_to(root).as_posix()
        if row["archive_path"] is None:
            if (
                row["archive_sha256"] is not None
                or row["local_archive_path"] is not None
                or row["model_started"] is not False
                or row["terminal_status"] != "infrastructure_failed"
            ):
                raise ValueError("no-archive attempt violates the synced ledger contract")
            no_archive.append(
                {
                    "task_id": task,
                    "launch_id": row["launch_id"],
                    "ledger": ledger_files[key],
                    "model_usage": None,
                }
            )
        else:
            relative = path.relative_to(root).with_suffix(".tar.gz")
            if (
                row["archive_path"] != "/mnt/results/" + relative.as_posix()
                or Path(row["local_archive_path"]).resolve() != (root / relative).resolve()
                or not re.fullmatch(r"[0-9a-f]{64}", str(row["archive_sha256"]))
                or not (root / relative).is_file()
                or orchestrator._sha(root / relative) != row["archive_sha256"]
            ):
                raise ValueError("attempt archive is missing, unsynced, or changed")
            evidence[relative.as_posix()] = row["archive_sha256"]

    canonical = {}
    for path in sorted(root.glob("index/task-*/completed.json")):
        row = read(path)
        task, worker = row["task_id"], row["worker"]
        if (
            type(task) is not int
            or str(task) not in campaign.plan["tasks"]
            or path.parent.name != f"task-{task:04d}"
            or worker not in by_worker
        ):
            raise ValueError("canonical receipt belongs to an unplanned task or worker")
        campaign._validated_receipt(row, worker, by_worker[worker]["sandbox_id"], task)
        archive = root / "archives" / f"task-{task:04d}.tar.gz"
        if not archive.is_file() or orchestrator._sha(archive) != row["archive_sha256"]:
            raise ValueError("canonical archive is missing, unsynced, or changed")
        matching = [
            ledger
            for (ledger_task, _), ledger in ledgers.items()
            if ledger_task == task
            and ledger["worker"] == worker
            and ledger["archive_sha256"] == row["archive_sha256"]
            and ledger["terminal_status"] == row["terminal_status"]
            and ledger["case_exit_code"] == 0
        ]
        if len(matching) != 1:
            raise ValueError("canonical receipt lacks its unique retained terminal attempt")
        checksum = path.parent / "archive.sha256"
        if checksum.read_text().strip() != row["archive_sha256"]:
            raise ValueError("canonical index archive checksum differs")
        evidence[checksum.relative_to(root).as_posix()] = orchestrator._sha(checksum)
        evidence[archive.relative_to(root).as_posix()] = row["archive_sha256"]
        canonical[task] = row
    for archive in sorted(root.glob("archives/task-*.tar.gz")):
        if archive.relative_to(root).as_posix() in evidence:
            continue
        task = int(archive.name.removeprefix("task-").removesuffix(".tar.gz"))
        digest = orchestrator._sha(archive)
        if not any(
            ledger_task == task and row["archive_sha256"] == digest
            for (ledger_task, _), row in ledgers.items()
        ):
            raise ValueError("local case archive lacks retained attempt evidence")
        evidence[archive.relative_to(root).as_posix()] = digest

    covered_ledgers = set()
    for path in sorted(root.glob("case-intents/*.json")):
        intent = read(path)
        task = intent["task_id"]
        if (
            str(task) not in campaign.plan["tasks"]
            or path.stem != f"task-{task:04d}"
            or any(intent.get(key) != value for key, value in identity.items())
        ):
            raise ValueError("case intent belongs to another campaign or task")
        for attempt in intent["attempts"]:
            worker = attempt["worker"]
            key = task, attempt["launch_id"]
            command = shlex.join(
                [
                    "bash",
                    "/mnt/results/run-worker.sh",
                    worker,
                    attempt["sandbox_id"],
                    "case",
                    attempt["launch_id"],
                    str(task),
                    ",".join(campaign.plan["tasks"][str(task)]["sites"]),
                ]
            )
            if (
                worker not in by_worker
                or attempt["sandbox_id"] != by_worker[worker]["sandbox_id"]
                or type(attempt.get("exit_code")) is not int
                or not attempt.get("command_id")
                or key not in ledgers
                or attempt["command"] != command
                or ledgers[key]["worker"] != worker
                or ledgers[key]["case_exit_code"] != attempt["exit_code"]
            ):
                raise ValueError(
                    "case command is active, ambiguous, or lacks synced terminal evidence"
                )
            covered_ledgers.add(key)
            by_worker[worker]["commands"].append(
                {
                    "command_id": attempt["command_id"],
                    "owner_file": path.relative_to(root).as_posix(),
                    "local_exit_code": attempt["exit_code"],
                }
            )

    for path in sorted(root.glob("worker-intents/*.json")):
        row = read(path)
        worker = path.stem.split("-", 1)[0]
        if worker not in campaign.plan["workers"]:
            raise ValueError("worker command owner is unplanned")
        if worker not in by_worker:
            if path.stem == worker and row["post_attempted"] is False:
                continue  # An explicitly rejected/not-submitted create is not a target.
            raise ValueError("worker creation or command has unresolved sandbox identity")
        target = by_worker[worker]
        if path.stem == worker:
            continue
        if row.get("sandbox_id") != target["sandbox_id"]:
            raise ValueError("worker command changed sandbox ownership")
        if path.name == f"{worker}-stage.json":
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("bundle_sha256", ""))):
                raise ValueError("worker stage proof lacks a bound bundle checksum")
            continue
        if not row.get("command_id"):
            raise ValueError("worker command submission remains ambiguous")
        if path.name not in {f"{worker}-dockerd.json", f"{worker}-prepare.json"}:
            if not re.fullmatch(rf"{worker}-(startup|full)-[1-9][0-9]*\.json", path.name):
                raise ValueError("worker command has an unknown lifecycle owner")
            keys = {
                key
                for key, ledger in ledgers.items()
                if ledger["worker"] == worker and ledger["launch_id"] == row["launch_id"]
            }
            mode = path.stem.split("-")[1]
            command = shlex.join(
                [
                    "bash",
                    "/mnt/results/run-worker.sh",
                    worker,
                    target["sandbox_id"],
                    mode,
                    row["launch_id"],
                ]
            )
            if row["command"] != command:
                raise ValueError("worker model command differs from its recorded launch owner")
            skipped = canonical.get(campaign.plan["workers"][worker]["startup_task"])
            if not keys and not (mode == "startup" and skipped and skipped["worker"] == worker):
                raise ValueError(
                    "worker model launch lacks retained attempt or skipped-case evidence"
                )
            covered_ledgers.update(keys)
        target["commands"].append(
            {
                "command_id": row["command_id"],
                "owner_file": path.relative_to(root).as_posix(),
            }
        )
    if covered_ledgers != set(ledgers):
        raise ValueError("retained attempt has no recorded command owner")
    for target in by_worker.values():
        ids = [row["command_id"] for row in target["commands"]]
        if len(ids) != len(set(ids)):
            raise ValueError("recorded command IDs have duplicate owners")
    journals = _journal_snapshot(root)
    if any(evidence.get(path) != digest for path, digest in journals.items()):
        raise ValueError("local command/evidence journals changed during cleanup validation")
    journal_sha = hashlib.sha256(
        json.dumps(journals, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for target in by_worker.values():
        target["journal_sha256"] = journal_sha
    return list(by_worker.values()), {
        **identity,
        "created_workers": sorted(by_worker),
        "planned_workers": len(campaign.plan["workers"]),
        "planned_tasks": campaign.plan["task_count"],
        "canonical_cases": len(canonical),
        "retained_attempts": len(ledgers),
        "explicit_no_archive_attempts": no_archive,
        "local_evidence_sha256": evidence,
        "journal_sha256": journal_sha,
        "preserved_local_evidence": True,
        "cleanup_model_requests_started": 0,
        "official_scoring_not_required_for_release": True,
    }


def preflight_target(session: Any, client: Any, target: dict[str, Any]) -> dict[str, Any]:
    """Observe current ownership and every recorded command without executing one."""
    result = {"sandbox_id": target["sandbox_id"], "commands": []}
    url = f"{client.base}/v1/sandboxes/{target['sandbox_id']}"
    try:
        response = session.get(url)
        if response.status_code == 404:
            return {**result, "outcome": "already_absent"}
        if response.status_code != 200:
            return {**result, "outcome": "read_before_failed", "http_status": response.status_code}
        if not ownership_matches(response.json(), target):
            return {**result, "outcome": "ownership_changed_refused"}
        for command in target["commands"]:
            status = client.command_status(target["sandbox_id"], command["command_id"])
            persistent = command["owner_file"] == (
                f"worker-intents/{target['metadata']['worker']}-dockerd.json"
            )
            observed = {
                "command_id": command["command_id"],
                "owner_file": command["owner_file"],
                "running": status.get("running"),
                "exit_code": status.get("exit_code"),
                "persistent_infrastructure": persistent,
            }
            result["commands"].append(observed)
            if persistent and status.get("running") is True:
                continue
            if status.get("running") is not False or type(status.get("exit_code")) is not int:
                return {**result, "outcome": "command_live_or_unknown_refused"}
            if "local_exit_code" in command and command["local_exit_code"] != status["exit_code"]:
                return {**result, "outcome": "command_exit_identity_changed_refused"}
        return {**result, "outcome": "ready"}
    except Exception as error:
        return {
            **result,
            "outcome": "observation_failed_refused",
            "error_type": type(error).__name__,
        }


def release_target(session: Any, client: Any, target: dict[str, Any]) -> dict[str, Any]:
    """Recheck one target, issue at most one DELETE, then reconcile only with GET."""
    before = preflight_target(session, client, target)
    if before["outcome"] != "ready":
        return {**before, "delete_requests": 0}
    url = f"{client.base}/v1/sandboxes/{target['sandbox_id']}"
    try:
        journals = _journal_snapshot(Path(target["campaign_root"]))
        actual_sha = hashlib.sha256(
            json.dumps(journals, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if actual_sha != target["journal_sha256"]:
            return {**before, "outcome": "local_journals_changed_refused", "delete_requests": 0}
        last = session.get(url)
        if last.status_code == 404:
            return {**before, "outcome": "already_absent", "delete_requests": 0}
        if last.status_code != 200 or not ownership_matches(last.json(), target):
            return {**before, "outcome": "last_read_ownership_refused", "delete_requests": 0}
    except Exception as error:
        return {
            **before,
            "outcome": "last_read_failed_refused",
            "delete_requests": 0,
            "error_type": type(error).__name__,
        }
    result = {
        **before,
        "delete_requests": 1,
        "delete_http_status": None,
        "delete_request_ambiguous": False,
    }
    try:
        deleted = session.delete(url)
        result["delete_http_status"] = deleted.status_code
    except Exception as error:
        result.update(delete_request_ambiguous=True, delete_error_type=type(error).__name__)
    try:
        after = session.get(url)
        result["readback_http_status"] = after.status_code
        if after.status_code == 404:
            result["outcome"] = (
                "deleted"
                if result["delete_http_status"] in (200, 202, 204)
                else "absent_after_unconfirmed_delete"
            )
            result["persistent_infrastructure_terminated_by_sandbox_release"] = [
                row["command_id"]
                for row in before["commands"]
                if row["persistent_infrastructure"] and row["running"] is True
            ]
        else:
            result["outcome"] = "delete_unconfirmed"
    except Exception as error:
        result.update(outcome="delete_readback_unknown", readback_error_type=type(error).__name__)
    return result


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(mode=0o700)
    with ExitStack() as stack:
        # The existing full dispatcher lock proves no coordinator is still
        # submitting model cases; open it read-only and never create a new lock.
        owner_path = args.root / "full-dispatch-owner.lock"
        if owner_path.exists():
            owner = stack.enter_context(owner_path.open("rb"))
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = orchestrator.Campaign(args.root, args.env)
        targets, proof = local_targets(campaign)
        write_json(
            args.output_dir / "release-plan.json",
            {
                **proof,
                "targets": targets,
                "delete_requested": args.delete,
                "tool_sha256": orchestrator._sha(Path(__file__)),
            },
        )

        def observe(target: dict[str, Any], *, delete: bool) -> dict[str, Any]:
            with httpx.Client(
                headers=campaign.client.headers, timeout=30, trust_env=False, follow_redirects=False
            ) as session:
                return (
                    release_target(session, campaign.client, target)
                    if delete
                    else preflight_target(session, campaign.client, target)
                )

        preflight = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(observe, row, delete=False): row for row in targets}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "sandbox_id": futures[future]["sandbox_id"],
                        "outcome": "observation_failed_refused",
                        "error_type": type(error).__name__,
                    }
                preflight.append(result)
                write_json(args.output_dir / "preflight" / f"{result['sandbox_id']}.json", result)
        write_json(
            args.output_dir / "preflight-summary.json",
            {
                "targets": len(targets),
                "results": preflight,
                "delete_requested": args.delete,
            },
        )
        if not args.delete:
            print(
                json.dumps(
                    {
                        "targets": len(targets),
                        "delete_requests": 0,
                        "ready": sum(row["outcome"] == "ready" for row in preflight),
                        "already_absent": sum(
                            row["outcome"] == "already_absent" for row in preflight
                        ),
                    }
                ),
                flush=True,
            )
            if any(row["outcome"] not in {"ready", "already_absent"} for row in preflight):
                raise SystemExit(2)
            return
        allowed = {
            row["sandbox_id"] for row in preflight if row["outcome"] in {"ready", "already_absent"}
        }
        receipts = [
            dict(row, delete_requests=0) for row in preflight if row["sandbox_id"] not in allowed
        ]
        for row in receipts:
            write_json(args.output_dir / "receipts" / f"{row['sandbox_id']}.json", row)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(observe, row, delete=True): row
                for row in targets
                if row["sandbox_id"] in allowed
            }
            for future in as_completed(futures):
                try:
                    receipt = future.result()
                except Exception as error:
                    receipt = {
                        "sandbox_id": futures[future]["sandbox_id"],
                        "outcome": "release_runner_unknown",
                        "delete_requests": None,
                        "error_type": type(error).__name__,
                    }
                receipts.append(receipt)
                write_json(args.output_dir / "receipts" / f"{receipt['sandbox_id']}.json", receipt)
        confirmed = {"deleted", "already_absent", "absent_after_unconfirmed_delete"}
        summary = {
            "requested": len(targets),
            "confirmed_absent": sum(row["outcome"] in confirmed for row in receipts),
            "unconfirmed_target_ids": [
                row["sandbox_id"] for row in receipts if row["outcome"] not in confirmed
            ],
            "foreign_not_targeted": True,
            "preserved_local_evidence": True,
            "cleanup_model_requests_started": 0,
            "receipts": receipts,
        }
        write_json(args.output_dir / "release-summary.json", summary)
        print(
            json.dumps({key: value for key, value in summary.items() if key != "receipts"}),
            flush=True,
        )
        if summary["unconfirmed_target_ids"]:
            raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not serialize exception messages: transport failures can carry URLs
        # or credentials, and malformed external JSON can include private values.
        print(
            json.dumps({"outcome": "cleanup_failed", "error_type": type(error).__name__}),
            flush=True,
        )
        raise SystemExit(2) from None
