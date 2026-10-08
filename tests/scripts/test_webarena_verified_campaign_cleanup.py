from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import io
import json
import shlex
import tarfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2] / "devs/exps/eval/browsergym.webarena_verified/campaign/cleanup.py"
)
SPEC = importlib.util.spec_from_file_location("webarena_verified_campaign_cleanup", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)

# Reuse the real campaign's manifest fixture, rather than reproducing a second
# ownership/config contract in cleanup tests.
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "webarena_cleanup_campaign_fixture",
    Path(__file__).with_name("test_webarena_verified_campaign_orchestrator.py"),
)
assert FIXTURE_SPEC is not None and FIXTURE_SPEC.loader is not None
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)


class Response:
    def __init__(self, status: int, value: dict | None = None):
        self.status_code = status
        self.value = value

    def json(self):
        return self.value


class Session:
    def __init__(self, reads, *, delete_status=204, delete_error=None):
        self.reads = list(reads)
        self.delete_status = delete_status
        self.delete_error = delete_error
        self.gets = []
        self.deletes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get(self, url):
        self.gets.append(url)
        result = self.reads.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def delete(self, url):
        self.deletes.append(url)
        if self.delete_error:
            raise self.delete_error
        return Response(self.delete_status)


class Client:
    base = "https://sandbox.example.test"
    project = "test"
    headers = {}

    def __init__(self, statuses=None):
        self.statuses = statuses or {}
        self.observed = []

    def command_status(self, sandbox, command):
        self.observed.append((sandbox, command))
        result = self.statuses.get(command, {"running": False, "exit_code": 0})
        if isinstance(result, Exception):
            raise result
        return result

    def command(self, *args, **kwargs):
        pytest.fail("cleanup must never start a remote command")

    def create(self, *args, **kwargs):
        pytest.fail("cleanup must never create a sandbox")

    def upload(self, *args, **kwargs):
        pytest.fail("cleanup must never upload or change remote evidence")


@pytest.fixture(autouse=True)
def no_real_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("cleanup regressions must not call a real remote service")

    for method in ("get", "post", "delete"):
        monkeypatch.setattr(cleanup.orchestrator.requests, method, forbidden)
    monkeypatch.setattr(cleanup.httpx.Client, "send", forbidden)


@pytest.fixture
def campaign(tmp_path):
    owner = fixtures.campaign.__wrapped__(tmp_path)
    owner.client = fixtures.FakeClient()
    owner.create(["01"])
    owner.client = Client()
    return owner


@pytest.fixture
def target(tmp_path):
    return {
        "sandbox_id": "owned-1",
        "project": "test",
        "metadata": {
            "owner": "cua-lite-webarena",
            "worker": "01",
            "run_id": "fixture-run",
            "plan_sha256": "1" * 64,
            "model": "fixture-model",
            "session_nonce": "2" * 32,
            "selection": "subset",
            "task_count": "1",
            "benchmark_version": "1.2.3",
        },
        "campaign_root": str(tmp_path),
        "journal_sha256": hashlib.sha256(b"{}").hexdigest(),
        "commands": [
            {
                "command_id": "case-command",
                "owner_file": "case-intents/task-0000.json",
                "local_exit_code": 0,
            }
        ],
    }


def detail(target):
    return {
        "id": target["sandbox_id"],
        "projectID": target["project"],
        "metadata": copy.deepcopy(target["metadata"]),
    }


def retain_case(campaign, *, canonical=True, no_archive=False, startup=False, terminal="completed"):
    root = campaign.root
    launch = "3" * 32
    folder = root / "attempts" / "task-0000"
    folder.mkdir(parents=True)
    path = folder / f"fixture-{launch}.json"
    archive = path.with_suffix(".tar.gz")
    if not no_archive:
        with tarfile.open(archive, "w:gz") as bundle:
            content = b'{"terminal_status":"fixture"}\n'
            member = tarfile.TarInfo("task-0/final_result.json")
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    else:
        digest = None
    identity = {
        "run_id": campaign.run_id,
        "plan_sha256": campaign.plan_sha,
        "runner_sha256": campaign.runner_sha,
        "source_sha256": campaign.version["source_sha256"],
        "image_lock_sha256": campaign.image_sha,
        "worker": "01",
        "sandbox_id": "owned-1",
        "task_id": 0,
    }
    row = {
        **identity,
        "launch_id": launch,
        "started_at": "2026-10-08T00:00:00Z",
        "finished_at": "2026-10-08T00:01:00Z",
        "runner_exit_code": 0 if canonical else 1,
        "case_exit_code": 0 if canonical else 1,
        "model_started": not no_archive,
        "terminal_status": terminal if canonical else "infrastructure_failed",
        "archive_path": "/mnt/results/" + archive.relative_to(root).as_posix()
        if not no_archive
        else None,
        "archive_sha256": digest,
        "local_archive_path": str(archive) if not no_archive else None,
    }
    path.write_text(json.dumps(row))
    if canonical:
        index = root / "index" / "task-0000"
        index.mkdir(parents=True)
        receipt = {
            **identity,
            "terminal_status": terminal,
            "archive_sha256": digest,
            "score": None if terminal == "evaluator_error" else 1,
        }
        (index / "completed.json").write_text(json.dumps(receipt))
        (index / "archive.sha256").write_text(digest + "\n")
        (root / "archives").mkdir()
        (root / "archives" / "task-0000.tar.gz").write_bytes(archive.read_bytes())
    if startup:
        intent = root / "worker-intents" / "01-startup-1.json"
        command = shlex.join(
            ["bash", "/mnt/results/run-worker.sh", "01", "owned-1", "startup", launch]
        )
        intent.write_text(
            json.dumps(
                {
                    "sandbox_id": "owned-1",
                    "launch_id": launch,
                    "command": command,
                    "command_id": "startup-command",
                }
            )
        )
    else:
        folder = root / "case-intents"
        folder.mkdir()
        intent = folder / "task-0000.json"
        command = shlex.join(
            [
                "bash",
                "/mnt/results/run-worker.sh",
                "01",
                "owned-1",
                "case",
                launch,
                "0",
                "shopping_admin",
            ]
        )
        intent.write_text(
            json.dumps(
                {
                    "run_id": campaign.run_id,
                    "plan_sha256": campaign.plan_sha,
                    "task_id": 0,
                    "attempts": [
                        {
                            "worker": "01",
                            "sandbox_id": "owned-1",
                            "launch_id": launch,
                            "command": command,
                            "command_id": "case-command",
                            "exit_code": row["case_exit_code"],
                        }
                    ],
                }
            )
        )
    return path, intent


def test_cli_defaults_read_only():
    args = cleanup._parse_args(["--root", "results/run", "--output-dir", "private/preflight"])
    assert args.env == Path(".env")
    assert not args.delete
    assert cleanup._parse_args(["--root", "r", "--output-dir", "o", "--delete"]).delete


def test_created_subset_smoke_needs_no_full_completion(campaign):
    targets, proof = cleanup.local_targets(campaign)
    assert len(targets) == 1 and targets[0]["sandbox_id"] == "owned-1"
    assert proof["created_workers"] == ["01"]
    assert proof["planned_workers"] == 2 and proof["planned_tasks"] == 2
    assert proof["canonical_cases"] == 0
    assert not (campaign.root / "PHASE_STATUS.json").exists()


@pytest.mark.parametrize("startup", [False, True])
@pytest.mark.parametrize("terminal", ["completed", "model_protocol_failed", "evaluator_error"])
def test_scored_or_unscored_canonical_is_hash_bound(campaign, startup, terminal):
    retain_case(campaign, startup=startup, terminal=terminal)
    targets, proof = cleanup.local_targets(campaign)
    assert proof["canonical_cases"] == proof["retained_attempts"] == 1
    assert len(targets[0]["commands"]) == 1
    assert "archives/task-0000.tar.gz" in proof["local_evidence_sha256"]
    assert "index/task-0000/archive.sha256" in proof["local_evidence_sha256"]


@pytest.mark.parametrize("no_archive", [False, True])
def test_partial_infra_attempt_is_retained_without_inventing_zero_usage(campaign, no_archive):
    retain_case(campaign, canonical=False, no_archive=no_archive)
    targets, proof = cleanup.local_targets(campaign)
    assert targets[0]["commands"][0]["local_exit_code"] == 1
    assert proof["canonical_cases"] == 0 and proof["retained_attempts"] == 1
    if no_archive:
        assert proof["explicit_no_archive_attempts"][0]["model_usage"] is None
    else:
        assert proof["explicit_no_archive_attempts"] == []


@pytest.mark.parametrize(
    "surface",
    [
        "cluster_nonce",
        "creation_owner",
        "creation_project",
        "case_run",
        "case_worker",
        "case_sandbox",
        "case_command",
        "case_exit",
        "case_id",
        "ledger_source",
        "ledger_launch",
        "ledger_archive",
        "ledger_finished",
        "canonical_archive",
        "canonical_ledger",
        "canonical_checksum",
        "orphan_ledger",
    ],
)
def test_foreign_or_unsynced_local_evidence_refuses(campaign, surface):
    ledger, intent = retain_case(campaign)
    if surface == "cluster_nonce":
        path = campaign.root / "sandbox-cluster.json"
        row = json.loads(path.read_text())
        row["01"]["nonce"] = "foreign"
    elif surface.startswith("creation_"):
        path = campaign.root / "worker-intents" / "01.json"
        row = json.loads(path.read_text())
        if surface == "creation_owner":
            row["payload"]["metadata"]["owner"] = "foreign"
        else:
            row["payload"]["extensions"]["project"] = "foreign"
    elif surface.startswith("case_"):
        path, row = intent, json.loads(intent.read_text())
        if surface == "case_run":
            row["run_id"] = "foreign"
        else:
            name = {
                "case_worker": "worker",
                "case_sandbox": "sandbox_id",
                "case_command": "command",
                "case_exit": "exit_code",
                "case_id": "command_id",
            }[surface]
            row["attempts"][0][name] = None if surface in {"case_id", "case_exit"} else "foreign"
    elif surface.startswith("ledger_"):
        path, row = ledger, json.loads(ledger.read_text())
        name = {
            "ledger_source": "source_sha256",
            "ledger_launch": "launch_id",
            "ledger_archive": "archive_sha256",
            "ledger_finished": "finished_at",
        }[surface]
        row[name] = "foreign" if surface != "ledger_finished" else ""
    elif surface == "canonical_archive":
        (campaign.root / "archives" / "task-0000.tar.gz").write_bytes(b"changed")
        path, row = None, None
    elif surface == "canonical_ledger":
        ledger.unlink()
        path, row = None, None
    elif surface == "canonical_checksum":
        (campaign.root / "index" / "task-0000" / "archive.sha256").write_text("foreign\n")
        path, row = None, None
    else:
        intent.unlink()
        path, row = None, None
    if path:
        path.write_text(json.dumps(row))
    with pytest.raises((ValueError, RuntimeError)):
        cleanup.local_targets(campaign)


def test_missing_attempt_archive_refuses(campaign):
    ledger, _ = retain_case(campaign, canonical=False)
    ledger.with_suffix(".tar.gz").unlink()
    with pytest.raises(ValueError, match="attempt archive"):
        cleanup.local_targets(campaign)


def test_explicit_no_archive_does_not_hide_started_model(campaign):
    path, _ = retain_case(campaign, canonical=False, no_archive=True)
    row = json.loads(path.read_text())
    row["model_started"] = True
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="no-archive"):
        cleanup.local_targets(campaign)


@pytest.mark.parametrize("name", ["01-dockerd", "01-prepare", "01-startup-1", "01-unknown"])
def test_submitted_worker_command_without_id_refuses(campaign, name):
    (campaign.root / "worker-intents" / f"{name}.json").write_text(
        json.dumps({"sandbox_id": "owned-1"})
    )
    with pytest.raises(ValueError, match="ambiguous"):
        cleanup.local_targets(campaign)


def test_unsynced_startup_model_command_refuses(campaign):
    launch = "4" * 32
    (campaign.root / "worker-intents" / "01-startup-1.json").write_text(
        json.dumps(
            {
                "sandbox_id": "owned-1",
                "launch_id": launch,
                "command_id": "startup",
                "command": shlex.join(
                    ["bash", "/mnt/results/run-worker.sh", "01", "owned-1", "startup", launch]
                ),
            }
        )
    )
    with pytest.raises(ValueError, match="lacks retained"):
        cleanup.local_targets(campaign)


def test_terminal_startup_skip_uses_existing_synced_case(campaign):
    retain_case(campaign, startup=True)
    launch = "4" * 32
    (campaign.root / "worker-intents" / "01-startup-2.json").write_text(
        json.dumps(
            {
                "sandbox_id": "owned-1",
                "launch_id": launch,
                "command_id": "skipped-startup",
                "command": shlex.join(
                    ["bash", "/mnt/results/run-worker.sh", "01", "owned-1", "startup", launch]
                ),
            }
        )
    )
    targets, proof = cleanup.local_targets(campaign)
    assert len(targets[0]["commands"]) == 2 and proof["retained_attempts"] == 1


@pytest.mark.parametrize("submitted", [False, True])
def test_uncreated_worker_requires_explicit_unsubmitted_intent(campaign, submitted):
    (campaign.root / "worker-intents" / "02.json").write_text(
        json.dumps({"post_attempted": submitted})
    )
    if submitted:
        with pytest.raises(ValueError, match="unresolved"):
            cleanup.local_targets(campaign)
    else:
        assert len(cleanup.local_targets(campaign)[0]) == 1


@pytest.mark.parametrize(
    "field",
    [
        "id",
        "projectID",
        "owner",
        "run_id",
        "worker",
        "model",
        "session_nonce",
        "plan_sha256",
        "selection",
        "task_count",
    ],
)
def test_live_identity_change_is_refused(target, field):
    live = detail(target)
    if field in {"id", "projectID"}:
        live[field] = "foreign"
    else:
        live["metadata"][field] = "foreign"
    session = Session([Response(200, live)])
    result = cleanup.release_target(session, Client(), target)
    assert result["outcome"] == "ownership_changed_refused"
    assert result["delete_requests"] == 0 and session.deletes == []


@pytest.mark.parametrize(
    "status",
    [
        {"running": True, "exit_code": None},
        {},
        {"running": False},
        {"running": False, "exit_code": True},
        {"running": None, "exit_code": 0},
    ],
)
def test_active_or_unknown_non_daemon_commands_are_refused(target, status):
    session = Session([Response(200, detail(target))])
    result = cleanup.release_target(session, Client({"case-command": status}), target)
    assert result["outcome"] == "command_live_or_unknown_refused"
    assert session.deletes == []


def test_local_remote_exit_mismatch_refuses(target):
    result = cleanup.preflight_target(
        Session([Response(200, detail(target))]),
        Client({"case-command": {"running": False, "exit_code": 1}}),
        target,
    )
    assert result["outcome"] == "command_exit_identity_changed_refused"


@pytest.mark.parametrize(
    "owner",
    [
        "worker-intents/01-dockerd.json",
        "worker-intents/02-dockerd.json",
        "worker-intents/01-prepare.json",
        "case-intents/task-0000.json",
    ],
)
def test_only_exact_owned_dockerd_may_remain_running(target, owner):
    target["commands"] = [{"command_id": "daemon", "owner_file": owner}]
    result = cleanup.preflight_target(
        Session([Response(200, detail(target))]),
        Client({"daemon": {"running": True, "exit_code": None}}),
        target,
    )
    assert result["outcome"] == (
        "ready" if owner == "worker-intents/01-dockerd.json" else "command_live_or_unknown_refused"
    )


@pytest.mark.parametrize("surface", ["sandbox", "command"])
def test_observation_error_is_sanitized_and_refused(target, surface):
    error = RuntimeError("private credential must not appear")
    session = Session([error if surface == "sandbox" else Response(200, detail(target))])
    client = Client({"case-command": error} if surface == "command" else None)
    result = cleanup.release_target(session, client, target)
    assert result["outcome"] == "observation_failed_refused"
    assert result["error_type"] == "RuntimeError" and "private credential" not in json.dumps(result)
    assert session.deletes == []


@pytest.mark.parametrize("at_last_read", [False, True])
def test_already_absent_requires_no_delete(target, at_last_read):
    reads = [Response(200, detail(target)), Response(404)] if at_last_read else [Response(404)]
    session = Session(reads)
    result = cleanup.release_target(session, Client(), target)
    assert result["outcome"] == "already_absent" and result["delete_requests"] == 0
    assert session.deletes == []


def test_identity_is_rechecked_immediately_before_delete(target):
    foreign = detail(target)
    foreign["metadata"]["session_nonce"] = "foreign"
    session = Session([Response(200, detail(target)), Response(200, foreign)])
    result = cleanup.release_target(session, Client(), target)
    assert result["outcome"] == "last_read_ownership_refused" and session.deletes == []


def test_new_local_dispatch_journal_refuses_delete(target):
    folder = Path(target["campaign_root"]) / "case-intents"
    folder.mkdir()
    (folder / "task-0001.json").write_text("{}")
    session = Session([Response(200, detail(target))])
    result = cleanup.release_target(session, Client(), target)
    assert result["outcome"] == "local_journals_changed_refused" and session.deletes == []


@pytest.mark.parametrize("delete_status", [204, 202, 500])
def test_one_delete_requires_get_404_confirmation(target, delete_status):
    session = Session(
        [Response(200, detail(target)), Response(200, detail(target)), Response(404)],
        delete_status=delete_status,
    )
    result = cleanup.release_target(session, Client(), target)
    assert result["outcome"] == (
        "deleted" if delete_status in {204, 202} else "absent_after_unconfirmed_delete"
    )
    assert result["delete_requests"] == len(session.deletes) == 1
    assert result["readback_http_status"] == 404 and len(session.gets) == 3


def test_timeout_then_404_reconciles_get_only(target):
    session = Session(
        [Response(200, detail(target)), Response(200, detail(target)), Response(404)],
        delete_error=TimeoutError("private transport value"),
    )
    result = cleanup.release_target(session, Client(), target)
    assert result["outcome"] == "absent_after_unconfirmed_delete"
    assert result["delete_request_ambiguous"] and len(session.deletes) == 1
    assert "private transport value" not in json.dumps(result)


@pytest.mark.parametrize("readback", [Response(200, {}), Response(503), TimeoutError("private")])
def test_delete_acknowledgement_never_replaces_readback_confirmation(target, readback):
    session = Session([Response(200, detail(target)), Response(200, detail(target)), readback])
    result = cleanup.release_target(session, Client(), target)
    assert result["outcome"] in {"delete_unconfirmed", "delete_readback_unknown"}
    assert len(session.deletes) == 1 and result["delete_requests"] == 1


def main_fixture(campaign, monkeypatch, tmp_path, sessions, *, delete=False):
    args = argparse.Namespace(
        root=campaign.root,
        env=tmp_path / ".env",
        output_dir=tmp_path / "private-cleanup",
        delete=delete,
    )
    monkeypatch.setattr(cleanup, "_parse_args", lambda: args)
    monkeypatch.setattr(cleanup.orchestrator, "SandboxClient", lambda *a, **kw: campaign.client)
    created = []

    def fake_http(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        session = sessions[len(created)]
        created.append(session)
        return session

    monkeypatch.setattr(cleanup.httpx, "Client", fake_http)
    return args, created


@pytest.mark.parametrize("delete", [False, True])
def test_main_uses_actual_manifests_and_private_subset_artifacts(
    campaign, monkeypatch, tmp_path, delete, capsys
):
    retain_case(campaign)
    target = cleanup.local_targets(campaign)[0][0]
    sessions = [Session([Response(200, detail(target))])]
    if delete:
        sessions.append(
            Session([Response(200, detail(target)), Response(200, detail(target)), Response(404)])
        )
    args, created = main_fixture(campaign, monkeypatch, tmp_path, sessions, delete=delete)
    before = cleanup._journal_snapshot(campaign.root)
    cleanup.main()
    plan = json.loads((args.output_dir / "release-plan.json").read_text())
    assert plan["created_workers"] == ["01"] and plan["planned_workers"] == 2
    assert plan["canonical_cases"] == 1 and plan["delete_requested"] is delete
    assert args.output_dir.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in args.output_dir.rglob("*.json"))
    assert cleanup._journal_snapshot(campaign.root) == before
    assert len(created) == (2 if delete else 1)
    assert sum(len(session.deletes) for session in sessions) == int(delete)
    stdout = json.loads(capsys.readouterr().out)
    if delete:
        assert stdout["confirmed_absent"] == 1
        assert (args.output_dir / "receipts" / "owned-1.json").is_file()
    else:
        assert stdout["delete_requests"] == 0 and stdout["ready"] == 1
        assert not (args.output_dir / "release-summary.json").exists()


def test_changed_source_manifest_fails_before_any_platform_observation(
    campaign, monkeypatch, tmp_path
):
    args, created = main_fixture(campaign, monkeypatch, tmp_path, [])
    (campaign.root / "cua-lite-source.tar.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source"):
        cleanup.main()
    assert created == [] and not (args.output_dir / "release-plan.json").exists()


def test_active_full_dispatch_lock_refuses_cleanup(campaign, monkeypatch, tmp_path):
    _, created = main_fixture(campaign, monkeypatch, tmp_path, [])
    with (campaign.root / "full-dispatch-owner.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            cleanup.main()
    assert created == []


def test_output_directory_must_be_new(campaign, monkeypatch, tmp_path):
    args, created = main_fixture(campaign, monkeypatch, tmp_path, [])
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        cleanup.main()
    assert created == []


def test_http_session_construction_failure_retains_private_preflight(
    campaign, monkeypatch, tmp_path, capsys
):
    args, _ = main_fixture(campaign, monkeypatch, tmp_path, [], delete=True)

    def failed_session(**kwargs):
        raise RuntimeError("private credential must not appear")

    monkeypatch.setattr(cleanup.httpx, "Client", failed_session)
    with pytest.raises(SystemExit) as exited:
        cleanup.main()
    assert exited.value.code == 2
    row = json.loads((args.output_dir / "preflight" / "owned-1.json").read_text())
    assert row["outcome"] == "observation_failed_refused" and row["error_type"] == "RuntimeError"
    receipt = json.loads((args.output_dir / "receipts" / "owned-1.json").read_text())
    assert receipt["delete_requests"] == 0
    assert "private credential" not in json.dumps(receipt) + capsys.readouterr().out
