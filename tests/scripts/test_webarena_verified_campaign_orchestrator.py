from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2]
    / "devs/exps/eval/browsergym.webarena_verified/campaign/orchestrate.py"
)
SPEC = importlib.util.spec_from_file_location("webarena_verified_campaign_orchestrator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
orchestrator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(orchestrator)

RUN_ID = "wav-generic-qwen38max-full120-20261007-v2"


@pytest.fixture
def campaign(tmp_path):
    benchmark = tmp_path / "benchmark"
    dataset = benchmark / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "webarena-verified.json").write_text(
        json.dumps(
            [
                {"task_id": 0, "sites": ["shopping_admin"]},
                {"task_id": 1, "sites": ["gitlab"]},
            ]
        )
    )
    root = benchmark / "results" / "run"
    root.mkdir(parents=True)
    runner = b"# pinned runner\n"
    receipt = b"# receipt validator\n"
    runner_sha = hashlib.sha256(runner).hexdigest()
    with tarfile.open(root / "cua-lite-source.tar.gz", "w:gz") as archive:
        for name, content in (
            ("./scripts/webarena_verified_generic.py", runner),
            ("./scripts/webarena_verified_case_receipt.py", receipt),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    (root / "VERSION_MANIFEST.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(
                    (root / "cua-lite-source.tar.gz").read_bytes()
                ).hexdigest(),
                "source_artifacts": {"scripts/webarena_verified_generic.py": runner_sha},
                "osworld_commit": "b138d348256078fa634fc3b73567a7337c793e6b",
            }
        )
    )
    (root / "campaign.env").write_text(
        f"RUN_ID={RUN_ID}\nRUNNER_SHA256={runner_sha}\nNAME_PREFIX=wav-test\n"
    )
    (root / "image-lock.env").write_text("QWENRUNNER_RUNTIME_IMAGE=pinned\n")
    for name in ("prepare-worker.sh", "run-case.sh", "run-worker.sh"):
        (root / name).write_text("#!/usr/bin/env bash\n")
    (root / "EXPERIMENT_CONFIG.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "model": "qwen3.8-max",
                "task_count": 2,
                "api_key_env": "QWEN_TEST_API_KEY",
                "entrypoint": "scripts/webarena_verified_generic.py",
            }
        )
    )
    workers = {
        "01": {"capabilities": ["shopping_admin"], "startup_task": 0, "tasks": 1},
        "02": {"capabilities": ["gitlab"], "startup_task": 1, "tasks": 1},
    }
    (root / "worker-plan.json").write_text(
        json.dumps(
            {
                "selection": "subset",
                "task_count": 2,
                "workers": workers,
                "tasks": {
                    "0": {"sites": ["shopping_admin"], "estimated_seconds": 20},
                    "1": {"sites": ["gitlab"], "estimated_seconds": 10},
                },
            }
        )
    )
    plan_paths = [root / "EXPERIMENT_CONFIG.json", root / "worker-plan.json"]
    for worker, task_id, site in (("01", 0, "shopping_admin"), ("02", 1, "gitlab")):
        folder = root / "workers" / worker
        folder.mkdir(parents=True)
        (folder / "tasks.tsv").write_text(f"{task_id}\t{site}\n")
        (folder / "capabilities.txt").write_text(site + "\n")
        plan_paths.extend((folder / "tasks.tsv", folder / "capabilities.txt"))
    (root / "PLAN_MANIFEST.sha256").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
            for path in plan_paths
        )
        + "\n"
    )
    env = tmp_path / ".env"
    env.write_text(
        "BASE=https://sandbox.example.test\nPROJECT=test\n"
        "OPEN_SANDBOX_ACCESS_KEY=access\nOPEN_SANDBOX_SECRET_KEY=secret\n"
        "QWEN_TEST_API_KEY=model-secret\nOPENAI_BASE_URL=https://model.example.test/v1\n"
    )
    return orchestrator.Campaign(root, env)


@pytest.mark.parametrize("worker_count", [1, 120, 128, 256])
def test_run_id_accepts_positive_worker_count(campaign, worker_count):
    run_id = f"wav-generic-qwen38max-full{worker_count}-20261007-v2"
    env = campaign.root / "campaign.env"
    env.write_text(env.read_text().replace(RUN_ID, run_id))
    config = campaign.root / "EXPERIMENT_CONFIG.json"
    config.write_text(config.read_text().replace(RUN_ID, run_id))
    manifest = campaign.root / "PLAN_MANIFEST.sha256"
    manifest.write_text(
        "\n".join(
            f"{hashlib.sha256(config.read_bytes()).hexdigest()}  EXPERIMENT_CONFIG.json"
            if line.endswith("  EXPERIMENT_CONFIG.json")
            else line
            for line in manifest.read_text().splitlines()
        )
        + "\n"
    )

    reloaded = orchestrator.Campaign(campaign.root, campaign.root.parents[2] / ".env")

    assert reloaded.run_id == run_id


@pytest.mark.parametrize("worker_count", ["0", "0120", "-120", "120x", "120 1"])
def test_run_id_rejects_invalid_worker_count(campaign, worker_count):
    env = campaign.root / "campaign.env"
    env.write_text(
        env.read_text().replace(RUN_ID, f"wav-generic-qwen38max-full{worker_count}-20261007-v2")
    )

    with pytest.raises(ValueError, match="invalid generic campaign run ID"):
        orchestrator.Campaign(campaign.root, campaign.root.parents[2] / ".env")


class FakeClient:
    def __init__(self):
        self.project = "test"
        self.items = []
        self.create_calls = 0
        self.command_calls = []

    def list_sandboxes(self):
        return list(self.items)

    def create(self, payload):
        self.create_calls += 1
        item = {"id": f"owned-{self.create_calls}", "state": "running", **payload}
        self.items.append(item)
        return item

    def detail(self, sandbox_id):
        return next(item for item in self.items if item["id"] == sandbox_id)

    def command(self, sandbox_id, command, *, background):
        self.command_calls.append((sandbox_id, command, background))
        return f"command-{len(self.command_calls)}"


def test_create_is_owned_and_idempotent(campaign):
    client = FakeClient()
    campaign.client = client
    campaign.create(["01"])
    campaign.create(["01"])
    assert client.create_calls == 1
    detail = campaign._owned_detail("01")
    assert detail["metadata"]["run_id"] == RUN_ID
    assert detail["metadata"]["plan_sha256"] == campaign.plan_sha


def test_stage_bundle_is_stable_across_retries(campaign):
    first = campaign._bundle()
    second = campaign._bundle()

    assert first == second
    assert first[4:8] == b"\0" * 4
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as bundle:
        assert bundle.extractfile("EXPERIMENT_CONFIG.json") is not None


def test_large_upload_assembles_verified_chunks(monkeypatch):
    client = orchestrator.SandboxClient(
        {
            "BASE": "https://sandbox.example.test",
            "PROJECT": "test",
            "OPEN_SANDBOX_ACCESS_KEY": "access",
            "OPEN_SANDBOX_SECRET_KEY": "secret",
            "GLM_TEST_API_KEY": "model-secret",
            "OPENAI_BASE_URL": "https://model.example.test/v1",
        },
        api_key_env="GLM_TEST_API_KEY",
    )
    uploaded = []
    commands = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

    def post(url, *, files, timeout):
        metadata = json.loads(files[0][1][1])
        uploaded.append((metadata["path"], files[1][1][1].read()))
        return Response()

    monkeypatch.setattr(orchestrator.requests, "post", post)
    monkeypatch.setattr(
        client,
        "checked_command",
        lambda sandbox_id, command, *, timeout_ms: commands.append(command),
    )
    payload = b"a" * (1024 * 1024) + b"b"

    client.upload("owned-1", "/tmp/bundle.tar.gz", payload)

    assert uploaded == [
        ("/tmp/bundle.tar.gz.part-0000", payload[: 1024 * 1024]),
        ("/tmp/bundle.tar.gz.part-0001", b"b"),
    ]
    assert len(commands) == 1
    assert "cat /tmp/bundle.tar.gz.part-0000 /tmp/bundle.tar.gz.part-0001" in commands[0]
    assert hashlib.sha256(payload).hexdigest() in commands[0]
    assert "chmod 600 /tmp/bundle.tar.gz.assembled" in commands[0]


def test_upload_retries_one_transient_timeout(monkeypatch):
    client = orchestrator.SandboxClient(
        {
            "BASE": "https://sandbox.example.test",
            "PROJECT": "test",
            "OPEN_SANDBOX_ACCESS_KEY": "access",
            "OPEN_SANDBOX_SECRET_KEY": "secret",
            "GLM_TEST_API_KEY": "model-secret",
            "OPENAI_BASE_URL": "https://model.example.test/v1",
        },
        api_key_env="GLM_TEST_API_KEY",
    )
    attempts = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

    def post(url, *, files, timeout):
        attempts.append((files[1][1][1].read(), timeout))
        if len(attempts) == 1:
            raise orchestrator.requests.Timeout("transient upload timeout")
        return Response()

    monkeypatch.setattr(orchestrator.requests, "post", post)
    monkeypatch.setattr(orchestrator.time, "sleep", lambda seconds: None)

    client.upload("owned-1", "/tmp/receipt.json", b"receipt")

    assert attempts == [(b"receipt", (15, 30)), (b"receipt", (15, 30))]


def test_create_reconciles_ambiguous_post_by_nonce(campaign):
    client = FakeClient()
    campaign.client = client
    campaign.create(["01"])
    record = json.loads((campaign.root / "worker-intents/01.json").read_text())
    record["post_attempted"] = True
    orchestrator._atomic_json(campaign.root / "worker-intents/01.json", record)
    (campaign.root / "sandbox-cluster.json").unlink()

    campaign.create(["01"])

    assert client.create_calls == 1
    assert campaign._cluster()["01"]["id"] == "owned-1"


def test_create_can_retry_an_explicit_rate_limit(campaign):
    class ThrottledClient(FakeClient):
        def create(self, payload):
            if self.create_calls == 0:
                self.create_calls += 1
                response = orchestrator.requests.Response()
                response.status_code = 429
                raise orchestrator.requests.HTTPError("rate limited", response=response)
            return super().create(payload)

    client = ThrottledClient()
    campaign.client = client
    with pytest.raises(orchestrator.requests.HTTPError):
        campaign.create(["01"])
    intent = json.loads((campaign.root / "worker-intents/01.json").read_text())
    assert intent["post_attempted"] is False

    campaign.create(["01"])

    assert client.create_calls == 2
    assert campaign._cluster()["01"]["id"] == "owned-2"


def test_foreign_sandbox_and_unscored_error_are_not_silent(campaign):
    client = FakeClient()
    campaign.client = client
    campaign.create(["01"])
    item = client.items[0]
    item["metadata"]["run_id"] = "another-run"
    with pytest.raises(RuntimeError, match="ownership mismatch"):
        campaign._owned_detail("01")

    receipt = {
        "run_id": RUN_ID,
        "plan_sha256": campaign.plan_sha,
        "worker": "01",
        "sandbox_id": "owned-1",
        "task_id": 0,
        "runner_sha256": campaign.runner_sha,
        "image_lock_sha256": campaign.image_sha,
        "source_sha256": campaign.version["source_sha256"],
        "archive_sha256": "a" * 64,
        "terminal_status": "evaluator_error",
        "score": None,
    }
    campaign._validated_receipt(receipt, "01", "owned-1", 0)
    receipt["score"] = 0
    with pytest.raises(RuntimeError, match="was scored"):
        campaign._validated_receipt(receipt, "01", "owned-1", 0)
    receipt.update(terminal_status="model_protocol_failed", score=0)
    campaign._validated_receipt(receipt, "01", "owned-1", 0)


def test_full_refuses_to_run_before_global_startup_gate(campaign):
    with pytest.raises(RuntimeError, match="startup gate"):
        campaign.full(["01"])


def test_launch_reuses_recorded_command_instead_of_double_submitting(campaign):
    client = FakeClient()
    campaign.client = client
    campaign.create(["01"])
    campaign._prepared = lambda worker, sandbox_id: True

    first = campaign._launch("01", "startup", 1)
    second = campaign._launch("01", "startup", 1)

    assert first == second
    assert len(client.command_calls) == 1
    assert first["launch_id"] in first["command"]


def test_case_claim_survives_ambiguous_post_without_changing_launch_or_worker(campaign):
    class AmbiguousClient(FakeClient):
        def command(self, sandbox_id, command, *, background):
            result = super().command(sandbox_id, command, background=background)
            if len(self.command_calls) == 1:
                raise orchestrator.requests.Timeout("response lost after command accepted")
            return result

    client = AmbiguousClient()
    campaign.client = client
    campaign.create(["01"])
    campaign._prepared = lambda worker, sandbox_id: True
    with pytest.raises(orchestrator.requests.Timeout):
        campaign._launch_case(0, "01")
    intent = json.loads((campaign.root / "case-intents/task-0000.json").read_text())
    first_launch = intent["attempts"][0]["launch_id"]
    first = campaign._launch_case(0, "01")
    second = campaign._launch_case(0, "01")
    assert first == second
    assert len(client.command_calls) == 2
    assert client.command_calls[0] == client.command_calls[1]
    assert first["launch_id"] == first_launch
    assert (
        len(json.loads((campaign.root / "case-intents/task-0000.json").read_text())["attempts"])
        == 1
    )
    assert campaign._case_owner(0) == ("01", "owned-1")


def test_dynamic_dispatch_reuses_live_handles_and_idle_compatible_worker(campaign, monkeypatch):
    class RunningClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.polls = {}
            self.done = set()
            self.transient_once = False

        def command_status(self, sandbox_id, command_id):
            command = self.command_calls[int(command_id.removeprefix("command-")) - 1][1]
            task = int(command.split()[-2])
            self.polls[task] = self.polls.get(task, 0) + 1
            if task == 2 and not self.transient_once:
                self.transient_once = True
                raise orchestrator.requests.Timeout("observation lost while command is live")
            if task == 2 and self.polls[task] < 3:
                return {"running": True}
            self.done.add(task)
            return {"running": False, "exit_code": 0}

    client = RunningClient()
    campaign.client = client
    campaign.create(["01", "02"])
    campaign._prepared = lambda worker, sandbox_id: True
    campaign._verify_startup_gate = lambda: None
    monkeypatch.setattr(orchestrator.time, "sleep", lambda seconds: None)
    for row in campaign.plan["workers"].values():
        row["capabilities"] = ["shopping_admin"]
    campaign.plan["tasks"].update(
        {
            str(task): {"sites": ["shopping_admin"], "estimated_seconds": duration}
            for task, duration in ((2, 100), (3, 50), (4, 10))
        }
    )
    campaign.plan["task_count"] = 5
    with (campaign.root / "workers/01/tasks.tsv").open("a") as output:
        output.write("2\tshopping_admin\n3\tshopping_admin\n4\tshopping_admin\n")
    campaign._initial_workers.update({2: "01", 3: "01", 4: "01"})
    complete = {0, 1}
    for task in complete:
        path = campaign.root / "index" / f"task-{task:04d}" / "completed.json"
        orchestrator._atomic_json(
            path, {"task_id": task, "score": 1, "terminal_status": "completed"}
        )
    campaign._local_receipt = lambda worker, sandbox_id, task: task in complete

    def sync_case(worker, sandbox_id, task):
        assert task in client.done
        assert campaign._case_owner(task) == (worker, sandbox_id)
        intent = json.loads((campaign.root / "case-intents" / f"task-{task:04d}.json").read_text())
        orchestrator._atomic_json(
            campaign.root / "attempts" / f"task-{task:04d}" / "attempt.json",
            {"launch_id": intent["attempts"][-1]["launch_id"]},
        )
        orchestrator._atomic_json(
            campaign.root / "index" / f"task-{task:04d}" / "completed.json",
            {"task_id": task, "score": 1, "terminal_status": "completed"},
        )
        complete.add(task)
        return "synced"

    campaign._sync_one = sync_case
    campaign.full(["01", "02"])
    assert len(client.command_calls) == 3
    assert [int(call[1].split()[-2]) for call in client.command_calls] == [2, 3, 4]
    assert campaign._case_owner(4) == ("02", "owned-2")
    assert client.polls[2] == 3
    assert complete == set(range(5))


def test_attempt_sync_retains_failed_archive_with_real_owner(campaign):
    payload = b"partial provider evidence"
    digest = hashlib.sha256(payload).hexdigest()
    remote = "/mnt/results/attempts/task-0000/20261007T000000-01-owned-1.json"
    record = {
        "run_id": RUN_ID,
        "plan_sha256": campaign.plan_sha,
        "worker": "01",
        "sandbox_id": "owned-1",
        "task_id": 0,
        "runner_sha256": campaign.runner_sha,
        "image_lock_sha256": campaign.image_sha,
        "source_sha256": campaign.version["source_sha256"],
        "model_started": True,
        "terminal_status": "infrastructure_failed",
        "archive_path": remote.removesuffix(".json") + ".tar.gz",
        "archive_sha256": digest,
    }

    class ArchiveClient:
        def checked_command(self, sandbox_id, command):
            assert sandbox_id == "owned-1"

        def download(self, sandbox_id, path):
            return (
                (remote + "\n").encode()
                if path.endswith("attempt-files.txt")
                else json.dumps(record).encode()
            )

        def download_large_file(self, sandbox_id, path, destination, expected_sha):
            assert sandbox_id == "owned-1" and expected_sha == digest
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)

    campaign.client = ArchiveClient()
    campaign._sync_attempts("01", "owned-1")
    ledger = campaign.root / "attempts/task-0000" / Path(remote).name
    local = json.loads(ledger.read_text())
    assert local["terminal_status"] == "infrastructure_failed"
    assert Path(local["local_archive_path"]).read_bytes() == payload
    campaign._sync_attempts("01", "owned-1")


def test_key_selection_uses_configured_model_key_without_fallback():
    values = {
        "BASE": "https://sandbox.example.test",
        "PROJECT": "test",
        "OPEN_SANDBOX_ACCESS_KEY": "access",
        "OPEN_SANDBOX_SECRET_KEY": "secret",
        "OPENAI_BASE_URL": "https://model.example.test/v1",
        "QWEN_TEST_API_KEY": "qwen-secret",
        "GLM_TEST_API_KEY": "glm-secret",
    }
    qwen = orchestrator.SandboxClient(values, api_key_env="QWEN_TEST_API_KEY")
    glm = orchestrator.SandboxClient(values, api_key_env="GLM_TEST_API_KEY")
    assert b"qwen-secret" in qwen.model_env and b"glm-secret" not in qwen.model_env
    assert b"glm-secret" in glm.model_env and b"qwen-secret" not in glm.model_env
    del values["QWEN_TEST_API_KEY"]
    with pytest.raises(ValueError, match="QWEN_TEST_API_KEY"):
        orchestrator.SandboxClient(values, api_key_env="QWEN_TEST_API_KEY")


def test_second_full_coordinator_cannot_acquire_dispatch_ownership(campaign, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        orchestrator,
        "_parse_args",
        lambda: SimpleNamespace(
            root=campaign.root, env=campaign.root / "unused", worker=None, phase="full"
        ),
    )
    monkeypatch.setattr(orchestrator, "Campaign", lambda root, env: campaign)
    launched = []
    campaign.full = lambda workers: launched.append(workers)
    with (campaign.root / "full-dispatch-owner.lock").open("a") as first:
        orchestrator.fcntl.flock(first, orchestrator.fcntl.LOCK_EX | orchestrator.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            orchestrator.main()
        assert launched == []
    orchestrator.main()
    assert launched == [["01", "02"]]
