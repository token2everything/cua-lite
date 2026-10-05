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

RUN_ID = "wav-glm53flash-infra-v6-test-20261005"


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
            ("./scripts/webarena_verified_qwen.py", runner),
            ("./scripts/webarena_verified_case_receipt.py", receipt),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    (root / "campaign.env").write_text(
        f"RUN_ID={RUN_ID}\nRUNNER_SHA256={runner_sha}\n"
        "NAME_PREFIX=wav-test\nMODEL_PROFILE=glm53flash\n"
    )
    (root / "image-lock.env").write_text("QWENRUNNER_RUNTIME_IMAGE=pinned\n")
    for name in ("prepare-worker.sh", "run-case.sh", "run-worker.sh"):
        (root / name).write_text("#!/usr/bin/env bash\n")
    (root / "EXPERIMENT_CONFIG.json").write_text(
        json.dumps({"run_id": RUN_ID, "model": "glm-5.3-flash"})
    )
    workers = {
        "01": {"capabilities": ["shopping_admin"], "startup_task": 0, "tasks": 1},
        "02": {"capabilities": ["gitlab"], "startup_task": 1, "tasks": 1},
    }
    (root / "worker-plan.json").write_text(
        json.dumps({"selection": "subset", "task_count": 2, "workers": workers})
    )
    plan_paths = [root / "worker-plan.json"]
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
        "GLM_TEST_API_KEY=model-secret\nOPENAI_BASE_URL=https://model.example.test/v1\n"
    )
    return orchestrator.Campaign(root, env)


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
        }
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
        }
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
