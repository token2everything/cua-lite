"""Fake-only checks for bounded Infra deferral and honest partial completion."""

from __future__ import annotations

import hashlib
import importlib.resources
import importlib.util
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).parents[2]
    / "devs/exps/eval/browsergym.webarena_verified/campaign/orchestrate.py"
)
SPEC = importlib.util.spec_from_file_location("native_campaign_management_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class FakeClient:
    """Record launches and observations without starting a remote executor."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, str]] = []
        self.observations: list[str] = []
        self.statuses: dict[str, list[dict]] = {}

    def command(self, sandbox_id: str, command: str, *, background: bool) -> str:
        assert background is True
        self.commands.append((sandbox_id, command))
        command_id = f"fake-new-{len(self.commands)}"
        self.statuses[command_id] = [{"running": False, "exit_code": 0}]
        return command_id

    def command_status(self, sandbox_id: str, command_id: str) -> dict:
        assert sandbox_id.startswith("owned-")
        self.observations.append(command_id)
        states = self.statuses[command_id]
        return states.pop(0) if len(states) > 1 else states[0]


@pytest.fixture(autouse=True)
def no_external_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Management regressions must not call a remote service")

    monkeypatch.setattr(native.requests, "get", forbidden)
    monkeypatch.setattr(native.requests, "post", forbidden)
    monkeypatch.setattr(native.requests, "delete", forbidden)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    campaign = native.Campaign.__new__(native.Campaign)
    campaign.root = tmp_path
    campaign.run_id = "wav-generic-glm53flash-full120-20261007-v2"
    campaign.plan_sha = "a" * 64
    campaign.client = FakeClient()
    campaign.plan = {
        "task_count": 4,
        "workers": {worker: {"capabilities": ["shopping_admin"]} for worker in ("01", "02")},
        "tasks": {
            str(task): {"sites": ["shopping_admin"], "estimated_seconds": 100 - task}
            for task in range(4)
        },
    }
    campaign._initial_workers = {0: "01", 1: "02", 2: "01", 3: "02"}
    campaign._cluster = lambda: {
        worker: {"id": f"owned-{worker}"} for worker in campaign.plan["workers"]
    }
    campaign._owned_detail = lambda worker: {"id": f"owned-{worker}", "state": "running"}
    campaign._prepared = lambda worker, sandbox_id: True
    campaign._verify_startup_gate = lambda: None
    campaign._local_receipt = lambda worker, sandbox_id, task: (
        campaign.root / "index" / f"task-{task:04d}" / "completed.json"
    ).is_file()
    monkeypatch.setattr(native.time, "sleep", lambda seconds: None)
    return campaign


def _terminal(campaign, task: int, score: int | None = 1) -> None:
    native._atomic_json(
        campaign.root / "index" / f"task-{task:04d}" / "completed.json",
        {
            "task_id": task,
            "terminal_status": "evaluator_error" if score is None else "completed",
            "score": score,
        },
    )


def _claimed(campaign, task: int, *, count: int, exited: bool, proof: bool = True) -> dict:
    worker = campaign._initial_workers[task]
    attempts = []
    for number in range(1, count + 1):
        launch_id = f"{task * 10 + number:032x}"
        attempt = {
            "worker": worker,
            "sandbox_id": f"owned-{worker}",
            "launch_id": launch_id,
            "command": shlex.join(
                [
                    "bash",
                    "/mnt/results/run-worker.sh",
                    worker,
                    f"owned-{worker}",
                    "case",
                    launch_id,
                    str(task),
                    "shopping_admin",
                ]
            ),
            "command_id": f"existing-{task}-{number}",
        }
        if number < count or exited:
            attempt["exit_code"] = 1
        if proof or number < count:
            native._atomic_json(
                campaign.root / "attempts" / f"task-{task:04d}" / f"{launch_id}.json",
                {"launch_id": launch_id, "terminal_status": "infrastructure_failed"},
            )
        attempts.append(attempt)
    native._atomic_json(
        campaign.root / "case-intents" / f"task-{task:04d}.json",
        {
            "run_id": campaign.run_id,
            "plan_sha256": campaign.plan_sha,
            "task_id": task,
            "attempts": attempts,
        },
    )
    return attempts[-1]


@pytest.mark.parametrize("exited", [True, False])
def test_three_exhausted_attempts_defer_without_fourth(campaign, exited):
    attempt = _claimed(campaign, 0, count=3, exited=exited)
    campaign.client.statuses[attempt["command_id"]] = [{"running": False, "exit_code": 1}]
    for task in (1, 2, 3):
        _terminal(campaign, task)
    campaign._sync_one = lambda worker, sandbox_id, task: "pending"
    assert campaign.full(["01", "02"]) == [0]
    summary = json.loads((campaign.root / "campaign-summary.json").read_text())
    exhausted = json.loads((campaign.root / "EXHAUSTED_TASKS.json").read_text())
    intent = json.loads((campaign.root / "case-intents/task-0000.json").read_text())
    assert len(intent["attempts"]) == 3
    assert campaign.client.commands == []
    assert summary["terminal"] == 3 and summary["planned"] == 4
    assert summary["pending"] == 1 and summary["unscored"] == 0
    assert summary["official_accuracy"] is None
    assert summary["exhausted_task_ids"] == exhausted["task_ids"] == [0]
    assert exhausted["retry_limit"] == 3
    assert not (campaign.root / "index/task-0000/completed.json").exists()


def test_exhausted_case_does_not_block_live_handle_or_other_case(campaign):
    _claimed(campaign, 0, count=3, exited=True)
    live = _claimed(campaign, 1, count=1, exited=False, proof=False)
    campaign.client.statuses[live["command_id"]] = [
        {"running": True},
        {"running": False, "exit_code": 0},
    ]
    _terminal(campaign, 3, score=0)

    def sync(worker, sandbox_id, task):
        if task == 0:
            return "pending"
        intent = json.loads((campaign.root / "case-intents" / f"task-{task:04d}.json").read_text())
        launch_id = intent["attempts"][-1]["launch_id"]
        native._atomic_json(
            campaign.root / "attempts" / f"task-{task:04d}" / f"{launch_id}.json",
            {"launch_id": launch_id, "terminal_status": "completed"},
        )
        _terminal(campaign, task)
        return "synced"

    campaign._sync_one = sync
    assert campaign.full(["01", "02"]) == [0]
    assert len(campaign.client.commands) == 1
    assert int(shlex.split(campaign.client.commands[0][1])[-2]) == 2
    assert campaign.client.observations.count(live["command_id"]) == 2
    intent = json.loads((campaign.root / "case-intents/task-0001.json").read_text())
    assert len(intent["attempts"]) == 1
    assert intent["attempts"][0]["command_id"] == live["command_id"]
    summary = json.loads((campaign.root / "campaign-summary.json").read_text())
    assert summary["successes"] == 2 and summary["official_accuracy"] is None


@pytest.mark.parametrize("exited", [True, False])
def test_missing_latest_attempt_evidence_still_refuses_deferral(campaign, exited):
    attempt = _claimed(campaign, 0, count=3, exited=exited, proof=False)
    campaign.client.statuses[attempt["command_id"]] = [{"running": False, "exit_code": 1}]
    for task in (1, 2, 3):
        _terminal(campaign, task)
    campaign._sync_one = lambda worker, sandbox_id, task: "pending"
    with pytest.raises(RuntimeError, match="retained attempt evidence"):
        campaign.full(["01", "02"])
    assert campaign.client.commands == []
    assert not (campaign.root / "EXHAUSTED_TASKS.json").exists()
    assert not (campaign.root / "campaign-summary.json").exists()


def test_business_zero_is_terminal_and_not_rerolled(campaign):
    for task in range(4):
        _terminal(campaign, task, score=0)
    campaign._sync_one = lambda *args: pytest.fail("Scored cases must not sync/retry")
    assert campaign.full(["01", "02"]) == []
    assert campaign.client.commands == []
    summary = json.loads((campaign.root / "campaign-summary.json").read_text())
    assert summary["terminal"] == 4 and summary["official_accuracy"] == 0


def test_existing_canonical_receipt_clears_deferral_without_replay(campaign):
    native._atomic_json(
        campaign.root / "EXHAUSTED_TASKS.json",
        {
            "run_id": campaign.run_id,
            "plan_sha256": campaign.plan_sha,
            "task_ids": [0],
            "retry_limit": 3,
        },
    )
    for task in range(4):
        _terminal(campaign, task, score=0)
    assert campaign.full(["01", "02"]) == []
    assert campaign.client.commands == []
    assert json.loads((campaign.root / "EXHAUSTED_TASKS.json").read_text())["task_ids"] == []


def test_evaluator_null_is_not_an_infra_pending_case(campaign):
    _terminal(campaign, 0, score=None)
    for task in (1, 2, 3):
        _terminal(campaign, task)
    assert campaign.full(["01", "02"]) == []
    summary = json.loads((campaign.root / "campaign-summary.json").read_text())
    assert summary["terminal"] == 4 and summary["scored"] == 3
    assert summary["pending"] == 0 and summary["unscored"] == 1
    assert summary["evaluator_errors"] == 1 and summary["official_accuracy"] is None
    assert summary["exhausted_task_ids"] == []


@pytest.mark.parametrize("exhausted", [[0], []])
def test_full_cli_returns_partial_exit_code_without_automatic_relaunch(
    campaign, monkeypatch, exhausted
):
    monkeypatch.setattr(
        native,
        "_parse_args",
        lambda: SimpleNamespace(
            root=campaign.root,
            env=campaign.root / "unused",
            worker=None,
            phase="full",
        ),
    )
    monkeypatch.setattr(native, "Campaign", lambda root, env: campaign)
    calls = []
    campaign.full = lambda workers: calls.append(workers) or exhausted
    if exhausted:
        with pytest.raises(SystemExit) as error:
            native.main()
        assert error.value.code == 2
    else:
        native.main()
    assert calls == [["01", "02"]]


@pytest.mark.parametrize("model", ["qwen3.8-max", "glm-5.3-flash"])
def test_real_checkout_initialization_is_consumed_without_remote_calls(tmp_path, model):
    """Freeze real tracked source, then consume the produced public run contract."""
    pytest.importorskip("webarena_verified")
    init_source = SOURCE.with_name("initialize.py")
    init_spec = importlib.util.spec_from_file_location(
        "campaign_initialize_integration", init_source
    )
    assert init_spec is not None and init_spec.loader is not None
    initializer = importlib.util.module_from_spec(init_spec)
    init_spec.loader.exec_module(initializer)
    assets = tmp_path / "assets"
    desktop = "portable-desktop"
    for relative in (
        *initializer.ASSET_FILES,
        desktop + "/desktop-python.tar.gz",
        desktop + "/osworld-b138d348.bundle",
    ):
        path = assets / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-only asset; no runtime extraction\n")
    for relative in initializer.ASSET_DIRECTORIES:
        (assets / relative).mkdir(parents=True, exist_ok=True)
    catalog = Path(
        str(
            importlib.resources.files("webarena_verified").joinpath(
                "assets/dataset/webarena-verified.json"
            )
        )
    )
    image_lock = tmp_path / "image-lock.env"
    image_lock.write_text(
        "".join(
            f"{name}=example/image@sha256:{'a' * 64}\n" for name in sorted(initializer.IMAGE_NAMES)
        )
    )
    root = tmp_path / "campaign"
    args = initializer._parse_args(
        [
            "--root",
            str(root),
            "--model",
            model,
            "--run-id",
            "wav-generic-publication-full120-20261009-v1",
            "--name-prefix",
            "publication-test",
            "--catalog",
            str(catalog),
            "--image-lock",
            str(image_lock),
            "--asset-root",
            str(assets),
            "--desktop-assets-relative",
            desktop,
            "--sandbox-image",
            "example/sandbox-dind:operator-pinned",
            "--asset-subpath",
            "benchmark/verified/1.2.3",
        ]
    )
    initializer.initialize(args)
    private_env = tmp_path / "private.env"
    private_env.write_text(
        "BASE=https://sandbox.example.test\nPROJECT=test\n"
        "OPEN_SANDBOX_ACCESS_KEY=fake-access\nOPEN_SANDBOX_SECRET_KEY=fake-secret\n"
        "QWEN_TEST_API_KEY=fake-qwen\nGLM_TEST_API_KEY=fake-glm\n"
        "OPENAI_BASE_URL=https://model.example.test/v1\n"
    )
    campaign = native.Campaign(root, private_env)
    assert campaign.config["model"] == model
    assert len(campaign.plan["workers"]) == 120
    assert set(campaign._initial_workers) == set(range(812))
    assert len(campaign.version["source_artifacts"]) > 1000
    assert (
        campaign.version["runner_sha256"]
        == hashlib.sha256(
            (SOURCE.parents[5] / "scripts/webarena_verified_generic.py").read_bytes()
        ).hexdigest()
    )
    assert not (root / "sandbox-cluster.json").exists()
    assert not (root / ".env").exists()
