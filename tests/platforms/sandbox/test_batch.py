"""Prove the shared queue can run an adapter without OSWorld tasks, health or artifacts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lite.platforms.sandbox import batch, lifecycle
from lite.platforms.sandbox.client import SandboxAPIError, write_json


def test_custom_adapter_owns_completion_and_queue_owns_sandbox_lifetime(tmp_path, monkeypatch):
    args = SimpleNamespace(
        state_dir=tmp_path / "state",
        resume=None,
        concurrency=2,
        poll_interval=1,
        task_timeout=30,
    )
    model_path = tmp_path / "api.json"
    write_json(
        model_path,
        {
            "api_type": "openai_chat",
            "model": "test-only",
            "api_key": "test-private-key",
            "base_url": "http://api.example/v1",
        },
    )
    deployment = {
        "project": "test",
        "base_url": "http://manager.example",
        "afs_url": "http://afs.example",
        "environment_id": "custom-template",
        "results_prefix": "custom-results",
    }
    state = batch.prepare(
        args,
        deployment,
        lambda: {
            "batch_id": "custom-batch",
            "job": {"model_config": str(model_path)},
            "tasks": [
                {"task_id": task, "experiment": f"custom-{task}", "state": "pending"}
                for task in ("alpha", "beta", "gamma")
            ],
        },
    )
    clock = SimpleNamespace(now=0)

    def sleep(seconds):
        clock.now += seconds
        assert clock.now < 20, "Queue failed to finish"

    monkeypatch.setattr(batch, "time", SimpleNamespace(time=lambda: clock.now, sleep=sleep))
    client = Mock(base_url=deployment["base_url"], afs_url=deployment["afs_url"], secrets=[])
    boxes, events = {}, []
    webhook = "test-only-webhook"

    def request(method, endpoint, payload=None):
        if method == "POST":
            assert endpoint == "/v1/sandboxes"
            sandbox_id = f"custom-{len(boxes)}"
            boxes[sandbox_id] = {
                "id": sandbox_id,
                "metadata": payload["metadata"],
                "state": "running",
                "extensions": {"webhook-feishu": webhook},
            }
            events.append(("create", sandbox_id, clock.now))
            return boxes[sandbox_id]
        sandbox_id = endpoint.rsplit("/", 1)[1]
        box = boxes[sandbox_id]
        if method == "GET":
            if box.get("deleted_at", 100) < clock.now:
                events.append(("absent", sandbox_id, clock.now))
                raise SandboxAPIError("not found", 404)
            return box
        assert method == "DELETE"
        events.append(("delete", sandbox_id, clock.now))
        box["deleted_at"] = clock.now
        return {}

    client.request.side_effect = request

    class CustomBatch:
        owner = "test-custom-benchmark"

        def submit(self, item):
            submission = {
                "experiment": item["experiment"],
                "batch_run": state["batch_run"],
                "afs_url": client.afs_url,
                "results_prefix": deployment["results_prefix"],
            }
            # The pre-create journal contract is independent of the job's contents.
            directory = args.state_dir / "jobs" / item["experiment"]
            directory.mkdir(parents=True, exist_ok=False)
            write_json(directory / "submission.json", submission)
            record = lifecycle.create(
                args.state_dir,
                client,
                {
                    "extensions": {"project": "test", "environmentId": "custom-template"},
                    "metadata": {"worker_run": item["experiment"]},
                },
                owner=self.owner,
                details={"submission": submission},
                webhook=webhook,
            )
            return lifecycle.verify_feishu(client, record, webhook)

        def observe(self, item, record):
            assert "release" not in record
            assert "worker_id" not in item
            return clock.now >= item["started_at"] + 1

        def collect(self, item, record):
            events.append(("collect", record["sandbox_id"], clock.now))
            item.update(workflow_completed=True, score=0.0, results="custom-artifact-format")

    result = batch.run(args, client, state, CustomBatch())
    assert result["completed"] == 3 and result["failed"] == 0
    active, maximum = set(), 0
    for event, sandbox_id, _ in events:
        if event == "create":
            active.add(sandbox_id)
            maximum = max(maximum, len(active))
        elif event == "absent":
            active.discard(sandbox_id)
    assert maximum == 2 and not active
    for sandbox_id in boxes:
        actions = [event for event, identity, _ in events if identity == sandbox_id]
        assert actions.index("collect") < actions.index("delete") < actions.index("absent")
    client.afs.assert_not_called()
    assert all(item["results"] == "custom-artifact-format" for item in result["tasks"])


def test_lifecycle_cannot_delete_another_benchmark_owner(tmp_path):
    client = Mock(base_url="http://manager.example")
    write_json(
        tmp_path / "worker-example.json",
        {
            "sandbox_id": "example",
            "project": "test",
            "base_url": client.base_url,
            "managed_by": "benchmark-a",
            "worker_run": "owned-nonce",
        },
    )
    with pytest.raises(ValueError, match="does not match this deployment"):
        lifecycle.delete(tmp_path, "example", client, project="test", owner="benchmark-b")
    client.request.assert_not_called()


def test_shared_modules_do_not_import_benchmark_implementations():
    import ast
    from pathlib import Path

    for module in (batch, lifecycle):
        source = Path(module.__file__).read_text()
        imports = [
            node.module
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        assert not any(".gym.envs" in name or ".sandbox.osworld" in name for name in imports)
        assert "agent-exit-code" not in source and "run.json" not in source
