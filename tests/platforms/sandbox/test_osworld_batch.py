"""Exercise the batch CLI against a deterministic platform/AFS, without live resources."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from lite.platforms.sandbox import batch
from lite.platforms.sandbox.client import AFSCommandError, SandboxAPIError, SandboxTransportError
from lite.platforms.sandbox.osworld_2 import manage


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.delenv("SANDBOX_FEISHU_WEBHOOK", raising=False)
    config = {
        "project": "test",
        "base_url": "http://manager.example",
        "afs_url": "http://afs.example",
        "environment_id": "0808-template",
        "release": "osworld-v2-2026.08.08",
        "results_prefix": "outputs/osworld",
        "state_dir": "state",
        "credentials": "keys.json",
        "job": {
            "model_config": "model.json",
            "experiment": "batch-test",
            "tasks": ["054", "007", "069"],
            "history_turns": 4,
            "results_root": "results",
        },
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    (tmp_path / "keys.json").write_text('{"ak":"test-ak","sk":"test-sk"}')
    (tmp_path / "model.json").write_text(
        json.dumps(
            {
                "api_type": "openai_chat",
                "model": "test-model",
                "api_key": "private-model-key",
                "base_url": "http://models.example/v1",
            }
        )
    )
    clock = SimpleNamespace(now=0)
    hooks = {"sleep": lambda: None}
    events = []

    def sleep(seconds):
        clock.now += seconds
        if clock.now > 100:
            raise AssertionError("Queue did not settle")
        hooks["sleep"]()

    monkeypatch.setattr(batch, "time", SimpleNamespace(time=lambda: clock.now, sleep=sleep))
    client = Mock(base_url=config["base_url"], afs_url=config["afs_url"])
    client.secrets = ["test-ak", "test-sk"]
    boxes, jobs = {}, {}
    faults = {}
    webhook = "https://open.feishu.cn/open-apis/bot/v2/hook/private-test-webhook"

    def request(method, endpoint, payload=None):
        if "sandbox-environments" in endpoint:
            if faults.pop("template", False):
                raise RuntimeError("template temporarily unavailable")
            return {
                "env": {
                    "OSWORLD_BENCHMARK_RELEASE": config["release"],
                    "CUA_AGENT_IMAGE": "registry.example/agent@sha256:" + "a" * 64,
                },
                "volumes": [{"subPath": "outputs/osworld", "mountPath": "/results"}],
                "extensions": {"webhook-feishu": webhook},
            }
        if endpoint == "/v1/sandboxes" and method == "POST":
            sandbox_id = f"sandbox-{len(boxes) + 1}"
            experiment = next(
                v["subPath"].rsplit("/", 1)[1]
                for v in payload["volumes"]
                if v["mountPath"] == "/opt/osworld-job"
            )
            boxes[sandbox_id] = {
                "id": sandbox_id,
                "state": "running",
                "metadata": payload["metadata"],
                "extensions": {"webhook-feishu": webhook},
                "created_at": clock.now,
                "experiment": experiment,
                "worker_id": f"pod-{len(boxes) + 1}",
            }
            events.append(("create", sandbox_id, clock.now))
            if faults.get("create"):
                raise SandboxTransportError("lost create response private-model-key")
            if faults.get("feishu"):
                boxes[sandbox_id]["extensions"] = {}
            return copy.deepcopy(boxes[sandbox_id])
        sandbox_id = endpoint.split("/")[3]
        box = boxes[sandbox_id]
        if method == "GET":
            if faults.get("read_forbidden"):
                raise SandboxAPIError("denied", 403)
            if box.get("delete_at", 1000) + 2 <= clock.now:
                events.append(("gone", sandbox_id, clock.now))
                raise SandboxAPIError("gone", 404)
            if faults.get("foreign_nonce"):
                box["metadata"]["worker_run"] = "someone-else"
            if faults.get("platform_failed"):
                box["state"] = "failed"
            return copy.deepcopy(box)
        if method == "DELETE":
            box["delete_at"] = clock.now
            events.append(("delete", sandbox_id, clock.now))
            if faults.pop("delete_response", False):
                raise RuntimeError("lost delete response")
            return {}
        if endpoint.endswith("renew-expiration"):
            box["expiresAt"] = "2100-01-01T00:00:00Z"
            events.append(("renew", sandbox_id, clock.now))
            return copy.deepcopy(box)
        raise AssertionError((method, endpoint))

    def afs(*args, timeout=None):
        if timeout is not None:
            raise AFSCommandError("progress temporarily unavailable private-model-key", 7)
        if args[0] == "copy" and not args[1].startswith("afs:"):
            directory = Path(args[1])
            jobs[directory.name] = json.loads((directory / "job.json").read_text())
            return ""
        if args[0] in ("lsf", "cat"):
            worker_id = args[1].split("/_workers/", 1)[1].split("/")[0]
            box = next(b for b in boxes.values() if b["worker_id"] == worker_id)
            if args[0] == "lsf":
                if faults.get("hang") or clock.now - box["created_at"] < 2:
                    return ""
                return "agent-exit-code\n"
            if faults.pop("empty_marker", False):
                return ""
            return str(faults.get("exit_code", 0))
        if "/_workers/" in args[1]:
            assert "/control.json" not in args  # No Worker credentials in mirrored diagnostics.
            return ""
        if faults.pop("sync", False):
            raise RuntimeError("AFS unavailable")
        if faults.get("sync_exit"):
            raise AFSCommandError("AFS copy failed", faults["sync_exit"])
        experiment = next(
            arg.strip("/").removesuffix("/**") for arg in args if arg.startswith("/batch-test")
        )
        argv = jobs[experiment]["rollout_args"]
        task = argv[argv.index("--tasks") + 1]
        target = Path(args[2]) / experiment / f"task_{task}" / f"run-{task}"
        target.mkdir(parents=True, exist_ok=True)
        if not faults.get("missing_result"):
            (target / "run.json").write_text(
                json.dumps(
                    {
                        "task_id": task,
                        "run_id": f"run-{task}",
                        "release": config["release"],
                        "status": faults.get("run_status", "completed"),
                        "closed": not faults.get("not_closed", False),
                        "score": 0.0,
                    }
                )
            )
        events.append(("sync", experiment, clock.now))
        return ""

    def health(sandbox_id, _client):
        if faults.pop("boot", False):
            raise SandboxAPIError("booting", 502)
        return "http://worker.example", {
            "release": config["release"],
            "worker_id": boxes[sandbox_id]["worker_id"],
        }

    client.request.side_effect = request
    client.afs.side_effect = afs
    monkeypatch.setattr(manage, "SandboxClient", Mock(return_value=client))
    monkeypatch.setattr(manage, "worker_health", health)
    return SimpleNamespace(
        path=path,
        config=config,
        client=client,
        boxes=boxes,
        jobs=jobs,
        clock=clock,
        hooks=hooks,
        events=events,
        faults=faults,
        state=tmp_path / "state/batches/batch-test/batch.json",
    )


def invoke(queue, *arguments):
    manage.main(["--config", str(queue.path), "batch", "--poll-interval", "1", *arguments])


def test_batch_collects_under_one_directory_and_manual_sync_keeps_the_saved_layout(queue):
    invoke(queue)
    state = json.loads(queue.state.read_text())
    output = queue.path.parent / "results" / "batch-test"
    assert state["results"] == str(output)
    assert sorted(p.name for p in output.iterdir()) == ["task_007", "task_054", "task_069"]
    for item in state["tasks"]:
        assert item["state"] == "completed"
        assert item["results"] == str(output)
        task = output / f"task_{item['task_id']}"
        assert (task / f"run-{item['task_id']}" / "run.json").is_file()
        assert not (output.parent / item["experiment"]).exists()
        record_file = queue.path.parent / f"state/worker-{item['sandbox_id']}.json"
        record = json.loads(record_file.read_text())
        assert record["submission"]["batch_id"] == state["batch_id"]
    launcher_targets = [
        call.args[2] for call in queue.client.afs.call_args_list
        if call.args[0] == "copy" and "/_workers/" in call.args[1]
    ]
    assert set(launcher_targets) == {
        str(output / f"task_{task}" / "_launcher") for task in ("054", "007", "069")
    }
    queue.config["job"]["experiment"] = "edited"
    queue.config["job"]["results_root"] = "another-results"
    queue.path.write_text(yaml.safe_dump(queue.config))
    manage.main(["--config", str(queue.path), "sync", state["tasks"][0]["sandbox_id"]])
    assert not (queue.path.parent / "another-results").exists()
    assert not list(output.parent.glob(".batch-sync-*"))


def test_monitor_uses_inherited_hook_and_failures_do_not_block_collection(queue, monkeypatch):
    from lite.platforms.sandbox import monitor

    queue.config["monitor"] = {"enabled": True}
    queue.path.write_text(yaml.safe_dump(queue.config))
    send = Mock(side_effect=RuntimeError("webhook unavailable private-model-key"))
    monkeypatch.setattr(monitor, "send_feishu_message", send)
    invoke(queue, "--concurrency", "2")
    state = json.loads(queue.state.read_text())
    assert all(i["state"] == "completed" for i in state["tasks"])
    assert send.call_count == 3  # Start, final, and one bounded retry for the final card.
    assert "private-test-webhook" in send.call_args.args[0]
    assert "private-model-key" not in json.dumps(state)
    assert "private-test-webhook" not in json.dumps(state)
    assert "notification" in state and "error" in state["notification"]
    assert "progress_error" in state["tasks"][0]


def test_monitor_dry_run_does_not_send_or_read_secrets(queue, monkeypatch):
    from lite.platforms.sandbox import monitor

    queue.config["monitor"] = {"enabled": True}
    queue.path.write_text(yaml.safe_dump(queue.config))
    send = Mock()
    monkeypatch.setattr(monitor, "send_feishu_message", send)
    invoke(queue, "--dry-run")
    send.assert_not_called()
    queue.client.request.assert_not_called()


def test_preview_full_0808_is_offline_and_does_not_create_files(queue, monkeypatch, capsys):
    queue.config["job"].pop("tasks")
    queue.config["job"]["task_set"] = "full"
    queue.path.write_text(yaml.safe_dump(queue.config))
    (queue.path.parent / "model.json").unlink()
    (queue.path.parent / "keys.json").unlink()
    monkeypatch.setattr(
        manage, "load_credentials", Mock(side_effect=AssertionError("private read"))
    )
    invoke(queue, "--concurrency", "8", "--dry-run")
    state = json.loads(capsys.readouterr().out)
    assert state["concurrency"] == 8
    assert len(state["tasks"]) == len({t["experiment"] for t in state["tasks"]}) == 108
    assert not queue.state.parent.exists()
    queue.client.request.assert_not_called()
    queue.client.afs.assert_not_called()


def test_per_task_resources_are_planned_and_submitted_with_template_inheritance(queue, capsys):
    override = {"026": {"cpu": "8", "memory": "32Gi"}}
    queue.config["task_resources"] = override
    queue.config["job"]["tasks"] = ["054", "026", "069"]
    queue.path.write_text(yaml.safe_dump(queue.config))
    invoke(queue, "--dry-run")
    state = json.loads(capsys.readouterr().out)
    assert state["deployment"]["task_resources"] == override
    queue.client.request.assert_not_called()
    invoke(queue)
    creates = [
        call.args[2] for call in queue.client.request.call_args_list if call.args[0] == "POST"
    ]
    assert [payload.get("resourceRequests") for payload in creates] == [
        None, override["026"], None,
    ]
    assert json.loads(queue.state.read_text())["deployment"]["task_resources"] == override


def test_resume_rejects_changed_resource_plan_without_touching_live_sandboxes(queue):
    queue.config["task_resources"] = {"026": {"cpu": "8", "memory": "32Gi"}}
    queue.path.write_text(yaml.safe_dump(queue.config))
    queue.faults["sync"] = True
    with pytest.raises(SystemExit, match="AFS unavailable"):
        invoke(queue, "--concurrency", "1")
    queue.config["task_resources"]["026"]["memory"] = "48Gi"
    queue.path.write_text(yaml.safe_dump(queue.config))
    queue.client.reset_mock()
    with pytest.raises(SystemExit, match="original batch deployment"):
        invoke(queue, "--resume", "batch-test")
    queue.client.request.assert_not_called()
    queue.client.afs.assert_not_called()
    assert json.loads(queue.state.read_text())["deployment"]["task_resources"]["026"] == {
        "cpu": "8", "memory": "32Gi",
    }


def test_two_slots_refill_after_confirmed_deletion_and_zero_score_is_completed(queue, capsys):
    queue.faults.update(boot=True, empty_marker=True)
    invoke(queue, "--concurrency", "2")
    result = json.loads(capsys.readouterr().out)
    assert (result["completed"], result["failed"]) == (3, 0)
    assert all(
        t["score"] == 0 and t["workflow_completed"] and t["sandbox_deleted"]
        for t in result["tasks"]
    )
    active = set()
    maximum = 0
    for event, identity, _time in queue.events:
        if event == "create":
            active.add(identity)
            maximum = max(maximum, len(active))
        elif event == "gone":
            active.discard(identity)
    assert maximum == 2
    assert not active
    assert not (queue.state.parent / ".scheduler.lock").exists()
    saved = queue.state.read_text()
    assert "private-model-key" not in saved and "private-test-webhook" not in saved
    for file in (queue.path.parent / "state").glob("worker-*.json"):
        assert json.loads(file.read_text())["feishu_webhook_verified"]


@pytest.mark.parametrize(
    "fault", ["not_closed", "missing_result", "run_status", "exit_code", "platform_failed"]
)
def test_failed_workflow_is_not_counted_as_zero_score_success(queue, fault, capsys):
    queue.faults[fault] = {"run_status": "failed", "exit_code": 1}.get(fault, True)
    with pytest.raises(SystemExit) as exit_info:
        invoke(queue)
    assert exit_info.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["completed"] == 0 and result["failed"] == 3
    assert all(t["sandbox_deleted"] for t in result["tasks"])


def test_sync_failure_holds_capacity_then_resume_uses_frozen_model(queue, capsys):
    queue.faults["sync"] = True
    with pytest.raises(SystemExit, match="AFS unavailable"):
        invoke(queue, "--concurrency", "2")
    assert len(queue.boxes) == 2
    assert not any(e[0] == "delete" for e in queue.events)
    (queue.path.parent / "model.json").unlink()
    queue.config["job"] = {"not": "read on resume"}
    queue.path.write_text(yaml.safe_dump(queue.config))
    invoke(queue, "--resume", "batch-test")
    assert len(queue.boxes) == 3
    assert json.loads(capsys.readouterr().out)["completed"] == 3
    # Completed batches can be inspected/resumed without creating any resources.
    invoke(queue, "--resume", "batch-test")
    assert len(queue.boxes) == 3


@pytest.mark.parametrize("fault", ["create", "feishu"])
def test_uncertain_or_unverified_submission_is_not_retried(queue, fault):
    queue.faults[fault] = True
    with pytest.raises(SystemExit):
        invoke(queue)
    assert len(queue.boxes) == 1
    assert "private-model-key" not in queue.state.read_text()
    with pytest.raises(SystemExit, match="Uncertain submission"):
        invoke(queue, "--resume", "batch-test")
    assert len(queue.boxes) == 1


def test_interrupt_after_verified_create_recovers_saved_ownership(queue, monkeypatch, capsys):
    original = manage.submit

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt()

    monkeypatch.setattr(manage, "submit", interrupted)
    with pytest.raises(KeyboardInterrupt):
        invoke(queue, "--concurrency", "2")
    monkeypatch.setattr(manage, "submit", original)
    invoke(queue, "--resume", "batch-test")
    assert len(queue.boxes) == 3
    assert json.loads(capsys.readouterr().out)["completed"] == 3


def test_lost_delete_response_can_resume_without_another_create(queue, capsys):
    queue.faults["delete_response"] = True
    with pytest.raises(SystemExit, match="lost delete"):
        invoke(queue, "--concurrency", "1")
    queue.clock.now += 3
    invoke(queue, "--resume", "batch-test")
    assert json.loads(capsys.readouterr().out)["completed"] == 3
    assert len([e for e in queue.events if e[0] == "delete"]) == 3


@pytest.mark.parametrize("status", [None, 429, 503])
def test_transient_platform_read_keeps_other_tasks_running_and_recovers(queue, capsys, status):
    request = queue.client.request.side_effect
    snapshots, other_reads = [], []

    def flaky(method, endpoint, payload=None):
        if method == "GET" and 0 < queue.clock.now <= 2:
            if endpoint == "/v1/sandboxes/sandbox-1":
                if status is None:
                    raise SandboxTransportError("timed out private-model-key")
                raise SandboxAPIError("temporary private-model-key", status)
            if endpoint == "/v1/sandboxes/sandbox-2":
                other_reads.append(queue.clock.now)
        return request(method, endpoint, payload)

    queue.client.request.side_effect = flaky
    queue.hooks["sleep"] = lambda: snapshots.append(json.loads(queue.state.read_text()))
    invoke(queue, "--concurrency", "2")
    result = json.loads(capsys.readouterr().out)
    assert result["completed"] == 3 and len(queue.boxes) == 3
    assert {1, 2} <= set(other_reads)
    unavailable = [s for s in snapshots if s["tasks"][0].get("poll_error")]
    assert unavailable and all(s["tasks"][0]["state"] == "running" for s in unavailable)
    assert all("error" not in s["tasks"][0] and "scheduler_error" not in s for s in unavailable)
    assert "private-model-key" not in json.dumps(unavailable)
    assert all("poll_error" not in t for t in result["tasks"])


def test_lost_delete_reply_reconciles_absence_without_stopping_or_recreating(queue, capsys):
    request = queue.client.request.side_effect
    lost = []

    def flaky(method, endpoint, payload=None):
        result = request(method, endpoint, payload)
        if method == "DELETE" and not lost:
            lost.append(endpoint)
            queue.boxes[endpoint.rsplit("/", 1)[-1]]["delete_at"] -= 3
            raise SandboxTransportError("DELETE: timed out")
        return result

    queue.client.request.side_effect = flaky
    invoke(queue, "--concurrency", "1")
    assert json.loads(capsys.readouterr().out)["completed"] == 3
    assert len(queue.boxes) == 3
    assert len([e for e in queue.events if e[0] == "delete"]) == 3


def test_long_scan_checks_notifications_between_tasks(queue, monkeypatch):
    from lite.platforms.sandbox import monitor
    from lite.platforms.sandbox.osworld_2.batch import OSWorldBatch

    queue.config["monitor"] = {"enabled": True}
    queue.path.write_text(yaml.safe_dump(queue.config))
    reports = []
    reporter = Mock()
    reporter.report.side_effect = lambda state, now, **kw: reports.append(now)
    monkeypatch.setattr(monitor, "BatchMonitor", Mock(return_value=reporter))
    observe = OSWorldBatch.observe

    def slow_observe(self, item, record):
        result = observe(self, item, record)
        queue.clock.now += 5
        return result

    monkeypatch.setattr(OSWorldBatch, "observe", slow_observe)
    invoke(queue)
    # A report opportunity follows each task instead of waiting for all three.
    assert {6, 11, 16} <= set(reports)


@pytest.mark.parametrize("fault", ["foreign_nonce", "read_forbidden"])
def test_wrong_ownership_or_auth_error_stops_without_deleting_or_refilling(queue, fault):
    queue.hooks["sleep"] = lambda: queue.faults.update({fault: True})
    with pytest.raises(SystemExit):
        invoke(queue, "--concurrency", "1")
    assert len(queue.boxes) == 1
    assert not any(e[0] == "delete" for e in queue.events)


def test_timeout_stops_writer_before_final_sync_and_frees_slots(queue, capsys):
    queue.faults["hang"] = True
    with pytest.raises(SystemExit) as exit_info:
        invoke(queue, "--concurrency", "1", "--task-timeout", "3")
    assert exit_info.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["failed"] == 3
    assert all(t["error"] == "Task exceeded task-timeout" for t in result["tasks"])
    assert next(i for i, e in enumerate(queue.events) if e[0] == "gone") < next(
        i for i, e in enumerate(queue.events) if e[0] == "sync"
    )


def test_resume_rejects_different_deployment_and_live_scheduler_lock(queue):
    queue.faults["sync"] = True
    with pytest.raises(SystemExit):
        invoke(queue)
    queue.config["results_prefix"] = "another/output"
    queue.path.write_text(yaml.safe_dump(queue.config))
    with pytest.raises(SystemExit, match="original batch deployment"):
        invoke(queue, "--resume", "batch-test")
    queue.config["results_prefix"] = "outputs/osworld"
    queue.path.write_text(yaml.safe_dump(queue.config))
    lock = queue.state.parent / ".scheduler.lock"
    lock.mkdir()
    with pytest.raises(SystemExit, match="already locked"):
        invoke(queue, "--resume", "batch-test")
    assert lock.is_dir()


def test_preflight_failure_before_staging_is_safe_to_resume(queue, capsys):
    queue.faults["template"] = True
    with pytest.raises(SystemExit, match="template temporarily unavailable"):
        invoke(queue)
    assert not queue.boxes
    invoke(queue, "--resume", "batch-test")
    assert len(queue.boxes) == 3
    assert json.loads(capsys.readouterr().out)["completed"] == 3


def test_child_name_collision_never_adopts_or_deletes_existing_submission(queue):
    original_experiment = queue.config["job"]["experiment"]
    queue.config["job"]["experiment"] += "-task-054"
    queue.path.write_text(yaml.safe_dump(queue.config))
    manage.main(["--config", str(queue.path), "submit"])
    queue.config["job"]["experiment"] = original_experiment
    queue.path.write_text(yaml.safe_dump(queue.config))
    for arguments in ((), ("--resume", "batch-test")):
        with pytest.raises(SystemExit, match="Child experiment already exists"):
            invoke(queue, *arguments)
    assert len(queue.boxes) == 1
    assert not any(e[0] == "delete" for e in queue.events)


@pytest.mark.parametrize("platform_failed", [False, True])
def test_durable_completed_receipt_survives_sandbox_disappearance(queue, capsys, platform_failed):
    def interrupt_after_worker_discovery():
        if queue.clock.now == 2:
            raise KeyboardInterrupt()

    queue.hooks["sleep"] = interrupt_after_worker_discovery
    with pytest.raises(KeyboardInterrupt):
        invoke(queue, "--concurrency", "1")
    # The last saved poll learned the Worker hostname, but the exit receipt was
    # written while the scheduler was offline and the Sandbox no longer runs.
    assert json.loads(queue.state.read_text())["tasks"][0]["worker_id"] == "pod-1"
    queue.clock.now += 3
    if platform_failed:
        queue.boxes["sandbox-1"]["state"] = "failed"
    else:
        queue.boxes["sandbox-1"]["delete_at"] = 0
    queue.hooks["sleep"] = lambda: None
    invoke(queue, "--resume", "batch-test")
    assert json.loads(capsys.readouterr().out)["completed"] == 3


def test_active_job_is_renewed_near_expiration(queue, capsys):
    def expiring():
        for box in queue.boxes.values():
            box.setdefault("expiresAt", "1970-01-01T00:08:00Z")

    queue.hooks["sleep"] = expiring
    invoke(queue)
    assert len([event for event in queue.events if event[0] == "renew"]) == 3
    assert json.loads(capsys.readouterr().out)["completed"] == 3


def test_failed_boot_with_no_afs_result_directory_is_retired(queue, capsys):
    queue.faults.update(platform_failed=True, sync_exit=3)
    with pytest.raises(SystemExit) as exit_info:
        invoke(queue, "--concurrency", "1")
    assert exit_info.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["failed"] == 3
    assert all(t["sandbox_deleted"] and "result directory" in t["error"] for t in result["tasks"])


def test_failed_boot_with_transfer_error_retains_resource_for_retry(queue):
    queue.faults.update(platform_failed=True, sync_exit=7)
    with pytest.raises(SystemExit, match="AFS copy failed"):
        invoke(queue, "--concurrency", "1")
    assert len(queue.boxes) == 1
    assert not any(e[0] == "delete" for e in queue.events)


@pytest.mark.parametrize("option", ["--concurrency", "--poll-interval", "--task-timeout"])
def test_rejects_nonpositive_limits_before_network(queue, option):
    with pytest.raises(SystemExit, match="positive"):
        invoke(queue, option, "0")
    queue.client.request.assert_not_called()
