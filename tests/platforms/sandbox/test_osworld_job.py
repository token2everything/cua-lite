"""Run configuration to Sandbox submission, with no live resources or model calls."""

from __future__ import annotations

import copy
import json
import shutil
import stat
import subprocess
from unittest.mock import Mock

import pytest
import yaml

from lite.platforms.sandbox.osworld_2 import manage

AGENT_IMAGE = "registry.example/cua-agent@sha256:" + "a" * 64


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("SANDBOX_FEISHU_WEBHOOK", raising=False)
    directory = tmp_path / ".sandbox"
    directory.mkdir()
    config = {
        "project": "test-project",
        "base_url": "http://manager.example",
        "afs_url": "http://afs.example",
        "environment_id": "template-v21",
        "results_prefix": "outputs/osworld",
        "release": "osworld-v2.1",
        "credentials": "AK-SK",
        "state_dir": "state",
        "job": {
            "model_config": "../models.json",
            "model_profile": "gui",
            "tasks": ["054", "069"],
            "history_turns": 4,
            "max_steps": 12,
            "results_root": "results",
            "experiment": "test-v21-job",
        },
    }
    path = directory / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    (directory / "AK-SK").write_text(json.dumps({"ak": "private-ak", "sk": "private-sk"}))
    model = tmp_path / "models.json"
    model.write_text(
        json.dumps(
            {
                "base_url": "http://model.example/v1",
                "api_key": "selected-private-key",
                "params": {"temperature": 0.2},
                "params_by_api_type": {"openai_chat": {"top_p": 0.8}},
                "models": {
                    "gui": {"api_type": "openai_chat", "model": "gui-model"},
                    "unused": {
                        "api_type": "openai_chat",
                        "model": "other",
                        "api_key": "unused-key",
                    },
                },
            }
        )
    )
    hook = "https://open.feishu.cn/open-apis/bot/v2/hook/test-private-hook"
    template = {
        "env": {
            "OSWORLD_BENCHMARK_RELEASE": "osworld-v2.1",
            "OSWORLD_VM_RAM_SIZE": "4G",
            "CUA_AGENT_IMAGE": AGENT_IMAGE,
        },
        "volumes": [
            {"subPath": "outputs/osworld", "mountPath": "/results", "readOnly": False},
            {
                "subPath": "benchmarks/osworld/osworld-v2.1",
                "mountPath": "/benchmark",
                "readOnly": True,
            },
        ],
        "extensions": {"webhook-feishu": hook},
    }
    client = Mock(base_url=config["base_url"], afs_url=config["afs_url"])
    client.secrets = ["private-ak", "private-sk"]
    created = {"id": "sandbox-one", "state": "running", "extensions": {"webhook-feishu": hook}}
    events = []

    def request(method, endpoint, payload=None):
        events.append((method, endpoint))
        if "sandbox-environments" in endpoint:
            return copy.deepcopy(template)
        if method == "POST":
            created["metadata"] = copy.deepcopy(payload["metadata"])
        return copy.deepcopy(created)

    client.request.side_effect = request
    client.afs.side_effect = lambda *args: events.append(("AFS", args)) or ""
    factory = Mock(return_value=client)
    monkeypatch.setattr(manage, "SandboxClient", factory)
    # Resolution must not depend on the invocation directory.
    cwd = tmp_path / "another-directory"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return path, config, model, template, client, created, factory, events


@pytest.mark.parametrize("explicit_config", [True, False])
def test_plan_never_opens_credentials_model_or_network(
    workspace, monkeypatch, capsys, explicit_config
):
    path, config, model, _, client, _, factory, _ = workspace
    model.unlink()
    (path.parent / "AK-SK").unlink()
    monkeypatch.setattr(
        manage, "load_credentials", Mock(side_effect=AssertionError("private read"))
    )
    if not explicit_config:
        monkeypatch.chdir(path.parent.parent)
    manage.main(["--config", str(path), "plan"] if explicit_config else ["plan"])
    value = json.loads(capsys.readouterr().out)
    assert value["requested_task_ids"] == ["054", "069"]
    assert value["model_config_file"] == str(model)
    assert value["results_root"] == str(path.parent / "results")
    assert value["state_dir"] == str(path.parent / "state")
    assert value["release"] == config["release"]
    assert not (path.parent / "state").exists()
    factory.assert_not_called()
    client.afs.assert_not_called()


@pytest.mark.parametrize(
    "job_change",
    [
        {"tasks": [44]},
        {"tasks": "054"},
        {"tasks": []},
        {"tasks": ["054", "054"]},
        {"tasks": ["999"]},
        {"task_set": "full"},
        {"history_turns": -1},
        {"max_steps": True},
        {"full_history": True},
        {"model_timeout": float("nan")},
        {"skip_unsupported": "false"},
        {"agent_bundle": "benchmarks/osworld/agent-builds/old-bundle"},
        {"agent_image": AGENT_IMAGE},
        {"typo": 4},
    ],
)
def test_bad_job_is_rejected_before_platform_access(workspace, job_change):
    path, config, _, _, _, _, factory, _ = workspace
    config["job"].update(job_change)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit):
        manage.main(["--config", str(path), "submit"])
    factory.assert_not_called()


@pytest.mark.parametrize(
    "resources",
    [
        None,
        {26: {"cpu": "8", "memory": "32Gi"}},
        {"999": {"cpu": "8", "memory": "32Gi"}},
        {"026": {"memory": "32Gi"}},
        {"026": {"cpu": 8, "memory": "32Gi"}},
        {"026": {"cpu": "0", "memory": "32Gi"}},
        {"026": {"cpu": "8", "memory": "-1Gi"}},
        {"026": {"cpu": "8", "memory": "32Gi", "gpu": "1"}},
    ],
)
def test_bad_task_resources_fail_before_private_reads_or_network(workspace, resources, monkeypatch):
    path, config, model, _, _, _, factory, _ = workspace
    config["task_resources"] = resources
    path.write_text(yaml.safe_dump(config))
    model.unlink()
    monkeypatch.setattr(
        manage, "load_credentials", Mock(side_effect=AssertionError("private read"))
    )
    with pytest.raises(SystemExit, match="task_resources"):
        manage.main(["--config", str(path), "submit"])
    factory.assert_not_called()


@pytest.mark.parametrize("tasks", [["026"], ["054"], ["026", "069"]])
def test_submit_applies_only_matching_outer_sandbox_resources(workspace, tasks, capsys):
    path, config, _, template, client, _, _, _ = workspace
    override = {"cpu": "8", "memory": "32Gi"}
    config["task_resources"] = {"026": override, "069": override}
    config["job"]["tasks"] = tasks
    path.write_text(yaml.safe_dump(config))
    manage.main(["--config", str(path), "plan"])
    expected = override if "026" in tasks else {}
    assert json.loads(capsys.readouterr().out)["resource_requests"] == expected
    manage.main(["--config", str(path), "submit"])
    payload = next(
        call.args[2] for call in client.request.call_args_list if call.args[0] == "POST"
    )
    assert payload.get("resourceRequests", {}) == expected
    assert payload["env"]["OSWORLD_VM_RAM_SIZE"] == "4G"
    assert "resourceRequests" not in template
    record = json.loads(capsys.readouterr().out)
    assert record["submission"].get("resource_requests", {}) == expected


@pytest.mark.parametrize("command", ["plan", "submit"])
def test_mixed_resources_in_one_sandbox_require_batch(workspace, monkeypatch, command):
    path, config, model, _, _, _, factory, _ = workspace
    config["task_resources"] = {"054": {"cpu": "8", "memory": "32Gi"}}
    path.write_text(yaml.safe_dump(config))
    model.unlink()
    monkeypatch.setattr(
        manage, "load_credentials", Mock(side_effect=AssertionError("private read"))
    )
    with pytest.raises(SystemExit, match="different task_resources; use batch"):
        manage.main(["--config", str(path), command])
    factory.assert_not_called()


def test_submit_matches_runner_and_snapshots_only_selected_model(workspace, capsys):
    from lite.gym.envs.osworld_2.rollout import _parse_args

    path, config, _, template, client, _, factory, events = workspace
    manage.main(["--config", str(path), "submit"])
    public = capsys.readouterr().out
    record = json.loads(public)
    assert record["feishu_webhook_verified"] is True
    assert events[:3] == [
        ("GET", "/v1/sandbox-environments/template-v21"),
        ("AFS", client.afs.call_args.args),
        ("POST", "/v1/sandboxes"),
    ]
    assert events[3] == ("GET", "/v1/sandboxes/sandbox-one")
    assert factory.call_args.kwargs["credentials"] == {
        "access_key": "private-ak",
        "secret_key": "private-sk",
    }
    directory = path.parent / "state/jobs/test-v21-job"
    snapshot = json.loads((directory / "model-api.json").read_text())
    assert set(snapshot["models"]) == {"gui"}
    assert snapshot["models"]["gui"]["params"] == {"temperature": 0.2, "top_p": 0.8}
    job = json.loads((directory / "job.json").read_text())
    assert set(job) == {"rollout_args", "model"}
    assert job["model"] == {"model_id": "gui-model", "api_type": "openai_chat"}
    args = _parse_args(job["rollout_args"])
    assert (args.release, args.tasks, args.history_turns, args.max_steps) == (
        "osworld-v2.1",
        ["054", "069"],
        4,
        12,
    )
    assert args.model_profile == "gui"
    assert "--immutable" in client.afs.call_args.args
    assert len([event for event in events if event[0] == "POST"]) == 1
    payload = client.request.call_args_list[1].args[2]
    assert payload["entrypoint"] == ["/bin/bash", "/opt/osworld-build/input/launch_rollout.sh"]
    assert payload["volumes"] == [
        *template["volumes"],
        {"subPath": "private/osworld-agent-jobs/test-v21-job",
         "mountPath": "/opt/osworld-job", "readOnly": True},
    ]
    assert payload["env"]["OSWORLD_AGENT_RESULTS_ROOT"] == str(path.parent / "results")
    assert payload["env"]["CUA_AGENT_IMAGE"] == AGENT_IMAGE
    assert record["agent_image"] == AGENT_IMAGE
    for secret in (
        "private-ak",
        "private-sk",
        "selected-private-key",
        "unused-key",
        "test-private-hook",
    ):
        assert secret not in public
    for file in directory.glob("*.json"):
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    saved = json.loads((path.parent / "state/worker-sandbox-one.json").read_text())
    assert saved["submission"]["experiment"] == config["job"]["experiment"]
    assert saved["submission"]["afs_url"] == config["afs_url"]
    assert saved["submission"]["results_prefix"] == config["results_prefix"]
    assert saved["submission"]["agent_image"] == AGENT_IMAGE
    # An explicit reused name cannot upload again or create another instance.
    events.clear()
    with pytest.raises(SystemExit, match="File exists"):
        manage.main(["--config", str(path), "submit"])
    assert all(event[0] == "GET" for event in events)


@pytest.mark.parametrize(
    "fault", ["release", "no-feishu", "output-path", "readonly", "job-exposure",
              "missing-agent-image", "floating-agent-image"]
)
def test_template_preflight_failure_does_not_upload_private_job(workspace, fault):
    path, _, _, template, client, _, _, events = workspace
    if fault == "release":
        template["env"]["OSWORLD_BENCHMARK_RELEASE"] = "osworld-v2-2026.08.08"
    elif fault == "no-feishu":
        template["extensions"].clear()
    elif fault == "output-path":
        template["volumes"][0]["subPath"] = "outputs/other"
    elif fault == "readonly":
        template["volumes"][0]["readOnly"] = True
    elif fault == "missing-agent-image":
        del template["env"]["CUA_AGENT_IMAGE"]
    elif fault == "floating-agent-image":
        template["env"]["CUA_AGENT_IMAGE"] = "registry.example/cua-agent:latest"
    else:
        template["volumes"].append({"subPath": "private", "mountPath": "/private"})
    with pytest.raises(SystemExit):
        manage.main(["--config", str(path), "submit"])
    assert events == [("GET", "/v1/sandbox-environments/template-v21")]
    assert not (path.parent / "state/jobs").exists()
    client.afs.assert_not_called()


@pytest.mark.parametrize("read_failure", [False, True])
def test_feishu_verification_failure_retains_ownership(workspace, read_failure):
    path, _, _, _, client, created, _, _ = workspace
    created["extensions"] = {}
    original = client.request.side_effect

    def request(method, endpoint, payload=None):
        if read_failure and (method, endpoint) == ("GET", "/v1/sandboxes/sandbox-one"):
            raise RuntimeError("read failed")
        return original(method, endpoint, payload)

    client.request.side_effect = request
    with pytest.raises(SystemExit, match="Sandbox sandbox-one created"):
        manage.main(["--config", str(path), "submit"])
    saved = json.loads((path.parent / "state/worker-sandbox-one.json").read_text())
    assert saved["worker_run"] == created["metadata"]["worker_run"]
    assert saved["feishu_webhook_verified"] is False
    assert len([c for c in client.request.call_args_list if c.args[0] == "POST"]) == 1


def test_lost_create_response_is_not_retried_and_nonce_is_retained(workspace):
    path, _, _, _, client, _, _, _ = workspace
    original = client.request.side_effect

    def request(method, endpoint, payload=None):
        if method == "POST":
            raise RuntimeError("connection lost selected-private-key")
        return original(method, endpoint, payload)

    client.request.side_effect = request
    with pytest.raises(SystemExit, match="do not resubmit blindly") as failure:
        manage.main(["--config", str(path), "submit"])
    assert "selected-private-key" not in str(failure.value)
    pending = json.loads((path.parent / "state/jobs/test-v21-job/submission.json").read_text())
    assert pending["worker_run"] in str(failure.value)
    assert len([c for c in client.request.call_args_list if c.args[0] == "POST"]) == 1


def test_yaml_feishu_file_supplies_missing_template_setting(workspace, capsys):
    path, config, _, template, client, created, _, _ = workspace
    hook = "https://open.feishu.cn/open-apis/bot/v2/hook/explicit-yaml-private"
    config["feishu_config"] = "feishu.json"
    path.write_text(yaml.safe_dump(config))
    (path.parent / "feishu.json").write_text(json.dumps({"webhook_url": hook}))
    template["extensions"].clear()
    created["extensions"]["webhook-feishu"] = hook
    manage.main(["--config", str(path), "submit"])
    payload = client.request.call_args_list[1].args[2]
    assert payload["extensions"]["webhook-feishu"] == hook
    output = capsys.readouterr().out
    assert json.loads(output)["feishu_webhook_verified"] is True
    assert "explicit-yaml-private" not in output


def test_sync_uses_submission_record_after_config_edit_and_model_removal(workspace, capsys):
    path, config, model, _, client, _, factory, events = workspace
    manage.main(["--config", str(path), "submit"])
    capsys.readouterr()
    model.unlink()
    config.update(
        project="another-project",
        base_url="http://another.example",
        afs_url="http://another-afs.example",
        results_prefix="outputs/new",
    )
    config["job"] = {"model_config": "no-longer-present", "invalid": True}
    path.write_text(yaml.safe_dump(config))
    events.clear()
    manage.main(["--config", str(path), "sync", "sandbox-one"])
    assert factory.call_args.kwargs["project"] == "test-project"
    assert factory.call_args.kwargs["afs_url"] == "http://afs.example"
    assert all(event[0] == "AFS" for event in events)
    assert len(events) == 1
    assert events[0][1][1] == "afs:outputs/osworld/agent-results"
    assert "/test-v21-job/**" in events[0][1]
    assert "/_environment/test-v21-job/**" in events[0][1]
    assert json.loads(capsys.readouterr().out)["results"] == str(
        path.parent / "results/test-v21-job"
    )


def test_status_and_delete_do_not_require_current_job_model(workspace, capsys):
    path, config, model, _, client, _, _, _ = workspace
    manage.main(["--config", str(path), "submit"])
    model.unlink()
    config["job"] = None
    path.write_text(yaml.safe_dump(config))
    manage.main(["--config", str(path), "status", "sandbox-one"])
    manage.main(["--config", str(path), "delete", "sandbox-one"])
    assert client.request.call_args.args == ("DELETE", "/v1/sandboxes/sandbox-one")
    assert "selected-private-key" not in capsys.readouterr().out


def test_afs_rejects_same_named_job_from_a_different_state_directory(workspace, tmp_path):
    rclone = shutil.which("rclone")
    if not rclone:
        pytest.skip("local rclone is required")
    path, config, _, _, client, _, _, events = workspace
    target = tmp_path / "fake-afs-job"

    def upload(*args):
        result = subprocess.run(
            [
                rclone,
                "--config",
                "/dev/null",
                args[0],
                args[1],
                str(target),
                *args[3:],
                "--size-only",
                "--retries",
                "1",
                "--low-level-retries",
                "1",
                "--stats",
                "0",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    client.afs.side_effect = upload
    manage.main(["--config", str(path), "submit"])
    original = (target / "model-api.json").read_bytes()
    config["state_dir"] = "state-from-another-workstation"
    path.write_text(yaml.safe_dump(config))
    events.clear()
    with pytest.raises(SystemExit, match="immutable file modified"):
        manage.main(["--config", str(path), "submit"])
    assert all(event[0] == "GET" for event in events)
    assert (target / "model-api.json").read_bytes() == original
    assert sorted(file.name for file in target.iterdir()) == ["job.json", "model-api.json"]


def test_sync_fetches_failure_without_environment_then_refreshes_mutable_results(
    workspace, tmp_path
):
    rclone = shutil.which("rclone")
    if not rclone:
        pytest.skip("local rclone is required")
    path, _, _, _, client, _, _, _ = workspace
    manage.main(["--config", str(path), "submit"])
    source = tmp_path / "fake-afs-results"
    run = source / "test-v21-job/task_054/run-one"
    run.mkdir(parents=True)
    (run / "run.json").write_text('{"steps":1}')
    other = source / "another-experiment"
    other.mkdir()
    (other / "run.json").write_text('{"steps":1}')

    def download(*args):
        return subprocess.run(
            [
                rclone,
                "--config",
                "/dev/null",
                args[0],
                str(source),
                *args[2:],
                "--size-only",
                "--retries",
                "1",
                "--stats",
                "0",
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout

    client.afs.side_effect = download
    manage.main(["--config", str(path), "sync", "sandbox-one"])
    output = path.parent / "results"
    saved = output / "test-v21-job/task_054/run-one/run.json"
    assert saved.read_text() == '{"steps":1}'
    assert not (output / "another-experiment").exists()
    assert not (output / "_environment").exists()
    (run / "run.json").write_text('{"steps":2}')
    environment = source / "_environment/test-v21-job/task_054"
    environment.mkdir(parents=True)
    (environment / "events.jsonl").write_text('{"action":"click"}\n')
    manage.main(["--config", str(path), "sync", "sandbox-one"])
    assert saved.read_text() == '{"steps":2}'
    assert (output / "_environment/test-v21-job/task_054/events.jsonl").is_file()
