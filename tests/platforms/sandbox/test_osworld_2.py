"""OSWorld Worker provisioning and artifact transfers without platform access."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import stat
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import Mock, call

import pytest

from lite.platforms.sandbox import lifecycle
from lite.platforms.sandbox.osworld_2 import manage

AGENT_IMAGE = "registry.example/cua-agent@sha256:" + "a" * 64


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    monkeypatch.delenv("SANDBOX_FEISHU_WEBHOOK", raising=False)
    config = {
        "base_url": "http://manager.example",
        "project": "test-project",
        "afs_url": "http://afs.example",
        "environment_id": "test-environment",
        "results_prefix": "outputs/osworld",
    }
    args = argparse.Namespace(
        state_dir=tmp_path,
        environment_id=None,
        dry_run=False,
        feishu_config=None,
        no_feishu=False,
        agent_job=None,
        agent_results_root="/results/agent-results",
        sandbox_id="new-sandbox",
    )
    client = Mock()
    client.base_url = config["base_url"]
    client.secrets = ["test-ak", "test-sk"]
    client.request.return_value = {
        "id": "new-sandbox", "state": "pending",
        "env": {"OSWORLD_BENCHMARK_RELEASE": "osworld-v2.1"},
    }
    return args, config, client


def test_create_persists_nonce_ownership_and_omits_secrets(deployment):
    args, config, client = deployment
    result = manage.create(args, config, client, "osworld-v2.1")
    assert client.request.call_args.args[:2] == ("POST", "/v1/sandboxes")
    payload = client.request.call_args.args[2]
    assert payload["extensions"] == {"environmentId": "test-environment", "project": "test-project"}
    saved = json.loads((args.state_dir / "worker-new-sandbox.json").read_text())
    assert saved["worker_run"] == payload["metadata"]["worker_run"]
    assert saved["base_url"] == config["base_url"]
    assert result["sandbox_id"] == "new-sandbox"
    assert "test-sk" not in json.dumps(result)


def test_create_dry_run_needs_no_api_and_masks_explicit_hook(deployment, monkeypatch):
    args, config, client = deployment
    args.dry_run = True
    monkeypatch.setenv(
        "SANDBOX_FEISHU_WEBHOOK", "https://open.feishu.cn/open-apis/bot/v2/hook/private-hook"
    )
    result = manage.create(args, config, None, "osworld-v2.1")
    assert result["extensions"]["webhook-feishu"] == "[redacted]"
    assert list(args.state_dir.iterdir()) == []
    client.request.assert_not_called()


def test_create_records_inherited_hook_without_copying_it_into_request(deployment):
    args, config, client = deployment
    hook = "https://open.feishu.cn/open-apis/bot/v2/hook/inherited-private-hook"
    client.request.side_effect = [
        {"extensions": {"webhook-feishu": hook},
         "env": {"OSWORLD_BENCHMARK_RELEASE": "osworld-v2.1"}},
        {"id": "new-sandbox", "state": "pending"},
    ]
    result = manage.create(args, config, client, "osworld-v2.1")
    assert client.request.call_args_list[0].args == (
        "GET", "/v1/sandbox-environments/test-environment"
    )
    assert "webhook-feishu" not in client.request.call_args.args[2]["extensions"]
    assert result["feishu_webhook_configured"] is True
    saved = (args.state_dir / "worker-new-sandbox.json").read_text()
    assert json.loads(saved)["feishu_webhook_configured"] is True
    assert hook not in saved
    assert hook not in json.dumps(result)
    assert hook in client.secrets
    assert "inherited-private-hook" in client.secrets


@pytest.mark.parametrize("disabled", [False, True])
def test_create_explicit_hook_choice_overrides_environment(deployment, monkeypatch, disabled):
    args, config, client = deployment
    hook = "https://open.feishu.cn/open-apis/bot/v2/hook/explicit-private-hook"
    monkeypatch.setenv("SANDBOX_FEISHU_WEBHOOK", hook)
    args.no_feishu = disabled
    result = manage.create(args, config, client, "osworld-v2.1")
    assert client.request.call_count == 2
    method, path, payload = client.request.call_args.args
    assert (method, path) == ("POST", "/v1/sandboxes")
    assert payload["extensions"]["webhook-feishu"] == ("" if disabled else hook)
    assert result["feishu_webhook_configured"] is (not disabled)
    assert hook not in json.dumps(result)


def test_create_dry_run_without_hook_does_not_read_environment(deployment):
    args, config, client = deployment
    args.dry_run = True
    result = manage.create(args, config, None, "osworld-v2.1")
    assert "webhook-feishu" not in result["extensions"]
    client.request.assert_not_called()


def test_create_refuses_wrong_release_template_before_allocating(deployment):
    args, config, client = deployment
    with pytest.raises(ValueError, match="OSWORLD_BENCHMARK_RELEASE=osworld-v2-2026.08.08"):
        manage.create(args, config, client, "osworld-v2-2026.08.08")
    assert client.request.call_count == 1
    assert client.request.call_args.args[0] == "GET"
    assert not list(args.state_dir.iterdir())


def test_create_uses_selected_release_for_template_and_record(deployment):
    args, config, client = deployment
    release = "osworld-v2-2026.08.08"
    client.request.return_value["env"]["OSWORLD_BENCHMARK_RELEASE"] = release
    record = manage.create(args, config, client, release)
    assert record["release"] == release
    assert client.request.call_args.args[2]["metadata"]["benchmark_release"] == release


def test_create_agent_preserves_worker_mounts_env_and_inherited_hook(deployment):
    args, config, client = deployment
    args.agent_job = "private/osworld-agent-jobs/job-one"
    args.agent_results_root = "/workspace/results"
    mounts = [{"subPath": "benchmark", "mountPath": "/opt/osworld-data", "readOnly": True}]
    template = {
        "env": {"OSWORLD_BUILD_ID": "build-one", "OSWORLD_VM_RAM_SIZE": "4G",
                "OSWORLD_BENCHMARK_RELEASE": "osworld-v2.1", "CUA_AGENT_IMAGE": AGENT_IMAGE},
        "volumes": mounts,
        "extensions": {"webhook-feishu": "https://open.feishu.cn/open-apis/bot/v2/hook/private"},
    }
    client.request.side_effect = [template, {"id": "new-sandbox", "state": "pending"}]
    result = manage.create(args, config, client, "osworld-v2.1")
    payload = client.request.call_args.args[2]
    assert payload["entrypoint"] == ["/bin/bash", "/opt/osworld-build/input/launch_rollout.sh"]
    assert payload["env"] == {**template["env"], "OSWORLD_AGENT_RESULTS_ROOT": "/workspace/results"}
    assert payload["volumes"] == [
        *mounts,
        {"subPath": args.agent_job, "mountPath": "/opt/osworld-job", "readOnly": True},
    ]
    assert template["volumes"] == mounts
    assert "OSWORLD_AGENT_RESULTS_ROOT" not in template["env"]
    assert result["agent_in_sandbox"] is True
    assert result["agent_image"] == AGENT_IMAGE
    assert result["feishu_webhook_configured"] is True


@pytest.mark.parametrize(
    ("job", "results"),
    [
        ("../job", "/results"),
        ("/job", "/results"),
        ("", "/results"),
        ("job", "results"),
        ("job", "/results:ro"),
        ("job", "/"),
    ],
)
def test_create_rejects_invalid_agent_mounts_before_api(deployment, job, results):
    args, config, client = deployment
    args.agent_job, args.agent_results_root = job, results
    with pytest.raises(ValueError):
        manage.create(args, config, client, "osworld-v2.1")
    client.request.assert_not_called()


@pytest.mark.parametrize(
    "image",
    [None, "registry.example/cua-agent:latest", "sha256:" + "a" * 64,
     "registry.example/cua-agent@sha256:abc", "registry.example/cua-agent@sha512:" + "a" * 64],
)
def test_create_agent_requires_template_digest_before_post(deployment, image):
    args, config, client = deployment
    args.agent_job = "private/osworld-agent-jobs/job-one"
    if image is not None:
        client.request.return_value["env"]["CUA_AGENT_IMAGE"] = image
    with pytest.raises(ValueError, match="CUA_AGENT_IMAGE.*sha256 digest"):
        manage.create(args, config, client, "osworld-v2.1")
    client.request.assert_called_once_with(
        "GET", "/v1/sandbox-environments/test-environment"
    )
    assert not list(args.state_dir.iterdir())


def test_create_does_not_expose_model_job_through_worker_results_mount(deployment):
    args, config, client = deployment
    args.agent_job = "outputs/osworld/private-jobs/job-one"
    client.request.return_value = {
        "volumes": [{"subPath": "outputs/osworld", "mountPath": "/results"}],
        "env": {"OSWORLD_BENCHMARK_RELEASE": "osworld-v2.1", "CUA_AGENT_IMAGE": AGENT_IMAGE},
    }
    with pytest.raises(ValueError, match="outside all inherited Worker"):
        manage.create(args, config, client, "osworld-v2.1")
    assert client.request.call_count == 1
    assert client.request.call_args.args[0] == "GET"


def test_delete_requires_local_create_and_matching_remote_nonce(deployment):
    args, config, client = deployment
    with pytest.raises(ValueError, match="No local create record"):
        lifecycle.delete(
            args.state_dir, args.sandbox_id, client, project=config["project"], owner=manage._OWNER,
        )
    client.request.assert_not_called()
    result = manage.create(args, config, client, "osworld-v2.1")
    client.request.reset_mock()
    client.request.return_value = {"metadata": {"worker_run": "someone-else"}}
    with pytest.raises(ValueError, match="Remote Sandbox ownership"):
        lifecycle.delete(
            args.state_dir, args.sandbox_id, client, project=config["project"], owner=manage._OWNER,
        )
    assert all(call.args[0] == "GET" for call in client.request.call_args_list)
    client.request.reset_mock()
    client.request.side_effect = [{"metadata": {"worker_run": result["worker_run"]}}, {}]
    assert lifecycle.delete(
            args.state_dir, args.sandbox_id, client, project=config["project"], owner=manage._OWNER,
        )["state"] == "deletion_requested"
    assert client.request.call_args_list[-1].args == ("DELETE", "/v1/sandboxes/new-sandbox")


def test_delete_rejects_different_deployment(deployment):
    args, config, client = deployment
    manage.create(args, config, client, "osworld-v2.1")
    client.request.reset_mock()
    client.base_url = "http://other-manager.example"
    with pytest.raises(ValueError, match="does not match this deployment"):
        lifecycle.delete(
            args.state_dir, args.sandbox_id, client, project=config["project"], owner=manage._OWNER,
        )
    client.request.assert_not_called()


@pytest.mark.parametrize("release", ["osworld-v2.1", "osworld-v2-2026.08.08"])
def test_attach_checks_both_id_and_release_and_writes_0600(deployment, monkeypatch, release):
    args, config, client = deployment
    health = {"worker_id": "worker-instance", "release": release, "state": "idle"}
    client.endpoint.return_value = "http://manager.example/v1/sandboxes/new-sandbox/proxy/8800"
    client.afs.return_value = json.dumps({**health, "token": "private-worker-token"})
    opener = Mock()
    opener.open.return_value = io.BytesIO(json.dumps(health).encode())
    monkeypatch.setattr(manage.urllib.request, "build_opener", Mock(return_value=opener))
    result = manage.attach(args, config, client, release)
    path = args.state_dir / "connection-new-sandbox.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["token"] == "private-worker-token"
    assert "private-worker-token" not in json.dumps(result)
    assert not (args.state_dir / "worker-new-sandbox.json").exists()
    client.afs.assert_called_once_with(
        "cat", "afs:outputs/osworld/_workers/worker-instance/control.json"
    )


@pytest.mark.parametrize("mismatch", ["live_release", "control_release", "control_identity"])
def test_attach_refuses_mismatched_worker_without_connection(deployment, monkeypatch, mismatch):
    args, config, client = deployment
    health = {"worker_id": "worker-instance", "release": "osworld-v2.1", "state": "idle"}
    control = {**health, "token": "private-worker-token"}
    if mismatch == "live_release":
        health["release"] = "old-release"
    elif mismatch == "control_release":
        control["release"] = "old-release"
    else:
        control["worker_id"] = "different-worker"
    client.endpoint.return_value = "http://worker.example"
    client.afs.return_value = json.dumps(control)
    opener = Mock()
    opener.open.return_value = io.BytesIO(json.dumps(health).encode())
    monkeypatch.setattr(manage.urllib.request, "build_opener", Mock(return_value=opener))
    with pytest.raises(ValueError):
        manage.attach(args, config, client, "osworld-v2.1")
    assert not (args.state_dir / "connection-new-sandbox.json").exists()


def test_cli_from_unrelated_cwd_uses_declared_config_and_private_state(
    deployment, monkeypatch, capsys
):
    args, config, client = deployment
    config_path = args.state_dir / "platform.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.chdir(args.state_dir)
    monkeypatch.setenv("OPEN_SANDBOX_ACCESS_KEY", "test-ak")
    monkeypatch.setenv("OPEN_SANDBOX_SECRET_KEY", "test-sk")
    monkeypatch.setattr(manage, "SandboxClient", Mock(return_value=client))
    manage.main(["--config", str(config_path), "create"])
    state = args.state_dir / ".sandbox"
    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    assert (state / "worker-new-sandbox.json").exists()
    assert "test-sk" not in capsys.readouterr().out


def test_cli_results_uses_configured_afs_prefix(deployment, monkeypatch, capsys):
    args, config, client = deployment
    config_path = args.state_dir / "platform.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(
        manage, "load_credentials", Mock(return_value={"access_key": "ak", "secret_key": "sk"})
    )
    monkeypatch.setattr(manage, "SandboxClient", Mock(return_value=client))
    output = args.state_dir / "result"
    manage.main(
        [
            "--config",
            str(config_path),
            "--state-dir",
            str(args.state_dir / "state"),
            "results",
            "run-one",
            "--output",
            str(output),
        ]
    )
    mutable_artifacts = "/{episode.json,result.json,result.txt,trajectory.jsonl,runtime.log}"
    assert client.afs.call_args_list == [
        call("copy", "afs:outputs/osworld/runs/run-one", str(output),
             "--filter", f"- {mutable_artifacts}", "--transfers", "4"),
        call("copy", "afs:outputs/osworld/runs/run-one", str(output),
             "--filter", f"+ {mutable_artifacts}", "--filter", "- **",
             "--ignore-times", "--ignore-size", "--multi-thread-streams", "0",
             "--transfers", "4"),
    ]
    assert json.loads(capsys.readouterr().out)["results"] == str(output)


def test_cli_results_refreshes_same_size_metadata_without_recopying_steps(
    deployment, monkeypatch
):
    rclone = shutil.which("rclone")
    if rclone is None:
        pytest.skip("rclone is required for the local transfer regression")
    args, config, client = deployment
    config_path = args.state_dir / "platform.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(
        manage, "load_credentials", Mock(return_value={"access_key": "ak", "secret_key": "sk"})
    )
    monkeypatch.setattr(manage, "SandboxClient", Mock(return_value=client))
    source = args.state_dir / "remote-run"
    source.mkdir()
    output = args.state_dir / "download"
    mutable = {
        "episode.json": ('{"steps": 6}\n', '{"steps": 8}\n'),
        "result.json": ('{"score": 0}\n', '{"score": 1}\n'),
        "result.txt": ("0\n", "1\n"),
        "trajectory.jsonl": ('{"step": 6}\n', '{"step": 8}\n'),
        "runtime.log": ("starting\n", "finished\n"),
    }
    steps = {
        "00001-operation.png": b"immutable screenshot",
        "00001-operation.json": b'{"step": 1}',
        "operation.operation.json": b'{"status": "succeeded"}',
        "cache/episode.json": b'{"cached": true}',
    }
    for name, (old, _) in mutable.items():
        (source / name).write_text(old)
    for name, contents in steps.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    copied = []

    def copy_with_size_only_comparison(*command):
        assert command[:2] == ("copy", "afs:outputs/osworld/runs/run-one")
        # Generic WebDAV cannot compare checksums/modtimes. Exercise that
        # comparison rule with real rclone and local files, without credentials.
        result = subprocess.run(
            [rclone, "--config", "/dev/null", "copy", str(source), *command[2:],
             "--size-only", "--use-json-log", "--log-level", "INFO", "--stats", "0"],
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            check=True, capture_output=True, text=True,
        )
        for line in result.stderr.splitlines():
            event = json.loads(line)
            if event.get("msg", "").startswith("Copied"):
                copied.append(event["object"])
        return result.stdout

    client.afs.side_effect = copy_with_size_only_comparison
    argv = ["--config", str(config_path), "--state-dir", str(args.state_dir / "state"),
            "results", "run-one", "--output", str(output)]
    manage.main(argv)
    assert set(copied) == set(mutable) | set(steps)
    copied.clear()
    for name, (old, new) in mutable.items():
        assert len(old) == len(new)
        path = source / name
        previous_mtime = path.stat().st_mtime
        path.write_text(new)
        os.utime(path, (previous_mtime + 120, previous_mtime + 120))

    # Reproduce the previous single-copy behavior: newer equal-size files stay old.
    copy_with_size_only_comparison("copy", "afs:outputs/osworld/runs/run-one", str(output))
    assert copied == []
    for name, (old, _) in mutable.items():
        assert (output / name).read_text() == old

    (source / "00002-operation.png").write_bytes(b"new screenshot")
    manage.main(argv)
    for name, (_, new) in mutable.items():
        assert (output / name).read_text() == new
    assert set(copied) == set(mutable) | {"00002-operation.png"}
    assert len(copied) == len(mutable) + 1
    copied.clear()
    manage.main(argv)
    assert set(copied) == set(mutable)
    assert len(copied) == len(mutable)


@pytest.mark.parametrize(
    ("artifact", "truncated", "thread_cutoff", "error"),
    [
        ("episode.json", False, "256Mi", None),
        ("trajectory.jsonl", False, "1", None),
        ("runtime.log", False, "256Mi", None),
        ("trajectory.jsonl", True, "1", "unexpected EOF"),
        ("00001-operation.png", False, "256Mi", "corrupted on transfer: sizes differ"),
    ],
)
def test_cli_results_handles_stale_webdav_sizes(
    deployment, monkeypatch, artifact, truncated, thread_cutoff, error
):
    rclone = shutil.which("rclone")
    if rclone is None:
        pytest.skip("rclone is required for the local WebDAV regression")
    args, config, client = deployment
    config_path = args.state_dir / "platform.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(
        manage, "load_credentials", Mock(return_value={"access_key": "ak", "secret_key": "sk"})
    )
    monkeypatch.setattr(manage, "SandboxClient", Mock(return_value=client))
    output = args.state_dir / "download"
    output.mkdir()
    old = b'{"steps": 1}\n'
    current = b'{"steps": 10, "notes": "' + b"." * (256 * 1024) + b'"}\n'
    (output / artifact).write_bytes(old)
    root = "/outputs/osworld/runs/run-one/"
    get_ranges = []

    class WebDAVHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_PROPFIND(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            # The writer appends a newline after PROPFIND and before GET.
            file_entry = (root + artifact, "", len(current) - 1)
            if self.path.rstrip("/") == root.rstrip("/"):
                entries = [(root, "<d:collection/>", 0)]
                if self.headers.get("Depth") != "0":
                    entries.append(file_entry)
            elif self.path == root + artifact:
                entries = [file_entry]
            else:
                self.send_error(404)
                return
            body = ('<?xml version="1.0"?><d:multistatus xmlns:d="DAV:">' + "".join(
                f"<d:response><d:href>{path}</d:href><d:propstat><d:prop>"
                f"<d:resourcetype>{kind}</d:resourcetype>"
                f"<d:getcontentlength>{size}</d:getcontentlength>"
                "<d:getlastmodified>Wed, 23 Sep 2026 00:00:00 GMT</d:getlastmodified>"
                "</d:prop><d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>"
                for path, kind, size in entries
            ) + "</d:multistatus>").encode()
            self.send_response(207)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != root + artifact:
                self.send_error(404)
                return
            get_ranges.append(self.headers.get("Range"))
            self.send_response(200)
            self.send_header("Content-Length", str(len(current)))
            self.end_headers()
            self.wfile.write(current[:-10] if truncated else current)

    with ThreadingHTTPServer(("127.0.0.1", 0), WebDAVHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def copy_from_webdav(*command):
            result = subprocess.run(
                [rclone, "--config", "/dev/null", *command,
                 "--retries", "1", "--low-level-retries", "1", "--stats", "0"],
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "NO_PROXY": "127.0.0.1",
                    "RCLONE_CONFIG_AFS_TYPE": "webdav",
                    "RCLONE_CONFIG_AFS_URL": f"http://127.0.0.1:{server.server_port}/",
                    "RCLONE_CONFIG_AFS_VENDOR": "other",
                    # Exercise the mutable single-GET requirement even for small fixtures.
                    "RCLONE_MULTI_THREAD_CUTOFF": thread_cutoff,
                },
                check=False, capture_output=True, text=True, timeout=20,
            )
            if result.returncode:
                raise RuntimeError(result.stderr)
            return result.stdout

        client.afs.side_effect = copy_from_webdav
        argv = ["--config", str(config_path), "--state-dir", str(args.state_dir / "state"),
                "results", "run-one", "--output", str(output)]
        try:
            if error:
                with pytest.raises(SystemExit) as failure:
                    manage.main(argv)
                assert error in str(failure.value)
                assert (output / artifact).read_bytes() == old
            else:
                manage.main(argv)
                assert (output / artifact).read_bytes() == current
                assert get_ranges == [None]
            assert not list(output.glob("*.partial"))
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_cli_feishu_test_does_not_load_platform_credentials(monkeypatch, capsys):
    monkeypatch.setenv(
        "SANDBOX_FEISHU_WEBHOOK", "https://open.feishu.cn/open-apis/bot/v2/hook/private-hook"
    )
    credentials = Mock(side_effect=AssertionError("must not load credentials"))
    monkeypatch.setattr(manage, "load_credentials", credentials)
    send = Mock(return_value={"feishu": "test_message_accepted", "code": 0})
    monkeypatch.setattr(manage, "feishu_test", send)
    manage.main(["--config", "/nonexistent/config.yaml", "feishu-test"])
    assert send.call_count == 1
    assert "private-hook" not in capsys.readouterr().out
