"""Exercise owned Map storage cleanup with real files and a fake Docker CLI."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import map_case_storage
import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = "wavgen-storage-test-6107001"
RUN = "storage-test"
NONCE = "owned-nonce"
SANDBOX = "owned-sandbox"
IMAGE_ID = "sha256:" + "a" * 64


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _script(source: str, tmp_path: Path) -> str:
    for original, local in (
        ("/mnt/results/v03", tmp_path / "infra"),
        ("/mnt/webarena-verified", tmp_path / "assets"),
        ("/var/lib/wav-runtime", tmp_path / "runtime"),
        ("/var/lib/wav-v03", tmp_path / "work"),
    ):
        source = source.replace(original, str(local))
    return source


def _fake_docker(tmp_path: Path, containers: list[dict] | None = None) -> dict[str, str]:
    program = (
        f"#!{sys.executable}\n"
        + """
import json, os, subprocess, sys
args = sys.argv[1:]
containers = json.loads(os.environ['FAKE_CONTAINERS'])
if args[0] == 'info' or args[0] == 'pull':
    sys.exit(0)
if args[:2] == ['image', 'inspect']:
    print(os.environ['FAKE_IMAGE_ID']); sys.exit(0)
if args[0] == 'ps':
    if os.environ.get('FAIL_DOCKER_PS'):
        sys.exit(29)
    print('\\n'.join(container['Id'] for container in containers)); sys.exit(0)
if args[0] == 'inspect':
    if '--format' in args:
        template = args[args.index('--format') + 1]
        if 'session-nonce' in template: print(os.environ['FAKE_NONCE'])
        elif 'run-id' in template: print(os.environ['FAKE_RUN'])
        elif 'State.Running' in template: print('true')
        else: sys.exit(31)
        sys.exit(0)
    selected = [container for container in containers if container['Id'] in args[1:]]
    if not selected: sys.exit(1)
    print(json.dumps(selected)); sys.exit(0)
if args[0] == 'run':
    if '--entrypoint' not in args or args[args.index('--entrypoint') + 1] != '/usr/bin/python3':
        sys.exit(33)
    assert '--rm' in args and args[args.index('--network') + 1] == 'none'
    assert args[args.index('--user') + 1] == '0:0'
    assert 'com.cua-lite.run-id=' + os.environ['FAKE_RUN'] in args
    assert any(argument.startswith('com.cua-lite.session-nonce=') for argument in args)
    image_index = args.index(os.environ['FAKE_HELPER_IMAGE'])
    with open(os.environ['FAKE_HELPER_LOG'], 'a') as log:
        log.write(json.dumps(args) + '\\n')
    source = sys.stdin.read()
    helper_mounts = [{'Source': args[index + 1].split(':')[0]}
                     for index, argument in enumerate(args) if argument == '-v']
    environment = dict(os.environ)
    environment['FAKE_CONTAINERS'] = json.dumps(containers + [{
        'Id': 'actual-helper-id', 'Name': '/' + args[args.index('--name') + 1],
        'Mounts': helper_mounts,
    }])
    result = subprocess.run([sys.executable, *args[image_index + 1:]], input=source,
                            env=environment, text=True, check=False)
    sys.exit(result.returncode)
raise RuntimeError('Unexpected fake Docker invocation: ' + repr(args))
"""
    )
    docker = tmp_path / "bin/docker"
    _write(docker, program)
    docker.chmod(0o755)
    unavailable = (
        "#!/bin/sh\nprintf called > "
        + shlex.quote(str(tmp_path / "host-python-was-called"))
        + "\nexit 127\n"
    )
    for path in (
        tmp_path / "bin/python3",
        tmp_path / "runtime/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12",
    ):
        _write(path, unavailable)
        path.chmod(0o755)
    return {
        **os.environ,
        "PATH": str(docker.parent) + os.pathsep + os.environ["PATH"],
        "FAKE_CONTAINERS": json.dumps(containers or []),
        "FAKE_IMAGE_ID": IMAGE_ID,
        "FAKE_NONCE": NONCE,
        "FAKE_RUN": RUN,
        "FAKE_HELPER_IMAGE": "map@" + IMAGE_ID,
        "FAKE_HELPER_LOG": str(tmp_path / "helper-log.jsonl"),
        "CUA_LITE_RUN_ID": RUN,
        "CUA_LITE_SESSION_NONCE": NONCE,
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _prepare_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    env = _fake_docker(tmp_path)
    _write(
        tmp_path / "infra/image-lock.env",
        'MAP_RUNTIME_IMAGE="map@'
        + IMAGE_ID
        + '"\n'
        + 'QWENRUNNER_RUNTIME_IMAGE="runner@'
        + IMAGE_ID
        + '"\n',
    )
    for name in ("desktop-python.tar.gz", "osworld-b138d348.bundle"):
        _write(tmp_path / "infra/runtime-inputs" / name, "fixed-runtime-input")
    seed = tmp_path / "assets/assets-full/map-volumes-webdav"
    for name in ("tile-db", "nominatim-db"):
        _write(seed / name / "fixed-seed.bin", name + "-fixed-seed")
    _write(tmp_path / "assets/assets-full/map-volumes/READY", "fixed-ready")
    _write(tmp_path / "assets/assets-full/map-volumes/PERMISSIONS_READY", "fixed-permissions")
    runtime = tmp_path / "runtime"
    helper = f"""#!/bin/bash
set -euo pipefail
if [[ "$1" = prepare ]]; then
  mkdir -p {shlex.quote(str(runtime / "venv/bin"))}
  ln -s /missing/host/python3 {shlex.quote(str(runtime / "venv/bin/python3"))}
elif [[ "$1" = prepare-map ]]; then
  for name in tile-db nominatim-db; do
    mkdir -p {shlex.quote(str(runtime / "map-volumes"))}/"$name"
    cp -a {shlex.quote(str(seed))}/"$name"/. {shlex.quote(str(runtime / "map-volumes"))}/"$name"/
    touch {shlex.quote(str(runtime / "map-volumes"))}/."$name".ready
  done
  printf 'entrypoint' > {shlex.quote(str(runtime / "map-volumes/entrypoint.sh"))}
  printf 'wrapper' > {shlex.quote(str(runtime / "map-wrapper.sh"))}
  mkdir -p {shlex.quote(str(runtime / "map-volumes/website-db"))}
fi
"""
    _write(tmp_path / "assets/tools/sandbox_stack.sh", helper)
    _write(tmp_path / "bin/tar", "#!/bin/bash\nexit 0\n")
    _write(
        tmp_path / "bin/git",
        '#!/bin/bash\nif [[ "$1" = clone ]]; then mkdir -p "$3"; else echo b138d348256078fa634fc3b73567a7337c793e6b; fi\n',
    )
    for name in ("tar", "git"):
        (tmp_path / "bin" / name).chmod(0o755)
    script = tmp_path / "prepare.sh"
    _write(script, _script((ROOT / "infra/prepare.sh").read_text(), tmp_path))
    return script, env


def test_prepare_removes_only_new_unmounted_db_copies(tmp_path):
    script, env = _prepare_fixture(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "map"], check=False, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr + result.stdout
    local = tmp_path / "runtime/map-volumes"
    for name in ("tile-db", "nominatim-db"):
        assert not (local / name).exists()
        assert not (local / ("." + name + ".ready")).exists()
        assert (
            tmp_path / "assets/assets-full/map-volumes-webdav" / name / "fixed-seed.bin"
        ).read_text() == name + "-fixed-seed"
    assert (local / "entrypoint.sh").read_text() == "entrypoint"
    assert (tmp_path / "runtime/map-wrapper.sh").read_text() == "wrapper"
    assert (local / "website-db").is_dir()
    assert (tmp_path / "assets/assets-full/map-volumes/READY").read_text() == "fixed-ready"
    assert (tmp_path / "infra/map-prepared-database-removal.json").is_file()
    helpers = [
        json.loads(line) for line in (tmp_path / "helper-log.jsonl").read_text().splitlines()
    ]
    assert len(helpers) == 1
    assert helpers[0][helpers[0].index("--entrypoint") + 1] == "/usr/bin/python3"
    assert not (tmp_path / "runtime/venv/bin/python3").exists()
    assert not (tmp_path / "host-python-was-called").exists()


def test_prepare_requires_coordinator_ownership_before_helper(tmp_path):
    script, env = _prepare_fixture(tmp_path)
    env.pop("CUA_LITE_SESSION_NONCE")
    result = subprocess.run(
        ["bash", str(script), "map"], check=False, env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert not (tmp_path / "helper-log.jsonl").exists()
    assert not (tmp_path / "runtime/map-volumes/tile-db").exists()


def test_prepare_keeps_preexisting_or_mounted_databases(tmp_path):
    script, env = _prepare_fixture(tmp_path)
    prior = tmp_path / "runtime/map-volumes/tile-db/prior.bin"
    _write(prior, "not-created-by-this-invocation")
    result = subprocess.run(
        ["bash", str(script), "map"], check=False, env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert prior.read_text() == "not-created-by-this-invocation"


def test_prepare_keeps_new_databases_if_any_container_mounts_them(tmp_path):
    script, env = _prepare_fixture(tmp_path)
    env["FAKE_CONTAINERS"] = json.dumps(
        [
            {
                "Id": "other-id",
                "Name": "/other-container",
                "Mounts": [{"Source": str(tmp_path / "runtime/map-volumes/tile-db")}],
            }
        ]
    )
    result = subprocess.run(
        ["bash", str(script), "map"], check=False, env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert (tmp_path / "runtime/map-volumes/tile-db/fixed-seed.bin").is_file()
    assert (tmp_path / "runtime/map-volumes/nominatim-db/fixed-seed.bin").is_file()
    assert not (tmp_path / "infra/map-prepared-database-removal.json").exists()


@pytest.mark.parametrize(
    "profile,attempt", [("author", "01"), ("teacher", "02"), ("teacher", "03")]
)
def test_case_claims_and_archives_ownership_before_failed_first_database_copy(
    tmp_path, profile, attempt
):
    env = _fake_docker(tmp_path)
    _write(tmp_path / "infra/image-lock.env", 'MAP_RUNTIME_IMAGE="map@' + IMAGE_ID + '"\n')
    _write(tmp_path / "infra/site-image-id.txt", IMAGE_ID)
    for name in ("run_single_site_case.sh", "bootstrap_single_site_execution.py"):
        _write(tmp_path / "infra" / name, "source fixture")
    for name in ("READY", "PERMISSIONS_READY"):
        _write(tmp_path / "assets/assets-full/map-volumes" / name, "immutable-marker")
    _write(tmp_path / "runtime/map-volumes/entrypoint.sh", "immutable-entrypoint")
    _write(tmp_path / "runtime/map-wrapper.sh", "immutable-wrapper")
    _write(
        tmp_path / "bin/cp",
        '#!/bin/bash\nif [[ "$*" = *\'/tile-db/.\'* ]]; then exit 35; fi\nexec /bin/cp "$@"\n',
    )
    (tmp_path / "bin/cp").chmod(0o755)
    script = tmp_path / "infra/run_single_site_case.sh"
    _write(script, _script((ROOT / "infra/run_single_site_case.sh").read_text(), tmp_path))
    result = subprocess.run(
        [
            "bash",
            str(script),
            "map",
            profile,
            TASK,
            RUN,
            NONCE,
            SANDBOX,
            attempt,
            "8080",
            "8081",
            "b" * 64,
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 35, result.stderr + result.stdout
    attempt_key = profile + "-" + attempt
    execution = tmp_path / "infra/single_site_executions/map" / TASK / attempt_key
    ownership = execution / "map-volume-ownership.json"
    archive_copy = (
        tmp_path / "work/results/baselines" / TASK / (attempt_key + "-map-volume-ownership.json")
    )
    assert archive_copy.read_bytes() == ownership.read_bytes()
    receipt = json.loads(ownership.read_text())
    assert receipt["run_id"] == RUN and receipt["session_nonce"] == NONCE
    assert receipt["map_root"] == str(tmp_path / "work/map-executions" / TASK / attempt_key)
    assert receipt["attempt"] == attempt and receipt["model_profile"] == profile
    assert not (execution / "inspection.json").exists()
    assert (Path(receipt["map_root"]) / "tile-db").is_dir()


def _case(
    tmp_path: Path, with_inspection: bool = False, profile: str = "author", attempt: str = "01"
) -> tuple[Path, Path, dict]:
    attempt_key = profile + "-" + attempt
    root = tmp_path / "work/map-executions" / TASK / attempt_key
    for name in ("tile-db", "nominatim-db", "website-db"):
        _write(root / name / "owned.bin", name + "-owned")
    execution = tmp_path / "infra/single_site_executions/map" / TASK / attempt_key
    ownership = {
        "site": "map",
        "run_id": RUN,
        "session_nonce": NONCE,
        "sandbox_id": SANDBOX,
        "generated_task_id": TASK,
        "model_profile": profile,
        "attempt": attempt,
        "container_name": "wav-v03-map-" + profile + "-" + TASK + "-" + attempt,
        "map_root": str(root),
    }
    _write(execution / "map-volume-ownership.json", json.dumps(ownership))
    if with_inspection:
        inspection = {key: value for key, value in ownership.items() if key != "map_root"}
        inspection["container_id"] = "actual-case-id"
        inspection["mounts"] = [
            {"Destination": target, "Source": str(root / name)}
            for target, name in (
                ("/data/database", "tile-db"),
                ("/data/nominatim/postgres", "nominatim-db"),
                ("/var/lib/postgresql/14/main", "website-db"),
            )
        ]
        _write(execution / "inspection.json", json.dumps(inspection))
    _write(tmp_path / "assets/immutable-seed.bin", "untouched-source")
    _write(tmp_path / "work/results/archived-output.json", "untouched-output")
    return root, execution, ownership


def _reap(
    tmp_path: Path,
    env: dict[str, str],
    nonce: str = NONCE,
    profile: str = "author",
    attempt: str = "01",
):
    ownership = (
        tmp_path
        / "infra/single_site_executions/map"
        / TASK
        / (profile + "-" + attempt)
        / "map-volume-ownership.json"
    )
    digest = hashlib.sha256(ownership.read_bytes()).hexdigest()
    return subprocess.run(
        [
            "bash",
            "-c",
            _script(map_case_storage.REMOTE_DISPATCH, tmp_path),
            "map-case-database-cleanup",
            TASK,
            profile,
            attempt,
            RUN,
            nonce,
            SANDBOX,
            digest,
            "map@" + IMAGE_ID,
        ],
        check=False,
        env=env,
        input=_script(map_case_storage.REMOTE_REAP, tmp_path),
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("with_inspection", [False, True])
@pytest.mark.parametrize(
    "profile,attempt", [("author", "01"), ("teacher", "02"), ("teacher", "03")]
)
def test_reap_owned_absent_case_preserves_source_outputs_and_receipts(
    tmp_path, with_inspection, profile, attempt
):
    root, execution, _ = _case(tmp_path, with_inspection, profile, attempt)
    result = _reap(tmp_path, _fake_docker(tmp_path), profile=profile, attempt=attempt)
    assert result.returncode == 0, result.stderr
    assert not root.exists()
    assert (execution / "map-volume-ownership.json").exists()
    assert (tmp_path / "assets/immutable-seed.bin").read_text() == "untouched-source"
    assert (tmp_path / "work/results/archived-output.json").read_text() == "untouched-output"
    receipt = json.loads(
        (
            tmp_path
            / "infra/diagnostics/map-case-prune"
            / (TASK + "-" + profile + "-" + attempt)
            / "receipt.json"
        ).read_text()
    )
    assert receipt["inspection_present"] is with_inspection
    assert receipt["container_absent"] is True
    assert receipt["attempt"] == attempt
    assert receipt["container_snapshot_precedes_helper"] is True
    assert receipt["helper_image_python"] == "/usr/bin/python3"
    assert not (tmp_path / "host-python-was-called").exists()
    helpers = [
        json.loads(line) for line in (tmp_path / "helper-log.jsonl").read_text().splitlines()
    ]
    assert len(helpers) == 1
    assert "map@" + IMAGE_ID in helpers[0]
    assert "com.cua-lite.session-nonce=" + NONCE in helpers[0]


@pytest.mark.parametrize(
    "blocker", ["wrong_nonce", "active_container", "other_mount", "daemon_error"]
)
def test_reap_retains_root_when_ownership_or_actual_container_absence_is_unproven(
    tmp_path, blocker
):
    root, _, ownership = _case(tmp_path)
    containers = []
    if blocker == "active_container":
        containers = [
            {"Id": "live-case-id", "Name": "/" + ownership["container_name"], "Mounts": []}
        ]
    if blocker == "other_mount":
        containers = [
            {"Id": "other-id", "Name": "/other", "Mounts": [{"Source": str(root / "tile-db")}]}
        ]
    env = _fake_docker(tmp_path, containers)
    if blocker == "daemon_error":
        env["FAIL_DOCKER_PS"] = "1"
    result = _reap(tmp_path, env, "wrong-nonce" if blocker == "wrong_nonce" else NONCE)
    assert result.returncode != 0
    assert (root / "tile-db/owned.bin").read_text() == "tile-db-owned"
    assert not (
        tmp_path / "infra/diagnostics/map-case-prune" / (TASK + "-author-01") / "receipt.json"
    ).exists()



def _collected_owner(tmp_path, profile="author", attempt="01", with_producer=True):
    directory = tmp_path / "runs/generated" / RUN
    _write(directory / "infra/image-lock.env", "MAP_RUNTIME_IMAGE=map@" + IMAGE_ID)
    attempt_key = profile + "-" + attempt
    label = "ground-" + TASK if profile == "author" else TASK + "-" + attempt_key
    ownership = {
        "site": "map", "run_id": RUN, "session_nonce": NONCE, "sandbox_id": SANDBOX,
        "generated_task_id": TASK, "model_profile": profile, "attempt": attempt,
        "container_name": "wav-v03-map-" + profile + "-" + TASK + "-" + attempt,
        "map_root": f"/var/lib/wav-v03/map-executions/{TASK}/{attempt_key}",
    }
    archive = directory / "archives" / (label + ".tar.gz")
    archive.parent.mkdir(parents=True)
    data = json.dumps(ownership).encode()
    with tarfile.open(archive, "w:gz") as packet:
        member = tarfile.TarInfo(f"baselines/{TASK}/{attempt_key}-map-volume-ownership.json"
                                 if with_producer else f"baselines/{TASK}/{attempt_key}.auth_failure.json")
        content = data if with_producer else b'{"model_started":false}'
        member.size = len(content)
        packet.addfile(member, io.BytesIO(content))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    receipt = archive.with_name(label + ".receipt.json")
    _write(receipt, json.dumps({"archive": str(archive), "SHA256": digest, "sandbox_id": SANDBOX}))
    handle = "terminal-" + attempt_key
    _write(directory / "commands" / (label + ".handle.json"), json.dumps({"command_id": handle}))
    status = {"id": handle, "running": False, "exit_code": 1}
    calls = []
    observed = []
    def command_status(sandbox, command):
        observed.append((sandbox, command))
        return status
    owner = SimpleNamespace(
        run=directory, site="map", args=SimpleNamespace(run=RUN),
        owned=lambda: {"id": SANDBOX, "metadata": {"session_nonce": NONCE}},
        command=lambda *args: calls.append(args),
        client=SimpleNamespace(command_status=command_status,
                               download=lambda *args: json.dumps({"removed_apparent_bytes": 42,
                                                                  "producer_ownership_SHA256": hashlib.sha256(data).hexdigest()})),
    )
    return owner, label, calls, observed, status, receipt, data


@pytest.mark.parametrize("blocker", ["archive_sha", "active_command", "unknown_exit", "unknown_handle", "wrong_nonce", "wrong_sandbox", "wrong_handle"])
def test_coordinator_gate_requires_verified_archive_and_actual_terminal_ownership(tmp_path, blocker):
    owner, label, calls, _observed, status, receipt, _data = _collected_owner(tmp_path)
    archive = json.loads(receipt.read_text())
    if blocker == "archive_sha":
        archive["SHA256"] = "invalid"
        receipt.write_text(json.dumps(archive))
    elif blocker == "wrong_sandbox":
        archive["sandbox_id"] = "other-sandbox"
        receipt.write_text(json.dumps(archive))
    elif blocker == "active_command":
        status.update(running=True, exit_code=None)
    elif blocker == "unknown_exit":
        status["exit_code"] = None
    elif blocker == "unknown_handle":
        (owner.run / "commands" / (label + ".handle.json")).unlink()
    elif blocker == "wrong_handle":
        status["id"] = "another-case-handle"
    else:
        owner.owned=lambda: {"id": SANDBOX, "metadata": {"session_nonce": "wrong-nonce"}}
    with pytest.raises((ValueError, FileNotFoundError)):
        map_case_storage.reap_collected_map_case(owner, label)
    assert calls == []


@pytest.mark.parametrize("profile,attempt", [("author", "01"), ("teacher", "01"), ("teacher", "02"), ("teacher", "03"), ("student", "01")])
def test_verified_terminal_archive_dispatches_its_own_attempt_and_reconciles_proof(tmp_path, profile, attempt):
    owner, case_label, calls, observed, status, receipt, data = _collected_owner(tmp_path, profile, attempt)
    result = map_case_storage.reap_collected_map_case(owner, case_label)
    assert observed == [(SANDBOX, "terminal-" + profile + "-" + attempt)]
    assert len(calls) == 1
    label, command = calls[0]
    assert label == "prune-databases-" + TASK + "-" + profile + "-" + attempt
    prefix, script = command.split(" <<'MAP_CLEANUP_PY'\n", 1)
    assert shlex.split(prefix)[:2] == ["bash", "-c"]
    assert script == map_case_storage.REMOTE_REAP + "\nMAP_CLEANUP_PY"
    assert shlex.split(prefix)[-8:] == [TASK, profile, attempt, RUN, NONCE, SANDBOX,
                                     hashlib.sha256(data).hexdigest(), "map@" + IMAGE_ID]
    assert result["terminal_case_handle"] == status["id"]
    assert result["durable_archive_SHA256"] == json.loads(receipt.read_text())["SHA256"]
    assert result["status"] == "reaped"
    assert map_case_storage.reap_collected_map_case(owner, case_label) == result
    assert len(calls) == 1


@pytest.mark.parametrize("presence", [
    {"producer_exists": False, "case_root_exists": False},
    {"producer_exists": True, "case_root_exists": False},
    {"producer_exists": False, "case_root_exists": True},
])
def test_missing_producer_requires_readonly_proof_and_never_deletes_a_guessed_root(tmp_path, presence):
    owner, label, calls, _observed, _status, _receipt, _data = _collected_owner(tmp_path, with_producer=False)
    remote_root=tmp_path/'actual-remote'
    def local(path):
        return Path(path.replace('/mnt/results/v03',str(remote_root/'infra'))
                        .replace('/var/lib/wav-v03',str(remote_root/'work')))
    if presence['producer_exists']:
        _write(local(f'/mnt/results/v03/single_site_executions/map/{TASK}/author-01/map-volume-ownership.json'),'actual-unarchived-producer')
    if presence['case_root_exists']:
        _write(local(f'/var/lib/wav-v03/map-executions/{TASK}/author-01/owned.bin'),'actual-root-retained')
    def command(label,body):
        calls.append((label,body))
        actual=body.replace('/mnt/results/v03',str(remote_root/'infra')).replace('/var/lib/wav-v03',str(remote_root/'work'))
        subprocess.run(['bash','-c',actual],check=True,capture_output=True,text=True)
    owner.command=command
    owner.client.download=lambda sandbox,path:local(path).read_bytes()
    if any(presence.values()):
        with pytest.raises(ValueError, match="actual producer/root exists"):
            map_case_storage.reap_collected_map_case(owner, label)
    else:
        result=map_case_storage.reap_collected_map_case(owner, label)
        assert result["status"] == "not_created"
        assert result["removed_apparent_bytes"] == 0
    assert len(calls) == 1
    assert calls[0][0] == "map-db-presence-" + TASK + "-author-01"
    assert "docker" not in calls[0][1] and "rmdir" not in calls[0][1] and "shutil" not in calls[0][1]


def test_map_explorer_and_non_map_collection_are_outside_case_reclamation(tmp_path):
    owner, label, calls, *_rest = _collected_owner(tmp_path)
    assert map_case_storage.reap_collected_map_case(owner, "explore-" + TASK) is None
    owner.site = "gitlab"
    assert map_case_storage.reap_collected_map_case(owner, label) is None
    assert calls == []
