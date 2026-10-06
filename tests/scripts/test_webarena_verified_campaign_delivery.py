from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
RELATIVE_CAMPAIGN = Path("devs/exps/eval/browsergym.webarena_verified/campaign")
CAMPAIGN = REPO / RELATIVE_CAMPAIGN
PREPARE = CAMPAIGN / "prepare-worker.sh"
CASE = CAMPAIGN / "run-case.sh"
CAPTURE_FILES = ("har/harTracer.js", "trace/recorder/tracing.js")
CONTAINER_SERVER = "/opt/venv/lib/python3.12/site-packages/playwright/driver/package/lib/server"
CONTAINER_LOCK = f"/root/cua-lite/{RELATIVE_CAMPAIGN}/playwright-capture.sha256"


def _run_shell(source: str, variables: dict[str, str]) -> subprocess.CompletedProcess:
    assignments = "\n".join(f"{name}={shlex.quote(value)}" for name, value in variables.items())
    return subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{assignments}\n{source}"],
        capture_output=True,
    )


@pytest.fixture
def short_capture_checks(tmp_path, monkeypatch):
    tools = tmp_path / "short-capture-checks"
    tools.mkdir()
    checksum = tools / "sha256sum"
    checksum.write_text(
        f"#!{shutil.which('bash')}\n"
        'for argument in "$@"; do\n'
        '    case "$argument" in --*|-s) exit 2 ;; esac\n'
        "done\n"
        f'exec {shlex.quote(shutil.which("sha256sum"))} "$@"\n'
    )
    checksum.chmod(0o700)
    compare = tools / "cmp"
    compare.write_text(
        f"#!{shutil.which('bash')}\n"
        'for argument in "$@"; do\n'
        '    case "$argument" in --*) exit 2 ;; esac\n'
        "done\n"
        f'exec {shlex.quote(shutil.which("cmp"))} "$@"\n'
    )
    compare.chmod(0o700)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])


@pytest.mark.parametrize("already_prepared", [False, True])
def test_prepare_refreshes_capture_assets_and_preserves_driver(tmp_path, already_prepared):
    spec = importlib.util.find_spec("playwright")
    if spec is None:
        pytest.skip("pinned Playwright is not installed")
    installed = Path(spec.origin).parent / "driver/package/lib/server"
    runtime = (
        tmp_path / "runtime/venv/lib/python3.12/site-packages/playwright/driver/package/lib/server"
    )
    for relative in CAPTURE_FILES:
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(installed / relative, target)
    original = {
        relative: ((runtime / relative).read_bytes(), (runtime / relative).stat().st_mtime_ns)
        for relative in CAPTURE_FILES
    }
    worker = tmp_path / "worker"
    extracted = worker / "source"
    stage = tmp_path / "stage"
    stage.mkdir()
    archive = stage / "cua-lite-source.tar.gz"
    runner = b"# frozen runner\n"
    assets = {
        "scripts/webarena_verified_qwen.py": runner,
        "scripts/webarena_verified_case_receipt.py": b"# frozen receipt\n",
        **{
            str(RELATIVE_CAMPAIGN / name): (CAMPAIGN / name).read_bytes()
            for name in (
                "build-playwright-capture.sh",
                "playwright-capture.patch",
                "playwright-capture.sha256",
            )
        },
    }
    with tarfile.open(archive, "w:gz") as bundle:
        for relative, data in assets.items():
            member = tarfile.TarInfo(f"./{relative}")
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    before_archive = archive.read_bytes()
    if already_prepared:
        extracted.mkdir(parents=True)
        (extracted / ".full-source-ready").touch()
        for relative in assets:
            stale = extracted / relative
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text("stale prepared asset\n")
    source = PREPARE.read_text()
    snippet = source[
        source.index("mkdir -p /var/lib/wav-qwen/source") : source.index(
            "if [ ! -d /var/lib/wav-qwen/desktop-python"
        )
    ]
    snippet = (
        snippet.replace("/var/lib/wav-runtime/venv", str(tmp_path / "runtime/venv"))
        .replace("/var/lib/wav-qwen", str(worker))
        .replace("/mnt/results", str(stage))
    )
    variables = {"RUNNER_SHA256": hashlib.sha256(runner).hexdigest()}
    first = _run_shell(snippet, variables)
    assert first.returncode == 0, first.stderr.decode()
    source_sha = hashlib.sha256(before_archive).hexdigest()
    overlay = worker / "playwright-capture" / source_sha
    patched = {relative: (overlay / relative).read_bytes() for relative in CAPTURE_FILES}
    second = _run_shell(snippet, variables)
    assert second.returncode == 0, second.stderr.decode()
    assert archive.read_bytes() == before_archive
    for relative, data in assets.items():
        assert (extracted / relative).read_bytes() == data
    for relative in CAPTURE_FILES:
        assert (
            (runtime / relative).read_bytes(),
            (runtime / relative).stat().st_mtime_ns,
        ) == original[relative]
        assert (overlay / relative).read_bytes() == patched[relative]


@pytest.fixture
def frozen_capture(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    source = tmp_path / "source"
    lock = source / RELATIVE_CAMPAIGN / "playwright-capture.sha256"
    lock.parent.mkdir(parents=True)
    content = {relative: f"frozen producer {relative}\n".encode() for relative in CAPTURE_FILES}
    hashes = {relative: hashlib.sha256(data).hexdigest() for relative, data in content.items()}
    lock.write_text("".join(f"{digest}  {relative}\n" for relative, digest in hashes.items()))
    archive = stage / "cua-lite-source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(f"./{RELATIVE_CAMPAIGN}/playwright-capture.sha256")
        member.size = len(lock.read_bytes())
        bundle.addfile(member, io.BytesIO(lock.read_bytes()))
    overlays = tmp_path / "playwright-capture"
    overlay = overlays / hashlib.sha256(archive.read_bytes()).hexdigest()
    for relative, data in content.items():
        target = overlay / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return stage, source, lock, overlays, overlay, hashes


@pytest.mark.parametrize(
    "invalid", [None, "missing-overlay", "missing-file", "wrong-hash", "stale-lock"]
)
def test_case_capture_gate_precedes_mutation_and_rejects_invalid(
    tmp_path, frozen_capture, short_capture_checks, invalid
):
    stage, source, lock, overlays, overlay, _ = frozen_capture
    if invalid == "missing-overlay":
        overlay.rename(overlays / "wrong-source-sha")
    elif invalid == "missing-file":
        (overlay / CAPTURE_FILES[0]).unlink()
    elif invalid == "wrong-hash":
        (overlay / CAPTURE_FILES[1]).write_text("different capture code\n")
    elif invalid == "stale-lock":
        lock.write_text("0" * 64 + "  har/harTracer.js\n")
    full_source = CASE.read_text()
    start = full_source.index("# Reject missing or changed capture code")
    end = full_source.index("test -f /var/lib/wav-qwen/private/model.env", start)
    assert (
        end < full_source.index("attempt_stamp=") < full_source.index('remove_owned "$CASE_NAME"')
    )
    snippet = (
        full_source[start:end]
        .replace("/mnt/results", str(stage))
        .replace("/var/lib/wav-qwen/source", str(source))
        .replace("/var/lib/wav-qwen/playwright-capture", str(overlays))
    )
    result = _run_shell(snippet + "printf 'site/model mutation reached\\n'", {})
    if invalid is None:
        assert result.returncode == 0 and result.stdout == b"site/model mutation reached\n"
    else:
        assert result.returncode != 0 and b"mutation reached" not in result.stdout


def _runner_argv() -> list[str]:
    source = CASE.read_text()
    snippet = source[
        source.index("runner_args=(") : source.index(
            "set +e\ndocker", source.index("runner_args=(")
        )
    ]
    variables = {
        "CASE_NAME": "wav-v18-01-task-0514",
        "RUN_ID": "wav-glm53flash-infra-v18-test-20261006",
        "PLAYWRIGHT_CAPTURE_DIR": "/var/lib/wav-qwen/playwright-capture/" + "a" * 64,
        "RUNTIME_IMAGE": "pinned-runtime-image",
        "TASK_ID": "514",
        "CASE_KEY": "task-0514",
        "MODEL_PROFILE": "glm53flash",
    }
    result = _run_shell(
        snippet + '\ndocker() { printf "%s\\0" "$@"; }\ndocker "${runner_args[@]}"', variables
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().rstrip("\0").split("\0")


def test_runner_fake_docker_argv_preserves_frozen_python_and_base_mounts():
    argv = _runner_argv()
    image_index = argv.index("pinned-runtime-image")
    mounts = [argv[index + 1] for index, value in enumerate(argv[:image_index]) if value == "-v"]
    overlay = "/var/lib/wav-qwen/playwright-capture/" + "a" * 64
    assert mounts == [
        "/var/lib/wav-runtime/venv:/opt/venv:ro",
        f"{overlay}/har/harTracer.js:{CONTAINER_SERVER}/har/harTracer.js:ro",
        f"{overlay}/trace/recorder/tracing.js:{CONTAINER_SERVER}/trace/recorder/tracing.js:ro",
        "/var/lib/wav-runtime/python:/root/.local/share/uv/python:ro",
        "/var/lib/wav-runtime/api-extra:/opt/api-extra:ro",
        "/var/lib/wav-runtime/nltk_data:/root/nltk_data:ro",
        "/var/lib/wav-qwen/desktop-python:/opt/desktop-python:ro",
        "/var/lib/wav-qwen/source:/root/cua-lite:ro",
        "/var/lib/wav-qwen/OSWorld:/opt/OSWorld:ro",
        "/var/lib/wav-qwen/results:/outputs",
        "/var/lib/wav-qwen/locks:/locks",
        "/var/lib/wav-qwen/baselines:/baselines:ro",
    ]
    assert argv[image_index + 1 : image_index + 3] == ["bash", "-c"]
    assert argv[image_index + 4] == "playwright-capture-entrypoint"
    assert argv[image_index + 5 :] == [
        "/root/cua-lite/scripts/webarena_verified_qwen.py",
        "--task-id",
        "514",
        "--osworld-root",
        "/opt/OSWorld",
        "--baseline-receipt",
        "/baselines/task-0514.json",
        "--output-dir",
        "/outputs/task-0514",
        "--lock-dir",
        "/locks",
        "--model-profile",
        "glm53flash",
        "--startup-page-gate",
        "--max-steps",
        "150",
        "--final-response-reserve-steps",
        "7",
    ]
    assert argv[argv.index("--env-file") + 1] == "/var/lib/wav-qwen/private/model.env"
    assert argv[argv.index("--network") + 1] == "host"
    assert "--shm-size=2g" in argv
    env = [argv[index + 1] for index, value in enumerate(argv[:image_index]) if value == "-e"]
    assert env == [
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=/opt/desktop-python:/opt/api-extra/python-extra:/root/cua-lite",
        "PLAYWRIGHT_BROWSERS_PATH=/root/.local/share/uv/python/ms-playwright",
        "NLTK_DATA=/root/nltk_data",
        "WA_SHOPPING=http://127.0.0.1:7770",
        "WA_SHOPPING_ADMIN=http://localhost:7780/admin",
        "WA_REDDIT=http://127.0.0.1:9999",
        "WA_GITLAB=http://127.0.0.1:8023",
        "WA_WIKIPEDIA=http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        "WA_MAP=http://localhost:8080",
        "WA_HOMEPAGE=http://127.0.0.1:4399",
    ]
    assert argv[:image_index] == [
        "run",
        "--rm",
        "--name",
        "wav-v18-01-task-0514",
        "--label",
        "com.cua-lite.run-id=wav-glm53flash-infra-v18-test-20261006",
        "--network",
        "host",
        "--shm-size=2g",
        "--env-file",
        "/var/lib/wav-qwen/private/model.env",
        *[argument for value in env for argument in ("-e", value)],
        *[argument for value in mounts for argument in ("-v", value)],
        "-w",
        "/outputs",
    ]


@pytest.mark.parametrize("invalid", [None, "wrong-hash", "missing-file"])
def test_actual_launch_envelope_preserves_cwd_and_checks_files_before_python(
    tmp_path, frozen_capture, short_capture_checks, invalid
):
    _, _, lock, _, overlay, _ = frozen_capture
    argv = _runner_argv()
    image_index = argv.index("pinned-runtime-image")
    envelope = argv[image_index + 3]
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$PWD" "$@"\n')
    fake_python.chmod(0o700)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    if invalid == "wrong-hash":
        (overlay / CAPTURE_FILES[0]).write_text("not the mounted producer\n")
    elif invalid == "missing-file":
        (overlay / CAPTURE_FILES[1]).unlink()
    envelope = (
        envelope.replace(CONTAINER_SERVER, str(overlay))
        .replace(CONTAINER_LOCK, str(lock))
        .replace("/opt/venv/bin/python", str(fake_python))
    )
    secret = "not-to-be-printed-test-key"
    result = subprocess.run(
        ["bash", "-c", envelope, *argv[image_index + 4 :]],
        capture_output=True,
        cwd=outputs,
        env={**os.environ, "GLM_TEST_API_KEY": secret},
    )
    assert secret.encode() not in result.stdout + result.stderr
    if invalid:
        assert result.returncode != 0 and not result.stdout
    else:
        assert result.returncode == 0
        assert result.stdout.decode().rstrip("\0").split("\0") == [
            str(outputs),
            "-B",
            *argv[image_index + 5 :],
        ]


def test_prepare_fake_docker_argv_mounts_only_capture_and_readonly_dependencies(tmp_path):
    source = PREPARE.read_text()
    start = source.index("docker run --rm -i --network none")
    snippet = source[start : source.index('\nmv "$WORKER_DIR/.prepared.json.tmp"', start)]
    captured = tmp_path / "argv"
    overlay = "/var/lib/wav-qwen/playwright-capture/" + "a" * 64
    result = _run_shell(
        'docker() { printf "%s\\0" "$@" > "$ARGV_PATH"; }\n' + snippet,
        {
            "WORKER_DIR": str(tmp_path),
            "ARGV_PATH": str(captured),
            "PLAYWRIGHT_CAPTURE_DIR": overlay,
            "QWENRUNNER_RUNTIME_IMAGE": "pinned-runtime-image",
        },
    )
    assert result.returncode == 0, result.stderr.decode()
    argv = captured.read_bytes().decode().rstrip("\0").split("\0")
    assert argv[:5] == ["run", "--rm", "-i", "--network", "none"]
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "-v"]
    assert mounts == [
        "/var/lib/wav-runtime/venv:/opt/venv:ro",
        "/var/lib/wav-runtime/python:/root/.local/share/uv/python:ro",
        "/var/lib/wav-qwen/source:/root/cua-lite:ro",
        f"{overlay}/har/harTracer.js:{CONTAINER_SERVER}/har/harTracer.js:ro",
        f"{overlay}/trace/recorder/tracing.js:{CONTAINER_SERVER}/trace/recorder/tracing.js:ro",
    ]
    assert argv[-3:] == ["pinned-runtime-image", "/opt/venv/bin/python", "-"]
    assert "--env-file" not in argv
    assert not any("private/" in argument or "api-extra" in argument for argument in argv)


@pytest.mark.parametrize("invalid", [False, True])
def test_prepared_marker_hashes_actual_installed_container_files(tmp_path, frozen_capture, invalid):
    _, _, lock, _, overlay, expected = frozen_capture
    package = tmp_path / "container/playwright"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for relative in CAPTURE_FILES:
        installed = package / "driver/package/lib/server" / relative
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_bytes((overlay / relative).read_bytes())
    if invalid:
        (package / "driver/package/lib/server" / CAPTURE_FILES[0]).write_text(
            "wrong container mount\n"
        )
    source = PREPARE.read_text()
    start = source.index("import hashlib\n")
    snippet = source[start : source.index("\nPY\nmv", start)].replace(CONTAINER_LOCK, str(lock))
    environment = {
        **os.environ,
        "PYTHONPATH": str(package.parent),
        "PYTHONDONTWRITEBYTECODE": "1",
        "RUN_ID": "v18-test",
        "WORKER": "01",
        "SANDBOX_ID": "owned-test",
        "CAPABILITIES": "shopping",
        "SOURCE_SHA": "source-sha",
        "IMAGE_LOCK_SHA": "image-sha",
        "PREPARED_AT": "2026-10-06T00:00:00Z",
        "IMAGES_JSON": "{}",
        "IMAGE_IDS_JSON": "{}",
        "PLAYWRIGHT_CAPTURE_DIR": str(overlay),
        "GLM_TEST_API_KEY": "not-to-be-printed-test-key",
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", snippet], capture_output=True, env=environment
    )
    assert environment["GLM_TEST_API_KEY"].encode() not in result.stdout + result.stderr
    if invalid:
        assert result.returncode != 0 and not result.stdout
    else:
        assert result.returncode == 0
        assert json.loads(result.stdout)["playwright_capture"] == expected


def test_receipt_cli_requires_same_frozen_capture_lock():
    source = CASE.read_text()
    receipt = source[source.index('if [ "$runner_rc" -eq 0 ] && runtime_python') :]
    assert f"--playwright-capture-lock {CONTAINER_LOCK} \\\n" in receipt
