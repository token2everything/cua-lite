"""Exercise the published runtime preparer with tiny assets and fake Docker.

These tests do not provision containers, call a model, or touch benchmark data.
They cover the managed-asset boundary used by the native generic campaign.
"""

from __future__ import annotations

import io
import shlex
import subprocess
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "devs/exps/eval/browsergym.webarena_verified/sandbox_stack.sh"
RUNTIME_FILES = (
    "python312-playwright144-chromium1117.tar.gz",
    "webarena-verified-venv.tar.gz",
    "webarena-api-extra.tar.gz",
    "nltk-punkt-tab.tar.gz",
    "webarena-eval-base-ubuntu22-playwright144.tar.gz",
    "start-eval.sh",
)


@pytest.fixture
def assets(tmp_path):
    asset_root = tmp_path / "assets"
    tools = asset_root / "tools"
    tools.mkdir(parents=True)
    members = {
        RUNTIME_FILES[0]: "cpython-3.12.12-linux-x86_64-gnu/bin/python3",
        RUNTIME_FILES[1]: "venv/lib/python3.12/site-packages/runtime.txt",
        RUNTIME_FILES[2]: "python-extra/api.txt",
        RUNTIME_FILES[3]: "nltk_data/tokenizers/punkt_tab/english/tokenizer.txt",
        RUNTIME_FILES[4]: "eval-image-placeholder",
    }
    for archive_name, member_name in members.items():
        payload = archive_name.encode()
        with tarfile.open(tools / archive_name, "w:gz") as archive:
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    (tools / "start-eval.sh").write_text("#!/bin/bash\nexit 0\n")
    map_source = asset_root / "assets-full/map-volumes"
    map_source.mkdir(parents=True)
    for name in ("READY", "PERMISSIONS_READY"):
        (map_source / name).touch()
    flatnode = map_source / "nominatim-flatnode"
    flatnode.mkdir()
    (flatnode / "flatnode.file").write_text("shared read-only flatnode\n")
    webdav = asset_root / "assets-full/map-volumes-webdav"
    for database in ("tile-db/postgres", "nominatim-db"):
        directory = webdav / database
        directory.mkdir(parents=True)
        (directory / "PG_VERSION").write_text("14\n")
        (directory / "postmaster.pid").write_text("stale pid\n")
    for profile in ("car", "bike", "foot"):
        directory = webdav / f"routing-{profile}"
        directory.mkdir()
        (directory / "us-northeast-latest.osrm.mldgr").write_text(profile)
    return asset_root


def run_stack(tmp_path, assets, mode, *, image_present=False, skip_eval=True, image_lock=None):
    """Run only against test-owned files and exported Docker/chown stubs."""
    log = tmp_path / "commands.log"
    env = {
        "PATH": "/usr/bin:/bin",
        "WEBARENA_VERIFIED_ASSET_ROOT": str(assets),
        "WEBARENA_VERIFIED_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "WEBARENA_VERIFIED_SKIP_EVAL_IMAGE": "1" if skip_eval else "0",
        "TEST_COMMAND_LOG": str(log),
        "TEST_IMAGE_PRESENT": "1" if image_present else "0",
    }
    if image_lock is not None:
        env["WEBARENA_VERIFIED_IMAGE_LOCK"] = str(image_lock)
    shell = r"""
docker() {
    printf '%q ' docker "$@" >> "$TEST_COMMAND_LOG"
    printf '\n' >> "$TEST_COMMAND_LOG"
    if [ "$1" = image ] && [ "$2" = inspect ]; then
        test "$TEST_IMAGE_PRESENT" = 1
    elif [ "$1" = run ]; then
        printf '%s\n' '#!/bin/bash' \
            'chown -R nominatim: $NOMINATIM_FLATNODE' 'exec supervisord'
    fi
}
chown() {
    printf '%q ' chown "$@" >> "$TEST_COMMAND_LOG"
    printf '\n' >> "$TEST_COMMAND_LOG"
}
export -f docker chown
exec bash "$1" "$2"
"""
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", shell, "fixture", str(SCRIPT), mode],
        env=env,
        capture_output=True,
        text=True,
    )
    commands = [shlex.split(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, commands


def test_stack_shell_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_prepare_preserves_offline_runtime_layout_and_ready_markers(tmp_path, assets):
    result, commands = run_stack(tmp_path, assets, "prepare")
    assert result.returncode == 0, result.stderr
    assert commands == []  # SKIP_EVAL_IMAGE performs no Docker operations.
    runtime = tmp_path / "runtime"
    expected = {
        "python/cpython-3.12.12-linux-x86_64-gnu/bin/python3": RUNTIME_FILES[0],
        "venv/lib/python3.12/site-packages/runtime.txt": RUNTIME_FILES[1],
        "api-extra/python-extra/api.txt": RUNTIME_FILES[2],
        "nltk_data/tokenizers/punkt_tab/english/tokenizer.txt": RUNTIME_FILES[3],
    }
    for relative, text in expected.items():
        assert (runtime / relative).read_text() == text
    for name in ("python", "venv", "api-extra", "nltk_data"):
        assert (runtime / name / ".ready").is_file()
    assert (runtime / "start-eval.sh").read_bytes() == (assets / "tools/start-eval.sh").read_bytes()
    assert (runtime / "start-eval.sh").stat().st_mode & 0o777 == 0o755

    # A ready runtime is reused, not replaced by a second extraction.
    runtime_file = runtime / "api-extra/python-extra/api.txt"
    runtime_file.write_text("already prepared")
    result, commands = run_stack(tmp_path, assets, "prepare")
    assert result.returncode == 0, result.stderr
    assert commands == []
    assert runtime_file.read_text() == "already prepared"


@pytest.mark.parametrize("filename", RUNTIME_FILES)
def test_prepare_requires_all_declared_assets_even_when_eval_load_is_skipped(
    tmp_path, assets, filename
):
    missing = assets / "tools" / filename
    missing.unlink()
    result, commands = run_stack(tmp_path, assets, "prepare")
    assert result.returncode == 1
    assert f"missing required asset: {missing}" in result.stderr
    assert commands == []


def test_prepare_uses_image_lock_for_eval_image_inspection(tmp_path, assets):
    lock = tmp_path / "images.env"
    image = "example/eval@sha256:" + "a" * 64
    lock.write_text(f"EVAL_IMAGE={image}\n")
    result, commands = run_stack(tmp_path, assets, "prepare", skip_eval=False, image_lock=lock)
    assert result.returncode == 0, result.stderr
    assert commands == [
        ["docker", "image", "inspect", image],
        ["docker", "load", "-i", str(assets / "tools" / RUNTIME_FILES[4])],
    ]


def test_missing_image_lock_fails_before_runtime_preparation(tmp_path, assets):
    lock = tmp_path / "missing.env"
    result, commands = run_stack(tmp_path, assets, "prepare", image_lock=lock)
    assert result.returncode == 1
    assert f"missing image lock: {lock}" in result.stderr
    assert commands == []
    assert not (tmp_path / "runtime").exists()


def test_prepare_map_copies_only_writable_databases_and_builds_bindfs_wrapper(tmp_path, assets):
    result, commands = run_stack(tmp_path, assets, "prepare-map", image_present=True)
    assert result.returncode == 0, result.stderr
    runtime = tmp_path / "runtime"
    map_root = runtime / "map-volumes"
    webdav = assets / "assets-full/map-volumes-webdav"
    for database in ("tile-db/postgres", "nominatim-db"):
        copied = map_root / database
        assert (copied / "PG_VERSION").read_text() == "14\n"
        assert not (copied / "postmaster.pid").exists()
        assert (webdav / database / "postmaster.pid").is_file()
        assert (copied / "pg_wal/archive_status").is_dir()
        assert (copied / "pg_logical/snapshots").is_dir()
        assert copied.stat().st_mode & 0o777 == 0o700
        assert ["chown", "-R", "104:106", str(copied)] in commands
    for name in ("tile-db", "nominatim-db"):
        assert (map_root / f".{name}.ready").is_file()
    assert not (map_root / "routing").exists()
    assert not (map_root / "nominatim-flatnode").exists()
    assert not any(str(webdav) in command for command in commands)
    entrypoint = (map_root / "entrypoint.sh").read_text()
    assert "chown -R nominatim" not in entrypoint
    assert ": # read-only shared flatnode" in entrypoint
    wrapper = (runtime / "map-wrapper.sh").read_text()
    assert "bindfs -r -u 1001 -g 1001" in wrapper
    assert "--chown-ignore --chgrp-ignore --chmod-ignore" in wrapper
    assert "/mnt/map-source/nominatim-flatnode /data/nominatim/flatnode" in wrapper
    assert "for profile in car bike foot; do" in wrapper
    assert '"/mnt/map-source/routing-$profile" "/data/routing/$profile"' in wrapper
    assert "sed -i 's/ --mmap//g'" in wrapper
    assert "file=/run/supervisord.sock" in wrapper
    assert 'exec /patched-entrypoint.sh "$@"' in wrapper
    subprocess.run(["bash", "-n", str(runtime / "map-wrapper.sh")], check=True)


@pytest.mark.parametrize("profile", ["car", "bike", "foot"])
def test_prepare_map_requires_all_three_shared_routing_profiles(tmp_path, assets, profile):
    missing = assets / f"assets-full/map-volumes-webdav/routing-{profile}"
    missing /= "us-northeast-latest.osrm.mldgr"
    missing.unlink()
    result, _ = run_stack(tmp_path, assets, "prepare-map", image_present=True)
    assert result.returncode == 1
    assert f"missing required asset: {missing}" in result.stderr


@pytest.mark.parametrize("marker", ["READY", "PERMISSIONS_READY"])
def test_prepare_map_requires_operator_prepared_baseline_markers(tmp_path, assets, marker):
    missing = assets / "assets-full/map-volumes" / marker
    missing.unlink()
    result, _ = run_stack(tmp_path, assets, "prepare-map", image_present=True)
    assert result.returncode == 1
    assert f"missing required asset: {missing}" in result.stderr


@pytest.mark.parametrize("pinned", [True, False])
def test_prepare_map_resolves_locked_runtime_image_without_changing_base_image(
    tmp_path, assets, pinned
):
    lock = tmp_path / "images.env"
    image = "example/map@sha256:" + "b" * 64 if pinned else "example/map:fixture"
    base_image = "example/official-map@sha256:" + "c" * 64
    lock.write_text(f"MAP_RUNTIME_IMAGE={image}\nMAP_IMAGE={base_image}\n")
    result, commands = run_stack(tmp_path, assets, "prepare-map", image_lock=lock)
    assert result.returncode == 0, result.stderr
    assert commands[0] == ["docker", "image", "inspect", image]
    if pinned:
        assert commands[1] == ["docker", "pull", image]
        assert not (tmp_path / "runtime/map-bindfs-image").exists()
    else:
        build_root = tmp_path / "runtime/map-bindfs-image"
        assert commands[1] == ["docker", "build", "--network", "host", "-t", image, str(build_root)]
        dockerfile = (build_root / "Dockerfile").read_text()
        assert dockerfile.startswith(f"FROM {base_image}\n")
        assert "apt-get install -y --no-install-recommends bindfs" in dockerfile
    assert [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "cat",
        base_image,
        "/entrypoint.sh",
    ] in commands
