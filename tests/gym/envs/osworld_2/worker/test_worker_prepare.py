"""Build contexts must use pinned committed source and never include local tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile

import pytest

from lite.gym.envs.osworld_2.catalog import load_release, release_names
from lite.gym.envs.osworld_2.worker.prepare import prepare


@pytest.fixture(params=release_names())
def source(tmp_path, request):
    repo = tmp_path / "source"
    repo.mkdir()
    release = load_release(request.param)
    (repo / "benchmark_releases").mkdir()
    manifest = repo / "benchmark_releases" / f"{release.name}.json"
    manifest.write_bytes((release.path / "release.json").read_bytes())
    (repo / "task_loader.py").write_text("pinned source\n")
    for command in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "pin"],
        ["tag", release.manifest["osworld_code"]["tag"]],
    ):
        subprocess.run(["git", "-C", str(repo), *command], check=True)
    (repo / "task_loader.py").write_text("local edited source\n")
    (repo / "private-task.py").write_text("must remain local\n")
    wheel = tmp_path / "lite-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"test build wheel")
    return argparse.Namespace(
        release=release.name, source=repo, output=tmp_path / "context", wheel=wheel,
    )


def test_context_uses_tag_without_touching_dirty_checkout(source):
    record = prepare(source)
    with tarfile.open(source.output / "osworld-code.tar.gz") as archive:
        assert archive.extractfile("task_loader.py").read() == b"pinned source\n"
        assert "private-task.py" not in archive.getnames()
    assert (source.source / "task_loader.py").read_text() == "local edited source\n"
    assert (source.source / "private-task.py").exists()
    assert json.loads((source.output / "release" / "build.json").read_text())["code_commit"] == (
        record["code_commit"]
    )
    release = load_release(source.release)
    assert (source.output / source.wheel.name).read_bytes() == source.wheel.read_bytes()
    assert len(record["cua_wheel_sha256"]) == 64
    assert (source.output / "launch_rollout.sh").is_file()
    for name in ("release.json", "task_hashes.json"):
        assert (source.output / "release" / name).read_bytes() == (release.path / name).read_bytes()
    with pytest.raises(FileExistsError):
        prepare(source)


def test_mismatching_tag_is_rejected_before_writing_context(source):
    manifest = source.source / "benchmark_releases" / f"{source.release}.json"
    manifest.write_text('{}\n')
    subprocess.run(["git", "-C", str(source.source), "add", str(manifest)], check=True)
    subprocess.run(
        ["git", "-C", str(source.source), "-c", "user.name=Test", "-c",
         "user.email=test@example.invalid", "commit", "-qm", "different manifest"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source.source), "tag", "-f",
         load_release(source.release).manifest["osworld_code"]["tag"]], check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="Source tag"):
        prepare(source)
    assert not source.output.exists()
