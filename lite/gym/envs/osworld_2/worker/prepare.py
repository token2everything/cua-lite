"""Export an immutable Worker build context from a selected upstream release.

python -m lite.gym.envs.osworld_2.worker.prepare --release osworld-v2.1 \
    --source /path/to/OSWorld-V2 --wheel dist/lite-0.1.0-py3-none-any.whl \
    --output /path/to/new-build-context
The source checkout is read-only; only committed files at the manifest tag are
exported. Task code, assets, VM disks and credentials remain runtime mounts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from lite.gym.envs.osworld_2.catalog import DEFAULT_RELEASE, load_release


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True, help="Built CUA-Lite wheel")
    return parser.parse_args(argv)


def prepare(args):
    """Pin code and metadata together without altering the caller's checkout."""
    release = load_release(args.release)
    wheel = args.wheel.expanduser().resolve()
    if not wheel.is_file() or not wheel.name.startswith("lite-") or wheel.suffix != ".whl":
        raise ValueError("--wheel must name the built CUA-Lite wheel")
    source = args.source.expanduser().resolve()
    tag = release.manifest["osworld_code"]["tag"]
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        text=True,
    ).strip()
    upstream_manifest = subprocess.check_output(
        ["git", "-C", str(source), "show", f"{revision}:benchmark_releases/{release.name}.json"]
    )
    if hashlib.sha256(upstream_manifest).hexdigest() != release.manifest_sha256:
        raise ValueError("Source tag and packaged release manifest differ")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "osworld-code.tar.gz"
    subprocess.run(
        ["git", "-C", str(source), "archive", "--format=tar.gz", f"--output={archive}", revision],
        check=True,
    )
    (output / "release").mkdir()
    for name in ("release.json", "task_hashes.json"):
        shutil.copyfile(release.path / name, output / "release" / name)
    for name in (
        "Dockerfile", "worker_runtime.py", "worker_service.py", "worker-entrypoint.sh",
        "provider_image_override.py",
    ):
        shutil.copyfile(Path(__file__).parent / name, output / name)
    shutil.copyfile(wheel, output / wheel.name)
    from lite.platforms.sandbox import osworld_2 as platform

    for name in ("launch_worker.sh", "launch_rollout.sh"):
        shutil.copyfile(Path(platform.__file__).parent / name, output / name)
    with archive.open("rb") as stream:
        archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {
        "release": release.name,
        "release_manifest_sha256": release.manifest_sha256,
        "code_tag": tag,
        "code_commit": revision,
        "code_archive_sha256": archive_sha256,
        "cua_wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }
    (output / "release" / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return {**record, "build_context": str(output)}


if __name__ == "__main__":
    print(json.dumps(prepare(_parse_args()), indent=2))
