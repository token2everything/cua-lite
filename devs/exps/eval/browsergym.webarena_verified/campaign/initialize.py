#!/usr/bin/env python3
"""Create an offline, credential-free native WebArena-Verified campaign bundle.

Run ``uv run python devs/exps/eval/browsergym.webarena_verified/campaign/initialize.py
--help`` for the required local catalog, image lock and shared asset inputs.
This command hashes assets and freezes tracked source; it never provisions a
sandbox, reads a model key, calls a model or starts an evaluation. A completed
bundle is consumed by ``orchestrate.py`` with a separately supplied private env.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
EVAL_RELATIVE = "devs/exps/eval/browsergym.webarena_verified"
DATASET_SHA256 = "d65275660814663375028e9017e1f929e3c38321041b125795e2713b52243d30"
OSWORLD_COMMIT = "b138d348256078fa634fc3b73567a7337c793e6b"
IMAGE_NAMES = {
    "SHOPPING_IMAGE",
    "SHOPPING_ADMIN_IMAGE",
    "REDDIT_IMAGE",
    "GITLAB_IMAGE",
    "MAP_IMAGE",
    "MAP_RUNTIME_IMAGE",
    "QWENRUNNER_RUNTIME_IMAGE",
    "WIKIPEDIA_RUNTIME_IMAGE",
}
ASSET_FILES = (
    "tools/python312-playwright144-chromium1117.tar.gz",
    "tools/webarena-verified-venv.tar.gz",
    "tools/webarena-api-extra.tar.gz",
    "tools/nltk-punkt-tab.tar.gz",
    "tools/webarena-eval-base-ubuntu22-playwright144.tar.gz",
    "tools/start-eval.sh",
    "tools/kiwix-serve-3.3.0-amd64.tar.gz",
    "assets-full/map-volumes/READY",
    "assets-full/map-volumes/PERMISSIONS_READY",
    "assets-full/map-volumes-webdav/routing-car/us-northeast-latest.osrm.mldgr",
    "assets-full/map-volumes-webdav/routing-bike/us-northeast-latest.osrm.mldgr",
    "assets-full/map-volumes-webdav/routing-foot/us-northeast-latest.osrm.mldgr",
    "assets-full/data/wikipedia_en_all_maxi_2022-05.zim",
    "assets-full/homepage-src/environment_docker/webarena-homepage/app.py",
)
ASSET_DIRECTORIES = (
    "assets-full/map-volumes-webdav/tile-db",
    "assets-full/map-volumes-webdav/nominatim-db",
    "assets-full/map-volumes-webdav/routing-car",
    "assets-full/map-volumes-webdav/routing-bike",
    "assets-full/map-volumes-webdav/routing-foot",
    "assets-full/map-volumes/nominatim-flatnode",
    "assets-full/homepage-src/environment_docker/webarena-homepage",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="new or empty run directory")
    parser.add_argument("--model", required=True, choices=("qwen3.8-max", "glm-5.3-flash"))
    parser.add_argument("--run-id", required=True, help="wav-generic-<slug>-full120-YYYYMMDD-vN")
    parser.add_argument("--name-prefix", required=True, help="unique lowercase Docker name prefix")
    parser.add_argument("--catalog", required=True, type=Path, help="official 1.2.3 catalog file")
    parser.add_argument(
        "--image-lock",
        required=True,
        type=Path,
        help="seven digest-pinned images plus offline-archive-bound Kiwix",
    )
    parser.add_argument("--asset-root", required=True, type=Path, help="local shared asset mount")
    parser.add_argument(
        "--desktop-assets-relative",
        required=True,
        help="directory within the asset mount containing desktop Python and OSWorld bundle",
    )
    parser.add_argument("--sandbox-image", required=True, help="operator-supplied DinD image URI")
    parser.add_argument("--asset-subpath", required=True, help="platform shared-volume subpath")
    parser.add_argument("--dockerd-no-proxy", default="localhost,127.0.0.1,::1")
    parser.add_argument(
        "--timing",
        type=Path,
        help="optional 812 JSONL task_id/estimated_seconds rows; absent means uniform scheduling",
    )
    return parser.parse_args(argv)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_path(path: Path, *, directory: bool = False, root: Path | None = None) -> Path:
    path = path.absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError(f"symlink paths are not allowed: {path}")
    resolved = path.resolve(strict=True)
    if root is not None and root != resolved and root not in resolved.parents:
        raise ValueError(f"asset path escapes its root: {path}")
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        raise ValueError(f"expected a regular {'directory' if directory else 'file'}: {path}")
    return resolved


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." == value
        or "\\" in value
        or str(path) != value
    ):
        raise ValueError(f"a normalized relative asset path is required: {value}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")
    path.chmod(0o600)


def initialize(args: argparse.Namespace) -> dict[str, Any]:
    """Validate local inputs, freeze actual tracked bytes, and publish one immutable bundle."""
    root = args.root.absolute()
    if any(parent.is_symlink() for parent in (root, *root.parents)):
        raise ValueError("run root must not contain symlinks")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("run root must be new or empty; refusing to overwrite artifacts")
    root = root.resolve()
    if not re.fullmatch(r"wav-generic-[a-z0-9]+-full120-[0-9]{8}-v[1-9][0-9]*", args.run_id):
        raise ValueError("run ID must use wav-generic-<slug>-full120-YYYYMMDD-vN")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,40}", args.name_prefix):
        raise ValueError("invalid Docker name prefix")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]*", args.sandbox_image):
        raise ValueError("invalid sandbox image URI")
    if not re.fullmatch(r"[A-Za-z0-9.,:_-]+", args.dockerd_no_proxy):
        raise ValueError("invalid dockerd no_proxy list")
    desktop_relative = _relative(args.desktop_assets_relative)
    asset_subpath = _relative(args.asset_subpath)
    asset_root = _checked_path(args.asset_root, directory=True)
    catalog = _checked_path(args.catalog)
    catalog_payload = catalog.read_bytes()
    if hashlib.sha256(catalog_payload).hexdigest() != DATASET_SHA256:
        raise ValueError("catalog must match the exact official WebArena-Verified 1.2.3 checksum")
    image_lock = _checked_path(args.image_lock)
    image_payload = image_lock.read_bytes()
    image_values = {}
    for line in image_payload.decode().splitlines():
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        pinned_digest = bool(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", value)
        )
        offline_kiwix = (
            name == "WIKIPEDIA_RUNTIME_IMAGE" and value == "ghcr.io/kiwix/kiwix-serve:wav-1.2.3"
        )
        if (
            not separator
            or name not in IMAGE_NAMES
            or name in image_values
            or not (pinned_digest or offline_kiwix)
        ):
            raise ValueError("image lock requires digest pins or the archive-bound Kiwix reference")
        image_values[name] = value
    if set(image_values) != IMAGE_NAMES:
        raise ValueError("image lock must provide all eight required images")
    # Copy validated image identities only; arbitrary comments are not campaign inputs.
    image_payload = "".join(f"{name}={value}\n" for name, value in image_values.items()).encode()
    if args.timing is not None:
        _checked_path(args.timing)

    # Shared runtime archives are external inputs, not copied into the source bundle.
    # Directory checks prove presence and safe traversal, not full database-content identity.
    asset_files = {
        relative: _checked_path(asset_root / relative, root=asset_root)
        for relative in (
            *ASSET_FILES,
            f"{desktop_relative}/desktop-python.tar.gz",
            f"{desktop_relative}/osworld-b138d348.bundle",
        )
    }
    for relative in ASSET_DIRECTORIES:
        directory = _checked_path(asset_root / relative, directory=True, root=asset_root)
        for current, directories, files in os.walk(directory, followlinks=False):
            for name in (*directories, *files):
                if (Path(current) / name).is_symlink():
                    raise ValueError(f"symlink within asset directory: {relative}")
    assets = {
        "schema_version": 1,
        "assets_mount": "/mnt/webarena-verified",
        "desktop_assets_relative": desktop_relative,
        "files": {
            relative: {"sha256": _sha(path), "size": path.stat().st_size}
            for relative, path in asset_files.items()
        },
        "directories": list(ASSET_DIRECTORIES),
        "offline_image_archives": {
            "WIKIPEDIA_RUNTIME_IMAGE": "tools/kiwix-serve-3.3.0-amd64.tar.gz"
        },
        "directory_verification": "presence_and_symlink_boundary_only_not_content_hash",
        "checked_at": datetime.now(UTC).isoformat(),
    }
    template_path = REPO_ROOT / EVAL_RELATIVE / "EXPERIMENT_CONFIG.generic-desktop-dual-v2.json"
    config = json.loads(template_path.read_text())
    selected = next(
        profile for profile in config.pop("experiments") if profile["model"] == args.model
    )
    config.update(selected)
    config.update(status="initialized_offline", run_id=args.run_id)
    config["sandbox"] = {
        "image": args.sandbox_image,
        "asset_subpath": asset_subpath,
        "owner": "cua-lite-webarena",
        "dockerd_no_proxy": args.dockerd_no_proxy,
    }
    config["assets"] = {"desktop_assets_relative": desktop_relative}
    if config["benchmark"]["dataset_sha256"] != DATASET_SHA256:
        raise ValueError("preset dataset checksum differs from the pinned benchmark")

    # Select only tracked, regular source files. Dotenv files, results, benchmark
    # data and host caches can never enter the source allowlist.
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO_ROOT).decode().split("\0")
    names = sorted(
        {
            name
            for name in tracked
            if name
            and (
                name.startswith(("lite/", "scripts/", f"{EVAL_RELATIVE}/campaign/"))
                or name in {"pyproject.toml", "uv.lock", f"{EVAL_RELATIVE}/sandbox_stack.sh"}
            )
        }
    )
    sources = {name: _checked_path(REPO_ROOT / name, root=REPO_ROOT) for name in names}
    required = {
        config["entrypoint"],
        "scripts/webarena_verified_case_receipt.py",
        f"{EVAL_RELATIVE}/sandbox_stack.sh",
        *(
            f"{EVAL_RELATIVE}/campaign/{name}"
            for name in (
                "prepare-worker.sh",
                "run-case.sh",
                "run-worker.sh",
            )
        ),
    }
    if not required <= sources.keys():
        raise ValueError("tracked source is missing a required campaign runner or worker file")
    source_artifacts = {name: _sha(path) for name, path in sources.items()}
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()

    # Every write belongs to this new run root; inputs and other runs stay untouched.
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    _write_json(root / "EXPERIMENT_CONFIG.json", config)
    _write_json(root / "ASSET_MANIFEST.json", assets)
    with (root / "webarena-verified.json").open("xb") as output:
        output.write(catalog_payload)
    with (root / "image-lock.env").open("xb") as output:
        output.write(image_payload)
    spec = importlib.util.spec_from_file_location(
        "webarena_campaign_plan", Path(__file__).with_name("generate_plan.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("campaign plan owner could not be loaded")
    plan_owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan_owner)
    plan_owner.generate_plan(
        root, root / "EXPERIMENT_CONFIG.json", root / "webarena-verified.json", args.timing
    )
    archive_path = root / "cua-lite-source.tar.gz"
    with archive_path.open("xb") as output, tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, path in sources.items():
            archive.add(path, arcname="./" + name, recursive=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        for name, expected in source_artifacts.items():
            member = archive.extractfile("./" + name)
            if member is None:
                raise ValueError(f"source archive omitted {name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: member.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"source changed during freeze: {name}")
    for name, relative in (
        *(
            (name, f"{EVAL_RELATIVE}/campaign/{name}")
            for name in (
                "prepare-worker.sh",
                "run-case.sh",
                "run-worker.sh",
            )
        ),
        ("sandbox_stack.sh", f"{EVAL_RELATIVE}/sandbox_stack.sh"),
    ):
        with (REPO_ROOT / relative).open("rb") as source, (root / name).open("xb") as output:
            shutil.copyfileobj(source, output)
        if _sha(root / name) != source_artifacts[relative]:
            raise ValueError(f"worker source changed during freeze: {name}")
    runner_sha = source_artifacts[config["entrypoint"]]
    with (root / "campaign.env").open("x") as output:
        output.write(
            f"RUN_ID={args.run_id}\nRUNNER_SHA256={runner_sha}\nNAME_PREFIX={args.name_prefix}\n"
        )
    manifest = {
        "experiment_version": config["experiment_version"],
        "baseline_git_commit": config["baseline_git_commit"],
        "source_git_commit": source_commit,
        "source_sha256": _sha(archive_path),
        "source_artifacts": source_artifacts,
        "runner_sha256": runner_sha,
        "image_lock_sha256": _sha(root / "image-lock.env"),
        "config_sha256": _sha(root / "EXPERIMENT_CONFIG.json"),
        "plan_sha256": _sha(root / "PLAN_MANIFEST.sha256"),
        "dataset_sha256": DATASET_SHA256,
        "evaluator_sha256": config["benchmark"]["evaluator_sha256"],
        "asset_manifest_sha256": _sha(root / "ASSET_MANIFEST.json"),
        "sandbox_stack_sha256": _sha(root / "sandbox_stack.sh"),
        "osworld_commit": OSWORLD_COMMIT,
        "harness": config["agent"]["class"],
        "loop": config["agent"]["loop"],
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "final_repair": "official schema validation and visible feedback once",
    }
    _write_json(root / "VERSION_MANIFEST.json", manifest)
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    return {"run_id": args.run_id, "model": args.model, "workers": 120, "tasks": 812}


def main(argv: list[str] | None = None) -> int:
    """Parse public inputs and construct a local bundle without remote side effects."""
    print(json.dumps(initialize(_parse_args(argv))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
