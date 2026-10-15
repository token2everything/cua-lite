"""Pinned ScaleCUA task-source locations for ``lite.scalecua``.

This module is intentionally small and import-light. Runtime registration uses
only local generated catalogs; upstream task-source import happens through
``scripts/utils/tasks.sh`` or the default ``install.sh`` path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from lite.gym.utils.config.manifest import load_asset_lock
from lite.utils.path import project_root

ENV_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = project_root()
CACHE_DIR = REPO_ROOT / ".cache" / "lite.scalecua_tasks"
CATALOG_DIR = ENV_DIR / "data"
STAGING_DIR = REPO_ROOT / ".cache" / "lite.scalecua_tasks.__staging__"
PROBE_SNAPSHOT = REPO_ROOT / ".cache" / "scalecua_hf_probe"
OSWORLD_EVAL_JSONL = (
    REPO_ROOT / "lite" / "gym" / "envs" / "lite" / "osworld" / "data" / "eval.jsonl"
)

PIN = load_asset_lock(ENV_DIR)
REPO = PIN.repo
REVISION = PIN.revision
FILE_CACHE_REVISION = "711e0811642364e7aa8f10a8918367d0b626d578"

ALLOW_PATTERNS = (
    "README.md",
    "osworld/generated_tasks/**/*.json",
    "osworld/rl_tasks/**/*.json",
    "osworld/judge_functions/generated_tasks/**",
    "osworld/judge_functions/rl_tasks/**",
)

HF_MAX_WORKERS = int(os.environ.get("LITE_SCALECUA_HF_MAX_WORKERS", "2"))
HF_TRANSPORT = os.environ.get("LITE_SCALECUA_HF_TRANSPORT", "git")


def asset_snapshot() -> dict[str, object]:
    return {
        "repo": REPO,
        "revision": REVISION,
        "file_cache_revision": FILE_CACHE_REVISION,
        "components": {
            name: component["path"]
            for name, component in sorted(PIN.components.items())
        },
    }


def asset_identity() -> str:
    return json.dumps(asset_snapshot(), sort_keys=True, separators=(",", ":"))


def ensure_hf_snapshot(dest: Path, *, force_download: bool = False) -> Path:
    """Return a local pinned ScaleCUA HF snapshot containing OSWorld tasks.

    The repo already carries a probe snapshot in ``.cache/scalecua_hf_probe`` on
    many developer machines. Reusing it avoids a multi-GB re-download while the
    lock still controls the generated task-source identity. If it is absent,
    fetch the pinned allowlist from Hugging Face.
    """
    osworld_dir = dest / "osworld"
    identity_file = dest / ".asset_identity"
    complete_file = dest / ".complete"
    expected_identity = asset_identity()
    if (
        osworld_dir.is_dir()
        and complete_file.is_file()
        and identity_file.is_file()
        and identity_file.read_text().strip() == expected_identity
        and not force_download
    ):
        return dest

    probe_osworld = PROBE_SNAPSHOT / "osworld"
    if probe_osworld.is_dir() and not force_download and _probe_matches_expected(PROBE_SNAPSHOT):
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(PROBE_SNAPSHOT, dest, ignore=shutil.ignore_patterns(".cache"))
        identity_file.write_text(expected_identity + "\n")
        complete_file.touch()
        return dest

    from huggingface_hub import snapshot_download

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if HF_TRANSPORT != "http":
        try:
            _git_sparse_snapshot(dest)
            identity_file.write_text(expected_identity + "\n")
            complete_file.touch()
            return dest
        except Exception as exc:
            print(
                "[lite.scalecua tasks] Git sparse fetch failed; "
                f"falling back to HF HTTP snapshot: {exc}",
                file=sys.stderr,
            )
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=list(ALLOW_PATTERNS),
        local_dir=str(dest),
        force_download=force_download,
        max_workers=HF_MAX_WORKERS,
    )
    identity_file.write_text(expected_identity + "\n")
    complete_file.touch()
    return dest


def _git_sparse_snapshot(dest: Path) -> None:
    """Fetch ScaleCUA task-source files as a sparse Git checkout.

    The official HF snapshot has tens of thousands of tiny JSON files. The HTTP
    snapshot path works but can hit HF rate limits on clean machines. Sparse Git
    fetches the same pinned commit with far fewer requests, then drops ``.git``
    so the cache has the same shape as ``snapshot_download``.
    """

    tmp = dest.with_name(dest.name + ".__git_tmp__")
    if tmp.exists():
        shutil.rmtree(tmp)
    url = f"https://huggingface.co/datasets/{REPO}"
    sparse_paths = [
        "README.md",
        "osworld/generated_tasks",
        "osworld/rl_tasks",
        "osworld/judge_functions/generated_tasks",
        "osworld/judge_functions/rl_tasks",
    ]
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--sparse",
                "--no-checkout",
                url,
                str(tmp),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp), "sparse-checkout", "set", *sparse_paths],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp), "checkout", "--quiet", REVISION],
            check=True,
        )
        shutil.rmtree(tmp / ".git", ignore_errors=True)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def _probe_matches_expected(root: Path) -> bool:
    """Cheap guard for the developer probe snapshot reused during import."""
    expected = {
        "osworld/generated_tasks": 20289,
        "osworld/rl_tasks": 2049,
    }
    for rel, count in expected.items():
        if len(list((root / rel).glob("*/*.json"))) != count:
            return False
    return all(
        (root / "osworld" / "judge_functions" / name / "getters.py").is_file()
        and (root / "osworld" / "judge_functions" / name / "metrics.py").is_file()
        for name in ("generated_tasks", "rl_tasks")
    )
