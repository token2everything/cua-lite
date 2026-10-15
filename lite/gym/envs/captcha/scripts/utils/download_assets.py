"""Download captcha env assets from HuggingFace into lite/gym/envs/captcha/.cache/assets/.

Assets are image files and JSON configs used by the captcha task servers at
runtime. They are hosted at https://huggingface.co/datasets/OnAnOrange/captcha-assets
and are NOT committed to the repo (``.cache/`` is git-ignored).

Data auto-downloads on first env use. This script is an optional way to
pre-download everything upfront (``install.sh`` calls it too).

Usage:
    uv run python lite/gym/envs/captcha/scripts/utils/download_assets.py
"""

from __future__ import annotations

from pathlib import Path

_REPO_ID = "OnAnOrange/captcha-assets"
# this file lives at <captcha>/scripts/utils/ → the env dir is parents[2]
_ASSETS_DIR = Path(__file__).resolve().parents[2] / ".cache" / "assets"


def download() -> Path:
    """Download captcha assets and return the local assets path."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        _REPO_ID,
        repo_type="dataset",
        local_dir=str(_ASSETS_DIR),
        ignore_patterns=[".gitignore"],
    )
    return Path(path)


def main() -> None:
    print(f"Downloading {_REPO_ID} → {_ASSETS_DIR} ...")
    path = download()
    print(f"Done. Assets at: {path}")

    task_dirs = [d for d in path.iterdir() if d.is_dir()]
    for d in sorted(task_dirs):
        n_files = sum(1 for _ in d.rglob("*") if _.is_file())
        print(f"  {d.name}/: {n_files} files")


if __name__ == "__main__":
    main()
