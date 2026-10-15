#!/usr/bin/env python3
"""Pre-download ScreenSpot-Pro dataset into the HuggingFace cache.

Uses huggingface_hub.snapshot_download() to fetch annotation JSONs and
screenshot PNGs. Data is cached in ~/.cache/huggingface/hub/.

Data auto-downloads on first env use. This script is an optional way to
pre-download everything upfront.

Usage:
    uv run python lite/gym/envs/screenspot_pro/scripts/utils/download_tasks.py
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ID = "likaixin/ScreenSpot-Pro"

def download() -> Path:
    """Download ScreenSpot-Pro and return the local snapshot path."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        _REPO_ID,
        repo_type="dataset",
        allow_patterns=["annotations/*.json", "images/**/*.png"],
    )
    return Path(path)

def main():
    print(f"Downloading {_REPO_ID} ...")
    path = download()
    print(f"Cached at: {path}")

    annotations_dir = path / "annotations"
    images_dir = path / "images"

    if annotations_dir.is_dir():
        json_files = sorted(annotations_dir.glob("*.json"))
        total_tasks = 0
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            total_tasks += len(data)
            print(f"  {jf.name}: {len(data)} tasks")
        print(f"\nAnnotation files: {len(json_files)}, total tasks: {total_tasks}")

    if images_dir.is_dir():
        img_count = sum(1 for _ in images_dir.rglob("*.png"))
        print(f"Images: {img_count}")

if __name__ == "__main__":
    main()
