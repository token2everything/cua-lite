#!/usr/bin/env python3
"""Clone the OSWorld-G upstream benchmark repo.

OSWorld-G's benchmark (564 grounding tasks across bbox / polygon / refusal
shapes) ships inside the GitHub repository ``xlang-ai/OSWorld-G`` rather
than as a HuggingFace dataset. This script clones the repo into the env's
``.cache/OSWorld-G`` directory (gitignored) so that ``main.py`` can register tasks.

Files pulled (under ``benchmark/``):
  - ``OSWorld-G.json`` / ``OSWorld-G_refined.json``: per-task spec
    (id, image_path, bbox/polygon coordinates, GUI_types, instruction).
  - ``classification_result.json`` / ``buckets.json``: paper-canonical
    5-class classification (text_matching / element_recognition /
    layout_understanding / fine_grained_manipulation / refusal). Read by
    ``main.py:_load_paper_categories`` and surfaced on each task's
    ``metadata.others["paper_category"]``. Auto-pulled by the
    shallow clone; if the file is absent, the env still loads and
    ``paper_category`` is just an empty list.

Idempotent: rerunning is a no-op if the clone already exists. Pass
``--force`` to wipe and re-clone.

Usage:
    uv run python lite/gym/envs/osworld_g/scripts/utils/download_tasks.py
    uv run python lite/gym/envs/osworld_g/scripts/utils/download_tasks.py --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

_REPO_URL = "https://github.com/xlang-ai/OSWorld-G.git"
# this file lives at <osworld_g>/scripts/utils/ → the env dir is parents[2]
_ENV_DIR = Path(__file__).resolve().parents[2]
_CACHE_PARENT = _ENV_DIR / ".cache"
_CLONE_DIR = _CACHE_PARENT / "OSWorld-G"


def clone(force: bool = False) -> Path:
    if _CLONE_DIR.exists():
        if force:
            print(f"Removing existing clone at {_CLONE_DIR} (--force) ...")
            shutil.rmtree(_CLONE_DIR)
        else:
            print(f"Already cloned at {_CLONE_DIR} (skip; pass --force to re-clone)")
            return _CLONE_DIR

    _CACHE_PARENT.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {_REPO_URL} → {_CLONE_DIR} ...")
    # Shallow clone to keep the working tree small (no commit history needed).
    subprocess.run(
        ["git", "clone", "--depth", "1", _REPO_URL, str(_CLONE_DIR)],
        check=True,
    )
    return _CLONE_DIR


def summarize(clone_path: Path) -> None:
    bench = clone_path / "benchmark"
    images = bench / "images"
    if not bench.is_dir():
        print(f"WARNING: {bench} not found in clone — repo layout may have changed.")
        return

    for name in ("OSWorld-G.json", "OSWorld-G_refined.json"):
        f = bench / name
        if f.is_file():
            with open(f) as fh:
                data = json.load(fh)
            cnt = Counter(t["box_type"] for t in data)
            print(f"  {name}: {len(data)} tasks  (box_type: {dict(cnt)})")

    if images.is_dir():
        n_imgs = sum(1 for _ in images.iterdir() if _.is_file())
        print(f"  images/: {n_imgs} files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--force", action="store_true", help="Re-clone even if .cache/OSWorld-G already exists.")
    args = ap.parse_args()

    path = clone(force=args.force)
    print()
    summarize(path)


if __name__ == "__main__":
    main()
