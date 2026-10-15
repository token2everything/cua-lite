"""GUIOdyssey → cua-lite mobile understanding preprocessor.

GUIOdyssey attaches a rich per-step ``description`` (a screen-only OCR/visual
description that does NOT reference the outer task or prior actions). That makes
each ``(screenshot, description)`` a clean screen-captioning pair, lifted out as
the ``understanding`` cohort alongside ``use``.

Prompt rotation is deterministic — picked from ``CAPTION_PROMPTS`` by
``crc32("ep:step") % len(CAPTION_PROMPTS)`` (a STABLE hash, not the salted
builtin ``hash()``) so a step always gets the same prompt across runs/processes.

Output: ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIOdyssey/mobile/understanding/<split>/<variant>.parquet

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount
    export CUA_LITE_DATASETS_ROOT=/path/to/canonical-output
    uv run python lite/data/preproc/guiodyssey/understanding.py \
        [--dry-run] [--verbose] [--head N]
"""

from __future__ import annotations

import argparse
import os
import sys
import zlib
from collections import Counter, defaultdict
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.data import staging
from lite.data.preproc.guiodyssey.utils import (
    OS,
    PLATFORM,
    SOURCE,
    iter_episodes,
    make_image_store,
    make_splitter,
    out_dir_for,
    screenshot_rel_path,
    stage_entry,
)
from lite.utils.path import resolve_path

VARIANT = "understanding"

CAPTION_PROMPTS: tuple[str, ...] = (
    "Describe this mobile screenshot in detail.",
    "Please describe what is shown on this Android screen.",
    "What UI elements are visible in this screenshot?",
    "Explain the content of this mobile screen.",
)


def _pick_prompt(ep_id: str, step_idx: int) -> str:
    # Stable across runs/processes (builtin hash() is PYTHONHASHSEED-salted for str/tuple).
    h = zlib.crc32(f"{ep_id}:{step_idx}".encode())
    return CAPTION_PROMPTS[h % len(CAPTION_PROMPTS)]


def step_to_record(ep_id: str, step: dict, device_info: dict, task_info: dict) -> dict | None:
    """One step → understanding row, or ``None`` to skip (per-record)."""
    step_idx = step.get("step")
    if not isinstance(step_idx, int):
        return None
    description = step.get("description")
    if not isinstance(description, str) or not description.strip():
        return None

    img_rel = screenshot_rel_path(ep_id, step_idx)
    try:
        img_abs = resolve_path(img_rel, "CUA_LITE_RAW_DATASETS_ROOT")
    except FileNotFoundError:
        return None

    dev_w, dev_h = device_info.get("w"), device_info.get("h")
    return {
        "images": [img_abs],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": _pick_prompt(ep_id, step_idx)},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": description}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=(PLATFORM, "understanding"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={
                "id": f"guiodyssey_{ep_id}_{step_idx}",
                "resolution": [dev_w, dev_h] if isinstance(dev_w, int) and isinstance(dev_h, int) else None,
                "os": OS,
                "source": SOURCE,
                "source_id": f"{ep_id}_{step_idx}",
                "category": task_info.get("category"),
                "apps": task_info.get("app"),
                "device_name": device_info.get("device_name"),
            },
        ).to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GUIOdyssey → cua-lite mobile understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true", help="Show counts, write nothing")
    parser.add_argument("--verbose", action="store_true", help="Per-episode skip logging")
    parser.add_argument("--head", type=int, default=None, help="Process at most N episodes (smoke test)")
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT must be set", file=sys.stderr)
        return 1
    if not args.dry_run and not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT must be set", file=sys.stderr)
        return 1

    out_dir = None if args.dry_run else out_dir_for()
    store = None if args.dry_run else make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    skip_reasons: Counter[str] = Counter()
    n_ep = n_steps = n_kept = 0

    for ep_id, episode in iter_episodes(raw_root, args.head):
        n_ep += 1
        device_info = episode.get("device_info")
        task_info = episode.get("task_info")
        steps = episode.get("steps")
        if not isinstance(device_info, dict) or not isinstance(task_info, dict):
            skip_reasons["bad_episode_meta"] += 1
            continue
        if not isinstance(steps, list):
            skip_reasons["bad_steps"] += 1
            continue
        for step in steps:
            n_steps += 1
            row = step_to_record(ep_id, step, device_info, task_info)
            if row is None:
                desc = step.get("description")
                if not isinstance(step.get("step"), int):
                    skip_reasons["bad_step_idx"] += 1
                elif not isinstance(desc, str) or not desc.strip():
                    skip_reasons["empty_description"] += 1
                else:
                    skip_reasons["missing_image"] += 1
                continue
            n_kept += 1
            if args.dry_run:
                continue
            try:
                bk, e = stage_entry(row, store=store, splitter=splitter, variant=VARIANT)
            except staging.CorruptImageError as img_err:
                # Corrupt/truncated screenshot → drop this step.
                n_kept -= 1
                skip_reasons["corrupt_image"] += 1
                if args.verbose:
                    print(f"  [skip-corrupt] {ep_id}: {img_err}", file=sys.stderr)
                continue
            buffers[bk].append(e)

    print(f"Episodes: {n_ep}  steps: {n_steps}  kept: {n_kept}  skipped: {n_steps - n_kept}")
    if skip_reasons:
        print("Skip reasons:", dict(skip_reasons))
    if args.dry_run:
        return 0

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
