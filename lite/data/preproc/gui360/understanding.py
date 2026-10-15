"""GUI-360 Screen Parsing Data Preprocessor.

Processes the ``screen_parsing_train_resize`` subset of GUI-360 into the
canonical ``understanding`` cohort (variant ``screen_parsing``):

    ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUI-360/desktop/understanding/<split>/<variant>.parquet

Screen parsing is a full-screen UI element-detection task: given one screenshot,
list every interactive control (from the Windows UIA accessibility tree) as
``{control_text, control_rect}``. It carries no tool call, so it belongs to the
``understanding`` family (plain structured-text assistant output).

The upstream answer carries three extra fields the prompt never requests
(``control_type``, ``source="uia"``, ``label``); we drop them and keep only
``{control_text, control_rect}``, with each rect normalized to [0, 1000]. The
user prompt is rewritten so it is consistent with the normalized coordinates.

Upstream rects are UIA bounds from the pre-resize screenshot, so 4.7% of them
fall outside the resized image. They are clipped to it, and a control whose
clipped rect has no area is dropped — see :func:`clip_rect_to_image` for the
measurement behind that split. Both tallies are printed on every run.

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount
    export CUA_LITE_DATASETS_ROOT=/path/to/canonical-output
    uv run python lite/data/preproc/gui360/understanding.py \
        [--dry-run] [--verbose] [--head N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from lite.core.metadata import LiteCUAMetadata
from lite.data import staging
from lite.data.preproc.common import prestage_images
from lite.data.preproc.gui360.utils import (
    OS,
    PLATFORM,
    SOURCE,
    iter_json_array,
    make_image_store,
    make_splitter,
    out_dir_for,
    stage_entry,
)
from lite.utils.path import resolve_path

VARIANT = "screen_parsing"
SUBSET_REL = "vyokky/GUI-360/processed_data/screen_parsing_train_resize"
SOURCE_ID = "screen_parsing_train_resize"

PROMPT = (
    "List all interactive controls (anything that can be clicked, typed into, "
    "or selected) visible in this screenshot. For each control, give its "
    "visible text and bounding box.\n\n"
    "Respond with a JSON array of objects, each "
    '{"control_text": <string>, "control_rect": [left, top, right, bottom]}. '
    "Use an empty string for controls with no visible text. Coordinates are "
    "normalized to the [0, 1000] range, where [0, 0] is the top-left corner and "
    "[1000, 1000] is the bottom-right corner. Output only the JSON array."
)


def normalize_rect(rect: tuple[float, float, float, float], width: int, height: int) -> list[int]:
    """``(left, top, right, bottom)`` pixels → [0, 1000] integers."""
    left, top, right, bottom = rect
    return [
        round(left / width * 1000),
        round(top / height * 1000),
        round(right / width * 1000),
        round(bottom / height * 1000),
    ]


def _parse_control_rect(rect: Any, record_id: str) -> tuple[float, float, float, float]:
    """Read one upstream ``control_rect`` as four numbers.

    The only question this asks is whether the annotation *is* a rect. Where it
    sits relative to the image is a geometric question, answered by
    :func:`clip_rect_to_image` — never by refusing the record.
    """
    if not (
        isinstance(rect, list)
        and len(rect) == 4
        and all(isinstance(v, (int, float)) for v in rect)
    ):
        raise ValueError(
            f"Malformed control_rect in screen_parsing record {record_id}: {rect}"
        )
    left, top, right, bottom = (float(v) for v in rect)
    return left, top, right, bottom


def clip_rect_to_image(
    rect: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Clip an upstream rect to the image; ``None`` when none of it is on screen.

    ``control_rect`` values are Windows-UIA bounds recorded against the
    *pre-resize* screenshot, so they routinely overflow the resized image by the
    resize rounding error alone. Measured over a 2,400-record / 801,796-control
    sample spread across 20 byte offsets of ``training_data.json``: 37,818 rects
    (4.7%) fall outside the image, and 24,898 of those overflow by exactly 2 px
    of 1036 — noise, for which clipping is a no-op on the control's identity.

    The remainder is not noise: 322 of the 37,818 (0.85%) have an **empty**
    intersection with the image — a spreadsheet row at ``top=728`` on a
    728-px-tall screenshot, a PowerPoint thumbnail at ``top=2931`` — joined by 98
    in-bounds ``[8, 8, 8, 8]`` phantoms with no text and no area. The prompt asks
    for the controls *"visible in this screenshot"*, so those have no answer to
    give and are dropped.

    A pixel tolerance cannot make this call: the two populations **overlap**
    between 20 px and 200 px of overflow (3,875 clippable vs 70 off-screen at
    6–20 px; 215 vs 100 at 51–200 px, minimum off-screen overflow 20 px). The
    intersection test separates them exactly, which is why the rule is geometric
    rather than a threshold.
    """
    left, top, right, bottom = rect
    left = min(max(left, 0.0), width)
    top = min(max(top, 0.0), height)
    right = min(max(right, 0.0), width)
    bottom = min(max(bottom, 0.0), height)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def build_controls(
    raw_controls: list[dict[str, Any]], width: int, height: int, record_id: str
) -> tuple[list[dict[str, Any]], int, int]:
    """Trim each control to ``{control_text, control_rect}`` with normalized rect.

    Returns ``(controls, n_clipped, n_offscreen)``; the caller prints both
    tallies unconditionally, so neither a clipped nor a dropped control is ever
    silent.
    """
    out = []
    n_clipped = 0
    n_offscreen = 0
    for ctrl in raw_controls:
        rect = _parse_control_rect(ctrl.get("control_rect"), record_id)
        clipped = clip_rect_to_image(rect, width, height)
        if clipped is None:
            n_offscreen += 1
            continue
        if clipped != rect:
            n_clipped += 1
        out.append(
            {
                "control_text": ctrl.get("control_text", ""),
                "control_rect": normalize_rect(clipped, width, height),
            }
        )
    return out, n_clipped, n_offscreen


def process_records(json_path: str, *, head: int | None, verbose: bool):
    """Yield canonical understanding/screen_parsing entries from the subset JSON."""
    n_seen = 0
    n_yielded = 0
    n_missing = 0
    n_empty = 0
    n_no_visible = 0
    n_clipped = 0
    n_offscreen = 0
    for record in iter_json_array(json_path):
        if head is not None and n_seen >= head:
            break
        n_seen += 1
        record_id = record["id"]

        images = record.get("images") or []
        if not images:
            raise ValueError(f"No image field in screen_parsing record {record_id}")
        rel = images[0].replace("\\", "/")
        try:
            abs_image = resolve_path(f"{SUBSET_REL}/{rel}", "CUA_LITE_RAW_DATASETS_ROOT")
        except FileNotFoundError as e:
            n_missing += 1
            if verbose:
                print(f"  Skip (missing image) {record_id}: {e}")
            continue

        gpt_text = record["conversation"][1]["value"]
        raw_controls = json.loads(gpt_text)

        # A handful of screens have no enabled controls; skip empty lists.
        if not raw_controls:
            n_empty += 1
            continue

        with Image.open(abs_image) as image:
            width, height = image.size
        controls, rec_clipped, rec_offscreen = build_controls(
            raw_controls, width, height, record_id
        )
        n_clipped += rec_clipped
        n_offscreen += rec_offscreen

        # Every control was off-screen: the screen has nothing to list, exactly
        # like the empty-upstream-list case above.
        if not controls:
            n_no_visible += 1
            if verbose:
                print(f"  Skip (no on-screen controls) {record_id}")
            continue

        answer = json.dumps(controls, ensure_ascii=False)

        n_yielded += 1
        yield {
            "images": [abs_image],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 0},
                        {"type": "text", "text": PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            ],
            "metadata": LiteCUAMetadata(
                dims=(PLATFORM, "understanding"),
                extra_tool_schemas=[],
                valid_actions=None,
                others={
                    "id": record_id,
                    "resolution": [width, height],
                    "os": OS,
                    "source": SOURCE,
                    "source_id": SOURCE_ID,
                },
            ).to_dict(),
        }

    # Printed unconditionally, not under --verbose: a per-control drop that only
    # shows up in verbose mode is how a partition silently loses a third of its
    # rows under a clean success line.
    print(f"  Records: {n_seen} read, {n_yielded} emitted")
    print(
        f"    skipped: {n_missing} missing image, {n_empty} no controls upstream, "
        f"{n_no_visible} no on-screen controls"
    )
    print(
        f"  Controls: {n_clipped} clipped to image bounds, "
        f"{n_offscreen} dropped (no visible area)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process GUI-360 screen-parsing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan, write nothing")
    parser.add_argument("--verbose", action="store_true", help="Per-record skip logging")
    parser.add_argument(
        "--head", type=int, default=None, help="Process at most N records (smoke test)"
    )
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT must be set")
        return 1
    json_path = os.path.join(raw_root, SUBSET_REL, "training_data.json")
    if not os.path.exists(json_path):
        print(f"Error: source not found: {json_path}")
        return 1

    if args.dry_run:
        print(
            f"=== DRY RUN ===\nWould read {json_path}\n"
            "Would write desktop/understanding/<split>/<variant>.parquet"
        )
        return 0

    if not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT must be set")
        return 1

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    n_corrupt = 0

    pending: list[dict[str, Any]] = []

    def stage_pending() -> None:
        nonlocal n_corrupt
        if not pending:
            return
        corrupt_paths = prestage_images(store, (entry["images"][0] for entry in pending))
        for entry in pending:
            if Path(entry["images"][0]) in corrupt_paths:
                n_corrupt += 1
                continue
            bk, staged = stage_entry(entry, store=store, splitter=splitter, variant=VARIANT)
            buffers[bk].append(staged)
        pending.clear()

    for entry in process_records(json_path, head=args.head, verbose=args.verbose):
        pending.append(entry)
        if len(pending) == 512:
            stage_pending()
    stage_pending()

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(f"Dropped {n_corrupt} rows with corrupt images")
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
