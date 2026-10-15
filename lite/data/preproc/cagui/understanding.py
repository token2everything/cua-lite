"""CAGUI Grounding (cap + ocr) -> CUA-Lite understanding preprocessor.

Converts the two grounding JSONL files in OpenBMB/CAGUI into the canonical
local layout consumed by ``lite.data.hf.upload`` and
``lite.train.export.export_sft``:

  - cap.jsonl  (bbox2function): Chinese functional description of a UI element
                                inside a given bounding box.
  - ocr.jsonl  (bbox2text):     Chinese OCR text inside a given bounding box.

Both files provide only a (bbox, ground-truth-text) pair — there is no
question prompt in the source. We invent a generic Chinese instruction that
embeds the bbox in [0, 1000] normalized integer coordinates (the project-wide
convention). The bbox is the *input* and the Chinese text the *answer*, so this
is region->text — the inverse of grounding — and is emitted as ``understanding``
with variants ``cap`` / ``ocr``. The ocr prompt asks for the box *content*
(not strictly "text"): upstream `text` labels non-text elements too (~7% are
avatar / cover / icon-button descriptions, e.g. "歌曲封面"), so a text-only
prompt would mislabel them.

Source record format (both files)::

    {
      "task": "bbox2function" | "bbox2text",
      "image": "grounding_eval/dataset/images/<...>/N.jpeg",
      "id": <int>,
      "abs_position": "<x1, y1, x2, y2>",       # pixels
      "rel_position": "<x1, y1, x2, y2>",       # normalized [0, 1] floats
      "text": "<ground truth Chinese description>"
    }

Image-path quirk: the ``image`` field's prefix ``grounding_eval/dataset/images/``
does NOT match the on-disk layout. Real images live at::

  ${CUA_LITE_RAW_DATASETS_ROOT}/OpenBMB/CAGUI/CAGUI_grounding/images/cap/<N>.jpeg
  ${CUA_LITE_RAW_DATASETS_ROOT}/OpenBMB/CAGUI/CAGUI_grounding/images/ocr/<N>.jpeg

We use ``os.path.basename(record["image"])`` to derive the filename, NOT the
``id`` field — in ``ocr.jsonl`` the ``id`` field is unrelated to the filename
(e.g. row 1 has ``id=629`` but image basename ``1.jpeg``).

Resolution is recovered from ``abs_position`` / ``rel_position``
(``width = abs_x / rel_x``); the raw ``abs_position`` is then dropped.

Output: ${CUA_LITE_DATASETS_ROOT}/cua-lite/CAGUI/mobile/understanding/<split>/{cap,ocr}.parquet

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw_datasets
    export CUA_LITE_DATASETS_ROOT=/path/to/output
    uv run python -m lite.data.preproc.cagui.understanding \
        [--subset cap|ocr|all] [--head N] [--dry-run] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import MAX_NORM, clamp_norm
from lite.data import staging
from lite.data.preproc.cagui.utils import (
    make_image_store,
    make_splitter,
    out_dir_for,
    stage_entry,
)
from lite.utils.path import resolve_path

# CAGUI grounding bbox is given as a string like "<x1, y1, x2, y2>"
# (angle brackets, comma-space separated). Both abs_position and rel_position
# share this shape.
BBOX_RE = re.compile(
    r"<\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*>"
)

# Both source files only provide a (bbox, text) pair with no question prompt, so
# we synthesize a Chinese instruction embedding the bbox in [0, 1000] ints.
#   - cap (bbox2function): ask for the element's *function*.
#   - ocr (bbox2text): the upstream `text` is "the OCR string OR an element
#     description" — ~7% of rows label non-text elements (avatar / cover / icon
#     button). So the prompt asks for the box *content* generically rather than
#     restricting to literal text, which would mislabel those rows.
CAP_PROMPT = "请描述截图中位于边界框 [{x1}, {y1}, {x2}, {y2}] 内的 UI 元素的功能。"
OCR_PROMPT = "请识别截图中位于边界框 [{x1}, {y1}, {x2}, {y2}] 内显示的内容。"

SUBSETS: dict[str, dict[str, str]] = {
    "cap": {
        "jsonl": "OpenBMB/CAGUI/CAGUI_grounding/code/cap.jsonl",
        "images_rel_dir": "OpenBMB/CAGUI/CAGUI_grounding/images/cap",
        "task_field": "bbox2function",
        "prompt": CAP_PROMPT,
    },
    "ocr": {
        "jsonl": "OpenBMB/CAGUI/CAGUI_grounding/code/ocr.jsonl",
        "images_rel_dir": "OpenBMB/CAGUI/CAGUI_grounding/images/ocr",
        "task_field": "bbox2text",
        "prompt": OCR_PROMPT,
    },
}


class _SkipReason:
    MISSING_IMAGE = "missing_image"
    EMPTY_TEXT = "empty_text"


def _parse_bbox(text: str) -> tuple[float, float, float, float]:
    """Parse a CAGUI '<x1, y1, x2, y2>' string into a tuple of floats."""
    m = BBOX_RE.match(text)
    if not m:
        raise ValueError(f"Unparseable bbox: {text!r}")
    return tuple(float(v) for v in m.groups())  # type: ignore[return-value]


def parse_rel_bbox(text: str) -> tuple[int, int, int, int]:
    """Parse a rel_position string (floats in [0, 1]) -> ints in [0, 1000]."""
    coords = []
    for v in _parse_bbox(text):
        coords.append(clamp_norm(int(round(v * MAX_NORM))))
    return tuple(coords)  # type: ignore[return-value]


def recover_resolution(abs_position: str, rel_position: str) -> list[int] | None:
    """Recover [width, height] in pixels from abs/rel bbox pairs.

    ``abs = rel * dimension`` so ``dimension = abs / rel``. Use the edge with
    the larger normalized value for numerical stability; bail to ``None`` if
    both edges sit at the origin (rel ~ 0), which would divide by zero.
    """
    try:
        ax1, ay1, ax2, ay2 = _parse_bbox(abs_position)
        rx1, ry1, rx2, ry2 = _parse_bbox(rel_position)
    except ValueError:
        return None

    def _dim(a1: float, r1: float, a2: float, r2: float) -> int | None:
        a, r = (a2, r2) if abs(r2) >= abs(r1) else (a1, r1)
        if r <= 1e-6:
            return None
        return int(round(a / r))

    width = _dim(ax1, rx1, ax2, rx2)
    height = _dim(ay1, ry1, ay2, ry2)
    if width is None or height is None:
        return None
    return [width, height]


def record_to_example(
    record: dict,
    subset_name: str,
    subset_cfg: dict[str, str],
    record_idx: int,
) -> dict | str:
    """Convert one CAGUI grounding record to a CUA-Lite understanding example.

    Returns a ``_SkipReason.*`` constant string if the record should be
    skipped (image missing, or empty/missing GT text — both silent skips,
    mirroring the missing-image skip policy in AGENTS.md). Raises ``ValueError``
    on truly malformed records (unparseable bbox, wrong task field).
    """
    task = record.get("task")
    if task != subset_cfg["task_field"]:
        raise ValueError(
            f"Expected task={subset_cfg['task_field']!r}, got {task!r} at record {record_idx}"
        )

    rel_position = record.get("rel_position")
    if not isinstance(rel_position, str):
        raise ValueError(
            f"Missing/invalid rel_position at record {record_idx}: {rel_position!r}"
        )
    x1, y1, x2, y2 = parse_rel_bbox(rel_position)

    # Resolve image: use basename of `image` field, NOT the `id` field.
    image_field = record.get("image")
    if not isinstance(image_field, str) or not image_field:
        raise ValueError(f"Missing/invalid image field at record {record_idx}")
    basename = os.path.basename(image_field)
    img_rel = os.path.join(subset_cfg["images_rel_dir"], basename)
    try:
        img_abs = resolve_path(img_rel, "CUA_LITE_RAW_DATASETS_ROOT")
    except FileNotFoundError:
        return _SkipReason.MISSING_IMAGE

    user_text = subset_cfg["prompt"].format(x1=x1, y1=y1, x2=x2, y2=y2)

    # Assistant content is the source text verbatim. Empty/missing text is a
    # labeling anomaly (1 occurrence in ocr.jsonl); skip it like a missing image.
    assistant_text = record.get("text")
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        return _SkipReason.EMPTY_TEXT

    resolution = recover_resolution(
        record.get("abs_position", ""), rel_position
    )

    return {
        "images": [img_abs],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("mobile", "understanding"),
            others={
                "id": f"cagui_{subset_name}_{basename.split('.')[0]}",
                "resolution": resolution,
                "os": "android",
                "source": "OpenBMB/CAGUI",
                "source_id": f"{subset_name}/{basename}",
                "language": "zh",
            },
        ).to_dict(),
    }


def iter_examples(
    subset_name: str,
    raw_root: str,
    head: int | None,
    verbose: bool,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Read one subset's JSONL and return (rows, total_seen, n_img_skip, n_text_skip)."""
    cfg = SUBSETS[subset_name]
    jsonl_abs = os.path.join(raw_root, cfg["jsonl"])
    if not os.path.exists(jsonl_abs):
        raise FileNotFoundError(f"source jsonl not found: {jsonl_abs}")

    rows: list[dict[str, Any]] = []
    skipped_image = 0
    skipped_text = 0
    total_seen = 0

    with open(jsonl_abs, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if head is not None and total_seen >= head:
                break
            total_seen += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error at {jsonl_abs}:{line_no}") from e

            entry = record_to_example(record, subset_name, cfg, line_no)
            if isinstance(entry, str):
                if entry == _SkipReason.MISSING_IMAGE:
                    skipped_image += 1
                    if verbose:
                        print(f"  Skip line {line_no}: image not resolvable")
                elif entry == _SkipReason.EMPTY_TEXT:
                    skipped_text += 1
                    if verbose:
                        print(f"  Skip line {line_no}: empty/missing text")
                continue
            rows.append(entry)

    return rows, total_seen, skipped_image, skipped_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CAGUI grounding (cap/ocr) -> CUA-Lite understanding"
    )
    parser.add_argument(
        "--subset",
        choices=["cap", "ocr", "all"],
        default="all",
        help="Which subset(s) to process (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Process only first N records per subset (debug only)",
    )
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT is not set", file=sys.stderr)
        return 1
    if not os.getenv("CUA_LITE_DATASETS_ROOT") and not args.dry_run:
        print("Error: CUA_LITE_DATASETS_ROOT is not set", file=sys.stderr)
        return 1

    subsets = ["cap", "ocr"] if args.subset == "all" else [args.subset]

    out_dir = None if args.dry_run else out_dir_for()
    store = None if args.dry_run else make_image_store(out_dir)
    splitter = None if args.dry_run else make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    n_corrupt = 0

    for name in subsets:
        print(f"\n=== Processing subset {name!r} ===")
        rows, total_seen, n_img, n_text = iter_examples(
            name, raw_root, args.head, args.verbose
        )
        print(
            f"  Read {total_seen} records -> {len(rows)} converted, "
            f"{n_img} skipped (missing image), {n_text} skipped (empty text)"
        )
        if args.dry_run:
            if rows:
                s = rows[0]
                print("  [DRY RUN] first user text:")
                print(f"    {s['messages'][0]['content'][1]['text'][:160]}")
                print(f"  [DRY RUN] first assistant text: {s['messages'][1]['content'][0]['text'][:120]}")
                print(f"  [DRY RUN] first resolution: {s['metadata']['others']['resolution']}")
            continue
        for entry in rows:
            assert store is not None and splitter is not None
            try:
                bk, e = stage_entry(entry, store=store, splitter=splitter, variant=name)
            except staging.CorruptImageError:
                n_corrupt += 1
                continue
            buffers[bk].append(e)

    if args.dry_run:
        return 0

    assert out_dir is not None and store is not None
    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(f"Dropped {n_corrupt} rows with corrupt images")
    print(f"\nWrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    print(f"Image store: {store.count()} unique images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
