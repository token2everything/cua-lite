"""GUIAct web-single Grounding Action preprocessor.

Reads ``${CUA_LITE_RAW_DATASETS_ROOT}/yiye2023/GUIAct/web-single_{train,test}_data.json``
and writes SFT-ready parquets to the canonical layout::

    ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIAct/browser/grounding.action/{train,validation}.parquet

web-single is structurally single-screenshot: each record has one image, one
instruction (``question``), one short human rationale (``thoughts``), and a
list of atomic UI actions to execute on that screenshot. There is no follow-up
screenshot and no multi-step feedback loop, so it maps to ``grounding.action``,
not ``use``. The upstream test split is honored verbatim as
``validation`` (see :func:`lite.data.preproc.guiact.utils.make_splitter`).

Image bytes are deduplicated into the dataset's content-addressed image store;
the row's ``images`` column carries paths relative to ``$CUA_LITE_DATASETS_ROOT``.

Usage::

    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw
    export CUA_LITE_DATASETS_ROOT=/path/to/processed
    # images must be extracted to disk first via scripts/process_raw_data.sh
    uv run python lite/data/preproc/guiact/grounding-action.py [--dry-run] [--verbose] [--head N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.messages import make_assistant_content
from lite.core.tools.calls import stamp_messages_tool_call_ids
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate
from lite.data.preproc.guiact.utils import (
    SkipTrajectoryError,
    convert_web_actions,
    make_image_store,
    make_splitter,
    out_dir_for,
    stage_entry,
)
from lite.data.utils.messages import extra_tool_schemas_for_messages
from lite.utils.path import resolve_path

VARIANT = "action"  # cohort variant; always its own folder

# (source filename, upstream split label) pairs. The label is consumed by the
# splitter's canonical_fn and stripped before persistence.
SOURCE_FILES = [
    ("yiye2023/GUIAct/web-single_train_data.json", "train"),
    ("yiye2023/GUIAct/web-single_test_data.json", "test"),
]


def record_to_example(record: dict[str, Any], *, split: str) -> dict[str, Any]:
    """Convert a web-single record into a ``grounding.action`` row.

    The assistant turn carries the human rationale (``thoughts``) as an
    ``inline_reasoning`` content part plus the action ``tool_calls``. There is
    no ``terminate`` — ``grounding.action`` has no end-of-episode concept, so a
    standalone ``answer`` maps to ``response(text)`` (``is_terminal=False``).
    """
    image_id = record["image_id"]
    image_rel = f"yiye2023/GUIAct/images/web-single/{image_id}.png"
    try:
        abs_image = resolve_path(image_rel, "CUA_LITE_RAW_DATASETS_ROOT")
    except FileNotFoundError:
        raise SkipTrajectoryError(f"Image not found: {image_rel}")

    actions_label = record["actions_label"]
    if not actions_label:
        raise SkipTrajectoryError(f"Empty actions_label for uid={record['uid']}")

    tool_calls = convert_web_actions(actions_label, is_terminal=False)

    thoughts = (record.get("thoughts") or "").strip()
    content = make_assistant_content(inline_reasoning=thoughts) if thoughts else []
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": tool_calls,
    }
    if content:
        assistant_msg["content"] = content

    messages: list[dict[str, Any]] = stamp_messages_tool_call_ids([
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": record["question"]},
            ],
        },
        assistant_msg,
    ], preserve=True)

    img_size = record.get("image_size") or {}
    resolution = [img_size["width"], img_size["height"]] if img_size else None

    return {
        "images": [abs_image],
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=("browser", "grounding.action"),
            extra_tool_schemas=extra_tool_schemas_for_messages(messages),
            valid_actions=None,
            others={
                "id": f"guiact_web_single_{record['uid']}",
                "resolution": resolution,
                "os": None,
                "source": "yiye2023/GUIAct",
                "source_id": record["uid"],
                "split": split,  # transient routing hint; stage_entry strips it
            },
        ).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process GUIAct web-single grounding.action data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true", help="Process without writing output")
    parser.add_argument("--verbose", action="store_true", help="Show skip details")
    parser.add_argument("--head", type=int, default=None, help="Process only first N records per file")
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT must be set")
        return 1
    if not args.dry_run and not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT must be set")
        return 1
    raw_root = Path(raw_root)

    out_dir = None if args.dry_run else out_dir_for()
    store = None if args.dry_run else make_image_store(out_dir)
    splitter = None if args.dry_run else make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    n_in = n_skipped = n_oob = n_corrupt = 0

    for rel_json, split in SOURCE_FILES:
        data_path = raw_root / rel_json
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        with open(data_path) as f:
            data = json.load(f)
        if args.head is not None:
            data = data[: args.head]
        print(f"\n{rel_json} (split={split}): {len(data)} records")
        n_in += len(data)

        for i, record in enumerate(data):
            try:
                entry = record_to_example(record, split=split)
            except SkipTrajectoryError as e:
                n_skipped += 1
                if args.verbose:
                    print(f"  Skipping record {i} (uid={record.get('uid', '?')}): {e}")
                continue

            if has_oob_coordinate(entry):
                n_oob += 1
                continue

            if args.dry_run:
                continue

            try:
                bk, e = stage_entry(entry, store=store, splitter=splitter, variant=VARIANT)
            except staging.CorruptImageError as img_err:
                n_corrupt += 1
                if args.verbose:
                    print(f"  Skipping corrupt record {i}: {img_err}")
                continue
            buffers[bk].append(e)

            if (i + 1) % 10000 == 0:
                print(f"  Progress: {i + 1}/{len(data)}")

    if args.dry_run:
        print(f"\n[dry-run] rows_in={n_in}, skipped={n_skipped}, oob={n_oob}")
        return 0

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(
        f"\nrows_in={n_in}, skipped={n_skipped}, oob_dropped={n_oob}, "
        f"corrupt_dropped={n_corrupt}"
    )
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
