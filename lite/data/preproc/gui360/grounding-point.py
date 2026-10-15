"""GUI-360 Grounding Point Data Preprocessor.

Processes the ``grounding_resize`` subset of GUI-360 into the canonical
``grounding.point`` cohort:

    ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUI-360/desktop/grounding.point/<split>/<variant>.parquet

Each source record pairs an instruction (the agent's step ``thought``, e.g.
"To use the ATAN function ... I will enter =ATAN(1) into cell H4 ...") with the
pixel coordinate of the element to operate on. We emit it as a single
``point(coordinate)`` tool call with coordinates normalized to [0, 1000].

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount
    export CUA_LITE_DATASETS_ROOT=/path/to/canonical-output
    uv run python lite/data/preproc/gui360/grounding-point.py \
        [--dry-run] [--verbose] [--head N]

Source record (processed_data/grounding_resize/training_data.json):
    {
      "id": "excel_4s_1_2",
      "images": ["images\\\\excel_4s_1/action_step4.png"],
      "conversation": [
        {"from": "human", "value": "<image>\\n...The instruction is:\\n<TEXT>\\n\\nOutput the coordinate...<coordinate> [x, y] </coordinate>"},
        {"from": "gpt", "value": "<coordinate> [630, 241] </coordinate>"}
      ]
    }
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Any

from PIL import Image

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import make_tool_call, stamp_messages_tool_call_ids
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate
from lite.data.preproc.gui360.utils import (
    OS,
    PLATFORM,
    SOURCE,
    iter_json_array,
    make_image_store,
    make_splitter,
    normalize_coordinate,
    out_dir_for,
    stage_entry,
)
from lite.utils.path import resolve_path

VARIANT = "point"
SUBSET_REL = "vyokky/GUI-360/processed_data/grounding_resize"
SOURCE_ID = "grounding_resize"

_INSTRUCTION_RE = re.compile(
    r"The instruction is:\s*\n(.+?)\n\s*\nOutput the coordinate", re.DOTALL
)
# Coordinates may be negative or out of range (upstream annotation noise);
# parse them as-is and let has_oob_coordinate drop the offending rows.
_COORD_RE = re.compile(
    r"<coordinate>\s*\[\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\]\s*</coordinate>"
)


def extract_instruction(human_text: str, record_id: str) -> str:
    """Pull the instruction body out of the upstream human prompt."""
    match = _INSTRUCTION_RE.search(human_text)
    if not match:
        raise ValueError(
            f"Failed to extract instruction for grounding record {record_id}: "
            f"{human_text[:200]}"
        )
    instruction = match.group(1).strip()
    if not instruction:
        raise ValueError(f"Empty instruction for grounding record {record_id}")
    return instruction


def resolve_image(record: dict[str, Any], record_id: str) -> str:
    """Resolve the record's single image to an absolute path (backslash-safe)."""
    images = record.get("images") or []
    if not images:
        raise ValueError(f"No image field in grounding record {record_id}")
    # Upstream paths sometimes use a Windows backslash for the first component
    # (e.g. ``images\\excel_4s_1/action_step4.png``); normalize to POSIX.
    rel = images[0].replace("\\", "/")
    return resolve_path(f"{SUBSET_REL}/{rel}", "CUA_LITE_RAW_DATASETS_ROOT")


def process_records(json_path: str, *, head: int | None, verbose: bool):
    """Yield canonical grounding.point entries from the subset JSON."""
    n_seen = 0
    n_missing = 0
    for record in iter_json_array(json_path):
        if head is not None and n_seen >= head:
            break
        n_seen += 1
        record_id = record["id"]

        try:
            abs_image = resolve_image(record, record_id)
        except FileNotFoundError as e:
            n_missing += 1
            if verbose:
                print(f"  Skip (missing image) {record_id}: {e}")
            continue

        conv = record["conversation"]
        human_text = conv[0]["value"]
        gpt_text = conv[1]["value"]

        instruction = extract_instruction(human_text, record_id)

        coord_match = _COORD_RE.search(gpt_text)
        if not coord_match:
            raise ValueError(
                f"Failed to parse <coordinate> in grounding record {record_id}: {gpt_text[:200]}"
            )
        px, py = float(coord_match.group(1)), float(coord_match.group(2))

        with Image.open(abs_image) as image:
            width, height = image.size
        coordinate = normalize_coordinate(px, py, width, height)

        yield {
            "images": [abs_image],
            "messages": stamp_messages_tool_call_ids([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 0},
                        {"type": "text", "text": instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "tool_calls": [make_tool_call("point", {"coordinate": coordinate})],
                },
            ], preserve=True),
            "metadata": LiteCUAMetadata(
                dims=(PLATFORM, "grounding.point"),
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

    # Unconditional: a zero here is the evidence that nothing was dropped, and
    # its absence used to be indistinguishable from nobody having counted.
    print(f"  Skipped {n_missing} record(s) with missing image")


def main():
    parser = argparse.ArgumentParser(
        description="Process GUI-360 point grounding data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan, write nothing")
    parser.add_argument("--verbose", action="store_true", help="Per-record skip logging")
    parser.add_argument("--head", type=int, default=None, help="Process at most N records (smoke test)")
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
        print(f"=== DRY RUN ===\nWould read {json_path}\nWould write desktop/grounding.point/<split>/<variant>.parquet")
        return 0

    if not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT must be set")
        return 1

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    n_oob = n_corrupt = 0

    for entry in process_records(json_path, head=args.head, verbose=args.verbose):
        if has_oob_coordinate(entry):
            n_oob += 1
            continue
        try:
            bk, e = stage_entry(entry, store=store, splitter=splitter, variant=VARIANT)
        except staging.CorruptImageError:
            n_corrupt += 1
            continue
        buffers[bk].append(e)

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(f"Dropped {n_oob} rows with out-of-bound coordinates")
    print(f"Dropped {n_corrupt} rows with corrupt images")
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
