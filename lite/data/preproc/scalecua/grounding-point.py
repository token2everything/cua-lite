"""
ScaleCUA Grounding Point Data Preprocessor

This script processes point grounding task data from ScaleCUA-Data dataset
and converts it to the standardized format for CUA-Lite training.

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/datasets
    uv run python lite/data/preproc/scalecua/grounding-point.py \
        [--dry-run] [--verbose] [--head N] [--head-entries N]

Requirements:
    - CUA_LITE_RAW_DATASETS_ROOT environment variable must be set
    - Source data should be in: ${CUA_LITE_RAW_DATASETS_ROOT}/OpenGVLab/ScaleCUA-Data/
    - Output: ${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA/{platform}/grounding.point/{split}/{variant}.parquet

Source data format:
    {
        "image": "path/to/image.png",
        "conversations": [
            {"from": "human", "value": "<image>\\nLocate the point contained in: <ref>description</ref>"},
            {"from": "gpt", "value": "<ref>description</ref><point>[[x, y]]</point>"}
        ],
        "width": 1920,
        "height": 1080
    }

Output format (see AGENTS.md for full schema):
    {
        "images": ["relative/path/to/image.jpg"],
        "messages": [...],
        "metadata": {
            "metadata_kind": "cua",
            "dims": ["...", "grounding.point"],
            "extra_tool_schemas": [],
            "valid_actions": null,
            "others": {"id": "...", "resolution": [...], ...}
        }
    }

Run accounting (printed unconditionally, no --verbose needed), at both levels
this adapter can lose data:

    Entries: meta entries with ``conv_style == "internvl_grounding"`` = selected
        + one named ``EXCLUDED (<reason>)`` line per entry the key-keyed platform
        lookup cannot place.
    Records read: R  kept: W  skipped: S   with   R == W + S
    Skip reasons: {<reason>: n, ...}
    plus one ``DROPPED ALL`` line per entry that read records and wrote no row.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import make_tool_call, stamp_messages_tool_call_ids
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate, prestage_images
from lite.data.preproc.scalecua.utils import (
    make_image_store,
    make_splitter,
    out_dir_for,
    stage_entry,
)
from lite.utils.path import resolve_path

VARIANT = "point"


def get_platform_type(key: str) -> str:
    """Determine platform type from key name."""
    key_lower = key.lower()
    if "iphone" in key_lower or "android" in key_lower:
        return "mobile"
    elif "windows" in key_lower or "mac" in key_lower or "ubuntu" in key_lower:
        return "desktop"
    elif "web" in key_lower:
        return "browser"
    else:
        raise ValueError(f"Cannot determine platform type from key: {key}")


def get_os_from_key(key: str) -> str:
    """Extract OS from key name."""
    key_lower = key.lower()
    if "windows" in key_lower:
        return "windows"
    elif "mac" in key_lower or "macos" in key_lower:
        return "macos"
    elif "ubuntu" in key_lower:
        return "ubuntu"
    elif "android" in key_lower:
        return "android"
    elif "iphone" in key_lower or "ios" in key_lower:
        return "ios"
    elif "web" in key_lower:
        return None
    else:
        return "unknown"


def clean_image_token(text: str) -> str:
    """Remove <image> token from text."""
    # Remove standalone <image>\n or <image> at the start
    text = re.sub(r"^<image>\n?", "", text)
    # Remove Image-N: <image>\n patterns
    text = re.sub(r"Image-\d+:\s*<image>\n?", "", text)
    return text.strip()


def extract_ref_description(text: str) -> str | None:
    """Extract the description from <ref>...</ref> tags."""
    match = re.search(r"<ref>(.+?)</ref>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def is_point_format(response_text: str) -> bool:
    """Check if the response is in point format (vs bbox format)."""
    return "<point>" in response_text


def parse_point_response(
    response_text: str,
    key: str,
    line_num: int,
) -> tuple[str, list[int]]:
    """
    Parse point from assistant response.

    Source format: <ref>description</ref><point>[[x, y]]</point>

    The InternVL `<point>` grammar is ALREADY in the canonical [0, 1000] range, so
    the integers are copied through verbatim. This function deliberately takes no
    `width`/`height`: there is no pixel normalization to do, and dividing by the
    image size would destroy every label. See the coordinate-system table in
    `AGENTS.md`.

    Args:
        response_text: The raw response text from source data
        key: Source key name for error messages
        line_num: Line number for error messages

    Returns:
        Tuple of (description, point)
        Point is [x, y], already in [0, 1000] range

    Raises:
        ValueError: If point format is detected but parsing fails
    """
    # Extract description from <ref>...</ref>
    ref_match = re.search(r"<ref>(.+?)</ref>", response_text, re.DOTALL)
    if not ref_match:
        raise ValueError(
            f"Failed to parse <ref> in point response at {key} line {line_num}: "
            f"{response_text[:200]}"
        )
    description = ref_match.group(1).strip()

    # Extract point from <point>[[x, y]]</point>
    point_match = re.search(r"<point>\[\[(\d+),\s*(\d+)\]\]</point>", response_text)
    if not point_match:
        raise ValueError(
            f"Failed to parse <point> coordinates in point response at {key} line {line_num}: "
            f"{response_text[:200]}"
        )

    # InternVL format: coordinates are already in [0, 1000] normalized range
    x_norm = int(point_match.group(1))
    y_norm = int(point_match.group(2))

    return description, [x_norm, y_norm]


def simplify_user_prompt(text: str) -> str:
    """
    Simplify user prompt by extracting the description from <ref> tags.

    Various source formats:
    - "Locate the point contained in: <ref>description</ref>"
    - "Query:<ref>description</ref>\\nOutput only the point in your response:"
    - "Output the point lying within: <ref>description</ref>"

    Output format:
    - "Locate the element: description"
    """
    description = extract_ref_description(text)
    if description:
        return f"Locate the element: {description}"
    # If no <ref> tag found, return cleaned text
    return text


def process_point_grounding_entry(
    key: str,
    value: dict[str, Any],
    base_dir: str,
    base_datasets_dir: str,
    platform_type: str,
    verbose: bool = False,
    head: int | None = None,
) -> tuple[list[dict[str, Any]], int, Counter[str]]:
    """
    Process a single point grounding entry from meta.json.

    Args:
        key: The metadata key name
        value: The metadata value dict
        base_dir: Absolute path to the source data directory
        base_datasets_dir: The CUA_LITE_RAW_DATASETS_ROOT path (for computing relative paths)
        platform_type: Platform type ("mobile", "desktop", or "browser")
        verbose: Whether to print verbose output
        head: Stop after reading this many annotation records (smoke test)

    Returns:
        ``(rows, n_records, skips)`` — the accounting travels WITH the rows so a
        caller cannot report the rows without it. One source record maps to at
        most one row, so the ledger closes exactly::

            n_records == len(rows) + sum(skips.values())

        and the function itself guarantees that by folding any remainder into
        ``skips["unaccounted"]``, so a ``continue`` added later without its own
        counter surfaces by name instead of vanishing.

        ``missing_image`` (the file is gone) is kept apart from
        ``missing_image_root_absent`` (the entry's whole image root is not on
        this host — an unextracted ``.tar.gz`` shard), because the two say
        different things about whether the adapter or the host lost the record.

        ``bbox_format`` is this adapter declining its sibling's format, which is
        the expected way the shared ``internvl_grounding`` pool is split.

    Raises:
        FileNotFoundError: If annotation file is not found
        ValueError: If data format is invalid (not including expected bbox format entries)
    """
    results = []
    skips: Counter[str] = Counter()
    n_records = 0

    # Get paths
    annotation_path = os.path.join(base_dir, value["annotation"])
    # Compute the relative path prefix from CUA_LITE_RAW_DATASETS_ROOT
    relative_root = os.path.relpath(os.path.join(base_dir, value["root"]), base_datasets_dir)
    root_present = os.path.isdir(os.path.join(base_datasets_dir, relative_root))
    missing_image_reason = "missing_image" if root_present else "missing_image_root_absent"

    # Check if annotation file exists
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    # Read annotation file
    with open(annotation_path, "r") as f:
        for line_num, line in enumerate(f):
            if head is not None and line_num >= head:
                break
            if not line.strip():
                continue
            n_records += 1

            data = json.loads(line)

            # Get image path
            image_field = data.get("image") or data.get("image_path")
            if not image_field:
                raise ValueError(f"No image field found in {key} line {line_num}")

            # Resolve to an absolute path; skip if missing.
            rel_image = os.path.join(relative_root, image_field)
            try:
                abs_image = resolve_path(rel_image, "CUA_LITE_RAW_DATASETS_ROOT")
            except FileNotFoundError as e:
                skips[missing_image_reason] += 1
                if verbose and root_present:
                    print(f"  Warning: Skipping record (missing image) line {line_num}: {e}")
                continue
            images = [abs_image]

            # Image size, published as others.resolution. NOT used to normalize
            # the <point> coordinates -- those are already [0, 1000].
            width = data.get("width")
            height = data.get("height")
            if not width or not height:
                raise ValueError(f"Missing resolution in {key} line {line_num}")
            resolution = [width, height]

            # Parse conversations
            conversations = data.get("conversations", [])
            if len(conversations) < 2:
                raise ValueError(f"Insufficient conversations in {key} line {line_num}")

            # Process assistant message first to check format type
            assistant_conv = conversations[1]
            if assistant_conv.get("from") not in ["gpt", "assistant"]:
                raise ValueError(
                    f"Second conversation is not from assistant in {key} line {line_num}"
                )

            assistant_value = assistant_conv.get("value", "")

            # Check if this is point format (vs bbox format)
            # Bbox format entries are expected and should be skipped silently
            if not is_point_format(assistant_value):
                skips["bbox_format"] += 1
                continue

            # Build messages
            messages = []

            # Process user message (first conversation)
            user_conv = conversations[0]
            if user_conv.get("from") not in ["human", "user"]:
                raise ValueError(f"First conversation is not from user in {key} line {line_num}")

            user_text = clean_image_token(user_conv.get("value", ""))
            user_text = simplify_user_prompt(user_text)

            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image", "index": 0}, {"type": "text", "text": user_text}],
                }
            )

            # Parse point from assistant response (raises ValueError on parse failure)
            description, point = parse_point_response(assistant_value, key, line_num)

            tool_calls = [make_tool_call("point", {"coordinate": point})]

            messages.append({"role": "assistant", "tool_calls": tool_calls})
            messages = stamp_messages_tool_call_ids(messages, preserve=True)

            entry = {
                "images": images,
                "messages": messages,
                "metadata": LiteCUAMetadata(
                    dims=(platform_type, "grounding.point"),
                    extra_tool_schemas=[],
                    valid_actions=None,
                    others={
                        "id": f"{key}_{line_num}",
                        "resolution": resolution,
                        "os": get_os_from_key(key),
                        "source": "OpenGVLab/ScaleCUA-Data",
                        "source_id": key,
                    },
                ).to_dict(),
            }
            results.append(entry)

    # Close the ledger. Every branch above either appends a row or names a
    # reason; a future one that does neither shows up here by name.
    unaccounted = n_records - len(results) - sum(skips.values())
    if unaccounted:
        skips["unaccounted"] += unaccounted

    return results, n_records, skips


def main():
    parser = argparse.ArgumentParser(
        description="Process ScaleCUA point grounding data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing data",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress information")
    parser.add_argument(
        "--head-entries",
        type=int,
        default=None,
        help="Process at most this many meta entries per platform (smoke test)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Process at most N records per annotation file (smoke test)",
    )
    args = parser.parse_args()

    # Get base directory from environment
    CUA_LITE_RAW_DATASETS_ROOT = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not CUA_LITE_RAW_DATASETS_ROOT:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT environment variable must be set")
        print("Please set it to the directory containing OpenGVLab/ScaleCUA-Data")
        print("\nExample:")
        print("  export CUA_LITE_RAW_DATASETS_ROOT=/path/to/datasets")
        print("  uv run python lite/data/preproc/scalecua/grounding-point.py")
        return 1

    # According to AGENTS.md:
    # Grounding data comes from: ${CUA_LITE_RAW_DATASETS_ROOT}/OpenGVLab/ScaleCUA-Data/
    base_source_dir = os.path.join(CUA_LITE_RAW_DATASETS_ROOT, "OpenGVLab", "ScaleCUA-Data")

    if not os.path.exists(base_source_dir):
        print(f"Error: Source directory not found: {base_source_dir}")
        print("Please ensure the ScaleCUA-Data dataset is available")
        return 1

    # Read meta.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(script_dir, "meta.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Filter entries for point grounding
    # Selection criteria: conv_style == "internvl_grounding" (exact match), then
    # the entry KEY must name a platform.
    # Note: The same conv_style is used for both bbox and point data
    # We filter by response format (<point> vs <box>) during processing.
    # The platform half is a real exclusion of whole entries, so it is counted
    # and named rather than skipped silently -- see "Entry selection" in
    # AGENTS.md for why the excluded entry stays out.
    point_grounding_entries: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    n_entries_in_pool = 0
    excluded_entries: dict[str, list[str]] = defaultdict(list)
    for key, value in meta.items():
        conv_style = value.get("conv_style", "")
        if conv_style != "internvl_grounding":
            continue
        n_entries_in_pool += 1
        try:
            platform_type = get_platform_type(key)
        except ValueError:
            excluded_entries["unknown_platform"].append(key)
            continue
        point_grounding_entries[platform_type].append((key, value))

    n_excluded = sum(len(keys) for keys in excluded_entries.values())
    print(
        f"internvl_grounding entries: {n_entries_in_pool}  "
        f"selected: {n_entries_in_pool - n_excluded}  excluded: {n_excluded}"
    )
    for reason, keys in sorted(excluded_entries.items()):
        for key in sorted(keys):
            print(f"  EXCLUDED ({reason}): {key}")

    if args.head_entries is not None:
        point_grounding_entries = {
            p: e[: args.head_entries] for p, e in point_grounding_entries.items()
        }
    print("Found internvl_grounding entries by platform type (will filter for point format):")
    for platform_type, entries in point_grounding_entries.items():
        print(f"  {platform_type}: {len(entries)} entries")

    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("No files will be written. Use without --dry-run to process data.")
        for platform_type, entries in point_grounding_entries.items():
            print(f"\n{platform_type}:")
            for key, value in entries:
                print(f"  - {key}: {value['annotation']}")
        return 0

    if not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT environment variable must be set")
        print("Please set it to the output directory for processed datasets")
        return 1

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    skips: Counter[str] = Counter()
    n_records_total = 0
    total_loss_entries: list[str] = []

    for platform_type, entries in point_grounding_entries.items():
        for key, value in entries:
            if args.verbose:
                print(f"Processing {key}...")
            results, n_records, entry_skips = process_point_grounding_entry(
                key,
                value,
                base_source_dir,
                CUA_LITE_RAW_DATASETS_ROOT,
                platform_type,
                verbose=args.verbose,
                head=args.head,
            )
            n_records_total += n_records
            skips.update(entry_skips)
            n_kept = 0
            kept = []
            for entry in results:
                if has_oob_coordinate(entry):
                    skips["oob_coordinate"] += 1
                    continue
                kept.append(entry)
            corrupt_paths = prestage_images(
                store, (path for entry in kept for path in entry["images"])
            )
            for entry in kept:
                if any(Path(path) in corrupt_paths for path in entry["images"]):
                    skips["image_corrupt_on_host"] += 1
                    entry_skips["image_corrupt_on_host"] += 1
                    continue
                bk, e = stage_entry(entry, store=store, splitter=splitter, variant=VARIANT)
                buffers[bk].append(e)
                n_kept += 1
            # A 100%-drop entry is what a broken adapter path looks like, and it
            # is also legitimate here (an unextracted image shard, or an entry
            # whose records are all the sibling's format). So it gets its own
            # unmissable line rather than an exit code: the failure mode is
            # silence, and a non-zero exit would fire on every legitimately-empty
            # --head window on this host.
            if n_records and not n_kept:
                total_loss_entries.append(key)
                print(
                    f"  DROPPED ALL: {platform_type}/{key}: {n_records} records -> 0 rows "
                    f"({dict(entry_skips)})"
                )
            elif args.verbose:
                print(f"  Processed {n_kept} point rows from {n_records} records")

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(
        f"Records read: {n_records_total}  kept: {n_rows}  "
        f"skipped: {sum(skips.values())}  "
        f"(entries in pool: {n_entries_in_pool}, excluded: {n_excluded}, "
        f"all-dropped: {len(total_loss_entries)})"
    )
    if skips:
        print("Skip reasons:", dict(skips))
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
