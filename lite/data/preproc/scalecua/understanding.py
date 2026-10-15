"""
ScaleCUA Understanding Data Preprocessor

This script processes understanding task data from ScaleCUA-Data-Understanding dataset
and converts it to the standardized format for CUA-Lite training.

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/datasets
    uv run python lite/data/preproc/scalecua/understanding.py \
        [--dry-run] [--verbose] [--head N] [--head-entries N]

Requirements:
    - CUA_LITE_RAW_DATASETS_ROOT environment variable must be set
    - Source data should be in: ${CUA_LITE_RAW_DATASETS_ROOT}/zyliu/ScaleCUA-Data-Understanding/
    - Output: ${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA/{platform}/understanding/{split}/{variant}.parquet

Output format (see AGENTS.md for full schema):
    {
        "images": ["relative/path/to/image.jpg"],
        "messages": [...],
        "metadata": {
            "metadata_kind": "cua",
            "dims": ["...", "understanding"],
            "extra_tool_schemas": [],
            "valid_actions": null,
            "others": {"id": "...", "resolution": [...], ...}
        }
    }

Run accounting (printed unconditionally, no --verbose needed):
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
from lite.data import staging
from lite.data.preproc.common import prestage_images
from lite.data.preproc.scalecua.utils import (
    make_image_store,
    make_splitter,
    out_dir_for,
    stage_entry,
)
from lite.utils.path import resolve_path


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


def get_task_type(key: str) -> str:
    """Determine task type from key name."""
    key_lower = key.lower()
    if "user_intention" in key_lower:
        return "user_intention"
    elif "dense_caption" in key_lower or "caption" in key_lower:
        return "caption"
    elif "screen_transition" in key_lower:
        return "screen_transition"
    else:
        raise ValueError(f"Cannot determine task type from key: {key}")


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


def normalize_action_format_desktop(text: str) -> str:
    """
    Convert action format from source data to standardized desktop format.

    Desktop Source format examples:
        - click(x=0.5365, y=0.4427)
        - scroll(x=0.5, y=0.5, clicks=-4)  (negative = up, positive = down)
        - drag(x1=..., y1=..., x2=..., y2=...)
        - type(text='hello')
        - key(keys=['ctrl', 'c'])

    Desktop Target format (coordinates normalized to [0, 1000]):
        - click(coordinate=[536, 442])
        - scroll(coordinate=[500, 500], direction="up")
        - drag(start_coordinate=[...], coordinate=[...])
        - type(text="hello")
        - key(keys=["ctrl", "c"])
    """

    def convert_click(match):
        x = float(match.group(1))
        y = float(match.group(2))
        # Normalize to [0, 1000]
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)
        return f"click(coordinate=[{x_norm}, {y_norm}])"

    def convert_scroll_with_direction(match):
        x = float(match.group(1))
        y = float(match.group(2))
        direction = match.group(3)
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)
        return f'scroll(coordinate=[{x_norm}, {y_norm}], direction="{direction}")'

    def convert_scroll_with_clicks(match):
        x = float(match.group(1))
        y = float(match.group(2))
        clicks = int(match.group(3))
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)
        # ScaleCUA desktop convention: negative clicks = scroll up, positive = down
        direction = "up" if clicks < 0 else "down"
        return f'scroll(coordinate=[{x_norm}, {y_norm}], direction="{direction}")'

    def convert_drag(match):
        x1 = float(match.group(1))
        y1 = float(match.group(2))
        x2 = float(match.group(3))
        y2 = float(match.group(4))
        x1_norm = int(x1 * 1000)
        y1_norm = int(y1 * 1000)
        x2_norm = int(x2 * 1000)
        y2_norm = int(y2 * 1000)
        return f"drag(start_coordinate=[{x1_norm}, {y1_norm}], coordinate=[{x2_norm}, {y2_norm}])"

    # Convert click(x=..., y=...) to click(coordinate=[...])
    text = re.sub(r"click\(x=([0-9.]+),\s*y=([0-9.]+)\)", convert_click, text)

    # Convert scroll(x=..., y=..., direction=...) to scroll(coordinate=[...], direction="...")
    text = re.sub(
        r"scroll\(x=([0-9.]+),\s*y=([0-9.]+),\s*direction=(\w+)\)",
        convert_scroll_with_direction,
        text,
    )

    # Convert scroll(x=..., y=..., clicks=...) to scroll(coordinate=[...], direction="...")
    text = re.sub(
        r"scroll\(x=([0-9.]+),\s*y=([0-9.]+),\s*clicks=(-?[0-9]+)\)",
        convert_scroll_with_clicks,
        text,
    )

    # Convert drag(x1=..., y1=..., x2=..., y2=...) to drag(start_coordinate=[...], coordinate=[...])
    text = re.sub(
        r"drag\(x1=([0-9.]+),\s*y1=([0-9.]+),\s*x2=([0-9.]+),\s*y2=([0-9.]+)\)", convert_drag, text
    )

    # Normalize string quotes: 'text' -> "text"
    text = re.sub(r"=\'([^\']*?)\'", r'="\1"', text)

    return text


def normalize_action_format_mobile(text: str) -> str:
    """
    Convert action format from source data to standardized mobile format.

    Mobile Source format examples:
        - click(x=0.0656, y=0.0840)
        - swipe(from_coord=[0.5, 0.75], to_coord=[0.5, 0.25])
        - type(text='hello')

    Mobile Target format (coordinates normalized to [0, 1000]):
        - tap(coordinate=[65, 84])
        - swipe(start_coordinate=[500, 750], coordinate=[500, 250])
        - type(text="hello")
    """

    def convert_click_to_tap(match):
        x = float(match.group(1))
        y = float(match.group(2))
        # Normalize to [0, 1000]
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)
        return f"tap(coordinate=[{x_norm}, {y_norm}])"

    def convert_swipe(match):
        # Parse from_coord and to_coord arrays
        from_x = float(match.group(1))
        from_y = float(match.group(2))
        to_x = float(match.group(3))
        to_y = float(match.group(4))
        # Normalize to [0, 1000]
        from_x_norm = int(from_x * 1000)
        from_y_norm = int(from_y * 1000)
        to_x_norm = int(to_x * 1000)
        to_y_norm = int(to_y * 1000)
        return f"swipe(start_coordinate=[{from_x_norm}, {from_y_norm}], coordinate=[{to_x_norm}, {to_y_norm}])"

    # Convert click(x=..., y=...) to tap(coordinate=[...])
    text = re.sub(r"click\(x=([0-9.]+),\s*y=([0-9.]+)\)", convert_click_to_tap, text)

    # Convert swipe(from_coord=[...], to_coord=[...]) to swipe(start_coordinate=[...], coordinate=[...])
    text = re.sub(
        r"swipe\(from_coord=\[([0-9.]+),\s*([0-9.]+)\],\s*to_coord=\[([0-9.]+),\s*([0-9.]+)\]\)",
        convert_swipe,
        text,
    )

    # Normalize string quotes: 'text' -> "text"
    text = re.sub(r"=\'([^\']*?)\'", r'="\1"', text)

    return text


def normalize_action_format(text: str, platform_type: str) -> str:
    """
    Convert action format from source data to standardized format based on platform type.

    Args:
        text: The text containing action definitions
        platform_type: Either "mobile", "desktop", or "browser"

    Returns:
        Text with normalized action format
    """
    if platform_type == "mobile":
        return normalize_action_format_mobile(text)
    else:
        # Desktop and browser use the same format (desktop-style actions)
        return normalize_action_format_desktop(text)


def clean_image_token(text: str) -> str:
    """Remove <image> token from text."""
    # Remove standalone <image>\n or <image> at the start
    text = re.sub(r"^<image>\n?", "", text)
    # Remove Image-N: <image>\n patterns
    text = re.sub(r"Image-\d+:\s*<image>\n?", "", text)
    return text.strip()


def process_understanding_entry(
    key: str,
    value: dict[str, Any],
    base_dir: str,
    base_datasets_dir: str,
    platform_type: str,
    task_type: str,
    verbose: bool = False,
    head: int | None = None,
) -> tuple[list[dict[str, Any]], int, Counter[str]]:
    """Process a single understanding entry from meta.json.

    Args:
        key: The metadata key name
        value: The metadata value dict
        base_dir: Absolute path to the source data directory (e.g., ${CUA_LITE_RAW_DATASETS_ROOT}/zyliu/ScaleCUA-Data-Understanding)
        base_datasets_dir: The CUA_LITE_RAW_DATASETS_ROOT path (for computing relative paths)
        platform_type: Platform type ("mobile", "desktop", or "browser")
        task_type: Task type ("user_intention", "caption", or "screen_transition")
        verbose: Whether to print verbose output
        head: Stop after reading this many annotation records (smoke test)

    Returns:
        ``(rows, n_records, skips)`` — the accounting travels WITH the rows so a
        caller cannot report the rows without it. One source record maps to at
        most one row, so the ledger closes exactly::

            n_records == len(rows) + sum(skips.values())

        and the function itself guarantees that by folding any remainder into
        ``skips["unaccounted"]``. A ``continue`` added later without its own
        counter therefore surfaces as ``unaccounted=N`` instead of vanishing —
        which is the whole reason the 360 dropped mobile records were invisible.

        ``missing_image`` (the file is gone) is kept apart from
        ``missing_image_root_absent`` (the entry's whole image root is not on
        this host — an unextracted ``.tar.gz`` shard), because the two say
        different things about whether the adapter or the host lost the record.
    """
    results = []
    skips: Counter[str] = Counter()
    n_records = 0

    # Get paths
    annotation_path = os.path.join(base_dir, value["annotation"])
    # Compute the relative path prefix from CUA_LITE_RAW_DATASETS_ROOT, then normalize to OpenGVLab/ScaleCUA-Data
    relative_root = os.path.relpath(os.path.join(base_dir, value["root"]), base_datasets_dir)
    if relative_root.startswith("zyliu/ScaleCUA-Data-Understanding"):
        relative_root = (
            "OpenGVLab/ScaleCUA-Data/" + relative_root[len("zyliu/ScaleCUA-Data-Understanding/") :]
        )
    root_present = os.path.isdir(os.path.join(base_datasets_dir, relative_root))
    missing_image_reason = "missing_image" if root_present else "missing_image_root_absent"

    # Check if annotation file exists
    if not os.path.exists(annotation_path):
        if verbose:
            print(f"Warning: Annotation file not found: {annotation_path}")
        # Nothing is added to the ledger: zero records were read, so there is no
        # record-level loss to attribute. main() reports it as an entry that
        # read no records, which is what an absent annotation file IS.
        return results, n_records, skips

    # Read annotation file
    with open(annotation_path, "r") as f:
        for line_num, line in enumerate(f):
            if head is not None and line_num >= head:
                break
            if not line.strip():
                continue
            n_records += 1
            try:
                data = json.loads(line)

                # Convert to required format
                # Assuming the source data has 'image', 'conversations' or similar fields
                # Adjust based on actual data format

                # Get image path(s) - try different field names
                # Each field could be a string or a list
                # Images are stored as relative paths from ${CUA_LITE_RAW_DATASETS_ROOT}
                images = None
                for field_name in ["image", "image_path", "images"]:
                    if field_name in data:
                        field_value = data[field_name]
                        # Handle both string and list
                        # Use os.path.join to construct relative paths
                        if isinstance(field_value, str):
                            images = [os.path.join(relative_root, field_value)]
                        elif isinstance(field_value, list):
                            images = [os.path.join(relative_root, img) for img in field_value]
                        break

                if images is None:
                    raise ValueError(
                        f"Missing or invalid image field in {annotation_path} "
                        f"line {line_num + 1}"
                    )

                # Resolve to absolute paths; skip if any image is missing.
                try:
                    images = [resolve_path(p, "CUA_LITE_RAW_DATASETS_ROOT") for p in images]
                except FileNotFoundError as e:
                    skips[missing_image_reason] += 1
                    if verbose and root_present:
                        print(f"Warning: Skipping record with missing image in {key}: {e}")
                    continue

                # Build messages from conversations
                messages = []
                if "conversations" in data:
                    for i, conv in enumerate(data["conversations"]):
                        role = conv.get("from", "")
                        value_text = conv.get("value", "")

                        # Map role names
                        if role in ["human", "user"]:
                            role = "user"
                        elif role in ["gpt", "assistant"]:
                            role = "assistant"

                        content = []

                        # Add image reference(s) for first user message
                        if role == "user" and i == 0:
                            if task_type == "screen_transition":
                                # screen_transition task uses two images (before and after)
                                content.append({"type": "image", "index": 0})
                                content.append({"type": "image", "index": 1})
                            else:
                                content.append({"type": "image", "index": 0})

                        # Clean and normalize text
                        if value_text:
                            # Remove <image> tokens
                            value_text = clean_image_token(value_text)
                            # Normalize action format based on platform type
                            # Desktop: click(x=0.5, y=0.5) -> click(coordinate=[500, 500])
                            # Mobile: click(x=0.5, y=0.5) -> tap(coordinate=[500, 500])
                            value_text = normalize_action_format(value_text, platform_type)

                            # Add coordinate system explanation for user_intention task (user message only)
                            if task_type == "user_intention" and role == "user":
                                value_text += "\n\nNote: The coordinate uses a [0, 1000] range where [0, 0] is the top-left corner and [1000, 1000] is the bottom-right corner."

                            # Understanding tasks: assistant replies are plain
                            # QA text (intent / caption / transition), not action
                            # descriptions — use ``text`` on both sides.
                            content.append({"type": "text", "text": value_text})

                        messages.append({"role": role, "content": content})

                # Skip records where assistant has no content (empty GPT response in upstream data)
                if any(m["role"] == "assistant" and not m.get("content") for m in messages):
                    skips["empty_assistant_response"] += 1
                    continue

                # Get resolution if available
                resolution = data.get("resolution", None)
                if resolution is None and "width" in data and "height" in data:
                    resolution = [data["width"], data["height"]]

                entry = {
                    "images": images,
                    "messages": messages,
                    "metadata": LiteCUAMetadata(
                        dims=(platform_type, "understanding"),
                        extra_tool_schemas=[],
                        valid_actions=None,
                        others={
                            # Keyed on the raw file position, not on how many
                            # records survived: SplitAssigner hashes this string,
                            # so a kept-row counter would re-roll the published
                            # train/validation assignment every time upstream
                            # filtering changes. The source schema is
                            # {conversations, image, width, height} -- it carries
                            # no id of its own -- so this is the only key, and the
                            # three sibling grounding adapters here spell it the
                            # same way.
                            "id": f"{key}_{line_num}",
                            "resolution": resolution,
                            "os": get_os_from_key(key),
                            "source": "OpenGVLab/ScaleCUA-Data",
                            "source_id": key,
                        },
                    ).to_dict(),
                }
                results.append(entry)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid ScaleCUA JSON in {annotation_path} line {line_num + 1}"
                ) from e

    # Close the ledger. Every branch above either appends a row or names a
    # reason; a future one that does neither shows up here by name.
    unaccounted = n_records - len(results) - sum(skips.values())
    if unaccounted:
        skips["unaccounted"] += unaccounted

    return results, n_records, skips


def main():
    parser = argparse.ArgumentParser(
        description="Process ScaleCUA understanding data",
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
        help="Process at most this many meta entries per category (smoke test)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Process at most N records per annotation file (smoke test)",
    )
    args = parser.parse_args()

    # Get base directory from environment or use default
    CUA_LITE_RAW_DATASETS_ROOT = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not CUA_LITE_RAW_DATASETS_ROOT:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT environment variable must be set")
        print(
            "Please set it to the directory containing zyliu/ScaleCUA-Data-Understanding (source); image paths in output use OpenGVLab/ScaleCUA-Data"
        )
        print("\nExample:")
        print("  export CUA_LITE_RAW_DATASETS_ROOT=/path/to/datasets")
        print("  uv run python lite/data/preproc/scalecua/understanding.py")
        return 1

    # According to AGENTS.md:
    # Understanding data comes from: ${CUA_LITE_RAW_DATASETS_ROOT}/zyliu/ScaleCUA-Data-Understanding/
    base_source_dir = os.path.join(
        CUA_LITE_RAW_DATASETS_ROOT, "zyliu", "ScaleCUA-Data-Understanding"
    )

    if not os.path.exists(base_source_dir):
        print(f"Error: Source directory not found: {base_source_dir}")
        print("Please ensure the ScaleCUA-Data-Understanding dataset is available")
        return 1

    # Read meta.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(script_dir, "meta.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Categorize entries
    understanding_entries = {}
    for key, value in meta.items():
        if "understanding" in value.get("conv_style", ""):
            platform_type = get_platform_type(key)
            task_type = get_task_type(key)

            category = f"{platform_type}_{task_type}"
            if category not in understanding_entries:
                understanding_entries[category] = []
            understanding_entries[category].append((key, value))

    if args.head_entries is not None:
        understanding_entries = {
            c: e[: args.head_entries] for c, e in understanding_entries.items()
        }
    print(f"Found {len(understanding_entries)} categories:")
    for category, entries in understanding_entries.items():
        print(f"  {category}: {len(entries)} entries")

    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("No files will be written. Use without --dry-run to process data.")
        return 0

    if not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT environment variable must be set")
        print("Please set it to the output directory for processed datasets")
        return 1

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)

    # Variant for understanding == sub-task name (caption / user_intention /
    # screen_transition). Each (platform, "understanding") cohort spans
    # multiple variants → multi-variant layout on disk.
    skips: Counter[str] = Counter()
    n_records_total = 0
    n_entries = n_entries_no_records = 0
    total_loss_entries: list[str] = []
    for category, entries in understanding_entries.items():
        platform_type, task_type = category.split("_", 1)
        for key, value in entries:
            if args.verbose:
                print(f"Processing {key}...")
            results, n_records, entry_skips = process_understanding_entry(
                key,
                value,
                base_source_dir,
                CUA_LITE_RAW_DATASETS_ROOT,
                platform_type,
                task_type,
                verbose=args.verbose,
                head=args.head,
            )
            n_entries += 1
            n_records_total += n_records
            skips.update(entry_skips)
            corrupt_paths = prestage_images(
                store, (path for entry in results for path in entry["images"])
            )
            n_kept = 0
            for entry in results:
                if any(Path(path) in corrupt_paths for path in entry["images"]):
                    skips["image_corrupt_on_host"] += 1
                    entry_skips["image_corrupt_on_host"] += 1
                    continue
                bk, e = stage_entry(entry, store=store, splitter=splitter, variant=task_type)
                buffers[bk].append(e)
                n_kept += 1
            # A 100%-drop entry is what a broken adapter path looks like, and it
            # is also legitimate here (an unextracted image shard, or a
            # category whose assistant responses are all empty upstream). So it
            # gets its own unmissable line rather than an exit code: the failure
            # mode is silence, not a bad exit status, and a non-zero exit would
            # fire on every legitimately-empty --head window on this host.
            if n_records == 0:
                n_entries_no_records += 1
                print(f"  {category}/{key}: read 0 records (annotation file absent or empty)")
            elif not n_kept:
                total_loss_entries.append(key)
                print(
                    f"  DROPPED ALL: {category}/{key}: {n_records} records -> 0 rows "
                    f"({dict(entry_skips)})"
                )
            elif args.verbose:
                print(f"  Processed {n_kept} entries")

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(
        f"Records read: {n_records_total}  kept: {n_rows}  "
        f"skipped: {sum(skips.values())}  "
        f"(entries: {n_entries}, read-nothing: {n_entries_no_records}, "
        f"all-dropped: {len(total_loss_entries)})"
    )
    if skips:
        print("Skip reasons:", dict(skips))
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
