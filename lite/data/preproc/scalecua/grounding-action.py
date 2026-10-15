"""ScaleCUA Grounding Action Data Preprocessor.

Reads ``${CUA_LITE_RAW_DATASETS_ROOT}/OpenGVLab/ScaleCUA-Data/`` and writes
SFT-ready parquets to the canonical layout:

    ${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA/<platform>/grounding.action/<split>/<variant>.parquet

Image bytes are deduplicated into the dataset's content-addressed image
store and the row's ``images`` column carries paths relative to
``$CUA_LITE_DATASETS_ROOT``.

Usage::

    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw
    export CUA_LITE_DATASETS_ROOT=/path/to/processed
    uv run python lite/data/preproc/scalecua/grounding-action.py \
        [--dry-run] [--verbose] [--head N] [--head-entries N]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LiteDesktopActionSet,
    LiteMobileActionSet,
    normalize_keys,
)
from lite.core.tools.action_space.keys import is_canonical_key_token
from lite.core.tools.calls import stamp_messages_tool_call_ids
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate, prestage_images
from lite.data.preproc.scalecua.utils import (
    make_image_store,
    make_splitter,
    out_dir_for,
    stage_entry,
)
from lite.utils.path import resolve_path

VARIANT = "action"  # cohort variant; always its own folder


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


def _parse_key_list_literal(
    keys_str: str, action_content: str, key: str, line_num: int
) -> list[str]:
    try:
        value = ast.literal_eval(f"[{keys_str}]")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Malformed key list in {key} line {line_num}: {action_content[:200]}"
        ) from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) for item in value)
    ):
        raise ValueError(
            f"Key list must be a non-empty list[str] in {key} line {line_num}: "
            f"{action_content[:200]}"
        )
    return value


def _checked_keys(keys: list[str], action_content: str, key: str, line_num: int) -> list[str]:
    try:
        normalized = normalize_keys(keys)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported key payload {keys!r} in {key} line {line_num}: "
            f"{action_content[:200]}"
        ) from exc
    bad = [value for value in normalized if not is_canonical_key_token(value)]
    if bad:
        raise ValueError(
            f"Unsupported key token(s) {bad!r} in {key} line {line_num}: "
            f"{action_content[:200]}"
        )
    return normalized


def parse_action(
    action_text: str, platform_type: str, key: str, line_num: int
) -> tuple[list[dict[str, Any]], int, Counter[str]]:
    """
    Parse action from source format and convert to tool_calls format.

    Source format: <action>\nclick(x=0.0703, y=0.9454)\n</action>

    Args:
        action_text: The raw action text from source data
        platform_type: "mobile", "desktop", or "browser"
        key: Source key name for error messages
        line_num: Line number for error messages

    Returns:
        List of tool_calls dicts

    Raises:
        ValueError: If action cannot be parsed
    """
    # Extract action content from <action>...</action> tags
    match = re.search(r"<action>\s*(.+?)\s*</action>", action_text, re.DOTALL)
    if not match:
        raise ValueError(
            f"Failed to find <action> tags in {key} line {line_num}: {action_text[:200]}"
        )

    action_content = match.group(1).strip()

    # Parse click(x=..., y=...) format
    click_match = re.match(r"click\(x=([0-9.]+),\s*y=([0-9.]+)\)", action_content)
    if click_match:
        x = float(click_match.group(1))
        y = float(click_match.group(2))

        # Normalize to [0, 1000]
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        # For mobile, use "tap" instead of "click"
        if platform_type == "mobile":
            return [LiteMobileActionSet.tap(coordinate=[x_norm, y_norm])]
        else:
            return [LiteDesktopActionSet.click(coordinate=[x_norm, y_norm])]

    # Parse rightClick(x=..., y=...) format
    right_click_match = re.match(r"rightClick\(x=([0-9.]+),\s*y=([0-9.]+)\)", action_content)
    if right_click_match:
        x = float(right_click_match.group(1))
        y = float(right_click_match.group(2))

        # Normalize to [0, 1000]
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        # For mobile, use "long_press" as equivalent to right-click
        if platform_type == "mobile":
            return [LiteMobileActionSet.long_press(coordinate=[x_norm, y_norm])]
        else:
            return [LiteDesktopActionSet.click(coordinate=[x_norm, y_norm], button="right")]

    # Parse doubleClick(x=..., y=...) format
    double_click_match = re.match(r"doubleClick\(x=([0-9.]+),\s*y=([0-9.]+)\)", action_content)
    if double_click_match:
        x = float(double_click_match.group(1))
        y = float(double_click_match.group(2))

        # Normalize to [0, 1000]
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        # For mobile, double-tap is two taps
        if platform_type == "mobile":
            return [
                LiteMobileActionSet.tap(coordinate=[x_norm, y_norm]),
                LiteMobileActionSet.tap(coordinate=[x_norm, y_norm]),
            ]
        else:
            return [LiteDesktopActionSet.click(coordinate=[x_norm, y_norm], clicks=2)]

    # Parse scroll(x=..., y=..., direction=...) format
    scroll_match = re.match(
        r"scroll\(x=([0-9.]+),\s*y=([0-9.]+),\s*direction=(\w+)\)", action_content
    )
    if scroll_match:
        x = float(scroll_match.group(1))
        y = float(scroll_match.group(2))
        direction = scroll_match.group(3)

        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        if platform_type == "mobile":
            raise ValueError(
                f"Mobile scroll grounding has no executable endpoint in {key} "
                f"line {line_num}: {action_content[:200]}"
            )
        return [
            LiteDesktopActionSet.scroll(direction=direction, amount=3, coordinate=[x_norm, y_norm])
        ]

    # Parse type(text='...') format
    type_match = re.match(r"type\(text=['\"](.+?)['\"]\)", action_content)
    if type_match:
        text = type_match.group(1)
        if platform_type == "mobile":
            return [LiteMobileActionSet.type(text=text)]
        else:
            return [LiteDesktopActionSet.type(text=text)]

    # Parse key(keys=[...]) format
    key_match = re.match(r"key\(keys=\[(.*?)\]\)", action_content)
    if key_match:
        if platform_type == "mobile":
            raise ValueError(
                f"Mobile key grounding has no canonical action in {key} "
                f"line {line_num}: {action_content[:200]}"
            )
        keys = _parse_key_list_literal(key_match.group(1), action_content, key, line_num)
        return [LiteDesktopActionSet.key(keys=_checked_keys(keys, action_content, key, line_num))]

    # If we can't parse the action, raise an error
    raise ValueError(
        f"Failed to parse action format in {key} line {line_num}: {action_content[:200]}"
    )


def process_action_grounding_entry(
    key: str,
    value: dict[str, Any],
    base_dir: str,
    base_datasets_dir: str,
    platform_type: str,
    verbose: bool = False,
    head: int | None = None,
) -> list[dict[str, Any]]:
    """
    Process a single action grounding entry from meta.json.

    ``head`` stops after reading that many annotation records (smoke test).

    Returns:
        ``(processed entries, records read, skip ledger)``

    Raises:
        FileNotFoundError: If annotation file is not found
        ValueError: If data format is invalid
    """
    results = []
    n_records = 0
    skips: Counter[str] = Counter()

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

            # Get resolution
            width = data.get("width")
            height = data.get("height")
            resolution = [width, height] if width and height else None

            # Parse conversations
            conversations = data.get("conversations", [])
            if len(conversations) < 2:
                raise ValueError(f"Insufficient conversations in {key} line {line_num}")

            # Build messages
            messages = []

            # Process user message (first conversation)
            user_conv = conversations[0]
            if user_conv.get("from") not in ["human", "user"]:
                raise ValueError(f"First conversation is not from user in {key} line {line_num}")

            user_text = clean_image_token(user_conv.get("value", ""))
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image", "index": 0}, {"type": "text", "text": user_text}],
                }
            )

            # Process assistant message (second conversation)
            assistant_conv = conversations[1]
            if assistant_conv.get("from") not in ["gpt", "assistant"]:
                raise ValueError(
                    f"Second conversation is not from assistant in {key} line {line_num}"
                )

            # Parse action from assistant response (raises ValueError on failure)
            tool_calls = parse_action(assistant_conv.get("value", ""), platform_type, key, line_num)

            messages.append({"role": "assistant", "tool_calls": tool_calls})
            messages = stamp_messages_tool_call_ids(messages, preserve=True)

            entry = {
                "images": images,  # abs paths; main() rehashes into store
                "messages": messages,
                "metadata": LiteCUAMetadata(
                    dims=(platform_type, "grounding.action"),
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

    return results, n_records, skips


def main():
    parser = argparse.ArgumentParser(
        description="Process ScaleCUA action grounding data",
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
        print("  uv run python lite/data/preproc/scalecua/grounding-action.py")
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

    # Filter entries for action grounding
    # Selection criteria: key contains "action_grounding" and conv_style matches "internvl2_5_*_grounding_v1"
    action_grounding_entries = {}
    for key, value in meta.items():
        conv_style = value.get("conv_style", "")
        # Check if this is an action grounding entry
        if "action_grounding" in key and re.match(r"internvl2_5_\w+_grounding_v1", conv_style):
            platform_type = get_platform_type(key)
            if platform_type not in action_grounding_entries:
                action_grounding_entries[platform_type] = []
            action_grounding_entries[platform_type].append((key, value))

    if args.head_entries is not None:
        action_grounding_entries = {
            p: e[: args.head_entries] for p, e in action_grounding_entries.items()
        }
    print("Found action grounding entries by platform type:")
    for platform_type, entries in action_grounding_entries.items():
        print(f"  {platform_type}: {len(entries)} entries")

    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("No files will be written. Use without --dry-run to process data.")
        for platform_type, entries in action_grounding_entries.items():
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
    n_records_total = 0
    skips: Counter[str] = Counter()

    for platform_type, entries in action_grounding_entries.items():
        for key, value in entries:
            if args.verbose:
                print(f"Processing {key}...")
            results, n_records, entry_skips = process_action_grounding_entry(
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
            kept = []
            for entry in results:
                if has_oob_coordinate(entry):
                    skips["oob_coordinate"] += 1
                    entry_skips["oob_coordinate"] += 1
                    continue
                kept.append(entry)
            corrupt_paths = prestage_images(
                store, (path for entry in kept for path in entry["images"])
            )
            n_kept = 0
            for entry in kept:
                if any(Path(path) in corrupt_paths for path in entry["images"]):
                    skips["image_corrupt_on_host"] += 1
                    entry_skips["image_corrupt_on_host"] += 1
                    continue
                bk, e = stage_entry(entry, store=store, splitter=splitter, variant=VARIANT)
                buffers[bk].append(e)
                n_kept += 1
            if args.verbose:
                print(f"  Processed {n_kept} entries")
            if n_records and not n_kept:
                print(
                    f"  DROPPED ALL: {platform_type}/{key}: {n_records} "
                    f"records -> 0 rows ({dict(entry_skips)})"
                )

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(
        f"Records read: {n_records_total}  kept: {n_rows}  "
        f"skipped: {sum(skips.values())}"
    )
    print(f"Skip reasons: {dict(skips)}")
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
