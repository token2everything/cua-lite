"""Aguvis Stage 1 → cua-lite grounding.action preprocessor.

Converts single-step grounding data from ``xlangai/aguvis-stage1`` into the
canonical ``grounding.action`` cohort. Each source record is a screenshot +
referring instruction → one ``pyautogui.*`` call (click/locate the element).
The upstream sub-dataset becomes the cohort *variant*:

    cua-lite/Aguvis/<platform>/grounding.action/<split>/<subset>.parquet

Source record: ``{"image": "<file>", "conversations": [{"from":"human","value":"<image>\\n<instr>"},
{"from":"gpt","value":"pyautogui.click(x=0.5, y=0.3)"}]}``. ``seeclick_mi`` packs
multiple (image, turn) pairs per record and is unpacked into separate samples.

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount
    export CUA_LITE_DATASETS_ROOT=/path/to/canonical-output
    uv run python lite/data/preproc/aguvis/grounding-action.py \
        [--subset all|<name>] [--dry-run] [--verbose] [--head N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import batched
from pathlib import Path
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import stamp_messages_tool_call_ids
from lite.data import staging
from lite.data.preproc.aguvis.utils import (
    SOURCE_STAGE1,
    AguvisParseError,
    clean_instruction,
    image_resolution,
    make_image_store,
    make_splitter,
    out_dir_for,
    pyautogui_to_tool_calls,
    stage_entry,
)
from lite.data.preproc.common import has_oob_coordinate, prestage_images
from lite.data.utils.messages import extra_tool_schemas_for_messages
from lite.utils.path import resolve_path

# variant = upstream sub-dataset. platform drives the action space.
SUBSETS: dict[str, dict[str, Any]] = {
    "seeclick":          {"json": "seeclick.json",           "image_dir": "seeclick/seeclick_web_imgs", "platform": "browser", "os": None},
    "seeclick_mi":       {"json": "seeclick_mi.json",        "image_dir": "seeclick/seeclick_web_imgs", "platform": "browser", "os": None, "multi_image": True},
    "guienv":            {"json": "guienv.json",             "image_dir": "guienvs/images",             "platform": "browser", "os": None},
    "webui":             {"json": "webui350k.json",          "image_dir": "webui350k/images",           "platform": "browser", "os": None},
    "ricosca":           {"json": "ricosca.json",            "image_dir": "ricosca/images",             "platform": "mobile",  "os": "android"},
    "rico_icon":         {"json": "ricoig16k.json",          "image_dir": "ricoig16k/images",           "platform": "mobile",  "os": "android"},
    "widget_captioning": {"json": "widget_captioning.json",  "image_dir": "widget_captioning/images",   "platform": "mobile",  "os": "android"},
    "ui_refexp":         {"json": "ui_refexp.json",          "image_dir": "ui_refexp/images",           "platform": "mobile",  "os": "android"},
    "omniact":           {"json": "omniact_fix.json",        "image_dir": "omniact/images",             "platform": "desktop", "os": None},
}


def _build_row(image_rel_abs: str, instruction: str, tool_calls: list[dict], cfg: dict, row_id: str, source_id: str) -> dict:
    messages = stamp_messages_tool_call_ids([
        {"role": "user", "content": [{"type": "image", "index": 0}, {"type": "text", "text": instruction}]},
        {"role": "assistant", "tool_calls": tool_calls},
    ], preserve=True)
    return {
        "images": [image_rel_abs],
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=(cfg["platform"], "grounding.action"),
            extra_tool_schemas=extra_tool_schemas_for_messages(messages),
            valid_actions=None,
            others={
                "id": row_id,
                # Read off the file: Aguvis coordinates arrive already
                # normalized, so nothing in the record names the pixel surface.
                "resolution": image_resolution(image_rel_abs),
                "os": cfg["os"],
                "source": SOURCE_STAGE1,
                "source_id": source_id,
            },
        ).to_dict(),
    }


def _resolve(image_name: str, cfg: dict) -> str | None:
    rel = f"{SOURCE_STAGE1}/{cfg['image_dir']}/{image_name}"
    try:
        return resolve_path(rel, "CUA_LITE_RAW_DATASETS_ROOT")
    except FileNotFoundError:
        return None


def process_subset(name: str, cfg: dict, raw_root: str, *, head: int | None, verbose: bool):
    """Yield (variant, entry) for one Stage-1 sub-dataset."""
    json_path = os.path.join(raw_root, SOURCE_STAGE1, cfg["json"])
    if not os.path.isfile(json_path):
        print(f"Warning: {cfg['json']} not found, skipping subset {name}", file=sys.stderr)
        return
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if head is not None:
        data = data[:head]
    stats: Counter = Counter()
    for idx, rec in enumerate(data):
        pairs: list[tuple[str, str, str]] = []  # (image_name, instruction, action_code)
        convs = rec.get("conversations", [])
        # SeeClick/webui/rico_icon records are multi-turn: one screenshot with
        # several (instruction → action) grounding pairs. We emit EVERY pair as
        # its own grounding sample (pairing human[i] with the following gpt[i] —
        # NOT human[0] with gpt[-1], which would mismatch the instruction with
        # the wrong action). seeclick_mi additionally carries one image per pair.
        images = rec.get("image", []) if cfg.get("multi_image") else None
        if cfg.get("multi_image") and not isinstance(images, list):
            stats["skip"] += 1
            continue
        image_name_single = rec.get("image", "")
        pi = ci = 0
        while ci < len(convs) - 1:
            h, g = convs[ci], convs[ci + 1]
            if h.get("from") in ("human", "user") and g.get("from") in ("gpt", "assistant"):
                if cfg.get("multi_image"):
                    if pi >= len(images):
                        break
                    img = images[pi]
                else:
                    img = image_name_single
                pairs.append((img, clean_instruction(h.get("value", "")), g.get("value", "").strip()))
                pi += 1
                ci += 2
            else:
                ci += 1
        if not pairs:
            stats["skip"] += 1
            continue
        for pj, (image_name, instruction, action_code) in enumerate(pairs):
            if not image_name or not instruction or not action_code:
                stats["skip"] += 1
                continue
            abs_path = _resolve(image_name, cfg)
            if abs_path is None:
                stats["missing_img"] += 1
                continue
            try:
                tool_calls = pyautogui_to_tool_calls(action_code, cfg["platform"])
            except AguvisParseError as e:
                stats["parse_err"] += 1
                if verbose:
                    print(f"  [skip] {name} idx={idx}: {e}", file=sys.stderr)
                continue
            if not tool_calls:
                stats["skip"] += 1
                continue
            stem = os.path.splitext(os.path.basename(str(image_name)))[0]
            row_id = f"aguvis_{name}_{idx}_{pj}"
            yield name, _build_row(abs_path, instruction, tool_calls, cfg, row_id, stem)
            stats["ok"] += 1
    print(f"  {name}: ok={stats['ok']} skip={stats['skip']} missing_img={stats['missing_img']} parse_err={stats['parse_err']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aguvis Stage 1 → cua-lite grounding.action",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--head", type=int, default=None, help="Process at most N records per subset")
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT must be set", file=sys.stderr)
        return 1
    subsets = list(SUBSETS) if args.subset == "all" else [args.subset]

    if args.dry_run:
        print("=== DRY RUN ===")
        for n in subsets:
            print(f"  {n}: {SUBSETS[n]['platform']}/grounding.action/.../{n}")
        return 0
    if not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT must be set", file=sys.stderr)
        return 1

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    n_oob = n_corrupt = 0
    for name in subsets:
        rows = process_subset(
            name, SUBSETS[name], raw_root, head=args.head, verbose=args.verbose
        )
        for batch in batched(rows, 512):
            # Grounding records are single-image, and many adjacent pairs reuse
            # one screenshot. Verify/hash unique source paths concurrently, then
            # stage rows sequentially so split assignment keeps source order.
            corrupt_paths = prestage_images(
                store, (entry["images"][0] for _, entry in batch)
            )
            for variant, entry in batch:
                if Path(entry["images"][0]) in corrupt_paths:
                    n_corrupt += 1
                    if args.verbose:
                        print(f"  [skip-corrupt] {variant}: {entry['images'][0]}", file=sys.stderr)
                    continue
                if has_oob_coordinate(entry):
                    n_oob += 1
                    continue
                bk, e = stage_entry(entry, store=store, splitter=splitter, variant=variant)
                buffers[bk].append(e)

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(f"Dropped {n_oob} rows with out-of-bound coordinates")
    print(f"Skipped {n_corrupt} rows with corrupt/unreadable images")
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
