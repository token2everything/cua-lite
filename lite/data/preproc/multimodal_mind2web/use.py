"""Multimodal-Mind2Web → cua-lite browser ``use`` preprocessor.

Converts the ``osunlp/Multimodal-Mind2Web`` web-agent benchmark (real
human-annotated multi-step episodes across 100+ websites) into the canonical
``use`` cohort:

    cua-lite/Multimodal-Mind2Web/browser/use/<split>/use.parquet

**Training-safe scope.** Only the ``train`` split is processed. The
``test_task`` / ``test_website`` / ``test_domain`` splits are the standard
Mind2Web held-out benchmarks and must NEVER be trained on — this adapter does
not read them. ``train`` is hash-split into train/validation (no upstream-split
label is written).

Each step carries a full-page screenshot, a task instruction, and a target DOM
element with a pixel-space bounding box. Action mapping (keyed off
``operation.original_op``):

    CLICK -> click(coordinate)
    TYPE  -> click(coordinate), type(text=value)
    HOVER -> mouse_move(coordinate)
    ENTER -> key(keys=["enter"])
    SELECT -> drop the whole episode (no native select tool; click+type won't
              replay native <select> typeahead reliably)

The coordinate is the centre of ``pos_candidates[0].bounding_box_rect``
(``x,y,w,h`` pixels) normalized by the per-step screenshot size to [0,1000].
True out-of-bounds centres are skipped -- reported as such, and distinctly from
an unparseable rect -- before endpoint clamping handles rounding noise.

Screenshot geometry varies *within* an episode (full-page shots grow and shrink
as the page navigates), so ``others.resolution`` is published only when every
step of the episode shares one size, and is ``None`` otherwise; coordinates are
normalized per step and are unaffected either way.
There is no per-step termination signal in the source. The raw rows carry one
screenshot per step and none after the final executable action, so the row ends
on that final action at EOF. Intermediate screenshots still become
``role:"tool"`` results for the previous action; no terminal result or ``Done.``
turn is fabricated.

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount
    export CUA_LITE_DATASETS_ROOT=/path/to/canonical-output
    uv run python lite/data/preproc/multimodal_mind2web/use.py \
        [--dry-run] [--verbose] [--head N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

import pyarrow.dataset as ds
from PIL import Image as PILImage

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    MAX_NORM,
    LiteDesktopActionSet,
    clamp_norm,
    merge_adjacent_lite_action_batches,
)
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate
from lite.data.preproc.multimodal_mind2web.utils import (
    RAW_PREFIX,
    SOURCE,
    make_image_store,
    make_splitter,
    out_dir_for,
    stage_entry,
)
from lite.data.utils.messages import (
    finalize_use_messages,
)
from lite.utils.path import resolve_path

# Mind2Web full-page screenshots can be ~100 Mpx; we only read the header
# (width/height), so lift PIL's decompression-bomb guard.
PILImage.MAX_IMAGE_PIXELS = None

VARIANT = "use"
PLATFORM = "browser"
SPLIT = "train"  # training-safe: never process the test_* benchmark holdouts

_COLUMNS = (
    "annotation_id", "target_action_index", "target_action_reprs",
    "confirmed_task", "operation", "pos_candidates",
    "website", "domain", "subdomain",
)
_COORD_EPS = 1e-6


class SkipEpisodeError(Exception):
    """Raised when an entire episode should be skipped."""


class _SelectSkip(SkipEpisodeError):
    """Episode contained a SELECT step; whole episode is dropped."""


def _parse_operation(op_str: str) -> tuple[str, str] | None:
    try:
        obj = json.loads(op_str)
        return str(obj["original_op"]), str(obj.get("value", ""))
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return None


def _parse_pos_candidate(cand_str: str) -> dict[str, Any] | None:
    """Parse ``pos_candidates[0]`` (whose ``attributes`` is a nested JSON string)."""
    try:
        obj = json.loads(cand_str)
        attrs_raw = obj.get("attributes", "{}")
        attrs = json.loads(attrs_raw) if isinstance(attrs_raw, str) else attrs_raw
        return attrs if isinstance(attrs, dict) else None
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return None


def _parse_bbox_rect(bbox_rect) -> tuple[float, float, float, float] | None:
    """Parse ``"x,y,w,h"`` (pixels) → floats, or ``None`` if unparseable."""
    if not isinstance(bbox_rect, str):
        return None
    parts = bbox_rect.split(",")
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (float(p) for p in parts)
    except (ValueError, TypeError):
        return None
    return x, y, w, h


def _bbox_center(
    rect: tuple[float, float, float, float], img_w: int, img_h: int
) -> list[int] | None:
    """Centre of a parsed rect → [cx, cy] normalized to [0, 1000].

    ``None`` means the centre falls outside the captured screenshot — a
    well-formed rect the screenshot does not cover, which is a different
    outcome from :func:`_parse_bbox_rect` failing, and is reported as such.
    """
    x, y, w, h = rect
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    if (
        cx < -_COORD_EPS
        or cx > 1 + _COORD_EPS
        or cy < -_COORD_EPS
        or cy > 1 + _COORD_EPS
    ):
        return None
    return [clamp_norm(round(cx * MAX_NORM)), clamp_norm(round(cy * MAX_NORM))]


def _read_image_size(abs_path: str) -> tuple[int, int] | None:
    try:
        with PILImage.open(abs_path) as im:
            return int(im.width), int(im.height)
    except (OSError, SyntaxError, ValueError):
        return None


def _build_tool_calls(original_op: str, value: str, coordinate: list[int]) -> list[dict] | None:
    if original_op == "CLICK":
        return [LiteDesktopActionSet.click(coordinate=coordinate)]
    if original_op == "TYPE":
        if not isinstance(value, str) or value == "":
            return None
        return [LiteDesktopActionSet.click(coordinate=coordinate),
                LiteDesktopActionSet.type(text=value)]
    if original_op == "HOVER":
        return [LiteDesktopActionSet.mouse_move(coordinate=coordinate)]
    if original_op == "ENTER":
        return [LiteDesktopActionSet.key(keys=["enter"])]
    if original_op == "SELECT":
        raise _SelectSkip("episode contains a SELECT step")
    raise ValueError(f"unknown Mind2Web operation: {original_op!r}")


def episode_to_entry(annotation_id: str, rows: list[dict]) -> dict:
    """Group an episode's step rows into one canonical use entry."""
    indexed: list[tuple[int, dict]] = []
    for row in rows:
        try:
            idx = int(row.get("target_action_index", ""))
        except (ValueError, TypeError):
            raise SkipEpisodeError(f"unparseable target_action_index={row.get('target_action_index')!r}")
        indexed.append((idx, row))
    indexed.sort(key=lambda t: t[0])
    if not indexed:
        raise SkipEpisodeError("empty episode")
    if len({i for i, _ in indexed}) != len(indexed):
        raise SkipEpisodeError("duplicate target_action_index within episode")

    step_indices = [i for i, _ in indexed]
    steps = [r for _, r in indexed]

    abs_images: list[str] = []
    resolutions: list[tuple[int, int]] = []
    for step_idx in step_indices:
        rel = f"{RAW_PREFIX}/images/{annotation_id}/{step_idx:02d}.jpg"
        try:
            abs_path = resolve_path(rel, "CUA_LITE_RAW_DATASETS_ROOT")
        except FileNotFoundError as e:
            raise SkipEpisodeError(f"image not found: {rel}") from e
        size = _read_image_size(abs_path)
        if size is None:
            raise SkipEpisodeError(f"unreadable image: {rel}")
        abs_images.append(abs_path)
        resolutions.append(size)

    confirmed_task = steps[0].get("confirmed_task")
    if not isinstance(confirmed_task, str) or not confirmed_task.strip():
        raise SkipEpisodeError("missing confirmed_task")

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "image", "index": 0}, {"type": "text", "text": confirmed_task}]},
    ]
    for i, (row, (img_w, img_h)) in enumerate(zip(steps, resolutions)):
        if i > 0:
            messages.append({"role": "user", "content": [{"type": "image", "index": i}]})
        op_parsed = _parse_operation(row.get("operation", ""))
        if op_parsed is None:
            raise SkipEpisodeError("malformed operation JSON")
        original_op, value = op_parsed

        pos_cands = row.get("pos_candidates") or []
        if not pos_cands:
            raise SkipEpisodeError("empty pos_candidates")
        attrs = _parse_pos_candidate(pos_cands[0])
        if attrs is None:
            raise SkipEpisodeError("malformed pos_candidates[0]")
        rect = _parse_bbox_rect(attrs.get("bounding_box_rect"))
        if rect is None:
            raise SkipEpisodeError(f"malformed bounding_box_rect: {attrs.get('bounding_box_rect')!r}")
        coordinate = _bbox_center(rect, img_w, img_h)
        if coordinate is None:
            raise SkipEpisodeError(
                f"target outside screenshot: bounding_box_rect="
                f"{attrs['bounding_box_rect']!r} screenshot={img_w}x{img_h}"
            )

        tool_calls = _build_tool_calls(original_op, value, coordinate)
        if tool_calls is None:
            raise SkipEpisodeError(f"degenerate step (op={original_op}, value={value!r})")
        tool_calls = merge_adjacent_lite_action_batches(tool_calls)

        action_desc = str(row.get("target_action_reprs") or "")
        content = [{"type": "action_description", "text": action_desc}] if action_desc else []
        assistant: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        if content:
            assistant["content"] = content
        messages.append(assistant)
    messages = finalize_use_messages(messages)

    first = steps[0]
    # ``others.resolution`` names the pixel size of the row's screenshots, and a
    # single pair can only say that when every step agrees. Mind2Web's full-page
    # shots mostly do not: over the first 1000 episodes by ``annotation_id``,
    # 942 carry more than one distinct size (up to 17 in one episode) and
    # 5,440 / 7,698 screenshots differ from step 0's -- median 1.25x, p90 3.2x,
    # p99 8.0x in height. Publishing step 0's pair regardless would hand a
    # consumer an authoritative-looking number that is wrong for 71% of the
    # images it describes, so the value is published only for the episodes where
    # it is true of all of them. ``None`` is how the other adapters already spell
    # "this row has no resolution" (guiact, guiodyssey, opencua, scalecua), it is
    # what the canonical gate expects (nothing in ``rows.py`` reads the key, and
    # aguvis publishes 4.1M rows without it), and it cannot be mistaken for a
    # measurement the way step 0's pair can.
    resolution = list(resolutions[0]) if len(set(resolutions)) == 1 else None
    return {
        "images": abs_images,
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=(PLATFORM, "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={
                "id": f"mind2web_{annotation_id}",
                "resolution": resolution,
                "os": None,
                "source": SOURCE,
                "source_id": annotation_id,
                "website": first.get("website"),
                "domain": first.get("domain"),
                "subdomain": first.get("subdomain"),
            },
        ).to_dict(),
    }


def load_train_episodes(raw_root: str) -> dict[str, list[dict]]:
    """Group ``train`` step rows by ``annotation_id`` (needed columns only)."""
    data_dir = os.path.join(raw_root, RAW_PREFIX, "data")
    shards = sorted(p for p in os.listdir(data_dir) if p.startswith(f"{SPLIT}-") and p.endswith(".parquet"))
    if not shards:
        raise FileNotFoundError(f"No {SPLIT}-*.parquet shards under {data_dir}")
    dataset = ds.dataset([os.path.join(data_dir, s) for s in shards], format="parquet")
    episodes: dict[str, list[dict]] = defaultdict(list)
    for batch in dataset.to_batches(columns=list(_COLUMNS)):
        for row in batch.to_pylist():
            ann = row.get("annotation_id")
            if isinstance(ann, str) and ann:
                episodes[ann].append(row)
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multimodal-Mind2Web → cua-lite browser use (train only)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--head", type=int, default=None, help="Process at most N episodes (smoke test)")
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT must be set", file=sys.stderr)
        return 1
    if not args.dry_run and not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT must be set", file=sys.stderr)
        return 1

    episodes = load_train_episodes(raw_root)
    ann_ids = sorted(episodes)
    if args.head is not None:
        ann_ids = ann_ids[: args.head]
    print(f"Loaded {len(episodes)} train episodes ({len(ann_ids)} selected)")

    if args.dry_run:
        print("=== DRY RUN === would write browser/use/<split>/use.parquet")
        return 0

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    n_ok = n_select = n_skip = n_oob = n_corrupt = 0

    for ann in ann_ids:
        try:
            entry = episode_to_entry(ann, episodes[ann])
        except _SelectSkip:
            n_select += 1
            continue
        except SkipEpisodeError as e:
            n_skip += 1
            if args.verbose:
                print(f"  [skip] {ann}: {e}", file=sys.stderr)
            continue
        if has_oob_coordinate(entry):
            n_oob += 1
            continue
        try:
            bk, e = stage_entry(entry, store=store, splitter=splitter, variant=VARIANT)
        except staging.CorruptImageError as img_err:
            n_corrupt += 1
            if args.verbose:
                print(f"  [skip-corrupt] {ann}: {img_err}", file=sys.stderr)
            continue
        buffers[bk].append(e)
        n_ok += 1

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(f"Episodes: ok={n_ok} select_dropped={n_select} skipped={n_skip} oob={n_oob} corrupt={n_corrupt}")
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
