"""CAGUI Agent -> CUA-Lite use preprocessor.

Converts the OpenBMB/CAGUI CAGUI_agent/domestic split into CUA-Lite's mobile
use format and writes the canonical local layout consumed by
``lite.data.hf.upload`` and ``lite.train.export.export_sft``. The dataset
contains 600 Chinese-language Android agent episodes (Eleme, WeChat, NetEase
Music, etc.), each delivered as one JSON file plus per-step JPEGs.

Per-step source record fields (relevant subset)::

    {
      "episode_id":        "<hex>",
      "step_id":           <int>,
      "instruction":       "<Chinese task instruction>",
      "image_path":        "domestic/<id>/<id>_<step>.jpeg",
      "image_width":       <int>,
      "image_height":      <int>,
      "ui_positions":      "[[y, x, h, w], ...]",   # normalized [0,1] boxes
      "result_action_type":  <int>,    # AITW ActionType enum (see mapping below)
      "result_action_text":  "<str>",  # populated for TYPE (code 3)
      "result_touch_yx":     "[y_norm, x_norm]",  # JSON-encoded; [-1,-1] sentinel
      "result_lift_yx":      "[y_norm, x_norm]"
    }

Action mapping (verified against ``OpenBMB/AgentCPM-GUI`` action_type.py and
empirically against the 600 domestic episodes)::

  0  LONG_POINT             -> long_press from result_touch_yx
  1  NO_ACTION              -> wait(duration / 1000)
  3  TYPE                   -> type(text=result_action_text)
  4  DUAL_POINT             -> tap if dist(touch, lift) <= 0.04 else swipe
  5  PRESS_BACK             -> system_button(button="Back")  (not seen in 600 ep)
  6  PRESS_HOME             -> system_button(button="Home")  (not seen in 600 ep)
  7  PRESS_ENTER            -> system_button(button="Enter") (not seen in 600 ep)
  10 STATUS_TASK_COMPLETE   -> terminator -> structural ``Done.`` final
  11 STATUS_TASK_IMPOSSIBLE -> terminator -> ``metadata.others.terminate_status="failure"``
  any other                 -> CAGUIActionError (raise)

Coordinate convention: source ``result_touch_yx`` is JSON-encoded ``[y, x]`` in
[0, 1] range — note y first. We swap to (x, y) and scale to [0, 1000] int. The
sentinel ``[-1, -1]`` is used by non-touch actions and is NOT consumed for those
codes.

CAGUI ships no thought / action-description text, so use assistant turns carry
only canonical wrapper ``tool_calls``. We never invent descriptions.

The per-step ``ui_positions`` (every on-screen element's box) is preserved under
``metadata.others.ui_positions`` as one entry per kept action step, each box
converted to [0, 1000] ``[x1, y1, x2, y2]``.

Terminator handling (matches ``opencua/use.py``'s convention): step codes
10/11 are NOT emitted as assistant turns. Code 10 is a purely structural
episode-end marker and is dropped — the episode ends with the content-only
``Done.`` final. Code 11 (STATUS_TASK_IMPOSSIBLE) is a source-asserted failure
label, so its payload is recorded in ``metadata.others`` via
``terminate_outcome_others``. No ``terminate`` tool_call is persisted. A
terminal step's image is retained only when the source uses it as the
post-action screenshot/result for the preceding real action.

Output: ${CUA_LITE_DATASETS_ROOT}/cua-lite/CAGUI/mobile/use/<split>/use.parquet

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw_datasets
    export CUA_LITE_DATASETS_ROOT=/path/to/output
    uv run python -m lite.data.preproc.cagui.use \
        [--subset domestic|all] [--head N] [--dry-run] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from glob import glob
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    MAX_NORM,
    LiteMobileActionSet,
    clamp_norm,
    merge_adjacent_lite_action_batches,
)
from lite.core.tools.calls import make_tool_call
from lite.data import staging
from lite.data.preproc.cagui.utils import (
    make_image_store,
    make_splitter,
    out_dir_for,
    stage_entry,
)
from lite.data.preproc.common import has_oob_coordinate
from lite.data.utils.messages import (
    extra_tool_schemas_for_messages,
    finalize_use_messages,
    structural_final_message,
    terminate_outcome_others,
)
from lite.utils.path import resolve_path

VARIANT = "use"

# AITW convention: tap if normalized [y,x] L2 distance between touch and lift
# is <= 0.04, else swipe.
SWIPE_DISTANCE_THRESHOLD = 0.04
_COORD_EPS = 1e-6


class SkipEpisodeError(Exception):
    """Raised when an episode should be skipped (missing image, unsupported anomaly)."""


class CAGUIActionError(RuntimeError):
    """Raised when an action code is unmapped or malformed — fail loud."""


SUBSETS: dict[str, dict[str, str]] = {
    "domestic": {
        "episode_glob": "OpenBMB/CAGUI/CAGUI_agent/domestic/*",
        "image_path_prefix": "OpenBMB/CAGUI/CAGUI_agent",
    },
}


def _parse_yx(s: str) -> tuple[float, float]:
    """Parse a CAGUI '[y, x]' JSON-encoded coordinate string.

    A corrupted source coordinate surfaces as ``SkipEpisodeError`` (the episode
    is skipped with a counter) rather than crashing the whole pipeline.
    """
    try:
        arr = json.loads(s)
        if not isinstance(arr, list) or len(arr) != 2:
            raise SkipEpisodeError(f"Expected [y, x] list, got {s!r}")
        return float(arr[0]), float(arr[1])
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise SkipEpisodeError(f"Unparseable [y, x] coord {s!r}: {e}") from e


def _norm_yx_to_xy_1000(y: float, x: float) -> list[int]:
    """Source order is (y, x) in [0, 1]. Return target [x, y] in int [0, 1000]."""
    if (
        y < -_COORD_EPS
        or y > 1 + _COORD_EPS
        or x < -_COORD_EPS
        or x > 1 + _COORD_EPS
    ):
        raise SkipEpisodeError(f"coordinate outside [0, 1]: y={y}, x={x}")
    xi = clamp_norm(int(round(x * MAX_NORM)))
    yi = clamp_norm(int(round(y * MAX_NORM)))
    return [xi, yi]


def _duration_seconds(step: dict) -> float:
    value = step.get("duration")
    if isinstance(value, bool):
        raise SkipEpisodeError(f"invalid duration {value!r}")
    try:
        seconds = float(value) / 1000
    except (TypeError, ValueError) as e:
        raise SkipEpisodeError(f"invalid duration {value!r}") from e
    if not math.isfinite(seconds) or seconds <= 0:
        raise SkipEpisodeError(f"invalid duration {value!r}")
    return seconds


def _convert_ui_positions(s: Any) -> list[list[int]]:
    """Convert a CAGUI ui_positions string of [y, x, h, w] normalized boxes to
    a list of [x1, y1, x2, y2] boxes in int [0, 1000] (clamped).

    Returns ``[]`` if the field is missing or unparseable — ui_positions is
    auxiliary metadata, not load-bearing, so a bad value never skips a step.
    """
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        boxes = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(boxes, list):
        return []
    out: list[list[int]] = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            y, x, h, w = (float(v) for v in box)
            x1 = clamp_norm(int(round(x * MAX_NORM)))
            y1 = clamp_norm(int(round(y * MAX_NORM)))
            x2 = clamp_norm(int(round((x + w) * MAX_NORM)))
            y2 = clamp_norm(int(round((y + h) * MAX_NORM)))
        except (ValueError, TypeError, OverflowError):
            continue
        out.append([x1, y1, x2, y2])
    return out


def step_to_tool_calls(step: dict) -> list[dict] | str:
    """Convert one CAGUI step to a list of CUA-lite mobile tool_calls.

    Returns either:
      - list[dict]                — normal action(s) for this step
      - "TERMINATE_SUCCESS"       — terminator (code 10), drop this step
      - "TERMINATE_FAILURE"       — terminator (code 11), drop this step

    Raises ``CAGUIActionError`` on unknown / malformed codes and
    ``SkipEpisodeError`` on malformed source values.
    """
    code = step.get("result_action_type")

    if code == 10:
        return "TERMINATE_SUCCESS"
    if code == 11:
        return "TERMINATE_FAILURE"

    if code == 3:
        text = step.get("result_action_text") or ""
        if not isinstance(text, str) or not text:
            raise CAGUIActionError(
                f"TYPE step has empty/invalid text at {step.get('episode_id')}/{step.get('step_id')}"
            )
        return [LiteMobileActionSet.type(text=text)]

    if code == 4:
        ty, tx = _parse_yx(step["result_touch_yx"])
        ly, lx = _parse_yx(step["result_lift_yx"])
        dist = ((ty - ly) ** 2 + (tx - lx) ** 2) ** 0.5
        if dist <= SWIPE_DISTANCE_THRESHOLD:
            return [LiteMobileActionSet.tap(coordinate=_norm_yx_to_xy_1000(ty, tx))]
        return [
            LiteMobileActionSet.swipe(
                start_coordinate=_norm_yx_to_xy_1000(ty, tx),
                coordinate=_norm_yx_to_xy_1000(ly, lx),
            )
        ]

    if code == 0:
        ty, tx = _parse_yx(step["result_touch_yx"])
        return [LiteMobileActionSet.long_press(
            coordinate=_norm_yx_to_xy_1000(ty, tx),
            duration=_duration_seconds(step),
        )]

    if code == 5:
        return [LiteMobileActionSet.system_button(button="Back")]
    if code == 6:
        return [LiteMobileActionSet.system_button(button="Home")]
    if code == 7:
        return [LiteMobileActionSet.system_button(button="Enter")]

    if code == 1:
        return [LiteMobileActionSet.wait(duration=_duration_seconds(step))]

    raise CAGUIActionError(
        f"Unmapped action_type={code!r} at {step.get('episode_id')}/{step.get('step_id')}"
    )


def build_episode(steps: list[dict], image_path_prefix: str) -> dict:
    """Build one use row from an episode's step list.

    Raises ``SkipEpisodeError`` if any image is unresolvable, an action is
    malformed or unsupported, or no actionable step precedes the terminator.
    Image paths are returned as ABSOLUTE paths (``stage_entry`` hashes them into
    the content-addressed store).
    """
    if not steps:
        raise SkipEpisodeError("empty episode")

    instruction = steps[0].get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise SkipEpisodeError(f"missing instruction in episode {steps[0].get('episode_id')}")
    episode_id = steps[0].get("episode_id") or "unknown"
    resolution = [steps[0].get("image_width"), steps[0].get("image_height")]

    images: list[str] = []
    assistant_calls: list[list[dict]] = []
    ui_positions: list[list[list[int]]] = []
    final_status: str | None = None

    def append_step_image(step: dict) -> None:
        rel_image_path = step.get("image_path")
        if not isinstance(rel_image_path, str) or not rel_image_path:
            raise SkipEpisodeError(
                f"missing image_path in episode {episode_id} step {step.get('step_id')}"
            )
        img_rel = os.path.join(image_path_prefix, rel_image_path)
        try:
            img_abs = resolve_path(img_rel, "CUA_LITE_RAW_DATASETS_ROOT")
        except FileNotFoundError as e:
            raise SkipEpisodeError(
                f"unresolvable image {img_rel} in episode {episode_id}"
            ) from e
        images.append(img_abs)
        ui_positions.append(_convert_ui_positions(step.get("ui_positions")))

    for step_pos, step in enumerate(steps):
        result = step_to_tool_calls(step)
        if isinstance(result, str):
            if step_pos != len(steps) - 1:
                raise SkipEpisodeError(
                    f"episode {episode_id} has steps after terminal step {step.get('step_id')}"
                )
            # Terminator: stop walking and remember status. The terminate call
            # itself is not persisted; keep this screenshot only when it is the
            # post-action result for a preceding real action.
            final_status = "success" if result == "TERMINATE_SUCCESS" else "failure"
            if assistant_calls:
                append_step_image(step)
            break

        append_step_image(step)
        assistant_calls.append(merge_adjacent_lite_action_batches(result))

    if not assistant_calls:
        raise SkipEpisodeError(f"episode {episode_id} has no actionable steps before terminator")

    # No ``terminate`` call is ever persisted: code 10 (STATUS_TASK_COMPLETE)
    # and code 11 (STATUS_TASK_IMPOSSIBLE) both end on the ``Done.`` final, and
    # code 11's source-asserted failure label moves to ``others``.
    terminate_call = (
        make_tool_call("terminate", {"status": final_status})
        if final_status is not None
        else None
    )

    # Build pre-finalization messages. Later screenshots are staged as user
    # images here, then finalize_use_messages() rewrites them into role:"tool"
    # results keyed by the preceding assistant call.
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    for i, calls in enumerate(assistant_calls):
        messages.append({"role": "assistant", "tool_calls": calls})
        if i + 1 < len(images):
            messages.append(
                {"role": "user", "content": [{"type": "image", "index": i + 1}]}
            )
    messages.append(structural_final_message())
    messages = finalize_use_messages(messages)

    assert len(images) in {len(assistant_calls), len(assistant_calls) + 1}, (
        f"images/assistant turns mismatch: {len(images)} vs {len(assistant_calls)}"
    )

    return {
        "images": images,
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=("mobile", "use"),
            extra_tool_schemas=extra_tool_schemas_for_messages(messages),
            valid_actions=None,
            others={
                "id": f"cagui_domestic_{episode_id}",
                "resolution": resolution,
                "os": "android",
                "source": "OpenBMB/CAGUI",
                "source_id": episode_id,
                "language": "zh",
                "ui_positions": ui_positions,
                **terminate_outcome_others(terminate_call),
            },
        ).to_dict(),
    }


def iter_episodes(subset_cfg: dict[str, str], raw_root: str, head: int | None):
    """Yield (episode_id, steps_list) in sorted order, capped by `head`.

    Raises ``FileNotFoundError`` if the glob matches zero directories or an
    episode directory lacks its required ``<id>.json`` annotation. Malformed
    annotations fail loud instead of disappearing before run accounting.
    """
    pattern = os.path.join(raw_root, subset_cfg["episode_glob"])
    ep_dirs = sorted(glob(pattern))
    if not ep_dirs:
        raise FileNotFoundError(
            f"No episode directories matched glob {pattern!r}. "
            "Check that CUA_LITE_RAW_DATASETS_ROOT points at the decoded CAGUI dataset."
        )
    if head is not None:
        ep_dirs = ep_dirs[:head]
    for ep_dir in ep_dirs:
        ep_id = os.path.basename(ep_dir)
        json_path = os.path.join(ep_dir, f"{ep_id}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing episode annotation: {json_path}")
        try:
            with open(json_path, encoding="utf-8") as f:
                steps = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid episode JSON: {json_path}") from e
        if not isinstance(steps, list):
            raise ValueError(
                f"Expected list at {json_path}, got {type(steps).__name__}"
            )
        yield ep_id, steps


def main() -> int:
    parser = argparse.ArgumentParser(description="CAGUI agent -> CUA-Lite use")
    parser.add_argument(
        "--subset",
        choices=["domestic", "all"],
        default="all",
        help="Which subset(s) to process (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Process only first N episodes per subset (debug only)",
    )
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT is not set", file=sys.stderr)
        return 1
    if not os.getenv("CUA_LITE_DATASETS_ROOT") and not args.dry_run:
        print("Error: CUA_LITE_DATASETS_ROOT is not set", file=sys.stderr)
        return 1

    subsets = ["domestic"] if args.subset == "all" else [args.subset]

    if args.dry_run:
        # Histogram + build-pass sanity check, no output written.
        for name in subsets:
            cfg = SUBSETS[name]
            print(f"\n=== DRY RUN subset {name!r} ===")
            action_ctr: Counter[int] = Counter()
            n_episodes = n_skipped = 0
            for ep_id, steps in iter_episodes(cfg, raw_root, args.head):
                n_episodes += 1
                for step in steps:
                    action_ctr[step.get("result_action_type")] += 1
                try:
                    build_episode(steps, cfg["image_path_prefix"])
                except SkipEpisodeError as e:
                    n_skipped += 1
                    if args.verbose:
                        print(f"  [skip] {ep_id}: {e}")
            print(f"  Episodes scanned: {n_episodes} ({n_skipped} would be skipped)")
            print("  Action type histogram (all steps):")
            for code, count in sorted(action_ctr.items(), key=lambda kv: (kv[0] is None, kv[0])):
                print(f"    {code}: {count}")
        return 0

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict]] = defaultdict(list)
    n_oob = n_corrupt = 0

    for name in subsets:
        cfg = SUBSETS[name]
        print(f"\n=== Processing subset {name!r} ===")
        n_total = n_skipped = n_converted = 0
        for ep_id, steps in iter_episodes(cfg, raw_root, args.head):
            n_total += 1
            try:
                row = build_episode(steps, cfg["image_path_prefix"])
            except SkipEpisodeError as e:
                n_skipped += 1
                if args.verbose:
                    print(f"  [skip] {ep_id}: {e}")
                continue
            if has_oob_coordinate(row):
                n_oob += 1
                continue
            try:
                bk, e = stage_entry(row, store=store, splitter=splitter, variant=VARIANT)
            except staging.CorruptImageError as img_err:
                n_corrupt += 1
                if args.verbose:
                    print(f"  [skip-corrupt] {ep_id}: {img_err}")
                continue
            buffers[bk].append(e)
            n_converted += 1
        print(f"  Episodes: {n_total} total, {n_converted} converted, {n_skipped} skipped")

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(f"Dropped {n_oob} trajectories with out-of-bound coordinates")
    print(f"Dropped {n_corrupt} trajectories with corrupt images")
    print(f"\nWrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    print(f"Image store: {store.count()} unique images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
