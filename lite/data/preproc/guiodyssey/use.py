"""GUIOdyssey → cua-lite mobile ``use`` preprocessor.

Converts the ``hflqf88888/GUIOdyssey`` dataset (8,334 long-horizon cross-app
Android task trajectories) into the canonical ``use`` cohort:

    ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIOdyssey/mobile/use/<split>/use.parquet

Action mapping (source ``info`` is already normalized to [0, 1000]; do not
scale it by device width/height):

  CLICK [[x,y],[x,y]]        → tap(coordinate)
  CLICK "KEY_HOME/BACK/APPSELECT" → system_button(button="Home"/"Back"/"Recent")
  TEXT "<str>"               → type(text)
  SCROLL [[sx,sy],[ex,ey]]   → swipe(start_coordinate, coordinate)
  LONG_PRESS [[x,y],[x,y]]   → long_press(coordinate)
  COMPLETE                   → structural content-only ``Done.`` final
  INCOMPLETE                 → ``metadata.others.terminate_status="failure"`` (+
                               ``terminate_reason`` from the step's ``ps``, which is
                               where the annotator's why-it-was-impossible text lives;
                               ``info`` is ``""`` for every INCOMPLETE step in the corpus)

Each assistant turn carries ``content`` built from ``inline_reasoning`` (the
step's first-person ``intention``) + ``action_description`` (the step's
``low_level_instruction``), plus the mapped ``tool_calls``.

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount
    export CUA_LITE_DATASETS_ROOT=/path/to/canonical-output
    uv run python lite/data/preproc/guiodyssey/use.py \
        [--dry-run] [--verbose] [--head N]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.messages import make_assistant_content
from lite.core.tools.action_space import LiteMobileActionSet, merge_adjacent_lite_action_batches
from lite.core.tools.calls import make_tool_call
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate
from lite.data.preproc.guiodyssey.utils import (
    OS,
    PLATFORM,
    SOURCE,
    SkipEpisodeError,
    iter_episodes,
    make_image_store,
    make_splitter,
    normalize_xy,
    out_dir_for,
    screenshot_rel_path,
    stage_entry,
)
from lite.data.utils.messages import (
    extra_tool_schemas_for_messages,
    finalize_use_messages,
    structural_final_message,
    terminate_outcome_others,
)
from lite.utils.path import resolve_path

VARIANT = "use"

_SYSTEM_KEY_MAP: dict[str, str] = {
    "KEY_HOME": "Home",
    "KEY_BACK": "Back",
    "KEY_APPSELECT": "Recent",
}


class GUIOdysseyActionError(RuntimeError):
    """Raised when a step's action is unmapped or malformed — fail loud."""


def _parse_xy_pair(info: Any) -> tuple[float, float] | None:
    """First (x, y) pixel point from a CLICK / LONG_PRESS ``info=[[x,y],[x,y]]``."""
    try:
        if not isinstance(info, list) or len(info) < 1:
            return None
        first = info[0]
        if not isinstance(first, (list, tuple)) or len(first) != 2:
            return None
        return float(first[0]), float(first[1])
    except (TypeError, ValueError):
        return None


def _parse_scroll_pair(info: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """(start, end) pixel points from a SCROLL ``info=[[sx,sy],[ex,ey]]``."""
    try:
        if not isinstance(info, list) or len(info) != 2:
            return None
        a, b = info
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == 2 and len(b) == 2):
            return None
        return (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
    except (TypeError, ValueError):
        return None


def step_to_tool_calls(step: dict) -> list[dict] | tuple[str, str | None]:
    """One step → tool_calls, or terminal metadata for the final ``Done.`` row.

    GUIOdyssey ``info`` coordinates are already normalized to [0, 1000] (see
    ``utils.normalize_xy``), so no device-resolution argument is needed.
    """
    action = step.get("action")
    info = step.get("info")
    step_idx = step.get("step")
    ep_id = step.get("_ep_id", "?")

    if action == "COMPLETE":
        return ("TERMINATE_SUCCESS", None)
    if action == "INCOMPLETE":
        # The annotator's why-it-was-impossible text lives in ``ps``, NOT in ``info``:
        # the upstream README says *"if action is any other value, info is empty
        # ("")"* and describes ``ps`` as *"any additional information from the
        # annotator about ... why it was impossible to complete"*. Measured over all
        # 8,334 annotation files: 472 INCOMPLETE steps, 0 with non-blank ``info``,
        # 472 with non-blank ``ps`` (median 17 chars, e.g. ``"No flights"``,
        # ``"Item not found"``), so reading ``info`` here dropped 472/472 authored
        # reasons and made ``others.terminate_reason`` unreachable for this source.
        ps = step.get("ps")
        reason = ps.strip() if isinstance(ps, str) and ps.strip() else None
        return ("TERMINATE_FAILURE", reason)

    if action == "CLICK":
        if isinstance(info, str):
            btn = _SYSTEM_KEY_MAP.get(info)
            if btn is None:
                raise GUIOdysseyActionError(f"Unmapped system key {info!r} at {ep_id}/{step_idx}")
            return [LiteMobileActionSet.system_button(button=btn)]
        parsed = _parse_xy_pair(info)
        if parsed is None:
            raise SkipEpisodeError(f"CLICK step {ep_id}/{step_idx} has unparseable info={info!r}")
        return [LiteMobileActionSet.tap(coordinate=normalize_xy(parsed[0], parsed[1]))]

    if action == "LONG_PRESS":
        parsed = _parse_xy_pair(info)
        if parsed is None:
            raise SkipEpisodeError(f"LONG_PRESS step {ep_id}/{step_idx} has unparseable info={info!r}")
        return [LiteMobileActionSet.long_press(coordinate=normalize_xy(parsed[0], parsed[1]))]

    if action == "SCROLL":
        parsed = _parse_scroll_pair(info)
        if parsed is None:
            raise SkipEpisodeError(f"SCROLL step {ep_id}/{step_idx} has unparseable info={info!r}")
        (sx, sy), (ex, ey) = parsed
        return [LiteMobileActionSet.swipe(
            start_coordinate=normalize_xy(sx, sy),
            coordinate=normalize_xy(ex, ey),
        )]

    if action == "TEXT":
        if not isinstance(info, str) or not info:
            raise SkipEpisodeError(f"TEXT step {ep_id}/{step_idx} has empty/invalid info={info!r}")
        return [LiteMobileActionSet.type(text=info)]

    raise GUIOdysseyActionError(f"Unmapped action={action!r} at {ep_id}/{step_idx}")


def build_episode(ep_id: str, episode: dict) -> dict:
    """Build one canonical use row from a GUIOdyssey episode."""
    device_info = episode.get("device_info")
    task_info = episode.get("task_info")
    steps = episode.get("steps")

    if not isinstance(device_info, dict) or not isinstance(task_info, dict):
        raise SkipEpisodeError(f"episode {ep_id} missing device_info/task_info")
    if not isinstance(steps, list) or not steps:
        raise SkipEpisodeError(f"episode {ep_id} has empty/invalid steps")

    dev_w, dev_h = device_info.get("w"), device_info.get("h")
    if not isinstance(dev_w, int) or not isinstance(dev_h, int) or dev_w <= 0 or dev_h <= 0:
        raise SkipEpisodeError(f"episode {ep_id} has invalid device resolution w={dev_w} h={dev_h}")

    instruction = task_info.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise SkipEpisodeError(f"episode {ep_id} missing task_info.instruction")

    images: list[str] = []
    assistant_turns: list[dict] = []
    final_status: str | None = None
    final_reason: str | None = None

    def append_step_image(step: dict) -> int:
        step_idx = step.get("step")
        if not isinstance(step_idx, int):
            raise SkipEpisodeError(f"episode {ep_id} step missing integer `step` field")
        img_rel = screenshot_rel_path(ep_id, step_idx)
        try:
            img_abs = resolve_path(img_rel, "CUA_LITE_RAW_DATASETS_ROOT")
        except FileNotFoundError as e:
            raise SkipEpisodeError(f"unresolvable image {img_rel} in episode {ep_id}") from e
        images.append(img_abs)
        return step_idx

    for step_pos, step in enumerate(steps):
        step["_ep_id"] = ep_id
        result = step_to_tool_calls(step)
        if isinstance(result, tuple):
            if step_pos != len(steps) - 1:
                raise SkipEpisodeError(
                    f"episode {ep_id} has steps after terminal step {step.get('step')}"
                )
            tag, reason = result
            final_status = "success" if tag == "TERMINATE_SUCCESS" else "failure"
            final_reason = reason
            if assistant_turns:
                append_step_image(step)
            break

        step_idx = append_step_image(step)
        low_level = step.get("low_level_instruction")
        if not isinstance(low_level, str) or not low_level.strip():
            raise SkipEpisodeError(f"episode {ep_id} step {step_idx} missing low_level_instruction")
        intention = step.get("intention")
        intention = intention.strip() if isinstance(intention, str) else ""

        assistant_turns.append({
            "role": "assistant",
            "content": make_assistant_content(
                inline_reasoning=intention,
                action_description=low_level.strip(),
            ),
            "tool_calls": merge_adjacent_lite_action_batches(result),
        })

    if not assistant_turns:
        raise SkipEpisodeError(f"episode {ep_id} has no actionable steps before terminator")

    # No ``terminate`` call is ever persisted: COMPLETE and INCOMPLETE both end
    # on the ``Done.`` final, and INCOMPLETE's source-asserted failure label
    # (with any authored reason) moves to ``others``.
    terminate_call = (
        make_tool_call("terminate", {"status": final_status, "reason": final_reason})
        if final_status is not None
        else None
    )

    messages: list[dict] = [{
        "role": "user",
        "content": [{"type": "image", "index": 0}, {"type": "text", "text": instruction}],
    }]
    for i, turn in enumerate(assistant_turns):
        messages.append(turn)
        if i + 1 < len(images):
            messages.append({"role": "user", "content": [{"type": "image", "index": i + 1}]})
    messages.append(structural_final_message())
    messages = finalize_use_messages(messages)

    assert len(images) in {len(assistant_turns), len(assistant_turns) + 1}, (
        f"images/assistant mismatch: {len(images)} vs {len(assistant_turns)}"
    )

    return {
        "images": images,
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=(PLATFORM, "use"),
            extra_tool_schemas=extra_tool_schemas_for_messages(messages),
            valid_actions=None,
            others={
                "id": f"guiodyssey_{ep_id}",
                "resolution": [dev_w, dev_h],
                "os": OS,
                "source": SOURCE,
                "source_id": ep_id,
                "category": task_info.get("category"),
                "apps": task_info.get("app"),
                "device_name": device_info.get("device_name"),
                **terminate_outcome_others(terminate_call),
            },
        ).to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GUIOdyssey → cua-lite mobile use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true", help="Action histogram, write nothing")
    parser.add_argument("--verbose", action="store_true", help="Per-episode skip logging")
    parser.add_argument("--head", type=int, default=None, help="Process at most N episodes (smoke test)")
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT must be set", file=sys.stderr)
        return 1

    if args.dry_run:
        action_ctr: Counter[str] = Counter()
        sys_key_ctr: Counter[str] = Counter()
        n_ep = n_skip = n_err = 0
        for ep_id, episode in iter_episodes(raw_root, args.head):
            n_ep += 1
            for step in episode.get("steps") or []:
                action_ctr[step.get("action")] += 1
                if step.get("action") == "CLICK" and isinstance(step.get("info"), str):
                    sys_key_ctr[step["info"]] += 1
            try:
                build_episode(ep_id, episode)
            except SkipEpisodeError as e:
                n_skip += 1
                if args.verbose:
                    print(f"  [skip] {ep_id}: {e}")
            except GUIOdysseyActionError as e:
                n_err += 1
                if args.verbose:
                    print(f"  [action-error] {ep_id}: {e}")
        print(f"Episodes: {n_ep}  skipped: {n_skip}  action-errors: {n_err}")
        print("Action histogram:", dict(action_ctr))
        print("CLICK system-key histogram:", dict(sys_key_ctr))
        return 0

    if not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT must be set", file=sys.stderr)
        return 1

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    n_total = n_skip = n_err = n_oob = n_corrupt = 0

    for ep_id, episode in iter_episodes(raw_root, args.head):
        n_total += 1
        try:
            row = build_episode(ep_id, episode)
        except SkipEpisodeError as e:
            n_skip += 1
            if args.verbose:
                print(f"  [skip] {ep_id}: {e}")
            continue
        except GUIOdysseyActionError as e:
            n_err += 1
            print(f"  [action-error] {ep_id}: {e}", file=sys.stderr)
            continue
        if has_oob_coordinate(row):
            n_oob += 1
            continue
        try:
            bk, e = stage_entry(row, store=store, splitter=splitter, variant=VARIANT)
        except staging.CorruptImageError as img_err:
            # Corrupt/truncated screenshot in the episode → drop the trajectory.
            n_corrupt += 1
            if args.verbose:
                print(f"  [skip-corrupt] {ep_id}: {img_err}", file=sys.stderr)
            continue
        buffers[bk].append(e)

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(f"Episodes: {n_total} total, {n_rows} written, {n_skip} skipped, "
          f"{n_err} action-errors, {n_oob} dropped (OOB coords), "
          f"{n_corrupt} skipped (corrupt images)")
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
