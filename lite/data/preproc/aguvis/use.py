"""Aguvis Stage 2 → cua-lite use preprocessor.

Converts multi-step trajectory data from ``xlangai/aguvis-stage2`` into the
canonical ``use`` cohort. Each source JSON is a per-step record list grouped
into episodes by image-name prefix. The upstream sub-dataset
becomes the cohort *variant*:

    cua-lite/Aguvis/<platform>/use/<split>/<subset>.parquet

Per step, the action is split across two gpt turns: the first ("Action: <desc>",
or COAT's "Observation/Thought/Action" block) → ``action_description`` (+
``inline_reasoning`` for COAT's Thought); the last (``pyautogui.*``) → tool_calls.
Episodes that end on a source terminator become the structural content-only
``Done.`` final: the source carries no answer text, only an optional terminator.
Source-asserted ``terminate(status!="success")`` calls are not persisted as tool
calls; their payload is moved to ``metadata.others`` by
``terminate_outcome_others``. A terminator step's screenshot is retained only
when it is the post-action result for the previous non-terminal action. When an
episode has no terminator step (most stage-2 subsets -- android_control, coat,
guide, miniwob -- carry none), the last action has no post-action screenshot, so
the row ends on that final action at EOF rather than fabricating a result or a
``Done.`` turn.

``guide.json`` spells 392 of its scrolls as a bare, argument-less
``pyautogui.scroll()`` whose direction is stated only in that step's own
``Action:`` prose -- the text the row publishes as its ``action_description``. It
is read there and passed to the parser resolved; see
``utils.scroll_direction_from_description``.

Run accounting (printed unconditionally, no --verbose needed). Rows are whole
episodes, so the ledger is denominated in SOURCE RECORDS -- the only unit in
which a run reconciles:
    Records read: R  kept: K  skipped: S   with   R == K + S
    Skip reasons: {<reason>: n, ...}
    plus one ``SOURCE ABSENT ON THIS HOST`` line per subset whose JSON is not on
    this host, which reads 0 records and so denominates nothing.

Usage:
    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw-data-mount
    export CUA_LITE_DATASETS_ROOT=/path/to/canonical-output
    uv run python lite/data/preproc/aguvis/use.py \
        [--subset all|<name>] [--dry-run] [--verbose] [--head N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.messages import make_assistant_content
from lite.core.tools.action_space import merge_adjacent_lite_action_batches
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.data import staging
from lite.data.preproc.aguvis.utils import (
    SOURCE_STAGE2,
    AguvisParseError,
    SkipEpisodeError,
    episode_key,
    extract_goal,
    extract_thought_and_desc,
    image_resolution,
    make_image_store,
    make_splitter,
    out_dir_for,
    pyautogui_to_tool_calls,
    scroll_direction_from_description,
    stage_entry,
)
from lite.data.preproc.common import has_oob_coordinate
from lite.data.utils.messages import (
    extra_tool_schemas_for_messages,
    finalize_use_messages,
    pop_terminal_terminate,
    structural_final_message,
    terminate_outcome_others,
)
from lite.utils.path import resolve_path

SUBSETS: dict[str, dict[str, Any]] = {
    "android_control": {"json": "android_control.json", "image_dir": "android_control/images", "platform": "mobile", "os": "android"},
    "aitw":            {"json": "aitw-l1.json",         "image_dir": "aitw-v1/images",         "platform": "mobile", "os": "android"},
    "coat":            {"json": "coat.json",            "image_dir": "coat/images",            "platform": "mobile", "os": "android"},
    "guide":           {"json": "guide.json",           "image_dir": "guide-v2/images",        "platform": "browser", "os": None},
    "miniwob":         {"json": "miniwob-l1.json",      "image_dir": "miniwob/images",         "platform": "browser", "os": None},
}

_RESULT_BOUNDARY_TOOLS_BY_PLATFORM: dict[str, frozenset[str]] = {
    "mobile": frozenset({
        "long_press",
        "open_app",
        "swipe",
        "system_button",
        "tap",
        "type",
        "wait",
    }),
    "browser": frozenset({
        "click",
        "drag",
        "key",
        "mouse_move",
        "open_app",
        "scroll",
        "type",
        "wait",
    }),
}


def build_episode(ep_id: str, steps: list[tuple[int, dict]], cfg: dict, raw_root: str) -> dict:
    """Group an episode's steps into one canonical use row."""
    steps.sort(key=lambda x: x[0])
    platform = cfg["platform"]
    images: list[str] = []
    messages: list[dict[str, Any]] = []
    goal: str | None = None
    terminate_call: dict[str, Any] | None = None
    cursor_coordinate: list[int] | None = None
    last_observation_key: tuple[int, str] | None = None

    for i, (step_idx, rec) in enumerate(steps):
        convs = rec.get("conversations", [])
        humans = [t for t in convs if t.get("from") in ("human", "user")]
        gpts = [t for t in convs if t.get("from") in ("gpt", "assistant")]
        if not humans or not gpts:
            raise SkipEpisodeError("bad_turns", f"step {step_idx}: missing human/gpt turns")

        if i == 0:
            first_human = humans[0].get("value", "")
            goal = extract_goal(first_human)
            if not goal:
                raise SkipEpisodeError("empty_goal", "cannot extract goal from first step")
            history = first_human.partition("Previous actions:")[2].strip()
            if history and history.splitlines()[0].strip(" .;").lower() != "none":
                raise SkipEpisodeError(
                    "partial_episode", "first retained record already has action history"
                )

        rel = f"{SOURCE_STAGE2}/{cfg['image_dir']}/{rec['image']}"
        try:
            abs_path = resolve_path(rel, "CUA_LITE_RAW_DATASETS_ROOT")
        except FileNotFoundError as e:
            raise SkipEpisodeError("missing_image", f"step {step_idx}: image not found: {rel}") from e

        thought, action_desc = extract_thought_and_desc(gpts[0].get("value", ""))
        action_code = gpts[-1].get("value", "").strip()
        if not action_code:
            raise SkipEpisodeError("empty_action_code", f"step {step_idx}: empty action code")
        try:
            # ``guide.json`` spells 392 scrolls as a bare ``pyautogui.scroll()``,
            # whose direction lives only in ``action_desc`` -- the very text this
            # step publishes as its ``action_description``. Pass it resolved.
            tool_calls = pyautogui_to_tool_calls(
                action_code, platform,
                scroll_direction=scroll_direction_from_description(action_desc),
                cursor_coordinate=cursor_coordinate,
            )
        except AguvisParseError as e:
            raise SkipEpisodeError("unsupported_action", f"step {step_idx}: parse error: {e}") from e
        tool_calls = merge_adjacent_lite_action_batches(tool_calls)
        if not tool_calls:
            raise SkipEpisodeError("no_tool_calls", f"step {step_idx}: no tool calls produced")

        if len(tool_calls) == 1 and tool_call_name(tool_calls[0]) == "terminate":
            if i != len(steps) - 1:
                raise SkipEpisodeError(
                    "post_terminal_steps", f"terminal source step {step_idx} is not final"
                )
            # No ``terminate`` call is ever persisted: the step is dropped and
            # replaced by the ``Done.`` final below, with any source-asserted
            # failure status/reason moved to ``others``. If a real action is
            # pending, keep this screenshot as that action's tool result.
            terminate_call = tool_calls[0]
            if messages:
                images.append(abs_path)
                messages.append({
                    "role": "user",
                    "content": [{"type": "image", "index": len(images) - 1}],
                })
            break

        for call in tool_calls:
            for action in tool_call_arguments(call).get("actions", []):
                if action.get("action") in {"click", "drag", "mouse_move"} and action.get("coordinate"):
                    cursor_coordinate = list(action["coordinate"])

        content = make_assistant_content(inline_reasoning=thought, action_description=action_desc)
        observation_key = (step_idx, rec["image"])
        if (
            observation_key == last_observation_key
            and messages
            and messages[-1].get("tool_calls")
        ):
            messages[-1]["tool_calls"] = merge_adjacent_lite_action_batches(
                [*messages[-1]["tool_calls"], *tool_calls]
            )
            existing = messages[-1].setdefault("content", [])
            for part in content:
                prior = next((item for item in existing if item["type"] == part["type"]), None)
                if prior is None:
                    existing.append(part)
                else:
                    prior["text"] += "\n" + part["text"]
            if not existing:
                messages[-1].pop("content", None)
            continue

        images.append(abs_path)
        last_observation_key = observation_key
        if len(images) == 1:
            # first emitted step (robust even if step 0 was a skipped terminate)
            user_content = [{"type": "image", "index": 0}, {"type": "text", "text": goal}]
        else:
            user_content = [{"type": "image", "index": len(images) - 1}]
        messages.append({"role": "user", "content": user_content})

        assistant: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
        if content:
            assistant["content"] = content
        messages.append(assistant)

    # A terminator's screenshot may already have been retained as the previous
    # action's result; the terminate call itself is still shed here by the same
    # shared helper the other cohorts and ``devs/migration`` use. If an
    # executable action remains at EOF, keep it as the final supervised label
    # rather than inventing a tool result or a later ``Done.`` turn.
    terminate_call = pop_terminal_terminate(messages) or terminate_call
    ends_on_action = bool(
        messages
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("tool_calls")
    )
    if messages and not ends_on_action:
        messages.append(structural_final_message())
    messages = finalize_use_messages(
        messages,
        result_boundary_tools=_RESULT_BOUNDARY_TOOLS_BY_PLATFORM.get(platform, frozenset()),
    )

    if not images:
        raise SkipEpisodeError(
            "too_few_images", f"only {len(images)} images after filtering (need >=1)")

    return {
        "images": images,
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=(platform, "use"),
            extra_tool_schemas=extra_tool_schemas_for_messages(messages),
            valid_actions=None,
            others={
                "id": f"aguvis_{cfg['_name']}_{ep_id}",
                # First emitted screenshot, like opencua/mm2w/gui360: an
                # episode is one screen surface, and Aguvis records no pixel
                # geometry of its own (its coordinates are pre-normalized).
                "resolution": image_resolution(images[0]),
                "os": cfg["os"],
                "source": SOURCE_STAGE2,
                "source_id": ep_id,
                **terminate_outcome_others(terminate_call),
            },
        ).to_dict(),
    }


def process_subset(
    name: str, cfg: dict, raw_root: str, *, head: int | None, verbose: bool,
) -> tuple[list[tuple[str, dict, int]], int, Counter]:
    """One Stage-2 sub-dataset -> ``(rows, records_read, skips)``.

    Each element of *rows* is ``(variant, entry, n_records)``: the accounting
    travels WITH the row, because the caller's own out-of-bounds and
    corrupt-image gates drop whole episodes and could otherwise only report them
    in trajectories. The ledger is denominated in SOURCE RECORDS — one row is a
    whole episode and consumes many records, so records are the only unit in
    which a run reconciles. ``skips`` is keyed by :class:`SkipEpisodeError`'s
    ``reason`` plus the grouping-stage buckets ``no_image_name`` and
    ``beyond_head``; ``missing_image`` is re-keyed to
    ``missing_image_dir_absent`` when this host has no image directory for the
    subset at all, because that is the host lacking data rather than the adapter
    dropping it.
    """
    cfg = {**cfg, "_name": name}
    json_path = os.path.join(raw_root, SOURCE_STAGE2, cfg["json"])
    if not os.path.isfile(json_path):
        # A host-data cause, not an adapter drop, and there is nothing to
        # denominate: no record was read.
        print(f"  {name}: SOURCE ABSENT ON THIS HOST: {cfg['json']} -> 0 records read")
        return [], 0, Counter()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    n_records = len(data)
    image_dir_present = os.path.isdir(os.path.join(raw_root, SOURCE_STAGE2, cfg["image_dir"]))
    skips: Counter[str] = Counter()

    episodes: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for rec in data:
        image_name = rec.get("image", "")
        if not isinstance(image_name, str) or not image_name:
            skips["no_image_name"] += 1
            continue
        ep_id, step_idx = episode_key(image_name)
        episodes[ep_id].append((step_idx, rec))

    ep_ids = sorted(episodes)
    if head is not None:
        for ep_id in ep_ids[head:]:
            skips["beyond_head"] += len(episodes[ep_id])
        ep_ids = ep_ids[:head]

    rows: list[tuple[str, dict, int]] = []
    for ep_id in ep_ids:
        steps = episodes[ep_id]
        try:
            row = build_episode(ep_id, steps, cfg, raw_root)
        except SkipEpisodeError as e:
            reason = e.reason
            if reason == "missing_image" and not image_dir_present:
                reason = "missing_image_dir_absent"
            skips[reason] += len(steps)
            if verbose:
                print(f"  [skip] {name}/{ep_id}: {e}", file=sys.stderr)
            continue
        rows.append((name, row, len(steps)))
    print(f"  {name}: {n_records} records → {len(episodes)} episodes → "
          f"{len(rows)} rows, {sum(skips.values())} records skipped")
    return rows, n_records, skips


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aguvis Stage 2 → cua-lite use",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--head", type=int, default=None, help="Process at most N episodes per subset")
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT must be set", file=sys.stderr)
        return 1
    subsets = list(SUBSETS) if args.subset == "all" else [args.subset]

    if args.dry_run:
        print("=== DRY RUN ===")
        for n in subsets:
            print(f"  {n}: {SUBSETS[n]['platform']}/use/.../{n}")
        return 0
    if not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT must be set", file=sys.stderr)
        return 1

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    skips: Counter[str] = Counter()
    n_read = n_kept = 0
    for name in subsets:
        rows, n_records, subset_skips = process_subset(
            name, SUBSETS[name], raw_root, head=args.head, verbose=args.verbose)
        n_read += n_records
        skips.update(subset_skips)
        for variant, entry, n_ep_records in rows:
            if has_oob_coordinate(entry):
                skips["oob_coordinate"] += n_ep_records
                continue
            try:
                bk, e = stage_entry(entry, store=store, splitter=splitter, variant=variant)
            except staging.CorruptImageError as img_err:
                # Corrupt/truncated image in the episode → drop the trajectory.
                skips["corrupt_image"] += n_ep_records
                if args.verbose:
                    print(f"  [skip-corrupt] {variant}: {img_err}", file=sys.stderr)
                continue
            buffers[bk].append(e)
            n_kept += n_ep_records

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    # Close the ledger: every record read is either behind a row or named by a
    # reason. A drop path added later without a bucket surfaces as
    # ``unaccounted`` instead of disappearing.
    unaccounted = n_read - n_kept - sum(skips.values())
    if unaccounted:
        skips["unaccounted"] += unaccounted
    # Both lines print unconditionally: an empty ``Skip reasons`` line is the
    # evidence that nothing was dropped, and its absence used to be
    # indistinguishable from nobody having counted.
    print(f"Records read: {n_read}  kept: {n_kept}  skipped: {sum(skips.values())}")
    print("Skip reasons:", dict(skips))
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
