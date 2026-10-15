"""GUIAct web-multi + smartphone Use preprocessor.

Reads the multi-step GUIAct subsets and writes SFT-ready parquets to the
canonical layout::

    ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIAct/browser/use/{train,validation}.parquet  # web-multi
    ${CUA_LITE_DATASETS_ROOT}/cua-lite/GUIAct/mobile/use/{train,validation}.parquet    # smartphone

Each source record is one step; contiguous runs are grouped into episodes by the
episode id embedded in the ``uid``, then sorted by step number. The source reuses
24 web-multi ids for later, separate tasks, so repeated runs receive a suffix;
one repeated run corrects an earlier final answer and replaces it. The first
user turn carries the episode goal
(``question``) + the initial screenshot; every later
post-action screenshot is rewritten by ``finalize_use_messages`` into a
``role:"tool"`` result keyed by the preceding assistant call. Assistant turns
carry ``tool_calls``; ``content`` appears only when the source supplies terminal
answer text for the final action turn.

GUIAct's terminal ``answer`` action spells two unrelated facts under one name
-- an AUTHORED ANSWER and a
CANNED TERMINATOR -- and which of them a subset can produce is declared once, in
``SUBSETS[...]["terminal_answer"]``:

* **web-multi is genuine QA** (``TERMINAL_ANSWER_AUTHORED_ONLY``). ``answer.text``
  is free-form prose and is the whole point of the task, so it becomes the final
  turn's text (5,491/5,501 train and 135/193 test episodes author one; 9 blank
  ones fall back to the ``Done.`` marker). Declaring the subset authored-only is
  what keeps a genuine answer reading "task complete" from being demoted.
* **smartphone is not QA at all** (``TERMINAL_ANSWER_MAY_BE_CANNED``). Its
  ``answer.text`` is a two-word status marker -- measured over both files, all
  7,476 episodes: ``"task complete"`` 7,236/7,246 train + 230/230 test,
  ``"task impossible"`` 10/7,246, zero other strings and zero blanks -- so the
  text is a *terminator*, not an answer, and is routed exactly as every sibling
  routes one: the episode ends on the structural ``Done.`` marker and a
  non-success status moves to ``metadata.others.terminate_status`` via
  ``terminate_outcome_others`` (the cagui code-10/11 and guiodyssey
  COMPLETE/INCOMPLETE shape). The two vocabularies are disjoint today -- no web
  answer spells either sentinel -- and a smartphone string outside the marker
  vocabulary is treated as authored prose rather than trusted to a lookup,
  because the source text is file I/O and this adapter does not own it.

The source has no screenshot after the last step, so that step's executable
actions end the row at EOF. If the same terminal step also authors answer text,
the text stays on that final assistant turn beside the action calls. A
**nonterminal** ``answer`` is
the one unpublishable shape: it becomes a standalone ``response`` call that the
following screenshot cannot pair with (a screenshot answers at most one call,
and the action-batch call beside it takes it), so those episodes are skipped (11
web-multi train episodes; smartphone has none). The upstream test split is
honored verbatim as ``validation``.

Image bytes are deduplicated into the dataset's content-addressed image store;
the row's ``images`` column carries paths relative to ``$CUA_LITE_DATASETS_ROOT``.

Usage::

    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw
    export CUA_LITE_DATASETS_ROOT=/path/to/processed
    # images must be extracted to disk first via scripts/process_raw_data.sh
    uv run python lite/data/preproc/guiact/use.py [--dry-run] [--verbose] \\
        [--subset web_multi|smartphone|all] [--head N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import merge_adjacent_lite_action_batches
from lite.core.tools.calls import make_tool_call
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate
from lite.data.preproc.guiact.utils import (
    SkipTrajectoryError,
    convert_mobile_action,
    convert_web_actions,
    make_image_store,
    make_splitter,
    out_dir_for,
    stage_entry,
)
from lite.data.utils.messages import (
    TERMINATE_TOOL_NAME,
    extra_tool_schemas_for_messages,
    finalize_use_messages,
    structural_final_message,
    terminate_outcome_others,
)
from lite.utils.path import resolve_path

VARIANT = "use"  # cohort variant; always its own folder

_WEB_MULTI_UID_RE = re.compile(r"^uid_record_(\d+)_step_(\d+)$")
_SMARTPHONE_UID_RE = re.compile(r"^uid_episode_(\d+)_step_(\d+)$")
_HISTORY_STEP_RE = re.compile(r"(?mi)^step\s+(\d+):")

# Whether a subset's terminal ``answer`` action may be a CANNED STATUS MARKER
# rather than an authored answer. GUIAct spells two unrelated facts with one
# action name, and only the subset says which are possible -- so the distinction
# is declared here, once, instead of being a string test scattered down the path.
#
# It is deliberately a statement about what is POSSIBLE, not about what the text
# is: within a subset that admits markers, prose and marker are still told apart
# per episode by :func:`_canned_terminator_status`. Web-multi is declared
# authored-only so that a genuine QA answer reading "task complete" could never
# be demoted to a terminator -- measured, no web answer spells either sentinel,
# and this makes that unrepresentable rather than merely absent.
TERMINAL_ANSWER_AUTHORED_ONLY = "authored_only"
TERMINAL_ANSWER_MAY_BE_CANNED = "may_be_canned"

#: smartphone's canned terminal-status vocabulary -> ``terminate`` ``status``.
#:
#: Measured over both smartphone files and both splits -- 7,476 episodes, each
#: with exactly one ``answer`` action and always on the terminal step:
#: ``"task complete"`` x 7,466, ``"task impossible"`` x 10. No third string, no
#: blank, no casing or whitespace variant. So the corpus as it stands is fully
#: covered; the map is nonetheless consulted as a TEST, never as a total
#: function, because source annotation text is file I/O and this adapter does
#: not own the vocabulary.
_CANNED_TERMINAL_STATUS: dict[str, str] = {
    "task complete": "success",
    "task impossible": "failure",
}


def _canned_terminator_status(text: str) -> str | None:
    """The ``terminate`` status *text* asserts as a canned marker, else ``None``.

    ``None`` means "this is not a marker", i.e. authored prose — and that is the
    whole distinction the ``answer`` action fails to draw: a marker is a
    TERMINATOR and must not be published as the model's answer, while prose is
    an ANSWER and must not be destroyed. Naming both cases keeps the unknown
    string from having to fall out of a lookup.
    """
    return _CANNED_TERMINAL_STATUS.get(text.strip().lower())


# subset -> config. Both train + test files are processed; the upstream split
# label rides on each row's transient metadata.others.split.
SUBSETS: dict[str, dict[str, Any]] = {
    "web_multi": {
        "files": [
            ("yiye2023/GUIAct/web-multi_train_data.json", "train"),
            ("yiye2023/GUIAct/web-multi_test_data.json", "test"),
        ],
        "image_subdir": "web-multi",
        "image_ext": "jpg",
        "split_images": True,
        "platform": "browser",
        "os": None,
        "uid_re": _WEB_MULTI_UID_RE,
        "id_prefix": "guiact_web_multi",
        "terminal_answer": TERMINAL_ANSWER_AUTHORED_ONLY,
    },
    "smartphone": {
        "files": [
            ("yiye2023/GUIAct/smartphone_train_data.json", "train"),
            ("yiye2023/GUIAct/smartphone_test_data.json", "test"),
        ],
        "image_subdir": "smartphone",
        "image_ext": "jpg",
        "platform": "mobile",
        "os": "android",
        "uid_re": _SMARTPHONE_UID_RE,
        "id_prefix": "guiact_smartphone",
        "terminal_answer": TERMINAL_ANSWER_MAY_BE_CANNED,
    },
}


def _parse_uid(uid: str, pattern: re.Pattern) -> tuple[str, int]:
    m = pattern.match(uid)
    if not m:
        raise ValueError(f"Cannot parse uid: {uid!r}")
    return m.group(1), int(m.group(2))


def _group_episodes(
    data: list[dict[str, Any]],
    uid_pattern: re.Pattern,
) -> dict[str, list[dict[str, Any]]]:
    """Group contiguous step-records into complete episodes.

    Source episode ids are not unique: 24 web-multi ids are reused for a
    different question, while one later run corrects the final answer for the
    same question. Suffix distinct runs and let a correction replace its earlier
    duplicate. Logical history continuity is validated by
    :func:`episode_to_example`; incomplete trajectories are rejected whole.
    """
    episodes: dict[str, dict[int, dict[str, Any]]] = {}
    occurrences: dict[str, int] = {}
    question_keys: dict[tuple[str, str], str] = {}
    current_source_id: str | None = None
    current_key: str | None = None
    for record in data:
        episode_id, step_num = _parse_uid(record["uid"], uid_pattern)
        if episode_id != current_source_id:
            question_key = (episode_id, record.get("question") or "")
            current_key = question_keys.get(question_key)
            if current_key is None:
                occurrences[episode_id] = occurrences.get(episode_id, 0) + 1
                occurrence = occurrences[episode_id]
                current_key = episode_id if occurrence == 1 else f"{episode_id}_run_{occurrence}"
                question_keys[question_key] = current_key
            episodes[current_key] = {}
            current_source_id = episode_id
        assert current_key is not None
        if step_num in episodes[current_key]:
            raise ValueError(
                f"Duplicate source step {step_num} in GUIAct episode {current_key}"
            )
        episodes[current_key][step_num] = record
    return {
        eid: sorted(
            source_steps.values(),
            key=lambda record: _history_step(record),
        )
        for eid, source_steps in episodes.items()
    }


def _history_step(record: dict[str, Any]) -> int:
    """Action index implied by GUIAct's accumulated logical history."""
    history = record.get("actions_history")
    if history is None:
        return int((record.get("uid") or record["image_id"]).rsplit("_step_", 1)[1])
    values = [int(value) for value in _HISTORY_STEP_RE.findall(history)]
    return max(values) + 1 if values else 0


_ANSWER_ACTION = "answer"


def _step_actions(actions_label: Any, *, platform: str) -> list[dict[str, Any]]:
    """One step's source actions as a list (smartphone stores a lone dict)."""
    return [actions_label] if platform == "mobile" else actions_label


def _has_answer_action(actions_label: Any, *, platform: str) -> bool:
    return any(
        str(action["name"]).lower() == _ANSWER_ACTION
        for action in _step_actions(actions_label, platform=platform)
    )


def _non_answer_actions(actions_label: Any, *, platform: str) -> list[dict[str, Any]]:
    """Executable source actions after terminal answer/status text is separated."""
    return [
        action
        for action in _step_actions(actions_label, platform=platform)
        if str(action["name"]).lower() != _ANSWER_ACTION
    ]


def _append_final_text(message: dict[str, Any], text: str) -> None:
    content = list(message.get("content") or [])
    content.append({"type": "text", "text": text})
    message["content"] = content


def _terminal_outcome(
    actions_label: Any,
    *,
    config: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """What the episode's last step asserted: ``(final_turn_text, others)``.

    GUIAct ends an episode with an ``answer`` action whose ``text`` is either an
    AUTHORED ANSWER or a CANNED TERMINATOR, and the two need opposite handling:

    * **an authored answer** is the point of the task, survives nowhere else
      (``terminate_outcome_others`` maps a success outcome to ``{}``), and so
      becomes the final assistant turn's text. Blank text is not an answer, so
      it falls back to the structural ``Done.`` marker.
    * **a canned terminator** is not the model's answer at all, so the final turn
      is the structural marker and only a NON-success status leaves a trace --
      ``others.terminate_status``, via the same :func:`terminate_outcome_others`
      cagui / guiodyssey / opencua use, which drops ``"success"`` because
      ``Done.`` already asserts exactly that.

    A terminator is only *possible* where the subset declares it
    (:data:`TERMINAL_ANSWER_MAY_BE_CANNED`); within such a subset, an
    unrecognised string is by definition not one of its markers, so it takes the
    authored branch -- the same branch web-multi always takes. The adapter
    therefore never destroys text it cannot classify, and never publishes a
    marker as an answer.
    """
    texts = [
        action.get("text", "")
        for action in _step_actions(actions_label, platform=config["platform"])
        if str(action["name"]).lower() == _ANSWER_ACTION
    ]
    text = next(
        (t for t in reversed(texts) if isinstance(t, str) and t.strip()),
        None,
    )
    if text is None:
        return None, {}
    status = (
        _canned_terminator_status(text)
        if config["terminal_answer"] == TERMINAL_ANSWER_MAY_BE_CANNED
        else None
    )
    if status is None:
        return text, {}
    return None, terminate_outcome_others(
        make_tool_call(TERMINATE_TOOL_NAME, {"status": status})
    )


def episode_to_example(
    episode_id: str,
    steps: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    split: str = "train",
) -> dict[str, Any]:
    """Convert one multi-step episode into a ``use`` row.

    Works for both web-multi (platform=browser, list-valued actions_label, desktop
    action space) and smartphone (platform=mobile, dict-valued actions_label,
    mobile action space).
    """
    image_ext = config["image_ext"]
    image_subdir = config["image_subdir"]
    platform = config["platform"]

    # ``actions_history`` numbers each step, so a gap names the actions the source
    # did not ship. An interior gap and a missing PREFIX are the same defect: a
    # trajectory starting at step 3 is one whose first three actions are absent,
    # and its first screenshot is mid-task while the row presents it as the start.
    logical_steps = [_history_step(step) for step in steps]
    if logical_steps and logical_steps[0] != 0:
        raise SkipTrajectoryError(
            f"trajectory does not start at step 0: history steps {logical_steps}"
        )
    if any(right != left + 1 for left, right in zip(logical_steps, logical_steps[1:])):
        raise SkipTrajectoryError(
            f"missing logical action record: history steps {logical_steps}"
        )

    images: list[str] = []
    for step in steps:
        image_dir = f"{image_subdir}/{split}" if config.get("split_images") else image_subdir
        image_rel = f"yiye2023/GUIAct/images/{image_dir}/{step['image_id']}.{image_ext}"
        try:
            abs_image = resolve_path(image_rel, "CUA_LITE_RAW_DATASETS_ROOT")
        except FileNotFoundError:
            raise SkipTrajectoryError(f"Image not found: {image_rel}")
        images.append(abs_image)

    messages: list[dict[str, Any]] = []
    question = steps[0]["question"]
    answer_text, terminal_others = _terminal_outcome(
        steps[-1]["actions_label"], config=config
    )

    for i, step in enumerate(steps):
        if i == 0:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": question},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "image", "index": i}],
            })

        actions_label = step["actions_label"]
        is_terminal = i == len(steps) - 1
        if not is_terminal and _has_answer_action(actions_label, platform=platform):
            raise SkipTrajectoryError(
                "non-terminal answer emits a standalone response call that the "
                "next screenshot cannot pair with"
            )

        executable_actions = _non_answer_actions(actions_label, platform=platform)
        if not executable_actions:
            continue

        if platform == "mobile":
            tool_calls = convert_mobile_action(
                executable_actions[0],
                is_terminal=is_terminal,
            )
            tool_calls = merge_adjacent_lite_action_batches(tool_calls)
        else:
            tool_calls = convert_web_actions(
                executable_actions,
                is_terminal=is_terminal,
            )

        assistant = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        if is_terminal and answer_text:
            _append_final_text(assistant, answer_text)
        messages.append(assistant)

    if answer_text is None and not any(message.get("tool_calls") for message in messages):
        raise SkipTrajectoryError(
            "no action and no answer; nothing publishable"
        )
    ends_on_action = bool(
        messages
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("tool_calls")
    )
    if not ends_on_action:
        messages.append(
            structural_final_message(answer_text) if answer_text
            else structural_final_message()
        )
    messages = finalize_use_messages(messages)

    img_size = steps[0].get("image_size") or {}
    resolution = [img_size["width"], img_size["height"]] if img_size else None

    scoped_episode_id = f"{split}_{episode_id}" if config.get("split_images") else episode_id
    return {
        "images": images,
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=(platform, "use"),
            extra_tool_schemas=extra_tool_schemas_for_messages(messages),
            valid_actions=None,
            others={
                "id": f"{config['id_prefix']}_{scoped_episode_id}",
                "resolution": resolution,
                "os": config["os"],
                "source": "yiye2023/GUIAct",
                "source_id": f"{config['id_prefix']}_{scoped_episode_id}",
                "split": None,  # set per-file below; stage_entry strips it
                **terminal_others,
            },
        ).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process GUIAct multi-step use data (web-multi + smartphone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--subset", choices=list(SUBSETS.keys()) + ["all"], default="all",
        help="Which subset(s) to process (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Process without writing output")
    parser.add_argument("--verbose", action="store_true", help="Show skip details")
    parser.add_argument("--head", type=int, default=None, help="Process only first N episodes per file")
    args = parser.parse_args()

    raw_root = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not raw_root:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT must be set")
        return 1
    if not args.dry_run and not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT must be set")
        return 1
    raw_root = Path(raw_root)

    subsets = list(SUBSETS.keys()) if args.subset == "all" else [args.subset]


    out_dir = None if args.dry_run else out_dir_for()
    store = None if args.dry_run else make_image_store(out_dir)
    splitter = None if args.dry_run else make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    n_episodes = n_skipped = n_oob = n_corrupt = 0

    for subset in subsets:
        config = SUBSETS[subset]
        for rel_json, split in config["files"]:
            data_path = raw_root / rel_json
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            with open(data_path) as f:
                data = json.load(f)
            episodes = _group_episodes(data, config["uid_re"])
            episode_ids = sorted(episodes)
            if args.head is not None:
                episode_ids = episode_ids[: args.head]
            step_counts = [len(episodes[e]) for e in episode_ids]
            shape = (
                f"steps/ep min={min(step_counts)}, max={max(step_counts)}, "
                f"avg={sum(step_counts) / len(step_counts):.1f}"
                if step_counts else "steps/ep n/a"
            )
            print(
                f"\n{rel_json} (split={split}): {len(data)} steps, "
                f"{len(episode_ids)} episodes ({shape})"
            )

            for idx, eid in enumerate(episode_ids):
                try:
                    entry = episode_to_example(eid, episodes[eid], config=config, split=split)
                except SkipTrajectoryError as e:
                    n_skipped += 1
                    if args.verbose:
                        print(f"  Skipping episode {eid} ({len(episodes[eid])} steps): {e}")
                    continue

                if has_oob_coordinate(entry):
                    n_oob += 1
                    continue

                n_episodes += 1
                if args.dry_run:
                    continue

                entry["metadata"]["others"]["split"] = split
                try:
                    bk, e = stage_entry(
                        entry, store=store, splitter=splitter, variant=VARIANT
                    )
                except staging.CorruptImageError as img_err:
                    n_episodes -= 1
                    n_corrupt += 1
                    if args.verbose:
                        print(f"  Skipping corrupt episode {eid}: {img_err}")
                    continue
                buffers[bk].append(e)

                if (idx + 1) % 1000 == 0:
                    print(f"  Progress: {idx + 1}/{len(episode_ids)} episodes")

    if args.dry_run:
        print(f"\n[dry-run] episodes={n_episodes}, skipped={n_skipped}, oob={n_oob}")
        return 0

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    print(
        f"\nepisodes={n_episodes}, skipped={n_skipped}, oob_dropped={n_oob}, "
        f"corrupt_dropped={n_corrupt}"
    )
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
