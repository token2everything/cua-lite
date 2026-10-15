"""Unstage a canonical cua-lite dataset back into a rollout log-root.

Inverse of :mod:`lite.data.hf.stage`. Turns a canonical dataset (the layout
``lite.data.hf.stage`` / the preproc adapters produce, and ``lite.data.hf.download``
restores)::

    <dataset>/
      images/<hash[:2]>/<hash>.<ext>            # content-addressed image store
      <platform>/<task_type>/<split>/<variant>.parquet    # rows: images(rel), messages, metadata

back into a rollout log-root that ``scripts/rollout.py`` recognises::

    <log_root>/<split>/<task_id>/sample_<NN>/
        summary.json        # the RESUME GATE — its mere presence => skip this sample
        trajectory.parquet  # images(abs)/messages/metadata, so a later hf.stage
                            #   re-absorbs OLD + NEW into one dataset

So you can **continue collecting on top of a published run**: download the dataset,
unstage it, point ``scripts/rollout.py --log-root <out>`` at it, and rollout's resume
re-runs ONLY the not-yet-collected ``(task, sample)`` pairs (``get_pending`` keys
off ``summary.json`` presence). Re-stage the grown log-root to publish OLD + NEW.

``--splits`` must match the split you will resume with: rollout's resume looks for
samples under ``<log_root>/<registry-split>/`` (``task_to_split`` from the env
registry), so a collect-on-``train`` run unstages with ``--splits train``. Every
canonical row lands under this one split dir (the canonical ``train``/``validation``
parquet split is a *dataset* carve for SFT, unrelated to the env task split).
The split output must be fresh by default; pass ``--overwrite`` only when replacing
an old reconstruction intentionally. This prevents stale sample dirs from surviving
and being interpreted as completed rollouts.

``--config-names`` (symmetric with ``stage``'s) reads ONLY the named configs' parquet.
A ONE-repo, MULTI-config dataset (e.g. ``desktop.use.rl`` + ``desktop.use.train``)
must land each config in its OWN registry-split dir — so unstage it once per config,
pairing ``--config-names desktop.use.rl --splits rl`` then ``--config-names
desktop.use.train --splits train``. Omit to read every config into the one ``--splits``
dir (fine for a single-config dataset).

The dataset's own ``train``/``validation`` carve survives the detour: each row
records the partition it came from as ``metadata.others.split``, the transient
routing hint ``lite.data.hf.stage`` prefers over a fresh ``hash_split`` (and pops
before writing).

A ``grounding.*`` / ``understanding`` row that is a single-turn SFT LABEL has no
episode, so there is nothing to resume collecting on it and a ``summary.json`` for
it could only carry a fabricated ``episode_return``. Such rows are counted and
skipped; a mixed dataset still unstages its ``use`` part. The skip keys off the
absence of ``metadata.others.episode_return``, not off ``task_type`` alone: a rollout
against a single-step grounding **env** (``screenspot_pro`` / ``osworld_g``)
produces ``grounding.point`` rows whose ``others.episode_return`` is really scored,
and dropping those would lose the outcome of every task in the run — measured on
``.logs/famval/{osworld_g,screenspot_pro}``, stage accepted 64/64 rows and
unstage rebuilt 0 sample dirs before this qualifier.

Lossy vs the original rollout, none of it load-bearing for resume or re-stage:
  * ``duration_seconds`` + per-turn timing are defaulted.
  * turn-level debug artifacts are never staged.

``sample_idx`` is synthesised by enumerating rows per ``task_id`` (the canonical
dataset does not preserve the original per-sample index; only the count matters for
resume).

Run:
    uv run python -m lite.data.hf.unstage \
        --dataset "${CUA_LITE_DATASETS_ROOT}/cua-lite/WebGymGPT" \
        --log-root .logs/rollout/webgymgpt_resumed --splits train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lite.agents.core.agent.logger import build_trajectory_summary
from lite.core.messages import peel_system_message
from lite.core.messages.turns import turn_spans
from lite.core.metadata import SINGLE_TURN_TASK_TYPES, LiteCUAMetadata, metadata_from_dict
from lite.data.staging import (
    coerce_messages,
    coerce_meta,
    prepare_output_dir,
    split_of_partition_file,
    to_plain,
)
from lite.utils.parquet import write_records_to_parquet


def _image_root(dataset: Path) -> Path:
    """The dir canonical image rel-paths resolve against.

    Rows store images as ``cua-lite/<name>/images/<hash[:2]>/<hash>.<ext>`` —
    relative to ``$CUA_LITE_DATASETS_ROOT`` (the dir ABOVE ``cua-lite/``), which is
    ``<dataset>/../..`` for the canonical ``<root>/cua-lite/<name>`` layout. Same
    contract ``export_sft --image-root`` uses.
    """
    return dataset.parent.parent


def _matches_config(p: Path, names: set[str], dataset: Path) -> bool:
    """True if partition parquet *p* belongs to one of the given config names.

    A config name is the ``--config-names`` label ``lite.data.hf.stage`` wrote it
    under; ``download`` restores that as the parquet stem (``desktop.use.rl.parquet``)
    or, for a sharded snapshot, the variant dir (``desktop.use.rl/shard-*.parquet``).
    Match either — an exact component/stem test against the path *inside* the
    dataset only (never its ancestors: a dataset stored under a dir named like a
    config must not make every parquet match), so ``desktop.use.rl`` never
    matches ``desktop.use.train``.
    """
    stem = p.name[: -len(".parquet")]
    return stem in names or any(part in names for part in p.relative_to(dataset).parts)


def _partition_split(p: Path, dataset: Path) -> str:
    """The canonical ``train``/``validation`` label partition parquet *p* lives under.

    The label is *only* in the path -- the rows carry no split marker -- so reading
    it here is what lets the reconstructed log-root hand the carve back to
    ``lite.data.hf.stage`` instead of having it re-drawn from scratch.
    """
    split = split_of_partition_file(p.relative_to(dataset))
    if split is None:
        raise SystemExit(
            f"{p} is not a canonical <platform>/<task_type>/<split>[/<variant>].parquet "
            f"partition under {dataset}; its split cannot be recovered"
        )
    return split


def unstage(
    dataset: Path,
    *,
    log_root: Path,
    splits: str,
    config_names: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    dataset = dataset.resolve()
    split_root = prepare_output_dir(
        log_root / splits,
        overwrite=overwrite,
        label="unstage split output directory",
        protected_roots=(dataset,),
    )
    image_root = _image_root(dataset)
    # All partition parquets (skip the CA image store, which holds no parquet).
    parquets = sorted(p for p in dataset.rglob("*.parquet") if "images" not in p.parts)
    if config_names:
        # Symmetric with stage's --config-names: read ONLY the named configs' parquet.
        # A multi-config dataset (e.g. desktop.use.rl + desktop.use.train in ONE repo)
        # must unstage each config into its OWN registry-split dir, so filter here —
        # otherwise every config's rows pile into the single --splits subdir and only
        # that split resumes correctly. Fail loud on a name that matches nothing (typo).
        wanted = set(config_names)
        parquets = [p for p in parquets if _matches_config(p, wanted, dataset)]
        if not parquets:
            # List config names, not shard-file stems: a sharded snapshot stores
            # <config>/shard-*.parquet, so the parent dir is the matchable name.
            available = sorted(
                {
                    q.parent.name if q.name.startswith("shard-") else q.name[: -len(".parquet")]
                    for q in dataset.rglob("*.parquet")
                    if "images" not in q.parts
                }
            )
            raise SystemExit(
                f"--config-names {config_names} matched no parquet under {dataset} "
                f"(available: {available})"
            )
    if not parquets:
        raise SystemExit(f"no <platform>/<task_type>/<split>/<variant>.parquet under {dataset}")

    # First pass groups rows per task_id (across all partitions) so sample_idx is a
    # stable 0..N-1 enumeration; the canonical dataset does not carry the original
    # index. A row is (task_id, metadata, images, messages).
    by_task: dict[str, list[dict]] = {}
    n_rows = n_noimg = n_single_turn = 0
    for pq in parquets:
        # The dataset's own train/validation carve, recorded per row as the
        # transient ``others["split"]`` routing hint ``lite.data.hf.stage``
        # consumes and pops.
        source_split = _partition_split(pq, dataset)
        df = pd.read_parquet(pq)
        for _, row in df.iterrows():
            n_rows += 1
            lite_meta = metadata_from_dict(coerce_meta(row["metadata"]))
            if not isinstance(lite_meta, LiteCUAMetadata):
                raise ValueError(
                    f"{pq}: lite.data.hf.unstage only supports CUA metadata; "
                    "generic publish layout is not supported by this plan"
                )
            md = lite_meta.to_dict()
            others = md["others"]
            others["split"] = source_split
            task_id = others.get("task_id")
            if not task_id:
                raise ValueError(f"{pq}: metadata.others.task_id is required")
            task_id = str(task_id)
            # A single-turn LABEL cohort has no episode to reconstruct, and the
            # whole point of a log-root is resuming COLLECTION on it. Writing one
            # would stamp a fabricated ``episode_return`` on a row that has no
            # scoring concept, behind the resume gate's authority.
            #
            # ``task_type`` alone does not say that, because ``screenspot_pro`` /
            # ``osworld_g`` are single-step ``grounding.point`` **envs**: a
            # rollout row from one carries a REAL scored ``metadata.others.episode_return``
            # (``lite/agents/core/agent/logger.py`` writes it ungated by
            # task_type) and there IS something to resume collecting on it. The
            # licence test is therefore the presence of that field, which is what
            # "fabricated" means here -- not the cohort name.
            if (
                isinstance(lite_meta, LiteCUAMetadata)
                and lite_meta.task_type in SINGLE_TURN_TASK_TYPES
                and "episode_return" not in others
            ):
                n_single_turn += 1
                continue
            imgs = to_plain(row["images"]) or []
            if not imgs:
                n_noimg += 1
            by_task.setdefault(task_id, []).append(
                {"md": md, "imgs": imgs, "msgs": coerce_messages(row["messages"])}
            )
        del df

    # Pre-flight: catch a mis-pointed --dataset early. Image rel-paths resolve
    # against _image_root (the dir ABOVE cua-lite/); if --dataset isn't the
    # canonical <root>/cua-lite/<name> shape that derivation is wrong and every
    # reconstructed image path would dangle silently. Fail loud on the first image.
    probe_rel = next((r["imgs"][0] for rows in by_task.values() for r in rows if r["imgs"]), None)
    if probe_rel is not None and not (image_root / probe_rel).resolve().exists():
        raise SystemExit(
            f"image {probe_rel!r} → {(image_root / probe_rel).resolve()} does not exist; "
            f"is --dataset the canonical <root>/cua-lite/<name> dir? (got {dataset})"
        )

    n_tasks = len(by_task)
    n_samples = 0
    for task_id, rows in sorted(by_task.items()):
        for sample_idx, r in enumerate(rows):
            sample_dir = split_root / task_id / f"sample_{sample_idx:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # trajectory.parquet — same 3-column schema the TrajectoryLogger writes.
            # Resolve CA-store rel-paths to ABSOLUTE so a later hf.stage (which
            # resolves relative paths against its cwd) finds the files regardless of
            # where it runs; hf.stage re-ingests them idempotently (content hash).
            abs_imgs = [str((image_root / p).resolve()) for p in r["imgs"]]
            write_records_to_parquet(
                [{"images": abs_imgs, "messages": r["msgs"], "metadata": r["md"]}],
                sample_dir / "trajectory.parquet",
                json_fields=("messages", "metadata"),
            )

            # summary.json — the resume gate, built through ``build_trajectory_summary`` so this
            # writer cannot drift from the logger / rollout ones. Outcomes come from
            # metadata.others; duration is unrecoverable → 0.0.
            #
            # ``n_turns`` is counted off the MESSAGES, the same fact
            # ``count_sample_turns`` gives ``adapter.unroll`` and therefore the same
            # number the logger writes as ``len(steps)``. Counting images instead
            # would be a second vocabulary for one fact and would drift two ways: a
            # batch of N actions stores N frames for ONE turn, and a staged rollout
            # keeps frames no message references.
            _, _turn_content = peel_system_message(r["msgs"])
            others = r["md"]["others"]
            summary = build_trajectory_summary(
                n_turns=len(turn_spans(_turn_content)),
                episode_return=float(others.get("episode_return", 0.0) or 0.0),
                terminated=bool(others.get("terminated", False)),
                truncated=bool(others.get("truncated", False)),
                duration_seconds=0.0,
            )
            (sample_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            n_samples += 1

    print(
        f"unstaged {n_rows} rows → {n_tasks} tasks × samples = {n_samples} sample dirs "
        f"under {split_root}/ (rows_with_no_image={n_noimg}, "
        f"single_turn_rows_skipped={n_single_turn})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, type=Path,
                    help="canonical dataset dir (e.g. $CUA_LITE_DATASETS_ROOT/cua-lite/<Name>)")
    ap.add_argument("--log-root", required=True, type=Path,
                    help="output rollout log-root to reconstruct (point scripts/rollout.py here)")
    ap.add_argument("--splits", default="train",
                    help="split subdir to write under — MUST match the --splits you resume "
                         "scripts/rollout.py with (the env-registry split, default: train)")
    ap.add_argument("--config-names", nargs="+", default=None,
                    help="symmetric with stage's --config-names: read ONLY these configs' "
                         "parquet (e.g. 'desktop.use.rl'). Use to unstage ONE config of a "
                         "multi-config dataset into its own --splits dir; omit to read all.")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing non-empty <log-root>/<splits> directory. Without this, "
             "unstage requires a fresh split output so stale sample dirs cannot survive.",
    )
    args = ap.parse_args()
    unstage(args.dataset, log_root=args.log_root, splits=args.splits,
            config_names=args.config_names, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
