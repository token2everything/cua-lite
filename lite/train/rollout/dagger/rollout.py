"""DAgger (online teacher-forcing) rollout for CUA-Lite agents.

The student rolls out in the env (states ~ d_{pi_theta}); each visited step's SFT
*target* is relabeled with a teacher's GENERATED target
(``lite.train.rollout.dagger.teacher.relabel_with_teacher``); training is plain ``sft_loss``
with advantages disabled. The student's executed actions and context history stay
its own — the teacher target is never executed.

Dataflow ~ OPD, with three swaps: teacher score->generate,
KL-to-advantage->relabel-target, policy_loss->sft_loss. Online loop, no buffer by
default (streaming) — the stock ``generate_rollout`` is reused unchanged. Rollout
infrastructure is shared via ``lite.train.rollout.core``; only ``generate`` (binds
the relabel hook) and ``convert_samples_to_train_data`` (SFT-style, no advantage)
differ from GRPO.

Usage:
  --custom-generate-function-path lite.train.rollout.dagger.generate
  --custom-convert-samples-to-train-data-path \
      lite.train.rollout.dagger.convert_samples_to_train_data
  --rollout-function-path lite.train.rollout.dagger.generate_rollout
  --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils.types import Sample

from lite.train.rollout.core import (
    bucket_trajectories,
    dummy_sample,
    flatten_and_align,
    generate_rollout,  # stock streaming driver (reused unchanged)
    get_episode_returns,
    safe_wandb_log,
    synthetic_dummy_sample,
)
from lite.train.rollout.core import (
    generate as _base_generate,
)
from lite.train.rollout.dagger.teacher import dagger_cfg, relabel_with_teacher

# Slime API: scripts/train/run_dagger.sh resolves these public symbols by
# dotted string. ``generate_rollout`` is re-exported from ``core`` and may have
# no in-module reference, so ``__all__`` is the intentional keep-alive surface.
# Keep ``__all__`` and launcher flags in sync when renaming.
__all__ = ["generate", "convert_samples_to_train_data", "generate_rollout"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# generate — student rollout + per-step teacher relabel (skipped at eval)
#
# Slime entrypoint: selected by ``--custom-generate-function-path``. Unlike
# GRPO/REINFORCE, DAgger defines ``generate`` locally to bind the relabel hook.
# ---------------------------------------------------------------------------

async def generate(args: Any, sample: Sample, sampling_params: dict,
                   evaluation: bool = False) -> list[Sample]:
    """Student env rollout, then swap each step's SFT target to the teacher target.

    Binds ``relabel_with_teacher`` as the shared ``generate()`` relabel hook. The hook
    is skipped when ``evaluation`` (eval measures only the student's env success; the
    teacher target is not needed and relabeling there would corrupt eval).
    """
    return await _base_generate(
        args, sample, sampling_params, evaluation=evaluation,
        relabel_fn=relabel_with_teacher,
    )


# ---------------------------------------------------------------------------
# convert_samples_to_train_data — SFT-style (no GRPO advantage)
#
# Slime entrypoint: selected by ``--custom-convert-samples-to-train-data-path``.
# ---------------------------------------------------------------------------

def convert_samples_to_train_data(args: Any, samples) -> dict:
    """Convert teacher-relabeled rollout samples into a training dict for ``sft_loss``.

    DAgger imitates EVERY student-visited state (the off-track ones are the point), so
    by default keep all trajectories — no success filter. ``relabel_with_teacher``
    already cleared steps where the teacher produced no usable target
    (``response_tokens=[]``), but a cleared step still emits a segment whose ``tokens``
    carry the full prompt — so ``_filter_errored_rollouts`` (which keys on ``s.tokens``)
    keeps it and the zero-loss filter below is what actually drops it. Reward
    is irrelevant to ``sft_loss`` (set uniform 1.0 for safety). The optional
    ``dagger.step_filter="success_traj"`` (anti-DAgger) gates on episode return.

    Returns: dict with tokens, response_lengths, rewards, loss_masks, … (same shape as
    GRPO's convert) but with **no advantage normalization** — run with
    ``--disable-compute-advantages-and-returns``.
    """
    if not samples:
        # Empty-convert guard: {} would KeyError in slime's data splitter.
        # Return a valid 1-group zero-gradient no-op instead.
        return flatten_and_align([[synthetic_dummy_sample()]], args)

    rollouts, n_valid, n_errored, n_missing, n_expected = bucket_trajectories(
        samples, args.rollout_batch_size * args.n_samples_per_prompt,
    )
    rollout_templates = list(rollouts)
    pre_filter_episode_returns = get_episode_returns(rollout_templates)
    pre_filter_nonzero_return_rate = (
        sum(1 for r in pre_filter_episode_returns if r > 0)
        / max(len(rollout_templates), 1)
    )
    n_zero_loss_segments = sum(
        1
        for rollout in rollouts
        for sample in rollout
        if not sample.response_length or not sum(sample.loss_mask or [])
    )
    if n_zero_loss_segments:
        trainable_rollouts = [
            [
                sample for sample in rollout
                if sample.response_length and sum(sample.loss_mask or [])
            ]
            for rollout in rollouts
        ]
        rollouts = [rollout for rollout in trainable_rollouts if rollout]
    episode_returns = get_episode_returns(rollouts)

    # DAgger imitates EVERY student-visited step by default (the off-track ones are the
    # point). The only filter is the anti-DAgger ablation: keep success trajectories only.
    step_filter = dagger_cfg(args).get("step_filter", "keep_all")
    if step_filter not in ("keep_all", "success_traj"):
        raise ValueError(
            "dagger.step_filter must be 'keep_all' or 'success_traj', "
            f"got {step_filter!r}"
        )
    if step_filter == "success_traj":
        kept = [r for r, er in zip(rollouts, episode_returns) if er > 0]
    else:  # keep_all: every trajectory (unusable steps were already cleared upstream)
        kept = list(rollouts)

    # Teacher-relabel quality: how many scored steps were dropped because the
    # teacher target was unusable or the teacher/render request failed. Keep
    # the metric name for dashboard continuity; its numerator is
    # ``dagger_n_dropped``.
    seen: dict = {}
    for s in samples:
        md = s.metadata or {}
        if "dagger_n_scored" in md:
            seen[s.group_id] = (md["dagger_n_scored"], md.get("dagger_n_dropped", 0))
    n_steps_scored = sum(a for a, _ in seen.values())
    n_steps_dropped = sum(b for _, b in seen.values())
    # NaN (not 0.0) when nothing was scored: no data should not read as 0%.
    unparseable_rate = (
        n_steps_dropped / n_steps_scored if n_steps_scored else float("nan")
    )

    n_truncated = sum(
        1 for r in rollout_templates if any(s.status == Sample.Status.TRUNCATED for s in r)
    )
    safe_wandb_log({
        # Honest mean reward over all valid rollouts, BEFORE filter + dummy-pad
        # (slime's post-pad ``rollout/raw_reward`` collapses to ≈ the kept fraction).
        "rollout/mean_episode_return": (
            sum(pre_filter_episode_returns) / max(len(rollout_templates), 1)
        ),
        "rollout/nonzero_return_rate": pre_filter_nonzero_return_rate,
        "rollout/n_kept": len(kept),
        "rollout/n_trajs_valid": n_valid,
        "rollout/n_trajs_errored": n_errored,
        "rollout/n_trajs_missing": n_missing,
        "rollout/n_trajs_expected": n_expected,
        "rollout/n_truncated": n_truncated,
        # teacher-relabel coverage
        "rollout/dagger_unparseable_rate": unparseable_rate,
        "rollout/dagger_n_steps_scored": n_steps_scored,
        "rollout/dagger_n_steps_dropped": n_steps_dropped,
        "rollout/dagger_n_zero_loss_segments_dropped": n_zero_loss_segments,
    })

    if not kept:
        # Nothing trainable (all errored / filtered) — dummy samples for a no-op
        # gradient step (loss_mask=[0], reward=0 → zero gradient), one per original
        # rollout to preserve group_index diversity (symmetric with REINFORCE).
        logger.warning("DAgger: no trajectories to train on — using dummy samples")
        kept = [[dummy_sample(r[0])] for r in rollout_templates]
    else:
        for rollout in kept:
            for s in rollout:
                s.reward = 1.0  # sft_loss ignores reward; uniform for safety

    logger.info(
        "DAgger: %d/%d trajectories kept (filter=%s), "
        "nonzero_return_rate=%.0f%%, %d errored; "
        "teacher dropped %d/%d scored steps (%.1f%%)",
        len(kept), len(rollout_templates), step_filter,
        pre_filter_nonzero_return_rate * 100, n_errored,
        n_steps_dropped, n_steps_scored, unparseable_rate * 100,
    )
    return flatten_and_align(kept, args)
