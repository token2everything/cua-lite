"""REINFORCE / filtered-BC rollout for CUA-Lite agents using slime framework.

Online filtered behavior cloning: rollout with current policy, filter to
successful trajectories only, train with SFT loss. Mathematically equivalent
to REINFORCE with binary reward and no baseline.

Rollout infrastructure is shared via lite.train.rollout.core. Only
convert_samples_to_train_data differs from GRPO (filtering instead of group
advantage normalization).

Usage:
  --custom-generate-function-path lite.train.rollout.reinforce.generate
  --custom-convert-samples-to-train-data-path \
      lite.train.rollout.reinforce.convert_samples_to_train_data
  --rollout-function-path lite.train.rollout.reinforce.generate_rollout
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils.types import Sample

from lite.train.rollout.core import (
    bucket_trajectories,
    # Also used by convert_samples_to_train_data:
    dummy_sample,
    flatten_and_align,
    generate,
    generate_rollout,
    get_episode_returns,
    safe_wandb_log,
    synthetic_dummy_sample,
)

# Slime API: scripts/train/run_reinforce.sh resolves these public symbols by
# dotted string. ``generate`` and ``generate_rollout`` are re-exported from
# ``core`` and may have no in-module references, so ``__all__`` is the
# intentional keep-alive surface. Keep ``__all__`` and launcher flags in sync
# when renaming.
__all__ = ["generate", "convert_samples_to_train_data", "generate_rollout"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# convert_samples_to_train_data (REINFORCE / filtered-BC)
#
# Slime entrypoint: selected by ``--custom-convert-samples-to-train-data-path``.
# ---------------------------------------------------------------------------

def convert_samples_to_train_data(args: Any, samples) -> dict:
    """Convert multi-turn rollout samples into a training-data dict for slime.

    REINFORCE-specific: filters to successful trajectories only (episode_return
    >= min_episode_return), then trains with uniform reward=1.0 (SFT loss).

    Args:
        args: Training args (from slime argument parser). Reads
            ``min_episode_return`` from custom config YAML (default 1.0, i.e.
            keep only successful trajectories under binary 0/1 rewards — lower
            the threshold to keep partially-rewarded trajectories).
        samples: Flat ``list[Sample]`` (after slime's chain.from_iterable
            flattening) or nested ``list[list[Sample]]``.

    Returns:
        dict with keys: tokens, response_lengths, rewards, raw_reward,
        truncated, sample_indices, loss_masks, and optionally
        rollout_log_probs and multimodal_train_inputs.
    """
    if not samples:
        # Empty-convert guard: {} would KeyError in slime's data splitter.
        # Return a valid 1-group zero-gradient no-op instead.
        return flatten_and_align([[synthetic_dummy_sample()]], args)

    # 1. Regroup + filter + 4-bucket classification (see bucket_trajectories
    # docstring). ``rollouts`` is the filtered list (may contain dummies when
    # all rollouts failed; n_trajs_valid is still 0 in that case).
    rollouts, n_trajs_valid, n_trajs_errored, n_trajs_missing, n_trajs_expected = (
        bucket_trajectories(
            samples, args.rollout_batch_size * args.n_samples_per_prompt,
        )
    )

    # 2. Per-rollout episode returns
    episode_returns = get_episode_returns(rollouts)

    # 3. Filtered-BC: keep only successful trajectories
    min_episode_return = getattr(args, "min_episode_return", 1.0)
    kept: list[list[Sample]] = []
    for rollout, episode_return in zip(rollouts, episode_returns):
        if episode_return >= min_episode_return:
            kept.append(rollout)

    n_total = len(rollouts)
    n_kept = len(kept)
    n_filtered = n_total - n_kept
    # Symmetric with GRPO / eval: fraction with positive episode return. NOT a true
    # success rate on partial-credit envs (counts r=0.33 etc.), hence ``nonzero_return_rate``.
    # Distinct from `n_kept`, which counts rollouts passing the (possibly
    # looser) ``min_episode_return`` filter used to select BC-training data.
    nonzero_return_rate = sum(1 for r in episode_returns if r > 0) / max(n_total, 1)

    logger.info(
        "REINFORCE: %d/%d kept (filter >=%.2f), nonzero_return_rate=%.0f%%, %d errored",
        n_kept, n_total, min_episode_return, nonzero_return_rate * 100, n_trajs_errored,
    )

    # Per-rollout truncated count (any turn truncated → rollout counts).
    # Use count (not ratio) to avoid collision with slime's per-turn
    # ``rollout/truncated_ratio`` from compute_metrics_from_samples.
    n_truncated = sum(
        1 for r in rollouts
        if any(s.status == Sample.Status.TRUNCATED for s in r)
    )

    safe_wandb_log({
        # Honest mean reward over all (valid) rollouts, BEFORE the success filter +
        # dummy-pad. slime's train-side ``rollout/raw_reward`` is computed post-pad
        # (kept reward=1, dummies force-zeroed) so it collapses to ≈ the kept fraction,
        # not the policy's reward — use this key for the real mean episode return.
        "rollout/mean_episode_return": sum(episode_returns) / max(n_total, 1),
        "rollout/nonzero_return_rate": nonzero_return_rate,
        "rollout/n_kept": n_kept,
        "rollout/n_filtered": n_filtered,
        "rollout/n_trajs_valid": n_trajs_valid,
        "rollout/n_trajs_errored": n_trajs_errored,
        "rollout/n_trajs_missing": n_trajs_missing,
        "rollout/n_trajs_expected": n_trajs_expected,
        "rollout/n_truncated": n_truncated,
    })

    if not kept:
        # 0% success — produce dummy samples for no-op gradient step
        # (loss_mask=[0] + reward=0.0 → zero gradient). Each dummy inherits
        # its ORIGINAL rollout's template (not a single shared one) to preserve
        # group_index diversity; symmetric with _filter_errored_rollouts.
        logger.warning("No successful trajectories — using dummy samples")
        kept = [[dummy_sample(r[0])] for r in rollouts]
    else:
        # 4. Set uniform reward (SFT loss treats all kept samples equally)
        for rollout in kept:
            for s in rollout:
                s.reward = 1.0

    # 5. Flatten, align, assemble
    return flatten_and_align(kept, args)
