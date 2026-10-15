"""GRPO rollout for CUA-Lite agents using slime framework.

Provides GRPO-specific convert_samples_to_train_data (group advantage
normalization). Rollout infrastructure is shared via lite.train.rollout.core.

Usage:
  --custom-generate-function-path lite.train.rollout.grpo.generate
  --custom-convert-samples-to-train-data-path lite.train.rollout.grpo.convert_samples_to_train_data
  --rollout-function-path lite.train.rollout.grpo.generate_rollout
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from slime.utils.types import Sample

from lite.train.rollout.core import (
    bucket_trajectories,
    # Used by convert_samples_to_train_data:
    dummy_sample,
    flatten_and_align,
    generate,
    generate_rollout,
    get_episode_returns,
    safe_wandb_log,
    synthetic_dummy_sample,
)

# Slime API: scripts/train/run_grpo.sh and run_gspo.sh resolve these public
# symbols by dotted string. ``generate`` and ``generate_rollout`` are
# re-exported from ``core`` and may have no in-module references, so ``__all__``
# is the intentional keep-alive surface. Keep ``__all__`` and launcher flags in
# sync when renaming.
__all__ = ["generate", "convert_samples_to_train_data", "generate_rollout"]

logger = logging.getLogger(__name__)

# One-shot guard: log incompatible DROP_ZERO_STD_GROUP configs once per process.
_DROP_SKIP_WARNED = False

# ---------------------------------------------------------------------------
# convert_samples_to_train_data (GRPO-specific)
#
# Slime entrypoint: selected by ``--custom-convert-samples-to-train-data-path``.
# ---------------------------------------------------------------------------

def convert_samples_to_train_data(args: Any, samples) -> dict:
    """Convert multi-turn rollout samples into a training-data dict for slime.

    Upstream rollout builds the same generic ``LiteRLStep`` contract used by
    SFT: rendered ``prompt``, ordered ``image_indices``, ``response`` text, and
    ``response_tokens``. The shared segmenter converts those steps to
    ``Sample.tokens`` / ``response_length`` / ``loss_mask`` before this GRPO
    converter applies group advantages.

    GRPO-specific: computes group advantages (reward - group_mean) / group_std.

    This function handles three concerns that the default slime converter doesn't:

    1. **Multi-turn segmentation**: Each trajectory (rollout) has one or more
       training segments. Prefix-extensional turns can be radix-packed into a
       multi-span segment; non-extensional turns stay as length-1 segments.
       Every segment from the same rollout shares the same advantage, computed
       from the per-rollout episode return.

    2. **Errored rollout handling**: Env failures (timeouts, connection errors,
       setup crashes) produce samples with ``tokens=[]``. These are discarded
       BEFORE advantage normalization so they don't pollute the GRPO baseline.
       This means a prompt group may have fewer than ``n_samples_per_prompt``
       rollouts — the advantage computation groups by ``group_index`` to handle
       variable group sizes correctly.

    3. **Batch-size alignment**: Slime trims sample counts to a multiple of
       ``global_batch_size`` BEFORE calling this function, but our filtering
       can reduce the count below that alignment. We re-trim (or pad with
       zero-gradient dummy samples) to restore alignment, preventing the
       ``AssertionError`` in Megatron's dynamic-batch-size data partitioner.

    Args:
        args: Training args (from slime argument parser).
        samples: Flat ``list[Sample]`` (after slime's chain.from_iterable
            flattening) or nested ``list[list[Sample]]``.

    Returns:
        dict with keys: tokens, response_lengths, rewards, raw_reward,
        truncated, sample_indices, loss_masks, and optionally
        rollout_log_probs and multimodal_train_inputs.
    """
    if not samples:
        # Empty-convert guard: returning {} would KeyError in slime's
        # _split_train_data_by_dp (data["group_ids"]/["tokens"]). Return a valid
        # 1-group zero-gradient no-op train_data instead.
        return flatten_and_align([[synthetic_dummy_sample()]], args)

    # The per-trajectory normalization (group_mask_sums emitted by
    # flatten_and_align) only takes effect on slime's sum_of_sample_mean reducer.
    # With --calculate-per-token-loss the reducer becomes sum_of_token
    # (slime cp_utils.py), which IGNORES group_mask_sums → silent revert to a
    # per-token loss and the §3 packing-invariant per-trajectory weighting is
    # lost. Fail fast rather than train with a silently-wrong objective.
    if getattr(args, "calculate_per_token_loss", False):
        raise ValueError(
            "GRPO requires calculate_per_token_loss=False: the per-trajectory "
            "normalization rides on group_mask_sums via sum_of_sample_mean; "
            "--calculate-per-token-loss switches to sum_of_token which ignores it."
        )

    # 1. Regroup + filter + 4-bucket classification (see bucket_trajectories
    # docstring). ``rollouts`` is the filtered list (may contain dummies when
    # all rollouts failed; n_trajs_valid is still 0 in that case).
    rollouts, n_trajs_valid, n_trajs_errored, n_trajs_missing, n_trajs_expected = (
        bucket_trajectories(
            samples, args.rollout_batch_size * args.n_samples_per_prompt,
        )
    )

    # 2. Per-rollout episode returns (one per rollout)
    episode_returns = get_episode_returns(rollouts)

    # 3. GRPO advantage normalization
    if (
        getattr(args, "advantage_estimator", None)
        in ("grpo", "gspo", "reinforce_plus_plus_baseline")
        and getattr(args, "rewards_normalization", True)
    ):
        group_map: dict[int, list[int]] = {}
        for i, rollout in enumerate(rollouts):
            group_map.setdefault(rollout[0].group_index, []).append(i)

        advantages: list[float] = [0.0] * len(rollouts)
        do_std = (
            getattr(args, "advantage_estimator", None) in ("grpo", "gspo")
            and getattr(args, "grpo_std_normalization", True)
        )

        for indices in group_map.values():
            rewards = torch.tensor(
                [episode_returns[i] for i in indices], dtype=torch.float
            )
            adv = rewards - rewards.mean()
            if do_std and len(indices) > 1:
                # Bessel-corrected std (torch default, correction=1) — aligned
                # with slime's own ``_post_process_rewards`` and verl's GRPO
                # reference. Using ``correction=0`` here would diverge by
                # √(N/(N-1)) on the advantage magnitude (√2 for N=2).
                adv = adv / (adv.std() + 1e-6)
            for j, idx in enumerate(indices):
                advantages[idx] = adv[j].item()
        # Log GRPO diagnostics
        n_groups = len(group_map)
        # Expected group count = one group per prompt. Missing groups
        # (``n_groups_expected - n_groups``) have every member dropped to
        # n_trajs_missing + n_trajs_errored.
        n_groups_expected = args.rollout_batch_size
        # Groups whose trajectories don't all share one reward. Zero-std groups
        # (the complement) contribute no GRPO gradient — reused below for the
        # optional zero-std → dummy compute saving.
        mixed_groups = [
            idxs for idxs in group_map.values()
            if len(set(episode_returns[i] for i in idxs)) > 1
        ]
        n_mixed = len(mixed_groups)
        n_nonzero_adv = sum(1 for a in advantages if abs(a) > 1e-6)
        mixed_ratio = n_mixed / max(n_groups, 1)
        # Fraction with positive return — NOT a true success rate on partial-credit
        # envs (counts r=0.33 etc.), hence ``nonzero_return_rate``.
        nonzero_return_rate = sum(1 for r in episode_returns if r > 0) / max(
            len(episode_returns), 1
        )
        # Per-rollout truncated count (any turn truncated → rollout counts).
        # Use a count (not ratio) to avoid collision with slime's per-turn
        # ``rollout/truncated_ratio`` (slime/ray/rollout.py:1210) which slime
        # emits from compute_metrics_from_samples and would overwrite us.
        # Symmetric with eval/<ds>/n_truncated in _eval_rollout.
        n_truncated = sum(
            1 for r in rollouts
            if any(s.status == Sample.Status.TRUNCATED for s in r)
        )
        logger.info(
            "GRPO: %d groups, %d mixed (%.0f%%), %d/%d non-zero advantages, "
            "nonzero_return_rate=%.2f",
            n_groups, n_mixed, mixed_ratio * 100, n_nonzero_adv,
            len(advantages), nonzero_return_rate,
        )
        safe_wandb_log({
            "rollout/mixed_group_ratio": mixed_ratio,
            "rollout/n_groups": n_groups,
            "rollout/n_groups_expected": n_groups_expected,
            "rollout/n_nonzero_advantages": n_nonzero_adv,
            "rollout/nonzero_return_rate": nonzero_return_rate,
            "rollout/n_trajs_valid": n_trajs_valid,
            "rollout/n_trajs_errored": n_trajs_errored,
            "rollout/n_trajs_missing": n_trajs_missing,
            "rollout/n_trajs_expected": n_trajs_expected,
            "rollout/n_truncated": n_truncated,
        })

        # DROP_ZERO_STD_GROUP replaces zero-advantage groups with zero-loss
        # dummies to skip dead compute while preserving raw episode-return
        # metrics. This is gradient-identical only for grpo/gspo with nsp>1 and
        # entropy/KL/advantage whitening off; otherwise keep full compute and
        # warn once. Set DROP_ZERO_STD_GROUP=0 to force off.
        if os.environ.get("DROP_ZERO_STD_GROUP", "1") == "1":
            drop_safe = (
                getattr(args, "advantage_estimator", None) in ("grpo", "gspo")
                and args.n_samples_per_prompt > 1
                and getattr(args, "entropy_coef", 0) == 0
                and not getattr(args, "use_kl_loss", False)
                and not getattr(args, "normalize_advantages", False)
            )
            if drop_safe:
                mixed_positions = {i for idxs in mixed_groups for i in idxs}
                rollouts = [
                    rollout if i in mixed_positions else [dummy_sample(rollout[0])]
                    for i, rollout in enumerate(rollouts)
                ]
            else:
                global _DROP_SKIP_WARNED
                if not _DROP_SKIP_WARNED:
                    _DROP_SKIP_WARNED = True
                    logger.warning(
                        "DROP_ZERO_STD_GROUP on but config not gradient-identical-safe "
                        "(needs grpo/gspo + nsp>1 + entropy/KL/advantage-norm off) — "
                        "skipping zero-std drop (full compute, gradient unchanged)."
                    )
    else:
        advantages = episode_returns

    # 4. Propagate the RAW per-trajectory advantage to every segment's .reward.
    # Per-trajectory weighting (so a 40-turn failure can't dominate a 3-turn
    # success) is now done slime-natively by the loss reducer dividing each
    # segment by ``group_mask_sums`` (the trajectory's total loss mask, emitted
    # by flatten_and_align) — an exactly packing-invariant, length-independent
    # token-weighted mean per trajectory. This supersedes the old
    # ``adv / n_turns`` hand-normalization.
    #
    # Pairing invariant: ``reward = adv`` here requires ``flatten_and_align`` to
    # emit ``group_mask_sums``. Changing only one side double-normalizes or
    # un-normalizes the gradient.
    for rollout, adv in zip(rollouts, advantages):
        for s in rollout:
            s.reward = adv

    # 5. Flatten, align, assemble
    return flatten_and_align(rollouts, args)
