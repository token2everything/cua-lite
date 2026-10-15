"""Dynamic sampling filters for multi-turn CUA-Lite rollouts.

Usage:
  --dynamic-sampling-filter-path lite.train.filters.check_reward_nonzero_std
  --dynamic-sampling-filter-path lite.train.filters.drop_capacity_failed_groups
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch
from slime.rollout.filter_hub.base_types import DynamicFilterOutput

logger = logging.getLogger(__name__)


def check_reward_nonzero_std(args, group, **kwargs):
    """Drop prompt groups where all trajectories have the same reward.

    Handles both multi-turn (group: list[list[Sample]]) and single-turn
    (group: list[Sample]) formats.
    """
    rewards = []
    for item in group:
        if isinstance(item, list):
            # Multi-turn: item is a trajectory (list[Sample])
            metadata = item[-1].metadata if item else {}
            episode_return = (metadata.get("others") or {}).get("episode_return")
            if episode_return is not None:
                rewards.append(float(episode_return))
            else:
                rewards.append(0.0)
        else:
            # Single-turn: item is a Sample
            rewards.append(float(item.get_reward_value(args)))
    keep = len(rewards) > 1 and torch.tensor(rewards, dtype=torch.float).std() > 0.0
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1) if rewards else 0}",
    )


# ---------------------------------------------------------------------------
# Capacity-failure group redraw
# ---------------------------------------------------------------------------
#
# When a GRPO group has ≥ K siblings whose env failed with a 503-shaped
# capacity error (typed CapacityExhausted raised by the env-server's
# bounded-pool fast-fail), drop the whole group. The outer over-sample
# loop in :mod:`lite.train.rollout.core.engine` (the
# ``while len(data) < target_data_size`` block) draws a fresh group
# automatically — no new retry machinery.
#
# Other failure categories (``task_crash``, ``reset_timeout``,
# ``step_timeout``, ``aborted``) are NOT redrawn — they reflect real env
# bugs / agent misbehavior that re-running would mask. Only transient
# backpressure (capacity) earns a redraw.
#
# Env-var knob:
#     CUA_LITE_CAPACITY_REDRAW_K (default 1, clamped to ≥1)
#         Group is dropped when ≥ K siblings have capacity_503. K=1 is
#         aggressive (any capacity failure triggers redraw); raise if
#         capacity pressure is sustained and constant redraws would
#         starve the loop.

#: Stable failure-reason values that the redraw filter treats as drop
#: triggers. Sourced from
#: :func:`lite.train.rollout.core.engine._classify_failure`. Wider than the
#: typed ``CapacityExhausted`` only — we accept any 503-shaped error
#: that survived classification, defensive against an intermediate
#: wrapper unwinding the type.
_CAPACITY_REASONS: frozenset[str] = frozenset({"capacity_503"})


def drop_capacity_failed_groups(
    _args: Any,
    group: list[Any],
    **_kwargs: Any,
) -> DynamicFilterOutput:
    """Drop groups dominated by capacity-503 failures so the over-sampler
    redraws them with fresh prompts.

    See module-level discussion above for the contract. Handles both
    multi-turn (group is ``list[list[Sample]]``) and single-turn
    (``list[Sample]``) layouts the same way: look at each item's
    metadata for the typed failure reason.
    """
    # Clamp K to ≥1: K=0 would drop every group (including those with
    # zero capacity failures), starving the over-sampling loop. If the
    # operator wants a fully aggressive policy, set K=1.
    k = max(1, int(os.environ.get("CUA_LITE_CAPACITY_REDRAW_K", "1")))
    n_capacity = _count_capacity_failed(group)
    if n_capacity >= k:
        logger.info(
            "capacity_redraw: dropping group of %d (capacity_failed=%d ≥ K=%d)",
            len(group), n_capacity, k,
        )
        return DynamicFilterOutput(keep=False, reason="capacity_redraw")
    return DynamicFilterOutput(keep=True)


def _count_capacity_failed(group: list[Any]) -> int:
    """Count siblings whose terminal sample's metadata flags capacity_503.

    Defensive about ``metadata`` being a non-dict or absent (it has been
    both at various points in slime's pipeline) and about multi-turn
    items being ``list[Sample]`` rather than a single ``Sample``.
    """
    n = 0
    for item in group:
        sample = item[-1] if isinstance(item, list) and item else item
        md = getattr(sample, "metadata", None)
        if not isinstance(md, dict):
            continue
        if md.get("lite_failure_reason") in _CAPACITY_REASONS:
            n += 1
    return n
