"""Shared rollout kernel for CUA-Lite RL/SFT training.

Internal to :mod:`lite.train.rollout` — NOT a slime entry point. The public
surface slime loads by dotted path is the thin per-algorithm shims one level
up (``lite.train.rollout.{grpo,reinforce,dagger,sft}``); those bind their
``generate`` / ``generate_rollout`` / ``convert_samples_to_train_data`` from
here. This ``__init__`` is the stable internal API the shims import, so the
split between :mod:`.engine` (async env→samples driver + batch assembly) and
:mod:`.segmenter` (pure steps→Sample-span packing) can move freely without
touching the shims or their slime config paths.

White-box tests reach into ``.engine`` / ``.segmenter`` directly (they test and
patch internals); only shim-facing symbols are re-exported below.
"""

from __future__ import annotations

from lite.train.rollout.core.engine import (
    bucket_trajectories,
    dummy_sample,
    flatten_and_align,
    generate,
    generate_rollout,
    get_episode_returns,
    safe_wandb_log,
    synthetic_dummy_sample,
)
from lite.train.rollout.core.segmenter import build_segment_samples

__all__ = [
    "build_segment_samples",
    "dummy_sample",
    "synthetic_dummy_sample",
    "bucket_trajectories",
    "flatten_and_align",
    "generate",
    "generate_rollout",
    "get_episode_returns",
    "safe_wandb_log",
]
