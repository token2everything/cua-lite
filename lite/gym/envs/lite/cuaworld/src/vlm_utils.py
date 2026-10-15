"""Upstream-compatible top-level ``vlm_utils`` wrapper for CUAWorld verifiers."""

from __future__ import annotations

from typing import Any

from lite.gym.envs.lite.cuaworld.src.vlm import get_final_screenshot, query_vlm
from lite.gym.envs.lite.cuaworld.src.vlm import (
    sample_trajectory_frames as _sample_trajectory_frames,
)

# `query_vlm` and `get_final_screenshot` are re-exported AS THEY ARE, never wrapped.
# A wrapper here can only ever subtract: re-declaring the signature dropped the
# `**kwargs` that absorbs `return_json`/`output_schema` (a TypeError inside a verifier
# is a hard 0), and it defeated the inverted-positional swap `vlm.py` exists to
# provide — upstream's `query_vlm(frames, prompt)` would bind frames->prompt and
# prompt->image, and the swap can no longer fire because `images` arrives as None.
# Only `sample_trajectory_frames` is wrapped, because it genuinely ADAPTS an
# upstream keyword (`n` -> `num_samples`).


def sample_trajectory_frames(
    traj: dict[str, Any],
    n: int = 3,
    **kwargs: Any,
) -> list[str]:
    return _sample_trajectory_frames(
        traj,
        num_samples=int(kwargs.pop("num_samples", n)),
        include_first=bool(kwargs.pop("include_first", True)),
        include_last=bool(kwargs.pop("include_last", True)),
    )


__all__ = ["get_final_screenshot", "query_vlm", "sample_trajectory_frames"]
