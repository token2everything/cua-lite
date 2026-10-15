"""Shared helpers for the GUIOdyssey (``hflqf88888/GUIOdyssey``) preproc adapters.

Used by both ``use.py`` and ``understanding.py``: episode iteration,
per-step coordinate normalization, screenshot-path construction, and the
standard :mod:`lite.data.staging` wiring (content-addressed image store, split
assigner, per-row staging helper).

GUIOdyssey is a long-horizon cross-app Android dataset — every row is
``platform=mobile`` / ``os=android`` / ``source=hflqf88888/GUIOdyssey``, so
those dataset-wide constants live here too.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from glob import glob
from typing import Any

from lite.data.preproc.common import SourceStaging

DATASET_NAME = "GUIOdyssey"
SOURCE = "hflqf88888/GUIOdyssey"
PLATFORM = "mobile"
OS = "android"

# Raw layout (relative to CUA_LITE_RAW_DATASETS_ROOT).
RAW_PREFIX = "hflqf88888/GUIOdyssey"
ANNOTATIONS_REL = f"{RAW_PREFIX}/annotations"
SCREENSHOTS_REL = f"{RAW_PREFIX}/screenshots"


class SkipEpisodeError(Exception):
    """Raised when a whole episode (use trajectory) should be skipped."""


_STAGING = SourceStaging(DATASET_NAME)
out_dir_for = _STAGING.out_dir_for
make_image_store = _STAGING.make_image_store
make_splitter = _STAGING.make_splitter
stage_entry = _STAGING.stage_entry


def iter_episodes(raw_root: str, head: int | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (episode_id, episode_dict) in sorted filename order.

    Raises ``FileNotFoundError`` if the annotations glob matches zero files (a
    mis-configured raw root). Malformed annotations fail loud with their path;
    they must not disappear before either task's run accounting.
    """
    pattern = os.path.join(raw_root, ANNOTATIONS_REL, "*.json")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No annotation files matched glob {pattern!r}. "
            "Check that CUA_LITE_RAW_DATASETS_ROOT points at the decoded GUIOdyssey dataset."
        )
    if head is not None:
        files = files[:head]
    for path in files:
        ep_id = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, encoding="utf-8") as f:
                episode = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid GUIOdyssey episode JSON: {path}") from e
        if not isinstance(episode, dict):
            raise ValueError(
                f"Expected object in GUIOdyssey episode {path}, got {type(episode).__name__}"
            )
        yield ep_id, episode


def normalize_xy(x: float, y: float) -> list[int]:
    """GUIOdyssey ``info`` coordinate (x, y) → [0, 1000] int.

    Per the upstream README, *"all the coordinates are normalized to the range
    of [0, 1000]"* already — they are NOT device pixels. So we just round to
    int (no division by ``device_info.w/h``: doing that, as a naive adapter
    would, scales every tap by ``1000/device_dim`` and displaces it). The rare
    genuine out-of-range value is caught by ``has_oob_coordinate`` and drops the
    whole trajectory, per the ``use`` contract in the parent ``AGENTS.md``.
    """
    return [int(round(x)), int(round(y))]


def screenshot_rel_path(ep_id: str, step_idx: int) -> str:
    """Relative path (from raw root) to a screenshot: ``screenshots/{ep}_{step}.png``."""
    return f"{SCREENSHOTS_REL}/{ep_id}_{step_idx}.png"
