"""Shared staging helpers for the Multimodal-Mind2Web adapter.

Single task type (``use``), so this just wires the canonical
:mod:`lite.data.staging` actions: content-addressed image store, hash-split
assigner, per-row staging helper, OOB check.
"""

from __future__ import annotations

from lite.data.preproc.common import SourceStaging

DATASET_NAME = "Multimodal-Mind2Web"
SOURCE = "osunlp/Multimodal-Mind2Web"
RAW_PREFIX = "osunlp/Multimodal-Mind2Web"

_STAGING = SourceStaging(DATASET_NAME)
out_dir_for = _STAGING.out_dir_for
make_image_store = _STAGING.make_image_store
make_splitter = _STAGING.make_splitter
stage_entry = _STAGING.stage_entry
