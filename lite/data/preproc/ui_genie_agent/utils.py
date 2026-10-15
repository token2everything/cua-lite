"""Shared helpers for the UI-Genie-Agent (``HanXiao1999/UI-Genie-Agent-16k``) adapter.

Canonical :mod:`lite.data.staging` wiring (content-addressed image store,
hash-split assigner, per-row staging helper, OOB check) shared by the two
subsets (``ui_genie`` + ``amex``), both mobile/android use.
"""

from __future__ import annotations

from lite.data.preproc.common import SourceStaging

DATASET_NAME = "UI-Genie-Agent"
SOURCE = "HanXiao1999/UI-Genie-Agent-16k"
PLATFORM = "mobile"
OS = "android"

_STAGING = SourceStaging(DATASET_NAME)
out_dir_for = _STAGING.out_dir_for
make_image_store = _STAGING.make_image_store
make_splitter = _STAGING.make_splitter
stage_entry = _STAGING.stage_entry
