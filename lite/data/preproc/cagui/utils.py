"""Shared helpers for the CAGUI (OpenBMB/CAGUI) preproc adapters.

Mirrors :mod:`lite.data.preproc.scalecua.utils` /
:mod:`lite.data.preproc.opencua.utils` — every dataset adapter wires
``ImageStore``, ``SplitAssigner``, and ``stage_entry`` the same way; only
``DATASET_NAME`` and the field accessors change.

Usage:
    from lite.data.preproc.cagui.utils import (
        DATASET_NAME, out_dir_for, make_image_store, make_splitter, stage_entry,
    )
"""

from __future__ import annotations

from lite.data.preproc.common import SourceStaging

DATASET_NAME = "CAGUI"

_STAGING = SourceStaging(DATASET_NAME)
out_dir_for = _STAGING.out_dir_for
make_image_store = _STAGING.make_image_store
make_splitter = _STAGING.make_splitter
stage_entry = _STAGING.stage_entry
