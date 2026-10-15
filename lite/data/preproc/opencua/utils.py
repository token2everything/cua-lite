"""Shared helpers for the OpenCUA (AgentNet) preproc adapter.

Wiring is the shared :class:`~lite.data.preproc.common.SourceStaging` default,
exactly as :mod:`lite.data.preproc.scalecua.utils` does — only ``DATASET_NAME``
changes. The splitter keys off ``metadata.others.id``: normally
``agentnet_{dataset_type}_{task_id}``, with a record-index suffix only for
upstream duplicate ``task_id`` values, so the produced id is per-row unique.

Usage:
    from lite.data.preproc.opencua.utils import (
        DATASET_NAME, out_dir_for, make_image_store, make_splitter, stage_entry,
    )
"""

from __future__ import annotations

from lite.data.preproc.common import SourceStaging

DATASET_NAME = "OpenCUA"

_STAGING = SourceStaging(DATASET_NAME)
out_dir_for = _STAGING.out_dir_for
make_image_store = _STAGING.make_image_store
make_splitter = _STAGING.make_splitter
stage_entry = _STAGING.stage_entry
