"""Shared helpers for the ScaleCUA preproc adapters.

Every per-task script (``understanding.py``, ``grounding-action.py``,
``grounding-bbox.py``, ``grounding-point.py``, ``use.py``) shares
the same wiring against :mod:`lite.data.staging`:

* image-store with the canonical ``cua-lite/ScaleCUA/images`` prefix,
* split assigner keyed off ``metadata.others.id`` (ScaleCUA's per-row
  unique id; ``source_id`` is a per-batch category label and would put
  whole batches in the same split),
* per-row staging helper that hashes images, fills the canonical
  metadata defaults, assigns a split, and produces a partition key.
"""

from __future__ import annotations

from lite.data.preproc.common import SourceStaging

DATASET_NAME = "ScaleCUA"
_STAGING = SourceStaging(DATASET_NAME)


out_dir_for = _STAGING.out_dir_for
make_image_store = _STAGING.make_image_store
make_splitter = _STAGING.make_splitter
stage_entry = _STAGING.stage_entry
