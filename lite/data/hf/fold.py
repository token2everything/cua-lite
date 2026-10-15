"""Image-dedup fold/unfold for the cua-lite HF transport layer.

Grounding/understanding cohorts are single-image-per-row, and many rows share
the *same* screenshot: SeeClick, for example, emits ~11 grounding instructions
per screen. The HF parquet embeds image bytes per row (so the Hub viewer can
render thumbnails), so a shared screenshot is duplicated once per instruction —
an ~11x storage blowup (seeclick alone was 1.43 TB, ~125 GB of unique images).

This module makes image-dedup a pure *transport* optimization, invisible to the
canonical local format and to every preproc script:

* :func:`fold_rows` (upload side) groups same-image rows into ONE row — the
  image is embedded once — and stashes the authoritative per-instruction
  payloads in a ``_folded`` JSON column.
* :func:`unfold_rows` (download side) reverses it exactly, reproducing the
  original one-row-per-instruction stream.

The canonical local layout (one instruction per row, ``images`` as
content-addressed paths) never changes. ``messages`` / ``metadata`` are opaque
JSON strings in that layout, so folding carries them verbatim — the round-trip
is byte-identical:

    unfold_rows(fold_rows(rows)) == rows        # as a multiset (order is grouped)

Only ``task_type`` in {``grounding.*``, ``understanding``} is folded (see
:func:`should_fold`); navigation is multi-image per row and rarely shares
screenshots across rows, so it is left untouched.

This module has no entry point; it is used by :mod:`lite.data.hf.upload` and
:mod:`lite.data.hf.download`.
"""

from __future__ import annotations

import json
from typing import Any

from lite.core.metadata import SINGLE_TURN_TASK_TYPES, LiteCUAMetadata

# Column that marks a folded row and carries the authoritative per-instruction
# members. Its presence is how the download side detects a folded cohort.
FOLD_COL = "_folded"


def should_fold(task_type: LiteCUAMetadata.TaskType | str) -> bool:
    """Whether a cohort's ``task_type`` is eligible for image-dedup folding.

    Foldable == single-turn: those cohorts are single-image-per-row, which is the
    property folding exploits. ``task_type`` reaches here verbatim from a staging
    path, so an unrecognised spelling answers "don't fold" rather than raising.
    """
    return task_type in SINGLE_TURN_TASK_TYPES


def _concat_messages(msg_strs: list[str]) -> str:
    """Concatenate members' message-lists under the single shared image.

    Viewer-only: the authoritative copy lives in ``_folded``. Each member's
    ``messages`` is a JSON list whose user turn references image index 0; since
    a folded row has exactly one image, concatenating keeps every reference
    valid (the viewer just shows the same thumbnail per turn).
    """
    merged: list[Any] = []
    for s in msg_strs:
        merged.extend(json.loads(s))
    return json.dumps(merged, ensure_ascii=False)


def fold_rows(rows: list[dict]) -> list[dict]:
    """Group same-image rows into one folded row each (image embedded once).

    Input rows are canonical: ``images`` a 1-element path list, ``messages`` /
    ``metadata`` opaque JSON strings. Rows that aren't single-image (should not
    occur for foldable cohorts) each become their own singleton group, so the
    output schema is uniform (every row carries ``_folded``).
    """
    groups: dict[Any, list[dict]] = {}
    order: list[Any] = []
    for row in rows:
        imgs = row.get("images") or []
        key = imgs[0] if len(imgs) == 1 else ("\x00multi", id(row))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    folded: list[dict] = []
    for key in order:
        members = groups[key]
        member_payload = [{"messages": m["messages"], "metadata": m["metadata"]} for m in members]
        folded.append({
            "images": list(members[0]["images"]),
            "messages": _concat_messages([m["messages"] for m in members]),
            "metadata": members[0]["metadata"],
            FOLD_COL: json.dumps(member_payload, ensure_ascii=False),
        })
    return folded


def unfold_rows(folded_rows: list[dict]) -> list[dict]:
    """Inverse of :func:`fold_rows`: expand each folded row back to one row per
    instruction, every member re-sharing the folded row's single image.

    Works whether the folded row's ``images`` holds path strings (local) or HF
    Image dicts with embedded bytes (post-snapshot) — the entry is copied
    verbatim into each member, and the caller's image-store rewrite dedups.
    """
    out: list[dict] = []
    for frow in folded_rows:
        members = json.loads(frow[FOLD_COL])
        imgs = frow["images"]
        for m in members:
            out.append({
                "images": list(imgs),
                "messages": m["messages"],
                "metadata": m["metadata"],
            })
    return out
