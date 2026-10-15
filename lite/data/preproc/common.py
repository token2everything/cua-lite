"""Source-neutral helpers for dataset preproc staging.

Dataset adapters should keep raw parsing and source-specific action policy in
their own ``utils.py`` files. This module owns only the repeated canonical
staging mechanics shared by those adapters, plus the out-of-bounds coordinate
gate :func:`has_oob_coordinate` and the canonical tool-call walker
:func:`action_argument_dicts` it is built on.

Run:
    uv run --extra data --extra dev pytest tests/data/preproc -q
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    action_coordinate_arguments_out_of_range,
)
from lite.core.metadata import LiteCUAMetadata, metadata_from_dict
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.data import staging
from lite.data.utils.rows import validate_canonical_rows


def _cua_metadata_for_stage(row: dict[str, Any]) -> LiteCUAMetadata:
    metadata = metadata_from_dict(row["metadata"])
    if not isinstance(metadata, LiteCUAMetadata):
        raise ValueError(
            "preproc staging only supports CUA metadata; generic publish layout "
            "is not supported by this plan"
        )
    return metadata


def prestage_images(
    store: staging.ImageStore,
    paths: Iterable[Path | str],
) -> set[Path]:
    """Stage unique images in parallel and return corrupt source paths.

    The common path stays fast. Only after a bulk verification failure do we
    retry each unique path to identify the bad files, while successful paths
    hit :class:`ImageStore`'s source-path cache.
    """
    unique = list(dict.fromkeys(Path(path) for path in paths))
    try:
        store.put_many(unique)
        return set()
    except staging.CorruptImageError:
        def retry(path: Path) -> Path | None:
            try:
                store.put(path)
            except staging.CorruptImageError:
                return path
            return None

        with ThreadPoolExecutor(max_workers=32) as executor:
            return {path for path in executor.map(retry, unique) if path is not None}


@dataclass(frozen=True)
class SourceStaging:
    """Default staging wiring for one preproc dataset source."""

    dataset_name: str
    val_cap: int = 2000
    # Output dirs whose ``split.json`` this instance has already written. The
    # policy is per-run, not per-row, and every adapter reaches ``stage_entry``.
    _described: set[Path] = field(default_factory=set, repr=False, compare=False)

    def out_dir_for(self, *, root: Path | str | None = None) -> Path:
        return staging.dataset_root(self.dataset_name, root=root)

    def make_image_store(self, out_dir: Path) -> staging.ImageStore:
        return staging.ImageStore(
            out_dir / "images",
            rel_prefix=staging.image_rel_prefix(self.dataset_name),
        )

    def make_splitter(self) -> staging.SplitAssigner:
        """Per-row id-keyed splitter, upstream split label honoured when present.

        ``others.id`` is INDEXED, not defaulted: the canonical format mandates it,
        and a ``.get(...) or ""`` fallback would hash every id-less row to the same
        key, i.e. collapse a whole cohort into one split bucket instead of failing.
        An absent ``id`` is an adapter bug and belongs at the adapter.
        """
        return staging.SplitAssigner(
            canonical_fn=lambda r: r["metadata"]["others"].get("split"),
            key_fn=lambda r: str(r["metadata"]["others"]["id"]),
            key_desc="metadata.others.id",
            bucket_fn=lambda r: tuple(_cua_metadata_for_stage(r).dims),
            bucket_desc="(metadata.dims[0], metadata.dims[1], variant)",
            val_cap=self.val_cap,
        )

    def stage_entry(
        self,
        entry: dict[str, Any],
        *,
        store: staging.ImageStore,
        splitter: staging.SplitAssigner,
        variant: str,
    ) -> tuple[tuple[str, str, str, str], dict[str, Any]]:
        """Hash the row's images, fill metadata defaults, assign a split.

        ``metadata.others.split`` is a *transient routing hint*: an adapter that
        honors an upstream split writes it for :meth:`make_splitter`'s
        ``canonical_fn`` to read here, and it must not reach the parquet — the
        split is already encoded by the partition path, and a persisted copy is
        NOT caught downstream -- ``lite/data/utils/rows.py`` rejects top-level
        ``metadata.split``, a DIFFERENT key, so a leaked ``others.split`` reaches a
        published row unchallenged. That is exactly why the pop must be reliable
        rather than backed by a gate. Consuming it means
        popping it, so the strip is a property of staging rather than something
        each adapter has to remember.

        Also the one place that holds BOTH the assigner and the output dir
        (``store.root`` is ``<out_dir>/images``), so it is where the assigner's
        ``describe()`` gets persisted -- once per output dir, see
        :meth:`_record_split_policy`.
        """
        self._record_split_policy(store.root.parent, splitter)
        # INDEXED: ``images`` is mandated. ``.get(...) or []`` would *inject* an empty
        # list here and hide the missing key from ``staging.content_fingerprint``,
        # which indexes it one call deeper.
        entry["images"] = store.put_many([Path(p) for p in entry["images"]])
        meta_obj = _cua_metadata_for_stage(entry)
        entry["metadata"] = meta_obj.to_dict()
        meta = entry["metadata"]
        split = splitter.assign(entry, bucket_extra=(variant,))
        meta["others"].pop("split", None)
        key = (meta_obj.platform.value, meta_obj.task_type.value, split, variant)
        validate_canonical_rows([entry], f"{self.dataset_name}/{variant}")
        return key, entry

    def _record_split_policy(self, out_dir: Path, splitter: staging.SplitAssigner) -> None:
        """Persist ``splitter.describe()`` to ``<out_dir>/split.json``.

        The split is a path artifact, so no row carries it and nothing on disk
        says how the carve was drawn. This is the record; the reader is a human
        looking at the dataset card, which ``lite.data.hf.upload`` folds it into.
        A dataset's several task scripts each write it -- same policy, one file,
        last writer wins with identical content.
        """
        if out_dir in self._described:
            return
        self._described.add(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "split.json").write_text(
            json.dumps({"split_policy": splitter.describe()}, indent=2) + "\n"
        )


def action_argument_dicts(tc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the argument dicts a canonical tool call actually executes.

    An action-batch ``computer``/``mobile`` call carries its child actions in
    ``arguments.actions``; every other call carries exactly one argument dict.
    :func:`has_oob_coordinate` walks this to scan coordinates uniformly across
    action-batch and standalone calls.
    """
    args = tool_call_arguments(tc)
    if tool_call_name(tc) not in LITE_ACTION_BATCH_TOOL_NAMES:
        return [args]
    actions = args["actions"]
    if not isinstance(actions, list):
        raise TypeError("computer/mobile tool_call arguments.actions must be a list")
    for action in actions:
        action["action"]
    return actions


def has_oob_coordinate(entry: dict[str, Any]) -> bool:
    """Return True if any tool_call coordinate in *entry* leaves ``[0, 1000]``.

    The last-line gate every adapter runs before staging a row (see the parent
    ``AGENTS.md`` §Common gotchas). Descends into **both** action-batch tools,
    not just ``computer``: GUIAct's smartphone subset emits ``mobile{actions:[...]}``
    and a per-source copy that hard-coded ``"computer"`` once let out-of-bounds
    mobile batches publish. Both ``coordinate`` and ``start_coordinate`` are
    scanned so drag / swipe start points are covered too, and the range test is
    :func:`~lite.core.tools.action_space.action_coordinate_arguments_out_of_range`, which
    accepts tuples as well as lists — the per-source copies tested ``list``
    only, so a tuple coordinate was silently never scanned.
    """
    for msg in entry.get("messages", []):
        for tc in msg.get("tool_calls", []):
            for args in action_argument_dicts(tc):
                if action_coordinate_arguments_out_of_range(args):
                    return True
    return False


__all__ = ["SourceStaging", "has_oob_coordinate", "prestage_images"]
