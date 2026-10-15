"""Generic record-list -> Parquet writer (no cua-lite semantics).

The leaf-tier parquet I/O used across pipelines (train export, infer rollout,
agent trajectory logging). Replaces the third-party-swappable mechanism that
used to live in ``lite.data.utils``; moved here so the model side
(``lite.agents.core.agent.logger``) no longer reaches up into ``lite.data`` (#40 DAG).

Usage:
    from lite.utils.parquet import write_records_to_parquet
    write_records_to_parquet(records, "/path/to/output.parquet")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert an object to a dict. Prefers .to_dict() method, falls back to shallow field extraction."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return {f: getattr(obj, f) for f in obj.__dataclass_fields__}
    return dict(obj)


#: Reserved child field injected into a dict path where PyArrow would otherwise
#: infer ``struct<>``. Always null-valued, so it carries no data — it exists only
#: to give the struct a child field, which is exactly the remedy PyArrow's own
#: error message names ("Consider adding a dummy child field").
_EMPTY_STRUCT_CHILD = "__lite_empty_struct__"


def _nonempty_dict_paths(obj: Any, path: tuple[str, ...], out: set[tuple[str, ...]]) -> None:
    """Collect the paths at which *obj* ever holds a NON-empty dict.

    Path components are dict keys; list indices are not part of the path, since
    Arrow gives every element of a list one shared element type. Two tool calls
    in the same trajectory therefore share the ``messages/tool_calls/arguments``
    path, which is precisely the granularity the schema is inferred at.
    """
    if isinstance(obj, dict):
        if obj:
            out.add(path)
        for key, value in obj.items():
            _nonempty_dict_paths(value, path + (key,), out)
    elif isinstance(obj, list):
        for value in obj:
            _nonempty_dict_paths(value, path, out)


def _normalize_for_arrow(
    obj: Any, path: tuple[str, ...], nonempty: set[tuple[str, ...]],
) -> Any:
    """Make *obj* writable by PyArrow while keeping ``{}`` distinct from ``None``.

    PyArrow cannot WRITE a struct with no child fields (``pa.Table.from_pylist``
    infers it happily; ``pq.write_table`` raises ``ArrowNotImplementedError:
    Cannot write struct type '<f>' with no child field to Parquet``). That
    constraint is real, but it only bites at a path where EVERY value is an empty
    dict — the only case that infers ``struct<>``.

    The old blanket ``{} -> None`` therefore over-applied. It also erased a
    meaningful distinction: a zero-argument tool call (``back``, ``new_tab``) has
    ``arguments == {}``, and rewriting that to ``None`` makes the golden record
    say "this call had no arguments field" rather than "this call took no
    arguments". Downstream that reads as a malformed tool_call — and, because a
    malformed call is not a call, it cascades into a spurious orphan
    ``role:"tool"`` result, so each defect is reported twice.

    So:

    * where the path holds a non-empty dict anywhere in the record set, ``{}`` is
      kept. Arrow writes it as a PRESENT struct whose children are all null,
      which reads back as a dict and stays distinguishable from a null struct.
    * where it does not (the genuine ``struct<>`` case), a single null-valued
      :data:`_EMPTY_STRUCT_CHILD` is injected so the struct has a child field.
      It reads back as a dict too.

    Either way ``{}`` never round-trips as ``None``.
    """
    if isinstance(obj, dict):
        if not obj:
            return {} if path in nonempty else {_EMPTY_STRUCT_CHILD: None}
        return {k: _normalize_for_arrow(v, path + (k,), nonempty) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_for_arrow(v, path, nonempty) for v in obj]
    return obj


def _json_default(obj: Any) -> Any:
    """JSON fallback for array/scalar values inside opaque parquet fields."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _promote_binary_lists_to_large(table: pa.Table) -> pa.Table:
    """Promote any ``list<binary>`` column to ``large_list<large_binary>``.

    PyArrow's default inference picks 32-bit-offset list/binary types. Once the
    combined binary bytes of a column exceed 2 GiB, the column is silently
    chunked, and downstream readers that call
    ``ParquetFile.iter_batches(batch_size=...)`` (e.g. slime's parquet loader at
    ``batch_size=4096``) hit ``"Nested data conversions not implemented for
    chunked array outputs"``. Promoting to 64-bit offsets fixes the read path.

    Cast in two stages — single-step ``cast(large_list<large_binary>)`` fails on
    already-chunked binary because pyarrow can't fuse 32-bit-offset chunks into
    a single 64-bit-offset array. We cast each chunk to ``list<large_binary>``
    (still 32-bit list offsets, but 64-bit binary offsets), then
    ``combine_chunks()`` is safe (large_binary fuses without offset overflow),
    and finally wrap in ``large_list``.

    Only ``list<binary>`` columns are touched; other column types pass through.
    """
    new_columns = []
    new_schema_fields = []
    for col, field in zip(table.columns, table.schema):
        ft = field.type
        if (
            pa.types.is_list(ft)
            and pa.types.is_binary(ft.value_type)
        ):
            inner_target = pa.list_(pa.large_binary())
            promoted_chunks = [chunk.cast(inner_target) for chunk in col.chunks]
            fused = pa.chunked_array(promoted_chunks).combine_chunks()
            # `fused` is a single-chunked list<large_binary>; cast wraps to large_list.
            promoted = fused.cast(pa.large_list(pa.large_binary()))
            new_columns.append(promoted)
            new_schema_fields.append(pa.field(field.name, pa.large_list(pa.large_binary())))
        else:
            new_columns.append(col)
            new_schema_fields.append(field)
    return pa.Table.from_arrays(new_columns, schema=pa.schema(new_schema_fields))


def write_records_to_parquet(
    records: list[dict[str, Any]],
    path: Path | str,
    *,
    json_fields: Iterable[str] | None = None,
    compression: str = "snappy",
    row_group_size: int | None = None,
) -> None:
    """
    Write a list of record dicts to a Parquet file.

    PyArrow infers schema from the records (including nested dict/list);
    the result is compatible with load_dataset("parquet", ...) and preserves
    columns like images, messages, meta.
    Fields in json_fields are serialized to JSON strings before writing.

    Binary-list columns (e.g. ``processed_images`` storing PNG bytes) are
    automatically promoted to ``large_list<large_binary>`` so the file stays
    readable via ``ParquetFile.iter_batches(batch_size=...)`` even when the
    combined image bytes exceed 2 GiB — see :func:`_promote_binary_lists_to_large`
    for the chunked-nested-array rationale.

    Args:
        records: list of dicts (e.g. CUA-lite samples with images, messages, meta).
        path: Output file path (.parquet).
        json_fields: Field names to serialize as JSON strings; None to skip (default: None).
        compression: Parquet compression (default "snappy").
        row_group_size: Rows per row group; None for default.
    """
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Two passes: the empty-dict decision is per PATH across the whole record
    # set (that is the granularity Arrow infers the schema at), so the paths
    # that ever hold real keys must be known before any row is rewritten.
    json_field_set = set(json_fields or ())
    dicts = []
    for record in records:
        row = _to_dict(record)
        if json_field_set:
            row = dict(row)
            for field in json_field_set:
                if field in row:
                    row[field] = json.dumps(row[field], default=_json_default)
        dicts.append(row)
    nonempty: set[tuple[str, ...]] = set()
    for row in dicts:
        _nonempty_dict_paths(row, (), nonempty)
    rows = []
    for row in dicts:
        row = _normalize_for_arrow(row, (), nonempty)
        rows.append(row)
    table = pa.Table.from_pylist(rows)
    table = _promote_binary_lists_to_large(table)
    write_options = {"compression": compression}
    if row_group_size is not None:
        write_options["row_group_size"] = row_group_size
    pq.write_table(table, path, **write_options)
