"""CLI: concatenate multiple data files (JSONL or parquet) into one.

Usage:
    python -m lite.data.merge \
      -i trajectory.parquet grounding.parquet understanding.parquet \
      -o all.parquet

    python -m lite.data.merge \
      -i trajectory.jsonl grounding.jsonl understanding.jsonl \
      -o all.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from lite.data.staging import OPAQUE_JSON_FIELDS, serialize_opaque_json_fields


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_rows(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        return _read_jsonl(path)
    if path.endswith(".parquet"):
        return pq.read_table(path).to_pylist()
    raise ValueError(f"Unsupported input format: {path}")


def _serialize_opaque_columns(table: pa.Table) -> pa.Table:
    """JSON-encode ``messages``/``metadata`` if they arrived as Arrow structs.

    Only those two columns are pulled into Python; every other column keeps its
    source Arrow type.
    """
    for name in OPAQUE_JSON_FIELDS:
        index = table.schema.get_field_index(name)
        if index < 0 or pa.types.is_string(table.schema.field(index).type):
            continue
        encoded = [
            serialize_opaque_json_fields({name: value})[name]
            for value in table.column(index).to_pylist()
        ]
        table = table.set_column(index, pa.field(name, pa.string()),
                                 pa.array(encoded, pa.string()))
    return table


def _read_table(path: str) -> pa.Table:
    if path.endswith(".parquet"):
        return _serialize_opaque_columns(pq.read_table(path))
    return pa.Table.from_pylist(
        [serialize_opaque_json_fields(row) for row in _read_rows(path)]
    )


def _merge_parquet(inputs: list[str], output: str) -> None:
    """Merge to parquet at the Arrow level, preserving each input's column types.

    Not through pandas: a ``read_parquet().to_dict()`` -> ``DataFrame().to_parquet()``
    round-trip re-infers the schema from Python objects and picks 32-bit offsets,
    turning ``large_list<large_binary>`` image columns into ``list<binary>``, which
    overflows past 2 GiB of blobs. It also materialises every row in memory.
    """
    tables = []
    for path in inputs:
        table = _read_table(path)
        print(f"  {path}: {table.num_rows} rows")
        tables.append(table)

    merged = pa.concat_tables(tables, promote_options="permissive")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    pq.write_table(merged, output)
    print(f"Wrote {merged.num_rows} rows to {output}")


def _merge_jsonl(inputs: list[str], output: str) -> None:
    """Merge files into JSONL output."""
    all_rows = []
    for path in inputs:
        rows = _read_rows(path)
        print(f"  {path}: {len(rows)} rows")
        all_rows.extend(rows)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} rows to {output}")

def main():
    parser = argparse.ArgumentParser(
        description="Concatenate multiple data files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--inputs", nargs="+", required=True,
                        help="Input files (JSONL or parquet)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output file path (.parquet or .jsonl)")
    args = parser.parse_args()

    if args.output.endswith(".parquet"):
        _merge_parquet(args.inputs, args.output)
    else:
        _merge_jsonl(args.inputs, args.output)

if __name__ == "__main__":
    main()
