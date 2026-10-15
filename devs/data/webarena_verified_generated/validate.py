#!/usr/bin/env python3
"""Validate a generated-task batch and emit post-generation leakage receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import (
    ContractError,
    assert_public_projection,
    contamination_result,
    load_official_catalog,
    validate_candidate_shape,
    validate_upstream_task_spec,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ContractError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ContractError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True, help="private candidate JSONL")
    parser.add_argument("--public", type=Path, required=True, help="public task JSONL")
    parser.add_argument(
        "--official-catalog",
        type=Path,
        required=True,
        help="pinned official catalog; post-generation checker only",
    )
    parser.add_argument("--receipts", type=Path, required=True, help="output leakage JSONL")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    private_rows = _load_jsonl(args.private)
    public_rows = _load_jsonl(args.public)
    if len(private_rows) != len(public_rows):
        raise ContractError("private/public row counts differ")

    official = load_official_catalog(args.official_catalog)
    receipts = []
    seen_ids: set[str] = set()
    for private, public in zip(private_rows, public_rows, strict=True):
        validate_candidate_shape(private)
        validate_upstream_task_spec(private["task_spec"])
        assert_public_projection(public, private)
        generated_id = private["generated_task_id"]
        if generated_id in seen_ids:
            raise ContractError(f"duplicate generated_task_id: {generated_id}")
        seen_ids.add(generated_id)
        receipts.append(contamination_result(private, official))

    blocked = [row for row in receipts if row["status"] != "pass"]
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    args.receipts.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in receipts)
    )
    if blocked:
        raise ContractError(
            f"{len(blocked)}/{len(receipts)} candidates are reject/review; none may enter training"
        )
    print(f"validated={len(receipts)} contamination_pass={len(receipts)}")


if __name__ == "__main__":
    main()

