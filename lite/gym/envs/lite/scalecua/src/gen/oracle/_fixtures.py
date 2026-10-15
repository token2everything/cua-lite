"""Shared helpers for generated lite.scalecua oracle fixture files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


CANONICAL_SPLITS = ("rl", "train")


def canonical_paths(output_dir: Path) -> dict[str, Path]:
    return {split: output_dir / f"{split}.jsonl" for split in CANONICAL_SPLITS}


def fixture_belongs_to_source(row: dict[str, Any], source: str) -> bool:
    fixture_id = str(row.get("fixture_id") or "")
    return row.get("source") == source or fixture_id.startswith(f"oracle_{source}_")


def _fixture_files(output_dir: Path) -> list[Path]:
    paths = canonical_paths(output_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        missing = [path.name for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "partial canonical oracle fixture set in "
                f"{output_dir}: missing {', '.join(missing)}"
            )
        return [paths[split] for split in CANONICAL_SPLITS]
    return sorted(output_dir.glob("*.jsonl"))


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def iter_fixture_rows(output_dir: Path) -> Iterator[dict[str, Any]]:
    for path in _fixture_files(output_dir):
        for _, row in iter_jsonl(path):
            yield row


def validate_unique(rows: list[dict[str, Any]]) -> None:
    by_fixture: dict[str, dict[str, Any]] = {}
    by_task: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        fixture_id = str(row.get("fixture_id") or "")
        if not fixture_id:
            raise ValueError("oracle row missing fixture_id")
        if fixture_id in by_fixture:
            raise ValueError(f"duplicate fixture_id: {fixture_id}")
        by_fixture[fixture_id] = row

        split = str(row.get("split") or "")
        task_id = str(row.get("task_id") or "")
        if not split or not task_id:
            raise ValueError(f"oracle row missing split/task_id: {fixture_id}")
        key = (split, task_id)
        if key in by_task:
            raise ValueError(f"duplicate oracle task: {split}:{task_id}")
        by_task[key] = row
