"""
Load CUA-Lite datasets from paths.

Public API:
    discover_files_under_paths(paths, splits=…) -> [(abs_path, rel_path), ...]
    load_file_as_dataset(file_path) -> Dataset

Paths must be absolute. Regex patterns are supported after a literal base,
e.g. "/data/OpenGVLab/ScaleCUA-Data/(desktop|browser)/use". Matched path
components remain part of each file's relative identity, so same-named
partitions from different platforms are both loaded.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path

from datasets import Dataset, load_dataset

from lite.data.staging import serialize_opaque_json_fields, split_of_partition_file

logger = logging.getLogger(__name__)


def load_file_as_dataset(file_path: Path) -> Dataset:
    """Load a single parquet or JSONL file as a HuggingFace Dataset."""
    if file_path.suffix == ".parquet":
        return load_dataset("parquet", data_files=str(file_path), split="train")
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(serialize_opaque_json_fields(json.loads(line)))
    return Dataset.from_list(rows) if rows else Dataset.from_list([])


def expand_path_pattern(path_pattern: str) -> list[Path]:
    """Expand a path pattern (literal base + optional regex) to matching directories.

    Examples:
        "/data/ScaleCUA-Data" -> [Path("/data/ScaleCUA-Data")]
        "/data/ScaleCUA-Data/(desktop|browser)/use" -> [.../desktop/use, .../browser/use]
    """
    base_match = re.match(r"^([^\[(]*)", path_pattern)
    base = (base_match.group(1) or "").rstrip("/")
    if not base:
        raise ValueError("Path pattern must have a literal base prefix")

    base_path = Path(base).resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {base_path}")

    rest = path_pattern[len(base):].lstrip("/")
    if not rest:
        return [base_path]

    full_regex = re.compile(str(base_path) + "/" + rest)
    result = []
    for root, dirs, _ in os.walk(base_path):
        for d in dirs:
            p = (Path(root) / d).resolve()
            if full_regex.fullmatch(str(p)):
                result.append(p)
    return sorted(result)


def _literal_base(path_pattern: str) -> Path:
    base_match = re.match(r"^([^\[(]*)", path_pattern)
    base = (base_match.group(1) or "").rstrip("/")
    if not base:
        raise ValueError("Path pattern must have a literal base prefix")
    return Path(base).resolve()


def discover_files_under_paths(
    paths: list[str], *, splits: Sequence[str] = ("train",)
) -> list[tuple[Path, str]]:
    """Discover parquet files under each path pattern. Returns (abs_path, rel_path).

    A canonical dataset encodes its train/validation carve in the partition path
    (``<platform>/<task_type>/<split>[/<variant>].parquet``), and no row carries a
    split marker, so a file whose path names a canonical split is kept only for a
    split in *splits* -- otherwise pointing a consumer at the dataset root sweeps
    the held-out shards in with the training ones.

    A file that names no canonical split (a rollout ``trajectory.parquet``, a
    preproc partition named after its cohort) has no layout to filter by and is
    always kept. Pass ``("train", "validation")`` to take the whole dataset.

    Every file the filter drops is reported at WARNING, so a row-count change is
    never silent: the same command on the same data says which splits it left out.
    Overlapping inputs deduplicate the same physical file; different files with
    the same partition name are both kept so canonical datasets can be mixed.
    """
    wanted = set(splits)
    files: list[tuple[Path, str]] = []
    seen_files: set[Path] = set()
    skipped: dict[str, int] = {}
    for path in paths:
        relative_root = _literal_base(path)
        for resolved_dir in expand_path_pattern(path):
            for file_path in resolved_dir.rglob("*.parquet"):
                resolved_file = file_path.resolve()
                if resolved_file in seen_files:
                    continue
                seen_files.add(resolved_file)
                # Keep regex-selected path components in the identity. For
                # ``root/(desktop|browser)/use``, both platforms may legitimately
                # contain ``train/use.parquet``.
                rel = str(file_path.relative_to(relative_root))
                file_split = split_of_partition_file(Path(rel))
                if file_split is not None and file_split not in wanted:
                    skipped[file_split] = skipped.get(file_split, 0) + 1
                    continue
                files.append((file_path, rel))
    if skipped:
        logger.warning(
            "splits=%s kept %d parquet file(s) and skipped %s; pass every split you want",
            tuple(splits), len(files),
            ", ".join(f"{n} in {s!r}" for s, n in sorted(skipped.items())),
        )
    return sorted(files, key=lambda x: (x[1], str(x[0])))
