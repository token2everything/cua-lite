"""Generate lite.scalecua oracle fixture JSONL files from Python modules.

Usage:
    uv run python -m lite.gym.envs.lite.scalecua.src.gen.oracle
    uv run python -m lite.gym.envs.lite.scalecua.src.gen.oracle --check
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from lite.gym.envs.lite.scalecua.src.utils.assets import CATALOG_DIR

from ._fixtures import (
    CANONICAL_SPLITS,
    canonical_paths,
    fixture_belongs_to_source,
    iter_fixture_rows,
    iter_jsonl,
    validate_unique,
)


# Domain-oriented oracle registry. The domain modules are the source of truth
# for the canonical aggregate fixture files under data/oracle/.
SHARDS: dict[str, str] = {
    "chrome": "domains.chrome",
    "gimp": "domains.gimp",
    "libreoffice_calc": "domains.libreoffice_calc",
    "libreoffice_impress": "domains.libreoffice_impress",
    "libreoffice_writer": "domains.libreoffice_writer",
    "multi_apps": "domains.multi_apps",
    "os": "domains.os",
    "thunderbird": "domains.thunderbird",
    "vlc": "domains.vlc",
    "vs_code": "domains.vs_code",
}

DEFAULT_OUTPUT_DIR = CATALOG_DIR / "oracle"


def _load_catalog(catalog_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    catalog: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(catalog_dir.glob("*.jsonl")):
        split = path.stem
        for _, row in iter_jsonl(path):
            catalog[(split, row["task_id"])] = row
    if not catalog:
        raise SystemExit(
            f"no task catalog under {catalog_dir} — run "
            "`bash lite/gym/envs/lite/scalecua/scripts/install.sh` (or "
            "`scripts/utils/tasks.sh generate`) to import the ScaleCUA catalogs first"
        )
    return catalog


def _load_module(shard: str):
    module_name = SHARDS[shard]
    return importlib.import_module(f"{__package__}.{module_name}")


def _validate_canonical_rows(
    rows: list[dict[str, Any]],
    catalog: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    dropped: list[tuple[str, str, str]] = []
    for row in rows:
        split = row["split"]
        task_id = row["task_id"]
        key = (split, task_id)
        if key in seen:
            raise ValueError(f"duplicate oracle row for {split}:{task_id}")
        seen.add(key)

        task = catalog.get(key)
        if task is None:
            raise KeyError(f"oracle row references missing task {split}:{task_id}")

        exclude_reason = task["metadata"].get("others", {}).get("exclude_reason")
        if exclude_reason:
            # Exclusion is the single source of truth for runnability. A curated
            # oracle whose task later became excluded (upstream corpus defect,
            # detector hit, ...) is not an error — it is simply out of scope, so
            # DROP it here. The generated fixture set is thus the set difference
            # {curated ORACLES} ∩ {runnable tasks}, which makes the fixture set
            # and the exclusion list impossible to drift apart. (Previously this
            # hard-failed, forcing a manual edit of the frozen ORACLES blob every
            # time a task was excluded — the source of the drift it meant to
            # prevent.) The dataset-side _apply_oracle_fixtures keeps its
            # hard-fail as a downstream assertion that this filtering happened.
            dropped.append((split, task_id, exclude_reason))
            continue

        oracle_actions = row.get("oracle_actions")
        oracle_trajectory = row.get("oracle_trajectory")
        if bool(oracle_actions) == bool(oracle_trajectory):
            raise ValueError(
                f"oracle row must populate exactly one replay field: {split}:{task_id}"
            )

        result.append(copy.deepcopy(row))
    if dropped:
        logger.info(
            "oracle: dropped %d row(s) for excluded task(s): %s",
            len(dropped),
            ", ".join(f"{s}:{t} ({r})" for s, t, r in dropped),
        )
    return result


def build_rows(
    shard: str,
    catalog_dir: Path = CATALOG_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[dict[str, Any]]:
    catalog = _load_catalog(catalog_dir)
    module = _load_module(shard)
    # Every domain module holds its fixture rows verbatim; validate them
    # against the imported catalog (existence, exclusion, replay fields).
    if not module.ORACLES_ARE_CANONICAL_ROWS:
        raise ValueError(f"domain module {shard} must set ORACLES_ARE_CANONICAL_ROWS")
    return _validate_canonical_rows(list(module.ORACLES), catalog)


def to_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )


def _load_existing_rows(output_dir: Path) -> list[dict[str, Any]]:
    return list(iter_fixture_rows(output_dir))


def _row_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("split") or ""),
        str(row.get("domain") or ""),
        str(row.get("source") or ""),
        str(row.get("fixture_id") or ""),
        str(row.get("task_id") or ""),
    )


def _belongs_to_any_shard(row: dict[str, Any], shards: set[str]) -> bool:
    return row.get("domain") in shards or any(
        fixture_belongs_to_source(row, shard) for shard in shards
    )


def build_canonical_rows(
    shards: list[str],
    *,
    catalog_dir: Path = CATALOG_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, list[dict[str, Any]]]:
    selected = set(shards)
    existing = [
        row
        for row in _load_existing_rows(output_dir)
        if not _belongs_to_any_shard(row, selected)
    ]
    generated: list[dict[str, Any]] = []
    for shard in shards:
        generated.extend(build_rows(shard, catalog_dir=catalog_dir, output_dir=output_dir))

    rows = existing + generated
    validate_unique(rows)
    by_split = {
        split: sorted(
            [row for row in rows if row.get("split") == split],
            key=_row_sort_key,
        )
        for split in CANONICAL_SPLITS
    }
    unknown = sorted({str(row.get("split")) for row in rows} - set(CANONICAL_SPLITS))
    if unknown:
        raise ValueError(f"unsupported oracle fixture split(s): {', '.join(unknown)}")
    return by_split


def check_canonical_rows(
    shards: list[str],
    *,
    catalog_dir: Path = CATALOG_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> bool:
    expected = build_canonical_rows(
        shards,
        catalog_dir=catalog_dir,
        output_dir=output_dir,
    )
    failed = False
    for split, rows in expected.items():
        path = canonical_paths(output_dir)[split]
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        text = to_jsonl(rows)
        if current != text:
            print(f"{path}: out of date")
            failed = True
        else:
            print(f"{path}: ok ({len(rows)} rows)")
    return not failed


def _selected_shards(values: list[str] | None) -> list[str]:
    if not values:
        return sorted(SHARDS)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", choices=sorted(SHARDS))
    parser.add_argument("--catalog-dir", type=Path, default=CATALOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    shards = _selected_shards(args.shard)
    if args.check:
        return 0 if check_canonical_rows(
            shards,
            catalog_dir=args.catalog_dir,
            output_dir=args.output_dir,
        ) else 1

    by_split = build_canonical_rows(
        shards,
        catalog_dir=args.catalog_dir,
        output_dir=args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in by_split.items():
        path = canonical_paths(args.output_dir)[split]
        path.write_text(to_jsonl(rows), encoding="utf-8")
        print(f"wrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
