"""Pinned WebArena-Verified task catalog.

The official dataset owns task identity, subset membership, and mutation type.
Pinning both distributions and both catalog payloads makes an upgrade an
explicit benchmark change instead of silently changing the score denominator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from functools import cache
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any

BROWSERGYM_PACKAGE_VERSION = "0.14.3"
PACKAGE_VERSION = "1.2.3"
DATASET_SHA256 = "d65275660814663375028e9017e1f929e3c38321041b125795e2713b52243d30"
HARD_SUBSET_SHA256 = "3b0a4df231bb5a0c642215e521c3fa97701a384f52a734dc2db8f617ad0591a7"
TASK_COUNT = 812
HARD_TASK_COUNT = 258


def _distribution_version(distribution: str, expected: str) -> None:
    installed = version(distribution)
    if installed != expected:
        raise RuntimeError(
            f"{distribution} version mismatch: expected {expected}, found {installed}. "
            "Run lite/gym/envs/browsergym/scripts/install.sh webarena_verified."
        )


def browsergym_package_root() -> Path:
    """Return the pinned BrowserGym integration package without importing it."""
    _distribution_version("browsergym-webarena-verified", BROWSERGYM_PACKAGE_VERSION)
    spec = importlib.util.find_spec("browsergym")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("browsergym")
    package_dir = Path(next(iter(spec.submodule_search_locations))) / "webarena_verified"
    if not (package_dir / "__init__.py").is_file():
        raise ModuleNotFoundError("browsergym.webarena_verified")
    return package_dir


def _package_root() -> Path:
    _distribution_version("webarena-verified", PACKAGE_VERSION)
    spec = importlib.util.find_spec("webarena_verified")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("webarena_verified")
    return Path(next(iter(spec.submodule_search_locations)))


def _read_checked(path: Path, expected_sha256: str) -> Any:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"WebArena-Verified catalog hash mismatch for {path.name}: "
            f"expected {expected_sha256}, found {actual}"
        )
    return json.loads(payload)


@cache
def task_records() -> tuple[dict[str, Any], ...]:
    """Return the validated official full-suite records in task-id order."""
    path = _package_root() / "assets/dataset/webarena-verified.json"
    records = _read_checked(path, DATASET_SHA256)
    if not isinstance(records, list) or len(records) != TASK_COUNT:
        found = len(records) if isinstance(records, list) else 0
        raise RuntimeError(f"WebArena-Verified must contain {TASK_COUNT} tasks, found {found}")
    task_ids = [record.get("task_id") for record in records]
    if task_ids != list(range(TASK_COUNT)):
        raise RuntimeError(
            "WebArena-Verified task_id values must be the unique contiguous range 0..811"
        )
    required = {"intent_template_id", "revision", "sites", "intent", "eval"}
    for record in records:
        missing = required - record.keys()
        if missing:
            raise RuntimeError(
                f"WebArena-Verified task {record['task_id']} missing fields: {sorted(missing)}"
            )
    return tuple(records)


@cache
def hard_task_ids() -> frozenset[int]:
    """Return task IDs in the official 258-task Hard subset."""
    path = _package_root() / "assets/dataset/subsets/webarena-verified-hard.json"
    subset = _read_checked(path, HARD_SUBSET_SHA256)
    task_ids = subset.get("task_ids") if isinstance(subset, dict) else None
    if not isinstance(task_ids, list) or len(task_ids) != HARD_TASK_COUNT:
        found = len(task_ids) if isinstance(task_ids, list) else 0
        raise RuntimeError(
            f"WebArena-Verified Hard must contain {HARD_TASK_COUNT} tasks, found {found}"
        )
    selected = frozenset(task_ids)
    if len(selected) != HARD_TASK_COUNT or not selected <= set(range(TASK_COUNT)):
        raise RuntimeError("WebArena-Verified Hard contains duplicate or unknown task IDs")
    return selected


@cache
def task_ids() -> tuple[str, ...]:
    """Return BrowserGym task IDs for the pinned full suite."""
    return tuple(
        f"webarena_verified.{record['intent_template_id']}.{record['task_id']}.{record['revision']}"
        for record in task_records()
    )


@cache
def _records_by_bgym_id() -> dict[str, dict[str, Any]]:
    return dict(zip(task_ids(), task_records(), strict=True))


def task_record(bgym_task_id: str) -> dict[str, Any] | None:
    """Return one official task record by BrowserGym task ID."""
    return _records_by_bgym_id().get(bgym_task_id)


def is_hard(bgym_task_id: str) -> bool:
    """Return whether a BrowserGym task belongs to the official Hard subset."""
    record = task_record(bgym_task_id)
    return record is not None and record["task_id"] in hard_task_ids()


def is_mutating(bgym_task_id: str) -> bool:
    """Return the exact mutation type declared by the response evaluator."""
    record = task_record(bgym_task_id)
    if record is None:
        return False
    response_evaluators = [
        item for item in record["eval"] if item.get("evaluator") == "AgentResponseEvaluator"
    ]
    if len(response_evaluators) != 1:
        raise RuntimeError(
            f"WebArena-Verified task {record['task_id']} must have one AgentResponseEvaluator"
        )
    return response_evaluators[0]["expected"]["task_type"] == "mutate"


def install_playwright_trace_loader() -> None:
    """Make Playwright's external request/response bodies visible to scoring.

    Playwright stores larger trace payloads under ``resources/<sha1>`` and leaves
    an empty ``text`` field plus ``_sha1`` in ``trace.network``. The pinned
    evaluator reads only ``trace.network``, so post-data assertions otherwise see
    ``None`` even though the browser sent the expected body. Hydrate those fields
    at the CUA-Lite integration boundary before the official evaluator parses
    them.
    """
    from webarena_verified.types import tracing
    from webarena_verified.types.tracing import NetworkEvent

    def load_playwright_trace(trace_path: Path) -> tuple[NetworkEvent, ...]:
        archive: zipfile.ZipFile | None = None
        if trace_path.suffix == ".zip":
            archive = zipfile.ZipFile(trace_path)
            raw_data = archive.read("trace.network").decode("utf-8")
            resource_root = None
        else:
            trace_file = trace_path / "trace.network" if trace_path.is_dir() else trace_path
            if not trace_file.exists():
                raise FileNotFoundError(f"Trace file {str(trace_file)!r} does not exist.")
            raw_data = trace_file.read_text(errors="ignore")
            resource_root = trace_file.parent / "resources"

        try:
            snapshots = []
            for index, line in enumerate(raw_data.splitlines()):
                item = json.loads(line)
                if item.get("type") != "resource-snapshot":
                    continue
                snapshot = item["snapshot"]
                for content in (
                    snapshot.get("request", {}).get("postData"),
                    snapshot.get("response", {}).get("content"),
                ):
                    if not isinstance(content, dict) or content.get("text"):
                        continue
                    resource_name = content.get("_sha1")
                    if not isinstance(resource_name, str):
                        continue
                    if archive is not None:
                        payload = archive.read(f"resources/{resource_name}")
                    else:
                        payload = (resource_root / resource_name).read_bytes()
                    content["text"] = payload.decode("utf-8", errors="replace")
                started_at = snapshot.get("startedDateTime")
                snapshots.append(
                    (
                        not isinstance(started_at, str),
                        started_at if isinstance(started_at, str) else "",
                        index,
                        snapshot,
                    )
                )
            snapshots.sort(key=lambda item: item[:3])
            return tuple(
                NetworkEvent(data=MappingProxyType(snapshot))
                for _, _, _, snapshot in snapshots
            )
        finally:
            if archive is not None:
                archive.close()

    tracing.load_playwright_trace = load_playwright_trace
