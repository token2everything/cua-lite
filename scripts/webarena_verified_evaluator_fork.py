#!/usr/bin/env python3
"""Replay archived WebArena-Verified cases with a named, offline dataset fork.

Example: uv run python scripts/webarena_verified_evaluator_fork.py \
    --run-root /path/to/glm-full-run --task-ids 699,700,811

The program reads the installed 1.2.3 dataset and archived model/browser inputs.
It writes a patched dataset only inside a temporary directory and prints a
counterfactual JSON report. It does not alter official results or site state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
import zipfile
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

from webarena_verified.api import WebArenaVerified
from webarena_verified.core.utils.checksum import compute_evaluator_checksum
from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaSite
from webarena_verified.types.tracing import NetworkEvent, NetworkTrace

PATCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "devs/exps/eval/browsergym.webarena_verified/EVALUATOR_FORK.v3-singleton-array.json"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def patched_dataset(
    official_data: list[dict[str, Any]], patch_spec: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply only the declared array schemas to an in-memory dataset copy."""
    fork_data = json.loads(json.dumps(official_data))
    records = {record["task_id"]: record for record in fork_data}
    for patch in patch_spec["patches"]:
        record = records[patch["task_id"]]
        evaluator = record["eval"][patch["evaluator_index"]]
        if evaluator["evaluator"] != "NetworkEventEvaluator" or "post_data_schema" in evaluator:
            raise ValueError(f"Task {patch['task_id']} has a different evaluator contract")
        properties = patch["post_data_schema"]["properties"]
        if not all(
            isinstance(evaluator["expected"]["post_data"][name], list)
            for name in properties
        ):
            raise ValueError(f"Task {patch['task_id']} no longer has the declared array fields")
        evaluator["post_data_schema"] = patch["post_data_schema"]
    return fork_data


def _task_config(
    dataset_file: Path, record: dict[str, Any], instruction: dict[str, Any]
) -> WebArenaVerifiedConfig:
    template_urls = record["start_urls"]
    rendered_urls = instruction["start_urls"]
    if len(template_urls) != 1 or len(rendered_urls) != 1 or len(record["sites"]) != 1:
        raise ValueError("This fork replay supports one-site, one-start-URL tasks only")
    site = WebArenaSite(record["sites"][0])
    if template_urls[0] != site.url_name_template:
        raise ValueError(f"Task {record['task_id']} has an unsupported start URL template")
    return WebArenaVerifiedConfig(
        test_data_file=dataset_file,
        environments={site: EnvironmentConfig(urls=[rendered_urls[0]])},
    )


def _extract_archived_inputs(archive_path: Path, task_id: int, target: Path) -> dict[str, Path]:
    names = {
        "har": "network.har",
        "trace": "trace.zip",
        "response": "agent_response.json",
    }
    paths: dict[str, Path] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for kind, name in names.items():
            source = archive.extractfile(f"task-{task_id:04d}/{name}")
            if source is None:
                raise FileNotFoundError(f"{name} missing from {archive_path}")
            destination = target / name
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            paths[kind] = destination
    return paths


def load_archived_trace(trace_path: Path) -> NetworkTrace:
    """Read resource bodies and request order without patching the installed package."""
    snapshots: list[tuple[bool, str, int, dict[str, Any]]] = []
    with zipfile.ZipFile(trace_path) as archive:
        for index, line in enumerate(archive.read("trace.network").decode("utf-8").splitlines()):
            item = json.loads(line)
            if item.get("type") != "resource-snapshot":
                continue
            snapshot = item["snapshot"]
            for content in (
                snapshot.get("request", {}).get("postData"),
                snapshot.get("response", {}).get("content"),
            ):
                if isinstance(content, dict) and not content.get("text") and isinstance(
                    content.get("_sha1"), str
                ):
                    content["text"] = archive.read(f"resources/{content['_sha1']}").decode(
                        "utf-8", errors="replace"
                    )
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
    events = tuple(NetworkEvent(data=MappingProxyType(item[3])) for item in snapshots)
    return NetworkTrace.model_construct(is_playwright=True, src_file=trace_path, events=events)


def _result_summary(result: Any) -> dict[str, Any]:
    return {
        "score": result.score,
        "status": result.status,
        "dataset_sha256": result.webarena_verified_data_checksum,
        "evaluator_sha256": result.webarena_verified_evaluator_checksum,
        "evaluators": [
            {
                "name": item.evaluator_name,
                "score": item.score,
                "status": item.status,
                "error": item.error_msg,
                "assertions": [assertion.assertion_name for assertion in item.assertions or ()],
            }
            for item in result.evaluators_results
        ],
    }


def replay(
    run_root: Path, task_ids: tuple[int, ...], patch_path: Path = PATCH_PATH
) -> dict[str, Any]:
    """Score unchanged archived inputs under official and fork dataset contracts."""
    patch_bytes = patch_path.read_bytes()
    patch_spec = json.loads(patch_bytes)
    patch_ids = {patch["task_id"] for patch in patch_spec["patches"]}
    if not set(task_ids) <= patch_ids:
        raise ValueError(f"Only patched task IDs may be replayed: {sorted(patch_ids)}")

    official_dataset_path = Path(
        str(files("webarena_verified").joinpath("assets/dataset/webarena-verified.json"))
    )
    official_bytes = official_dataset_path.read_bytes()
    official_dataset_sha256 = _sha256(official_bytes)
    if official_dataset_sha256 != patch_spec["official_dataset_sha256"]:
        raise ValueError("Installed official dataset differs from the fork's pinned source")
    official_data = json.loads(official_bytes)
    fork_data = patched_dataset(official_data, patch_spec)
    evaluator_sha256 = compute_evaluator_checksum()
    fork_evaluator_sha256 = _sha256(
        evaluator_sha256.encode() + b"\0" + Path(__file__).read_bytes() + b"\0" + patch_bytes
    )

    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="wav-evaluator-fork-") as scratch:
        scratch_path = Path(scratch)
        fork_dataset_path = scratch_path / "webarena-verified-fork.json"
        fork_dataset_path.write_text(
            json.dumps(fork_data, ensure_ascii=False, separators=(",", ":"))
        )
        fork_dataset_sha256 = _sha256(fork_dataset_path.read_bytes())

        for task_id in task_ids:
            index = run_root / "index" / f"task-{task_id:04d}"
            recorded = json.loads((index / "evaluation.json").read_text())
            instruction = json.loads((index / "instruction.json").read_text())
            record = official_data[task_id]
            if record["task_id"] != task_id:
                raise ValueError(f"Official dataset index does not match task {task_id}")
            case_dir = scratch_path / f"task-{task_id:04d}"
            case_dir.mkdir()
            paths = _extract_archived_inputs(
                run_root / "archives" / f"task-{task_id:04d}.tar.gz", task_id, case_dir
            )
            raw_response = json.loads(paths["response"].read_text())["raw_content"]
            receipt = recorded["input_receipt"]
            for kind, receipt_name in (("har", "network_har"), ("trace", "playwright_trace")):
                if _sha256(paths[kind].read_bytes()) != receipt[receipt_name]["sha256"]:
                    raise ValueError(f"Task {task_id} archived {kind} differs from scored input")
            if _sha256(raw_response.encode()) != receipt["agent_response_raw"]["sha256"]:
                raise ValueError(
                    f"Task {task_id} archived final response differs from scored input"
                )

            har = NetworkTrace.from_har(paths["har"])
            trace = load_archived_trace(paths["trace"])
            official = WebArenaVerified(
                config=_task_config(official_dataset_path, record, instruction)
            )
            fork = WebArenaVerified(config=_task_config(fork_dataset_path, record, instruction))
            scores: dict[str, Any] = {}
            for kind, network in (("har", har), ("playwright_trace", trace)):
                official_result = official.evaluate_task(
                    task_id=task_id, agent_response=raw_response, network_trace=network
                )
                archived_result = recorded[kind]
                if official_result.model_dump(mode="json") != archived_result:
                    raise ValueError(
                        f"Task {task_id} official {kind} replay disagrees with archive"
                    )
                fork_result = fork.evaluate_task(
                    task_id=task_id, agent_response=raw_response, network_trace=network
                )
                scores[kind] = {
                    "official": _result_summary(official_result),
                    "fork": _result_summary(fork_result),
                }
            cases.append({"task_id": task_id, "scores": scores})

    return {
        "kind": "offline_counterfactual_not_official_score",
        "fork_id": patch_spec["fork_id"],
        "official_dataset_sha256": official_dataset_sha256,
        "official_evaluator_sha256": evaluator_sha256,
        "fork_dataset_sha256": fork_dataset_sha256,
        "fork_evaluator_sha256": fork_evaluator_sha256,
        "patch_sha256": _sha256(patch_bytes),
        "cases": cases,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-ids", default="699,700,811")
    parser.add_argument("--patch", type=Path, default=PATCH_PATH)
    return parser.parse_args()


def main() -> None:
    """Replay the selected archived cases and print one versioned JSON report."""
    args = _parse_args()
    task_ids = tuple(int(value) for value in args.task_ids.split(","))
    logging.getLogger("WebArena-Verified").setLevel(logging.CRITICAL)
    print(json.dumps(replay(args.run_root, task_ids, args.patch), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
