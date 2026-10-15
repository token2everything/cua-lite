#!/usr/bin/env python3
"""Contracts and fail-closed gates for generated WebArena-Verified tasks."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

OFFICIAL_CATALOG_SHA256 = "d65275660814663375028e9017e1f929e3c38321041b125795e2713b52243d30"
OFFICIAL_TASK_IDS = range(812)
GENERATED_ID_FLOOR = 2_000_000
ALLOWED_SITES = frozenset({"shopping", "shopping_admin", "reddit", "gitlab", "wikipedia", "map"})
ALLOWED_GENERATION_MODES = frozenset({"direct", "regrounded_semantic_seed"})
PRIVATE_KEYS = frozenset(
    {
        "eval",
        "expected",
        "instantiation_dict",
        "ground_truth",
        "judge",
        "credentials",
        "cookies",
        "har",
        "trace",
    }
)
OFFICIAL_SOURCE_MARKERS = (
    "webarena-verified.json",
    "webarena_verified/assets/dataset",
    "test.raw.json",
    OFFICIAL_CATALOG_SHA256,
)


class ContractError(ValueError):
    """Raised when a generated task is unsafe or internally inconsistent."""


def canonical_json(value: Any) -> str:
    """Serialize a JSON value for stable hashing and fingerprints."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def _task_type(task_spec: Mapping[str, Any]) -> str:
    response_evals = [
        item
        for item in task_spec.get("eval", ())
        if isinstance(item, Mapping) and item.get("evaluator") == "AgentResponseEvaluator"
    ]
    if len(response_evals) != 1:
        raise ContractError("task_spec must contain exactly one AgentResponseEvaluator")
    expected = response_evals[0].get("expected")
    if not isinstance(expected, Mapping):
        raise ContractError("AgentResponseEvaluator.expected must be an object")
    task_type = str(expected.get("task_type", "")).lower()
    if task_type not in {"retrieve", "navigate", "mutate"}:
        raise ContractError("expected.task_type must be retrieve, navigate, or mutate")
    return task_type


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def assert_public_projection(public: Mapping[str, Any], private: Mapping[str, Any]) -> None:
    """Check that a public row matches its private task without leaking judge data."""
    task_spec = private.get("task_spec")
    if not isinstance(task_spec, Mapping):
        raise ContractError("private task is missing task_spec")
    expected = {
        "generated_task_id": private.get("generated_task_id"),
        "sites": task_spec.get("sites"),
        "start_urls": task_spec.get("start_urls"),
        "intent": task_spec.get("intent"),
    }
    if dict(public) != expected:
        raise ContractError("public task must be the exact projection of the private task spec")
    leaked = PRIVATE_KEYS.intersection(_walk_keys(public))
    if leaked:
        raise ContractError(f"public task contains private keys: {sorted(leaked)}")


def validate_candidate_shape(candidate: Mapping[str, Any]) -> None:
    """Validate project-level invariants before invoking the upstream model."""
    required = {"generated_task_id", "generation_mode", "source", "task_spec"}
    missing = required - candidate.keys()
    if missing:
        raise ContractError(f"candidate missing fields: {sorted(missing)}")

    generated_id = candidate["generated_task_id"]
    if not isinstance(generated_id, str) or not re.fullmatch(
        r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", generated_id
    ):
        raise ContractError("generated_task_id must match wavgen-<batch>-<instance>")
    if candidate["generation_mode"] not in ALLOWED_GENERATION_MODES:
        raise ContractError("unsupported generation_mode")

    source = candidate["source"]
    if not isinstance(source, Mapping):
        raise ContractError("source must be an object")
    source_text = canonical_json(source).lower()
    if any(marker.lower() in source_text for marker in OFFICIAL_SOURCE_MARKERS):
        raise ContractError("official WebArena data cannot be a generation source")
    if not source.get("artifact_sha256") or not source.get("record_sha256"):
        raise ContractError("source must pin artifact_sha256 and record_sha256")

    task_spec = candidate["task_spec"]
    if not isinstance(task_spec, Mapping):
        raise ContractError("task_spec must be an object")
    task_id = task_spec.get("task_id")
    template_id = task_spec.get("intent_template_id")
    if type(task_id) is not int or task_id < GENERATED_ID_FLOOR:
        raise ContractError("generated task_id must be an integer >= 2_000_000")
    if type(template_id) is not int or template_id < GENERATED_ID_FLOOR:
        raise ContractError("generated intent_template_id must be an integer >= 2_000_000")

    sites = task_spec.get("sites")
    if not isinstance(sites, list) or len(sites) != 1 or sites[0] not in ALLOWED_SITES:
        raise ContractError("P0 tasks must contain exactly one supported site")
    start_urls = task_spec.get("start_urls")
    expected_placeholder = f"__{sites[0].upper()}__"
    if not isinstance(start_urls, list) or not start_urls:
        raise ContractError("task_spec.start_urls must be a non-empty list")
    if any(not isinstance(url, str) or expected_placeholder not in url for url in start_urls):
        raise ContractError("every start URL must use the task site's URL placeholder")

    task_type = _task_type(task_spec)
    evals = task_spec.get("eval")
    assert isinstance(evals, list)
    network_evals = [item for item in evals if item.get("evaluator") == "NetworkEventEvaluator"]
    if task_type in {"navigate", "mutate"} and not network_evals:
        raise ContractError("navigate and mutate tasks require a NetworkEventEvaluator")
    response_eval = next(
        item for item in evals if item.get("evaluator") == "AgentResponseEvaluator"
    )
    retrieved = response_eval["expected"].get("retrieved_data")
    status = str(response_eval["expected"].get("status", "")).upper()
    if task_type == "retrieve" and status == "SUCCESS":
        if not isinstance(retrieved, list) or not retrieved:
            raise ContractError(
                "successful retrieve tasks require non-empty grounded retrieved_data"
            )
    elif retrieved is not None:
        raise ContractError("navigation, mutation, and error outcomes require null retrieved_data")


def validate_upstream_task_spec(task_spec: Mapping[str, Any]) -> None:
    """Run the pinned package's canonical Pydantic validation."""
    try:
        from webarena_verified.types import WebArenaVerifiedTask
    except ImportError as error:
        raise ContractError(
            "webarena-verified is required; run this gate in the pinned authoring environment"
        ) from error
    WebArenaVerifiedTask.model_validate(task_spec)


def public_projection(candidate: Mapping[str, Any]) -> dict[str, Any]:
    task_spec = candidate["task_spec"]
    return {
        "generated_task_id": candidate["generated_task_id"],
        "sites": task_spec["sites"],
        "start_urls": task_spec["start_urls"],
        "intent": task_spec["intent"],
    }


def _ngrams(value: str, size: int = 5) -> set[str]:
    compact = normalize_text(value).replace(" ", "")
    if not compact:
        return set()
    if len(compact) < size:
        return {compact}
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def text_containment(candidate: str, reference: str) -> float:
    candidate_grams = _ngrams(candidate)
    if not candidate_grams:
        return 0.0
    return len(candidate_grams & _ngrams(reference)) / len(candidate_grams)


def _oracle_scalars(task_spec: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in {"task_type", "status", "evaluator"}:
                    continue
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif value is not None:
            normalized = normalize_text(str(value))
            if normalized:
                values.add(normalized)

    for evaluator in task_spec.get("eval", ()):
        if isinstance(evaluator, Mapping):
            visit(evaluator.get("expected"))
    return values


def oracle_jaccard(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_values = _oracle_scalars(left)
    right_values = _oracle_scalars(right)
    union = left_values | right_values
    return len(left_values & right_values) / len(union) if union else 0.0


def _without_identity(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in task_spec.items()
        if key not in {"task_id", "intent_template_id", "revision"}
    }


def _evaluator_fingerprint(task_spec: Mapping[str, Any]) -> str:
    return sha256_json(task_spec.get("eval", ()))


def contamination_result(
    candidate: Mapping[str, Any], official_records: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare one already-generated candidate against the official catalog."""
    validate_candidate_shape(candidate)
    task_spec = candidate["task_spec"]
    task_type = _task_type(task_spec)
    candidate_sites = sorted(task_spec["sites"])
    candidate_hash = sha256_json(_without_identity(task_spec))
    candidate_intent = normalize_text(task_spec["intent"])
    candidate_template = normalize_text(task_spec["intent_template"])
    candidate_eval = _evaluator_fingerprint(task_spec)

    best: dict[str, Any] | None = None
    reasons: list[str] = []
    for official in official_records:
        try:
            official_type = _task_type(official)
        except ContractError:
            continue
        if sorted(official.get("sites", ())) != candidate_sites or official_type != task_type:
            continue
        text_score = text_containment(task_spec["intent"], str(official.get("intent", "")))
        oracle_score = oracle_jaccard(task_spec, official)
        current = {
            "task_id": official.get("task_id"),
            "text_containment": round(text_score, 6),
            "oracle_jaccard": round(oracle_score, 6),
        }
        if best is None or (text_score, oracle_score) > (
            best["text_containment"],
            best["oracle_jaccard"],
        ):
            best = current

        if sha256_json(_without_identity(official)) == candidate_hash:
            reasons.append(f"official content hash match: {official.get('task_id')}")
        if normalize_text(str(official.get("intent", ""))) == candidate_intent:
            reasons.append(f"normalized intent match: {official.get('task_id')}")
        if normalize_text(str(official.get("intent_template", ""))) == candidate_template:
            reasons.append(f"normalized template match: {official.get('task_id')}")
        if _evaluator_fingerprint(official) == candidate_eval:
            reasons.append(f"evaluator fingerprint match: {official.get('task_id')}")
        if text_score >= 0.90:
            reasons.append(f"text containment >= 0.90: {official.get('task_id')}")
        if oracle_score >= 0.85 and text_score >= 0.65:
            reasons.append(f"oracle/text near duplicate: {official.get('task_id')}")

    if reasons:
        status = "reject"
    elif best and best["text_containment"] >= 0.75:
        status = "review"
    else:
        status = "pass"
    return {
        "gate_version": 1,
        "official_catalog_sha256": OFFICIAL_CATALOG_SHA256,
        "generated_task_id": candidate["generated_task_id"],
        "candidate_sha256": sha256_json(candidate),
        "status": status,
        "reasons": sorted(set(reasons)),
        "nearest_official_task": best,
    }


def load_official_catalog(path: Path) -> list[dict[str, Any]]:
    actual = sha256_file(path)
    if actual != OFFICIAL_CATALOG_SHA256:
        raise ContractError(
            f"official catalog hash mismatch: expected {OFFICIAL_CATALOG_SHA256}, found {actual}"
        )
    records = json.loads(path.read_text())
    if not isinstance(records, list) or len(records) != 812:
        raise ContractError("official catalog must contain exactly 812 records")
    if [record.get("task_id") for record in records] != list(OFFICIAL_TASK_IDS):
        raise ContractError("official catalog IDs must be the contiguous range 0..811")
    return records
