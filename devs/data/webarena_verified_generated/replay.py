#!/usr/bin/env python3
"""Validate reference evidence and independent answer/action contrasts."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import gitlab_episode_evaluator as episode_owner
from contract import ContractError, sha256_file, sha256_json
from jsonpath_ng.ext import parse as parse_jsonpath
from jsonpath_ng.jsonpath import Fields
from webarena_verified.core.evaluation.evaluators import EVALUATOR_REGISTRY
from webarena_verified.core.utils.checksum import compute_evaluator_checksum
from webarena_verified.core.utils.jsonpath_utils import (
    extract_jsonpath_value,
    is_jsonpath_key,
    is_regexp_jsonpath_key,
)
from webarena_verified.core.utils.pattern_utils import is_regexp
from webarena_verified.types import WebArenaSite, WebArenaVerifiedTask
from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
from webarena_verified.types.eval import EvalStatus, TaskEvalContext
from webarena_verified.types.tracing import NetworkTrace

from lite.gym.envs.browsergym.webarena_verified import install_playwright_trace_loader


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _evaluate(
    task: WebArenaVerifiedTask,
    response: Any,
    trace: NetworkTrace,
    config: WebArenaVerifiedConfig,
) -> list[dict[str, Any]]:
    context = TaskEvalContext(
        task=task,
        agent_response_raw=response,
        network_trace=trace,
        config=config,
    )
    return [
        EVALUATOR_REGISTRY[evaluator_config.evaluator]()
        .evaluate(context=context, config=evaluator_config)
        .model_dump(mode="json")
        for evaluator_config in task.eval
    ]


def _artifact_path(batch_root: Path, candidate: dict[str, Any], name: str) -> Path:
    relative = candidate["grounding"]["artifacts"][name]
    path = (batch_root / relative).resolve()
    if not path.is_relative_to(batch_root.resolve()):
        raise ContractError(f"artifact path escapes batch root: {relative}")
    return path


def _validate_receipts(batch_root: Path, candidate: dict[str, Any]) -> dict[str, str]:
    if candidate["task_definition_sha256"] != sha256_json(candidate["task_definition"]):
        raise ContractError("task-definition receipt hash mismatch")
    if candidate.get("baseline_sha256") != sha256_json(candidate.get("baseline")):
        raise ContractError("baseline receipt hash mismatch")
    if candidate.get("grounding_sha256") != sha256_json(candidate.get("grounding")):
        raise ContractError("grounding receipt hash mismatch")
    if candidate["grounding"].get("task_spec_sha256") != sha256_json(candidate["task_spec"]):
        raise ContractError("grounding/task-spec hash mismatch")

    hashes = {}
    expected_hashes = candidate["grounding"].get("artifact_sha256", {})
    for name in ("har", "trace", "screenshot", "observation"):
        path = _artifact_path(batch_root, candidate, name)
        if not path.is_file():
            raise ContractError(f"missing grounding artifact: {path}")
        hashes[name] = sha256_file(path)
        if expected_hashes.get(name) != hashes[name]:
            raise ContractError(f"grounding artifact hash mismatch: {name}")
    return hashes


def validate_network_replay(batch_root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    artifact_hashes = _validate_receipts(batch_root, candidate)
    task = WebArenaVerifiedTask.model_validate(candidate["task_spec"])
    site = WebArenaSite(candidate["task_spec"]["sites"][0])
    config = WebArenaVerifiedConfig(
        test_data_file=batch_root / "private.jsonl",
        environments={site: EnvironmentConfig(urls=[candidate["baseline"]["site_url"]])},
    )
    har_trace = NetworkTrace.from_har(_artifact_path(batch_root, candidate, "har"))
    install_playwright_trace_loader()
    playwright_trace = NetworkTrace.from_playwright_trace(
        _artifact_path(batch_root, candidate, "trace")
    )
    response = json.dumps(candidate["reference_response"])
    positive_har = _evaluate(task, response, har_trace, config)
    positive_playwright = _evaluate(task, response, playwright_trace, config)
    # Change answer, task type, and status separately; retain the correct trace.
    wrong_response = dict(candidate["reference_response"])
    wrong_response["retrieved_data"] = (
        [] if wrong_response["retrieved_data"] else ["__UNEXPECTED_RETRIEVED_DATA__"]
    )
    wrong_task_type = dict(candidate["reference_response"])
    wrong_task_type["task_type"] = (
        "NAVIGATE" if wrong_task_type["task_type"] == "RETRIEVE" else "RETRIEVE"
    )
    wrong_status = dict(candidate["reference_response"])
    wrong_status["status"] = "UNKNOWN_ERROR" if wrong_status["status"] == "SUCCESS" else "SUCCESS"
    negative_wrong = _evaluate(task, json.dumps(wrong_response), har_trace, config)
    negative_task_type = _evaluate(task, json.dumps(wrong_task_type), har_trace, config)
    negative_status = _evaluate(task, json.dumps(wrong_status), har_trace, config)

    # Keep the correct answer when testing absent action evidence.
    empty_trace = NetworkTrace.model_construct(
        is_playwright=False,
        src_file=Path("<empty-negative>"),
        events=(),
    )
    negative_empty = _evaluate(task, response, empty_trace, config)
    answer_negative_scores = [
        0.0 if item.evaluator == "AgentResponseEvaluator" else 1.0 for item in task.eval
    ]
    empty_trace_scores = [
        0.0 if item.evaluator == "NetworkEventEvaluator" and not item.should_not_exist else 1.0
        for item in task.eval
    ]
    results_by_probe = {
        "positive_har": positive_har,
        "positive_playwright": positive_playwright,
        "negative_wrong_response": negative_wrong,
        "negative_wrong_task_type": negative_task_type,
        "negative_wrong_status": negative_status,
        "negative_empty_trace": negative_empty,
    }
    expected_scores = {
        "positive_har": [1.0] * len(task.eval),
        "positive_playwright": [1.0] * len(task.eval),
        "negative_wrong_response": answer_negative_scores,
        "negative_wrong_task_type": answer_negative_scores,
        "negative_wrong_status": answer_negative_scores,
        "negative_empty_trace": empty_trace_scores,
    }

    # Derive synthetic post-body contrasts from real selected requests. They are
    # evaluator fixtures, never independently executed reference evidence.
    post_data_contrasts = []
    negative_post_data_targets = {}
    source_entries = [event.model_dump(mode="json")["data"] for event in har_trace.events]
    for evaluator_index, evaluator_config in enumerate(task.eval):
        if (
            evaluator_config.evaluator != "NetworkEventEvaluator"
            or evaluator_config.should_not_exist
            or not evaluator_config.expected.post_data
        ):
            continue
        selected_requests = positive_har[evaluator_index]["actual"] or []
        candidate_indices = [
            index
            for index, event in enumerate(har_trace.events)
            if any(
                selected["url"] == event.url
                and selected["http_method"] == event.http_method
                and selected["response_status"] == event.request_status
                and selected.get("post_data") == event.post_data
                for selected in selected_requests
            )
        ]
        # The evaluator's raw actual list contains endpoint candidates, not
        # only events satisfying every expected request/response criterion.
        post_evaluator = EVALUATOR_REGISTRY["NetworkEventEvaluator"]()
        selected_indices = []
        for index in candidate_indices:
            single_event = NetworkTrace.model_construct(
                is_playwright=har_trace.is_playwright,
                src_file=har_trace.src_file,
                events=(har_trace.events[index],),
            )
            matched = post_evaluator.evaluate(
                context=TaskEvalContext(
                    task=task,
                    agent_response_raw=response,
                    network_trace=single_event,
                    config=config,
                ),
                config=evaluator_config,
            )
            if matched.status == EvalStatus.SUCCESS:
                selected_indices.append(index)
        field_schemas = (evaluator_config.post_data_schema or {}).get("properties", {})
        mime_type = (
            "application/json"
            if any(
                isinstance(har_trace.events[index].post_data, list)
                or any(
                    isinstance(value, (dict, list))
                    for value in (har_trace.events[index].post_data or {}).values()
                )
                for index in selected_indices
            )
            else "application/x-www-form-urlencoded"
        )
        for field, expected_value in evaluator_config.expected.post_data.items():
            if not isinstance(expected_value, (str, int, float, bool)):
                continue
            paths = {}
            if is_jsonpath_key(field):
                regex_path = is_regexp_jsonpath_key(field) or is_regexp(field)
                expression = None if regex_path else parse_jsonpath(field)
                for index in selected_indices:
                    post_body = har_trace.events[index].post_data
                    body = (
                        post_body
                        if isinstance(post_body, list)
                        else {
                            key: value
                            for key, value in (post_body or {}).items()
                            if not post_evaluator._should_ignore_post_param(key, evaluator_config)
                        }
                    )
                    scalar = extract_jsonpath_value(body, field, strict=False)
                    if not isinstance(scalar, (str, int, float, bool)):
                        continue
                    matches = (
                        [
                            Fields(key)
                            for key, value in body.items()
                            if extract_jsonpath_value({key: value}, field, strict=False) is not None
                        ]
                        if regex_path
                        else [match.full_path for match in expression.find(body)]
                    )
                    if len(matches) == 1 and isinstance(scalar, (str, int, float, bool)):
                        paths[index] = matches[0]
            else:
                paths = {
                    index: field
                    for index in selected_indices
                    if field in (har_trace.events[index].post_data or {})
                }
            if not selected_indices or len(paths) != len(selected_indices):
                continue
            field_type = field_schemas.get(field, {}).get("type", "string")
            if field_type == "boolean":
                value = (
                    EVALUATOR_REGISTRY["NetworkEventEvaluator"]()
                    .value_normalizer.normalize_single(expected_value, "boolean", strict=True)
                    .normalized
                )
                wrong_value = "0" if value else "1"
                equivalent_value = "true" if value else "false"
            elif field_type == "number":
                value = (
                    EVALUATOR_REGISTRY["NetworkEventEvaluator"]()
                    .value_normalizer.normalize_single(expected_value, "number", strict=True)
                    .normalized
                )
                wrong_value = str(value + 1)
                equivalent_value = str(float(value))
            else:
                wrong_value = "__WRONG_CRITICAL_PARAMETER__"
                equivalent_value = None

            for kind, replacement in (
                ("negative_post_data", wrong_value),
                ("equivalent_post_data", equivalent_value),
            ):
                entries = deepcopy(source_entries)
                for index in selected_indices:
                    body = deepcopy(har_trace.events[index].post_data)
                    if replacement is not None:
                        if is_jsonpath_key(field):
                            paths[index].update(body, replacement)
                        else:
                            body[field] = replacement
                    entries[index]["request"]["postData"] = {
                        "mimeType": mime_type,
                        "text": (
                            json.dumps(body)
                            if mime_type == "application/json"
                            else urlencode(body, doseq=True)
                        ),
                    }
                    for header in entries[index]["request"]["headers"]:
                        if header["name"].lower() == "content-type":
                            header["value"] = mime_type
                name = f"{kind}_e{evaluator_index}_{field}"
                results_by_probe[name] = _evaluate(
                    task, response, NetworkTrace.from_content(entries), config
                )
                scores = [1.0] * len(task.eval)
                if kind == "negative_post_data":
                    scores[evaluator_index] = 0.0
                    negative_post_data_targets[name] = evaluator_index
                expected_scores[name] = scores
                post_data_contrasts.append(
                    {
                        "probe": name,
                        "evaluator_index": evaluator_index,
                        "field": field,
                        "field_type": field_type,
                        "artifact_kind": "synthetic_contrast",
                        "source_artifact": "har",
                        "source_artifact_sha256": artifact_hashes["har"],
                        "synthetic_trace_sha256": sha256_json(entries),
                        "post_encoding": mime_type,
                    }
                )
    checks = {
        f"{name}_expected_vector": all(
            # A wrong POST field can also violate another network constraint.
            # The target must fail; answers and all other probes stay exact.
            (result["score"], result["status"]) in ((0.0, "failure"), (1.0, "success"))
            if name in negative_post_data_targets
            and index != negative_post_data_targets[name]
            and evaluator_config.evaluator == "NetworkEventEvaluator"
            else result["score"] == expected_score
            and result["status"] == ("success" if expected_score else "failure")
            for index, (evaluator_config, result, expected_score) in enumerate(
                zip(task.eval, results, expected_scores[name], strict=True)
            )
        )
        for name, results in results_by_probe.items()
    }
    checks["har_playwright_results_equal"] = [
        (result["status"], result["score"]) for result in positive_har
    ] == [(result["status"], result["score"]) for result in positive_playwright]
    evaluator_errors = [
        {"probe": name, "evaluator_index": index, "result": result}
        for name, results in results_by_probe.items()
        for index, result in enumerate(results)
        if result["status"] == "error"
    ]
    checks["no_evaluator_errors"] = not evaluator_errors
    contrast_limitations = [
        name
        for name in results_by_probe
        if name.startswith(("negative_", "equivalent_")) and not checks[f"{name}_expected_vector"]
    ]
    post_contrast_status = (
        "not_applicable"
        if not post_data_contrasts
        else "pass"
        if all(checks[f"{probe['probe']}_expected_vector"] for probe in post_data_contrasts)
        else "quarantine"
    )
    if evaluator_errors or contrast_limitations:
        status = "quarantine"
    else:
        status = "pass" if all(checks.values()) else "reject"

    return {
        "gate_version": 3,
        "generated_task_id": candidate["generated_task_id"],
        "status": status,
        "checks": checks,
        "baseline_sha256": candidate["baseline_sha256"],
        "grounding_sha256": candidate["grounding_sha256"],
        "task_definition_sha256": candidate["task_definition_sha256"],
        "task_spec_sha256": sha256_json(candidate["task_spec"]),
        "reference_response_sha256": sha256_json(candidate["reference_response"]),
        "webarena_verified_version": version("webarena-verified"),
        "webarena_verified_evaluator_checksum": compute_evaluator_checksum(),
        "artifact_sha256": artifact_hashes,
        "expected_scores": expected_scores,
        "evaluator_errors": evaluator_errors,
        "evaluator_limitations": contrast_limitations,
        "gate_phase": "complete_official_native_network_vectors_only_not_GT",
        "mandatory_original_episode_and_visible_GUI_gate_pending":
            "episode_contract" in candidate["grounding"],
        "post_data_contrasts": {
            "status": post_contrast_status,
            "probes": post_data_contrasts,
        },
        **results_by_probe,
    }


def validate_replay(batch_root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """Require official network vectors and any explicitly declared episode policy."""
    result = validate_network_replay(batch_root, candidate)
    har = NetworkTrace.from_har(_artifact_path(batch_root, candidate, "har"))
    install_playwright_trace_loader()
    playwright = NetworkTrace.from_playwright_trace(_artifact_path(batch_root, candidate, "trace"))
    empty = NetworkTrace.model_construct(is_playwright=False, src_file=Path("<empty-negative>"), events=())
    episode = episode_owner.evaluate_recording(
        candidate,
        {"positive_har": har, "positive_playwright": playwright, "negative_empty_trace": empty},
        candidate["baseline"]["site_url"],
        artifact_root=batch_root,
        har_path=_artifact_path(batch_root, candidate, "har"),
        trace_path=_artifact_path(batch_root, candidate, "trace"),
        expected_scope="author_context_only",
    )
    result["checks"].update(episode.pop("checks"))
    result.update(episode, mandatory_original_episode_and_visible_GUI_gate_pending=False)
    if episode["episode_gate_status"] != "not_applicable":
        result["gate_phase"] = "complete_official_network_and_public_original_episode_GUI"
    result["status"] = "hold" if episode["episode_gate_status"] == "hold" else (
        "quarantine" if result["evaluator_errors"] or result["evaluator_limitations"] else
        "pass" if all(result["checks"].values()) else "reject")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--receipts", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    private = args.private or args.batch_root / "private.jsonl"
    receipts_path = args.receipts or args.batch_root / "replay_receipts.jsonl"
    receipts = [validate_replay(args.batch_root, row) for row in _load_jsonl(private)]
    receipts_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in receipts)
    )
    blocked = [row for row in receipts if row["status"] != "pass"]
    if blocked:
        raise ContractError(f"{len(blocked)}/{len(receipts)} replay checks failed")
    print(f"replay_validated={len(receipts)} all_pass=true")


if __name__ == "__main__":
    main()
