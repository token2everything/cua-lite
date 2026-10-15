#!/usr/bin/env python3
"""Compile grounded site definitions into generated task and evaluator files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from contract import (
    canonical_json,
    public_projection,
    sha256_file,
    sha256_json,
    validate_candidate_shape,
    validate_upstream_task_spec,
)
from cross_site_evaluator import derive_value
from jsonschema import Draft202012Validator

WEBHARBOR_COMMIT = "381dbd9882680806c81ed626d226f114c464a321"
WEBHARBOR_MANIFEST_SHA256 = "b80e9daec5375e65a01ebfdf3431db2015f23c5a66bfc193948947465e126637"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    return parser.parse_args()


def _answer_description(schema: dict[str, Any]) -> str:
    item_type = schema["type"]
    if item_type == "object":
        properties = schema["properties"]
        fields = ", ".join(
            f"{name} ({_answer_description(field)})" for name, field in properties.items()
        )
        description = f"objects with fields {fields}"
    elif item_type == "array":
        description = f"lists of {_answer_description(schema['items'])}"
    else:
        description = {
            "string": "strings",
            "number": "numbers",
            "integer": "integers",
            "boolean": "booleans",
        }[item_type]
    if schema.get("format"):
        description += f" in {schema['format']} format"
    if schema.get("description"):
        description += f" ({schema['description']})"
    return description


def _render_event_value(value: Any, parameters: dict[str, Any]) -> Any:
    """Render network expectations from the same named task parameters as Query."""
    if isinstance(value, str):
        return value.format_map(parameters)
    if isinstance(value, dict):
        return {key: _render_event_value(child, parameters) for key, child in value.items()}
    if isinstance(value, list):
        return [_render_event_value(child, parameters) for child in value]
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _load_drafts(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError("draft JSON must be an array of objects")
        return value
    return _load_jsonl(path)


def _webharbor_record(
    source_root: Path, relative_path: str, record_id: str
) -> tuple[Path, dict[str, Any]]:
    source_root = source_root.resolve()
    artifact = (source_root / relative_path).resolve()
    if not artifact.is_relative_to(source_root):
        raise ValueError("source artifact must stay under the WebHarbor checkout")
    matches = [row for row in _load_jsonl(artifact) if row.get("id") == record_id]
    if len(matches) != 1:
        raise ValueError(f"expected one WebHarbor source record {record_id}, found {len(matches)}")
    return artifact, matches[0]


def _task_spec(draft: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    definition = draft["task_definition"]
    if set(definition) != {
        "task_type",
        "goal_template",
        "parameters",
        "answer",
        "completion_events",
        "dependencies",
    }:
        raise ValueError(
            "task_definition must contain task_type, goal_template, parameters, "
            "answer, completion_events, dependencies"
        )
    parameters = definition["parameters"]
    answer = definition["answer"]
    if set(answer) != {"status", "values", "item_schema", "ordered"}:
        raise ValueError("answer must contain status, values, item_schema, ordered")
    task_type = definition["task_type"]
    if task_type not in {"RETRIEVE", "NAVIGATE", "MUTATE"}:
        raise ValueError("task_type must be RETRIEVE, NAVIGATE, or MUTATE")
    intent = definition["goal_template"].format_map(parameters)
    intent_template = definition["goal_template"]
    if task_type == "RETRIEVE":
        item_schema = answer["item_schema"]
        if not isinstance(item_schema, dict) or item_schema.get("type") not in {
            "string",
            "number",
            "boolean",
            "object",
        }:
            raise ValueError(
                "RETRIEVE item_schema must describe strings, numbers, booleans, or objects"
            )
        result_instruction = (
            " Return the results in the retrieved_data field of the final JSON object "
            f"as a list of {_answer_description(item_schema)}."
        )
        if answer["ordered"]:
            result_instruction += " Preserve the requested order."
        intent += result_instruction
        intent_template += result_instruction
    elif answer["item_schema"] is not None or answer["values"] is not None:
        raise ValueError("NAVIGATE and MUTATE require null item_schema and values")

    response = {
        "task_type": task_type,
        "status": answer["status"],
        "retrieved_data": answer["values"],
        "error_details": None,
    }
    results_schema = (
        {"type": "array", "items": answer["item_schema"]}
        if task_type == "RETRIEVE" and answer["status"] == "SUCCESS"
        else {"type": "null"}
    )
    Draft202012Validator.check_schema(results_schema)
    Draft202012Validator(results_schema).validate(answer["values"])
    # Original bench leaves error_details unscored; retain it in the public response only.
    expected = {key: value for key, value in response.items() if key != "error_details"}
    evaluators = [
        {
            "evaluator": "AgentResponseEvaluator",
            "ordered": answer["ordered"],
            "results_schema": results_schema,
            "expected": expected,
        }
    ]
    for event in definition["completion_events"]:
        expected_keys = {
            "http_method",
            "response_status",
            "headers",
            "query_params",
            "post_data",
            "response_content",
            "response_cookies",
        }
        config_keys = {
            "last_event_only",
            "ignored_query_params",
            "ignored_query_params_patterns",
            "decode_base64_query",
            "query_params_schema",
            "post_data_schema",
            "ignored_post_data_params_patterns",
            "should_not_exist",
        }
        unexpected = set(event) - expected_keys - config_keys - {"url_template"}
        if unexpected:
            raise ValueError(f"unsupported completion event fields: {sorted(unexpected)}")
        event_expected = {"url": _render_event_value(event["url_template"], parameters)}
        event_expected.update(
            {
                key: _render_event_value(value, parameters)
                for key, value in event.items()
                if key in expected_keys
            }
        )
        evaluators.append(
            {
                "evaluator": "NetworkEventEvaluator",
                "last_event_only": False,
                **{key: value for key, value in event.items() if key in config_keys},
                "expected": event_expected,
            }
        )

    return {
        "sites": draft["sites"],
        "task_id": draft["task_id"],
        "intent_template_id": draft["intent_template_id"],
        "start_urls": draft["start_urls"],
        "intent": intent,
        "eval": evaluators,
        "intent_template": intent_template,
        "instantiation_dict": parameters,
        "revision": draft["revision"],
    }, response


def compile_candidate(draft: dict[str, Any], source_root: Path, batch_root: Path) -> dict[str, Any]:
    artifact, source_record = _webharbor_record(
        source_root, draft["source_artifact"], draft["source_record_id"]
    )
    task_spec, reference_response = _task_spec(draft)
    generated_task_id = draft["generated_task_id"]
    grounding = dict(draft["grounding"])
    artifact_paths = grounding.get("artifacts")
    if not isinstance(artifact_paths, dict) or set(artifact_paths) != {
        "har",
        "trace",
        "screenshot",
        "observation",
    }:
        raise ValueError("grounding artifacts must name har, trace, screenshot, and observation")
    artifact_hashes = {}
    for name, relative in artifact_paths.items():
        path = (batch_root / relative).resolve()
        if not path.is_relative_to(batch_root.resolve()) or not path.is_file():
            raise ValueError(f"invalid grounding artifact path: {relative}")
        artifact_hashes[name] = sha256_file(path)
    grounding["artifact_sha256"] = artifact_hashes
    grounding["task_spec_sha256"] = sha256_json(task_spec)
    field_evidence = grounding["field_evidence"]
    if not isinstance(field_evidence, list) or not field_evidence:
        raise ValueError("grounding requires non-empty field_evidence")
    resolved_evidence = []
    for record in field_evidence:
        pointer = record["field"]
        if not pointer.startswith("/task_definition/"):
            raise ValueError("field evidence must point inside /task_definition/")
        value: Any = draft
        for token in pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            value = value[int(token)] if isinstance(value, list) else value[token]
        if not record.get("location") or record["modality"] not in {"text", "image"}:
            raise ValueError("field evidence must identify location and text/image modality")
        resolved_evidence.append(
            {
                **record,
                "field_value_sha256": sha256_json(value),
                "artifact_sha256": artifact_hashes[record["artifact"]],
            }
        )
    required_fields: list[str] = []
    pending = (
        []
        if reference_response["retrieved_data"] is None
        else [("/task_definition/answer/values", reference_response["retrieved_data"])]
    )
    for dependency in draft["task_definition"]["dependencies"]:
        names = [
            dependency["source_url_parameter"],
            *dependency["source_entity"].values(),
            *dependency["fact_readouts"],
        ]
        names.extend(
            re.findall(
                r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}",
                canonical_json(
                    draft["task_definition"]["completion_events"][dependency["target_event_index"]]
                ),
            )
        )
        pending.extend(
            (f"/task_definition/parameters/{name}", draft["task_definition"]["parameters"][name])
            for name in names
        )
    if reference_response["retrieved_data"] is not None or draft["task_definition"]["dependencies"]:
        while pending:
            pointer, value = pending.pop()
            if isinstance(value, (list, dict)):
                entries = enumerate(value) if isinstance(value, list) else value.items()
                pending.extend(
                    (f"{pointer}/{str(key).replace('~', '~0').replace('/', '~1')}", child)
                    for key, child in entries
                )
            else:
                required_fields.append(pointer)
    missing_evidence = set(required_fields) - {record["field"] for record in resolved_evidence}
    if missing_evidence:
        raise ValueError(
            f"answer or dependency leaves lack field evidence: {sorted(missing_evidence)}"
        )
    for record in resolved_evidence:
        if "rule" not in record:
            continue
        rule = record["rule"]
        if rule["operation"] in {"format", "sum", "multiply", "ratio_percent"}:
            named = draft["task_definition"]["parameters"]
            inputs = rule["parameters"]
            if any(
                f"/task_definition/parameters/{name}"
                not in {item["field"] for item in resolved_evidence}
                for name in inputs
            ):
                raise ValueError("derived values require evidence for every input")
            rendered = derive_value(rule, named)
            if sha256_json(rendered) != record["field_value_sha256"]:
                raise ValueError("derived rule does not reproduce its semantic value")
        else:
            raise ValueError("unsupported field-evidence derivation rule")
    remaining = {
        record["field"].removeprefix("/task_definition/parameters/"): set(
            record["rule"]["parameters"]
        )
        for record in resolved_evidence
        if "rule" in record
    }
    resolved = set(draft["task_definition"]["parameters"]) - set(remaining)
    while remaining:
        ready = {name for name, inputs in remaining.items() if inputs <= resolved}
        if not ready:
            raise ValueError("parameter derivations must form an acyclic grounded graph")
        resolved.update(ready)
        for name in ready:
            del remaining[name]
    grounding["field_evidence"] = resolved_evidence
    baseline = draft["baseline"]
    candidate = {
        "generated_task_id": generated_task_id,
        "generation_mode": "regrounded_semantic_seed",
        "source": {
            "dataset": "aiming-lab/WebHarbor",
            "repository_commit": WEBHARBOR_COMMIT,
            "artifact_path": draft["source_artifact"],
            "artifact_sha256": sha256_file(artifact),
            "record_id": draft["source_record_id"],
            "record_sha256": sha256_json(source_record),
            "reuse_scope": "abstract_task_family_only",
        },
        "baseline": baseline,
        "baseline_sha256": sha256_json(baseline),
        "grounding": grounding,
        "grounding_sha256": sha256_json(grounding),
        "reference_response": reference_response,
        "task_definition": draft["task_definition"],
        "task_definition_sha256": sha256_json(draft["task_definition"]),
        "task_spec": task_spec,
    }
    validate_candidate_shape(candidate)
    validate_upstream_task_spec(task_spec)
    return candidate


def main() -> None:
    args = _parse_args()
    source_manifest = args.source_root / "source-manifest.json"
    if sha256_file(source_manifest) != WEBHARBOR_MANIFEST_SHA256:
        raise ValueError("Bundled semantic seed snapshot manifest differs")
    manifest = json.loads(source_manifest.read_text())
    for relative, expected in manifest["files"].items():
        if sha256_file(args.source_root / relative) != expected:
            raise ValueError("Bundled semantic seed file differs: " + relative)

    drafts = _load_drafts(args.drafts)
    candidates = [compile_candidate(draft, args.source_root, args.batch_root) for draft in drafts]
    for identity in ("generated_task_id", "task_id"):
        values = [
            candidate[identity]
            if identity == "generated_task_id"
            else candidate["task_spec"][identity]
            for candidate in candidates
        ]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {identity} in authoring batch")
    families: dict[int, str] = {}
    for candidate in candidates:
        definition = candidate["task_definition"]
        family = {
            "site": candidate["task_spec"]["sites"],
            "task_type": definition["task_type"],
            "goal_template": definition["goal_template"],
            "answer": {
                key: value for key, value in definition["answer"].items() if key != "values"
            },
        }
        template_id = candidate["task_spec"]["intent_template_id"]
        fingerprint = sha256_json(family)
        if template_id in families and families[template_id] != fingerprint:
            raise ValueError(f"template ID {template_id} describes different task families")
        families[template_id] = fingerprint
    public = [public_projection(candidate) for candidate in candidates]
    args.private.parent.mkdir(parents=True, exist_ok=True)
    args.public.parent.mkdir(parents=True, exist_ok=True)
    args.private.write_text("".join(canonical_json(candidate) + "\n" for candidate in candidates))
    args.public.write_text("".join(canonical_json(row) + "\n" for row in public))
    print(f"compiled={len(candidates)} private={args.private} public={args.public}")


if __name__ == "__main__":
    main()
