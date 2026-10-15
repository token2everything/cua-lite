"""Clean native JsonlTrajectoryLogger rows and publish reversible V3 candidates.

Use the pinned InternFlow environment with ``python -B``. The JSONL input
manifest explicitly supplies task_id, trajectory, request_dir, score_receipt,
public_tasks and canonical_tools. Formal rows also supply model_execution and
its frozen SHA256. Native identity, all API requests/returns and successful
owner scoring must match that teacher snapshot before strict V3/round-trip audit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
from clean_v3_projection import audit_roundtrip, project_v3, visible_content
from internflow.formats import build_default_format_registry
from internflow.formats.contracts import (
    AdmissionConfig,
    DeclaredFormatRejection,
    EmptyFormatParams,
    FinalConversionDefinition,
    FinalFormatDefinition,
    FinalizationConfig,
)
from internflow.operations.executor import execute_resolved_invocation
from internflow.operations.models import OperationContext, PipelineDefinition
from internflow.operations.planner import prepare_pipeline_run
from internflow.operations.profiles import ProfileRepository
from internflow.operations.registry import build_default_operation_registry
from internvalidata.data_models.messages_types.agent.constants import ROLLOUT_MODEL_ALLOWLIST
from internvalidata.data_models.messages_types.multimodal_image_text_messages import (
    MultimodalImageTextMessagesV3,
)
from PIL import Image
from pydantic import ValidationError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--formal", action="store_true")
    scope.add_argument("--smoke", action="store_true", help="Never counted as formal v0.3 data.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Hash provenance artifacts without loading large HAR/trace files into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_rewrite_score(
    entry: dict, score: dict, native: dict, public: dict, score_path: Path,
) -> dict:
    """Verify official completion evidence without claiming independent state read-back.

    PASS is the official all-or-nothing score on both network representations,
    with equal normalized evaluator contracts and a valid submitted response.
    This ingestion owner accepts benchmark-derived, potentially multi-site tasks;
    generated candidates retain their separate grounding and final-state gate.
    """
    if score["evidence_kind"] != "official_task_rewrite":
        raise ValueError("Official rewrite manifest and receipt evidence kinds differ")
    required_artifacts = {
        "source_task", "source_dataset", "public_task", "evaluation", "status",
        "agent_response", "agent_response_raw", "baseline", "instruction", "har", "trace",
        "original_instruction", "instruction_text",
        "native_jsonl", "canonical_tools",
    }
    if not required_artifacts <= score["artifacts"].keys():
        raise ValueError("Official rewrite receipt lacks its source or scoring artifacts")
    artifacts = {}
    for name, relative in score["artifacts"].items():
        path = (score_path.parent / relative).resolve()
        path.relative_to(score_path.parent)
        if sha256_file(path) != score["artifact_sha256"][name]:
            raise ValueError(f"Official rewrite artifact differs from receipt: {name}")
        artifacts[name] = path
    native_lines = artifacts["native_jsonl"].read_text().splitlines()
    if len(native_lines) != 1 or json.loads(native_lines[0]) != native:
        raise ValueError("Official rewrite native episode differs from its scored artifact")
    if json.loads(artifacts["canonical_tools"].read_text()) != native["metadata"]["tools"]:
        raise ValueError("Official rewrite tool contract differs from its scored native episode")
    dataset_sha = sha256_file(artifacts["source_dataset"])
    if dataset_sha != "d65275660814663375028e9017e1f929e3c38321041b125795e2713b52243d30":
        raise ValueError("Official rewrite source dataset differs from the pinned 812-task catalog")
    source = json.loads(artifacts["source_task"].read_text())
    official_id = score["official_task_id"]
    catalog = json.loads(artifacts["source_dataset"].read_text())
    if (
        type(official_id) is not int
        or not 0 <= official_id < 812
        or catalog[official_id] != source
    ):
        raise ValueError("Official rewrite source task differs from its pinned catalog record")
    if source["task_id"] != official_id:
        raise ValueError("Official rewrite source task identity differs from the receipt")
    if set(public) != {"source_task_id", "official_task_id", "sites", "start_urls", "intent"}:
        raise ValueError("Official rewrite public task must be its exact public projection")
    if (
        public["source_task_id"] != entry["task_id"]
        or score["source_task_id"] != entry["task_id"]
        or public["official_task_id"] != official_id
        or public["sites"] != source["sites"]
        or not public["sites"]
        or not isinstance(public["intent"], str)
        or not public["intent"].strip()
        or public["intent"] == source["intent"]
    ):
        raise ValueError("Official rewrite public identity or rewritten intent is invalid")
    if json.loads(artifacts["public_task"].read_text()) != public:
        raise ValueError("Official rewrite archived public task differs from the manifest")
    instruction = json.loads(artifacts["instruction"].read_text())
    if (
        instruction["intent"] != public["intent"]
        or instruction["start_urls"] != public["start_urls"]
    ):
        raise ValueError("Official rewrite model instruction differs from its public projection")
    original_instruction = json.loads(artifacts["original_instruction"].read_text())
    if original_instruction != {**instruction, "intent": source["intent"]}:
        raise ValueError("Official rewrite changed original instruction fields besides intent")
    instruction_text = artifacts["instruction_text"].read_text().removesuffix("\n")
    initial_user = next(message for message in native["messages"] if message["role"] == "user")
    initial_text = "".join(
        part["text"] for part in initial_user["content"] if part["type"] == "text"
    )
    if initial_text != instruction_text or f"Task: {public['intent']}\n\n" not in initial_text:
        raise ValueError("Official rewrite native model input differs from its bound instruction")
    baseline = json.loads(artifacts["baseline"].read_text())
    for field, value in (("task_spec_sha256", source), ("baseline_sha256", baseline)):
        digest = hashlib.sha256(json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        if score[field] != digest:
            raise ValueError(f"Official rewrite receipt differs from its {field} artifact")
    if baseline["task_id"] != official_id or baseline["fresh_service_processes"] is not True:
        raise ValueError("Official rewrite baseline is not fresh or belongs to another task")
    evaluation = json.loads(artifacts["evaluation"].read_text())
    status = json.loads(artifacts["status"].read_text())
    response = json.loads(artifacts["agent_response"].read_text())
    raw_response = artifacts["agent_response_raw"].read_text()
    final_assistant = next(
        message for message in reversed(native["messages"]) if message["role"] == "assistant"
    )
    final_calls = final_assistant.get("tool_calls", [])
    if final_calls:
        submitted = [call["function"]["arguments"]["text"] for call in final_calls
                     if call["function"]["name"] == "response"]
        if submitted != [raw_response]:
            raise ValueError("Official rewrite scored response differs from the native final call")
    elif "".join(
        part["text"] for part in final_assistant["content"] if part["type"] == "text"
    ) != raw_response:
        raise ValueError("Official rewrite scored response differs from the native final answer")
    if (
        score["status"] != "pass" or score["execution_status"] != "completed"
        or status["status"] != "completed" or status.get("error")
        or status["task_id"] != official_id or status["model"] != score["model_id"]
        or status["score"] != 1 or status["truncated"] is not False
        or evaluation["score"] != 1 or evaluation["integrity_status"] != "passed"
        or evaluation["publishable"] is not True
        or evaluation["inputs_agree"] is not True
        or evaluation["score_inputs_agree"] is not True
        or response["valid"] is not True or response["raw_content"] != raw_response
    ):
        raise ValueError("Official rewrite episode did not complete with a valid official PASS")
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_response, re.DOTALL)
    parsed = json.loads(match.group(1) if match else raw_response)
    normalized_response = {
        "task_type": parsed["task_type"].upper(), "status": parsed["status"].upper(),
        "retrieved_data": parsed.get("retrieved_data"),
        "error_details": parsed.get("error_details"),
    }
    if (
        normalized_response != response["validated_projection"]
        or set(response["validated_projection"])
        != {"task_type", "status", "retrieved_data", "error_details"}
    ):
        raise ValueError("Official rewrite final response differs from its validated projection")
    try:
        jsonschema.validate(normalized_response, instruction["final_response_schema"])
    except jsonschema.ValidationError as error:
        raise ValueError(
            "Official rewrite final response violates its original response schema"
        ) from error
    har = evaluation["har"]
    trace = evaluation["playwright_trace"]
    contract_fields = {
        "task_id", "intent_template_id", "sites", "task_revision", "status", "score",
        "error_msg", "webarena_verified_version", "webarena_verified_evaluator_checksum",
        "webarena_verified_data_checksum", "evaluators_results",
    }
    normalized = []
    for result in (har, trace):
        if not contract_fields <= result.keys():
            raise ValueError("Official rewrite evaluator returned an incomplete task contract")
        if (
            result["task_id"] != official_id
            or result["intent_template_id"] != source["intent_template_id"]
            or result["sites"] != source["sites"]
            or result["task_revision"] != source["revision"]
            or result["status"] != "success" or result["score"] != 1
            or result["webarena_verified_version"] != "1.2.3"
            or result["webarena_verified_data_checksum"] != dataset_sha
            or result["webarena_verified_evaluator_checksum"]
            != "35c3385b1db4b3378657589f95f50defd4234bd36e5b93d44733fd561b01db4e"
        ):
            raise ValueError("Official rewrite evaluator identity or score differs from its source")
        rows = result["evaluators_results"]
        evaluator_fields = {
            "evaluator_name", "status", "score", "actual", "actual_normalized",
            "expected", "assertions", "error_msg", "should_not_exist",
        }
        if len(rows) != len(source["eval"]) or any(
            not evaluator_fields <= row.keys()
            or row["evaluator_name"] != definition["evaluator"]
            or row["status"] != "success" or row["score"] != 1
            for row, definition in zip(rows, source["eval"], strict=True)
        ):
            raise ValueError("Official rewrite did not pass every original evaluator")
        # Raw actual network objects differ by recording format; normalized
        # values, expectations, assertions and every task-level field must agree.
        normalized.append({**result, "evaluators_results": [
            {key: value for key, value in row.items() if key != "actual"} for row in rows
        ]})
    if normalized[0] != normalized[1]:
        raise ValueError("Official rewrite HAR and trace normalized contracts disagree")
    for name, artifact_name in (
        ("network_har", "har"), ("playwright_trace", "trace"),
        ("agent_response_raw", "agent_response_raw"),
    ):
        identity = evaluation["input_receipt"][name]
        path = artifacts[artifact_name]
        if identity["sha256"] != sha256_file(path) or identity["size"] != path.stat().st_size:
            raise ValueError("Official rewrite score input differs from the executed artifact")
    return {
        "evidence_kind": "official_task_rewrite", "official_task_id": official_id,
        "benchmark_derived": True, "evaluation_scope": "evaluation_only",
        "independent_final_state": "not_verified",
        "score_checks": {
            "response_contract_valid": True, "all_evaluators_pass": True,
            "har_playwright_results_equal": True, "official_source_unchanged": True,
            "public_projection_bound": True,
        },
        "pass_rule": "official score == 1 on HAR and trace; every original evaluator succeeds; "
        "normalized contracts agree; submitted final response is valid",
    }


def verify_execution_evidence(entry: dict, native: dict, public: dict, formal: bool) -> dict:
    """Bind actual teacher request/response logs and owner score to this task."""
    score_path = Path(entry["score_receipt"]).resolve()
    score = json.loads(score_path.read_text())
    official = entry.get("evidence_kind") == "official_task_rewrite"
    rewrite_evidence = {}
    if official:
        if not formal:
            raise ValueError("Official rewrite ingestion requires formal execution evidence")
        rewrite_evidence = verify_official_rewrite_score(entry, score, native, public, score_path)
        checks = rewrite_evidence["score_checks"]
    else:
        for key in (
            "response_contract_valid",
            "all_evaluators_pass",
            "har_playwright_results_equal",
            "final_state_confirmed",
        ):
            if score["checks"][key] is not True:
                raise ValueError(f"Owner scoring check did not pass: {key}")
        checks = score["checks"]
        if (
            score["generated_task_id"] != entry["task_id"]
            or score["status"] != "pass"
            or any(value is not True for value in checks.values())
        ):
            raise ValueError("Owner scoring receipt does not pass all required checks")
    public_sha = hashlib.sha256(
        json.dumps(
            public,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if score["public_projection_sha256"] != public_sha:
        raise ValueError("Scoring receipt is bound to a different public task")
    model_id = native["metadata"]["model_id"]
    api_type = entry["api_type"]
    expected_teacher = None
    execution_sha = None
    if formal:
        execution_path = Path(entry["model_execution"]).resolve()
        execution_sha = sha256_file(execution_path)
        if execution_sha != entry["model_execution_sha256"]:
            raise ValueError("Teacher execution snapshot differs from the frozen manifest hash")
        expected_teacher = json.loads(execution_path.read_text())["teacher"]
        if (
            model_id != expected_teacher["requested_model"]
            or api_type != expected_teacher["api_type"]
        ):
            raise ValueError("Native teacher identity differs from the frozen batch teacher")
    if formal and (score["model_id"] != model_id or score["api_type"] != api_type):
        raise ValueError("Owner score teacher identity differs from native source")
    if not formal and f"openai/{model_id}" not in ROLLOUT_MODEL_ALLOWLIST:
        raise ValueError(f"Real source teacher is outside pinned allowlist: {model_id}")
    request_dir = Path(entry["request_dir"]).resolve()
    request_files = sorted(request_dir.glob("*.request.json"))
    if not request_files:
        raise ValueError("Real model request logs are required")
    assistant_count = sum(message["role"] == "assistant" for message in native["messages"])
    if len(request_files) != assistant_count:
        raise ValueError("Actual model requests do not align with native assistant decisions")
    responses = []
    for request_path in request_files:
        request = json.loads(request_path.read_text())
        if official and request_path == request_files[0]:
            request_user = next(
                message for message in request["messages"] if message["role"] == "user"
            )
            request_text = "".join(
                part["text"] for part in request_user["content"] if part["type"] == "text"
            )
            native_user = next(
                message for message in native["messages"] if message["role"] == "user"
            )
            native_text = "".join(
                part["text"] for part in native_user["content"] if part["type"] == "text"
            )
            if request_text != native_text:
                raise ValueError(
                    "Official rewrite first request differs from its bound native query"
                )
        response_path = request_path.with_name(
            request_path.name.replace(".request.json", ".response.json")
        )
        response = json.loads(response_path.read_text())
        if request["model"] != model_id:
            raise ValueError("Recorded requested model differs from native teacher")
        returned = response["model"]
        if formal and returned != expected_teacher["returned_probe_model"]:
            raise ValueError("Actual returned teacher differs from the frozen probe identity")
        if formal:
            history_key = "messages" if api_type == "openai_chat" else "input"
            if set(request) != {"model", "tools", history_key, *expected_teacher["params"]}:
                raise ValueError("Actual request parameter keys differ from the frozen teacher")
            for name, value in expected_teacher["params"].items():
                if request[name] != value:
                    raise ValueError(
                        f"Actual model parameter differs from the frozen teacher: {name}"
                    )
        if not formal and returned != model_id and not returned.startswith(model_id + "-"):
            raise ValueError("Technical-smoke returned model differs from its actual source")
        if api_type == "openai_responses":
            if response["status"] != "completed" or not isinstance(response["output"], list):
                raise ValueError("Recorded Responses output did not complete")
        elif api_type == "openai_chat":
            if not response["choices"] or response["choices"][0]["finish_reason"] not in {
                "stop",
                "tool_calls",
            }:
                raise ValueError("Recorded Chat response is incomplete")
        else:
            raise ValueError(f"Explicit evidence parser does not support api_type={api_type}")
        responses.append(returned)
    if formal and not official:
        for name, relative in score["artifacts"].items():
            artifact = score_path.parent / relative
            if sha256_file(artifact) != score["artifact_sha256"][name]:
                raise ValueError(f"Scored reference artifact differs from receipt: {name}")
    return {
        **rewrite_evidence,
        "model_id": model_id,
        "api_type": api_type,
        "returned_models": sorted(set(responses)),
        "source_api_model_id": model_id,
        "model_execution_sha256": execution_sha,
        "expected_teacher": expected_teacher,
        "request_count": len(request_files),
        "score_checks": checks,
        "score_receipt_sha256": sha256_file(score_path),
        "public_projection_sha256": public_sha,
        "task_spec_sha256": score["task_spec_sha256"],
        "baseline_sha256": score["baseline_sha256"],
        "request_evidence_sha256": {
            str(path.relative_to(request_dir)): sha256_file(path)
            for path in sorted(request_dir.rglob("*"))
            if path.is_file()
        },
        "files": sorted(path for path in request_dir.rglob("*") if path.is_file()),
    }


def main() -> int:
    """Run deterministic cleaning while preserving a whole native trajectory."""
    args = _parse_args()
    versions = {name: importlib.metadata.version(name) for name in ("internflow", "internvalidata")}
    if versions != {"internflow": "0.2.0", "internvalidata": "0.4.17"}:
        raise ValueError(f"Wrong pinned cleaning versions: {versions}")
    entries = [json.loads(line) for line in args.input_manifest.open() if line.strip()]
    if len({entry["task_id"] for entry in entries}) != len(entries):
        raise ValueError("Input manifest must contain unique source tasks")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("input", "original", "images", "processed", "rejected", "audits"):
        (output / name).mkdir()
    shutil.copyfile(args.input_manifest, output / "input_manifest.jsonl")
    shutil.copyfile(args.profile, output / "profile.yaml")
    execution_sha = None
    execution = None
    if args.formal:
        (output / "model_executions").mkdir()
        contract = None
        snapshots = {}
        for entry in entries:
            execution_path = Path(entry["model_execution"]).resolve()
            digest = sha256_file(execution_path)
            if digest != entry["model_execution_sha256"]:
                raise ValueError("Formal teacher snapshot hash differs from the manifest")
            current = json.loads(execution_path.read_text())
            comparable = {key: value for key, value in current.items() if key != "student"}
            if contract is not None and comparable != contract:
                raise ValueError("Formal sources use different frozen teacher execution contracts")
            contract = comparable
            relative = "model_executions/" + digest + ".json"
            if digest not in snapshots:
                shutil.copyfile(execution_path, output / relative)
            snapshots[digest] = relative
        execution = {**contract, "source_executions": snapshots}
        (output / "model_execution.json").write_text(json.dumps(execution, indent=2) + "\n")
        execution_sha = sha256_file(output / "model_execution.json")
    formats = build_default_format_registry()
    formats.register_final(
        FinalFormatDefinition(
            key="multimodal_image_text_dialogs_v3",
            contract_version="v0.4.17",
            model=MultimodalImageTextMessagesV3,
            serialize_by_alias=True,
        )
    )
    formats.register_final_conversion(
        FinalConversionDefinition(
            key="native_agent_pipeline_record_to_image_v3",
            contract_version="v2.1.0",
            pipeline_record_format="agent_pipeline_record",
            final_format="multimodal_image_text_dialogs_v3",
            params_model=EmptyFormatParams,
            converter=project_v3,
        )
    )
    operations = build_default_operation_registry()
    repository = ProfileRepository([args.profile.parent])
    profile = repository.load_file(args.profile)
    plan = prepare_pipeline_run(
        pipeline=PipelineDefinition.model_validate({"blocks": [{"profile": profile.profile.name}]}),
        registry=operations,
        profile_repository=repository,
        format_registry=formats,
        admission=AdmissionConfig(source_format="draft_agent_sample"),
        finalization=FinalizationConfig(final_format="multimodal_image_text_dialogs_v3"),
        apply_ref_use_loss_seq=True,
    )
    (output / "prepared_plan.json").write_text(plan.model_dump_json(indent=2))
    summary = {
        "scope": "v03_formal" if args.formal else "technical_smoke_not_v03_data",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "source_format": "JsonlTrajectoryLogger.native_viewer",
        "format_projection": "reversible_image_v3_v2_1",
        "input_trajectories": len(entries),
        "accepted": 0,
        "rejected": 0,
        "groups": len(plan.groups),
        "operations": sum(len(group.operations) for group in plan.groups),
        "plan_hash": plan.plan_hash,
        "review_status": "pending_human_review",
        "training_promotion": "not_promoted",
        "rows": [],
        "model_execution_sha256": execution_sha,
        "expected_teacher": execution["teacher"] if execution is not None else None,
    }
    manifest_images = {}
    for entry in entries:
        task_id = entry["task_id"]
        receipt = {"task_id": task_id, "accepted": False}
        try:
            trajectory = Path(entry["trajectory"]).resolve()
            lines = [line for line in trajectory.read_text().splitlines() if line.strip()]
            if len(lines) != 1:
                raise ValueError("Native viewer source must contain exactly one complete row")
            native = json.loads(lines[0])
            if native["metadata"]["status"] != "completed" or not native["messages"]:
                raise ValueError("Native trajectory is not completed")
            public_matches = []
            identity = (
                "source_task_id"
                if entry.get("evidence_kind") == "official_task_rewrite"
                else "generated_task_id"
            )
            for line in Path(entry["public_tasks"]).open():
                public_row = json.loads(line)
                if public_row[identity] == task_id:
                    public_matches.append(public_row)
            if len(public_matches) != 1 or (
                len(public_matches[0]["sites"]) != 1
                and entry.get("evidence_kind") != "official_task_rewrite"
            ):
                raise ValueError(
                    "Manifest task must resolve to one generated single-site public task"
                )
            public = public_matches[0]
            evidence = verify_execution_evidence(entry, native, public, args.formal)
            original_dir = output / "original" / task_id
            original_dir.mkdir()
            shutil.copyfile(trajectory, original_dir / "agent_trajectory.jsonl")
            shutil.copyfile(entry["score_receipt"], original_dir / "reference_receipt.json")
            shutil.copyfile(entry["canonical_tools"], original_dir / "canonical_tools.json")
            (original_dir / "public_task.json").write_text(json.dumps(public, ensure_ascii=False))
            logs_dir = original_dir / "requests"
            logs_dir.mkdir()
            for path in evidence["files"]:
                destination = logs_dir / path.relative_to(Path(entry["request_dir"]).resolve())
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
            if args.formal:
                score = json.loads(Path(entry["score_receipt"]).read_text())
                for relative in score["artifacts"].values():
                    artifact = Path(entry["score_receipt"]).parent / relative
                    destination = original_dir / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(artifact, destination)
            canonical_tools = json.loads(Path(entry["canonical_tools"]).read_text())
            if not isinstance(canonical_tools, list):
                raise ValueError("Canonical tools must be the actual frozen Lite schema array")
            if args.formal and native["metadata"]["tools"] != canonical_tools:
                raise ValueError("Formal native tool contract differs from the frozen owner schema")
            aliases = {}
            for message in native["messages"]:
                if not isinstance(message.get("content"), list):
                    continue
                for part in message["content"]:
                    if part["type"] != "image":
                        continue
                    source_url = part["image"]["url"]
                    path = Path(source_url)
                    if "media_mount" in entry:
                        mapping = entry["media_mount"]
                        source_root = Path(mapping["source_root"])
                        if not path.is_absolute():
                            path = source_root / path
                        relative_source = path.relative_to(source_root)
                        path = Path(mapping["local_root"]) / relative_source
                    elif not path.is_absolute():
                        path = trajectory.parent / path
                    payload = path.read_bytes()
                    digest = hashlib.sha256(payload).hexdigest()
                    with Image.open(path) as image:
                        image.load()
                        dimensions = list(image.size)
                    if dimensions != list(part["image"]["image_wh"]):
                        raise ValueError("Native screenshot dimensions differ from actual bytes")
                    relative = f"images/{digest}{path.suffix.lower()}"
                    if relative not in manifest_images:
                        shutil.copyfile(path, output / relative)
                        manifest_images[relative] = {
                            "sha256": digest,
                            "bytes": len(payload),
                            "image_wh": dimensions,
                        }
                    aliases[source_url] = {
                        "url": relative,
                        "sha256": digest,
                        "image_wh": dimensions,
                    }
            # Native/request log writers share the run's images directory. Mirror
            # every actual file so unchanged ../images references in raw requests
            # and images references in the native row resolve in the archive.
            local_media_root = (
                Path(entry["media_mount"]["local_root"])
                if "media_mount" in entry
                else trajectory.parent
            )
            source_media_archive = {}
            native_media_dir = local_media_root / "images"
            if native_media_dir.is_dir():
                for path in sorted(native_media_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    relative_media = path.relative_to(local_media_root)
                    destination = original_dir / relative_media
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
                    source_media_archive[str(path.relative_to(native_media_dir))] = {
                        "url": str(destination.relative_to(output)),
                        "sha256": sha256_file(path),
                    }
            calls = {
                call["id"]: call["function"]["name"]
                for message in native["messages"]
                for call in message.get("tool_calls", [])
            }
            messages = []
            for index, original in enumerate(native["messages"]):
                message = {
                    "role": original["role"],
                    "content": visible_content(original, aliases),
                    "meta": {"native_boundary": {"source_index": index}},
                }
                for name in ("name", "reasoning_content", "tool_calls", "loss"):
                    if name in original:
                        message[name] = copy.deepcopy(original[name])
                if message["role"] in {"assistant", "system"} and isinstance(
                    message["content"], list
                ):
                    message["content"] = "".join(part["text"] for part in message["content"])
                if message["role"] == "assistant":
                    message["loss"] = message.get("loss", True)
                elif message["role"] == "tool":
                    message["tool_call_id"] = original["tool_call_id"]
                    message["name"] = calls[original["tool_call_id"]]
                if isinstance(message["content"], list):
                    for part in message["content"]:
                        if part["type"] == "image":
                            part["image"]["image_wh"] = tuple(part["image"]["image_wh"])
                messages.append(message)
            evidence_for_metadata = {
                key: value for key, value in evidence.items() if key != "files"
            }
            # The intermediate optional enum label cannot identify this new
            # teacher. Preserve its real API identity separately; final V3's
            # unrestricted model_name receives that actual source identifier.
            model_metadata = (
                {"params": evidence["expected_teacher"]["params"]}
                if args.formal
                else {"model_name": f"openai/{evidence['model_id']}"}
            )
            source = {
                "id": task_id,
                "schema_version": "v0.4.17",
                "provider": "cua-lite",
                "project": "agent-tool_use",
                "for_agent": True,
                "for_rl": False,
                "messages": messages,
                "tools": canonical_tools,
                "metadata": {
                    "source": (
                        "webarena_verified_official_task_rewrite"
                        if evidence.get("evidence_kind") == "official_task_rewrite"
                        else "webarena_verified_generated"
                    ),
                    **({key: evidence[key] for key in (
                        "evidence_kind", "official_task_id", "benchmark_derived",
                        "evaluation_scope",
                        "independent_final_state",
                    )} if evidence.get("evidence_kind") == "official_task_rewrite" else {}),
                    **({"training_promotion": "not_promoted"}
                       if evidence.get("evidence_kind") == "official_task_rewrite" else {}),
                    "language": "en",
                    "user_query": public["intent"],
                    "model": model_metadata,
                    "source_model_id": evidence["model_id"],
                    "source_api_type": evidence["api_type"],
                    "source_api_model_id": evidence["source_api_model_id"],
                    "source_returned_models": evidence["returned_models"],
                    "source_task_id": task_id,
                    "sites": public["sites"],
                    "scope": summary["scope"],
                    "source_trajectory_sha256": sha256_file(trajectory),
                    "image_aliases": aliases,
                    "native_messages": native["messages"],
                    "native_row_fields": {
                        key: value for key, value in native.items() if key != "messages"
                    },
                    "canonical_tools_sha256": sha256_file(Path(entry["canonical_tools"])),
                    "source_media_mount": entry.get("media_mount"),
                    "source_media_archive": source_media_archive,
                    "execution_evidence": evidence_for_metadata,
                    "review_status": "pending_human_review",
                },
            }
            if evidence["api_type"] == "openai_chat":
                source["metadata"]["harness"] = {"harness_name": "openai_chat"}
            (output / "input" / f"{task_id}.json").write_text(
                json.dumps(source, ensure_ascii=False)
            )
            record = formats.admit("draft_agent_sample", source)
            source_messages = record.model_dump(mode="python", exclude_none=True)["messages"]
            for group_index, group in enumerate(plan.groups, start=1):
                for invocation in group.operations:
                    result = execute_resolved_invocation(
                        registry=operations,
                        invocation=invocation,
                        context=OperationContext(
                            sample=record,
                            invocation_id=invocation.invocation_id,
                            resources=None,
                            file_name=trajectory.name,
                            sample_index=0,
                        ),
                        format_registry=formats,
                        dependency_graph=plan.dependency_graph,
                    )
                    if len(result.emissions) != 1:
                        raise ValueError("Whole-trajectory fixed rules must remain one-to-one")
                    emission = result.emissions[0]
                    record = emission.sample
                    if not emission.finding.passed:
                        raise ValueError(
                            f"Cleaning rejected at {invocation.operation_key}: "
                            f"{emission.finding.reason}"
                        )
                stage = output / "processed" / f"{group_index:03d}_{group.name}"
                stage.mkdir(exist_ok=True)
                (stage / f"{task_id}.json").write_text(record.model_dump_json())
            cleaned_messages = record.model_dump(mode="python", exclude_none=True)["messages"]
            # Cleaning may recommend loss masks, but actual calls and content stay intact.
            if cleaned_messages != source_messages:
                raise ValueError("Deterministic cleaning altered source messages/actions")
            final = formats.finalize(
                record, "multimodal_image_text_dialogs_v3", apply_ref_use_loss_seq=True
            )
            projected = json.loads(final.model_dump_json(by_alias=True, exclude_none=True))
            # V3 permits nested schema extras but its typed serializer drops them.
            # Preserve the owning tool contract in the actual published JSON.
            projected["tools"] = canonical_tools
            serialized = json.dumps(projected, ensure_ascii=False)
            MultimodalImageTextMessagesV3.model_validate_json(serialized)
            audit = audit_roundtrip(projected, native, output)
            audit["source_trajectory_sha256"] = sha256_file(trajectory)
            (output / "audits" / f"{task_id}.json").write_text(json.dumps(audit, indent=2))
            with (output / "multimodal_image_text_dialogs_v3.jsonl").open("a") as handle:
                handle.write(serialized + "\n")
            receipt.update(
                {
                    "accepted": True,
                    "model_id": evidence["model_id"],
                    "source_sha256": sha256_file(trajectory),
                    "audit": audit,
                    "evidence": evidence_for_metadata,
                }
            )
            summary["accepted"] += 1
        except (ValueError, KeyError, OSError, ValidationError, DeclaredFormatRejection) as error:
            summary["rejected"] += 1
            receipt["error"] = str(error)
            with (output / "rejected" / "rejected_samples.jsonl").open("a") as handle:
                handle.write(json.dumps(receipt, ensure_ascii=False) + "\n")
        summary["rows"].append(receipt)
    (output / "image_manifest.json").write_text(json.dumps(manifest_images, indent=2))
    published = output / "multimodal_image_text_dialogs_v3.jsonl"
    if published.exists():
        summary["published_sha256"] = sha256_file(published)
    (output / "verification.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps({key: summary[key] for key in ("scope", "operations", "accepted", "rejected")})
    )
    return 0 if summary["rejected"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
