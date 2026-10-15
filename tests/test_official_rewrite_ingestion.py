from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
from clean_native_trajectories import sha256_file, verify_execution_evidence
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _json_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


@pytest.fixture
def official_episode(tmp_path):
    catalog_path = ROOT / "validation/official-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    source = next(task for task in catalog if len(task["sites"]) > 1)
    task_id = source["task_id"]
    row_id = f"task-{task_id:04d}"
    public = {
        "source_task_id": row_id, "official_task_id": task_id,
        "sites": source["sites"], "start_urls": ["http://site.test/"],
        "intent": "Please " + source["intent"],
    }
    schema = {
        "type": "object",
        "properties": {
            "task_type": {"enum": ["RETRIEVE", "NAVIGATE", "MUTATE"]},
            "status": {"enum": ["SUCCESS"]},
            "retrieved_data": {"type": ["array", "null"]},
            "error_details": {"type": ["string", "null"]},
        },
        "required": ["task_type", "status"],
    }
    instruction = {
        "intent": public["intent"], "start_urls": public["start_urls"],
        "final_response_schema": schema,
    }
    instruction_text = f"Task: {public['intent']}\n\nShared answer schema."
    native = {
        "metadata": {"model_id": "qwen3.8-max", "status": "completed", "tools": []},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": instruction_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Finished."}]},
        ],
    }
    baseline = {"task_id": task_id, "fresh_service_processes": True}
    response = {
        "task_type": "NAVIGATE", "status": "SUCCESS", "retrieved_data": None,
        "error_details": None,
    }
    raw_response = "```json\n" + json.dumps(response) + "\n```"
    native["messages"][1]["content"] = [{"type": "text", "text": raw_response}]
    official_result = {
        "task_id": task_id, "intent_template_id": source["intent_template_id"],
        "sites": source["sites"], "task_revision": source["revision"],
        "status": "success", "score": 1, "error_msg": None,
        "webarena_verified_version": "1.2.3",
        "webarena_verified_evaluator_checksum":
        "35c3385b1db4b3378657589f95f50defd4234bd36e5b93d44733fd561b01db4e",
        "webarena_verified_data_checksum": sha256_file(catalog_path),
        "evaluators_results": [{
            "evaluator_name": item["evaluator"], "status": "success", "score": 1,
            "actual": {"format": "har"}, "actual_normalized": {"fixture": True},
            "expected": item["expected"], "assertions": [], "error_msg": None,
            "should_not_exist": item.get("should_not_exist"),
        } for item in source["eval"]],
    }
    trace_result = copy.deepcopy(official_result)
    for item in trace_result["evaluators_results"]:
        item["actual"] = {"format": "playwright_trace"}
    evaluation = {
        "score": 1, "integrity_status": "passed", "publishable": True,
        "inputs_agree": True, "score_inputs_agree": True,
        "har": official_result, "playwright_trace": trace_result, "input_receipt": {},
    }
    artifacts = {
        "source_dataset": "source_dataset.json", "source_task": "source_task.json",
        "public_task": "public_task.json", "evaluation": "evaluation.json",
        "status": "status.json", "baseline": "baseline_receipt.json",
        "instruction": "instruction.json", "original_instruction": "original_instruction.json",
        "instruction_text": "instruction.txt", "agent_response": "agent_response.json",
        "agent_response_raw": "agent_response_raw.txt", "har": "network.har", "trace": "trace.zip",
        "native_jsonl": "agent_trajectory.jsonl", "canonical_tools": "canonical_tools.json",
    }
    (tmp_path / artifacts["source_dataset"]).write_bytes(catalog_path.read_bytes())
    (tmp_path / artifacts["har"]).write_bytes(b"fixture HAR")
    (tmp_path / artifacts["trace"]).write_bytes(b"fixture trace")
    (tmp_path / artifacts["agent_response_raw"]).write_text(raw_response)
    (tmp_path / artifacts["instruction_text"]).write_text(instruction_text + "\n")
    (tmp_path / artifacts["native_jsonl"]).write_text(json.dumps(native) + "\n")
    (tmp_path / artifacts["canonical_tools"]).write_text(json.dumps(native["metadata"]["tools"]))
    for name, filename in (("network_har", "network.har"), ("playwright_trace", "trace.zip"),
                           ("agent_response_raw", "agent_response_raw.txt")):
        path = tmp_path / filename
        evaluation["input_receipt"][name] = {
            "sha256": sha256_file(path), "size": path.stat().st_size,
        }
    documents = {
        "source_task": source, "public_task": public, "evaluation": evaluation,
        "baseline": baseline, "instruction": instruction,
        "original_instruction": {**instruction, "intent": source["intent"]},
        "agent_response": {
            "valid": True, "raw_content": raw_response, "validated_projection": response,
        },
        "status": {"status": "completed", "task_id": task_id, "model": "qwen3.8-max",
                   "score": 1, "truncated": False},
    }
    for name, document in documents.items():
        (tmp_path / artifacts[name]).write_text(json.dumps(document))
    params = {"max_tokens": 32768, "reasoning_effort": "xhigh", "stream": False}
    execution = {"teacher": {
        "requested_model": "qwen3.8-max", "returned_probe_model": "qwen3.8-max",
        "api_type": "openai_chat", "params": params,
    }}
    (tmp_path / "model_execution.json").write_text(json.dumps(execution))
    requests = tmp_path / "requests"
    requests.mkdir()
    (requests / "00001.request.json").write_text(json.dumps({
        "model": "qwen3.8-max", "messages": [native["messages"][0]], "tools": [], **params,
    }))
    (requests / "00001.response.json").write_text(json.dumps({
        "model": "qwen3.8-max", "choices": [{"finish_reason": "stop"}],
    }))
    score = {
        "evidence_kind": "official_task_rewrite", "source_task_id": row_id,
        "official_task_id": task_id, "status": "pass", "execution_status": "completed",
        "model_id": "qwen3.8-max", "api_type": "openai_chat",
        "public_projection_sha256": _json_sha(public), "task_spec_sha256": _json_sha(source),
        "baseline_sha256": _json_sha(baseline), "artifacts": artifacts,
        "artifact_sha256": {name: sha256_file(tmp_path / relative)
                            for name, relative in artifacts.items()},
    }
    score_path = tmp_path / "reference_receipt.json"
    score_path.write_text(json.dumps(score))
    entry = {
        "task_id": row_id, "score_receipt": str(score_path), "evidence_kind": "official_task_rewrite",
        "api_type": "openai_chat", "request_dir": str(requests),
        "model_execution": str(tmp_path / "model_execution.json"),
        "model_execution_sha256": sha256_file(tmp_path / "model_execution.json"),
    }
    return entry, native, public, score, documents, tmp_path


def _rewrite_document(episode, name, value):
    _, _, _, score, _, root = episode
    path = root / score["artifacts"][name]
    path.write_text(json.dumps(value))
    score["artifact_sha256"][name] = sha256_file(path)
    (root / "reference_receipt.json").write_text(json.dumps(score))


def test_official_rewrite_multisite_pass_preserves_evaluation_only_scope(official_episode):
    entry, native, public, *_ = official_episode
    assert len(public["sites"]) > 1
    evidence = verify_execution_evidence(entry, native, public, True)
    assert evidence["evidence_kind"] == "official_task_rewrite"
    assert evidence["benchmark_derived"] is True
    assert evidence["evaluation_scope"] == "evaluation_only"
    assert evidence["independent_final_state"] == "not_verified"
    assert "final_state_confirmed" not in evidence["score_checks"]


@pytest.mark.parametrize("field,value", [
    ("integrity_status", "evaluator_error"), ("inputs_agree", False), ("score", 0),
])
def test_official_rewrite_requires_real_complete_scoring(official_episode, field, value):
    documents = official_episode[4]
    changed = copy.deepcopy(documents["evaluation"])
    changed[field] = value
    _rewrite_document(official_episode, "evaluation", changed)
    with pytest.raises(ValueError, match="valid official PASS"):
        verify_execution_evidence(*official_episode[:3], True)


def test_equal_scores_and_flags_do_not_hide_normalized_contract_disagreement(official_episode):
    changed = copy.deepcopy(official_episode[4]["evaluation"])
    changed["playwright_trace"]["evaluators_results"][0]["actual_normalized"] = "wrong object"
    _rewrite_document(official_episode, "evaluation", changed)
    with pytest.raises(ValueError, match="normalized contracts disagree"):
        verify_execution_evidence(*official_episode[:3], True)


def test_both_score_inputs_must_include_complete_evaluator_contracts(official_episode):
    changed = copy.deepcopy(official_episode[4]["evaluation"])
    for source in ("har", "playwright_trace"):
        del changed[source]["evaluators_results"][0]["assertions"]
    _rewrite_document(official_episode, "evaluation", changed)
    with pytest.raises(ValueError, match="every original evaluator"):
        verify_execution_evidence(*official_episode[:3], True)


def test_scored_answer_must_be_the_native_final_response(official_episode):
    native = official_episode[1]
    native["messages"][1]["content"] = [{"type": "text", "text": "Unscored alternate answer"}]
    _rewrite_document(official_episode, "native_jsonl", native)
    with pytest.raises(ValueError, match="native final answer"):
        verify_execution_evidence(*official_episode[:3], True)


def test_official_source_task_cannot_change_evaluator(official_episode):
    changed = copy.deepcopy(official_episode[4]["source_task"])
    changed["eval"][0]["expected"] = {"status": "SUCCESS"}
    _rewrite_document(official_episode, "source_task", changed)
    with pytest.raises(ValueError, match="pinned catalog record"):
        verify_execution_evidence(*official_episode[:3], True)


def test_rewrite_cannot_change_start_urls_or_answer_schema(official_episode):
    changed = copy.deepcopy(official_episode[4]["original_instruction"])
    changed["start_urls"] = ["http://other.test/"]
    _rewrite_document(official_episode, "original_instruction", changed)
    with pytest.raises(ValueError, match="fields besides intent"):
        verify_execution_evidence(*official_episode[:3], True)


def test_official_rewrite_rejects_tampered_scored_artifact(official_episode):
    (official_episode[-1] / "network.har").write_bytes(b"different network recording")
    with pytest.raises(ValueError, match="artifact differs"):
        verify_execution_evidence(*official_episode[:3], True)


def test_official_rewrite_requires_original_response_schema(official_episode):
    changed = copy.deepcopy(official_episode[4]["agent_response"])
    changed["validated_projection"]["retrieved_data"] = "not an array"
    raw = "```json\n" + json.dumps(changed["validated_projection"]) + "\n```"
    changed["raw_content"] = raw
    official_episode[1]["messages"][1]["content"] = [{"type": "text", "text": raw}]
    _rewrite_document(official_episode, "native_jsonl", official_episode[1])
    _rewrite_document(official_episode, "agent_response", changed)
    root = official_episode[-1]
    score = official_episode[3]
    (root / "agent_response_raw.txt").write_text(raw)
    score["artifact_sha256"]["agent_response_raw"] = sha256_file(root / "agent_response_raw.txt")
    (root / "reference_receipt.json").write_text(json.dumps(score))
    with pytest.raises(ValueError, match="original response schema"):
        verify_execution_evidence(*official_episode[:3], True)


def test_generated_scoring_still_requires_independent_final_state(official_episode):
    entry, native, public, score, _, root = official_episode
    del entry["evidence_kind"]
    score["generated_task_id"] = entry["task_id"]
    score["checks"] = {
        "response_contract_valid": True, "all_evaluators_pass": True,
        "har_playwright_results_equal": True, "final_state_confirmed": False,
    }
    (root / "reference_receipt.json").write_text(json.dumps(score))
    with pytest.raises(ValueError, match="final_state_confirmed"):
        verify_execution_evidence(entry, native, public, True)


def test_official_rewrite_native_prompt_must_match_public_intent(official_episode):
    official_episode[1]["messages"][0]["content"][0]["text"] = "Task: a different task"
    _rewrite_document(official_episode, "native_jsonl", official_episode[1])
    with pytest.raises(ValueError, match="native model input"):
        verify_execution_evidence(*official_episode[:3], True)


def test_fixed46_multisite_clean_and_absolute_bundle_export(official_episode, monkeypatch):
    import clean_native_trajectories
    import export_review_bundle

    entry, native, public, score, documents, root = official_episode
    raw_response = documents["agent_response"]["raw_content"]
    tools = [{"type": "function", "function": {
        "name": "response", "description": "Submit the final response.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"], "additionalProperties": False},
    }}]
    native["metadata"]["tools"] = tools
    native["messages"][0]["content"].append({
        "type": "image", "image": {"url": "images/screen.png", "image_wh": [1920, 1080]},
    })
    native["messages"][1]["content"] = [{"type": "text", "text": raw_response}]
    image = root / "images/screen.png"
    image.parent.mkdir()
    Image.new("RGB", (1920, 1080), (240, 240, 240)).save(image)
    _rewrite_document(official_episode, "native_jsonl", native)
    _rewrite_document(official_episode, "canonical_tools", tools)
    public_path = root / "public.jsonl"
    public_path.write_text(json.dumps(public) + "\n")
    entry.update(trajectory=str(root / "agent_trajectory.jsonl"), public_tasks=str(public_path),
                 canonical_tools=str(root / "canonical_tools.json"))
    manifest = root / "clean_manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n")
    cleaned = root / "cleaned"
    monkeypatch.setattr(sys, "argv", [
        "clean_native_trajectories.py", "--input-manifest", str(manifest),
        "--output-root", str(cleaned), "--profile", str(ROOT / "profiles/clean_cua_fixed.yaml"),
        "--formal",
    ])
    assert clean_native_trajectories.main() == 0
    verification = json.loads((cleaned / "verification.json").read_text())
    assert verification["operations"] == 46
    assert verification["accepted"] == 1
    # Rejected inputs can leave manifest images outside the accepted projection.
    unreferenced = cleaned / "images/rejected-only.png"
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(unreferenced)
    image_manifest = json.loads((cleaned / "image_manifest.json").read_text())
    image_manifest["images/rejected-only.png"] = {
        "sha256": sha256_file(unreferenced),
        "bytes": unreferenced.stat().st_size,
        "image_wh": [1920, 1080],
    }
    (cleaned / "image_manifest.json").write_text(json.dumps(image_manifest))
    target = root / "exported"
    monkeypatch.setattr(sys, "argv", [
        "export_review_bundle.py", "--clean-output", str(cleaned), "--target", str(target),
        "--require-trajectories", "1",
    ])
    assert export_review_bundle.main() == 0
    assert json.loads((target / "image_manifest.json").read_text()) == image_manifest
    assert (target / "images/rejected-only.png").read_bytes() == unreferenced.read_bytes()
    review = json.loads((target / "review_manifest.json").read_text())
    assert review["benchmark_derived"] is True
    assert review["evaluation_scope"] == "evaluation_only"
    assert review["review_status"] == "pending_human_review"
    assert review["training_promotion"] == "not_promoted"
    row = json.loads((target / "multimodal_image_text_dialogs_v3.jsonl").read_text())
    assert row["metadata"]["source"] == "webarena_verified_official_task_rewrite"
    assert row["metadata"]["independent_final_state"] == "not_verified"
    assert len(row["metadata"]["sites"]) > 1
    exported_json = json.loads((target / "json_absolute" / (entry["task_id"] + ".json")).read_text())
    exported_jsonl = json.loads((target / "jsonl_absolute" / (entry["task_id"] + ".jsonl")).read_text())
    assert row == exported_json == exported_jsonl
    for message in row["messages"]:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part["type"] == "image":
                    assert Path(part["image"]["url"]).is_absolute()
                    assert Path(part["image"]["url"]).is_file()
