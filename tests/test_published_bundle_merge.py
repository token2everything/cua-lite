"""Synthetic published-source contracts exercise real evidence/export owners."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import merge_published_bundles
import clean_native_trajectories
import export_review_bundle
import pytest
from clean_native_trajectories import sha256_file, verify_execution_evidence
from export_review_bundle import export_absolute_jsonl, export_trajectory_files
from internvalidata.data_models.messages_types.multimodal_image_text_messages import (
    MultimodalImageTextMessagesV3,
)
from PIL import Image


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")


def published_fixture(root, task, revision, color, student=None):
    root.mkdir()
    original = root / "original" / task
    original.mkdir(parents=True)
    image = original / "images/screen.png"
    image.parent.mkdir()
    Image.new("RGB", (2, 2), color).save(image)
    digest = sha256_file(image)
    visible = root / "images" / (digest + ".png")
    visible.parent.mkdir()
    shutil.copyfile(image, visible)
    uri = f"/synthetic-unit-episode/{task}/screen.png"
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Synthetic contract fixture; no live sampling."},
            {"type": "image", "image": {"url": uri, "image_wh": [2, 2]}},
        ],
    }
    assistant = {"role": "assistant", "content": "Synthetic answer."}
    native = {"messages": [user, assistant], "metadata": {"model_id": "qwen3.8-max", "tools": [],
                                                          "status": "completed"}}
    write_json(original / "agent_trajectory.jsonl", native)
    write_json(original / "canonical_tools.json", [])
    public = {"generated_task_id": task, "sites": ["gitlab"], "intent": "Synthetic contract fixture."}
    write_json(original / "public_task.json", public)
    teacher = {
        "requested_model": "qwen3.8-max",
        "returned_probe_model": "qwen3.8-max",
        "api_type": "openai_chat",
        "params": {"max_tokens": 32768, "temperature": 0.0},
    }
    execution = {"teacher": teacher, "revision": revision}
    if student is not None:
        execution["student"] = {"model": student}
    write_json(root / "model_execution.json", execution)
    execution_sha = sha256_file(root / "model_execution.json")
    write_json(
        original / "requests/00001.request.json",
        {"model": "qwen3.8-max", "tools": [], "messages": [user], **teacher["params"]},
    )
    write_json(
        original / "requests/00001.response.json",
        {"model": "qwen3.8-max", "choices": [{"finish_reason": "stop"}]},
    )
    public_sha = hashlib.sha256(
        json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifacts = {
        "native_jsonl": "agent_trajectory.jsonl",
        "canonical_tools": "canonical_tools.json",
        "public_task": "public_task.json",
    }
    receipt = {
        "generated_task_id": task,
        "status": "pass",
        "model_id": "qwen3.8-max",
        "api_type": "openai_chat",
        "public_projection_sha256": public_sha,
        "task_spec_sha256": "synthetic-task-spec",
        "baseline_sha256": "synthetic-baseline",
        "checks": {
            "response_contract_valid": True,
            "all_evaluators_pass": True,
            "har_playwright_results_equal": True,
            "final_state_confirmed": True,
        },
        "artifacts": artifacts,
        "artifact_sha256": {key: sha256_file(original / path) for key, path in artifacts.items()},
    }
    write_json(original / "reference_receipt.json", receipt)
    evidence = verify_execution_evidence(
        {
            "task_id": task,
            "request_dir": str(original / "requests"),
            "score_receipt": str(original / "reference_receipt.json"),
            "api_type": "openai_chat",
            "model_execution": str(root / "model_execution.json"),
            "model_execution_sha256": execution_sha,
        },
        native,
        public,
        True,
    )
    evidence.pop("files")
    row = {
        "_id": task,
        "model_name": "qwen3.8-max",
        "tools": [],
        "messages": [
            {
                "role": "user",
                "content": [
                    user["content"][0],
                    {
                        "type": "image",
                        "image": {"url": str(visible.relative_to(root)), "image_wh": [2, 2]},
                    },
                ],
                "meta": {
                    "native_projection": [
                        {
                            "source_index": 0,
                            "source_message": user,
                            "content_start": 0,
                            "content_end": 2,
                        }
                    ]
                },
            },
            {
                "role": "assistant",
                "content": assistant["content"],
                "loss": True,
                "meta": {"native_projection": [{"source_index": 1, "source_message": assistant}]},
            },
        ],
        "metadata": {
            "source_task_id": task,
            "source_model_id": "qwen3.8-max",
            "source_api_model_id": "qwen3.8-max",
            "source_api_type": "openai_chat",
            "scope": "v03_formal",
            "sites": ["gitlab"],
            "source_trajectory_sha256": sha256_file(original / "agent_trajectory.jsonl"),
            "canonical_tools_sha256": sha256_file(original / "canonical_tools.json"),
            "execution_evidence": evidence,
            "internflow": {"accepted": True},
            "image_aliases": {
                uri: {"url": str(visible.relative_to(root)), "sha256": digest, "image_wh": [2, 2]}
            },
            "source_media_archive": {
                "screen.png": {
                    "url": str(image.relative_to(root)),
                    "sha256": digest,
                    "image_wh": [2, 2],
                }
            },
        },
    }
    row = MultimodalImageTextMessagesV3.model_validate(row).model_dump(
        by_alias=True, exclude_none=True, mode="json"
    )
    relative = root / "audits/cleaning_projection.relative.jsonl"
    write_json(relative, row)
    for stage in (
        "001_common_format_fixes",
        "002_common_format_filters",
        "003_gui_behavior_filters",
    ):
        write_json(root / "processed" / stage / (task + ".json"), row)
    (root / "rejected").mkdir()
    write_json(
        root / "verification.json",
        {
            "scope": "v03_formal",
            "accepted": 1,
            "published_sha256": sha256_file(relative),
            "model_execution_sha256": execution_sha,
            "operations": 46,
            "groups": 3,
            "versions": {"internflow": "0.2.0", "internvalidata": "0.4.17"},
            "fixture_scope": "synthetic unit contract; no real fixed46 or model execution",
        },
    )
    for name in (
        "prepared_plan.json",
        "image_manifest.json",
        "export_audits.json",
        "copied_bundle_audits.json",
    ):
        write_json(root / name, {"fixture_scope": "synthetic"})
    (root / "profile.yaml").write_text("fixture_scope: synthetic\n")
    (root / "input_manifest.jsonl").write_text("{}\n")
    exported = export_absolute_jsonl(
        root, relative, root / "multimodal_image_text_dialogs_v3.jsonl"
    )
    export_trajectory_files(
        root / "multimodal_image_text_dialogs_v3.jsonl", root / "json_absolute", "json"
    )
    export_trajectory_files(
        root / "multimodal_image_text_dialogs_v3.jsonl", root / "jsonl_absolute", "jsonl"
    )
    write_json(
        root / "review_manifest.json",
        {"scope": "v03_formal", "sample_rows": 1, "jsonl_sha256": exported["jsonl_sha256"]},
    )
    return root


def index_published_snapshots(bundle, sources):
    snapshots = {}
    contract = None
    for source in sources:
        path = source / "model_execution.json"
        digest = sha256_file(path)
        current = {key: value for key, value in json.loads(path.read_text()).items() if key != "student"}
        assert contract is None or current == contract
        contract = current
        relative = "model_executions/" + digest + ".json"
        (bundle / "model_executions").mkdir(exist_ok=True)
        shutil.copyfile(path, bundle / relative)
        snapshots[digest] = relative
    write_json(bundle / "model_execution.json", {**contract, "source_executions": snapshots})
    verification = json.loads((bundle / "verification.json").read_text())
    verification["model_execution_sha256"] = sha256_file(bundle / "model_execution.json")
    verification["expected_teacher"] = contract["teacher"]
    verification["format_projection"] = "reversible_image_v3_v2_1"
    write_json(bundle / "verification.json", verification)


def test_real_cleaner_archives_each_source_snapshot_without_rebinding(tmp_path, monkeypatch):
    sources = [published_fixture(tmp_path / name, task, 3, color, model)
        for name, task, color, model in [("first", "first", "red", "glm-5.3-flash"),
                                         ("second", "second", "blue", "z-ai/glm-5.3-flash")]]
    before = {path: path.read_bytes() for source in sources for path in source.rglob("*") if path.is_file()}
    entries = []
    for source in sources:
        task = source.name
        original = source / "original" / task
        entries.append({
            "task_id": task, "trajectory": str(original / "agent_trajectory.jsonl"),
            "request_dir": str(original / "requests"), "score_receipt": str(original / "reference_receipt.json"),
            "public_tasks": str(original / "public_task.json"), "canonical_tools": str(original / "canonical_tools.json"),
            "api_type": "openai_chat", "model_execution": str(source / "model_execution.json"),
            "model_execution_sha256": sha256_file(source / "model_execution.json"),
            "media_mount": {"source_root": "/synthetic-unit-episode/" + task, "local_root": str(original / "images")},
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    output = tmp_path / "cleaned"
    profile = Path(clean_native_trajectories.__file__).parents[1] / "profiles/clean_cua_fixed.yaml"
    monkeypatch.setattr(sys, "argv", ["clean_native_trajectories.py", "--input-manifest", str(manifest),
        "--output-root", str(output), "--profile", str(profile), "--formal"])
    # These deliberately empty-tool unit episodes fail fixed46 E18. Snapshot
    # archival and real request/score validation must still preserve provenance.
    assert clean_native_trajectories.main() == 2
    descriptor = json.loads((output / "model_execution.json").read_text())
    assert len(descriptor["source_executions"]) == 2 and "student" not in descriptor
    for entry in entries:
        archived = output / descriptor["source_executions"][entry["model_execution_sha256"]]
        assert archived.read_bytes() == before[Path(entry["model_execution"])]
    assert all(path.read_bytes() == value for path, value in before.items())
    verification = json.loads((output / "verification.json").read_text())
    assert verification["model_execution_sha256"] == sha256_file(output / "model_execution.json")
    assert verification["expected_teacher"] == descriptor["teacher"]
    assert verification["accepted"] == 0 and verification["rejected"] == 2
    assert all("agent_tools_empty" in row["error"] for row in verification["rows"])
    assert all((output / "original" / source.name / "requests/00001.request.json").read_bytes()
               == (source / "original" / source.name / "requests/00001.request.json").read_bytes()
               for source in sources)


@pytest.mark.parametrize("count", [1, 2])
def test_export_preserves_source_index_and_real_row_snapshot_hash(tmp_path, monkeypatch, count):
    bundle = published_fixture(tmp_path / "cleaned", "new-c", 3, "green", "z-ai/glm-5.3-flash")
    snapshot = tmp_path / "unmodified-execution"
    snapshot.mkdir()
    shutil.copyfile(bundle / "model_execution.json", snapshot / "model_execution.json")
    snapshots = [snapshot]
    relative = bundle / "audits/cleaning_projection.relative.jsonl"
    if count == 2:
        second = published_fixture(tmp_path / "second", "new-d", 3, "blue", "glm-5.3-flash")
        snapshots.append(second)
        for name in ["original", "images", "processed"]:
            shutil.copytree(second / name, bundle / name, dirs_exist_ok=True)
        relative.write_text(relative.read_text() + (second / "audits/cleaning_projection.relative.jsonl").read_text())
    index_published_snapshots(bundle, snapshots)
    shutil.copyfile(relative, bundle / "multimodal_image_text_dialogs_v3.jsonl")
    rows = [json.loads(line) for line in relative.read_text().splitlines()]
    images = {ref["url"]: {"sha256": ref["sha256"]} for row in rows for ref in row["metadata"]["image_aliases"].values()}
    write_json(bundle / "image_manifest.json", images)
    verification = json.loads((bundle / "verification.json").read_text())
    verification.update(accepted=count, published_sha256=sha256_file(relative))
    write_json(bundle / "verification.json", verification)
    before = {path: path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    target = tmp_path / "exported"
    monkeypatch.setattr(sys, "argv", ["export_review_bundle.py", "--clean-output", str(bundle),
        "--target", str(target), "--require-trajectories", str(count)])
    assert export_review_bundle.main() == 0
    descriptor = json.loads((target / "model_execution.json").read_text())
    assert len(descriptor["source_executions"]) == count
    for row, source in zip(rows, snapshots):
        task = row["metadata"]["source_task_id"]
        exported = json.loads((target / "jsonl_absolute" / (task + ".jsonl")).read_text())
        digest = exported["metadata"]["execution_evidence"]["model_execution_sha256"]
        assert (target / descriptor["source_executions"][digest]).read_bytes() == (source / "model_execution.json").read_bytes()
        assert exported["metadata"]["execution_evidence"] == row["metadata"]["execution_evidence"]
    assert all(path.read_bytes() == value for path, value in before.items())


def test_merge_preserves_new_indexed_snapshot_and_legacy_original_snapshots(tmp_path, monkeypatch):
    legacy, new, report, roots = merge_inputs(tmp_path)
    snapshot = tmp_path / "new-snapshot"
    snapshot.mkdir()
    shutil.copyfile(new / "model_execution.json", snapshot / "model_execution.json")
    index_published_snapshots(new, [snapshot])
    before = {path: path.read_bytes() for base in (legacy, new, *roots) for path in base.rglob("*") if path.is_file()}
    target = tmp_path / "merged"
    assert run_merge(monkeypatch, legacy, new, report, target) == 0
    manifest = json.loads((target / "review_manifest.json").read_text())
    assert manifest["snapshot_count"] == 3
    digest = sha256_file(snapshot / "model_execution.json")
    row = next(row for row in json.loads((target / "row_provenance.json").read_text()) if row["task"] == "new-c")
    assert row["model_execution_sha256"] == digest
    assert Path(row["archived_snapshot"]).read_bytes() == (snapshot / "model_execution.json").read_bytes()
    assert all(path.read_bytes() == value for path, value in before.items())


def merge_inputs(tmp_path, *, duplicate=False):
    roots = [
        published_fixture(tmp_path / "source-a", "legacy-a", 1, "red"),
        published_fixture(tmp_path / "source-b", "legacy-b", 2, "blue"),
    ]
    new_task = "legacy-a" if duplicate else "new-c"
    new = published_fixture(tmp_path / "new", new_task, 3, "green")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    pairs = []
    lines = []
    for source, task in zip(roots, ("legacy-a", "legacy-b")):
        pair = {
            "task_id": task,
            "existing_export_audit": str(
                source / "multimodal_image_text_dialogs_v3.export_audit.json"
            ),
        }
        for suffix in ("json", "jsonl"):
            original = source / (suffix + "_absolute") / (task + "." + suffix)
            dest = legacy / (suffix + "_absolute") / original.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(original, dest)
            pair[suffix] = {"source": str(original), "SHA256": sha256_file(original)}
        pairs.append(pair)
        lines.append((source / "multimodal_image_text_dialogs_v3.jsonl").read_text())
    combined = legacy / "multimodal_image_text_dialogs_v3.jsonl"
    combined.write_text("".join(lines))
    original_report = tmp_path / "original-student-report.json"
    write_json(original_report, {"fixture_scope": "synthetic original frozen assessment"})
    write_json(
        legacy / "delivery_manifest.json",
        {
            "combined_rows": 2,
            "pairs": 2,
            "formal_order_case_ids": ["legacy-a", "legacy-b"],
            "combined_JSONL_SHA256": sha256_file(combined),
            "pairs_provenance": pairs,
            "final_report_SHA256": sha256_file(original_report),
            "GLM_pass": 0,
            "GLM_fail": 1,
            "GLM_HOLD": 1,
            "existing_formal_methods_GLM_and_report_references": {
                original_report.name: {
                    "path": str(original_report),
                    "SHA256": sha256_file(original_report),
                }
            },
        },
    )
    report = tmp_path / "student-report/final300_GLM_after_hold21_reevaluation.json"
    write_json(
        report,
        {
            "original_report_SHA256": sha256_file(original_report),
            "reevaluation_rows": [],
            "rows": [
                {"case_id": "legacy-a", "category": "pass", "effective_scored": True},
                {"case_id": "legacy-b", "category": "fail", "effective_scored": True},
            ],
        },
    )
    write_json(report.parent / "files_SHA256.json", {"files": {report.name: sha256_file(report)}})
    student_rows = []
    for source, task, status in zip(roots, ("legacy-a", "legacy-b"), ("PASS", "FAIL")):
        raw = tmp_path / "legacy-students" / task / "student-01"
        shutil.copytree(source / "original" / task, raw)
        native = json.loads((raw / "agent_trajectory.jsonl").read_text())
        native["metadata"]["model_id"] = "glm-5.3-flash"
        write_json(raw / "agent_trajectory.jsonl", native)
        for path in (raw / "requests").iterdir():
            value = json.loads(path.read_text())
            value["model"] = "glm-5.3-flash"
            write_json(path, value)
        receipt = json.loads((raw / "reference_receipt.json").read_text())
        receipt["model_id"] = "glm-5.3-flash"
        receipt["status"] = status.lower()
        receipt["artifact_sha256"]["native_jsonl"] = sha256_file(raw / "agent_trajectory.jsonl")
        write_json(raw / "reference_receipt.json", receipt)
        student_rows.append(
            {
                "task": task,
                "source": str(raw),
                "source_report": str(report),
                "source_report_sha256": sha256_file(report),
                "origin": "original_valid_episode",
                "attempt": "student-01",
                "receipt_sha256": sha256_file(raw / "reference_receipt.json"),
                "reported_category": status.lower(),
                "status": status,
                "effective_scored": True,
                "artifacts_sha256": {
                    relative: receipt["artifact_sha256"][name]
                    for name, relative in receipt["artifacts"].items()
                },
                "actual_identity": {
                    "receipt_model_id": "glm-5.3-flash",
                    "api_type": "openai_chat",
                    "native_model_id": "glm-5.3-flash",
                    "wire_request_models": ["glm-5.3-flash"],
                    "wire_successful_response_models": ["glm-5.3-flash"],
                },
                "request_evidence_sha256": {
                    str(path.relative_to(raw / "requests")): sha256_file(path)
                    for path in (raw / "requests").rglob("*")
                    if path.is_file()
                },
            }
        )
    write_json(
        report.parent / "student_index.json",
        {
            "schema": "legacy_student_episode_index_v1",
            "count": 2,
            "rows": student_rows,
            "legacy_student_report_sha256": sha256_file(report),
            "audits": [],
        },
    )
    (new / "evidence").mkdir()
    write_json(
        new / "evidence/author.json", {"fixture_scope": "synthetic preserved author evidence"}
    )
    write_json(
        new / "evidence_manifest.json",
        {
            "files_SHA256": {"author.json": sha256_file(new / "evidence/author.json")},
            "credential_files_included": False,
        },
    )
    write_json(
        new / "difficulty_report.json",
        {
            "rows": [{"task": new_task, "status": "HOLD"}],
            "difficulty_methods_and_rules": [],
            "teacher_rollout_statistics": {
                "attempt_distribution": {"1": 1, "2": 0, "3": 0},
                "first_attempt_success_rate": 1.0,
                "cumulative_success_rate": 1.0,
            },
        },
    )
    return legacy, new, report, roots


def run_merge(monkeypatch, legacy, new, report, target):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_published_bundles.py",
            "--legacy-delivery",
            str(legacy),
            "--new-delivery",
            str(new),
            "--legacy-student-report",
            str(report),
            "--legacy-student-evidence-index",
            str(report.parent / "student_index.json"),
            "--target",
            str(target),
            "--require-trajectories",
            "3",
        ],
    )
    return merge_published_bundles.main()


def test_mixed_snapshots_rebase_all_media_and_preserve_source_bytes(tmp_path, monkeypatch):
    legacy, new, report, roots = merge_inputs(tmp_path)
    before = {
        path: sha256_file(path)
        for base in (legacy, new, report.parent, tmp_path / "legacy-students", *roots)
        for path in base.rglob("*")
        if path.is_file()
    }
    target = tmp_path / "merged"
    assert run_merge(monkeypatch, legacy, new, report, target) == 0
    manifest = json.loads((target / "review_manifest.json").read_text())
    assert manifest["snapshot_count"] == 3 and manifest["sample_rows"] == 3
    assert manifest["fixed46_reexecuted"] is False and manifest["source_files_unchanged"] is True
    originals = {}
    for base in (*roots, new):
        row = json.loads((base / "multimodal_image_text_dialogs_v3.jsonl").read_text())
        originals[row["metadata"]["source_task_id"]] = row
    ids = []
    for line in (target / "multimodal_image_text_dialogs_v3.jsonl").open():
        row = json.loads(line)
        task = row["metadata"]["source_task_id"]
        ids.append(task)
        assert json.loads((target / "json_absolute" / (task + ".json")).read_text()) == row
        assert json.loads((target / "jsonl_absolute" / (task + ".jsonl")).read_text()) == row
        assert (
            row["metadata"]["execution_evidence"]
            == originals[task]["metadata"]["execution_evidence"]
        )
        for field in ("image_aliases", "source_media_archive"):
            for key, reference in row["metadata"][field].items():
                for value in (key, reference["url"]):
                    path = Path(value)
                    path.relative_to(target)
                    assert path.is_absolute() and path.is_file()
        for message in row["messages"]:
            for boundary in message["meta"]["native_projection"]:
                content = boundary["source_message"]["content"]
                if isinstance(content, list):
                    for part in content:
                        if part["type"] == "image":
                            assert Path(part["image"]["url"]).is_relative_to(target)
    assert ids == ["legacy-a", "legacy-b", "new-c"]
    assert all(sha256_file(path) == digest for path, digest in before.items())
    difficulty = json.loads((target / "difficulty_report.json").read_text())
    assert difficulty["student_counts"] == {"PASS": 1, "FAIL": 1, "HOLD": 1}
    assert difficulty["student_effective"] == 2 and difficulty["student_success_rate"] == 0.5
    assert difficulty["legacy_historical_manifest_counts"] == {"PASS": 0, "FAIL": 1, "HOLD": 1}
    assert difficulty["uniform_current_reevaluation"] is False
    assert difficulty["student_outcomes_used_for_selection"] is False
    assert difficulty["legacy_student_raw_archived"] == 2
    for task in ("legacy-a", "legacy-b"):
        assert sha256_file(
            target / "provenance/legacy_students" / task / "reference_receipt.json"
        ) == sha256_file(tmp_path / "legacy-students" / task / "student-01/reference_receipt.json")
    assert difficulty["new_cohort_report"]["teacher_rollout_statistics"] == {
        "attempt_distribution": {"1": 1, "2": 0, "3": 0},
        "first_attempt_success_rate": 1.0,
        "cumulative_success_rate": 1.0,
    }


def test_duplicate_published_id_is_rejected_before_target_creation(tmp_path, monkeypatch):
    legacy, new, report, _ = merge_inputs(tmp_path, duplicate=True)
    target = tmp_path / "merged"
    with pytest.raises(ValueError, match="Duplicate published task ID"):
        run_merge(monkeypatch, legacy, new, report, target)
    assert not target.exists()


def test_actual_request_parameter_drift_is_rejected(tmp_path, monkeypatch):
    legacy, new, report, _ = merge_inputs(tmp_path)
    request = new / "original/new-c/requests/00001.request.json"
    value = json.loads(request.read_text())
    value["temperature"] = 0.9
    write_json(request, value)
    with pytest.raises(ValueError, match="Actual model parameter differs from the frozen teacher"):
        run_merge(monkeypatch, legacy, new, report, tmp_path / "merged")
    assert not (tmp_path / "merged/review_manifest.json").exists()


def test_merge_cannot_write_inside_a_published_source(tmp_path, monkeypatch):
    legacy, new, report, _ = merge_inputs(tmp_path)
    target = new / "nested-merge"
    with pytest.raises(ValueError, match="outside the published source directories"):
        run_merge(monkeypatch, legacy, new, report, target)
    assert not target.exists()


def test_legacy_student_index_must_cover_the_full_published_cohort(tmp_path, monkeypatch):
    legacy, new, report, _ = merge_inputs(tmp_path)
    path = report.parent / "student_index.json"
    index = json.loads(path.read_text())
    index["rows"].pop()
    index["count"] = 1
    write_json(path, index)
    target = tmp_path / "merged"
    with pytest.raises(ValueError, match="cover every published legacy task once"):
        run_merge(monkeypatch, legacy, new, report, target)
    assert not target.exists()


def test_legacy_student_wire_bytes_are_bound_to_the_audited_index(tmp_path, monkeypatch):
    legacy, new, report, _ = merge_inputs(tmp_path)
    path = tmp_path / "legacy-students/legacy-a/student-01/requests/00001.request.json"
    request = json.loads(path.read_text())
    request["model"] = "another-model"
    write_json(path, request)
    target = tmp_path / "merged"
    with pytest.raises(ValueError, match="student request evidence differs"):
        run_merge(monkeypatch, legacy, new, report, target)
    assert not (target / "review_manifest.json").exists()
