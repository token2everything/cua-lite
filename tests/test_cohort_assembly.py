from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from sandbox import _atomic_json, _sha

import assemble


def write_source(root, name, execution):
    source = root / "runs/generated" / name
    repo = Path(__file__).resolve().parents[1]
    (source / "release/infra").mkdir(parents=True)
    (source / "release/profiles").mkdir()
    for filename in [
        "clean_native_trajectories.py",
        "clean_v3_projection.py",
        "export_review_bundle.py",
    ]:
        shutil.copyfile(repo / "infra" / filename, source / "release/infra" / filename)
    shutil.copyfile(
        repo / "profiles/clean_cua_fixed.yaml", source / "release/profiles/clean_cua_fixed.yaml"
    )
    _atomic_json(source / "plans/model_execution.json", execution)
    _atomic_json(
        source / "admitted.json", {"tasks": ["wavgen-test-01", "wavgen-test-02", "wavgen-test-03"]}
    )
    (source / "cleaned").mkdir()
    (source / "cleaned/multimodal_image_text_dialogs_v3.jsonl").write_text(
        "".join(
            json.dumps({"metadata": {"source_task_id": task}}) + "\n"
            for task in ["wavgen-test-02", "wavgen-test-03"]
        )
    )
    entries = [
        {
            "task_id": task,
            "model_execution": str(source / "plans/model_execution.json"),
            "model_execution_sha256": _sha(source / "plans/model_execution.json"),
        }
        for task in ["wavgen-test-01", "wavgen-test-02", "wavgen-test-03"]
    ]
    (source / "plans/clean_manifest.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in entries)
    )
    (source / "private.jsonl").write_text("{}\n")
    (source / "public.jsonl").write_text("{}\n")
    for task in ["wavgen-test-01", "wavgen-test-02", "wavgen-test-03"]:
        attempt = {
            "task": task,
            "profile": "teacher",
            "status": "PASS",
            "owner_status": "pass",
            "effective_scored": True,
            "steps": 22,
            "attempt": "teacher-01",
            "quality": {"status": "PASS", "reason": "fixed46_and_V3_pass"},
        }
        _atomic_json(
            source / "model_results" / (task + ".teacher.json"),
            {
                **attempt,
                "selected_attempt": "teacher-01",
                "semantic_episode_budget": 3,
                "attempts": [attempt],
                "attempt_counts": {"PASS": 1, "FAIL": 0, "HOLD": 0},
                "quality_counts": {"PASS": 1, "REJECT": 0, "HOLD": 0},
                "quality_status": "PASS",
                "reason": "first_teacher_PASS",
            },
        )
    return source


def test_assembly_selects_teacher_quality_and_rebinds_one_active_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(assemble, "ROOT", tmp_path)
    source = write_source(tmp_path, "site01", {"teacher": {"requested_model": "qwen3.8-max"}})
    # No student result files exist: build must depend only on teacher quality.
    spec = tmp_path / "spec.json"
    spec.write_text('[{"run":"site01","count":1}]')
    monkeypatch.setattr(
        sys, "argv", ["assemble.py", "plan", "--run", "final01", "--spec", str(spec)]
    )
    assemble.main()
    monkeypatch.setattr(sys, "argv", ["assemble.py", "build", "--run", "final01"])
    assemble.main()
    target = tmp_path / "runs/assembled/final01"
    selected = json.loads((target / "selection.json").read_text())
    assert selected["selected"] == [{"task": "wavgen-test-02", "run": "site01"}]
    assert selected["student_results_read_for_selection"] is False
    entry = json.loads((target / "plans/clean_manifest.jsonl").read_text())
    assert Path(entry["model_execution"]) == target / "plans/model_execution.json"
    assert (
        entry["model_execution_sha256"]
        == _sha(source / "plans/model_execution.json")
        == _sha(Path(entry["model_execution"]))
    )


@pytest.mark.parametrize("quality_status", ["HOLD", "REJECT"])
def test_assembly_skips_aggregate_accepted_teacher_without_original_quality_pass(
    tmp_path, monkeypatch, quality_status
):
    monkeypatch.setattr(assemble, "ROOT", tmp_path)
    source = write_source(tmp_path, "site01", {"teacher": {"requested_model": "qwen3.8-max"}})
    teacher_path = source / "model_results/wavgen-test-02.teacher.json"
    teacher = json.loads(teacher_path.read_text())
    teacher["quality_status"] = quality_status
    teacher["quality"] = {"status": quality_status}
    teacher["attempts"][0]["quality"] = teacher["quality"]
    teacher["quality_counts"] = {"PASS": 0, "REJECT": 0, "HOLD": 0}
    teacher["quality_counts"][quality_status] = 1
    _atomic_json(teacher_path, teacher)
    original_teacher_sha = _sha(teacher_path)
    assert teacher["status"] == "PASS"
    assert not list((source / "model_results").glob("*.student.json"))
    target = tmp_path / "runs/assembled/final01"
    _atomic_json(target / "plans/cohorts.json", [{"run": "site01", "count": 1}])
    (target / "infra").mkdir()
    monkeypatch.setattr(sys, "argv", ["assemble.py", "build", "--run", "final01"])
    assemble.main()
    selected = json.loads((target / "selection.json").read_text())
    assert selected["selected"] == [{"task": "wavgen-test-03", "run": "site01"}]
    assert selected["student_results_read_for_selection"] is False
    assert _sha(teacher_path) == original_teacher_sha


def test_assembly_rejects_quota_when_original_teacher_quality_passes_are_insufficient(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(assemble, "ROOT", tmp_path)
    source = write_source(tmp_path, "site01", {"teacher": {"requested_model": "qwen3.8-max"}})
    teacher_path = source / "model_results/wavgen-test-02.teacher.json"
    teacher = json.loads(teacher_path.read_text())
    teacher["quality_status"] = "HOLD"
    teacher["quality"] = {"status": "HOLD", "reason": "cleaner_completion_unproven"}
    teacher["attempts"][0]["quality"] = teacher["quality"]
    teacher["quality_counts"] = {"PASS": 0, "REJECT": 0, "HOLD": 1}
    _atomic_json(teacher_path, teacher)
    assert not list((source / "model_results").glob("*.student.json"))
    target = tmp_path / "runs/assembled/final01"
    _atomic_json(target / "plans/cohorts.json", [{"run": "site01", "count": 2}])
    (target / "infra").mkdir()
    monkeypatch.setattr(sys, "argv", ["assemble.py", "build", "--run", "final01"])
    with pytest.raises(ValueError, match="Teacher/fixed46 quality leaves a cohort short"):
        assemble.main()
    assert not (target / "selection.json").exists()


def test_assembly_rejects_different_teacher_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(assemble, "ROOT", tmp_path)
    write_source(tmp_path, "site01", {"teacher": {"requested_model": "qwen3.8-max"}})
    write_source(tmp_path, "site02", {"teacher": {"requested_model": "another-model"}})
    spec = tmp_path / "spec.json"
    spec.write_text('[{"run":"site01","count":1},{"run":"site02","count":1}]')
    monkeypatch.setattr(
        sys, "argv", ["assemble.py", "plan", "--run", "final01", "--spec", str(spec)]
    )
    assemble.main()
    monkeypatch.setattr(sys, "argv", ["assemble.py", "build", "--run", "final01"])
    with pytest.raises(ValueError, match="different frozen"):
        assemble.main()
    assert not (tmp_path / "runs/assembled/final01/selection.json").exists()


def test_aggregate_report_uses_delivered_cohort_and_excludes_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(assemble, "ROOT", tmp_path)
    execution = {
        "teacher": {"requested_model": "configured-teacher"},
        "student": {"model": "z-ai/glm-5.3-flash"},
    }
    source = write_source(tmp_path, "site01", execution)
    private = []
    for number, kind in [(1, "RETRIEVE"), (2, "NAVIGATE"), (3, "MUTATE")]:
        task = f"wavgen-test-{number:02}"
        private.append(
            {
                "generated_task_id": task,
                "task_spec": {"sites": ["shopping"]},
                "task_definition": {
                    "task_type": kind,
                    "goal_template": "Synthetic report fixture {name}",
                    "parameters": {"name": str(number)},
                    "completion_events": [],
                },
                "grounding": {
                    "family": "report-fixture",
                    "final_state_checks": [],
                    "field_evidence": [],
                },
            }
        )
    (source / "private.jsonl").write_text("".join(json.dumps(row) + "\n" for row in private))
    for number, status, steps in [(2, "PASS", 4), (3, "HOLD", 10)]:
        task = f"wavgen-test-{number:02}"
        _atomic_json(
            source / "model_results" / (task + ".student.json"), {"task": task, "status": status}
        )
        teacher_path = source / "model_results" / (task + ".teacher.json")
        teacher = json.loads(teacher_path.read_text())
        teacher["steps"] = steps
        teacher["attempts"][0]["steps"] = steps
        _atomic_json(teacher_path, teacher)
    _atomic_json(
        source / "model_results/wavgen-test-01.student.json",
        {"task": "wavgen-test-01", "status": "FAIL"},
    )
    target = tmp_path / "runs/assembled/final01"
    _atomic_json(target / "plans/model_execution.json", execution)
    _atomic_json(target / "plans/cohorts.json", [{"run": "site01", "count": 2}])
    _atomic_json(
        target / "selection.json",
        {"selected": [{"task": f"wavgen-test-{i:02}", "run": "site01"} for i in [2, 3]]},
    )
    _atomic_json(
        target / "cleaned/verification.json",
        {"input_trajectories": 2, "accepted": 2, "rejected": 0},
    )
    out = tmp_path / "shared"
    out.mkdir()
    _atomic_json(target / "export.receipt.json", {"target": str(out)})
    monkeypatch.setattr(sys, "argv", ["assemble.py", "report", "--run", "final01"])
    assemble.main()
    result = json.loads((out / "difficulty_report.json").read_text())
    assert result["teacher_model"] == "configured-teacher"
    assert result["student_model"] == "z-ai/glm-5.3-flash"
    assert result["student_counts"] == {"PASS": 1, "FAIL": 0, "HOLD": 1}
    assert result["student_effective"] == 1 and result["student_success_rate"] == 1
    assert result["candidate_count"] == 3 and result["delivered_count"] == 2
    assert result["teacher_step_statistics"]["median"] == 7
    assert result["task_type_distribution"] == {"NAVIGATE": 1, "MUTATE": 1}
