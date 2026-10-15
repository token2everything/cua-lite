from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sandbox import _atomic_json, _sha

import assemble
import pipeline


@pytest.fixture
def sampling(tmp_path, monkeypatch):
    coordinator = pipeline.Coordinator.__new__(pipeline.Coordinator)
    coordinator.run = tmp_path / "run"
    coordinator.run.mkdir()
    coordinator.args = SimpleNamespace(
        run="run", task=None, clean_python=Path("/fake/clean-python")
    )
    coordinator.sites = ["shopping", "gitlab"]
    coordinator.release_root = coordinator.run / "release"
    coordinator.config = {"helper_SHA256": "a" * 64, "source_SHA256": "b" * 64}
    coordinator.worker = coordinator.run / "infra/workers/execution.json"
    task = "wavgen-test-0001"
    execution = coordinator.run / "plans/model_execution.json"
    _atomic_json(
        execution,
        {
            "teacher": {"requested_model": "qwen3.8-max"},
            "student": {"model": "z-ai/glm-5.3-flash"},
            "attempt_policy": {
                "semantic_episodes": {"teacher": 3, "student": 1},
                "provider_transport_retries": 2,
            },
        },
    )
    _atomic_json(
        coordinator.run / "model_probe_pass.json", {"teacher_snapshot_SHA256": _sha(execution)}
    )
    (coordinator.run / "private.jsonl").write_text("{}\n")
    (coordinator.run / "public.jsonl").write_text("{}\n")
    _atomic_json(
        coordinator.run / "admitted.json",
        {"tasks": [task], "private_SHA256": _sha(coordinator.run / "private.jsonl")},
    )
    metadata = {"run_id": "run", "session_nonce": "test-nonce"}
    _atomic_json(coordinator.worker, {"id": "owned-sandbox", "metadata": metadata})

    class Client:
        def __init__(self):
            self.commands = {}
            self.episode_calls = []

        def detail(self, sandbox_id):
            assert sandbox_id == "owned-sandbox"
            return {"metadata": metadata}

        def command(self, sandbox_id, command, *, label, **kwargs):
            assert sandbox_id == "owned-sandbox"
            assert label not in self.commands, "A completed attempt was submitted again"
            handle = "original-" + label
            self.commands[label] = {"body": command, "handle": handle}
            if not label.endswith("-archive-directories"):
                self.episode_calls.append(label)
            return handle

        def command_status(self, sandbox_id, handle):
            return {"id": handle, "running": False, "exit_code": 0}

    client = Client()
    coordinator.client = client
    outcomes = {"teacher": ["fail", "fail", "fail"], "student": ["fail"]}

    def collect(label, members):
        selected_task, profile, number = label.rsplit("-", 2)
        attempt = profile + "-" + number
        prefix = "reference_attempts" if profile == "teacher" else "glm_attempts"
        assert members == [
            "cases/" + label,
            prefix + "/" + selected_task + "/" + attempt,
            "baselines/" + selected_task,
        ]
        body = client.commands[label]["body"]
        assert "/mnt/results/v04/run_case.sh" in body
        assert '["shopping", "gitlab"]' in body
        assert profile + "-" + number in body
        assert "test-nonce owned-sandbox " + number in body
        episode = coordinator.run / prefix / selected_task / attempt
        episode.mkdir(parents=True, exist_ok=False)
        status = outcomes[profile][int(number) - 1]
        if status == "absent":
            return
        _atomic_json(
            episode / "reference_receipt.json",
            {
                "status": status,
                "steps": 7 + int(number),
                "execution_status": "completed" if status in {"pass", "fail"} else "failed",
                "checks": {"har_playwright_results_equal": True},
                "har_results": [{"status": "success", "score": int(status == "pass")}],
                "playwright_results": [{"status": "success", "score": int(status == "pass")}],
            },
        )

    monkeypatch.setattr(coordinator, "collect", collect)
    quality_outcomes = {}
    cleaned = []

    def cleaner(argv, **kwargs):
        assert kwargs["check"] is False
        assert argv[0] == "/fake/clean-python" and argv[1] == "-B"
        assert argv[-1] == "--formal"
        manifest = Path(argv[argv.index("--input-manifest") + 1])
        output = Path(argv[argv.index("--output-root") + 1])
        row = json.loads(manifest.read_text())
        assert row["model_execution_sha256"] == _sha(execution)
        assert row["task_id"] in row["media_mount"]["source_root"]
        assert output.name in row["media_mount"]["source_root"]
        assert row["trajectory"].endswith(output.name + "/agent_trajectory.jsonl")
        cleaned.append(output.name)
        outcome = quality_outcomes.get(output.name, "pass")
        if outcome == "launch_error":
            raise FileNotFoundError("synthetic missing cleaner interpreter")
        output.mkdir(parents=True, exist_ok=False)
        if outcome == "crash":
            return SimpleNamespace(returncode=1, stdout="", stderr="synthetic cleaner bug")
        _atomic_json(
            output / "verification.json",
            {
                "scope": "v04_formal",
                "operations": 46,
                "input_trajectories": 1,
                "accepted": int(outcome == "pass"),
                "rejected": int(outcome == "reject"),
            },
        )
        return SimpleNamespace(
            returncode=0 if outcome == "pass" else 2, stdout="quality receipt", stderr=""
        )

    monkeypatch.setattr(pipeline.subprocess, "run", cleaner)
    return SimpleNamespace(
        coordinator=coordinator,
        task=task,
        client=client,
        outcomes=outcomes,
        quality_outcomes=quality_outcomes,
        cleaned=cleaned,
    )


def test_teacher_three_failures_preserve_all_scores_and_stop_at_budget(sampling):
    case = sampling
    case.coordinator.sample()
    summary = json.loads(
        (case.coordinator.run / "model_results" / (case.task + ".teacher.json")).read_text()
    )
    assert summary["status"] == "FAIL" and summary["selected_attempt"] is None
    assert summary["attempt_counts"] == {"PASS": 0, "FAIL": 3, "HOLD": 0}
    assert [row["attempt"] for row in summary["attempts"]] == [
        "teacher-01",
        "teacher-02",
        "teacher-03",
    ]
    assert all(row["har_results"][0]["score"] == 0 for row in summary["attempts"])
    assert all(row["original_handle"].startswith("original-") for row in summary["attempts"])
    assert len(case.client.episode_calls) == 4
    assert not any("teacher-04" in label for label in case.client.commands)
    assert len(list((case.coordinator.run / "attempt_results").glob("*.teacher-*.json"))) == 3
    assert case.cleaned == []


def test_teacher_stops_at_second_owner_pass_independently_of_student(sampling):
    case = sampling
    case.outcomes["teacher"] = ["fail", "pass", "pass"]
    case.outcomes["student"] = ["error"]
    case.coordinator.sample()
    summary = json.loads(
        (case.coordinator.run / "model_results" / (case.task + ".teacher.json")).read_text()
    )
    assert summary["status"] == "PASS" and summary["attempt"] == "teacher-02"
    assert summary["selected_attempt"] == "teacher-02"
    assert summary["attempt_counts"] == {"PASS": 1, "FAIL": 1, "HOLD": 0}
    assert summary["quality_status"] == "PASS" and case.cleaned == ["teacher-02"]
    assert case.client.episode_calls == [
        case.task + "-teacher-01",
        case.task + "-teacher-02",
        case.task + "-student-01",
    ]
    student = json.loads(
        (case.coordinator.run / "model_results" / (case.task + ".student.json")).read_text()
    )
    assert student["status"] == "HOLD" and len(student["attempts"]) == 1


def test_successful_teacher_with_quality_rejection_is_not_rolled_again(sampling):
    case = sampling
    case.outcomes["teacher"] = ["pass", "pass", "pass"]
    case.quality_outcomes["teacher-01"] = "reject"
    case.coordinator.sample()
    summary = json.loads(
        (case.coordinator.run / "model_results" / (case.task + ".teacher.json")).read_text()
    )
    assert summary["status"] == "PASS" and summary["quality_status"] == "REJECT"
    assert summary["attempt_counts"] == {"PASS": 1, "FAIL": 0, "HOLD": 0}
    assert summary["quality_counts"] == {"PASS": 0, "REJECT": 1, "HOLD": 0}
    assert summary["attempt"] == "teacher-01" and case.cleaned == ["teacher-01"]
    assert case.client.episode_calls == [case.task + "-teacher-01", case.task + "-student-01"]


@pytest.mark.parametrize("outcome", ["crash", "launch_error"])
def test_cleaner_infrastructure_failure_is_quality_hold_with_owner_pass(sampling, outcome):
    case = sampling
    case.outcomes["teacher"] = ["pass", "pass", "pass"]
    case.quality_outcomes["teacher-01"] = outcome
    case.coordinator.sample()
    summary = json.loads(
        (case.coordinator.run / "model_results" / (case.task + ".teacher.json")).read_text()
    )
    assert summary["status"] == "PASS" and summary["quality_status"] == "HOLD"
    assert summary["attempt_counts"] == {"PASS": 1, "FAIL": 0, "HOLD": 0}
    assert summary["quality_counts"]["HOLD"] == 1
    assert len(case.client.episode_calls) == 2
    assert (case.coordinator.run / "attempt_quality" / case.task / "teacher-01.stderr.txt").exists()


def test_three_teacher_holds_are_preserved_as_unscored(sampling):
    case = sampling
    case.outcomes["teacher"] = ["error", "hold", "absent"]
    case.coordinator.sample()
    summary = json.loads(
        (case.coordinator.run / "model_results" / (case.task + ".teacher.json")).read_text()
    )
    assert summary["status"] == "HOLD" and summary["effective_scored"] is False
    assert summary["attempt_counts"] == {"PASS": 0, "FAIL": 0, "HOLD": 3}
    assert len(case.client.episode_calls) == 4 and case.cleaned == []


def test_resume_reconstructs_summary_without_model_post_or_reclean(sampling):
    case = sampling
    case.outcomes["teacher"] = ["fail", "pass", "pass"]
    case.coordinator.sample()
    result = case.coordinator.run / "model_results" / (case.task + ".teacher.json")
    before = result.read_bytes()
    records = {
        path: path.read_bytes()
        for path in (case.coordinator.run / "attempt_results").glob("*.json")
    }
    calls = list(case.client.episode_calls)
    result.unlink()
    case.coordinator.sample()
    assert result.read_bytes() == before
    assert case.client.episode_calls == calls and case.cleaned == ["teacher-02"]
    assert all(path.read_bytes() == content for path, content in records.items())


def test_task_filter_samples_only_selected_task(sampling):
    case = sampling
    second = "wavgen-test-0002"
    _atomic_json(
        case.coordinator.run / "admitted.json",
        {
            "tasks": [case.task, second],
            "private_SHA256": _sha(case.coordinator.run / "private.jsonl"),
        },
    )
    case.coordinator.args.task = second
    case.outcomes["teacher"] = ["pass", "pass", "pass"]
    case.coordinator.sample()
    assert case.client.episode_calls == [second + "-teacher-01", second + "-student-01"]
    assert not (case.coordinator.run / "model_results" / (case.task + ".teacher.json")).exists()


@pytest.mark.parametrize("status", ["pass", "fail", "error"])
def test_student_budget_is_one_episode_for_every_outcome(sampling, status):
    case = sampling
    case.outcomes["teacher"] = ["pass", "pass", "pass"]
    case.outcomes["student"] = [status]
    case.coordinator.sample()
    summary = json.loads(
        (case.coordinator.run / "model_results" / (case.task + ".student.json")).read_text()
    )
    assert summary["semantic_episode_budget"] == 1
    assert [row["attempt"] for row in summary["attempts"]] == ["student-01"]
    assert summary["status"] == (status.upper() if status != "error" else "HOLD")
    assert case.client.episode_calls.count(case.task + "-student-01") == 1
    assert not any("student-02" in label for label in case.client.commands)


def test_interrupted_quality_gate_is_not_relaunched_or_used_to_reroll(sampling):
    case = sampling
    case.outcomes["teacher"] = ["pass", "pass", "pass"]
    case.coordinator.sample()
    result = case.coordinator.run / "model_results" / (case.task + ".teacher.json")
    result.unlink()
    attempt_path = case.coordinator.run / "attempt_results" / (case.task + ".teacher-01.json")
    info = json.loads(attempt_path.read_text())
    info["quality"] = None
    _atomic_json(attempt_path, info)
    command_path = case.coordinator.run / "attempt_quality" / case.task / "teacher-01.command.json"
    _atomic_json(command_path, {"status": "started", "argv": ["synthetic interrupted cleaner"]})
    calls = list(case.client.episode_calls)
    case.coordinator.sample()
    summary = json.loads(result.read_text())
    assert summary["status"] == "PASS" and summary["quality_status"] == "HOLD"
    assert case.client.episode_calls == calls and case.cleaned == ["teacher-01"]


def test_initialize_freezes_absolute_episode_budgets_and_transport_retries(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for directory in ["infra", "runtime", "vendor", "lite", "devs", "scripts"]:
        (root / directory).mkdir(parents=True)
    (root / pipeline.HELPER).write_text("# synthetic frozen helper\n")
    (root / "runtime/image-lock.env").write_text("# synthetic fixed image lock\n")
    (root / "vendor/evaluator_build.tar.gz").write_bytes(b"synthetic archive")
    models = {
        "models": {
            profile: {
                "model": profile,
                "expected_returned_model": profile + "-response",
                "api_type": "openai_chat",
                "params": {},
                "history_turns": 19,
            }
            for profile in ["teacher", "student"]
        }
    }
    (root / "runtime/models.example.json").write_text(json.dumps(models))
    monkeypatch.setattr(pipeline, "ROOT", root)
    monkeypatch.setattr(
        pipeline,
        "_read_env",
        lambda path: {
            "OPENAI_BASE_URL": "https://synthetic.test",
            "OPENAI_API_KEY": "synthetic-test-key",
        },
    )
    pipeline.initialize(
        SimpleNamespace(run="test-init", sites=["shopping", "gitlab"], env=root / "unused.env")
    )
    execution = json.loads(
        (root / "runs/generated/test-init/plans/model_execution.json").read_text()
    )
    policy = execution["attempt_policy"]
    assert execution["teacher"]["requested_model"] == "teacher"
    assert execution["teacher"]["expected_returned_model"] == "teacher-response"
    assert execution["student"]["model"] == "student"
    assert execution["student"]["expected_returned_model"] == "student-response"
    assert "returned_probe_model" not in execution["teacher"]
    assert "returned_probe_model" not in execution["student"]
    assert policy["semantic_episodes"] == {"teacher": 3, "student": 1}
    assert policy["provider_transport_retries"] == 2
    assert "semantic_episode_retries" not in policy
    assert "proven_infrastructure_retries" not in policy


@pytest.fixture
def reported_attempts(sampling):
    """Produce four actual sampler summaries with fake transport and cleaner boundaries."""
    case = sampling
    tasks = [f"wavgen-report-{number:04d}" for number in range(1, 5)]
    definitions = [
        {
            "generated_task_id": task,
            "task_spec": {"sites": ["shopping", "gitlab"]},
            "task_definition": {
                "task_type": "MUTATE",
                "goal_template": "Synthetic reporting fixture {name}",
                "parameters": {"name": task},
                "completion_events": [],
            },
            "grounding": {
                "family": "synthetic-report",
                "final_state_checks": [],
                "field_evidence": [],
            },
        }
        for task in tasks
    ]
    private = case.coordinator.run / "private.jsonl"
    private.write_text("".join(json.dumps(row) + "\n" for row in definitions))
    _atomic_json(
        case.coordinator.run / "admitted.json",
        {"tasks": tasks, "private_SHA256": _sha(private)},
    )
    for task, row in zip(tasks, definitions, strict=True):
        candidate = case.coordinator.run / "candidates" / task / "private.jsonl"
        candidate.parent.mkdir(parents=True)
        candidate.write_text(json.dumps(row) + "\n")
    scenarios = [
        (["fail", "pass", "pass"], "pass", "pass"),
        (["pass", "pass", "pass"], "reject", "fail"),
        (["fail", "fail", "fail"], "pass", "error"),
        (["error", "hold", "absent"], "pass", "pass"),
    ]
    for task, (teacher, quality, student) in zip(tasks, scenarios, strict=True):
        case.coordinator.args.task = task
        case.outcomes["teacher"] = teacher
        case.outcomes["student"] = [student]
        case.quality_outcomes.update({f"teacher-{number:02d}": quality for number in [1, 2, 3]})
        case.coordinator.sample()
    case.tasks = tasks
    return case


@pytest.mark.parametrize("after_export", [False, True])
def test_report_keeps_first_success_retries_and_quality_as_separate_measures(
    reported_attempts, after_export
):
    case = reported_attempts
    run = case.coordinator.run
    if after_export:
        target = run / "synthetic-review"
        target.mkdir()
        (target / "multimodal_image_text_dialogs_v3.jsonl").write_text(
            json.dumps({"metadata": {"source_task_id": case.tasks[0]}}) + "\n"
        )
        _atomic_json(run / "export.receipt.json", {"target": str(target)})
    case.coordinator.report()
    report = json.loads((run / "difficulty_report.json").read_text())
    assert report["teacher_model"] == "qwen3.8-max"
    assert report["student_model"] == "z-ai/glm-5.3-flash"
    sampling_result = report["teacher_sampling"]
    assert sampling_result == {
        "question_count": 4,
        "success_at_1": 1,
        "effective_at_1": 3,
        "success_rate_at_1": pytest.approx(1 / 3),
        "success_within_3": 2,
        "effective_within_3": 3,
        "success_rate_within_3": pytest.approx(2 / 3),
        "attempt_count_distribution": {"1": 1, "2": 1, "3": 2},
        "quality_counts": {"PASS": 1, "REJECT": 1, "HOLD": 0},
    }
    teachers = {row["task"]: row for row in report["rows"] if row["profile"] == "teacher"}
    assert teachers[case.tasks[0]]["selected_attempt"] == "teacher-02"
    assert teachers[case.tasks[1]]["status"] == "PASS"
    assert teachers[case.tasks[1]]["quality_status"] == "REJECT"
    assert len(teachers[case.tasks[1]]["attempts"]) == 1
    assert teachers[case.tasks[2]]["status"] == "FAIL"
    assert teachers[case.tasks[3]]["status"] == "HOLD"
    assert report["teacher_PASS"] == 2
    assert report["candidate_count"] == report["admitted_count"] == 4
    if after_export:
        assert report["delivered_count"] == 1
        assert report["student_effective"] == 1
        assert report["student_counts"] == {"PASS": 1, "FAIL": 0, "HOLD": 0}
        assert json.loads((target / "difficulty_report.json").read_text()) == report
    else:
        assert report["delivered_count"] is None
        assert report["student_effective"] == 3
        assert report["student_counts"] == {"PASS": 2, "FAIL": 1, "HOLD": 1}


def test_report_all_holds_have_no_effective_success_rate(reported_attempts):
    case = reported_attempts
    for path in (case.coordinator.run / "model_results").glob("*.json"):
        if not path.name.startswith(case.tasks[3] + "."):
            path.unlink()
    case.coordinator.report()
    result = json.loads((case.coordinator.run / "difficulty_report.json").read_text())[
        "teacher_sampling"
    ]
    assert result["question_count"] == 1
    assert result["success_at_1"] == result["success_within_3"] == 0
    assert result["effective_at_1"] == result["effective_within_3"] == 0
    assert result["success_rate_at_1"] is result["success_rate_within_3"] is None
    assert result["quality_counts"] == {"PASS": 0, "REJECT": 0, "HOLD": 0}


@pytest.mark.parametrize("entrypoint", ["pipeline", "assemble"])
@pytest.mark.parametrize("released", [False, True])
def test_exports_preserve_every_original_attempt_and_quality_artifact(
    reported_attempts, tmp_path, monkeypatch, entrypoint, released
):
    case = reported_attempts
    source = case.coordinator.run
    for name in [
        "run.json",
        "release_manifest.json",
        "replay_receipts.jsonl",
        "contamination_receipts.jsonl",
        "prepared.json",
        "release.intent.json",
        "release.receipt.json",
    ]:
        if not released and name in {"release.intent.json", "release.receipt.json"}:
            continue
        _atomic_json(source / name, {"synthetic": name})
    (source / "infra/source-runtime.tar.gz").write_bytes(b"synthetic frozen source archive")
    _atomic_json(source / "infra/source-runtime.tar.manifest.json", {"synthetic": True})
    for relative, content in {
        "pipeline.py": b"# frozen coordinator\n",
        "infra/author_cross_site.py": b"# frozen author\n",
        "runtime/models.example.json": b'{"synthetic": "frozen configuration"}\n',
        "vendor/fixed-input.bin": b"\x00\xfffrozen vendor input\n",
    }.items():
        path = source / "release" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    author_source = source / "author_provenance/original-author"
    _atomic_json(author_source / "release_manifest.json", {"synthetic": "author release"})
    _atomic_json(
        author_source / "commands/ground.handle.json",
        {"command_id": "original-author-handle", "sandbox_id": "original-author-sandbox"},
    )
    (author_source / "source-runtime.tar.gz").write_bytes(b"synthetic original author source")
    execution_source = source / "execution_provenance/original-execution"
    _atomic_json(execution_source / "release_manifest.json", {"synthetic": "execution release"})
    _atomic_json(
        execution_source / "commands/teacher-01.handle.json",
        {"command_id": "original-execution-handle", "sandbox_id": "original-execution-sandbox"},
    )
    _atomic_json(
        execution_source / "attempt_results/teacher-01.json",
        {"status": "HOLD", "source_SHA256": "original-execution-source"},
    )
    review_artifacts = {
        "runtime_gui_preflight/cross_site_gui_frames.jsonl":
            b'{"screenshot":"desktop/step_007.png"}\n',
        "runtime_gui_preflight/desktop/step_007.png": b"\x89PNG\r\n\x1a\npreflight screenshot",
        "runtime_gui_preflight/network.har": b'{"log":{"entries":[]}}\n',
        "runtime_gui_preflight/trace.zip": b"PK\x03\x04preflight trace",
        "runtime_gui_preflight_script.py": b"# native GUI preflight\n",
        "runtime_gui_preflight.log": b"original preflight command output\n",
        "runtime_gui_preflight.coordinator.log": b"original coordinator output\n",
        "runtime_gui_preflight.coordinator.receipt.json": b'{"pid":123,"argv":["preflight"]}\n',
        "continuation.coordinator.log": b"prior attempts retained\n",
        "continuation.receipt.json": b'{"pid":124,"script_SHA256":"original"}\n',
        "continuation.status.json": b'{"phase":"sample","prior_attempts_retained":6}\n',
        "r8_migration_full_readonly_review_20261013.json": b'{"status":"PASS"}\n',
        "visible_source_audit.wavgen-test.teacher-01.json": b'{"original_owner_status":"pass"}\n',
        "visible_source_audit.wavgen-test.student-01.json": b'{"original_owner_status":"pass"}\n',
        "audit_r8_teacher3_visible_proof.py": b"# readonly teacher screenshot review\n",
        "audit_r8_student3_visible_proof.py": b"# readonly student screenshot review\n",
        "review_r8_student3_native.py": b"# readonly native response projection review\n",
        "native_response_projection_review.wavgen-test.student-01.json":
            b'{"all_native_messages_preserved":true}\n',
        "7000003.teacher-01.first_response_identity.json": b'{"actual_returned_model":"qwen3.8-max"}\n',
        "plans/root_audits/root_review.json": b'{"status":"PASS","original_path":"root-review.json"}\n',
    }
    for relative, content in review_artifacts.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _atomic_json(
        source / "runtime_gui_preflight/preflight_receipt.json",
        {"status": "PASS", "artifact_SHA256": {
            name: _sha(source / "runtime_gui_preflight" / name)
            for name in ["desktop/step_007.png", "network.har", "trace.zip"]
        }},
    )
    _atomic_json(
        source / "runtime_gui_preflight_pass.json",
        {"status": "PASS", "run": "cohort",
         "script_SHA256": _sha(source / "runtime_gui_preflight_script.py"),
         "receipt_SHA256": _sha(source / "runtime_gui_preflight/preflight_receipt.json")},
    )
    _atomic_json(
        source / "archives/runtime-gui-preflight.receipt.json",
        {"archive": "runtime-gui-preflight.tar.gz", "SHA256": "original archive hash"},
    )
    _atomic_json(source / "private/models_qwen_v2.json", {"api_key": "synthetic private credential"})
    (source / ".env").write_text("OPENAI_API_KEY=synthetic-private-credential\n")
    originals = {
        str(path.relative_to(source)): path.read_bytes()
        for directory in [
            "release", "attempt_results", "attempt_quality", "author_provenance",
            "execution_provenance", "runtime_gui_preflight", "plans"
        ]
        for path in (source / directory).rglob("*")
        if path.is_file()
    }
    originals.update(
        {
            name: (source / relative).read_bytes()
            for name, relative in {
                "release_manifest.json": "release_manifest.json",
                "source-runtime.tar.gz": "infra/source-runtime.tar.gz",
                "source-runtime.tar.manifest.json": "infra/source-runtime.tar.manifest.json",
            }.items()
        }
    )
    originals.update({name: (source / name).read_bytes() for name in review_artifacts})
    originals.update({
        name: (source / name).read_bytes()
        for name in ["runtime_gui_preflight_pass.json"]
    })
    originals["archive_receipts/runtime-gui-preflight.receipt.json"] = (
        source / "archives/runtime-gui-preflight.receipt.json"
    ).read_bytes()
    assert len(list((source / "attempt_results").glob("*.teacher-*.json"))) == 9
    assert any(name.endswith("teacher-01.stderr.txt") for name in originals)
    target = tmp_path / (entrypoint + "-review")

    def export_rows(argv, **kwargs):
        assert "export_review_bundle.py" in argv[2]
        assert Path(argv[argv.index("--target") + 1]) == target
        target.mkdir()
        (target / "multimodal_image_text_dialogs_v3.jsonl").write_text(
            json.dumps({"metadata": {"source_task_id": case.tasks[0]}}) + "\n"
        )

    monkeypatch.setattr(pipeline.subprocess, "run", export_rows)
    if entrypoint == "pipeline":
        case.coordinator.args.target = target
        case.coordinator.args.count = 1
        case.coordinator.export()
        exported = target / "evidence"
        manifest_prefix = ""
    else:
        repo = tmp_path / "aggregate-repo"
        cohort = repo / "runs/generated/cohort"
        shutil.copytree(source, cohort)
        run = repo / "runs/assembled/final"
        _atomic_json(run / "plans/cohorts.json", [{"run": "cohort", "count": 1}])
        _atomic_json(
            run / "selection.json", {"selected": [{"task": case.tasks[0], "run": "cohort"}]}
        )
        monkeypatch.setattr(assemble, "ROOT", repo)
        monkeypatch.setattr(
            sys, "argv", ["assemble.py", "export", "--run", "final", "--target", str(target)]
        )
        if not released:
            with pytest.raises(FileNotFoundError, match="release.intent.json"):
                assemble.main()
            assert not (target / "evidence/cohort").exists()
            return
        assemble.main()
        exported = target / "evidence/cohort"
        manifest_prefix = "cohort/"
    manifest = json.loads((target / "evidence_manifest.json").read_text())["files_SHA256"]
    for relative, original in originals.items():
        artifact = exported / relative
        assert artifact.read_bytes() == original
        assert manifest[manifest_prefix + relative] == _sha(artifact)
    assert not (exported / "commands/ground.handle.json").exists()
    assert not (exported / "commands/teacher-01.handle.json").exists()
    assert not (exported / "private").exists()
    assert not (exported / ".env").exists()
    assert not any("private/models" in name or name.endswith(".env") for name in manifest)
    gate = json.loads((exported / "runtime_gui_preflight_pass.json").read_text())
    assert gate["script_SHA256"] == _sha(exported / "runtime_gui_preflight_script.py")
    assert gate["receipt_SHA256"] == _sha(exported / "runtime_gui_preflight/preflight_receipt.json")
    receipt = json.loads((exported / "runtime_gui_preflight/preflight_receipt.json").read_text())
    for name, digest in receipt["artifact_SHA256"].items():
        assert digest == _sha(exported / "runtime_gui_preflight" / name)
    for name in ["release.intent.json", "release.receipt.json"]:
        assert (exported / "resources" / name).exists() is released
    for task in case.tasks:
        summary = json.loads((exported / "model_results" / (task + ".teacher.json")).read_text())
        assert len(summary["attempts"]) in {1, 2, 3}
