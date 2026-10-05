from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/webarena_verified_qwen.py"
SPEC = importlib.util.spec_from_file_location("webarena_verified_scoring_integrity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _evidence(*, status: str, score: float = 0.0, assertion: str | None = None) -> dict:
    return {
        "task_id": 811,
        "intent_template_id": 999,
        "sites": ["gitlab"],
        "task_revision": 2,
        "status": status,
        "score": score,
        "error_msg": None,
        "webarena_verified_version": "1.2.3",
        "webarena_verified_evaluator_checksum": "evaluator-checksum",
        "webarena_verified_data_checksum": "dataset-checksum",
        "evaluators_results": [
            {
                "evaluator_name": "NetworkEventEvaluator",
                "status": status,
                "score": score,
                "actual_normalized": {"issue": {"assignee_ids": [2330]}},
                "expected": {"issue": {"assignee_ids": [2330]}},
                "assertions": (
                    [{"assertion_name": assertion, "status": "failure"}] if assertion else None
                ),
                "error_msg": "normalization failed" if status == "error" else None,
                "should_not_exist": False,
            }
        ],
    }


def _score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, har: dict, trace: dict) -> dict:
    import webarena_verified.types as official_types
    import webarena_verified.types.eval as official_eval
    import webarena_verified.types.tracing as official_tracing

    from lite.gym.envs.browsergym import webarena_verified

    monkeypatch.setattr(webarena_verified, "install_playwright_trace_loader", lambda: None)
    monkeypatch.setattr(
        official_types,
        "WebArenaVerifiedTask",
        SimpleNamespace(model_validate=lambda config: object()),
    )
    monkeypatch.setattr(
        official_eval,
        "TaskEvalContext",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        official_tracing,
        "NetworkTrace",
        SimpleNamespace(
            from_har=lambda path: "har",
            from_playwright_trace=lambda path: "trace",
        ),
    )

    class Evaluator:
        config = object()

        def evaluate_task(self, *, context):
            evidence = har if context.network_trace == "har" else trace
            return SimpleNamespace(
                score=evidence["score"], model_dump=lambda mode: evidence
            )

    task = SimpleNamespace(
        config={}, evaluator=SimpleNamespace(evaluator=Evaluator())
    )
    har_path = tmp_path / "network.har"
    trace_path = tmp_path / "trace.zip"
    har_path.touch()
    trace_path.touch()
    return runner.score_artifacts(task, "{}", trace_path, har_path)


def test_identical_model_failure_is_scored_and_publishable(tmp_path, monkeypatch):
    failure = _evidence(status="failure")
    result = _score(tmp_path, monkeypatch, failure, failure)

    assert result["score"] == 0.0
    assert result["score_metric"] == "TaskEvalResult.all_or_nothing"
    assert result["inputs_agree"] is True
    assert result["integrity_status"] == "passed"
    assert result["publishable"] is True


def test_same_numeric_score_does_not_hide_evaluator_error(tmp_path, monkeypatch):
    failure = _evidence(status="failure")
    error = _evidence(status="error")
    result = _score(tmp_path, monkeypatch, failure, error)

    assert result["score"] == 0.0
    assert result["score_inputs_agree"] is True
    assert result["inputs_agree"] is False
    assert result["integrity_status"] == "evaluator_error"
    assert result["publishable"] is False


def test_matching_evaluator_errors_are_still_not_publishable(tmp_path, monkeypatch):
    error = _evidence(status="error")
    result = _score(tmp_path, monkeypatch, error, error)

    assert result["inputs_agree"] is True
    assert result["integrity_status"] == "evaluator_error"
    assert result["publishable"] is False


def test_assertion_difference_is_evidence_disagreement(tmp_path, monkeypatch):
    har = _evidence(status="failure", assertion="missing_event")
    trace = _evidence(status="failure", assertion="wrong_value")
    result = _score(tmp_path, monkeypatch, har, trace)

    assert result["score_inputs_agree"] is True
    assert result["inputs_agree"] is False
    assert result["integrity_status"] == "evidence_disagreement"
    assert result["publishable"] is False
