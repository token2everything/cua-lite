"""A failed Map case cannot start the next copy without its owner proof."""

from __future__ import annotations

import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sandbox import _atomic_json

DRIVER = Path(__file__).resolve().parents[1] / "runs/campaigns/ext35020261012a/run_cohort.py"


def execute(tmp_path, monkeypatch, site, failures):
    campaign = tmp_path / "runs/campaigns/fixture"
    campaign.mkdir(parents=True)
    (campaign / "execution_logs").mkdir()
    script = campaign / "run_cohort.py"
    shutil.copyfile(DRIVER, script)
    directory = tmp_path / "runs/generated/cohortfixture"
    tasks = [f"wavgen-driver-{site}-610799{i}" for i in range(1, 4)]
    config = directory / "plans/task_configs.json"
    _atomic_json(config, {"tasks" if site == "map" else "cases": [
        {"generated_task_id": task} for task in tasks
    ]})
    calls = []
    errors = []

    def run(command, **kwargs):
        phase = command[3]
        if phase != "ground":
            calls.append(phase)
            return SimpleNamespace(returncode=0)
        task = command[command.index("--task") + 1]
        number = tasks.index(task) + 1
        calls.append(task)
        if number not in failures:
            return SimpleNamespace(returncode=0)
        mode = failures[number]
        handle = task + "-original-handle"
        if mode != "unknown_handle":
            _atomic_json(directory / "commands" / ("ground-" + task + ".handle.json"),
                         {"command_id": handle})
        if mode != "missing":
            proof = {"status": "not_created" if mode == "not_created" else "reaped",
                     "task": task, "run": "cohortfixture", "profile": "author", "attempt": "01",
                     "terminal_case_handle": handle, "terminal_case_exit_code": 35}
            if mode == "wrong_task":
                proof["task"] = tasks[-1]
            elif mode == "wrong_run":
                proof["run"] = "another-run"
            elif mode == "wrong_handle":
                proof["terminal_case_handle"] = "another-original-handle"
            elif mode == "wrong_attempt":
                proof["attempt"] = "02"
            elif mode == "wrong_profile":
                proof["profile"] = "teacher"
            elif mode == "not_terminal":
                proof["terminal_case_exit_code"] = None
            elif mode == "pending":
                proof["status"] = "pending"
            _atomic_json(directory / "commands" / ("prune-databases-" + task + "-author-01.receipt.json"), proof)
        error = subprocess.CalledProcessError(1, command)
        errors.append(error)
        raise error

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [str(script), "--run", "cohortfixture", "--site", site, "--phase", "ground"])
    return script, directory, tasks, calls, errors


@pytest.mark.parametrize("mode", ["missing", "wrong_task", "wrong_run", "wrong_handle", "wrong_attempt", "wrong_profile", "not_terminal", "unknown_handle", "pending"])
def test_map_unproven_reclamation_stops_before_next_case_and_preserves_failure(tmp_path, monkeypatch, mode):
    script, directory, tasks, calls, errors = execute(tmp_path, monkeypatch, "map", {1: mode})
    with pytest.raises(subprocess.CalledProcessError) as raised:
        runpy.run_path(str(script), run_name="__main__")
    assert raised.value is errors[0]
    assert calls == [tasks[0]]
    receipt = json.loads((directory / "author_cohort_receipt.json").read_text())
    assert receipt["cohort_completed"] is False and receipt["stopped_task"] == tasks[0]
    assert receipt["failed"][0]["map_database_reclamation_confirmed"] is False
    assert receipt["model_calls"] == 0


@pytest.mark.parametrize("mode", ["reaped", "not_created"])
def test_map_proven_original_case_can_continue_without_reclassifying_failure(tmp_path, monkeypatch, mode):
    script, directory, tasks, calls, _errors = execute(tmp_path, monkeypatch, "map", {1: mode})
    runpy.run_path(str(script), run_name="__main__")
    assert calls == [*tasks, "check"]
    receipt = json.loads((directory / "author_cohort_receipt.json").read_text())
    assert [row["task"] for row in receipt["failed"]] == [tasks[0]]
    assert receipt["failed"][0]["map_database_reclamation_confirmed"] is True


def test_map_later_unproven_case_retains_earlier_recorded_failure_and_stops(tmp_path, monkeypatch):
    script, directory, tasks, calls, errors = execute(tmp_path, monkeypatch, "map", {1: "reaped", 2: "missing"})
    with pytest.raises(subprocess.CalledProcessError) as raised:
        runpy.run_path(str(script), run_name="__main__")
    assert raised.value is errors[-1]
    assert calls == tasks[:2]
    receipt = json.loads((directory / "author_cohort_receipt.json").read_text())
    assert [row["task"] for row in receipt["failed"]] == tasks[:2]
    assert receipt["stopped_task"] == tasks[1]
    assert receipt["failed"][0]["map_database_reclamation_confirmed"] is True


def test_gitlab_failure_keeps_its_existing_continue_then_check_behavior(tmp_path, monkeypatch):
    script, directory, tasks, calls, _errors = execute(tmp_path, monkeypatch, "gitlab", {1: "missing"})
    runpy.run_path(str(script), run_name="__main__")
    assert calls == [*tasks, "check"]
    receipt = json.loads((directory / "author_cohort_receipt.json").read_text())
    assert len(receipt["failed"]) == 1
    assert "map_database_reclamation_confirmed" not in receipt["failed"][0]
