from __future__ import annotations

import importlib.resources
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2]
    / "devs/exps/eval/browsergym.webarena_verified/campaign/generate_plan.py"
)
SPEC = importlib.util.spec_from_file_location("webarena_verified_campaign_plan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plan_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan_module)


def test_full_plan_matches_approved_120_layout_and_binds_config(tmp_path, monkeypatch):
    pytest.importorskip("webarena_verified")
    dataset = Path(
        str(
            importlib.resources.files("webarena_verified").joinpath(
                "assets/dataset/webarena-verified.json"
            )
        )
    )
    config = json.loads(
        (SCRIPT.parents[1] / "EXPERIMENT_CONFIG.generic-desktop-dual-v2.json").read_text()
    )
    config.update(run_id="wav-generic-qwen38max-full120-20261007-v2", task_count=812)
    (tmp_path / "EXPERIMENT_CONFIG.json").write_text(json.dumps(config))
    baseline = datetime(2026, 10, 6, tzinfo=UTC)
    rows = []
    for task in range(812):
        index = tmp_path / "history" / "index" / f"task-{task:04d}"
        index.mkdir(parents=True)
        (index / "baseline_receipt.json").write_text(
            json.dumps({"created_at": baseline.isoformat()})
        )
        (index / "completed.json").write_text(
            json.dumps(
                {
                    "completed_at": (baseline + timedelta(seconds=task + 1)).isoformat(),
                }
            )
        )
        rows.append(json.dumps({"task_id": task, "origin_result_root": str(tmp_path / "history")}))
    history = tmp_path / "timing.jsonl"
    history.write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--config",
            str(tmp_path / "EXPERIMENT_CONFIG.json"),
            "--historical-timing",
            str(history),
            "--dataset",
            str(dataset),
        ],
    )
    plan_module.main()
    plan = json.loads((tmp_path / "worker-plan.json").read_text())
    assert len(plan["workers"]) == 120
    assert plan["task_count"] == 812
    assert sum("map" in row["capabilities"] for row in plan["workers"].values()) == 45
    counts = {}
    for row in plan["workers"].values():
        key = tuple(row["capabilities"])
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        ("map",): 36,
        ("map", "wikipedia"): 8,
        ("shopping_admin",): 24,
        ("shopping",): 18,
        ("gitlab",): 20,
        ("reddit",): 6,
        ("gitlab", "reddit"): 4,
        ("gitlab", "wikipedia"): 1,
        ("reddit", "shopping"): 2,
        ("map", "shopping_admin"): 1,
    }
    assigned = [
        int(line.split("\t", 1)[0])
        for worker in plan["workers"]
        for line in (tmp_path / "workers" / worker / "tasks.tsv").read_text().splitlines()
    ]
    assert sorted(assigned) == list(range(812))
    manifest = (tmp_path / "PLAN_MANIFEST.sha256").read_text()
    assert "  EXPERIMENT_CONFIG.json\n" in manifest
    assert len(manifest.splitlines()) == 242
    with pytest.raises(ValueError, match="already-published"):
        plan_module.main()


def test_lpt_assignment_preserves_long_first_after_short_qualification():
    catalog = {task: {"map"} for task in range(6)}
    profiles = {"01": {"capabilities": ["map"]}, "02": {"capabilities": ["map"]}}
    durations = {0: 1, 1: 2, 2: 90, 3: 80, 4: 70, 5: 60}
    assignments = plan_module.assign_full(catalog, profiles, durations)
    assert assignments == {"01": [0, 2, 5], "02": [1, 3, 4]}
