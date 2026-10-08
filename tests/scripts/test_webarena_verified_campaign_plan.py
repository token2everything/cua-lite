from __future__ import annotations

import hashlib
import importlib.resources
import importlib.util
import json
import sys
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


@pytest.mark.parametrize("use_timing", [False, True])
def test_full_plan_matches_approved_120_layout_and_binds_config(tmp_path, monkeypatch, use_timing):
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
    timing = tmp_path / "timing.jsonl"
    timing.write_text(
        "".join(
            json.dumps({"task_id": task, "estimated_seconds": task + 1}) + "\n"
            for task in range(812)
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--config",
            str(tmp_path / "EXPERIMENT_CONFIG.json"),
            "--catalog",
            str(dataset),
            *(["--timing", str(timing)] if use_timing else []),
        ],
    )
    plan_module.main()
    plan = json.loads((tmp_path / "worker-plan.json").read_text())
    assert len(plan["workers"]) == 120
    assert plan["task_count"] == 812
    assert plan["timing_source"]["mode"] == ("provided_estimates" if use_timing else "uniform_cost")
    assert plan["timing_source"]["sha256"] == (
        hashlib.sha256(timing.read_bytes()).hexdigest() if use_timing else None
    )
    assert plan["tasks"]["811"]["estimated_seconds"] == (812.0 if use_timing else 1.0)
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


@pytest.mark.parametrize(
    "row",
    [
        {"task_id": 0, "estimated_seconds": 0},
        {"task_id": 0, "estimated_seconds": float("inf")},
        {"task_id": 0, "estimated_seconds": True},
        {"task_id": True, "estimated_seconds": 1},
        {"task_id": 812, "estimated_seconds": 1},
        {"task_id": 0, "origin_result_root": "/private/prior/result"},
        {"task_id": 0, "estimated_seconds": 1, "answer": "private"},
    ],
)
def test_timing_contract_rejects_invalid_or_private_rows(tmp_path, row):
    catalog = tmp_path / "catalog.json"
    payload = json.dumps([{"task_id": task, "sites": ["map"]} for task in range(812)])
    catalog.write_text(payload)
    config = tmp_path / "EXPERIMENT_CONFIG.json"
    config.write_text(
        json.dumps(
            {
                "task_count": 812,
                "benchmark": {"dataset_sha256": hashlib.sha256(payload.encode()).hexdigest()},
            }
        )
    )
    timing = tmp_path / "timing.jsonl"
    timing.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        plan_module.generate_plan(tmp_path, config, catalog, timing)
    assert not (tmp_path / "workers").exists()


def test_plan_rejects_catalog_checksum_mismatch_before_writing(tmp_path):
    config = tmp_path / "EXPERIMENT_CONFIG.json"
    config.write_text(json.dumps({"benchmark": {"dataset_sha256": "0" * 64}}))
    catalog = tmp_path / "catalog.json"
    catalog.write_text("[]")
    with pytest.raises(ValueError, match="catalog differs"):
        plan_module.generate_plan(tmp_path, config, catalog)
    assert not (tmp_path / "worker-plan.json").exists()
