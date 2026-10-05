from __future__ import annotations

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


def test_disk_aware_full_plan_covers_812_with_four_map_workers(tmp_path, monkeypatch):
    pytest.importorskip("webarena_verified")
    dataset = importlib.resources.files("webarena_verified").joinpath(
        "assets/dataset/webarena-verified.json"
    )
    monkeypatch.setattr(plan_module, "ROOT", tmp_path)
    monkeypatch.setattr(plan_module, "DATASET", Path(str(dataset)))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--full", "--disk-aware-50"])

    plan_module.main()

    plan = json.loads((tmp_path / "worker-plan.json").read_text())
    assert plan["selection"] == "full"
    assert plan["task_count"] == 812
    assert list(plan["workers"]) == [f"{worker:02d}" for worker in range(1, 51)]
    assert sum("map" in worker["capabilities"] for worker in plan["workers"].values()) == 4
    assigned = [
        int(line.split("\t", 1)[0])
        for worker in plan["workers"]
        for line in (tmp_path / "workers" / worker / "tasks.tsv").read_text().splitlines()
    ]
    assert sorted(assigned) == list(range(812))
    assert max(worker["tasks"] for worker in plan["workers"].values()) == 32
    assert len((tmp_path / "PLAN_MANIFEST.sha256").read_text().splitlines()) == 101
