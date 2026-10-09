from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2] / "devs/exps/eval/browsergym.webarena_verified/campaign/run-worker.sh"
)


@pytest.mark.parametrize("script_name", ["prepare-worker.sh", "run-case.sh", "run-worker.sh"])
@pytest.mark.parametrize(
    ("worker_count", "exit_code"),
    [("1", 0), ("120", 0), ("128", 0), ("256", 0), ("0", 2), ("0120", 2), ("-120", 2)],
)
def test_worker_scripts_validate_positive_run_id_worker_count(script_name, worker_count, exit_code):
    source = SCRIPT.with_name(script_name).read_text()
    validation = next(line for line in source.splitlines() if line.startswith('[[ "$RUN_ID" =~'))
    result = subprocess.run(
        ["bash", "-c", validation],
        env={"RUN_ID": f"wav-generic-qwen38max-full{worker_count}-20261007-v2"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == exit_code, result.stderr


def test_worker_task_plan_is_not_consumed_by_case_stdin(tmp_path):
    source = SCRIPT.read_text()
    start = source.index("while IFS=$'\\t' read -r task_id sites <&3; do")
    end = source.index("\ndone", start) + len("\ndone")
    tasks = tmp_path / "tasks.tsv"
    tasks.write_text("0\tshopping_admin\n44\tgitlab\n63\tshopping_admin\n")
    shell = f"""
set -uo pipefail
TASKS={shlex.quote(str(tasks))}
WORKER_DIR={shlex.quote(str(tmp_path))}
WORKER=01
SANDBOX_ID=owned-test
MODE=full
receipt_valid() {{ return 1; }}
bash() {{ IFS= read -r swallowed || true; return 0; }}
exec 3< "$TASKS"
{source[start:end]}
"""
    subprocess.run(["bash", "-c", shell], check=True, stdin=subprocess.DEVNULL)
    rows = (tmp_path / "progress.tsv").read_text().splitlines()
    assert [row.split("\t")[1] for row in rows] == ["0", "44", "63"]


def test_preparation_consumes_run_owned_stack_and_configured_desktop_assets():
    source = SCRIPT.with_name("prepare-worker.sh").read_text()
    assert "CAL_ROOT" not in source
    assert "wav-qwenrunner-calibration" not in source
    assert "bash /mnt/results/sandbox_stack.sh prepare\n" in source
    assert "bash /mnt/results/sandbox_stack.sh prepare-map" in source
    assert 'config["assets"]["desktop_assets_relative"]' in source
    assert 'tar -xzf "$DESKTOP_ASSETS/desktop-python.tar.gz"' in source
    assert 'git clone "$DESKTOP_ASSETS/osworld-b138d348.bundle"' in source
    assert '"asset_manifest_sha256": os.environ["ASSET_MANIFEST_SHA"]' in source
    assert '"sandbox_stack_sha256": os.environ["SANDBOX_STACK_SHA"]' in source


@pytest.mark.parametrize(
    "condition", ["intact", "changed_bytes", "wrong_size", "different_relative"]
)
def test_desktop_asset_gate_verifies_manifest_before_unpacking(tmp_path, condition):
    assets = tmp_path / "assets"
    relative = "portable/desktop"
    desktop = assets / relative
    desktop.mkdir(parents=True)
    manifest = {
        "assets_mount": str(assets),
        "desktop_assets_relative": relative,
        "files": {},
    }
    for name in ("desktop-python.tar.gz", "osworld-b138d348.bundle"):
        payload = b"pinned desktop dependency"
        (desktop / name).write_bytes(payload)
        manifest["files"][relative + "/" + name] = {
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    config = {
        "assets": {
            "desktop_assets_relative": relative,
            "sandbox_map_data_relative": "assets-full/map-volumes",
        }
    }
    if condition == "changed_bytes":
        (desktop / "desktop-python.tar.gz").write_bytes(b"changed desktop payload!!")
    elif condition == "wrong_size":
        manifest["files"][relative + "/desktop-python.tar.gz"]["size"] += 1
    elif condition == "different_relative":
        config["assets"]["desktop_assets_relative"] = "other/desktop"
    config_path, manifest_path = tmp_path / "config.json", tmp_path / "assets.json"
    config_path.write_text(json.dumps(config))
    manifest_path.write_text(json.dumps(manifest))
    source = SCRIPT.with_name("prepare-worker.sh").read_text()
    start = source.index("import hashlib\nimport json\nimport sys\nfrom pathlib import Path\n")
    code = source[start : source.index("\nPY\n)", start)]
    code = code.replace('Path("/mnt/webarena-verified")', f"Path({str(assets)!r})")
    code = code.replace(
        'Path("/mnt/results/EXPERIMENT_CONFIG.json")', f"Path({str(config_path)!r})"
    )
    code = code.replace('Path("/mnt/results/ASSET_MANIFEST.json")', f"Path({str(manifest_path)!r})")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, "shopping"],
        capture_output=True,
        text=True,
    )
    if condition == "intact":
        assert result.returncode == 0, result.stderr
        assert result.stdout == relative + "\tassets-full/map-volumes\n"
    else:
        assert result.returncode != 0
        assert "differs from" in result.stderr


@pytest.mark.parametrize(
    "condition", ["intact", "changed_control", "changed_prefix", "wrong_size", "missing"]
)
def test_map_asset_gate_reads_explicit_sandbox_view_before_copy(tmp_path, condition):
    assets = tmp_path / "assets"
    desktop_relative = "desktop"
    map_relative = "assets-full/readable-map"
    manifest = {
        "assets_mount": str(assets),
        "desktop_assets_relative": desktop_relative,
        "files": {},
        "read_probes": {},
    }
    for name in ("desktop-python.tar.gz", "osworld-b138d348.bundle"):
        path = assets / desktop_relative / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"desktop")
        manifest["files"][desktop_relative + "/" + name] = {
            "size": 7,
            "sha256": hashlib.sha256(b"desktop").hexdigest(),
        }
    control_relative = "assets-full/map-volumes/tile-db/postgres/global/pg_control"
    remote_control = map_relative + "/tile-db/postgres/global/pg_control"
    control = assets / remote_control
    control.parent.mkdir(parents=True)
    control.write_bytes(b"database identity")
    manifest["files"][control_relative] = {
        "sandbox_relative": remote_control,
        "size": control.stat().st_size,
        "sha256": hashlib.sha256(control.read_bytes()).hexdigest(),
    }
    probe_relative = "assets-full/map-volumes/routing-car/us-northeast-latest.osrm.fileIndex"
    remote_probe = map_relative + "/routing-car/us-northeast-latest.osrm.fileIndex"
    probe = assets / remote_probe
    probe.parent.mkdir(parents=True)
    probe.write_bytes(b"a" * 5000)
    manifest["read_probes"][probe_relative] = {
        "sandbox_relative": remote_probe,
        "size": 5000,
        "prefix_sha256": hashlib.sha256(b"a" * 4096).hexdigest(),
    }
    if condition == "changed_control":
        control.write_bytes(b"changed DB state!")
    elif condition == "changed_prefix":
        probe.write_bytes(b"b" * 5000)
    elif condition == "wrong_size":
        probe.write_bytes(b"a" * 5001)
    elif condition == "missing":
        probe.unlink()
    config_path, manifest_path = tmp_path / "config.json", tmp_path / "manifest.json"
    config_path.write_text(
        json.dumps(
            {
                "assets": {
                    "desktop_assets_relative": desktop_relative,
                    "sandbox_map_data_relative": map_relative,
                }
            }
        )
    )
    manifest_path.write_text(json.dumps(manifest))
    source = SCRIPT.with_name("prepare-worker.sh").read_text()
    start = source.index("import hashlib\nimport json\nimport sys\nfrom pathlib import Path\n")
    code = source[start : source.index("\nPY\n)", start)]
    code = code.replace('Path("/mnt/webarena-verified")', f"Path({str(assets)!r})")
    code = code.replace(
        'Path("/mnt/results/EXPERIMENT_CONFIG.json")', f"Path({str(config_path)!r})"
    )
    code = code.replace('Path("/mnt/results/ASSET_MANIFEST.json")', f"Path({str(manifest_path)!r})")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, "map"], capture_output=True, text=True
    )
    if condition == "intact":
        assert result.returncode == 0, result.stderr
        assert result.stdout == desktop_relative + "\t" + map_relative + "\n"
        assert not (assets / "assets-full/map-volumes").exists()
    else:
        assert result.returncode != 0
        assert "differs from" in result.stderr or "FileNotFoundError" in result.stderr
