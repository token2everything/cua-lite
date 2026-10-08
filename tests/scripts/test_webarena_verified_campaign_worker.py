from __future__ import annotations

import shlex
import subprocess
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
