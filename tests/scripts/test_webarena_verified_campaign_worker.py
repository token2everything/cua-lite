from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2]
    / "devs/exps/eval/browsergym.webarena_verified/campaign/run-worker.sh"
)


def test_worker_task_plan_is_not_consumed_by_case_stdin(tmp_path):
    source = SCRIPT.read_text()
    start = source.index("while IFS=$'\\t' read -r task_id sites <&3; do")
    end = source.index('done 3< "$TASKS"', start) + len('done 3< "$TASKS"')
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
{source[start:end]}
"""
    subprocess.run(["bash", "-c", shell], check=True, stdin=subprocess.DEVNULL)
    rows = (tmp_path / "progress.tsv").read_text().splitlines()
    assert [row.split("\t")[1] for row in rows] == ["0", "44", "63"]
