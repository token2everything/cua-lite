from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2] / "devs/exps/eval/browsergym.webarena_verified/campaign/run-case.sh"
)


def test_gitlab_waits_for_real_page_after_controller():
    source = SCRIPT.read_text()
    controller = source.index("wait_controller http://127.0.0.1:8024/status")
    start = source.index("gitlab_ready=false", controller)
    end = source.index("            ;;", start)
    snippet = source[start:end]
    shell = f"""
set -euo pipefail
attempts=0
curl() {{ attempts=$((attempts + 1)); [ "$attempts" -ge 3 ]; }}
sleep() {{ :; }}
{snippet}
printf '%s\\n' "$attempts"
"""
    result = subprocess.run(["bash", "-c", shell], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "7"


def test_gitlab_page_wait_requires_consecutive_successes():
    source = SCRIPT.read_text()
    start = source.index("gitlab_ready=false")
    end = source.index("            ;;", start)
    snippet = source[start:end]
    shell = f"""
set -euo pipefail
attempts=0
curl() {{
    attempts=$((attempts + 1))
    [ "$attempts" -eq 1 ] || [ "$attempts" -ge 3 ]
}}
sleep() {{ :; }}
{snippet}
printf '%s\\n' "$attempts"
"""
    result = subprocess.run(["bash", "-c", shell], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "7"


def test_gitlab_page_wait_fails_when_only_controller_is_healthy():
    source = SCRIPT.read_text()
    start = source.index("gitlab_ready=false")
    end = source.index("            ;;", start)
    snippet = source[start:end]
    shell = f"""
set -euo pipefail
curl() {{ return 22; }}
sleep() {{ SECONDS=$((SECONDS + 301)); }}
{snippet}
"""
    result = subprocess.run(["bash", "-c", shell], capture_output=True, text=True)
    assert result.returncode == 1
    assert "GitLab root did not become healthy" in result.stderr


def test_attempt_exit_handler_archives_failure_identity_without_changing_exit_code(tmp_path):
    source = SCRIPT.read_text()
    start = source.index("finish_attempt() {")
    end = source.index("trap finish_attempt EXIT", start) + len("trap finish_attempt EXIT")
    archive = tmp_path / "attempt.tar.gz"
    archive.write_bytes(b"partial native request evidence")
    receipt = tmp_path / "attempt.json"
    shell = f"""
set -euo pipefail
RUN_ID=wav-generic-qwen38max-full120-20261007-v2
WORKER=01
SANDBOX_ID=owned-test
TASK_ID=3
RUNNER_SHA256=runner-hash
IMAGE_LOCK_SHA256=image-hash
PLAN_SHA256=plan-hash
SOURCE_SHA256=source-hash
attempt_started_at=2026-10-07T00:00:00Z
ATTEMPT_ARCHIVE={shlex.quote(str(archive))}
ATTEMPT_RECEIPT={shlex.quote(str(receipt))}
CASE_PATH={shlex.quote(str(tmp_path / "unused-case"))}
INDEX={shlex.quote(str(tmp_path / "missing-index"))}
runner_rc=1
model_started=true
runtime_python() {{ {shlex.quote(sys.executable)} "$@"; }}
{source[start:end]}
exit 7
"""
    result = subprocess.run(["bash", "-c", shell], capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    value = json.loads(receipt.read_text())
    assert value["terminal_status"] == "infrastructure_failed"
    assert value["model_started"] is True
    assert value["runner_exit_code"] == 1 and value["case_exit_code"] == 7
    assert value["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
