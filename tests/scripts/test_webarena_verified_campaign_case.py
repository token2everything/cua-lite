from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2]
    / "devs/exps/eval/browsergym.webarena_verified/campaign/run-case.sh"
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
    assert result.stdout.strip() == "3"


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
