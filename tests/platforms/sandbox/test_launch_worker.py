"""Exercise Worker environment forwarding without platform, Docker or KVM access."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def launch_worker(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "assert args[0] == 'run'\n"
        "environment = {}\n"
        "for index, arg in enumerate(args):\n"
        "    if arg == '--env':\n"
        "        name, separator, value = args[index + 1].partition('=')\n"
        "        environment[name] = value if separator else os.environ[name]\n"
        "Path(os.environ['FAKE_DOCKER_CALL']).write_text(json.dumps({\n"
        "    'arguments': args, 'environment': environment\n"
        "}))\n"
    )
    fake_docker.chmod(0o755)
    launcher = (
        Path(__file__).resolve().parents[3] / "lite/platforms/sandbox/osworld_2/launch_worker.sh"
    )
    # Execute the actual launch section after platform preflight. The preflight
    # owns /proc, KVM and mounted files, which this offline test must not touch.
    _, boundary, launch = launcher.read_text().partition("\nextra=()\n")
    assert boundary, "Worker launch section not found"

    def run(overrides):
        capture = tmp_path / "docker-call.json"
        environment = {
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            "FAKE_DOCKER_CALL": str(capture),
            "OSWORLD_WORKER_IMAGE": "registry.test/osworld-worker@sha256:" + "a" * 64,
            **overrides,
        }
        subprocess.run(
            ["bash", "-euo", "pipefail", "-c",
             "worker_id=test-worker\nruntime_dir=.\nextra=()\n" + launch],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return json.loads(capture.read_text())

    return run


def test_launcher_forwards_explicit_simulator_and_evaluator_configuration(launch_worker):
    supplied = {
        "OSWORLD_USER_SIM_PROVIDER": "openai_compatible",
        "OSWORLD_USER_SIM_MODEL": "simulator-model",
        "OSWORLD_USER_SIM_API_KEY": "test-only simulator key\nwith spaces and $characters",
        "OSWORLD_USER_SIM_API_KEY_ENV": "OPENAI_API_KEY",
        "OSWORLD_USER_SIM_BASE_URL": "http://simulator.example/v1",
        "OSWORLD_USER_SIM_TEMPERATURE": "0.25",
        "OSWORLD_USER_SIM_MAX_TOKENS": "256",
        "OSWORLD_USER_SIM_DEBUG": "0",
        "OSWORLD_EVAL_MODEL_PROVIDER": "openai_compatible",
        "OSWORLD_EVAL_MODEL_NAME": "judge-model",
        "OSWORLD_EVAL_MODEL_API_KEY": "test-only judge key",
        "OSWORLD_EVAL_MODEL_API_KEY_ENV": "OPENAI_API_KEY",
        "OSWORLD_EVAL_MODEL_BASE_URL": "http://judge.example/v1",
        "OSWORLD_EVAL_MODEL_TEMPERATURE": "0",
        "OSWORLD_EVAL_MODEL_MAX_OUTPUT_TOKENS": "512",
        "OSWORLD_EVAL_MODEL_RETRY_ATTEMPTS": "3",
        "OSWORLD_EVAL_MODEL_RETRY_DELAY": "2",
        "OSWORLD_EVAL_MODEL_IMAGE_DETAIL": "high",
        "OSWORLD_EVAL_MODEL_REASONING_EFFORT": "low",
        "OSWORLD_EVAL_MODEL_DEBUG": "0",
        "OSWORLD_EVAL_SAVE_RAW_DIR": "/results/evaluator calls",
    }
    call = launch_worker({**supplied, "OSWORLD_USER_SIM_UNSUPPORTED": "not-forwarded"})
    assert {key: call["environment"].get(key) for key in supplied} == supplied
    assert "OSWORLD_USER_SIM_UNSUPPORTED" not in call["environment"]
    assert supplied["OSWORLD_USER_SIM_API_KEY"] not in call["arguments"]
    assert supplied["OSWORLD_EVAL_MODEL_API_KEY"] not in call["arguments"]
    assert call["arguments"][-1] == "registry.test/osworld-worker@sha256:" + "a" * 64


def test_shared_credentials_do_not_override_official_model_defaults(launch_worker):
    supplied = {
        "OPENAI_API_KEY": "test-only shared credential",
        "OPENAI_BASE_URL": "http://gateway.example/v1",
    }
    call = launch_worker(supplied)
    assert {key: call["environment"].get(key) for key in supplied} == supplied
    assert not any(
        name.startswith(("OSWORLD_USER_SIM_", "OSWORLD_EVAL_MODEL_"))
        for name in call["environment"]
    )
