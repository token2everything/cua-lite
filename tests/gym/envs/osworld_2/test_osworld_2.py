"""Public OSWorld release registration stays independent of Worker deployment.

Episode, action, and cleanup behavior is covered under ``worker/``. These tests
exercise discovery and per-release config through the public gym entry point.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from lite.gym.envs.osworld_2.catalog import load_release

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_PREFIXES = ("OSWORLD_2", "OSWORLD_V2026_08_08", "OSWORLD_V2_1")


def _isolated_env() -> dict[str, str]:
    return {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CUA_LITE_ENV_SERVER_")
            and key not in {
                "CUA_LITE_REGISTRATION_MODULES",
                "OSWORLD_WORKER_CONNECTION",
                *(f"{prefix}_CONFIG" for prefix in _CONFIG_PREFIXES),
            }
        },
        "PYTHONPATH": str(_REPO_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def test_all_release_discovery_and_construction_need_no_local_backend(tmp_path):
    script = '''
import importlib.abc
import sys

blocked = {
    "desktop_env",
    "docker",
    "lite.gym.envs.osworld.main",
    "lite.gym.remote.reaper",
    "lite.gym.utils.backend.docker",
    "lite.gym.utils.backend.ports",
}
class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("task discovery imported local backend: " + fullname)
sys.meta_path.insert(0, ImportGuard())

def no_infrastructure(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "subprocess.Popen"}:
        raise AssertionError("task discovery contacted infrastructure: " + event)
sys.addaudithook(no_infrastructure)

import lite.gym as gym
from lite.gym.envs.osworld_2.catalog import load_release, release_names
from lite.gym.envs.osworld_2.main import ENV_CONFIG_DIRS
from lite.gym.envs.osworld_2.worker_env import OSWorldWorkerEnv
from lite.gym.registry import ensure_services
from lite.gym.services import BackendFamily, EnvServices, family_of, services_for

for name in release_names():
    release = load_release(name)
    assert len(gym.registry.task_ids(release.env_id, split="eval")) == 108
    assert family_of(release.env_id) is BackendFamily.REMOTE
    assert type(services_for(release.env_id)) is EnvServices
    ensure_services(release.env_id)
    assert ENV_CONFIG_DIRS[release.env_id] == release.path
    assert "connection_file" in gym.registry.env_supported_kwargs(release.env_id)
    env = gym.make(f"{release.env_id}@054", connection_file="/missing/worker.json")
    assert isinstance(env.unwrapped, OSWorldWorkerEnv)
    assert env.unwrapped.release.name == name
    assert env.unwrapped._admin is None
    assert env.metadata.others == gym.registry.task_metadata(release.env_id, "054").others
assert not blocked.intersection(sys.modules)
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=_isolated_env(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("release_name", "prefix"),
    [
        ("osworld-v2-2026.06.24", "OSWORLD_2"),
        ("osworld-v2-2026.08.08", "OSWORLD_V2026_08_08"),
        ("osworld-v2.1", "OSWORLD_V2_1"),
    ],
)
def test_absolute_config_override_is_isolated_to_its_release(tmp_path, release_name, prefix):
    release = load_release(release_name)
    config = yaml.safe_load((release.path / "configs" / "default.yaml").read_text())
    config["env_kwargs"]["max_steps"] = 37
    config["make_kwargs"]["reset_timeout"] = 1234.0
    override = tmp_path / "override.yaml"
    override.write_text(yaml.safe_dump(config))
    script = '''
import os
import sys

import lite.gym as gym
from lite.gym.envs.osworld_2.catalog import load_release, release_names
from lite.gym.utils import config as env_config

selected_name, prefix = sys.argv[1:]
for name in release_names():
    release = load_release(name)
    env = gym.make(f"{release.env_id}@054")
    cfg = env_config.load(str(release.path))
    selected = name == selected_name
    assert env.unwrapped._max_steps == (37 if selected else 200)
    assert gym.registry.env_make_kwargs(release.env_id)["reset_timeout"] == (
        1234.0 if selected else 2100.0
    )
    assert gym.registry.env_make_kwargs(release.env_id)["loop_detect"] == 0
    assert cfg.config_source == ("override" if selected else "default")
    if selected:
        assert cfg.config_path == os.environ[f"{prefix}_CONFIG"]
'''
    result = subprocess.run(
        [sys.executable, "-c", script, release_name, prefix],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**_isolated_env(), f"{prefix}_CONFIG": str(override)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
