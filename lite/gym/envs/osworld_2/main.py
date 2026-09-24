"""Register OSWorld 2 releases from one catalog.

``osworld_2@<task>`` selects the 2026.06.24 release;
``osworld_2.v2026_08_08@<task>`` and ``osworld_2.v2_1@<task>`` select
2026.08.08 and 2.1. All releases use release-matched external Workers.
Task discovery requires neither Docker nor an upstream OSWorld checkout.
"""

from __future__ import annotations

from functools import partial

from lite.gym.envs.osworld_2.catalog import load_release, release_names
from lite.gym.envs.osworld_2.worker_env import OSWorldWorkerEnv
from lite.gym.registry import register, registry
from lite.gym.services import BackendFamily, EnvServices, register_family, register_services
from lite.gym.utils import config as env_config

ENV_CONFIG_DIRS = {}

for _name in release_names():
    _release = load_release(_name)
    ENV_CONFIG_DIRS[_release.env_id] = _release.path
    _cfg = env_config.load(str(_release.path))
    for _task_id in _release.tasks:
        register(
            key=f"{_release.env_id}@{_task_id}",
            split="eval",
            entry_point=partial(
                OSWorldWorkerEnv,
                task_id=_task_id,
                release=_release.name,
                **_cfg.env_kwargs,
            ),
            metadata=OSWorldWorkerEnv._task_metadata(_task_id, _release),
        )
    registry.set_env_supported_kwargs(
        _release.env_id,
        {
            "connection_file",
            "max_steps",
            "post_action_delay",
            "valid_actions",
            "extra_tools",
            "experiment",
            "results_root",
            "run_id",
            "phase_index",
        },
    )
    register_services(_release.env_id, EnvServices())
    register_family(_release.env_id, BackendFamily.REMOTE)
    registry.set_env_make_kwargs(_release.env_id, _cfg.make_kwargs)
