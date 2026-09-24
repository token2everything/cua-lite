"""Discovery must need no source sibling, credentials, network or Worker."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

import lite.gym as gym
from lite.gym.envs.osworld_2.catalog import load_release, release_names
from lite.gym.envs.osworld_2.worker_env import OSWorldWorkerEnv


@pytest.mark.parametrize(
    ("release_name", "env_id"),
    [
        ("osworld-v2-2026.06.24", "osworld_2"),
        ("osworld-v2-2026.08.08", "osworld_2.v2026_08_08"),
        ("osworld-v2.1", "osworld_2.v2_1"),
    ],
)
def test_catalog_and_hash_manifest_are_the_same_pinned_release(
    monkeypatch, tmp_path, release_name, env_id
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSWORLD_WORKER_CONNECTION", raising=False)
    monkeypatch.delenv("CUA_LITE_REGISTRATION_MODULES", raising=False)
    release = load_release(release_name)
    task_ids = gym.registry.task_ids(env_id)["eval"]
    assert len(task_ids) == 108
    assert len(set(task_ids)) == 108
    assert task_ids == list(release.tasks)
    assert gym.registry.is_known_env_id(env_id)
    assert env_id == "osworld_2" or env_id in gym.registry.sub_env_ids("osworld_2")
    reasons = []
    for task_id in task_ids:
        env = OSWorldWorkerEnv(task_id=task_id, release=release_name)
        metadata = gym.registry.task_metadata(env_id, task_id)
        assert env._admin is None
        assert metadata.others["benchmark_release"] == release_name
        assert metadata.others["release_manifest_sha256"] == release.manifest_sha256
        assert metadata.others["task_sets"] == [
            name for name, tasks in release.task_sets.items() if task_id in tasks
        ]
        if "exclude_reason" in metadata.others:
            reasons.append(task_id)
    assert sorted(reasons) == sorted(release.exclusions)
    digest = hashlib.sha256((release.path / "task_hashes.json").read_bytes()).hexdigest()
    assert "sha256:" + digest == release.manifest["task_hash_manifest"]["sha256"]


def test_offline_task_set_reuses_release_tasks_without_filtering_protocols():
    release = replace(load_release(), exclusions={"007": "Test protocol exclusion"})
    assert release.select(task_set="full") == list(release.tasks)
    offline = release.select(task_set="offline")
    assert len(offline) == 82
    assert set(offline) < set(release.tasks)
    assert "054" in offline
    assert "007" in offline and "007" in release.exclusions


def test_all_versions_share_catalog_and_preserve_their_distinct_sources():
    assert release_names() == (
        "osworld-v2-2026.06.24", "osworld-v2-2026.08.08", "osworld-v2.1",
    )
    june = load_release("osworld-v2-2026.06.24")
    august = load_release("osworld-v2-2026.08.08")
    current = load_release()
    assert june.env_id == "osworld_2"
    assert len(june.select(task_set="full")) == 108
    assert june.task_revision == "v2026.06.24"
    assert august.env_id == "osworld_2.v2026_08_08"
    assert august.task_revision == "v2026.08.08"
    assert len(august.select(task_set="full")) == 108
    assert set(august.task_sets) == {"full"}
    # An identical VM archive does not establish task/runtime compatibility.
    assert june.manifest["provider_images"] == august.manifest["provider_images"]
    assert june.manifest["task_hash_manifest"]["sha256"] != (
        august.manifest["task_hash_manifest"]["sha256"]
    )
    # The 0808 tag publishes no offline task set; never borrow 2.1's list.
    with pytest.raises(ValueError, match="Unknown task set"):
        august.select(task_set="offline")
    with pytest.raises(ValueError, match="Unknown task set"):
        june.select(task_set="offline")
    assert current.env_id == "osworld_2.v2_1"
    assert current.task_revision == current.manifest["tasks"]["commit"]
    for release in (june, august, current):
        assert release.path.name == release.name == release.manifest["release"]


def test_june_protocols_remain_excluded_until_verified_on_its_own_runtime():
    june = load_release("osworld-v2-2026.06.24")
    assert set(june.exclusions) == {"007", "023", "024", "026", "034", "069", "095", "098"}
    assert all("awaits live protocol verification" in reason for reason in june.exclusions.values())
    assert "054" not in june.exclusions
    assert load_release("osworld-v2-2026.08.08").exclusions == {}
    assert load_release("osworld-v2.1").exclusions == {}


def test_native_server_resolves_the_registered_release_config():
    from lite.gym.remote.server import _env_config_dir_for_env_id
    from lite.gym.utils import config as env_config

    for name, prefix, timeout in (
        ("osworld-v2-2026.06.24", "OSWORLD_2", 2100.0),
        ("osworld-v2-2026.08.08", "OSWORLD_V2026_08_08", 2100.0),
        ("osworld-v2.1", "OSWORLD_V2_1", 2100.0),
    ):
        release = load_release(name)
        gym.registry.task_ids(release.env_id)
        directory = _env_config_dir_for_env_id(release.env_id)
        assert directory == release.path
        cfg = env_config.load(str(directory))
        assert cfg.env_var_prefix == prefix
        assert cfg.make_kwargs["reset_timeout"] == timeout
