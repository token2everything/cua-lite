"""Pinned release configuration reaches native task loading and provider settings."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest

from lite.gym.envs.osworld_2.worker import provider_image_override, worker_runtime


@pytest.fixture
def runtime_files(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    source = repo / "evaluation_examples/task_class/task_054.py"
    source.parent.mkdir(parents=True)
    content = b"# Task loading belongs to the native loader, not its class name.\n"
    source.write_bytes(content)
    hashes = tmp_path / "task_hashes.json"
    hashes.write_text(
        json.dumps(
            {
                "files": {
                    source.name: {
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                }
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(repo))
    for name in ("OSWORLD_BENCHMARK_RELEASE", "OSWORLD_FILE_BASE_URL", "NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "original-test-value")
    env = SimpleNamespace(
        provider=SimpleNamespace(container=SimpleNamespace(id="fake-vm")),
        reset=Mock(),
        user_simulator=None,
    )
    desktop = ModuleType("desktop_env.desktop_env")
    desktop.DesktopEnv = Mock(return_value=env)
    loader = ModuleType("task_loader")
    loader.load_task_from_file = Mock(return_value={"volume_size": 48})
    monkeypatch.setitem(sys.modules, "desktop_env", ModuleType("desktop_env"))
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", desktop)
    monkeypatch.setitem(sys.modules, "task_loader", loader)
    monkeypatch.setattr(
        worker_runtime.OSWorldRuntime,
        "_wait_for_desktop",
        lambda self: {"screenshot": b"test-image", "instruction": "test instruction"},
    )
    return SimpleNamespace(
        source=source,
        desktop=desktop,
        loader=loader,
        env=env,
        config={
            "repo": str(repo),
            "task_hashes": str(hashes),
            "assets": "/assets",
            "vm": "/vm",
            "output": str(tmp_path / "output"),
        },
    )


@pytest.mark.parametrize("release", ["osworld-v2-2026.08.08", "osworld-v2.1"])
def test_runtime_uses_selected_release_and_native_loader(runtime_files, release):
    files = runtime_files
    runtime = worker_runtime.OSWorldRuntime({**files.config, "release": release})
    result = runtime.reset("054")
    files.loader.load_task_from_file.assert_called_once_with(str(files.source))
    files.env.reset.assert_called_once_with(task_config=runtime.task)
    assert os.environ["OSWORLD_BENCHMARK_RELEASE"] == release
    assert os.environ["OSWORLD_FILE_BASE_URL"] == "/assets"
    assert files.desktop.DesktopEnv.call_args.kwargs["action_space"] == "pyautogui"
    assert files.desktop.DesktopEnv.call_args.kwargs["volume_size"] == 48
    assert result["phase_count"] == 1 and result["has_user_simulator"] is False
    assert result["task_current_date"] is None
    assert result["runtime"]["vm_container_id"] == "fake-vm"


@pytest.mark.parametrize("same_size", [True, False])
def test_task_content_is_verified_before_native_loading(runtime_files, same_size):
    files = runtime_files
    content = files.source.read_bytes()
    files.source.write_bytes(b"x" * len(content) if same_size else content + b"changed")
    runtime = worker_runtime.OSWorldRuntime({**files.config, "release": "osworld-v2-2026.08.08"})
    with pytest.raises(RuntimeError, match="does not match the pinned release"):
        runtime.reset("054")
    files.loader.load_task_from_file.assert_not_called()
    files.desktop.DesktopEnv.assert_not_called()


@pytest.mark.parametrize("release", ["osworld-v2-2026.08.08", "osworld-v2.1"])
def test_multiphase_reset_uses_native_first_setup_once(runtime_files, release):
    phases = [
        {"instruction": "first phase", "setup": Mock(), "evaluate": Mock(return_value=0.35)},
        {"instruction": "second phase", "setup": Mock(), "evaluate": Mock(return_value=0.3)},
    ]

    class Task(dict):
        task_current_date = "2030-01-02"

        def get_phases(self):
            runtime_files.env.reset.assert_not_called()
            return phases

    task = Task()
    runtime_files.loader.load_task_from_file.return_value = task
    runtime_files.env.reset.side_effect = lambda **kwargs: kwargs["task_config"].get(
        "setup", phases[0]["setup"]
    )()
    runtime = worker_runtime.OSWorldRuntime({**runtime_files.config, "release": release})
    result = runtime.reset("054")
    phases[0]["setup"].assert_called_once_with()
    phases[1]["setup"].assert_not_called()
    assert result["phase_index"] == 0 and result["phase_count"] == 2
    assert result["task_current_date"] == "2030-01-02"
    assert result["observation"]["instruction"] == "first phase"


@pytest.fixture(params=["osworld-v2-2026.08.08", "osworld-v2.1"])
def phase_runtime(request):
    runtime = worker_runtime.OSWorldRuntime({"release": request.param})
    runtime.task_current_date = "2030-01-02"
    runtime.env = SimpleNamespace(
        instruction="phase 1",
        _step_no=7,
        _traj_no=3,
        action_history=["pyautogui.click(1, 1)", "DONE"],
        current_use_proxy=True,
        setup_controller=object(),
        is_environment_used=False,
        user_simulator=SimpleNamespace(
            respond=Mock(return_value="A simulated answer"), reset=Mock()
        ),
        step=Mock(),
        evaluate=Mock(),
    )
    runtime.env._get_obs = Mock(
        side_effect=lambda: {"instruction": runtime.env.instruction, "screenshot": b"image"}
    )
    runtime.phases = [
        {
            "name": f"Phase {index + 1}",
            "instruction": f"phase {index + 1}",
            "setup": Mock(),
            "evaluate": Mock(return_value=weight),
            **({"gate_min_score": weight} if index < 3 else {}),
        }
        for index, weight in enumerate((0.35, 0.3, 0.1, 0.25))
    ]
    return runtime


def test_all_phases_cache_evaluation_and_advance_native_lifecycle(phase_runtime, monkeypatch):
    runtime = phase_runtime
    sleep = Mock()
    monkeypatch.setattr(worker_runtime.time, "sleep", sleep)
    simulator = runtime.env.user_simulator
    for index, phase in enumerate(runtime.phases):
        result = runtime.evaluate()
        assert runtime.evaluate() is result
        phase["evaluate"].assert_called_once_with(runtime.env)
        assert result["phase_score"] == (0.35, 0.3, 0.1, 0.25)[index]
        assert result["phase_index"] == index and result["phase_count"] == 4
        assert result["complete"] is (index == 3)
        assert result["gate_failed"] is False
        if index < 3:
            runtime.env.action_history.append("DONE")
            runtime.env._step_no = 9
            advanced = runtime.advance()
            assert advanced["phase_index"] == index + 1
            assert advanced["phase_count"] == 4
            assert advanced["task_current_date"] == "2030-01-02"
            assert advanced["observation"]["instruction"] == f"phase {index + 2}"
            assert runtime.env._step_no == 0 and runtime.env.action_history == []
            assert runtime.env._traj_no == index + 4
            assert runtime.env.is_environment_used is True
            runtime.phases[index + 1]["setup"].assert_called_once_with(
                runtime.env.setup_controller, use_proxy=True
            )
    assert result["score"] == 1.0 and len(result["phase_results"]) == 4
    assert sleep.call_args_list == [call(5)] * 3
    assert runtime.env.user_simulator is simulator
    simulator.reset.assert_not_called()
    with pytest.raises(ValueError, match="No evaluated phase"):
        runtime.advance()


@pytest.mark.parametrize("pause", [0, 2.5])
def test_phase_setup_uses_declared_pause(phase_runtime, monkeypatch, pause):
    runtime = phase_runtime
    runtime.phases[1]["pause_after_setup_seconds"] = pause
    sleep = Mock()
    monkeypatch.setattr(worker_runtime.time, "sleep", sleep)
    runtime.evaluate()
    runtime.advance()
    assert sleep.call_args_list == ([call(pause)] if pause else [])


@pytest.mark.parametrize(
    ("score", "gate"),
    [(0.35 - 1e-10, {"gate_min_score": 0.35}), (0.0, {"gate": True}), (-0.1, {"gate": True})],
)
def test_native_phase_gates_have_no_epsilon(phase_runtime, score, gate):
    runtime = phase_runtime
    phase = runtime.phases[0]
    phase.pop("gate_min_score")
    phase.update(gate)
    phase["evaluate"].return_value = score
    result = runtime.evaluate()
    assert result["phase_score"] == score
    assert result["gate_failed"] is True and result["complete"] is True
    with pytest.raises(ValueError, match="No evaluated phase"):
        runtime.advance()
    runtime.phases[1]["setup"].assert_not_called()


def test_phase_total_is_rounded_and_clamped(phase_runtime, monkeypatch):
    runtime = phase_runtime
    runtime.phases = runtime.phases[:2]
    for phase in runtime.phases:
        phase.pop("gate_min_score")
        phase["evaluate"].return_value = 0.666666
    monkeypatch.setattr(worker_runtime.time, "sleep", Mock())
    first = runtime.evaluate()
    assert first["phase_score"] == 0.666666 and first["score"] == 0.6667
    runtime.advance()
    assert runtime.evaluate()["score"] == 1.0


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_invalid_phase_score_never_becomes_a_cached_zero(phase_runtime, score):
    runtime = phase_runtime
    runtime.phases[0]["evaluate"].return_value = score
    with pytest.raises(ValueError, match="non-finite"):
        runtime.evaluate()
    assert runtime.evaluation is None and runtime.phase_results == []


def test_single_phase_preserves_native_result_and_caches_once(phase_runtime):
    runtime = phase_runtime
    runtime.phases = []
    native = {"score": 0.5, "details": {"checked": True}}
    runtime.env.evaluate.return_value = native
    result = runtime.evaluate()
    assert result["result"] is native
    assert result["score"] == result["phase_score"] == 0.5
    assert result["complete"] is True and result["phase_count"] == 1
    assert runtime.evaluate() is result
    runtime.env.evaluate.assert_called_once_with()


def test_ask_user_returns_answer_without_a_gui_action(phase_runtime):
    runtime = phase_runtime
    history = list(runtime.env.action_history)
    result = runtime.ask_user("Could you clarify?")
    assert result["question"] == "Could you clarify?"
    assert result["answer"] == result["observation"]["user_response"] == "A simulated answer"
    runtime.env.user_simulator.respond.assert_called_once_with("Could you clarify?")
    runtime.env.step.assert_not_called()
    assert runtime.env.action_history == history and runtime.env._step_no == 7
    runtime.evaluate()
    with pytest.raises(ValueError, match="phase was evaluated"):
        runtime.ask_user("Another question")
    assert runtime.env.user_simulator.respond.call_count == 1


def test_missing_or_failed_user_simulator_does_not_fake_an_answer(phase_runtime):
    runtime = phase_runtime
    runtime.env.user_simulator.respond.side_effect = RuntimeError("model unavailable")
    with pytest.raises(RuntimeError, match="model unavailable"):
        runtime.ask_user("Could you clarify?")
    runtime.env._get_obs.assert_not_called()
    runtime.env.user_simulator = None
    with pytest.raises(ValueError, match="no user simulator"):
        runtime.ask_user("Could you clarify?")
    runtime.env.step.assert_not_called()
    runtime.env._get_obs.assert_not_called()


@pytest.mark.parametrize("release", ["osworld-v2-2026.08.08", "osworld-v2.1"])
def test_provider_uses_manifest_image_and_preserves_memory_network_cleanup(
    tmp_path, monkeypatch, release
):
    packaged = Path(worker_runtime.__file__).parent.parent / "releases" / release / "release.json"
    manifest = json.loads(packaged.read_text())
    (tmp_path / "release.json").write_bytes(packaged.read_bytes())
    monkeypatch.setenv("OSWORLD_RELEASE_DIR", str(tmp_path))
    image = manifest["provider_images"]["docker"]["ubuntu"]["runtime_image"]
    provider = tmp_path / "provider.py"
    provider.write_text(
        "import os\n"
        "class Provider:\n"
        "    def __init__(self):\n"
        '        self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", '
        '"CPU_CORES": "4"}  # Modify if needed\n'
        f"        self.image = {json.dumps(image)}\n"
        "    def close(self):\n"
        "        self.container.remove()\n"
        "    def retry(self):\n"
        "        self.container.remove()\n"
    )
    provider_image_override.apply(provider)
    patched = provider.read_text()
    provider_image_override.apply(provider)
    assert provider.read_text() == patched
    namespace = {}
    exec(compile(patched, str(provider), "exec"), namespace)
    for name in ("OSWORLD_DOCKER_IMAGE", "OSWORLD_VM_RAM_SIZE", "OSWORLD_VM_NETWORK"):
        monkeypatch.delenv(name, raising=False)
    instance = namespace["Provider"]()
    assert instance.image == image
    assert instance.environment == {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}
    monkeypatch.setenv("OSWORLD_DOCKER_IMAGE", "sha256:local-runtime-image")
    monkeypatch.setenv("OSWORLD_VM_RAM_SIZE", "16G")
    monkeypatch.setenv("OSWORLD_VM_NETWORK", "user")
    instance = namespace["Provider"]()
    assert instance.image == "sha256:local-runtime-image"
    assert instance.environment["RAM_SIZE"] == "16G"
    assert instance.environment["NETWORK"] == "N"
    assert instance.environment["ARGUMENTS"] == provider_image_override.USER_NETWORK
    instance.container = Mock()
    instance.close()
    instance.retry()
    assert instance.container.remove.call_count == 2
    assert all(call.kwargs == {"v": True} for call in instance.container.remove.call_args_list)


def test_provider_rejects_a_different_source_image_without_guessing(tmp_path, monkeypatch):
    (tmp_path / "release.json").write_text(
        json.dumps({"provider_images": {"docker": {"ubuntu": {"runtime_image": "expected-image"}}}})
    )
    monkeypatch.setenv("OSWORLD_RELEASE_DIR", str(tmp_path))
    provider = tmp_path / "provider.py"
    original = 'image = "different-image"\n'
    provider.write_text(original)
    with pytest.raises(ValueError, match="expected image call"):
        provider_image_override.apply(provider)
    assert provider.read_text() == original
