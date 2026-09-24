"""A failed reset remains the primary diagnostic when Worker cleanup also fails."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from lite.gym.envs.osworld_2 import smoke
from lite.gym.envs.osworld_2.catalog import release_names
from lite.gym.errors import TrueInfraFailure


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_replace_reset_error(monkeypatch, tmp_path):
    original = TrueInfraFailure("WEBSITE_HOST_SUFFIX must be set")
    env = SimpleNamespace(
        reset=AsyncMock(side_effect=original),
        close=AsyncMock(side_effect=TrueInfraFailure("Worker is broken")),
    )
    monkeypatch.setattr(smoke.gym, "make", lambda *args, **kwargs: env)
    args = SimpleNamespace(
        release=smoke.DEFAULT_RELEASE,
        task="021",
        connection="private.json",
        experiment="test",
        results_root=tmp_path,
    )
    with pytest.raises(TrueInfraFailure) as caught:
        await smoke.run(args)
    assert caught.value is original
    assert caught.value.__notes__ == ["Worker cleanup also failed: Worker is broken"]
    env.reset.assert_awaited_once()
    env.close.assert_awaited_once()
    (record,) = tmp_path.glob("test/task_021/*/run.json")
    saved = json.loads(record.read_text())
    assert saved["status"] == "failed" and saved["closed"] is False
    assert saved["error"] == "WEBSITE_HOST_SUFFIX must be set"
    assert saved["cleanup_error"] == "Worker is broken"
    assert "score" not in saved
    assert saved["smoke"]["transport_passed"] is False


@pytest.mark.asyncio
async def test_construction_error_is_recorded_without_a_stale_running_episode(
    monkeypatch, tmp_path
):
    def fail(*args, **kwargs):
        raise ValueError("Unknown environment")

    monkeypatch.setattr(smoke.gym, "make", fail)
    args = SimpleNamespace(
        release=smoke.DEFAULT_RELEASE,
        task="054",
        connection="private.json",
        experiment="make-failure",
        results_root=tmp_path,
    )
    with pytest.raises(ValueError, match="Unknown environment"):
        await smoke.run(args)
    (record,) = tmp_path.glob("make-failure/task_054/*/run.json")
    saved = json.loads(record.read_text())
    assert saved["status"] == "failed" and saved["closed"] is False


@pytest.mark.parametrize("cleanup_error", [None, "Worker close failed"])
@pytest.mark.parametrize("release_name", release_names())
async def test_smoke_keeps_evaluation_and_transport_report_in_run_record(
    monkeypatch, tmp_path, cleanup_error, release_name
):
    image = io.BytesIO()
    Image.new("RGB", (2, 3)).save(image, format="PNG")
    screenshot = image.getvalue()
    gui_result = SimpleNamespace(
        terminated=False,
        truncated=False,
        results=[SimpleNamespace(error=None, images=[screenshot])],
    )
    env = SimpleNamespace(
        reset=AsyncMock(return_value=SimpleNamespace(image=screenshot)),
        step=AsyncMock(
            side_effect=[gui_result, gui_result, SimpleNamespace(terminated=True, reward=0.0)]
        ),
        close=AsyncMock(side_effect=RuntimeError(cleanup_error) if cleanup_error else None),
    )
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    make = Mock(return_value=env)
    monkeypatch.setattr(smoke.gym, "make", make)
    connection = tmp_path / "connection.json"
    connection.write_text(
        json.dumps({"endpoint": "http://unused.test", "token": "test-only", "worker_id": "fake"})
    )
    args = SimpleNamespace(
        release=release_name,
        task="054",
        connection=str(connection),
        experiment="smoke",
        results_root=tmp_path,
    )

    def wait_closed(run_id, worker_id, release):
        record = tmp_path / "smoke" / "task_054" / run_id / "run.json"
        saved = json.loads(record.read_text())
        assert saved["status"] == "running" and saved["closed"] is False
        assert saved["score"] == 0.0 and saved["smoke"]["transport_passed"] is True
        assert worker_id == "fake" and release == release_name

    worker = SimpleNamespace(wait_closed=Mock(side_effect=wait_closed))
    monkeypatch.setattr(smoke, "WorkerClient", lambda *args: worker)
    if cleanup_error:
        with pytest.raises(RuntimeError, match=cleanup_error):
            await smoke.run(args)
        worker.wait_closed.assert_not_called()
    else:
        await smoke.run(args)
        worker.wait_closed.assert_called_once()

    env.reset.assert_awaited_once()
    assert env.step.await_count == 3
    env.close.assert_awaited_once()
    (record,) = tmp_path.glob("smoke/task_054/*/run.json")
    saved = json.loads(record.read_text())
    assert make.call_args.args == (smoke.load_release(release_name).env_id + "@054",)
    assert saved["release"] == release_name
    manifest = json.loads((tmp_path / "smoke/experiment.json").read_text())
    assert manifest["release"] == release_name
    assert saved["status"] == ("failed" if cleanup_error else "completed")
    assert saved["closed"] is (cleanup_error is None)
    assert saved["score"] == 0.0
    assert saved["smoke"]["transport_passed"] is True
    assert saved["smoke"]["task_solution_attempted"] is False
    assert saved["smoke"]["elapsed_seconds"] >= 0
    assert set(saved["smoke"]) == {"transport_passed", "task_solution_attempted", "elapsed_seconds"}
    assert {path.name for path in record.parent.iterdir()} == {"run.json"}
    summary = json.loads((tmp_path / "smoke/summary.json").read_text())
    assert summary["runs"] == [saved]
