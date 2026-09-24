"""A rejected action costs ITSELF and nothing else — and must say why.

R4 repealed the abort policy this file used to pin: a MODEL fault never reached
the screen, so the state the tail was chosen against is exactly the state still
on it. What survives is the rejection itself and its wording -- the model
duration hard-caps and the coordinate validation -- now asserted alongside the
sibling actions that DO still run.

Historical note, kept deliberately: the four ``record_batch_abort`` unit tests
below still pass because the helper survives for INFRA aborts (R1), where the
action may have half executed and the screen is unknown. They go when R1 lands.
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock

import pytest

import lite.gym.envs.osworld.main as osworld
from lite.core.tools import make_tool_call
from lite.gym.envs.waa.main import WindowsAgentArenaEnv, _load_tasks
from lite.gym.utils.feedback.errors import (
    BATCH_ABORT_PREFIX,
    BATCH_ABORT_SIBLING_MESSAGE,
    ToolErrorFeedback,
    batch_abort_message,
    record_batch_abort,
)

_PNG_B64 = base64.b64encode(b"screenshot").decode("ascii")


class _FakeContainer:
    base_url = "http://localhost:12345"
    name = "lite-env-local-osworld-task-abc"


def _osworld_env(monkeypatch, module, config):
    env = module.__new__(module)
    env._config = config
    env._max_steps = 15
    env._post_action_delay = 0.0
    env._step_count = 0
    env._pending_cf_future = None
    env._container = _FakeContainer()
    env._valid_actions = None
    env._extra_tool_schemas = module.extra_tool_schemas(None)

    calls: list[tuple[str, dict]] = []

    def fake_rpc(base, path, body=None, timeout=None):
        calls.append((path, body or {}))
        if path == "/evaluate":
            return {"reward": 1.0}
        if path == "/screenshot":
            return {"screenshot_b64": _PNG_B64}
        return {"ok": True}

    return env, calls, fake_rpc


def _make_osworld(monkeypatch):
    env, calls, fake_rpc = _osworld_env(
        monkeypatch,
        osworld.OSWorldEnv,
        osworld.OSWorldConfig(domain="libreoffice_calc", task_id="t1"),
    )
    monkeypatch.setattr(osworld, "_rpc", fake_rpc)
    return env, calls


def _batch(actions, call_id):
    return make_tool_call("computer", {"actions": actions}, call_id=call_id)


def _step_cmds(calls):
    return [body.get("cmd") for path, body in calls if path == "/step"]


# --------------------------------------------------------------------------
# The helper itself
# --------------------------------------------------------------------------


def test_record_batch_abort_is_a_noop_when_nothing_was_dropped():
    errors: dict[str, ToolErrorFeedback] = {}

    record_batch_abort(errors, "call_a", [])

    assert errors == {}


def test_record_batch_abort_reports_the_count_to_the_failing_call():
    errors: dict[str, ToolErrorFeedback] = {}

    record_batch_abort(errors, "call_a", [({}, "call_a"), ({}, "call_a")])

    assert errors["call_a"].message == batch_abort_message(2)
    assert "2 later actions were not executed" in errors["call_a"].message


def test_record_batch_abort_uses_singular_wording_for_one_dropped_action():
    errors: dict[str, ToolErrorFeedback] = {}

    record_batch_abort(errors, "call_a", [({}, "call_a")])

    assert "1 later action was not executed" in errors["call_a"].message


def test_record_batch_abort_warns_sibling_calls_whose_actions_never_ran():
    errors: dict[str, ToolErrorFeedback] = {}

    record_batch_abort(
        errors,
        "call_a",
        [({}, "call_b"), ({}, "call_b"), ({}, "call_c"), ({}, None)],
    )

    assert errors["call_b"].message == BATCH_ABORT_SIBLING_MESSAGE
    assert errors["call_c"].message == BATCH_ABORT_SIBLING_MESSAGE
    assert BATCH_ABORT_PREFIX in errors["call_a"].message


# --------------------------------------------------------------------------
# R7 — capped duration rejects, and the drop is announced
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_osworld_capped_hold_key_duration_refuses_only_that_action(monkeypatch):
    env, calls = _make_osworld(monkeypatch)

    result = await env.step([
        _batch(
            [
                {"action": "hold_key", "keys": ["shift"], "duration": 10},
                {"action": "click", "coordinate": [500, 500]},
                {"action": "type", "text": "hello"},
            ],
            call_id="call_batch",
        )
    ])

    # R4: the capped action is refused, and its siblings still run.
    assert _step_cmds(calls), "R4: the tail must still execute"
    error = result.results[0].error
    assert "hold_key.duration must be <= 5" in error


@pytest.mark.asyncio
async def test_osworld_capped_wait_duration_refuses_only_that_action(monkeypatch):
    env, calls = _make_osworld(monkeypatch)

    result = await env.step([
        _batch(
            [
                {"action": "wait", "duration": 60},
                {"action": "click", "coordinate": [500, 500]},
            ],
            call_id="call_batch",
        )
    ])

    # R4: the refused action costs itself; its siblings still run.
    assert _step_cmds(calls)
    error = result.results[0].error
    assert "wait.duration must be <= 30" in error


@pytest.mark.asyncio
async def test_osworld_fault_leaves_its_sibling_call_alone(monkeypatch):
    env, calls = _make_osworld(monkeypatch)

    result = await env.step([
        _batch([{"action": "wait", "duration": 60}], call_id="call_first"),
        _batch([{"action": "click", "coordinate": [500, 500]}], call_id="call_second"),
    ])

    # R4: the refused action costs itself; its siblings still run.
    assert _step_cmds(calls)
    by_id = {r.tool_call_id: r for r in result.results}
    assert "wait.duration must be <= 30" in by_id["call_first"].error
    # R4: the second CALL is a sibling too. Its action ran, so it returns clean
    # -- there is no dropped tail to warn about any more.
    assert by_id["call_second"].error is None


@pytest.mark.asyncio
async def test_waa_capped_hold_key_duration_refuses_only_that_action():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=[],
    )
    env._last_screenshot = b"last-shot"
    env._bridge = AsyncMock()
    env._bridge.post.return_value = {
        "ok": True,
        "screenshot_b64": base64.b64encode(b"fresh-shot").decode(),
    }

    result = await env.step([
        _batch(
            [
                {"action": "hold_key", "keys": ["shift"], "duration": 10},
                {"action": "click", "coordinate": [500, 500]},
            ],
            call_id="call_batch",
        )
    ])

    # R4 + R3: the refused action earns a frame (a fresh /screenshot) and its
    # siblings still reach the guest.
    posted = [c.args[0] for c in env._bridge.post.await_args_list]
    assert "/screenshot" in posted
    error = result.results[0].error
    assert "hold_key.duration must be <= 5" in error


# --------------------------------------------------------------------------
# R8 — missing coordinate rejects (no fabricated centre click), drop announced
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_osworld_missing_coordinate_refuses_only_that_action(monkeypatch):
    env, calls = _make_osworld(monkeypatch)

    result = await env.step([
        _batch(
            [
                {"action": "click", "button": "left"},
                {"action": "type", "text": "hello"},
            ],
            call_id="call_batch",
        )
    ])

    # Baseline emitted pyautogui.click(960, 540) here — a fabricated action.
    # R4: the refused action costs itself; its siblings still run.
    assert _step_cmds(calls)
    error = result.results[0].error
    assert "invalid arguments for click" in error


@pytest.mark.asyncio
async def test_waa_short_coordinate_refuses_only_that_action():
    task = _load_tasks()[0]
    env = WindowsAgentArenaEnv(
        task_config=task["config"],
        domain=task["domain"],
        variant=task["variant"],
        extra_tools=[],
    )
    env._last_screenshot = b"last-shot"
    env._bridge = AsyncMock()
    env._bridge.post.return_value = {
        "ok": True,
        "screenshot_b64": base64.b64encode(b"fresh-shot").decode(),
    }

    # ``[500]`` passes ingress: the coordinate schema declares no minItems.
    result = await env.step([
        _batch(
            [
                {"action": "click", "coordinate": [500]},
                {"action": "type", "text": "hello"},
            ],
            call_id="call_batch",
        )
    ])

    # R4 + R3: the refused action earns a frame (a fresh /screenshot) and its
    # siblings still reach the guest.
    posted = [c.args[0] for c in env._bridge.post.await_args_list]
    assert "/screenshot" in posted
    error = result.results[0].error
    assert "invalid arguments for click" in error
