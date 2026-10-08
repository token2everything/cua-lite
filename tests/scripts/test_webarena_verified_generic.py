"""The native desktop bridge uses fake GUI/browser/provider owners only."""

from __future__ import annotations

import importlib.util
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from pydantic import BaseModel

from lite.agents.core.agent.base import AdapterBasedAgent
from lite.agents.extensions.chat_gateway import ChatGatewayAgent, ChatGatewayError
from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools.calls import make_tool_call
from lite.gym.types import LiteEnvObservation

SPEC = importlib.util.spec_from_file_location(
    "webarena_verified_generic", Path(__file__).parents[2] / "scripts/webarena_verified_generic.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Answer(BaseModel):
    answer: str


class GUI:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return call


def arguments(tmp_path, **overrides):
    args = runner._parse_args(
        [
            "--task-id",
            "0",
            "--osworld-root",
            "/unused/osworld",
            "--baseline-receipt",
            str(tmp_path / "baseline.json"),
            "--output-dir",
            str(tmp_path),
            "--lock-dir",
            str(tmp_path / "locks"),
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def env_ready(tmp_path, *, owner=None, **overrides):
    env = (owner or runner.WebArenaFullDesktopEnv)(arguments(tmp_path, **overrides), tmp_path)
    env._opened = True
    env._response_model = Answer
    env._pyautogui = GUI()
    env._page = SimpleNamespace(wait_for_timeout=lambda milliseconds: None)
    stream = io.BytesIO()
    Image.new("RGB", (1920, 1080)).save(stream, format="PNG")
    env._image = stream.getvalue()
    env._capture = lambda: env._image

    def finish():
        env.evaluation = {"score": 1.0}
        env._browser_finalized = True

    env._finish_sync = finish
    return env


def computer(call_id="gui-call", **action):
    return make_tool_call("computer", {"actions": [action]}, call_id=call_id)


def test_cli_model_profile_selects_literal_id_effort_and_explicit_reserve(tmp_path):
    args = arguments(tmp_path)
    assert (args.model_id, args.reasoning_effort) == ("qwen3.8-max", "xhigh")
    assert (args.max_steps, args.final_response_reserve_steps) == (150, 7)
    assert args.repair_invalid_response and args.startup_page_gate
    args = arguments(tmp_path, model_profile="glm53flash")
    parsed = runner._parse_args(
        [
            "--task-id",
            "0",
            "--osworld-root",
            "/unused",
            "--baseline-receipt",
            "/unused/receipt",
            "--output-dir",
            "/unused/output",
            "--lock-dir",
            "/unused/locks",
            "--model-profile",
            "glm53flash",
        ]
    )
    assert (parsed.model_id, parsed.reasoning_effort) == ("glm-5.3-flash", "max")


@pytest.mark.parametrize("max_steps,reserve", [(0, 0), (150, 150), (150, -1)])
def test_cli_rejects_invalid_budgets(tmp_path, max_steps, reserve):
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--task-id",
                "0",
                "--osworld-root",
                "/unused",
                "--baseline-receipt",
                "/unused/receipt",
                "--output-dir",
                "/unused/output",
                "--lock-dir",
                "/unused/locks",
                "--max-steps",
                str(max_steps),
                "--final-response-reserve-steps",
                str(reserve),
            ]
        )


async def test_native_click_uses_full_desktop_normalized_coords_and_paired_results(tmp_path):
    env = env_ready(tmp_path)
    try:
        result = await env.step([computer(action="click", coordinate=[500, 500])])
        assert env._step_count == 1
        assert env._pyautogui.calls == [("click", (960, 540), {"button": "left"})]
        assert result.results[0].tool_call_id == "gui-call"
        assert result.results[0].images == [env._image]
        assert not result.terminated and not result.truncated
        assert [schema["function"]["name"] for schema in env.metadata.extra_tool_schemas] == [
            "response"
        ]
    finally:
        await env.close()


async def test_action_batch_costs_one_model_turn_and_preserves_results_for_multiple_calls(tmp_path):
    env = env_ready(tmp_path)
    actions = [
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "key", "keys": ["ctrl", "a"]},
                    {"action": "type", "text": "hello<world", "press_enter": True},
                ]
            },
            call_id="first",
        ),
        computer(call_id="second", action="click", coordinate=[1000, 1000]),
    ]
    try:
        result = await env.step(actions)
        assert env._step_count == 1
        assert [item.tool_call_id for item in result.results] == ["first", "second"]
        assert not result.results[0].images and result.results[1].images
        assert env._pyautogui.calls[-1][1] == (1919, 1079)
        assert ("press", ("enter",), {}) in env._pyautogui.calls
        assert ("hotkey", ("shift", ","), {}) in env._pyautogui.calls
    finally:
        await env.close()


async def test_invalid_coordinate_returns_model_visible_error_without_execution(tmp_path):
    env = env_ready(tmp_path)
    try:
        result = await env.step([computer(action="click", coordinate=[1500, 500])])
        assert not env._pyautogui.calls
        assert "normalized" in result.results[0].error
        assert result.results[0].tool_call_id == "gui-call"
        assert result.results[0].images == [env._image]
    finally:
        await env.close()


async def test_invalid_scroll_amount_cannot_create_executable_source(tmp_path):
    env = env_ready(tmp_path)
    try:
        result = await env.step([computer(action="scroll", amount="1); print('bad');")])
        assert not env._pyautogui.calls
        assert "finite number" in result.results[0].error
        assert result.results[0].tool_call_id == "gui-call"
    finally:
        await env.close()


@pytest.mark.parametrize("explicit", [False, True])
async def test_invalid_final_has_one_visible_repair_and_accepts_raw_valid_answer(
    tmp_path, explicit
):
    env = env_ready(tmp_path)
    invalid = (
        [make_tool_call("response", {"text": "not json"}, call_id="answer-bad")]
        if explicit
        else make_no_tool_call_final_actions("not json")
    )
    valid = '{"answer":"verified"}'
    try:
        result = await env.step(invalid)
        assert not result.terminated and env.response_repair_used
        feedback = result.results[0]
        assert feedback.tool_call_id == ("answer-bad" if explicit else None)
        assert "one repair attempt" in feedback.text and feedback.error
        result = await env.step(make_no_tool_call_final_actions(valid))
        assert result.terminated and env.final_content == valid
        assert env.response_parse_error is None
        assert not env._pyautogui.calls
    finally:
        await env.close()


async def test_second_invalid_final_ends_as_model_protocol_failure(tmp_path):
    env = env_ready(tmp_path)
    try:
        await env.step(make_no_tool_call_final_actions("bad"))
        result = await env.step(make_no_tool_call_final_actions("still bad"))
        assert result.terminated and env.model_protocol_error
        assert env.final_content == "still bad"
        assert env._step_count == 2
    finally:
        await env.close()


async def test_final_repair_cannot_execute_gui_action(tmp_path):
    env = env_ready(tmp_path)
    try:
        await env.step(make_no_tool_call_final_actions("bad"))
        result = await env.step([computer(action="click", coordinate=[500, 500])])
        assert result.terminated and env.model_protocol_error
        assert not env._pyautogui.calls
    finally:
        await env.close()


@pytest.mark.parametrize("answer_first", [True, False])
async def test_mixed_gui_final_turn_is_rejected_before_any_execution(tmp_path, answer_first):
    env = env_ready(tmp_path)
    response = make_tool_call("response", {"text": '{"answer":"verified"}'}, call_id="final")
    gui = computer(action="click", coordinate=[500, 500])
    try:
        result = await env.step([response, gui] if answer_first else [gui, response])
        assert not result.terminated and env.response_repair_used
        assert env._repair_pending and not env._pyautogui.calls
        assert all(item.error for item in result.results)
        result = await env.step(make_no_tool_call_final_actions('{"answer":"verified"}'))
        assert result.terminated and not env._pyautogui.calls
    finally:
        await env.close()


async def test_final_repair_rejects_unusable_response_tool_without_silent_continuation(tmp_path):
    env = env_ready(tmp_path)
    try:
        await env.step(make_no_tool_call_final_actions("bad"))
        result = await env.step([make_tool_call("response", {}, call_id="missing-text")])
        assert result.terminated and env.model_protocol_error
        assert not env._pyautogui.calls
    finally:
        await env.close()


async def test_reserve_starts_at_round144_and_blocks_two_gui_proposals(tmp_path):
    env = env_ready(tmp_path)
    env._step_count = 142
    try:
        result = await env.step([computer(action="screenshot")])
        assert not result.truncated and env._step_count == 143
        assert "7 model turns" in result.results[0].text
        result = await env.step([computer(action="click", coordinate=[500, 500])])
        assert not result.truncated and env.finalization_attempts == 1
        assert result.info["reserve_blocked"] and result.results[0].error
        result = await env.step([computer(action="click", coordinate=[500, 500])])
        assert result.truncated and env.finalization_attempts == 2
        assert not env._pyautogui.calls and env.final_content == ""
    finally:
        await env.close()


async def test_wait_honors_requested_duration_and_releases_held_keys(tmp_path):
    env = env_ready(tmp_path)
    waits = []
    env._page = SimpleNamespace(wait_for_timeout=waits.append)
    try:
        await env.step([computer(action="hold_key", keys=["ctrl"], duration=1.2341)])
        assert waits == [1235, 350]
        assert env._pyautogui.calls == [("keyDown", ("ctrl",), {}), ("keyUp", ("ctrl",), {})]
    finally:
        await env.close()


async def test_sync_phases_stay_on_one_thread_outside_native_event_loop(tmp_path):
    env = env_ready(tmp_path)
    owner_thread = threading.get_ident()
    threads = []
    env._opened = False
    env._reset_sync = lambda: (
        threads.append(threading.get_ident()) or LiteEnvObservation(None, "goal")
    )
    env._step_sync = lambda actions: threads.append(threading.get_ident())
    env._close_sync = lambda: threads.append(threading.get_ident()) or setattr(env, "_closed", True)
    await env.reset()
    await env.step([])
    await env.close()
    assert len(set(threads)) == 1 and threads[0] != owner_thread


def test_finalization_scores_raw_text_and_binds_all_input_receipts(tmp_path, monkeypatch):
    env = env_ready(tmp_path)
    env._finish_sync = runner.WebArenaFullDesktopEnv._finish_sync.__get__(env)
    env._task = object()
    env._context, env._browser = object(), object()
    env._tracing_started = True
    raw = 'Finished.\n```json\n{"answer":"verified"}\n```'
    env.final_content = raw
    finalized = []
    scored = []

    def finalize(context, browser, trace_path, *, tracing_started):
        finalized.append((context, browser, tracing_started))
        runner.desktop._atomic_bytes(trace_path, b"trace-evidence")
        runner.desktop._atomic_bytes(tmp_path / "network.har", b"har-evidence")

    def score(task, answer, trace, har):
        scored.append((task, answer, trace, har))
        return {"score": 1.0, "integrity_status": "passed", "publishable": True}

    runner.desktop._atomic_json(tmp_path / "startup_receipt.json", {"result": "passed"})
    monkeypatch.setattr(runner.desktop, "finalize_browser_artifacts", finalize)
    monkeypatch.setattr(runner.desktop, "score_artifacts", score)
    try:
        env._finish_sync()
        env._finish_sync()
        assert len(finalized) == len(scored) == 1
        assert scored[0][1] == raw
        assert (tmp_path / "agent_response_raw.txt").read_text() == raw
        assert set(env.evaluation["input_receipt"]) == {
            "agent_response_raw",
            "network_har",
            "playwright_trace",
            "startup_receipt",
        }
        assert env.evaluation["input_receipt"][
            "agent_response_raw"
        ] == runner.desktop.content_identity(raw)
        projection = json.loads((tmp_path / "agent_response.json").read_text())
        assert projection["validated_projection"] == {"answer": "verified"}
    finally:
        env._context = env._browser = None
        env._executor.shutdown()


async def test_real_native_sample_repairs_content_only_answer_with_full_history(
    tmp_path, monkeypatch
):
    env = env_ready(tmp_path)
    env._reset_sync = lambda: LiteEnvObservation(env._image, "Return JSON with answer.")
    requests = []
    outputs = ["not valid json", '{"answer":"verified"}']
    original_client = httpx.AsyncClient

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": outputs[len(requests) - 1],
                            "reasoning_content": f"recorded-reasoning-{len(requests)}",
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    agent = ChatGatewayAgent(
        model_id="qwen3.8-max",
        api_key="fake",
        api_base="https://fake.example/v1",
        metadata=env.metadata,
        request_params={
            "max_tokens": 32768,
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
        },
        history_turns=19,
        image_fold_size=10,
        api_retry_max=0,
        request_timeout=httpx.Timeout(130, read=600),
    )
    assert type(agent).sample is AdapterBasedAgent.sample
    sample = await agent.sample(env, max_steps=150)
    assert sample.terminated and len(sample.steps) == 2 and env._closed
    assert len(requests) == 2
    for request in requests:
        assert "temperature" not in request and "top_p" not in request
        assert request["reasoning_effort"] == "xhigh"
    assistant = next(
        message for message in requests[1]["messages"] if message["role"] == "assistant"
    )
    assert assistant["content"] == "not valid json"
    assert assistant["reasoning_content"] == "recorded-reasoning-1"
    assert any("one repair attempt" in str(message) for message in requests[1]["messages"])


@pytest.fixture
def synthetic_case(tmp_path, monkeypatch):
    """Run the real native loop with synthetic desktop/scoring resources."""
    from lite.gym.envs.browsergym import webarena_verified

    output = tmp_path / "case"
    args = arguments(
        tmp_path, output_dir=output, api_key="fake-key", base_url="https://fake.test/v1"
    )
    args.osworld_root = tmp_path / "cursor-source"
    cursor = args.osworld_root / "desktop_env/server/pyxcursor.py"
    cursor.parent.mkdir(parents=True)
    cursor.write_text("# synthetic cursor source\n")
    args.baseline_receipt.write_text(
        json.dumps(
            {
                "session_id": "fake-run",
                "sandbox_id": "fake-sandbox",
                "task_id": 0,
                "created_at": "2026-10-07T00:00:00Z",
                "container_digests": {"fake": "a" * 64},
            }
        )
    )
    monkeypatch.setattr(webarena_verified, "task_records", lambda: [{}] * 812)
    monkeypatch.setattr(runner.desktop, "validate_osworld_checkout", lambda path: path)
    env_class = runner.WebArenaFullDesktopEnv
    created = []

    def factory(case_args, case_output):
        env = env_ready(case_output, owner=env_class)
        env.args = case_args
        env._task = object()
        env._context = env._browser = object()
        env._tracing_started = True
        env._finish_sync = env_class._finish_sync.__get__(env)
        env._reset_sync = lambda: LiteEnvObservation(env._image, "Return one final JSON answer.")
        runner.desktop._atomic_bytes(case_output / "desktop/step_000.png", env._image)
        runner.desktop._atomic_json(case_output / "startup_receipt.json", {"result": "passed"})
        created.append(env)
        return env

    def finalize(context, browser, trace_path, *, tracing_started):
        runner.desktop._atomic_bytes(trace_path, b"synthetic-trace")
        runner.desktop._atomic_bytes(output / "network.har", b"synthetic-har")

    def score(task, raw, trace, har):
        assert raw == ""  # No final answer was submitted; do not reconstruct one.
        return {"score": 0.0, "integrity_status": "passed", "publishable": True}

    class NoDelayChatGateway(ChatGatewayAgent):
        def __post_init__(self):
            super().__post_init__()
            self.api_retry_base_delay = self.api_retry_max_delay = 0

    monkeypatch.setattr(runner, "WebArenaFullDesktopEnv", factory)
    monkeypatch.setattr(runner.desktop, "finalize_browser_artifacts", finalize)
    monkeypatch.setattr(runner.desktop, "score_artifacts", score)
    monkeypatch.setattr(runner, "ChatGatewayAgent", NoDelayChatGateway)
    return args, output, created


def native_gui_response(call_id="first"):
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "computer",
                                "arguments": json.dumps(
                                    {"actions": [{"action": "click", "coordinate": [500, 500]}]}
                                ),
                            },
                        }
                    ],
                },
            }
        ],
    }


@pytest.mark.parametrize("prior_gui", [False, True])
async def test_successful_wire_bad_arguments_are_terminal_without_case_retry(
    synthetic_case, monkeypatch, prior_gui
):
    args, output, created = synthetic_case
    requests = []
    raw = native_gui_response("malformed")
    raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "not-json"
    original_client = httpx.AsyncClient

    def handler(request):
        requests.append(request)
        body = native_gui_response() if prior_gui and len(requests) == 1 else raw
        return httpx.Response(200, json=body)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    status = await runner.run_case(args)
    assert status["status"] == "model_protocol_failed" and status["score"] == 0
    assert status["model_protocol_failure_phase"] == "provider_projection"
    assert status["model_protocol_error_type"] == "ChatGatewayProtocolError"
    assert "error" not in status
    assert len(requests) == 1 + int(prior_gui)
    env = created[0]
    assert len(env._pyautogui.calls) == int(prior_gui)
    assert env._closed and env.close_error is None
    rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == list(range(len(requests)))
    assert rows[-1]["prediction_failed"] is True
    assert rows[-1]["actions"] == rows[-1]["executed_actions"] == rows[-1]["feedback"] == []
    assert json.loads((output / rows[-1]["raw_response"]).read_text()) == raw
    assert (output / "agent_response_raw.txt").read_text() == ""
    viewer = json.loads((output / "agent_trajectory.jsonl").read_text())
    assert viewer["metadata"]["status"] == "failed"
    assert sum(message["role"] == "assistant" for message in viewer["messages"]) == int(prior_gui)


async def test_http503_stays_infra_after_bounded_api_retries_without_gui_replay(
    synthetic_case, monkeypatch
):
    args, output, created = synthetic_case
    requests = []
    original_client = httpx.AsyncClient

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json=native_gui_response())
        return httpx.Response(503, json={"error": "upstream unavailable"})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    with pytest.raises(ChatGatewayError, match="HTTP 503"):
        await runner.run_case(args)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["error"]["type"] == "ChatGatewayError"
    assert status["model_protocol_failure_phase"] is None
    assert len(requests) == 6  # One successful decision plus five transport attempts.
    assert len(created[0]._pyautogui.calls) == 1
    assert created[0]._closed and created[0].evaluation["score"] == 0
    assert len((output / "trajectory.jsonl").read_text().splitlines()) == 1


async def test_swallowed_close_error_prevents_protocol_failure_from_being_published(
    synthetic_case, monkeypatch
):
    args, output, created = synthetic_case
    raw = native_gui_response()
    raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "not-json"
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=raw)), **kwargs
        ),
    )
    monkeypatch.setattr(
        runner.desktop,
        "score_artifacts",
        lambda *args: (_ for _ in ()).throw(RuntimeError("synthetic scoring failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic scoring failure"):
        await runner.run_case(args)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["model_protocol_error"]
    assert created[0]._closed and created[0].close_error is not None
