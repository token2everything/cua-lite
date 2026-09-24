"""Chat gateway tests use fake HTTP and an in-memory environment only."""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json

import httpx
import pytest
from PIL import Image

from lite.agents.core.agent.jsonl_logger import JsonlTrajectoryLogger
from lite.agents.extensions.chat_gateway import ChatGatewayAgent, ChatGatewayError
from lite.agents.extensions.chat_gateway.prompt import DESKTOP_SYSTEM_PROMPT
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools.action_space import validate_lite_action_batch_structure
from lite.core.tools.action_space.geometry import pixel_to_norm
from lite.core.tools.calls import tool_call_id, tool_call_name
from lite.core.tools.results import LiteToolResult
from lite.gym.envs.osworld.actions import to_pyautogui
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

_MODEL = "glm-5.3-flash"
_KEY = "test-private-api-key"
_BASE = "https://gateway.test/v1"


class _Env:
    def __init__(self):
        self.metadata = LiteCUAMetadata(
            dims=("desktop", "use"), valid_actions=["click", "screenshot"]
        )
        stream = io.BytesIO()
        Image.new("RGB", (800, 600), "blue").save(stream, format="PNG")
        self.screenshot = stream.getvalue()
        self.actions = []
        self.closed = False

    async def reset(self):
        return LiteEnvObservation(image=self.screenshot, text="Complete the sample task.")

    async def step(self, actions):
        for action in actions:
            if tool_call_name(action) == "computer":
                _, error = validate_lite_action_batch_structure(
                    "computer", action["function"]["arguments"]
                )
                assert error is None, error
        self.actions.append(copy.deepcopy(actions))
        done = any(tool_call_name(action) == "response" for action in actions)
        return LiteEnvStepResult(
            terminated=done,
            reward=1.0 if done else 0.0,
            results=[
                LiteToolResult(
                    tool_call_id=tool_call_id(action), images=[self.screenshot], text="ok"
                )
                for action in actions
            ],
        )

    async def close(self):
        self.closed = True


def _response(message=None, finish_reason="stop"):
    return {
        "id": "chatcmpl_original",
        "model": _MODEL,
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": message or {"role": "assistant", "content": "Done."},
        }],
    }


def _call(call_id="call_provider_729", name="computer", arguments=None):
    if arguments is None:
        arguments = {"actions": [{"action": "click", "coordinate": [500, 400]}]}
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _agent(**kwargs):
    return ChatGatewayAgent(
        model_id=_MODEL,
        api_base=kwargs.pop("api_base", _BASE),
        api_key=_KEY,
        api_retry_max=kwargs.pop("api_retry_max", 0),
        **kwargs,
    )


@pytest.fixture
def fake_http(monkeypatch):
    original_client = httpx.AsyncClient
    clients = []

    def install(handler):
        transport = httpx.MockTransport(handler)

        def client(**kwargs):
            clients.append(dict(kwargs))
            return original_client(transport=transport, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client)
        return clients

    return install


@pytest.mark.parametrize("shared_images", [False, True])
async def test_native_loop_replays_full_history_reasoning_ids_and_tool_screenshots(
    shared_images, fake_http, tmp_path,
):
    requests = []
    first = {
        "role": "assistant", "content": "Clicking the button.",
        "reasoning_content": "  I can see the button.\nUse its center.  ",
        "tool_calls": [_call()],
    }
    second = {
        "role": "assistant", "content": None,
        "reasoning_content": "Check the new screen.",
        "tool_calls": [_call(
            call_id="call_original_screenshot",
            arguments={"actions": [{"action": "screenshot"}]},
        )],
    }
    outputs = [_response(first, "tool_calls"), _response(second, "tool_calls"), _response()]

    def handle(request):
        assert request.url == f"{_BASE}/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {_KEY}"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=outputs[len(requests) - 1])

    clients = fake_http(handle)
    env = _Env()
    params = {
        "temperature": 0.2,
        "extra_body": {"enable_thinking": True, "clear_thinking": False},
    }
    original_params = copy.deepcopy(params)
    agent = _agent(
        metadata=env.metadata, request_params=params, request_log_dir=tmp_path / "model",
        request_image_dir=str(tmp_path / "images") if shared_images else None,
        request_timeout=30,
    )
    trajectory_logger = JsonlTrajectoryLogger(
        tmp_path, model_id=_MODEL, get_tools=agent._build_tools,
    )
    sample = await agent.sample(env, max_steps=5, hooks=[trajectory_logger])

    assert env.closed and sample.terminated and not sample.truncated
    assert len(env.actions) == len(requests) == 3
    assert agent.adapter.resolution is None
    assert params == original_params
    for request in requests:
        assert request["model"] == _MODEL
        assert request["messages"][0] == {
            "role": "system", "content": [{"type": "text", "text": DESKTOP_SYSTEM_PROMPT}],
        }
        assert sum(m["role"] == "system" for m in request["messages"]) == 1
        assert request["clear_thinking"] is False and request["enable_thinking"] is True
        assert "extra_body" not in request
        assert "truncation" not in request
        assert [tool["function"]["name"] for tool in request["tools"]] == ["computer"]
    schema = json.dumps(requests[0]["tools"])
    assert "1000" in schema and "screenshot" in schema
    assert "cursor_position" not in schema
    assert [client["trust_env"] for client in clients] == [False, False, False]
    assert all(
        client["follow_redirects"] is False and client["timeout"] == 30 for client in clients
    )
    assert [part["role"] for part in requests[1]["messages"]] == [
        "system", "user", "assistant", "tool", "user",
    ]
    replayed = requests[1]["messages"][2]
    assert replayed["reasoning_content"] == first["reasoning_content"]
    assert replayed["tool_calls"][0]["id"] == "call_provider_729"
    assert json.loads(replayed["tool_calls"][0]["function"]["arguments"]) == json.loads(
        first["tool_calls"][0]["function"]["arguments"]
    )
    assert requests[1]["messages"][3]["tool_call_id"] == "call_provider_729"
    assert "call_provider_729" in requests[1]["messages"][4]["content"][0]["text"]
    assert requests[2]["messages"][:5] == requests[1]["messages"]
    assert sample.steps[2].image_indices == (0, 1, 2)
    assert env.actions[0][0]["id"] == "call_provider_729"
    assert env.actions[0][0]["function"]["arguments"]["actions"][0] == {
        "action": "click", "coordinate": [500, 400],
    }
    assert [m["role"] for m in sample.lite_sample.messages] == [
        "system", "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]
    assert sample.lite_sample.messages[0] == requests[0]["messages"][0]
    exported = (tmp_path / "agent_trajectory.jsonl").read_text()
    assert len(exported.splitlines()) == 1
    row = json.loads(exported)
    assert "tools" not in row
    assert row["metadata"]["tools"] == requests[0]["tools"] == requests[-1]["tools"]
    assert row["messages"][0] == requests[0]["messages"][0]
    assert row["messages"][2]["reasoning_content"] == first["reasoning_content"]
    assert row["messages"][3]["role"] == "tool"
    raw = json.loads((tmp_path / "model" / "00001.response.json").read_text())
    assert raw == outputs[0]
    logged_request = json.loads((tmp_path / "model" / "00003.request.json").read_text())
    image_url = next(
        part["image_url"]["url"] for part in logged_request["messages"][1]["content"]
        if part["type"] == "image_url"
    )
    image_path = tmp_path / "model" / image_url
    assert image_path.read_bytes() == env.screenshot
    model_image = next(
        part["image_url"]["url"] for part in requests[0]["messages"][1]["content"]
        if part["type"] == "image_url"
    )
    assert base64.b64decode(model_image.split(";base64,", 1)[1]) == env.screenshot
    assert not list((tmp_path / "model").glob("*.attempt-*.response.json"))
    if shared_images:
        assert agent.request_image_dir == tmp_path / "images"
        assert image_url.startswith("../images/")
        assert not (tmp_path / "model" / "images").exists()
        viewer_ref = next(
            part["image"]["url"] for part in row["messages"][1]["content"]
            if part["type"] == "image"
        )
        assert image_path.resolve() == (tmp_path / viewer_ref).resolve()
    else:
        assert image_url.startswith("images/")
        assert len(list((tmp_path / "model" / "images").glob("*.png"))) == 1
    assert len(list((tmp_path / "images").glob("*.png"))) == 1


async def test_parallel_tool_results_precede_associated_user_images(fake_http):
    requests = []
    message = {
        "role": "assistant", "content": None,
        "tool_calls": [_call("first"), _call("second")],
    }

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=(
            _response(message, "tool_calls") if len(requests) == 1 else _response()
        ))

    fake_http(handle)
    sample = await _agent().sample(_Env(), max_steps=3)
    messages = requests[1]["messages"]
    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "tool", "tool", "user",
    ]
    assert [m["tool_call_id"] for m in messages[3:5]] == ["first", "second"]
    assert [p["type"] for p in messages[5]["content"]] == [
        "text", "image_url", "text", "image_url",
    ]
    assert sample.steps[1].image_indices == (0, 1, 2)


@pytest.mark.parametrize("history_turns,expected_images", [
    (4, (2, 3, 4, 5, 6)),
    (3, (3, 4, 5, 6)),
    (1, (5, 6)),
    (0, (6,)),
    (None, (0, 1, 2, 3, 4, 5, 6)),
])
@pytest.mark.parametrize("system_content", [
    None,
    "Keep the original system instructions.",
    [{"type": "text", "text": "Keep the original system instructions."}],
])
def test_screenshot_window_keeps_all_text_and_tool_pairs_without_mutating_history(
    history_turns, expected_images, system_content, monkeypatch,
):
    import lite.agents.extensions.chat_gateway.agent as gateway

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "完成图片任务 🖼"}, {"type": "image", "index": 0},
        ]},
    ]
    if system_content is not None:
        messages.insert(0, {"role": "system", "content": system_content})
    for index in range(6):
        call = _call(f"call_{index}")
        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        messages.extend([
            {"role": "assistant", "content": [{"type": "text", "text": f"decision {index}"}],
             "reasoning_content": f"reasoning {index}",
             "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": [
                {"type": "text", "text": f"observation {index + 1}"},
                {"type": "image", "index": index + 1},
            ]},
        ])
        if index == 0:
            messages.append({"role": "user", "content": "Keep the user's early correction."})
    original = copy.deepcopy(messages)
    images = [Image.new("RGB", (4, 3), (index, 0, 0)) for index in range(7)]
    sample = LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use")), messages=messages,
                        images=images)
    encoded = []
    encoder = gateway.encode_image_b64

    def encode(image):
        encoded.append(image.getpixel((0, 0))[0])
        return encoder(image)

    monkeypatch.setattr(gateway, "encode_image_b64", encode)
    request, indices = _agent(history_turns=history_turns)._build_request_kwargs(sample, images)
    assert indices == expected_images
    assert encoded == list(expected_images)  # Excluded images were never encoded for HTTP.
    assert sample.messages == original and len(sample.images) == 7
    wire = request["messages"]
    expected_system = (
        DESKTOP_SYSTEM_PROMPT
        if system_content is None
        else "Keep the original system instructions."
    )
    assert wire[0] == {"role": "system", "content": [{
        "type": "text", "text": expected_system,
    }]}
    assert sum(message["role"] == "system" for message in wire) == 1
    assert sum("完成图片任务" in json.dumps(m, ensure_ascii=False) for m in wire) == 1
    seen = []
    for message in wire:
        if message["role"] == "assistant":
            seen.extend(call["id"] for call in message.get("tool_calls", []))
        elif message["role"] == "tool":
            assert message["tool_call_id"] in seen
    assert seen == [f"call_{i}" for i in range(6)]
    assert [m["tool_call_id"] for m in wire if m["role"] == "tool"] == seen
    assert [m["content"] for m in wire if m["role"] == "tool"] == [
        f"observation {i + 1}" for i in range(6)
    ]
    assert [m["content"] for m in wire if m["role"] == "assistant"] == [
        f"decision {i}" for i in range(6)
    ]
    assert [m["reasoning_content"] for m in wire if m["role"] == "assistant"] == [
        f"reasoning {i}" for i in range(6)
    ]
    assert sum("Keep the user's early correction." in json.dumps(m) for m in wire) == 1


async def test_default_window_keeps_entire_single_row_trajectory(fake_http, tmp_path):
    requests = []

    def handle(request):
        assert request.content.isascii()
        body = json.loads(request.content)
        requests.append(body)
        index = len(requests) - 1
        if index == 6:
            return httpx.Response(200, json=_response())
        return httpx.Response(200, json=_response({
            "role": "assistant", "content": None, "reasoning_content": f"观察第{index}轮 🖼",
            "tool_calls": [_call(f"call_{index}")],
        }, "tool_calls"))

    fake_http(handle)
    agent = _agent(request_log_dir=tmp_path / "model")
    sample = await agent.sample(
        _Env(), max_steps=8, hooks=[JsonlTrajectoryLogger(tmp_path, model_id=_MODEL)],
    )
    assert agent.history_turns == 4
    assert [len(step.image_indices) for step in sample.steps] == [1, 2, 3, 4, 5, 5, 5]
    assert sample.steps[-1].image_indices == (2, 3, 4, 5, 6)
    assert len(sample.lite_sample.images) == 7
    for request in requests:
        systems = [m for m in request["messages"] if m["role"] == "system"]
        assert systems == [{
            "role": "system", "content": [{"type": "text", "text": DESKTOP_SYSTEM_PROMPT}],
        }]
    last = requests[-1]["messages"]
    assert [m["reasoning_content"] for m in last if m["role"] == "assistant"] == [
        f"观察第{i}轮 🖼" for i in range(6)
    ]
    exported = (tmp_path / "agent_trajectory.jsonl").read_text()
    assert len(exported.splitlines()) == 1
    row = json.loads(exported)
    assert len([m for m in row["messages"] if m["role"] == "assistant"]) == 7
    assert len(list((tmp_path / "images").glob("*.png"))) == 1
    assert sum(
        part["type"] == "image"
        for message in row["messages"] for part in message.get("content", [])
    ) == 7
    assert row["messages"][0] == requests[0]["messages"][0]
    assert row["messages"][2]["tool_calls"][0]["id"] == "call_0"
    assert row["messages"][2]["reasoning_content"] == "观察第0轮 🖼"


@pytest.mark.parametrize("history_turns,expected_images", [
    (4, (2, 3, 4, 5, 6)),
    (1, (5, 6)),
    (0, (6,)),
])
async def test_window_boundary_keeps_all_parallel_call_results(
    fake_http, history_turns, expected_images,
):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        index = len(requests) - 1
        if index == 3:
            return httpx.Response(200, json=_response())
        return httpx.Response(200, json=_response({
            "role": "assistant", "content": None,
            "tool_calls": [_call(f"call_{index}_a"), _call(f"call_{index}_b")],
        }, "tool_calls"))

    fake_http(handle)
    sample = await _agent(history_turns=history_turns).sample(_Env(), max_steps=5)
    last = requests[-1]["messages"]
    assert [m["tool_call_id"] for m in last if m["role"] == "tool"] == [
        "call_0_a", "call_0_b", "call_1_a", "call_1_b", "call_2_a", "call_2_b",
    ]
    calls = [m["tool_calls"] for m in last if m["role"] == "assistant"]
    assert [[call["id"] for call in turn] for turn in calls] == [
        [f"call_{i}_a", f"call_{i}_b"] for i in range(3)
    ]
    assert [m["role"] for m in last if m["role"] != "user"] == [
        "system", "assistant", "tool", "tool",
        "assistant", "tool", "tool", "assistant", "tool", "tool",
    ]
    assert sample.steps[-1].image_indices == expected_images
    assert sum(
        part["type"] == "image_url"
        for message in last
        if isinstance(message.get("content"), list)
        for part in message["content"]
    ) == len(expected_images)
    assert len(sample.lite_sample.images) == 7


@pytest.mark.parametrize("history_turns", [-1, True, 1.5])
def test_rejects_invalid_history_window(history_turns):
    with pytest.raises(ValueError, match="history_turns"):
        _agent(history_turns=history_turns)


def test_sample_metadata_owns_extra_tools_and_action_filter():
    extra = {
        "type": "function",
        "function": {
            "name": "lookup", "description": "Look up a resource.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    metadata = LiteCUAMetadata(
        dims=("desktop", "use"), valid_actions=[], extra_tool_schemas=[extra]
    )
    sample = LiteSample(metadata=metadata, messages=[{"role": "user", "content": "Task."}])
    agent = _agent()
    body, image_indices = agent._build_request_kwargs(sample, [])
    assert body["tools"] == [extra]
    assert image_indices == ()
    metadata.extra_tool_schemas = []
    body, _ = agent._build_request_kwargs(sample, [])
    assert "tools" not in body


@pytest.mark.parametrize("api_base,expected", [
    ("https://gateway.test", "https://gateway.test/v1"),
    ("https://gateway.test/chat/completions", "https://gateway.test"),
    ("https://gateway.test/custom/chat/completions/", "https://gateway.test/custom"),
])
def test_explicit_endpoint_normalization(api_base, expected):
    assert _agent(api_base=api_base).api_base == expected


@pytest.mark.parametrize("params", [
    {"model": "fake-gpt-id"},
    {"extra_body": {"messages": []}},
    {"extra_body": {"tools": []}},
    {"extra_body": {"api_key": "other-secret"}},
    {"extra_body": []},
    {"stream": True},
    {"n": 2},
])
def test_rejects_configuration_that_would_replace_owned_protocol(params):
    with pytest.raises(ValueError):
        _agent(request_params=params)


async def test_explicit_false_and_null_extra_body_fields_survive(fake_http):
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    fake_http(handle)
    agent = _agent(request_params={
        "enable_thinking": True,
        "extra_body": {
            "enable_thinking": False, "clear_thinking": False, "reasoning_effort": None,
        },
    })
    await agent.sample(_Env())
    assert seen[0]["enable_thinking"] is False
    assert seen[0]["clear_thinking"] is False
    assert seen[0]["reasoning_effort"] is None


@pytest.mark.parametrize("failure", ["http", "timeout"])
async def test_retries_only_model_transport_and_never_reexecutes_actions(
    failure, fake_http, tmp_path,
):
    requests = []

    def handle(request):
        requests.append(request.content)
        if len(requests) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout(f"secret: {_KEY} {_BASE}")
            return httpx.Response(503, json={"error": f"private {_KEY} at {_BASE}"})
        return httpx.Response(200, json=_response())

    fake_http(handle)
    env = _Env()
    agent = _agent(
        request_log_dir=tmp_path, api_retry_max=1,
        api_retry_base_delay=0, api_retry_max_delay=0,
    )
    sample = await agent.sample(env)
    assert env.closed and sample.terminated
    assert len(requests) == 2 and requests[0] == requests[1]
    assert len(env.actions) == 1
    assert (tmp_path / "00001.attempt-01.error.json").is_file()
    assert (tmp_path / "00001.attempt-01.response.json").exists() is (failure == "http")
    assert not (tmp_path / "00001.attempt-02.response.json").exists()
    assert json.loads((tmp_path / "00001.response.json").read_text()) == _response()
    archived = "\n".join(p.read_text() for p in tmp_path.glob("*.json"))
    assert _KEY not in archived and _BASE not in archived
    assert len(list(tmp_path.glob("*.request.json"))) == 1


@pytest.mark.parametrize("status", [301, 400, 401])
async def test_http_errors_are_not_retried_or_leaked(status, fake_http, tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, json={
            "error": f"private {_KEY} at {_BASE}",
            "authorization": f"Bearer {_KEY}", "access_token": "other-secret",
        }, headers={"Location": "https://elsewhere.test"})

    fake_http(handle)
    env = _Env()
    agent = _agent(request_log_dir=tmp_path, api_retry_max=3)
    with pytest.raises(ChatGatewayError) as failure:
        await agent.sample(env)
    assert str(failure.value) == f"Chat gateway returned HTTP {status}"
    assert env.closed and env.actions == [] and len(requests) == 1
    archived = "\n".join(p.read_text() for p in tmp_path.glob("*.json"))
    assert _KEY not in archived and _BASE not in archived and "other-secret" not in archived
    assert _KEY not in repr(agent) and _BASE not in repr(agent)


@pytest.mark.parametrize("message,finish_reason", [
    ({"role": "assistant", "content": "Done", "tool_calls": {}}, "stop"),
    ({"role": "assistant", "content": "Done", "tool_calls": [
        {"type": "function", "function": {"name": "computer", "arguments": "{}"}},
    ]}, "tool_calls"),
    ({"role": "assistant", "tool_calls": [
        _call(), _call(),
    ]}, "tool_calls"),
    ({"role": "assistant", "content": None, "reasoning_content": "Thinking only."}, "stop"),
    ({"role": "assistant", "content": "Done", "tool_calls": []}, "tool_calls"),
    ({"role": "assistant", "content": "Done"}, None),
    ({"role": "assistant", "content": "Done"}, "content_filter"),
    ({"role": "assistant", "content": "Done", "reasoning_content": []}, "stop"),
])
async def test_malformed_provider_output_never_becomes_successful_final(
    message, finish_reason, fake_http, tmp_path,
):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=_response(message, finish_reason))

    fake_http(handle)
    env = _Env()
    with pytest.raises(ChatGatewayError):
        await _agent(request_log_dir=tmp_path, api_retry_max=2).sample(env)
    assert len(requests) == 1 and env.actions == [] and env.closed
    assert json.loads((tmp_path / "00001.response.json").read_text())["choices"][0][
        "message"
    ] == message


async def test_invalid_argument_json_is_sanitized_and_raw_response_preserved(fake_http, tmp_path):
    call = _call()
    call["function"]["arguments"] = f"invalid JSON containing {_KEY} {_BASE}"
    message = {"role": "assistant", "tool_calls": [call]}
    fake_http(lambda _: httpx.Response(200, json=_response(message, "tool_calls")))
    with pytest.raises(ChatGatewayError, match="invalid function arguments") as failure:
        await _agent(request_log_dir=tmp_path).sample(_Env())
    assert _KEY not in str(failure.value) and _BASE not in str(failure.value)
    raw = (tmp_path / "00001.response.json").read_text()
    assert "invalid JSON containing" in raw and _KEY not in raw and _BASE not in raw


@pytest.mark.parametrize("failure", ["http", "cancelled"])
async def test_first_request_failure_keeps_system_in_initial_trajectory(
    failure, fake_http, tmp_path,
):
    requests = []
    initial_rows = []

    def handle(request):
        requests.append(json.loads(request.content))
        initial_rows.append(json.loads((tmp_path / "agent_trajectory.jsonl").read_text()))
        if failure == "cancelled":
            raise asyncio.CancelledError()
        return httpx.Response(400, json={"error": "request failed"})

    fake_http(handle)
    env = _Env()
    agent = _agent(api_retry_max=2, metadata=env.metadata)
    trajectory_logger = JsonlTrajectoryLogger(
        tmp_path, model_id=_MODEL, get_tools=agent._build_tools,
    )
    expected_error = asyncio.CancelledError if failure == "cancelled" else ChatGatewayError
    with pytest.raises(expected_error):
        await agent.sample(env, hooks=[trajectory_logger])
    assert len(requests) == 1 and env.actions == [] and env.closed
    assert initial_rows[0]["metadata"]["status"] == "running"
    assert [m["role"] for m in initial_rows[0]["messages"]] == ["system", "user"]
    assert initial_rows[0]["messages"][0] == requests[0]["messages"][0] == {
        "role": "system", "content": [{"type": "text", "text": DESKTOP_SYSTEM_PROMPT}],
    }
    row = json.loads((tmp_path / "agent_trajectory.jsonl").read_text())
    assert row["metadata"]["status"] == "failed"
    assert row["metadata"]["tools"] == initial_rows[0]["metadata"]["tools"] == requests[0]["tools"]
    assert row["messages"] == initial_rows[0]["messages"]


async def test_environment_failure_keeps_last_assistant_in_single_live_jsonl(fake_http, tmp_path):
    message = {
        "role": "assistant", "content": None,
        "reasoning_content": "The button is ready.", "tool_calls": [_call()],
    }
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=_response(message, "tool_calls"))

    class FailingEnv(_Env):
        async def step(self, actions):
            await super().step(actions)
            raise RuntimeError("environment action failed")

    fake_http(handle)
    env = FailingEnv()
    trajectory_logger = JsonlTrajectoryLogger(tmp_path, model_id=_MODEL)
    with pytest.raises(RuntimeError, match="environment action failed"):
        await _agent(api_retry_max=2).sample(env, hooks=[trajectory_logger])
    text = (tmp_path / "agent_trajectory.jsonl").read_text()
    assert len(text.splitlines()) == 1
    row = json.loads(text)
    assert [turn["role"] for turn in row["messages"]] == ["system", "user", "assistant"]
    assert row["messages"][-1]["reasoning_content"] == message["reasoning_content"]
    assert row["messages"][-1]["tool_calls"][0]["id"] == "call_provider_729"
    assert env.closed and len(env.actions) == 1 and len(calls) == 1


async def test_null_function_arguments_are_preserved_and_env_rejections_remain_visible(fake_http):
    requests = []
    call = _call(name="lookup", arguments={"optional": None})

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=(
            _response({"role": "assistant", "tool_calls": [call]}, "tool_calls")
            if len(requests) == 1 else _response()
        ))

    class RejectingEnv(_Env):
        async def step(self, actions):
            result = await super().step(actions)
            if not result.terminated:
                result.results[0].error = "tool not available"
            return result

    fake_http(handle)
    env = RejectingEnv()
    await _agent().sample(env, max_steps=3)
    assert env.actions[0][0]["function"]["arguments"] == {"optional": None}
    assert json.loads(requests[1]["messages"][2]["tool_calls"][0]["function"]["arguments"]) == {
        "optional": None,
    }
    assert "tool not available" in requests[1]["messages"][3]["content"]


@pytest.mark.parametrize("pixel,normalized", [
    ((960, 540), [500, 500]),
    ((209, 46), [109, 43]),
])
def test_prompt_coordinate_formula_matches_vm_projection(pixel, normalized):
    assert "[round(1000 * x / W), round(1000 * y / H)]" in DESKTOP_SYSTEM_PROMPT
    assert "(960, 540) in a 1920 by 1080 screenshot becomes [500, 500]" in DESKTOP_SYSTEM_PROMPT
    assert "never absolute pixel coordinates" in DESKTOP_SYSTEM_PROMPT
    assert pixel_to_norm(*pixel, 1920, 1080) == normalized
    assert to_pyautogui("click", {"coordinate": normalized}, 1920, 1080) == (
        f"pyautogui.click({pixel[0]}, {pixel[1]}, button='left')"
    )


@pytest.mark.parametrize("keys,command", [
    (["ctrl", "l"], "pyautogui.hotkey('ctrl', 'l')"),
    (["ctrl", "shift", "s"], "pyautogui.hotkey('ctrl', 'shift', 's')"),
])
def test_prompt_key_examples_match_vm_projection(keys, command):
    assert json.dumps(keys) in DESKTOP_SYSTEM_PROMPT
    assert "lowercase canonical key names" in DESKTOP_SYSTEM_PROMPT
    assert "chord keys in separate array items" in DESKTOP_SYSTEM_PROMPT
    assert to_pyautogui("key", {"keys": keys}, 1920, 1080) == command
