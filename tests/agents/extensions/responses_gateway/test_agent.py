"""Stateless Responses gateway transport tests; every HTTP request is mocked."""

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
from lite.agents.extensions.responses_gateway.agent import (
    ResponsesGatewayAgent,
    ResponsesGatewayError,
)
from lite.core import LiteCUAMetadata
from lite.core.tools.calls import tool_call_id, tool_call_name
from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

_MODEL = "glm-5.3-responses"
_KEY = "test-private-api-key"
_BASE = "https://gateway.test/v1"


class _Env:
    def __init__(self):
        self.metadata = LiteCUAMetadata(dims=("desktop", "use"))
        stream = io.BytesIO()
        Image.new("RGB", (800, 600), "blue").save(stream, format="PNG")
        self.screenshot = stream.getvalue()
        self.actions = []
        self.closed = False

    async def reset(self):
        return LiteEnvObservation(image=self.screenshot, text="Complete the sample task.")

    async def step(self, actions):
        self.actions.append(actions)
        done = any(tool_call_name(action) in {"response", "terminate"} for action in actions)
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


def _response(output=None, **fields):
    if output is None:
        output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Done."}],
            }
        ]
    return {
        "id": "resp_original_id",
        "status": "completed",
        "output": output,
        "usage": {},
        **fields,
    }


def _agent(**kwargs):
    return ResponsesGatewayAgent(
        model_id=_MODEL,
        api_base=_BASE,
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


@pytest.mark.parametrize("singular", [True, False])
@pytest.mark.parametrize("shared_images", [False, True])
async def test_rollout_keeps_real_model_reasoning_raw_history_and_computer_actions(
    singular,
    shared_images,
    fake_http,
    tmp_path,
):
    first_output = [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Click the button."}],
            "encrypted_content": "opaque-reasoning",
        },
        {
            "type": "computer_call",
            "call_id": "call_original",
            "action": {
                "type": "click",
                "x": 400,
                "y": 300,
                "button": "left",
            },
        },
    ]
    if not singular:
        first_output[1]["actions"] = [first_output[1].pop("action"), {"type": "screenshot"}]
    expected_raw = copy.deepcopy(first_output)
    posts, gets = [], []

    def handler(request):
        if request.method == "GET":
            gets.append(request)
            return httpx.Response(404, json={"error": "Response state is not retained"})
        assert request.headers["Authorization"] == f"Bearer {_KEY}"
        assert request.content.isascii()
        posts.append(json.loads(request.content))
        return httpx.Response(200, json=_response(first_output if len(posts) == 1 else None))

    clients = fake_http(handler)
    params = {
        "reasoning": {"context": "all_turns", "effort": "max", "provider_option": "保留 🖼"},
        "max_output_tokens": 512,
        "include": ["message.output_text.logprobs"],
    }
    log_dir = tmp_path / "model"
    agent = _agent(
        request_params=params,
        request_log_dir=log_dir,
        request_image_dir=str(tmp_path / "images") if shared_images else None,
        api_kwargs={"chain_previous_response": True, "reasoning_summary": "concise"},
        full_history_size=1,
    )
    env = _Env()
    result = await agent.sample(
        env, max_steps=3, hooks=[JsonlTrajectoryLogger(tmp_path, model_id=_MODEL)],
    )

    assert len(posts) == 2
    assert env.closed and result.terminated and result.episode_return == 1.0
    assert agent.full_history_size is None
    assert agent.api_kwargs["chain_previous_response"] is False
    assert all(
        client["trust_env"] is False and client["follow_redirects"] is False
        for client in clients
        if client["timeout"] == 600
    )
    for body in posts:
        assert body["model"] == _MODEL
        assert body["reasoning"] == params["reasoning"]
        assert body["store"] is False and body["stream"] is False
        assert body["include"] == ["message.output_text.logprobs", "reasoning.encrypted_content"]
        assert body["tools"] == [{"type": "computer"}]
        assert (
            not {"previous_response_id", "truncation", "extra_body", "prompt_cache_key"}
            & body.keys()
        )
    assert params["include"] == ["message.output_text.logprobs"]
    assert first_output == expected_raw
    assert posts[1]["input"][1:3] == expected_raw
    assert posts[1]["input"][3]["type"] == "computer_call_output"
    assert posts[1]["input"][3]["call_id"] == "call_original"
    assert (
        len(
            [
                item
                for item in posts[1]["input"]
                if item.get("role") == "user" and "Complete the sample task." in json.dumps(item)
            ]
        )
        == 1
    )
    assert all("resp_original_id/input_items" in str(request.url) for request in gets)
    assert len(gets) == 2
    assert tool_call_name(env.actions[0][0]) == "computer"
    sent_image = posts[0]["input"][0]["content"][1]["image_url"]
    assert base64.b64decode(sent_image.split(";base64,", 1)[1]) == env.screenshot
    assistant = next(
        message for message in result.lite_sample.messages if message["role"] == "assistant"
    )
    assert assistant["content"][0] == {"type": "inline_reasoning", "text": "Click the button."}

    raw = json.loads((log_dir / "00001.response.json").read_text())
    assert raw["output"] == expected_raw
    logged_request = json.loads((log_dir / "00002.request.json").read_text())
    assert logged_request["input"][1:3] == expected_raw
    image_ref = logged_request["input"][0]["content"][1]["image_url"]
    assert (log_dir / image_ref).read_bytes() == env.screenshot
    assert len(list((tmp_path / "images").iterdir())) == 1
    assert not list(log_dir.glob("*.attempt-*.response.json"))
    if shared_images:
        assert agent.request_image_dir == tmp_path / "images"
        assert image_ref.startswith("../images/")
        assert not (log_dir / "images").exists()
        row = json.loads((tmp_path / "agent_trajectory.jsonl").read_text())
        viewer_ref = next(
            part["image"]["url"] for part in row["messages"][0]["content"]
            if part["type"] == "image"
        )
        assert (log_dir / image_ref).resolve() == (tmp_path / viewer_ref).resolve()
    else:
        assert image_ref.startswith("images/")
        assert len(list((log_dir / "images").iterdir())) == 1
    assert _KEY not in "".join(path.read_text() for path in log_dir.glob("*.json"))


async def test_native_function_tools_still_use_the_canonical_parser(fake_http):
    schema = {
        "type": "function",
        "function": {
            "name": "record_result",
            "description": "Record a result.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
    env = _Env()
    env.metadata.extra_tool_schemas = [schema]
    posts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(404)
        posts.append(json.loads(request.content))
        output = [
            {
                "type": "function_call",
                "name": "record_result",
                "call_id": "fn_original",
                "arguments": '{"value":"checked"}',
            }
        ]
        return httpx.Response(200, json=_response(output if len(posts) == 1 else None))

    fake_http(handler)
    await _agent(metadata=env.metadata).sample(env, max_steps=3)
    assert posts[0]["tools"][1]["name"] == "record_result"
    assert tool_call_name(env.actions[0][0]) == "record_result"
    feedback = next(
        item for item in posts[1]["input"] if item.get("type") == "function_call_output"
    )
    assert feedback["call_id"] == "fn_original"


async def test_plain_reasoning_is_replayed_raw_and_exported_with_annotations(
    fake_http,
    tmp_path,
):
    reasoning = {
        "type": "reasoning",
        "encrypted_content": None,
        "content": [
            {"type": "reasoning_text", "text": "Inspect the current desktop.\n"},
            {"type": "reasoning_text", "text": "Then take a screenshot."},
        ],
        "summary": [{"type": "summary_text", "text": "Brief summary."}],
    }
    first_output = [
        reasoning,
        {
            "type": "computer_call",
            "call_id": "c1",
            "actions": [{"type": "screenshot"}],
        },
    ]
    original = copy.deepcopy(first_output)
    posts = []
    live_trace = []
    trace_dir = tmp_path / "trace"

    def handler(request):
        if request.method == "GET":
            return httpx.Response(404)
        posts.append(json.loads(request.content))
        if len(posts) == 2:
            live_trace.append(json.loads((trace_dir / "agent_trajectory.jsonl").read_text()))
        return httpx.Response(200, json=_response(first_output if len(posts) == 1 else None))

    fake_http(handler)
    agent = _agent(request_log_dir=tmp_path / "requests")
    result = await agent.sample(
        _Env(),
        max_steps=3,
        hooks=[JsonlTrajectoryLogger(trace_dir, model_id=_MODEL)],
    )
    expected_text = "Inspect the current desktop.\n\nThen take a screenshot."
    assistant = next(
        message for message in result.lite_sample.messages if message["role"] == "assistant"
    )
    assert assistant["reasoning_content"] == expected_text
    assert assistant["content"] == [{"type": "inline_reasoning", "text": "Brief summary."}]
    assert posts[1]["input"][1:3] == original
    assert first_output == original
    assert json.loads((tmp_path / "requests/00001.response.json").read_text())["output"] == original
    for row in [*live_trace, json.loads((trace_dir / "agent_trajectory.jsonl").read_text())]:
        message = next(message for message in row["messages"] if message["role"] == "assistant")
        assert message["reasoning_content"] == expected_text
        assert message["content"] == []
        assert message["content_annotations"] == [{
            "index": 0, "content": {"type": "inline_reasoning", "text": "Brief summary."},
        }]


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "reasoning_text", "text": "  "}],
        [{"type": "output_text", "text": "not a native reasoning block"}],
    ],
)
async def test_summary_without_full_reasoning_is_not_treated_as_replayable(content, fake_http):
    reasoning = {
        "type": "reasoning",
        "encrypted_content": None,
        "content": content,
        "summary": [{"type": "summary_text", "text": "Only a summary."}],
    }
    fake_http(lambda request: httpx.Response(200, json=_response([reasoning])))
    with pytest.raises(ResponsesGatewayError, match="replayable reasoning"):
        await _agent()._aresponses_with_retry(model=_MODEL, input=[])


@pytest.mark.parametrize("kind", ["missing_reasoning", "failed", "invalid_json", "empty"])
async def test_invalid_responses_are_logged_then_fail_before_environment_actions(
    kind,
    fake_http,
    tmp_path,
):
    if kind == "missing_reasoning":
        response = httpx.Response(200, json=_response([{"type": "reasoning", "summary": []}]))
    elif kind == "failed":
        response = httpx.Response(200, json=_response(status="failed", error={"message": _KEY}))
    elif kind == "empty":
        response = httpx.Response(200, json=_response(**{"output": []}))
    else:
        response = httpx.Response(200, text=f"invalid JSON {_KEY}")
    fake_http(lambda request: response)
    env = _Env()
    with pytest.raises(ResponsesGatewayError) as error:
        await _agent(request_log_dir=tmp_path).sample(env, max_steps=2)
    assert _KEY not in str(error.value)
    assert env.actions == [] and env.closed
    assert (tmp_path / "00001.response.json").is_file()
    assert _KEY not in (tmp_path / "00001.response.json").read_text()


async def test_transient_http_retry_reuses_exact_request_and_preserves_attempt_logs(
    fake_http,
    tmp_path,
):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, json={"api_key": _KEY, "error": f"try again {_KEY}"})
        return httpx.Response(200, json=_response())

    fake_http(handler)
    agent = _agent(
        request_log_dir=tmp_path,
        api_retry_max=1,
        api_retry_base_delay=0,
        api_retry_max_delay=0,
    )
    response = await agent._aresponses_with_retry(model=_MODEL, input=[])
    assert response["id"] == "resp_original_id"
    assert len(requests) == 2 and requests[0].content == requests[1].content
    failure = json.loads((tmp_path / "00001.attempt-01.response.json").read_text())
    assert failure == {"api_key": "[redacted]", "error": "try again [redacted]"}
    assert not (tmp_path / "00001.attempt-02.response.json").exists()
    assert json.loads((tmp_path / "00001.response.json").read_text()) == _response()


@pytest.mark.parametrize("status", [302, 400, 401])
async def test_nonretryable_http_errors_are_sanitized_and_redirects_are_not_followed(
    status,
    fake_http,
    tmp_path,
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status,
            headers={"Location": "https://other.test/"},
            json={"authorization": f"Bearer {_KEY}", "error": f"{_BASE} rejected {_KEY}"},
        )

    fake_http(handler)
    agent = _agent(request_log_dir=tmp_path, api_retry_max=1)
    with pytest.raises(ResponsesGatewayError, match=f"HTTP {status}") as error:
        await agent._aresponses_with_retry(model=_MODEL, input=[])
    assert len(requests) == 1
    assert _KEY not in str(error.value) and _BASE not in str(error.value)
    assert _KEY not in "".join(path.read_text() for path in tmp_path.glob("*.json"))
    assert _BASE not in "".join(path.read_text() for path in tmp_path.glob("*.json"))


async def test_timeout_reports_safe_error_and_cancellation_is_not_retried(fake_http, tmp_path):
    calls = []

    def timeout(request):
        calls.append(request)
        raise httpx.ReadTimeout(f"{_BASE} {_KEY}", request=request)

    fake_http(timeout)
    with pytest.raises(ResponsesGatewayError, match="ReadTimeout") as error:
        await _agent(request_log_dir=tmp_path)._aresponses_with_retry(model=_MODEL, input=[])
    assert _KEY not in str(error.value) and _BASE not in str(error.value)
    assert len(calls) == 1

    async def cancelled(request):
        calls.append(request)
        raise asyncio.CancelledError

    fake_http(cancelled)
    env = _Env()
    with pytest.raises(asyncio.CancelledError):
        await _agent(api_retry_max=2).sample(env, max_steps=2)
    assert len(calls) == 2
    assert env.closed and env.actions == []


@pytest.mark.parametrize(
    "call",
    [
        {"type": "computer_call", "call_id": "c1"},
        {"type": "computer_call", "call_id": "c1", "actions": []},
        {"type": "computer_call", "call_id": "c1", "actions": [None]},
    ],
)
def test_empty_or_invalid_computer_actions_do_not_become_successful_final_text(call):
    with pytest.raises(ResponsesGatewayError, match="valid action list"):
        _agent()._parse_output_items([call], resolution=(800, 600))


@pytest.mark.parametrize(
    "base, expected",
    [
        ("https://gateway.test", "https://gateway.test/v1"),
        ("https://gateway.test/responses", "https://gateway.test"),
        ("https://gateway.test/v1/responses", "https://gateway.test/v1"),
    ],
)
def test_gateway_base_accepts_explicit_responses_endpoint(base, expected):
    assert ResponsesGatewayAgent(model_id=_MODEL, api_key=_KEY, api_base=base).api_base == expected


@pytest.mark.parametrize(
    "params",
    [
        {"model": "fake-gpt-name"},
        {"store": True},
        {"stream": True},
        {"previous_response_id": "stale"},
        {"truncation": "auto"},
        {"include": "wrong"},
    ],
)
def test_conflicting_stateless_parameters_are_rejected(params):
    with pytest.raises(ValueError):
        _agent(request_params=params)
