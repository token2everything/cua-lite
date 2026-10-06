from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).parents[2] / "scripts/webarena_verified_qwen.py"
SPEC = importlib.util.spec_from_file_location("webarena_verified_qwen", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakePyAutoGUI:
    def __init__(self) -> None:
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return call


class FakeQwenBase:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeMessage:
    def __init__(self, content="done", reasoning_content="reason") -> None:
        self.content = content
        self.reasoning_content = reasoning_content

    def model_dump(self):
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
        }


class FakeUsage:
    def model_dump(self, mode="json"):
        return {"prompt_tokens": 10, "completion_tokens": 2}


class FakeResponse:
    def __init__(self, content="done", reasoning_content="reason") -> None:
        self.choices = [
            SimpleNamespace(
                message=FakeMessage(content, reasoning_content),
                finish_reason="stop",
            )
        ]
        self.usage = FakeUsage()

    def model_dump(self, mode="json"):
        return {
            "choices": [
                {
                    "message": self.choices[0].message.model_dump(),
                    "finish_reason": "stop",
                }
            ],
            "usage": self.usage.model_dump(),
        }


class RetryError(Exception):
    pass


class FakeAPIStatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeOpenAI:
    APIConnectionError = RetryError
    APITimeoutError = RetryError
    RateLimitError = RetryError
    BadRequestError = RetryError
    InternalServerError = RetryError
    APIStatusError = FakeAPIStatusError


def fake_client_factory(outcomes, captured):
    class Completions:
        def create(self, **kwargs):
            captured.append(kwargs)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    return lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
        constructor_kwargs=kwargs,
    )


def build_agent(
    tmp_path, *, outcomes, enable_thinking=True, sleep_fn=lambda _: None, profile="qwen"
):
    captured = []
    agent_class = runner.make_pinned_agent_class(FakeQwenBase, FakeOpenAI, model_profile=profile)
    agent = agent_class(
        model=runner.MODEL,
        max_tokens=runner.MAX_TOKENS,
        temperature=runner.TEMPERATURE,
        top_p=runner.TOP_P,
        enable_thinking=enable_thinking,
        base_url="https://model.example/v1",
        api_key="secret-key",
        request_log_dir=tmp_path / "requests",
        request_image_dir=tmp_path / "images",
        preserve_thinking=True,
        reasoning_effort="xhigh",
        client_factory=fake_client_factory(outcomes, captured),
        sleep_fn=sleep_fn,
    )
    return agent, captured


def test_validate_osworld_checkout_rejects_wrong_sha(tmp_path, monkeypatch):
    (tmp_path / "mm_agents/qwen").mkdir(parents=True)
    (tmp_path / "mm_agents/qwen/main.py").write_text("")

    def run(*args, **kwargs):
        return SimpleNamespace(stdout="wrong\n")

    monkeypatch.setattr(runner.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        runner.validate_osworld_checkout(tmp_path)


def test_validate_osworld_checkout_rejects_dirty_sources(tmp_path, monkeypatch):
    (tmp_path / "mm_agents/qwen").mkdir(parents=True)
    (tmp_path / "mm_agents/qwen/main.py").write_text("")
    results = iter(
        [
            SimpleNamespace(stdout=runner.OSWORLD_SHA + "\n"),
            SimpleNamespace(stdout=" M mm_agents/qwen/main.py\n"),
        ]
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="sources are dirty"):
        runner.validate_osworld_checkout(tmp_path)


def test_baseline_receipt_requires_matching_task_and_immutable_digests(tmp_path):
    path = tmp_path / "receipt.json"
    receipt = {
        "session_id": "session-1",
        "sandbox_id": "sandbox-1",
        "task_id": 17,
        "created_at": "2026-09-28T12:00:00Z",
        "container_digests": {"gitlab": "sha256:" + "a" * 64},
    }
    path.write_text(json.dumps(receipt))
    assert runner.validate_baseline_receipt(path, 17) == receipt
    with pytest.raises(ValueError, match="not requested task"):
        runner.validate_baseline_receipt(path, 18)
    receipt["container_digests"]["gitlab"] = "latest"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="immutable SHA-256"):
        runner.validate_baseline_receipt(path, 17)


def test_private_output_permissions_and_refuses_overwrite(tmp_path):
    output = runner.prepare_output_directory(tmp_path / "case")
    assert output.stat().st_mode & 0o777 == 0o700
    runner._atomic_json(output / "result.json", {"ok": True})
    assert (output / "result.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        runner.prepare_output_directory(output)


@pytest.mark.parametrize(
    "code, expected",
    [
        ("pyautogui.click(12, 34)", ("click", (12, 34), {})),
        (
            "pyautogui.dragTo(10, 20, duration=0.5)",
            ("dragTo", (10, 20), {"duration": 0.5}),
        ),
        ("pyautogui.hotkey('ctrl', 'l')", ("hotkey", ("ctrl", "l"), {})),
    ],
)
def test_safe_pyautogui_accepts_upstream_literal_calls(code, expected):
    module = FakePyAutoGUI()
    runner.execute_safe_pyautogui(code, module)
    assert module.calls == [expected]


@pytest.mark.parametrize(
    "code",
    [
        "__import__('os').system('id')",
        "pyautogui.click(x, 2)",
        "pyautogui.click(*[1, 2])",
        "pyautogui.screenshot()",
        "pyautogui.click(1, 2); pyautogui.click(3, 4)",
        "pyautogui.position.x()",
    ],
)
def test_safe_pyautogui_rejects_code_outside_allowlist(code):
    with pytest.raises(ValueError):
        runner.execute_safe_pyautogui(code, FakePyAutoGUI())


def test_screenshot_matches_osworld_linux_cursor_overlay():
    desktop = Image.new("RGB", (10, 10), "white")
    cursor_pixels = np.full((2, 2, 4), (255, 0, 0, 255), dtype=np.uint8)

    class PyAutoGUI:
        screenshot = staticmethod(lambda: desktop.copy())
        position = staticmethod(lambda: (3, 4))

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def getCursorImageArrayFast(self):
            return cursor_pixels

    screenshot = Image.open(__import__("io").BytesIO(runner.screenshot_bytes(PyAutoGUI, Cursor)))
    assert screenshot.getpixel((3, 4)) == (255, 0, 0)


def test_thinking_request_is_logged_without_key_and_images_are_content_addressed(tmp_path):
    agent, captured = build_agent(tmp_path, outcomes=[FakeResponse()])
    png = b"\x89PNG\r\n\x1a\ncontent"
    data_url = "data:image/png;base64," + base64.b64encode(png).decode()
    result = agent.call_llm(
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": data_url}]}]},
        runner.MODEL,
    )
    assert result == "<think>\nreason\n</think>\n\ndone"
    assert captured[0]["extra_body"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "xhigh",
    }
    request = json.loads((tmp_path / "requests/step_000_request.json").read_text())
    image_ref = request["messages"][0]["content"][0]["image_url"]
    assert image_ref.startswith("../images/")
    assert (tmp_path / "requests" / image_ref).read_bytes() == png
    assert "secret-key" not in (tmp_path / "requests/step_000_request.json").read_text()
    metadata = json.loads((tmp_path / "requests/step_000_metadata.json").read_text())
    assert metadata["reasoning_content"] == "reason"
    assert metadata["usage"]["completion_tokens"] == 2
    assert metadata["finish_reason"] == "stop"


def test_structured_repair_request_uses_json_schema_response_format(tmp_path):
    agent, captured = build_agent(tmp_path, outcomes=[FakeResponse(content='{"status":"SUCCESS"}')])
    schema = {"type": "object", "properties": {"status": {"type": "string"}}}
    agent.structured_response_schema = schema
    assert (
        agent.call_llm({"messages": []}, runner.MODEL)
        == '<think>\nreason\n</think>\n\n{"status":"SUCCESS"}'
    )
    assert captured[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "final_agent_response",
            "strict": True,
            "schema": schema,
        },
    }
    request = json.loads((tmp_path / "requests/step_000_request.json").read_text())
    assert request["response_format"]["json_schema"]["strict"] is True


def test_no_thinking_request_shape(tmp_path):
    agent, captured = build_agent(
        tmp_path,
        outcomes=[FakeResponse(reasoning_content="")],
        enable_thinking=False,
    )
    assert agent.call_llm({"messages": []}, runner.MODEL) == "done"
    assert captured[0]["extra_body"] == {
        "enable_thinking": False,
        "preserve_thinking": True,
        "reasoning_effort": "xhigh",
    }


def test_no_thinking_is_a_single_request_body_variable(tmp_path):
    enabled, enabled_calls = build_agent(tmp_path / "on", outcomes=[FakeResponse()])
    disabled, disabled_calls = build_agent(
        tmp_path / "off",
        outcomes=[FakeResponse(reasoning_content="")],
        enable_thinking=False,
    )
    enabled.call_llm({"messages": []}, runner.MODEL)
    disabled.call_llm({"messages": []}, runner.MODEL)
    enabled_body = enabled_calls[0]["extra_body"]
    disabled_body = disabled_calls[0]["extra_body"]
    assert {key for key in enabled_body if enabled_body[key] != disabled_body[key]} == {
        "enable_thinking"
    }


def test_model_request_retries_five_times_with_upstream_backoff(tmp_path):
    sleeps = []
    agent, captured = build_agent(
        tmp_path,
        outcomes=[RetryError("retry")] * runner.MODEL_ATTEMPTS,
        sleep_fn=sleeps.append,
    )
    with pytest.raises(RetryError):
        agent.call_llm({"messages": []}, runner.MODEL)
    assert len(captured) == runner.MODEL_ATTEMPTS
    assert sleeps == [5.0, 10.0, 15.0, 20.0, 25.0]


@pytest.mark.parametrize("profile, expected_requests", [("qwen", 15), ("glm53flash", 5)])
@pytest.mark.parametrize("recover", [False, True])
def test_glm_retry_owner_is_bounded_without_changing_qwen_sdk_defaults(
    tmp_path, profile, expected_requests, recover
):
    constructors, wire_requests, sleeps = [], [], []

    class Client:
        def __init__(self, **kwargs):
            constructors.append(kwargs)
            self.max_retries = kwargs.get("max_retries", 2)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            # Model the two internal SDK retries verified in the runtime image.
            for _ in range(self.max_retries + 1):
                wire_requests.append(kwargs)
                if recover and len(wire_requests) == 3:
                    return FakeResponse(reasoning_content="")
            raise RetryError("secret-key timeout")

    agent, _ = build_agent(tmp_path, outcomes=[], profile=profile, sleep_fn=sleeps.append)
    agent.client_factory = Client
    if recover:
        assert agent.call_llm({"messages": []}, runner.MODEL) == "done"
        assert len(wire_requests) == 3
        outer_attempts = 3 if profile == "glm53flash" else 1
        metadata = json.loads((tmp_path / "requests/step_000_metadata.json").read_text())
        assert metadata["attempt"] == outer_attempts
        assert sleeps == ([5.0, 10.0] if profile == "glm53flash" else [])
    else:
        with pytest.raises(RetryError, match="timeout"):
            agent.call_llm({"messages": []}, runner.MODEL)
        assert len(wire_requests) == expected_requests
        outer_attempts = runner.MODEL_ATTEMPTS
        assert sleeps == [5.0, 10.0, 15.0, 20.0, 25.0]
        assert not (tmp_path / "requests/step_000_metadata.json").exists()
    expected_constructor = {
        "base_url": "https://model.example/v1",
        "api_key": "secret-key",
        "timeout": runner.MODEL_TIMEOUT_SECONDS,
    }
    if profile == "glm53flash":
        expected_constructor["max_retries"] = 0
        expected_constructor["timeout"] = runner.httpx.Timeout(
            runner.MODEL_TIMEOUT_SECONDS, read=runner.GLM_READ_TIMEOUT_SECONDS
        )
        assert constructors[0]["timeout"].as_dict() == {
            "connect": 130.0,
            "read": 600.0,
            "write": 130.0,
            "pool": 130.0,
        }
    assert constructors == [expected_constructor]
    errors = sorted((tmp_path / "requests").glob("step_000_attempt_*_error.json"))
    assert len(errors) == (outer_attempts - 1 if recover else outer_attempts)
    for index, path in enumerate(errors, 1):
        assert path.name == f"step_000_attempt_{index:02d}_error.json"
        error = json.loads(path.read_text())
        assert error["type"] == "RetryError"
        assert "secret-key" not in error["message"]
    assert all(request == wire_requests[0] for request in wire_requests)


@pytest.mark.parametrize("profile", ["qwen", "glm53flash"])
def test_retry_policy_does_not_swallow_nonretryable_errors(tmp_path, profile):
    sleeps = []
    agent, captured = build_agent(
        tmp_path,
        outcomes=[ValueError("invalid local request")],
        profile=profile,
        sleep_fn=sleeps.append,
    )
    with pytest.raises(ValueError, match="invalid local request"):
        agent.call_llm({"messages": []}, runner.MODEL)
    assert len(captured) == 1
    assert sleeps == []
    assert not list((tmp_path / "requests").glob("*error.json"))


@pytest.mark.parametrize("profile", ["qwen", "glm53flash"])
def test_transport_error_details_are_glm_only_and_redacted(tmp_path, monkeypatch, profile):
    inner = TimeoutError("secret-key waiting for headers")
    error = RetryError("secret-key SDK timeout")
    error.__cause__ = inner
    error.status_code = 503
    error.request_id = "provider-request-123"
    timestamps = iter([10.0, 140.5, 145.5] if profile == "glm53flash" else [])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(timestamps))
    sleeps = []
    agent, captured = build_agent(
        tmp_path,
        outcomes=[error, FakeResponse(reasoning_content="")],
        profile=profile,
        sleep_fn=sleeps.append,
    )
    assert agent.call_llm({"messages": []}, runner.MODEL) == "done"
    assert len(captured) == 2 and captured[0] == captured[1]
    assert sleeps == [5.0]
    error_path = tmp_path / "requests/step_000_attempt_01_error.json"
    logged = json.loads(error_path.read_text())
    assert "secret-key" not in error_path.read_text()
    assert logged["type"] == "RetryError"
    assert logged["message"] == "[redacted] SDK timeout"
    if profile == "qwen":
        assert set(logged) == {"type", "message"}
    else:
        assert logged["elapsed_seconds"] == 130.5
        assert logged["causes"] == [
            {"type": "TimeoutError", "message": "[redacted] waiting for headers"}
        ]
        assert logged["http_status"] == 503
        assert logged["request_id"] == "provider-request-123"


@pytest.mark.parametrize("status", [400, 408, 409, 429, 500, 503])
@pytest.mark.parametrize("recover", [False, True])
def test_glm_outer_loop_takes_over_retryable_http_statuses(tmp_path, status, recover):
    outcomes = [FakeAPIStatusError(status)] * (2 if recover else runner.MODEL_ATTEMPTS)
    if recover:
        outcomes.append(FakeResponse(reasoning_content=""))
    sleeps = []
    agent, captured = build_agent(
        tmp_path, outcomes=outcomes, profile="glm53flash", sleep_fn=sleeps.append
    )
    if recover:
        assert agent.call_llm({"messages": []}, runner.MODEL) == "done"
        metadata = json.loads((tmp_path / "requests/step_000_metadata.json").read_text())
        assert metadata["attempt"] == 3
        assert len(captured) == 3
        assert sleeps == [5.0, 10.0]
    else:
        with pytest.raises(FakeAPIStatusError):
            agent.call_llm({"messages": []}, runner.MODEL)
        assert len(captured) == 5
        assert sleeps == [5.0, 10.0, 15.0, 20.0, 25.0]
    assert len(list((tmp_path / "requests").glob("*error.json"))) == (2 if recover else 5)


@pytest.mark.parametrize("profile", ["qwen", "glm53flash"])
@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_nonretryable_http_statuses_remain_immediate_failures(tmp_path, profile, status):
    sleeps = []
    agent, captured = build_agent(
        tmp_path, outcomes=[FakeAPIStatusError(status)], profile=profile, sleep_fn=sleeps.append
    )
    with pytest.raises(FakeAPIStatusError):
        agent.call_llm({"messages": []}, runner.MODEL)
    assert len(captured) == 1
    assert sleeps == []
    assert not list((tmp_path / "requests").glob("*error.json"))


def test_final_json_uses_content_not_reasoning():
    response = FakeResponse(
        content='```json\n{"status": "SUCCESS"}\n```',
        reasoning_content='{"status": "FAILURE"}',
    )
    content, reasoning, _, _ = runner.response_content(response)
    assert runner.parse_final_json(content) == {"status": "SUCCESS"}
    assert "FAILURE" in reasoning


def test_final_json_matches_official_fence_extraction_with_surrounding_text():
    content = 'Completed successfully.\n```json\n{"status": "SUCCESS"}\n```'
    assert runner.parse_final_json(content) == {"status": "SUCCESS"}


def test_public_instruction_does_not_leak_eval_expected():
    config = {
        "intent": "Find the visible value",
        "start_urls": ["http://site.example/start"],
        "eval": [{"expected": {"retrieved_data": ["SECRET ANSWER"]}}],
    }
    instruction = runner.build_public_instruction(config, {"type": "object"})
    assert "Find the visible value" in instruction
    assert "http://site.example/start" in instruction
    assert "SECRET ANSWER" not in instruction
    assert "send_msg_to_user" not in instruction
    assert "```json fenced code block" in instruction


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"sites": ["wikipedia"], "intent_template_id": 1}, 50),
        ({"sites": ["shopping_admin"], "intent_template_id": 1}, 60),
        ({"sites": ["shopping_admin"], "intent_template_id": 276}, 65),
        ({"sites": ["gitlab", "reddit"], "intent_template_id": 1}, 65),
    ],
)
def test_resolve_step_budget_layers_site_and_intent(config, expected):
    assert runner.resolve_step_budget(config, 50, layered=True) == expected
    assert runner.resolve_step_budget(config, 50, layered=False) == 50


def test_normalize_agent_response_validates_and_canonicalizes_fenced_json():
    class Validated:
        def model_dump(self, mode="json"):
            return {"task_type": "RETRIEVE", "status": "SUCCESS"}

    class ResponseModel:
        seen = None

        @classmethod
        def model_validate(cls, value):
            cls.seen = value
            return Validated()

    normalized, raw = runner.normalize_agent_response(
        '```json\n{"status": "SUCCESS", "task_type": "RETRIEVE"}\n```',
        ResponseModel,
    )
    assert ResponseModel.seen == {"status": "SUCCESS", "task_type": "RETRIEVE"}
    assert normalized == {"task_type": "RETRIEVE", "status": "SUCCESS"}
    assert raw == '{"task_type":"RETRIEVE","status":"SUCCESS"}'


def test_failure_finalization_flushes_trace_and_har(tmp_path):
    trace_path = tmp_path / "trace.zip"
    har_path = tmp_path / "network.har"

    class Tracing:
        def stop(self, *, path):
            path.write_bytes(b"trace")

    class Context:
        tracing = Tracing()

        def close(self):
            har_path.write_text('{"log": {"entries": []}}')

    browser = SimpleNamespace(close=lambda: None)
    error = runner.finalize_browser_artifacts(
        Context(),
        browser,
        trace_path,
        tracing_started=True,
        sleep_fn=lambda _: None,
    )
    assert error is None
    assert trace_path.read_bytes() == b"trace"
    assert har_path.is_file()


def test_capture_failure_closes_owned_browser_before_har_flush(tmp_path):
    events = []
    capture_error = RuntimeError("Finished response metadata capture timeout")

    class Tracing:
        def stop(self, *, path):
            events.append("trace")
            raise capture_error

    class Context:
        tracing = Tracing()

        def close(self):
            assert events == ["trace", "browser"]
            events.append("har")

    browser = SimpleNamespace(close=lambda: events.append("browser"))
    error = runner.finalize_browser_artifacts(
        Context(),
        browser,
        tmp_path / "trace.zip",
        tracing_started=True,
        sleep_fn=lambda _: None,
    )
    assert error is capture_error
    assert events == ["trace", "browser", "har"]


def test_capture_failure_preserved_when_owned_browser_close_also_fails(tmp_path):
    events = []
    capture_error = RuntimeError("incomplete response metadata")

    class Tracing:
        def stop(self, *, path):
            raise capture_error

    class Context:
        tracing = Tracing()

        def close(self):
            events.append("har")

    def close_browser():
        events.append("browser")
        raise RuntimeError("browser already disconnected")

    error = runner.finalize_browser_artifacts(
        Context(),
        SimpleNamespace(close=close_browser),
        tmp_path / "trace.zip",
        tracing_started=True,
        sleep_fn=lambda _: None,
    )
    assert error is capture_error
    assert events == ["browser", "har"]


def test_secure_result_tree_makes_browser_artifacts_private(tmp_path):
    output = tmp_path / "output"
    nested = output / "requests"
    nested.mkdir(parents=True)
    artifact = nested / "network.har"
    artifact.write_text("{}")
    output.chmod(0o755)
    nested.chmod(0o755)
    artifact.chmod(0o644)
    runner.secure_result_tree(output)
    assert output.stat().st_mode & 0o777 == 0o700
    assert nested.stat().st_mode & 0o777 == 0o700
    assert artifact.stat().st_mode & 0o777 == 0o600


def test_append_trajectory_is_private_jsonl(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    runner.append_trajectory(path, {"step": 0})
    runner.append_trajectory(path, {"step": 1})
    assert path.stat().st_mode & 0o777 == 0o600
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"step": 0},
        {"step": 1},
    ]


def test_artifact_receipt_binds_exact_bytes(tmp_path):
    path = tmp_path / "trace.zip"
    path.write_bytes(b"trace bytes")
    file_identity = runner.artifact_identity(path)
    content_identity = runner.content_identity("trace bytes")
    assert file_identity["size"] == content_identity["size"] == 11
    assert file_identity["sha256"] == content_identity["sha256"]


def test_retry_error_log_redacts_api_key(tmp_path):
    agent, _ = build_agent(
        tmp_path,
        outcomes=[RetryError("secret-key must not leak")] * runner.MODEL_ATTEMPTS,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(RetryError):
        agent.call_llm({"messages": []}, runner.MODEL)
    error_log = (tmp_path / "requests/step_000_attempt_01_error.json").read_text()
    assert "secret-key" not in error_log
    assert "[redacted]" in error_log


def test_normalize_origin_to_v1():
    assert runner.normalize_api_base("https://gateway.example") == "https://gateway.example/v1"
    assert runner.normalize_api_base("https://gateway.example/custom/v1/") == (
        "https://gateway.example/custom/v1"
    )


class FakeStartPage:
    def __init__(self, url, *, status=200, title="Task page", body="Task content", reloads=None):
        self.url = url
        self.status = status
        self.title = title
        self.body = body
        self.reloads = list(reloads or [])
        self.goto_calls = []

    def wait_for_load_state(self, state, *, timeout):
        assert state == "domcontentloaded"
        assert timeout == 10000

    def evaluate(self, script):
        assert "document.body" in script
        return {
            "title": self.title,
            "body": self.body,
            "hasBody": True,
            "readyState": "complete",
        }

    def goto(self, url, *, wait_until, timeout):
        self.goto_calls.append(url)
        assert wait_until == "domcontentloaded"
        assert timeout == 30000
        if self.reloads:
            self.status, self.title, self.body = self.reloads.pop(0)
        self.url = url
        return SimpleNamespace(status=self.status)


@pytest.mark.parametrize(
    "task_id,start_url,task_url,error_url,error_status",
    [
        (
            0,
            "http://localhost:7780/admin",
            "http://localhost:7780/admin/admin/dashboard/",
            "http://localhost:7780/admin",
            502,
        ),
        (
            3,
            "http://localhost:7780/admin",
            "http://localhost:7780/admin/admin/dashboard/",
            "http://localhost:7780/admin",
            200,
        ),
        (
            449,
            "http://127.0.0.1:8023",
            "http://127.0.0.1:8023/",
            "http://127.0.0.1:8023/users/sign_in",
            502,
        ),
    ],
)
def test_startup_gate_selects_task_page_before_leaked_login_error_tab(
    tmp_path, task_id, start_url, task_url, error_url, error_status
):
    task_page = FakeStartPage(task_url, body="Authenticated task page")
    failed_login = FakeStartPage(
        error_url, status=error_status, title="502 Bad Gateway", body="502 Bad Gateway"
    )
    context = SimpleNamespace(pages=[task_page, failed_login])
    receipt_path = tmp_path / f"startup_{task_id}.json"

    selected = runner.select_healthy_start_page(
        context,
        [start_url],
        [
            (task_page, start_url, 302),
            (task_page, task_url, 200),
            (failed_login, error_url, error_status),
        ],
        receipt_path,
        sleep_fn=lambda _: None,
    )

    assert selected is task_page
    receipt = json.loads(receipt_path.read_text())
    assert receipt["result"] == "passed"
    assert receipt["selected_page_index"] == 0
    assert receipt["pages"][1]["url"] == failed_login.url
    assert receipt["pages"][1]["document_responses"][-1]["status"] == error_status
    assert receipt["checks"][0]["attempts"][0]["document_status"] == 200


def test_startup_gate_preserves_last_configured_page_with_multiple_start_urls(tmp_path):
    first_url = "http://localhost:8080"
    second_url = "http://localhost:7780/admin"
    first = FakeStartPage(first_url, title="Reddit")
    second = FakeStartPage(second_url, title="Admin")
    leaked = FakeStartPage("http://localhost:7780/admin/login", title="404 Not Found")
    context = SimpleNamespace(pages=[first, second, leaked])

    selected = runner.select_healthy_start_page(
        context,
        [first_url, second_url],
        [(first, first_url, 200), (second, second_url, 200), (leaked, leaked.url, 404)],
        tmp_path / "receipt.json",
        sleep_fn=lambda _: None,
    )

    assert selected is second
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert [check["page_index"] for check in receipt["checks"]] == [0, 1]
    assert receipt["selected_page_index"] == 1


def test_startup_gate_retries_502_and_records_recovered_response(tmp_path):
    start_url = "http://127.0.0.1:8023"
    page = FakeStartPage(
        start_url,
        status=502,
        title="502 Bad Gateway",
        body="502 Bad Gateway",
        reloads=[(200, "GitLab", "Projects")],
    )
    sleeps = []

    selected = runner.select_healthy_start_page(
        SimpleNamespace(pages=[page]),
        [start_url],
        [(page, start_url, 502)],
        tmp_path / "receipt.json",
        sleep_fn=sleeps.append,
    )

    assert selected is page
    assert page.goto_calls == [start_url]
    assert sleeps == [1]
    attempts = json.loads((tmp_path / "receipt.json").read_text())["checks"][0]["attempts"]
    assert [(item["document_status"], item["healthy"]) for item in attempts] == [
        (502, False),
        (200, True),
    ]


def test_startup_gate_fails_before_model_on_persistent_error_page(tmp_path):
    start_url = "http://127.0.0.1:8023"
    page = FakeStartPage(start_url, status=200, title="502 Bad Gateway", body="502 Bad Gateway")
    receipt_path = tmp_path / "receipt.json"

    with pytest.raises(RuntimeError, match="Startup page gate failed"):
        runner.select_healthy_start_page(
            SimpleNamespace(pages=[page]),
            [start_url],
            [(page, start_url, 200)],
            receipt_path,
            sleep_fn=lambda _: None,
        )

    receipt = json.loads(receipt_path.read_text())
    assert receipt["result"] == "failed"
    assert len(receipt["checks"][0]["attempts"]) == 3
    assert receipt["selected_page_index"] is None


def test_startup_page_gate_is_opt_in_for_existing_qwen_cli():
    required = [
        "--task-id",
        "0",
        "--osworld-root",
        "/tmp/osworld",
        "--baseline-receipt",
        "/tmp/receipt",
        "--output-dir",
        "/tmp/output",
        "--lock-dir",
        "/tmp/lock",
    ]
    assert runner._parse_args(required).startup_page_gate is False
    assert runner._parse_args([*required, "--startup-page-gate"]).startup_page_gate is True
