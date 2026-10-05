from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/webarena_verified_qwen.py"
SPEC = importlib.util.spec_from_file_location("webarena_verified_qwen_glm_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class ResponseModel:
    @classmethod
    def model_validate(cls, value):
        if not isinstance(value, dict) or value.get("status") != "SUCCESS":
            raise ValueError("status must be SUCCESS")
        return SimpleNamespace(model_dump=lambda mode: value)

    @classmethod
    def model_json_schema(cls):
        return {"type": "object", "properties": {"status": {"const": "SUCCESS"}}}


def tool_call(action, **params):
    fields = {"action": action, **params}
    body = "\n".join(
        f"<parameter={name}>{json.dumps(value) if isinstance(value, list) else value}</parameter>"
        for name, value in fields.items()
    )
    return f"Action: use the GUI\n<tool_call><function=computer_use>{body}</function></tool_call>"


def classify(content, actions, *, native=False, parser_error=None):
    return runner.classify_glm_turn(
        content,
        actions,
        ResponseModel,
        processed_size=(1920, 1088),
        original_size=(1920, 1080),
        native_tool_calls=native,
        parser_error=parser_error,
    )


@pytest.mark.parametrize(
    "content, actions, expected",
    [
        ('Done.\n```json\n{"status":"SUCCESS"}\n```', ["DONE"], "final"),
        (tool_call("left_click", coordinate=[1897, 345]), ["pyautogui.click(1897, 342)"], "action"),
        (
            "Click Reports.\n</tool_call>\n"
            + tool_call("left_click", coordinate=[50, 570]),
            ["pyautogui.click(50, 565)"],
            "action",
        ),
        ("I will click the tab now.", ["DONE"], "action_error"),
        (
            "<tool_call><function=computer_use><action>left_click</action></function></tool_call>",
            ["DONE"],
            "action_error",
        ),
        (
            "<tool_call><function=wrong><parameter=action>left_click</parameter></function></tool_call>",
            ["DONE"],
            "action_error",
        ),
        (tool_call("`left_click`", coordinate=[20, 30]), ["DONE"], "action_error"),
        (tool_call("terminate", status="success"), ["DONE"], "final_error"),
        (
            tool_call("left_click", coordinate=[20, 30]) + '\n```json\n{"status":"SUCCESS"}\n```',
            ["pyautogui.click(20, 30)"],
            "action_error",
        ),
    ],
)
def test_glm_turn_classification_never_accepts_parser_fabricated_done(content, actions, expected):
    assert classify(content, actions).kind == expected


def test_glm_orphan_closing_tag_does_not_hide_multiple_or_trailing_calls():
    call = tool_call("left_click", coordinate=[50, 570])
    assert classify("</tool_call>" + call + call, ["DONE"]).kind == "action_error"
    assert classify("</tool_call>" + call + "</tool_call>", ["DONE"]).kind == "action_error"
    assert classify("<function=computer_use></tool_call>" + call, ["DONE"]).kind == (
        "action_error"
    )


@pytest.mark.parametrize(
    "coordinate",
    [[1920, 10], [10, 1088], [-1, 10], [10], [10, 20, 30], [True, 20], "150, 22", ["nan", 20]],
)
def test_glm_rejects_supplied_bad_coordinates(coordinate):
    turn = classify(tool_call("left_click", coordinate=coordinate), ["pyautogui.click()"])
    assert turn.kind == "action_error"
    assert "coordinate" in turn.error


def test_glm_absolute_coordinate_maps_processed_image_to_desktop():
    turn = classify(
        tool_call("left_click", coordinate=[1897, 345]),
        ["pyautogui.click(1897, 342)"],
    )
    assert turn.kind == "action"
    assert turn.coordinate == (1897, 345)
    assert turn.desktop_coordinate == (1897, 342)
    assert classify(tool_call("left_click"), ["pyautogui.click()"]).kind == "action"
    assert classify(tool_call("mouse_move"), ["pyautogui.moveTo(0, 0)"]).kind == "action_error"


def test_glm_wait_preserves_seconds_and_screenshot_does_not_wait():
    wait = classify(tool_call("wait", time=15), ["WAIT"])
    screenshot = classify(tool_call("screenshot"), ["WAIT"])
    assert wait.kind == screenshot.kind == "action"
    assert wait.wait_seconds == 15
    assert screenshot.wait_seconds is None
    assert classify(tool_call("wait", time="nan"), ["WAIT"]).kind == "action_error"


class FakeProviderResponse:
    def __init__(self, content, reasoning, *, tool_calls=None):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                ),
                finish_reason="stop",
            )
        ]
        self.usage = None

    def model_dump(self, mode="json"):
        return {
            "choices": [
                {
                    "message": {
                        "content": self.choices[0].message.content,
                        "reasoning_content": self.choices[0].message.reasoning_content,
                        "tool_calls": self.choices[0].message.tool_calls,
                    }
                }
            ]
        }


class FakeBaseAgent:
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.parsed = None

    def _parse_response(self, response, **sizes):
        self.parsed = response
        if "<tool_call>" in response:
            return "click", ["pyautogui.click(100, 100)"]
        return "done", ["DONE"]


def make_agent(tmp_path, response, profile, *, base_class=FakeBaseAgent):
    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )

    class OpenAI:
        OpenAI = Client

    agent_class = runner.make_pinned_agent_class(base_class, OpenAI, model_profile=profile)
    return agent_class(
        model="glm-5.3-flash" if profile == "glm53flash" else runner.MODEL,
        enable_thinking=True,
        base_url="https://example.test/v1",
        api_key="test-key",
        request_log_dir=tmp_path / "requests",
        request_image_dir=tmp_path / "images",
        preserve_thinking=True,
        reasoning_effort="xhigh",
    )


def test_glm_parses_only_content_and_keeps_reasoning_in_history(tmp_path):
    reasoning = tool_call("left_click", coordinate=[10, 10])
    content = tool_call("left_click", coordinate=[100, 100])
    agent = make_agent(tmp_path, FakeProviderResponse(content, reasoning), "glm53flash")
    response = agent.call_llm({"messages": []}, agent.model)
    assert reasoning in response
    _, actions = agent._parse_response(
        response,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )
    assert agent.parsed == content
    assert len(actions) == 1
    assert classify(agent.parsed, actions).kind == "action"


def test_glm_projects_real_native_child_fields_without_changing_raw_history(tmp_path):
    content = (
        "Action: Click the REPORTS icon.\n"
        "<tool_call>\n<function=computer_use>\n"
        "<action>left_click</action>\n"
        "<coordinate>[[50, 570]]</coordinate>\n"
        "</function>\n</tool_call>"
    )

    class CanonicalParser(FakeBaseAgent):
        def _parse_response(self, response, **sizes):
            self.parsed = response
            assert "<parameter=action>left_click</parameter>" in response
            assert "<parameter=coordinate>[50, 570]</parameter>" in response
            return "click", ["pyautogui.click(50, 565)"]

    agent = make_agent(
        tmp_path, FakeProviderResponse(content, ""), "glm53flash", base_class=CanonicalParser
    )
    response = agent.call_llm({"messages": []}, agent.model)
    _, actions = agent._parse_response(
        response,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )
    assert response == agent.last_provider_content == content
    assert agent.last_projected_content == agent.parsed
    turn = classify(agent.last_projected_content, actions)
    assert turn.kind == "action"
    assert turn.coordinate == (50, 570)
    assert turn.desktop_coordinate == (50, 565)


def test_qwen_default_still_parses_composite_response(tmp_path):
    reasoning = tool_call("left_click", coordinate=[10, 10])
    content = '{"status":"SUCCESS"}'
    agent = make_agent(tmp_path, FakeProviderResponse(content, reasoning), "qwen")
    response = agent.call_llm({"messages": []}, agent.model)
    agent._parse_response(
        response,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )
    assert agent.parsed == response
    assert agent.model == runner.MODEL


def test_glm_native_tool_calls_are_rejected_before_execution(tmp_path):
    agent = make_agent(
        tmp_path,
        FakeProviderResponse("", "", tool_calls=[{"function": {"name": "computer_use"}}]),
        "glm53flash",
    )
    agent.call_llm({"messages": []}, agent.model)
    assert agent.last_native_tool_calls is True
    assert (
        classify(agent.last_provider_content, [], native=agent.last_native_tool_calls).kind
        == "action_error"
    )


def test_glm_upstream_parser_exception_becomes_model_visible_action_error(tmp_path):
    class ExplodingParser(FakeBaseAgent):
        def _parse_response(self, response, **sizes):
            raise TypeError("int() argument must be a number")

    content = tool_call("left_click", coordinate=[None, 2])
    agent = make_agent(
        tmp_path, FakeProviderResponse(content, ""), "glm53flash", base_class=ExplodingParser
    )
    response = agent.call_llm({"messages": []}, agent.model)
    _, actions = agent._parse_response(
        response,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )
    assert actions == []
    assert "TypeError" in agent.last_upstream_parse_error
    turn = classify(content, actions, parser_error=agent.last_upstream_parse_error)
    assert turn.kind == "action_error"
    assert "invalid computer_use parameters" in turn.error


def test_model_profile_defaults_to_original_qwen_path():
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
    assert runner._parse_args(required).model_profile == "qwen"
    assert (
        runner._parse_args([*required, "--model-profile", "glm53flash"]).model_profile
        == "glm53flash"
    )


def test_glm_profile_prefers_its_key_and_qwen_keeps_original_key(monkeypatch):
    monkeypatch.setenv("GLM_TEST_API_KEY", "glm-test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "qwen-test-secret")
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
    assert runner._parse_args(required).api_key == "qwen-test-secret"
    assert runner._parse_args([*required, "--model-profile", "glm53flash"]).api_key == (
        "glm-test-secret"
    )
    monkeypatch.delenv("GLM_TEST_API_KEY")
    assert runner._parse_args([*required, "--model-profile", "glm53flash"]).api_key == (
        "qwen-test-secret"
    )


def run_fake_glm_case(tmp_path, monkeypatch, turns, evaluation=None, *, setup_error=None):
    """Exercise the runner without a browser, sandbox, or model API."""
    from playwright import sync_api
    from webarena_verified import types as response_types

    from lite.gym.envs.browsergym import webarena_verified

    class Task:
        config = {
            "intent": "Open the page",
            "start_urls": ["http://example.test"],
            "sites": ["wikipedia"],
            "intent_template_id": 1,
        }

        def __init__(self, **kwargs):
            pass

        def setup(self, page):
            if setup_error is not None:
                raise setup_error

    class Page:
        def __init__(self):
            self.waits = []
            self.url = "http://example.test"

        def bring_to_front(self):
            pass

        def wait_for_timeout(self, milliseconds):
            self.waits.append(milliseconds)

    page = Page()
    context = SimpleNamespace(
        pages=[page],
        new_page=lambda: page,
        on=lambda *args: None,
        remove_listener=lambda *args: None,
    )
    browser = SimpleNamespace(new_context=lambda **kwargs: context, close=lambda: None)
    playwright = SimpleNamespace(
        chromium=SimpleNamespace(launch=lambda **kwargs: browser), stop=lambda: None
    )

    class Agent:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            page.reasoning_effort = kwargs["reasoning_effort"]
            self.last_native_tool_calls = False
            self.last_upstream_parse_error = None
            self.last_processed_size = (1920, 1088)
            self.last_original_size = (1920, 1080)

        def reset(self, logger):
            pass

        def predict(self, instruction, observation):
            self.last_provider_content, actions = turns.pop(0)
            self.last_projected_content = self.last_provider_content
            return self.last_provider_content, actions

    monkeypatch.setattr(webarena_verified, "task_records", lambda: [dict(task_id=0)])
    monkeypatch.setattr(webarena_verified, "DATASET_SHA256", "dataset")
    browsergym_package = ModuleType("browsergym.webarena_verified")
    browsergym_package.__path__ = []
    task_module = ModuleType("browsergym.webarena_verified.task")
    task_module.WebArenaVerifiedTask = Task
    browsergym_package.task = task_module
    monkeypatch.setitem(sys.modules, "browsergym.webarena_verified", browsergym_package)
    monkeypatch.setitem(sys.modules, "browsergym.webarena_verified.task", task_module)
    monkeypatch.setitem(sys.modules, "openai", ModuleType("openai"))
    monkeypatch.setattr(response_types, "FinalAgentResponse", ResponseModel)
    monkeypatch.setattr(
        sync_api, "sync_playwright", lambda: SimpleNamespace(start=lambda: playwright)
    )
    monkeypatch.setitem(sys.modules, "pyautogui", ModuleType("pyautogui"))
    monkeypatch.setattr(runner, "validate_baseline_receipt", lambda *args: {"sandbox_id": "one"})
    monkeypatch.setattr(runner, "validate_osworld_checkout", lambda *args: tmp_path)
    monkeypatch.setattr(runner, "runtime_metadata", lambda *args: {})
    monkeypatch.setattr(runner, "sandbox_lock", lambda *args: nullcontext())
    monkeypatch.setattr(runner, "xvfb", lambda *args: nullcontext())
    monkeypatch.setattr(runner, "working_directory", lambda *args: nullcontext())
    monkeypatch.setattr(runner, "patch_pyautogui", lambda *args: None)
    monkeypatch.setattr(runner, "load_cursor_class", lambda *args: object)
    monkeypatch.setattr(runner, "load_qwen_agent", lambda *args: object)
    monkeypatch.setattr(runner, "make_pinned_agent_class", lambda *args, **kwargs: Agent)
    monkeypatch.setattr(runner, "screenshot_bytes", lambda *args: b"fake png")
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        runner,
        "score_artifacts",
        lambda *args: evaluation
        or {"score": 0, "integrity_status": "passed", "publishable": True},
    )
    monkeypatch.setattr(runner, "secure_result_tree", lambda *args: None)

    def finalize(context, browser, trace_path, **kwargs):
        (tmp_path / "case" / "network.har").write_text("{}")
        return None

    monkeypatch.setattr(runner, "finalize_browser_artifacts", finalize)
    args = SimpleNamespace(
        task_id=0,
        baseline_receipt=tmp_path / "baseline.json",
        osworld_root=tmp_path,
        output_dir=tmp_path / "case",
        lock_dir=tmp_path / "locks",
        base_url="https://example.test/v1",
        api_key="test-key",
        display=":99",
        max_steps=4,
        layered_step_budget=False,
        final_response_reserve_steps=0,
        repair_invalid_response=True,
        no_thinking=False,
        startup_page_gate=setup_error is not None,
        model_profile="glm53flash",
    )
    status = runner.run_case(args)
    trajectories = [
        json.loads(line)
        for line in (tmp_path / "case" / "trajectory.jsonl").read_text().splitlines()
    ]
    return status, trajectories, page


def test_glm_startup_failure_records_status_before_step_loop(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="GitLab unavailable"):
        run_fake_glm_case(
            tmp_path,
            monkeypatch,
            [],
            setup_error=RuntimeError("GitLab unavailable"),
        )
    status = json.loads((tmp_path / "case" / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["failure_phase"] == "startup"
    assert status["fixed_parameters"]["max_steps"] == 4
    assert status["error"]["message"] == "GitLab unavailable"


def test_glm_runner_executes_wait_time_and_then_accepts_final_json(tmp_path, monkeypatch):
    status, trajectories, page = run_fake_glm_case(
        tmp_path,
        monkeypatch,
        [(tool_call("wait", time=1.5), ["WAIT"]), ('{"status":"SUCCESS"}', ["DONE"])],
    )
    assert page.waits == [1500]
    assert trajectories[0]["wait_seconds_requested"] == 1.5
    assert trajectories[0]["executed_actions"] == ["WAIT"]
    assert trajectories[1]["executed_actions"] == ["FINAL_JSON"]
    assert status["status"] == "completed"
    assert status["model"] == "glm-5.3-flash"
    assert status["fixed_parameters"]["coordinate_type"] == "absolute"
    assert status["fixed_parameters"]["reasoning_effort"] == page.reasoning_effort == "max"


def test_glm_runner_screenshot_refreshes_without_waiting(tmp_path, monkeypatch):
    status, trajectories, page = run_fake_glm_case(
        tmp_path,
        monkeypatch,
        [(tool_call("screenshot"), ["WAIT"]), ('{"status":"SUCCESS"}', ["DONE"])],
    )
    assert page.waits == []
    assert trajectories[0]["executed_actions"] == ["WAIT"]
    assert "wait_seconds_requested" not in trajectories[0]
    assert status["status"] == "completed"


def test_glm_runner_final_json_repair_has_independent_budget(tmp_path, monkeypatch):
    status, trajectories, page = run_fake_glm_case(
        tmp_path,
        monkeypatch,
        [('{"status":"FAILURE"}', ["DONE"]), ('{"status":"SUCCESS"}', ["DONE"])],
    )
    assert page.waits == []
    assert trajectories[0]["repair_requested"] == "final_error"
    assert trajectories[0]["executed_actions"] == []
    assert status["status"] == "completed"
    assert status["response_repair_used"] is True
    assert status["action_repair_count"] == 0


def test_glm_final_repair_cannot_execute_a_tool_call(tmp_path, monkeypatch):
    status, trajectories, page = run_fake_glm_case(
        tmp_path,
        monkeypatch,
        [
            ('{"status":"FAILURE"}', ["DONE"]),
            (tool_call("left_click", coordinate=[100, 100]), ["pyautogui.click(100, 99)"]),
        ],
    )
    assert page.waits == []
    assert trajectories[1]["protocol_failure"] == "final_error"
    assert trajectories[1]["executed_actions"] == []
    assert status["status"] == "model_protocol_failed"


def test_glm_runner_repairs_bad_action_then_records_distinct_failure(tmp_path, monkeypatch):
    status, trajectories, page = run_fake_glm_case(
        tmp_path,
        monkeypatch,
        [("I will click the tab now.", ["DONE"]), ("Still planning the click.", ["DONE"])],
    )
    assert page.waits == []
    assert trajectories[0]["repair_requested"] == "action_error"
    assert trajectories[0]["executed_actions"] == []
    assert trajectories[1]["protocol_failure"] == "action_error"
    assert trajectories[1]["executed_actions"] == []
    assert status["status"] == "model_protocol_failed"
    assert status["action_repair_count"] == 1
    assert status["response_repair_used"] is False


def test_glm_runner_evaluator_error_is_not_a_model_zero(tmp_path, monkeypatch):
    status, _, _ = run_fake_glm_case(
        tmp_path,
        monkeypatch,
        [('{"status":"SUCCESS"}', ["DONE"])],
        evaluation={"score": 0, "integrity_status": "evaluator_error", "publishable": False},
    )

    assert status["status"] == "evaluator_error"
    assert status["evaluation_integrity"] == "evaluator_error"
    assert status["score"] == 0


def test_glm_runner_evidence_disagreement_is_separate_from_model_failure(
    tmp_path, monkeypatch
):
    status, _, _ = run_fake_glm_case(
        tmp_path,
        monkeypatch,
        [('{"status":"SUCCESS"}', ["DONE"])],
        evaluation={
            "score": 0,
            "integrity_status": "evidence_disagreement",
            "publishable": False,
        },
    )

    assert status["status"] == "evidence_invalid"
    assert status["evaluation_integrity"] == "evidence_disagreement"
