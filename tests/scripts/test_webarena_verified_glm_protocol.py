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
            "Click Reports.\n</tool_call>\n" + tool_call("left_click", coordinate=[50, 570]),
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
    assert classify("<function=computer_use></tool_call>" + call, ["DONE"]).kind == ("action_error")


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


@pytest.mark.parametrize(
    "coordinate_field, expected",
    [
        ("<coordinate>558, 113</coordinate>", (558, 113)),
        ("<coordinate>1222 908</coordinate>", (1222, 908)),
        ("<parameter=coordinate>558, 113</parameter>", (558, 113)),
        ("<parameter=coordinate>1222 908</parameter>", (1222, 908)),
        ("<coordinate>559</coordinate><parameter>113</parameter>", (559, 113)),
        ("<coordinate>1473</coordinate><coordinate>115</coordinate>", (1473, 115)),
        ("<parameter=coordinate>(1204, 561)</parameter>", (1204, 561)),
        ("<parameter=coordinate>[[573, 479]]</parameter>", (573, 479)),
    ],
)
def test_glm_projects_two_numeric_pixel_tokens_before_gui_parser(
    tmp_path, coordinate_field, expected
):
    action_field = (
        "<parameter=action>left_click</parameter>"
        if coordinate_field.startswith("<parameter=")
        else "<action>left_click</action>"
    )
    content = (
        f"<tool_call><function=computer_use>{action_field}{coordinate_field}</function></tool_call>"
    )
    agent = make_agent(tmp_path, FakeProviderResponse(content, ""), "glm53flash")
    response = agent.call_llm({"messages": []}, agent.model)
    _, actions = agent._parse_response(
        response,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )

    assert agent.last_provider_content == content
    assert f"<parameter=coordinate>[{expected[0]}, {expected[1]}]</parameter>" in (
        agent.last_projected_content
    )
    assert classify(agent.last_projected_content, actions).coordinate == expected


def test_glm_projects_observed_list_wrapper_and_inline_keys(tmp_path):
    coordinate_content = (
        "Action: Click the project.\n<li><function=computer_use>"
        "<parameter=action>left_click</parameter>"
        "<parameter=coordinate>[[573, 479]]</parameter>"
        "</function></li>"
    )
    keys_content = (
        "<tool_call><function=computer_use><action>key</action>"
        '<keys=["ctrl", "shift", "home"]></function></tool_call>'
    )
    for content, expected in (
        (coordinate_content, "<parameter=coordinate>[573, 479]</parameter>"),
        (keys_content, '<parameter=keys>["ctrl", "shift", "home"]</parameter>'),
    ):
        agent = make_agent(tmp_path, FakeProviderResponse(content, ""), "glm53flash")
        response = agent.call_llm({"messages": []}, agent.model)
        _, actions = agent._parse_response(
            response,
            original_width=1920,
            original_height=1080,
            processed_width=1920,
            processed_height=1088,
        )
        assert response == agent.last_provider_content == content
        assert expected in agent.last_projected_content
        assert classify(agent.last_projected_content, actions).kind == "action"


@pytest.mark.parametrize(
    "call",
    [
        "<head><function=computer_use><parameter=action>left_click</parameter>"
        "<parameter=coordinate>[1209, 561]</parameter></function></head>",
        "<head>" + tool_call("left_click", coordinate=[1183, 109]).split("\n", 1)[1] + "</head>",
        "<html><body>"
        + tool_call("left_click", coordinate=[559, 114]).split("\n", 1)[1]
        + "</body></html>",
        "<table>" + tool_call("left_click", coordinate=[558, 113]).split("\n", 1)[1] + "</table>",
        "<table><function=computer_use><parameter=action>left_click</parameter>"
        "<parameter=coordinate>[558, 113]</parameter></function></table>",
    ],
)
def test_glm_projects_balanced_observed_wrappers_around_complete_call(tmp_path, call):
    content = "Action: Click the target.\n" + call
    agent = make_agent(tmp_path, FakeProviderResponse(content, ""), "glm53flash")
    agent.last_provider_content = content
    _, actions = agent._parse_response(
        content,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )
    assert agent.last_provider_content == content
    assert classify(agent.last_projected_content, actions).kind == "action"
    assert agent.last_projected_content.endswith("</tool_call>")


@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ("<html><body>", "</html></body>"),
        ("<head>", "</body>"),
        ("<head>", "</head>extra text"),
        ("<html><body>", "</body>"),
        ("<body>unexpected text", "</body>"),
        ("<table>", "</body>"),
        ("<table>", "</table>extra text"),
        ("<table>unexpected text", "</table>"),
        ('<table class="action">', "</table>"),
    ],
)
def test_glm_wrapper_projection_keeps_mismatched_or_nonempty_wrappers_rejected(
    tmp_path, prefix, suffix
):
    content = prefix + tool_call("left_click", coordinate=[559, 114]).split("\n", 1)[1] + suffix
    agent = make_agent(tmp_path, FakeProviderResponse(content, ""), "glm53flash")
    agent.last_provider_content = content
    _, actions = agent._parse_response(
        content,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )
    assert agent.last_provider_content == content
    assert classify(agent.last_projected_content, actions).kind == "action_error"


@pytest.mark.parametrize(
    "prefix",
    [
        "<table>",
        "<html><body>Action: Click Forums.</body></html>\n",
        "Action: <li>Click Forums.</li>\n",
        "<table><tr><td>Action: Click Forums.</td></tr></table>\n",
        "Action: Type the literal markup <table>\n",
    ],
)
def test_glm_keeps_unprojected_prefix_text_and_its_existing_action_meaning(tmp_path, prefix):
    content = prefix + tool_call("left_click", coordinate=[558, 113]).split("\n", 1)[1]
    agent = make_agent(tmp_path, FakeProviderResponse(content, ""), "glm53flash")
    agent.last_provider_content = content
    _, actions = agent._parse_response(
        content,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )
    assert agent.last_projected_content == agent.last_provider_content == content
    assert classify(agent.last_projected_content, actions).kind == "action"


@pytest.mark.parametrize(
    "prefix",
    [
        "<table><tr><td>Action: Click Forums.</td></tr></table>\n",
        "Action: Click the item labeled <table>.\n",
        "<html>prefix prose\n",
        "</body>",
    ],
)
def test_glm_projects_adjacent_balanced_wrappers_without_interpreting_prefix_prose(
    tmp_path, prefix
):
    call = tool_call("left_click", coordinate=[558, 113]).split("\n", 1)[1]
    content = prefix + "<html><body>" + call + "</body></html>"
    agent = make_agent(tmp_path, FakeProviderResponse(content, ""), "glm53flash")
    agent.last_provider_content = content
    _, actions = agent._parse_response(
        content,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )
    assert agent.last_provider_content == content
    assert agent.last_projected_content == prefix + call
    assert classify(agent.last_projected_content, actions).kind == "action"


@pytest.mark.parametrize(
    "fields",
    [
        "<coordinate>559</coordinate>",
        "<coordinate>559</coordinate><parameter>unknown</parameter>",
        "<coordinate>559</coordinate><coordinate>113</coordinate><coordinate>90</coordinate>",
        "<coordinate>[559, 113]</coordinate><coordinate>[80, 90]</coordinate>",
        "<coordinate>NaN</coordinate><coordinate>113</coordinate>",
        "<coordinate>559</coordinate><coordinate>1088</coordinate>",
        "<coordinate>(559, 113</coordinate>",
        "<coordinate>559, 113)</coordinate>",
        '<keys=["ctrl", 3]>',
        '<keys=["ctrl"]><unexpected>113</unexpected>',
    ],
)
def test_glm_projection_keeps_incomplete_ambiguous_and_invalid_fields_rejected(tmp_path, fields):
    content = (
        "<tool_call><function=computer_use><action>left_click</action>"
        f"{fields}</function></tool_call>"
    )
    agent = make_agent(tmp_path, FakeProviderResponse(content, ""), "glm53flash")
    response = agent.call_llm({"messages": []}, agent.model)
    _, actions = agent._parse_response(
        response,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )
    assert agent.last_provider_content == content
    assert classify(agent.last_projected_content, actions).kind == "action_error"


@pytest.mark.parametrize(
    "text",
    [
        '<keys=["ctrl"]>',
        "<coordinate>559</coordinate><coordinate>113</coordinate>",
        "<li><function=computer_use><action>left_click</action>"
        "<coordinate>50, 70</coordinate></function></li>",
        "<table>literal HTML text</table>",
    ],
)
def test_glm_projection_preserves_tag_like_type_text(tmp_path, text):
    content = (
        "<tool_call><function=computer_use><action>type</action>"
        f"<text>{text}</text></function></tool_call>"
    )
    agent = make_agent(tmp_path, FakeProviderResponse(content, ""), "glm53flash")
    agent.last_provider_content = content
    _, actions = agent._parse_response(
        content,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )
    assert f"<parameter=text>{text}</parameter>" in agent.last_projected_content
    assert classify(agent.last_projected_content, actions).kind == "action"


@pytest.mark.parametrize("profile", ["glm53flash", "qwen"])
def test_turn_feedback_is_bound_to_current_glm_observation_and_logged(tmp_path, profile):
    captured = []
    agent = make_agent(tmp_path, FakeProviderResponse("", ""), profile)
    agent.client_factory = lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **request: captured.append(request) or FakeProviderResponse("", "")
            )
        )
    )
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Original task"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Malformed call"}]},
            {"role": "user", "content": [{"type": "text", "text": "Current screenshot"}]},
        ]
    }
    original = json.loads(json.dumps(payload))
    agent.turn_feedback = "No action was executed. Budget: 2 model turns remain."
    agent.call_llm(payload, agent.model)
    request = captured[0]
    logged = json.loads((tmp_path / "requests" / "step_000_request.json").read_text())
    assert request["messages"] == logged["messages"]
    assert payload == original
    if profile == "glm53flash":
        assert request["messages"][:-1] == original["messages"][:-1]
        assert request["messages"][-1]["content"][-1]["text"] == agent.turn_feedback
    else:
        assert request["messages"] == original["messages"]
    agent.turn_feedback = ""
    agent.call_llm(payload, agent.model)
    assert captured[1]["messages"] == original["messages"]


def test_glm_does_not_project_incomplete_coordinate_text(tmp_path):
    content = (
        "<tool_call><function=computer_use>"
        "<parameter=action>left_click</parameter>"
        "<parameter=coordinate>559</parameter>"
        "</function></tool_call>"
    )
    agent = make_agent(tmp_path, FakeProviderResponse(content, ""), "glm53flash")
    response = agent.call_llm({"messages": []}, agent.model)
    _, actions = agent._parse_response(
        response,
        original_width=1920,
        original_height=1080,
        processed_width=1920,
        processed_height=1088,
    )

    assert agent.last_projected_content == content
    assert classify(agent.last_projected_content, actions).kind == "action_error"


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


def run_fake_glm_case(
    tmp_path, monkeypatch, turns, evaluation=None, *, setup_error=None, reserve_steps=0
):
    """Exercise the runner without a browser, sandbox, or model API."""
    from playwright import sync_api
    from webarena_verified import types as response_types

    from lite.gym.envs.browsergym import webarena_verified

    class Task:
        def __init__(self, **kwargs):
            pass

        def setup(self, page):
            if setup_error is not None:
                raise setup_error
            self.config = {
                "intent": "Open the page",
                "start_urls": ["http://example.test"],
                "sites": ["wikipedia"],
                "intent_template_id": 1,
            }

    class Page:
        def __init__(self):
            self.waits = []
            self.url = "http://example.test"

        def bring_to_front(self):
            pass

        def wait_for_timeout(self, milliseconds):
            self.waits.append(milliseconds)

    page = Page()
    page.turn_feedback = []
    page.turn_instructions = []
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
            page.turn_feedback.append(self.turn_feedback)
            page.turn_instructions.append(instruction)
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
        lambda *args: evaluation or {"score": 0, "integrity_status": "passed", "publishable": True},
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
        final_response_reserve_steps=reserve_steps,
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
    assert status["fixed_parameters"]["max_steps"] is None
    assert status["error"]["message"] == "GitLab unavailable"
    startup = json.loads((tmp_path / "case" / "startup_receipt.json").read_text())
    assert startup["result"] == "failed"
    assert startup["error"] == "RuntimeError: GitLab unavailable"


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


def test_glm_remaining_budget_is_bound_to_current_observation(tmp_path, monkeypatch):
    _, _, page = run_fake_glm_case(
        tmp_path,
        monkeypatch,
        [(tool_call("screenshot"), ["WAIT"])] * 3 + [('{"status":"SUCCESS"}', ["DONE"])],
        reserve_steps=2,
    )
    assert page.turn_feedback[:2] == ["", ""]
    assert "Budget: 2 model turn(s) remain" in page.turn_feedback[2]
    assert "Budget: 1 model turn(s) remain" in page.turn_feedback[3]
    assert all("Budget:" not in instruction for instruction in page.turn_instructions)


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
    assert page.turn_feedback[0] == ""
    assert "No action was executed." in page.turn_feedback[1]
    assert "Invalid computer_use response" not in page.turn_instructions[1]


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


def test_glm_runner_evidence_disagreement_is_separate_from_model_failure(tmp_path, monkeypatch):
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
