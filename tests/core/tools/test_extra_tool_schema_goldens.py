"""Core extra-tool schema goldens."""

from __future__ import annotations

from lite.core.tools.extra_tools import (
    LiteAppLaunchToolSet,
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
    LiteUserInteractionToolSet,
    make_open_app_tool,
)
from lite.core.tools.schemas import tool_schema_name


def _assert_canonical(schemas):
    assert all(set(schema) == {"type", "function"} for schema in schemas)
    assert all(schema["type"] == "function" for schema in schemas)
    assert all(isinstance(schema["function"], dict) for schema in schemas)


# fmt: off
GOLDEN_BROWSER_NAV = [{'type': 'function',
  'function': {'name': 'goto',
               'description': 'Navigate the browser directly to a URL.',
               'parameters': {'type': 'object',
                              'properties': {'url': {'type': 'string',
                                                     'description': 'The URL to '
                                                                    'navigate to '
                                                                    '(should start '
                                                                    'with https://).'}},
                              'required': ['url']}}},
 {'type': 'function',
  'function': {'name': 'back',
               'description': 'Navigate to the previous page in browser history.',
               'parameters': {'type': 'object', 'properties': {}, 'required': []}}},
 {'type': 'function',
  'function': {'name': 'forward',
               'description': 'Navigate to the next page in browser history.',
               'parameters': {'type': 'object', 'properties': {}, 'required': []}}},
 {'type': 'function',
  'function': {'name': 'new_tab',
               'description': 'Open a new browser tab and make it active.',
               'parameters': {'type': 'object', 'properties': {}, 'required': []}}},
 {'type': 'function',
  'function': {'name': 'switch_tab',
               'description': 'Switch to (activate) the tab at the given index.',
               'parameters': {'type': 'object',
                              'properties': {'index': {'type': 'integer',
                                                       'description': 'Zero-based '
                                                                      'index of the '
                                                                      'tab to '
                                                                      'activate.'}},
                              'required': ['index']}}},
 {'type': 'function',
  'function': {'name': 'close_tab',
               'description': 'Close the current browser tab.',
               'parameters': {'type': 'object', 'properties': {}, 'required': []}}}]

GOLDEN_FINISH = [{'type': 'function',
  'function': {'name': 'response',
               'description': 'Submit a final answer to the task. This is the only '
                              'tool that carries answer\n'
                              'text — use this, not `terminate`, whenever the task '
                              'asks for an answer.',
               'parameters': {'type': 'object',
                              'properties': {'text': {'type': 'string',
                                                      'description': 'Final answer '
                                                                     'text.'}},
                              'required': ['text']}}},
 {'type': 'function',
  'function': {'name': 'terminate',
               'description': 'End the task with no answer text (actions done, or '
                              'infeasible). Does NOT\n'
                              'submit an answer — use `response` whenever the task '
                              'asks for one.',
               'parameters': {'type': 'object',
                              'properties': {'status': {'type': 'string',
                                                        'enum': ['success', 'failure'],
                                                        'description': 'Terminal '
                                                                       'status.'},
                                             'reason': {'type': 'string',
                                                        'description': 'Optional '
                                                                       'reason for '
                                                                       'ending the '
                                                                       'task.'}},
                              'required': ['status']}}}]

GOLDEN_OPEN_APP = [{'type': 'function',
  'function': {'name': 'open_app',
               'description': 'Launch the named app on the device.',
               'parameters': {'type': 'object',
                              'properties': {'app_name': {'type': 'string',
                                                          'description': 'Exact name '
                                                                         'of the app '
                                                                         'to launch.'}},
                              'required': ['app_name']}}}]

GOLDEN_ASK_USER = [{'type': 'function',
  'function': {'name': 'ask_user',
               'description': 'Ask the user for information needed to complete '
                              'the task. Returns their reply.',
               'parameters': {'type': 'object',
                              'properties': {'text': {'type': 'string',
                                                      'description': 'The question '
                                                                     'to ask the user.'}},
                              'required': ['text']}}}]

# The runtime-enum path: four env mains call ``make_open_app_tool(apps)`` to
# stamp the env's real installed-app catalog onto ``app_name``. Pinned
# separately because the enum is appended AFTER ``description``, so the key
# ORDER inside the property is part of the wire bytes.
GOLDEN_OPEN_APP_ENUM = {'type': 'function',
 'function': {'name': 'open_app',
              'description': 'Launch the named app on the device.',
              'parameters': {'type': 'object',
                             'properties': {'app_name': {'type': 'string',
                                                         'description': 'Exact name of '
                                                                        'the app to '
                                                                        'launch.',
                                                         'enum': ['Settings']}},
                             'required': ['app_name']}}}
# fmt: on


def test_browser_nav_all_six_schema_golden():
    schemas = LiteBrowserNavToolSet.get_tool_schemas(
        include=["goto", "back", "forward", "new_tab", "switch_tab", "close_tab"]
    )
    _assert_canonical(schemas)
    assert schemas == GOLDEN_BROWSER_NAV


def test_finish_schema_golden():
    schemas = LiteFinishToolSet.get_tool_schemas(include=["response", "terminate"])
    _assert_canonical(schemas)
    assert schemas == GOLDEN_FINISH
    assert LiteFinishToolSet.get_tool_schemas() == GOLDEN_FINISH


def test_open_app_schema_golden():
    schemas = LiteAppLaunchToolSet.get_tool_schemas(include=["open_app"])
    _assert_canonical(schemas)
    assert schemas == GOLDEN_OPEN_APP
    assert LiteAppLaunchToolSet.get_tool_schemas() == GOLDEN_OPEN_APP


def test_ask_user_schema_golden():
    schemas = LiteUserInteractionToolSet.get_tool_schemas(include=["ask_user"])
    _assert_canonical(schemas)
    assert schemas == GOLDEN_ASK_USER
    assert LiteUserInteractionToolSet.get_tool_schemas() == GOLDEN_ASK_USER
    assert LiteUserInteractionToolSet.ask_user("Which file should I use?") == {
        "type": "function",
        "function": {"name": "ask_user", "arguments": {"text": "Which file should I use?"}},
    }
    assert LiteUserInteractionToolSet.get_tool_names().isdisjoint(
        LiteFinishToolSet.get_tool_names()
    )


def test_open_app_runtime_enum_schema_golden():
    schema = make_open_app_tool(["Settings"])
    _assert_canonical([schema])
    assert schema == GOLDEN_OPEN_APP_ENUM
    assert LiteAppLaunchToolSet.get_tool_schemas() == GOLDEN_OPEN_APP


def test_core_extra_tool_order_locked():
    assert [
        tool_schema_name(t)
        for t in LiteBrowserNavToolSet.get_tool_schemas(
            include=["goto", "back", "forward", "new_tab", "switch_tab", "close_tab"]
        )
    ] == ["goto", "back", "forward", "new_tab", "switch_tab", "close_tab"]
    assert [tool_schema_name(t) for t in LiteFinishToolSet.get_tool_schemas()] == [
        "response",
        "terminate",
    ]
    assert [tool_schema_name(t) for t in LiteAppLaunchToolSet.get_tool_schemas()] == ["open_app"]
