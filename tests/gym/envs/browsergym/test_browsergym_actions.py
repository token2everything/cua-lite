"""BrowserGym action-surface, translation, config, and seed tests."""

from __future__ import annotations

import asyncio
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")

from lite.core.tools.calls import make_tool_call
from lite.core.tools.extra_tools import LiteBrowserNavToolSet
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters
from lite.gym.envs.browsergym.main import (
    _DEFAULT_TASK_SEED,
    BrowserGymConfig,
    BrowserGymEnv,
    _bgym_param_info,
    _format_bgym_call,
    _tool_schema_from_signature,
    _tools_for_subsets,
)
from tests.gym.envs.browsergym._support import _fake_bgym_step_obs, _make_fake, _png_bytes

# ---------------------------------------------------------------------------
# Action surface introspection
# ---------------------------------------------------------------------------


class TestBgymParamInfo:
    def test_known_action_signature(self):
        info = _bgym_param_info("click")
        assert info is not None
        names = [p[0] for p in info]
        # First param of bgym `click` is `bid`.
        assert names[0] == "bid"

    def test_unknown_action(self):
        assert _bgym_param_info("totally_made_up_action") is None

    def test_fill_signature(self):
        info = _bgym_param_info("fill")
        names = [p[0] for p in info]
        assert names[:2] == ["bid", "value"]


class TestToolSchemaFromSignature:
    def test_click_schema(self):
        schema = _tool_schema_from_signature("click", description="click bid")
        assert schema is not None
        assert tool_schema_name(schema) == "click"
        assert schema["function"]["description"] == "click bid"
        parameters = tool_schema_parameters(schema)
        assert "bid" in parameters["properties"]
        assert "bid" in parameters["required"]

    def test_unknown_returns_none(self):
        assert _tool_schema_from_signature("xxx") is None


class TestToolsForSubsets:
    def test_webarena_actions(self):
        # WA preset has 15 actions; we drop noop → 14.
        names = sorted(tool_schema_name(t) for t in _tools_for_subsets(("webarena",)))
        assert "click" in names
        assert "fill" in names
        assert "response" in names
        assert "terminate" in names
        assert "send_msg_to_user" not in names
        assert "report_infeasible" not in names
        assert "noop" not in names
        assert len(names) == 14

    def test_visualwebarena_adds_upload_file(self):
        wa = {tool_schema_name(t) for t in _tools_for_subsets(("webarena",))}
        vwa = {tool_schema_name(t) for t in _tools_for_subsets(("visualwebarena",))}
        assert vwa - wa == {"upload_file"}

    def test_visualwebarena_upload_file_schema_accepts_string_or_list(self):
        from jsonschema import Draft202012Validator, ValidationError

        schema = next(
            t
            for t in _tools_for_subsets(("visualwebarena",))
            if tool_schema_name(t) == "upload_file"
        )
        validator = Draft202012Validator(tool_schema_parameters(schema))
        validator.validate({"bid": "a47", "file": "receipt.pdf"})
        validator.validate({"bid": "a47", "file": ["receipt.pdf", "/tmp/photo.jpg"]})
        with pytest.raises(ValidationError):
            validator.validate({"bid": "a47", "file": 42})

    def test_coord_mode_advertises_canonical_nav(self):
        # Coord/vision mode uses the browser platform's desktop-coordinate
        # action surface (native computer_use for click/scroll/type/key). So
        # _tools_for_subsets:
        #   - drops the bgym coord preset (mouse_*/keyboard_*) — native tool;
        #   - maps send_msg_to_user/report_infeasible to canonical response/terminate;
        #   - KEEPS nav but advertises it with CANONICAL names sourced from
        #     LiteBrowserNavToolSet.get_tool_schemas(include=) (so it's identical to
        #     webgym -> SFT transfer);
        tools = _tools_for_subsets(("coord", "chat", "infeas", "nav", "tab"))
        names = {tool_schema_name(t) for t in tools}
        assert not any(n.startswith("mouse_") for n in names)
        assert not any(n.startswith("keyboard_") for n in names)
        assert {"response", "terminate"} <= names
        assert "send_msg_to_user" not in names
        assert "report_infeasible" not in names
        # Canonical nav (NOT bgym names go_back/tab_focus) is advertised:
        assert {"goto", "back", "forward", "new_tab", "switch_tab", "close_tab"} <= names
        assert not (names & {"go_back", "go_forward", "tab_focus", "tab_close"})

    def test_bid_mode_advertises_canonical_nav(self):
        tools = _tools_for_subsets(("webarena",))
        names = {tool_schema_name(t) for t in tools}
        assert {"response", "terminate"} <= names
        assert "send_msg_to_user" not in names
        assert "report_infeasible" not in names
        assert {"goto", "back", "forward", "new_tab", "switch_tab", "close_tab"} <= names
        assert not (names & {"go_back", "go_forward", "tab_focus", "tab_close"})

    @pytest.mark.parametrize(
        "subsets",
        [("webarena",), ("coord", "chat", "infeas", "nav", "tab")],
        ids=["bid", "coord"],
    )
    def test_terminal_tool_descriptions_end_episode_and_name_a_callable_tool(self, subsets):
        # Model-visible text. Two invariants, both regressions once:
        #   * the "ENDS the episode" hint (added for observed misuse on
        #     GPT-5.4 + WA) must survive the canonical rename -- the terminal
        #     branches return early, so a hint appended after them is dead code;
        #   * BG's examples must not name a function the model cannot call.
        #     ``send_msg_to_user``/``report_infeasible`` are rejected by ``step``
        #     as noncanonical input tools whenever they are not what is
        #     advertised, so a verbatim example is an instruction to make the
        #     one call guaranteed to fail.
        by_name = {tool_schema_name(t): t for t in _tools_for_subsets(subsets)}
        finish = [by_name["response"]]
        # ``report_infeasible`` only when advertised under its bgym name.
        finish.append(by_name.get("terminate") or by_name["report_infeasible"])
        for schema in finish:
            description = schema["function"]["description"]
            assert description.endswith(" Calling this ENDS the episode.")
            assert "Examples: " in description
            examples = description.split("Examples: ", 1)[1]
            for advertised, alias in (
                ("response", "send_msg_to_user"),
                ("terminate", "report_infeasible"),
            ):
                if tool_schema_name(schema) == advertised:
                    assert f"{alias}(" not in examples
                    assert f"{advertised}(" in examples

    def test_coord_nav_schemas_match_single_source(self):
        # The advertised nav schemas are byte-identical to LiteBrowserNavToolSet (single
        # source of truth), so webgym/browsergym vision offer the same nav.
        tools = _tools_for_subsets(("coord", "chat", "infeas", "nav", "tab"))
        nav_names = {"goto", "back", "forward", "new_tab", "switch_tab", "close_tab"}
        for t in tools:
            name = tool_schema_name(t)
            if name in nav_names:
                assert t == LiteBrowserNavToolSet.get_tool_schema(name)

    def test_canonical_nav_is_advertised_in_declaration_order(self):
        """The nav block follows ``LiteBrowserNavToolSet`` declaration order, NOT the
        order BrowserGym's own ``action_set`` iterates.

        ``("webarena",)`` advertises
        ``goto, back, forward, new_tab, switch_tab, close_tab``. Keep this order
        aligned with ``lite/core/tools/extra_tools.py``.
        """
        declared = [tool_schema_name(s) for s in LiteBrowserNavToolSet.get_tool_schemas()]
        for subsets in (("webarena",), ("bid", "nav"), ("coord", "nav")):
            tools = _tools_for_subsets(subsets)
            nav = [tool_schema_name(t) for t in tools if tool_schema_name(t) in set(declared)]
            assert nav == [name for name in declared if name in set(nav)], subsets

    def test_empty_subsets(self):
        assert _tools_for_subsets(()) == []

    def test_unknown_subset_returns_empty(self):
        # Bad key → catches ValueError/KeyError, returns [].
        assert _tools_for_subsets(("not_a_real_subset",)) == []


class TestFormatBgymCall:
    def test_click_bid(self):
        assert _format_bgym_call("click", {"bid": "a47"}) == "click('a47')"

    def test_fill(self):
        assert _format_bgym_call("fill", {"bid": "a47", "value": "hi"}) == "fill('a47', 'hi')"

    def test_goto(self):
        assert (
            _format_bgym_call("goto", {"url": "https://example.com"})
            == "goto('https://example.com')"
        )

    def test_scroll_bid_shape(self):
        # Bgym scroll(delta_x, delta_y) — totally distinct from cua-lite scroll.
        assert _format_bgym_call("scroll", {"delta_x": 0, "delta_y": 100}) == "scroll(0, 100)"

    def test_unknown_action_returns_none(self):
        assert _format_bgym_call("not_a_real_action", {"bid": "a"}) is None

    def test_first_param_missing_returns_none(self):
        # ``click`` requires bid; cua-lite-shape ``click(coordinate=...)`` has no bid
        # → must return None so the coord branch can take over.
        assert _format_bgym_call("click", {"coordinate": [500, 500]}) is None

    def test_partial_args_truncate(self):
        # When a tail param is missing, generation stops at the gap.
        # (Bgym defaults take over inside Python.)
        out = _format_bgym_call("fill", {"bid": "a47"})
        assert out == "fill('a47')"


# ---------------------------------------------------------------------------
# Action translation: CUA-Lite → BrowserGym Python code
# ---------------------------------------------------------------------------


class TestActionTranslation:
    def _translate(self, name: str, args: dict) -> str | None:
        env = _make_fake()
        return env._to_bgym_code(name, args)

    # ─── coord (vision pipeline) path ────────────────────────────────────

    def test_click_left(self):
        code = self._translate("click", {"coordinate": [500, 500]})
        assert "mouse_click" in code
        assert "250.0" in code  # 500/1000*500
        assert "160.0" in code  # 500/1000*320

    def test_click_double(self):
        code = self._translate("click", {"coordinate": [500, 500], "clicks": 2})
        assert "mouse_dblclick" in code

    def test_click_right_button(self):
        code = self._translate("click", {"coordinate": [500, 500], "button": "right"})
        assert '"right"' in code

    def test_mouse_move(self):
        code = self._translate("mouse_move", {"coordinate": [0, 1000]})
        assert "mouse_move" in code
        assert "0.0" in code
        assert "320.0" in code

    def test_mouse_down_uses_cursor_state(self):
        env = _make_fake()
        env._to_bgym_code("mouse_move", {"coordinate": [600, 500]})
        code = env._to_bgym_code("mouse_down", {"button": "left"})
        # 600/1000*500 = 300, 500/1000*320 = 160
        assert "300.0" in code
        assert "160.0" in code
        assert "mouse_down" in code

    def test_mouse_up_uses_cursor_state(self):
        env = _make_fake()
        env._to_bgym_code("mouse_move", {"coordinate": [200, 200]})
        code = env._to_bgym_code("mouse_up", {"button": "left"})
        assert "mouse_up" in code

    def test_type(self):
        code = self._translate("type", {"text": "hello world"})
        assert code == 'keyboard_type("hello world")'

    def test_type_escapes_quotes(self):
        code = self._translate("type", {"text": 'say "hi"'})
        assert '\\"' in code

    # Keys arrive as canonical named/glyph tokens through the browser action-space
    # factory; the env asserts that and projects to Playwright names.
    def test_key(self):
        assert self._translate("key", {"keys": ["ctrl", "c"]}) == 'keyboard_press("Control+c")'

    def test_key_accepts_canonical_glyphs(self):
        assert self._translate("key_down", {"keys": ["+", "-", "="]}) == (
            'keyboard_down("+")\nkeyboard_down("-")\nkeyboard_down("=")'
        )

    def test_key_empty_raises_model_visible_error(self):
        # ``keys`` is required with no default and env ingress does not check
        # argument presence, so an empty list must raise (into ``step``'s
        # ``except MODEL_ACTION_ERROR_TYPES``) rather than translate to ``None``
        # -- a ``None`` here was reported to the model as "unsupported action: key".
        with pytest.raises(ValueError, match="key.keys must not be empty"):
            self._translate("key", {"keys": []})

    @pytest.mark.parametrize(
        ("keys", "expected"),
        [
            (["plus"], "unknown key token 'plus'"),
            ([" "], "unknown key token ' '"),
        ],
    )
    def test_key_rejects_noncanonical_key_tokens(self, keys, expected):
        with pytest.raises(ValueError, match=expected):
            self._translate("key", {"keys": keys})

    def test_key_down(self):
        assert self._translate("key_down", {"keys": ["shift"]}) == 'keyboard_down("Shift")'

    def test_key_down_multiple(self):
        out = self._translate("key_down", {"keys": ["shift", "alt"]})
        assert out == 'keyboard_down("Shift")\nkeyboard_down("Alt")'

    def test_key_down_empty_raises_model_visible_error(self):
        with pytest.raises(ValueError, match="key_down.keys must not be empty"):
            self._translate("key_down", {"keys": []})

    def test_key_up(self):
        assert self._translate("key_up", {"keys": ["shift"]}) == 'keyboard_up("Shift")'

    def test_drag(self):
        code = self._translate(
            "drag",
            {
                "start_coordinate": [0, 0],
                "coordinate": [1000, 1000],
            },
        )
        assert "mouse_drag_and_drop" in code
        assert "0.0" in code
        assert "500.0" in code  # 1000/1000*500
        assert "320.0" in code  # 1000/1000*320

    def test_drag_without_start_uses_cursor_state(self):
        # start_coordinate optional → drag origin is the tracked cursor from a
        # preceding mouse_move (600/1000*500=300, 500/1000*320=160), not [500,500].
        env = _make_fake()
        env._to_bgym_code("mouse_move", {"coordinate": [600, 500]})
        code = env._to_bgym_code("drag", {"coordinate": [1000, 1000]})
        assert code == "mouse_drag_and_drop(300.0, 160.0, 500.0, 320.0)"

    def test_drag_requires_coordinate(self):
        with pytest.raises(KeyError, match="coordinate"):
            self._translate("drag", {"start_coordinate": [0, 0]})

    def test_scroll_down_with_coord(self):
        # coord subset exposes scroll_at(x, y, dx, dy) — NOT bare `scroll`
        # (which NameErrors → silent no-op). With a coord, anchor the wheel there.
        code = self._translate(
            "scroll", {"coordinate": [500, 500], "direction": "down", "amount": 3}
        )
        assert code == "scroll_at(250.0, 160.0, 0, 300)"

    def test_scroll_up_no_coord(self):
        # no coord → anchor at the tracked cursor (fresh env → 0.0, 0.0)
        code = self._translate("scroll", {"direction": "up", "amount": 2})
        assert code == "scroll_at(0.0, 0.0, 0, -200)"

    def test_scroll_left(self):
        code = self._translate("scroll", {"direction": "left", "amount": 1})
        assert code == "scroll_at(0.0, 0.0, -100, 0)"

    def test_scroll_right(self):
        code = self._translate("scroll", {"direction": "right", "amount": 1})
        assert code == "scroll_at(0.0, 0.0, 100, 0)"

    def test_scroll_unknown_direction_falls_back_down(self):
        code = self._translate("scroll", {"direction": "weird", "amount": 1})
        # unknown direction → else branch → dy=+amount*100
        assert code == "scroll_at(0.0, 0.0, 0, 100)"

    def test_back(self):
        assert self._translate("back", {}) == "go_back()"

    def test_goto(self):
        # `goto` has bgym shape (url=...), routes through _format_bgym_call.
        assert (
            self._translate("goto", {"url": "https://example.com"}) == "goto('https://example.com')"
        )

    def test_wait_has_no_browsergym_code(self):
        assert self._translate("wait", {"duration": 1.0}) is None

    def test_screenshot_noop(self):
        assert self._translate("screenshot", {}) is None

    def test_cursor_position_noop(self):
        assert self._translate("cursor_position", {}) is None

    def test_unknown_returns_none_with_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            assert self._translate("totally_unknown", {}) is None
        assert any("Unknown action" in r.message for r in caplog.records)

    # ─── bid (text+AXTree pipeline) path ─────────────────────────────────

    def test_click_bid_routes_to_format_bgym_call(self):
        code = self._translate("click", {"bid": "a47"})
        assert code == "click('a47')"

    def test_fill_bid(self):
        code = self._translate("fill", {"bid": "a47", "value": "hello"})
        assert code == "fill('a47', 'hello')"

    def test_select_option_bid(self):
        code = self._translate("select_option", {"bid": "a47", "options": "FOO"})
        assert "select_option" in code
        assert "'a47'" in code


class TestTypeExecution:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "arguments", [{"press_enter": True}, {"press_enter": False}, {"press_enter": None}, {}],
    )
    async def test_type_executes_optional_enter_as_single_native_calls(
        self, arguments, monkeypatch,
    ):
        from browsergym.core.action.base import execute_python_code
        from browsergym.core.action.highlevel import HighLevelActionSet

        env = _make_fake(cursor=False, multiaction=False)
        try:
            await env.reset()
            action_set = HighLevelActionSet(subsets=["coord"], multiaction=False, strict=True)
            native_calls = []
            keyboard_events = []
            backend_threads = []
            frames = [_png_bytes(4, 4), _png_bytes(5, 5)]
            page = SimpleNamespace(keyboard=SimpleNamespace(
                type=lambda text, delay: keyboard_events.append(("type", text)),
                press=lambda key: keyboard_events.append(("press", key)),
            ))

            def backend_step(code):
                backend_threads.append(threading.get_ident())
                native_calls.append(code)
                execute_python_code(action_set.to_python_code(code), page, None, None)
                obs = _fake_bgym_step_obs(screenshot=frames[len(native_calls) - 1])
                return obs, 0.0, False, False, {}

            env._env = SimpleNamespace(step=backend_step)
            monkeypatch.setattr("lite.gym.envs.browsergym.main._with_playwright", lambda fn: fn())
            result = await env.step([
                make_tool_call("computer", {"actions": [{"action": "type", "text": "technology",
                                                         **arguments}]}, call_id="call_type"),
            ])

            expected_calls = ['keyboard_type("technology")']
            expected_events = [("type", "technology")]
            if arguments.get("press_enter") is True:
                expected_calls.append('keyboard_press("Enter")')
                expected_events.append(("press", "Enter"))
            assert native_calls == expected_calls
            assert keyboard_events == expected_events
            assert all(t != threading.get_ident() for t in backend_threads)
            assert result.info["executed_actions"] == [{"call": c} for c in expected_calls]
            assert len(result.results) == 1
            assert result.results[0].tool_call_id == "call_type"
            assert result.results[0].images == [frames[len(expected_calls) - 1]]
            assert result.results[0].error is None
            assert env._step_count == 1
        finally:
            env._env = None
            await env.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("typing_result", [None, {"last_action_error": "typing failed"}])
    async def test_typing_failure_does_not_send_enter(self, typing_result):
        env = _make_fake(use_screenshot=False)
        try:
            await env.reset()

            async def execute(code):
                if typing_result is None:
                    env._last_action_execution_error = "typing failed"
                return typing_result

            env._execute_bgym_action = AsyncMock(side_effect=execute)
            result = await env.step([
                make_tool_call("computer", {"actions": [{"action": "type", "text": "technology",
                                                         "press_enter": True}]},
                               call_id="call_type"),
            ])

            env._execute_bgym_action.assert_awaited_once_with('keyboard_type("technology")')
            assert result.info["executed_actions"] == [{"call": 'keyboard_type("technology")'}]
            assert result.results[0].tool_call_id == "call_type"
            assert result.results[0].metadata == {"is_error": True}
            assert result.results[0].error == "type failed: execution failed"
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_enter_failure_is_visible_under_the_type_call_id(self):
        env = _make_fake(use_screenshot=False)
        try:
            await env.reset()
            env._execute_bgym_action = AsyncMock(side_effect=[
                _fake_bgym_step_obs(), {"last_action_error": "Enter failed"},
            ])
            result = await env.step([
                make_tool_call("computer", {"actions": [{"action": "type", "text": "technology",
                                                         "press_enter": True}]},
                               call_id="call_type"),
            ])

            assert env._execute_bgym_action.await_count == 2
            assert result.results[0].tool_call_id == "call_type"
            assert result.results[0].metadata == {"is_error": True}
            assert result.results[0].error == "type failed: execution failed"
            assert env._step_count == 1
        finally:
            await env.close()


class TestWaitExecution:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("arguments", "seconds"),
        [({"duration": 2.5}, 2.5), ({}, 1.0), ({"duration": 0}, 0.0)],
    )
    async def test_wait_finishes_before_its_frame(self, arguments, seconds, monkeypatch):
        env = _make_fake(cursor=False)
        try:
            await env.reset()
            env._env = SimpleNamespace()
            env._execute_bgym_action = AsyncMock()
            events = []
            real_sleep = asyncio.sleep
            frame = _png_bytes()

            async def sleep(duration):
                events.append(("wait_started", duration))
                await real_sleep(0)
                events.append(("wait_finished", duration))

            async def screenshot():
                events.append(("frame", None))
                return frame

            sleeper = AsyncMock(side_effect=sleep)
            monkeypatch.setattr("lite.gym.envs.browsergym.main.asyncio.sleep", sleeper)
            env._take_screenshot = AsyncMock(side_effect=screenshot)
            result = await env.step([
                make_tool_call("computer", {"actions": [{"action": "wait", **arguments}]},
                               call_id="call_wait"),
            ])

            sleeper.assert_awaited_once_with(seconds)
            assert events == [
                ("wait_started", seconds), ("wait_finished", seconds), ("frame", None),
            ]
            assert result.results[0].images == [frame]
            assert result.results[0].error is None
            env._execute_bgym_action.assert_not_awaited()
        finally:
            env._env = None
            await env.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("duration", [-1, 31, True, float("nan")])
    async def test_invalid_wait_is_visible_to_model_without_sleep(self, duration, monkeypatch):
        env = _make_fake(cursor=False)
        try:
            await env.reset()
            env._execute_bgym_action = AsyncMock()
            sleeper = AsyncMock()
            monkeypatch.setattr("lite.gym.envs.browsergym.main.asyncio.sleep", sleeper)
            result = await env.step([
                make_tool_call("computer", {"actions": [{"action": "wait", "duration": duration}]},
                               call_id="call_wait"),
            ])

            sleeper.assert_not_awaited()
            env._execute_bgym_action.assert_not_awaited()
            assert result.results[0].tool_call_id == "call_wait"
            assert "wait.duration" in result.results[0].error
            assert result.results[0].metadata == {"is_error": True}
        finally:
            await env.close()


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestBrowserGymConfig:
    def test_defaults(self):
        cfg = BrowserGymConfig()
        assert cfg.use_screenshot is True
        assert cfg.use_ax_tree is False
        assert cfg.use_focused_element is False
        assert cfg.action_subsets == ("coord", "chat", "infeas", "nav", "tab")
        assert cfg.headless is True
        # extract_coords is a Literal["False","center","box"] string, not bool.
        assert cfg.extract_coords == "False"

    def test_text_only_override(self):
        cfg = BrowserGymConfig(use_screenshot=False, use_ax_tree=True, action_subsets=["bid"])
        assert cfg.use_screenshot is False
        assert cfg.use_ax_tree is True
        assert cfg.action_subsets == ["bid"]

    def test_env_seed_default_none_and_override(self):
        # Seed lives on the env (``self._seed``), not the config — matching the
        # cua-lite convention (androidworld/mobilegym). Bare construction is
        # unseeded (legacy RandomState(None)); registration supplies the fixed seed.
        config = BrowserGymConfig(bgym_task_id="miniwob.click-dialog", benchmark="miniwob")
        assert BrowserGymEnv(config=config, use_fake=True)._seed is None
        assert BrowserGymEnv(config=config, use_fake=True, seed=5)._seed == 5


class TestSeedRegistration:
    """Registration-time fixed seed (reproducible task instances).

    Mirrors the cua-lite eval convention: a fixed ``seed`` is forwarded as a
    ``register(seed=...)`` kwarg → ``BrowserGymEnv(seed=...)`` → ``self._seed``
    (androidworld/mobilegym do the same). The seed threads into
    ``env.reset(seed=...)`` → BrowserGym's ``task_entrypoint(seed=...)``, the
    sole randomness source.
    """

    def _make(self, task: str = "click-checkboxes", **kw) -> BrowserGymEnv:
        # Force LOCAL construction. gym.make routes to a remote LiteEnvClient
        # (which has no ``_seed``) whenever CUA_LITE_ENV_SERVER_URL is set —
        # via an ``or``-chain that a kwarg can't override — so these
        # registration-path tests would otherwise depend on shell state. The
        # var only matters during make(); pop it for the call, restore after.
        import lite.gym as gym

        with patch.dict(os.environ):
            os.environ.pop("CUA_LITE_ENV_SERVER_URL", None)
            return gym.make(f"browsergym.miniwob@{task}", use_fake=True, **kw)

    @pytest.mark.asyncio
    async def test_registered_env_seed_is_fixed(self):
        env = self._make()
        try:
            assert env._seed == _DEFAULT_TASK_SEED
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_yaml_seed_override_threads_to_env(self):
        env = self._make(seed=123)
        try:
            assert env._seed == 123
        finally:
            await env.close()
