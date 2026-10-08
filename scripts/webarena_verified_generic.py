#!/usr/bin/env python3
"""Run one fresh WebArena-Verified desktop case with the native generic agent.

Example (the coordinator supplies a fresh baseline and private model environment):
    uv run python -m scripts.webarena_verified_generic --task-id 0 \
        --osworld-root /opt/OSWorld --baseline-receipt /results/baseline.json \
        --output-dir /results/task-0000 --lock-dir /results/locks \
        --model-profile qwen --startup-page-gate

OPENAI_API_KEY / OPENAI_BASE_URL configure the Chat-compatible transport. The
pinned OSWorld checkout supplies cursor capture only, never the decision agent.
The inherited AdapterBasedAgent.sample owns every model turn. This entry owns
the full-desktop environment, terminal answer contract, and official scoring.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import importlib.util
import inspect
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from lite.agents.core.agent.base import AdapterBasedAgent
from lite.agents.core.agent.hooks import SampleHook, SampleStepData
from lite.agents.core.agent.jsonl_logger import JsonlTrajectoryLogger
from lite.agents.core.agent.logger import TrajectoryLogger
from lite.agents.core.agent.utils.request_logs import redact
from lite.agents.extensions.chat_gateway import ChatGatewayAgent, ChatGatewayProtocolError
from lite.agents.extensions.chat_gateway import prompt as chat_prompt
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER,
    LiteDesktopActionSet,
)
from lite.core.tools.calls import canonical_lite_tool_call, runtime_rejected_reason
from lite.core.tools.results import make_tool_result
from lite.gym.base import LiteBaseEnv
from lite.gym.envs.osworld.actions import to_pyautogui
from lite.gym.types import EXECUTED_ACTIONS_INFO_KEY, LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.backend.model_inputs import (
    coerce_model_duration,
    coerce_model_numeric,
    project_model_keys,
)
from lite.gym.utils.feedback.errors import MODEL_ACTION_ERROR_TYPES
from lite.gym.utils.feedback.ingress import prepare_env_tool_calls
from lite.gym.utils.feedback.results import ordered_tool_call_ids
from lite.utils.image import encode_png

_DESKTOP_SPEC = importlib.util.spec_from_file_location(
    "webarena_desktop_helpers", Path(__file__).with_name("webarena_verified_qwen.py")
)
if _DESKTOP_SPEC is None or _DESKTOP_SPEC.loader is None:
    raise ImportError("Cannot load sibling WebArena desktop/scoring helpers")
desktop = importlib.util.module_from_spec(_DESKTOP_SPEC)
_DESKTOP_SPEC.loader.exec_module(desktop)

GUI_ACTIONS = [
    name for name in LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER if name != "cursor_position"
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument(
        "--osworld-root", required=True, type=Path, help="Pinned cursor source only."
    )
    parser.add_argument("--baseline-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lock-dir", required=True, type=Path)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--display", default=":99")
    parser.add_argument("--model-profile", choices=("qwen", "glm53flash"), default="qwen")
    parser.add_argument("--model-id")
    parser.add_argument("--reasoning-effort", choices=("xhigh", "max"))
    parser.add_argument("--max-steps", type=int, default=desktop.MAX_STEPS)
    parser.add_argument(
        "--final-response-reserve-steps", type=int, default=desktop.FINAL_RESPONSE_RESERVE_STEPS
    )
    parser.add_argument(
        "--repair-invalid-response", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--startup-page-gate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if not 0 <= args.final_response_reserve_steps < args.max_steps:
        parser.error("--final-response-reserve-steps must be in [0, max_steps)")
    if args.model_id is None:
        args.model_id = desktop.GLM_MODEL if args.model_profile == "glm53flash" else desktop.MODEL
    if args.reasoning_effort is None:
        args.reasoning_effort = "max" if args.model_profile == "glm53flash" else "xhigh"
    if args.api_key is None:
        key_name = "GLM_TEST_API_KEY" if args.model_profile == "glm53flash" else "QWEN_TEST_API_KEY"
        args.api_key = os.environ.get(key_name)
    return args


class WebArenaFullDesktopEnv(LiteBaseEnv):
    """One fresh site baseline and headed browser on a 1920x1080 desktop.

    Sync Playwright, PyAutoGUI, and their cleanup stay on the same dedicated
    thread. Step accounting is per model decision, not per child GUI action.
    Rejected final answers return visible feedback through native result IDs.
    """

    def __init__(self, args: argparse.Namespace, output: Path):
        self.args, self.output = args, output
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wav-desktop")
        self._step_count = 0
        self._opened = self._finished = self._closed = False
        self._browser_finalized = False
        self._xvfb = self._cwd = self._playwright = self._browser = self._context = None
        self._page = self._task = self._pyautogui = self._cursor_class = None
        self._response_model = None
        self._image = None
        self._tracing_started = False
        self.final_content = ""
        self.response_parse_error = None
        self.response_repair_used = False
        self._repair_pending = False
        self.finalization_attempts = 0
        self.model_protocol_error = None
        self.model_protocol_failure_phase = None
        self.model_protocol_error_type = None
        self.truncated = False
        self.terminal_action = None
        self.startup_failure = False
        self.evaluation = None
        self.close_error = None
        self._extra_schemas = self.extra_tool_schemas(["response"])

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(
            dims=("desktop", "use"),
            valid_actions=GUI_ACTIONS,
            extra_tool_schemas=self._extra_schemas,
            others={"benchmark": "webarena-verified", "task_id": str(self.args.task_id)},
        )

    async def reset(self) -> LiteEnvObservation:
        """Consume the coordinator's baseline and open authenticated task pages."""
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._reset_sync)

    def _reset_sync(self) -> LiteEnvObservation:
        if self._opened or self._closed:
            raise RuntimeError("A fresh environment instance is required for every case")
        self._xvfb = desktop.xvfb(self.args.display)
        self._xvfb.__enter__()
        import pyautogui
        from browsergym.webarena_verified.task import WebArenaVerifiedTask
        from playwright.sync_api import sync_playwright
        from webarena_verified.types import FinalAgentResponse

        self._pyautogui = pyautogui
        desktop.patch_pyautogui(pyautogui)
        self._cursor_class = desktop.load_cursor_class(self.args.osworld_root)
        self._response_model = FinalAgentResponse
        self._cwd = desktop.working_directory(self.output)
        self._cwd.__enter__()
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=False,
            args=[
                f"--window-size={desktop.SCREEN_WIDTH},{desktop.SCREEN_HEIGHT}",
                "--start-maximized",
            ],
        )
        self._context = self._browser.new_context(
            no_viewport=True,
            record_har_path=str(self.output / "network.har"),
            record_har_content="embed",
        )
        self._page = self._context.new_page()
        self._task = WebArenaVerifiedTask(seed=0, task_id=self.args.task_id)
        navigation_responses = []

        def record_navigation(response: Any) -> None:
            if not response.request.is_navigation_request():
                return
            try:
                page = response.frame.page
                if response.frame is not page.main_frame:
                    return
            except Exception:
                page = None
            navigation_responses.append((page, response.url, response.status))

        if self.args.startup_page_gate:
            self._context.on("response", record_navigation)
        try:
            # Official setup authenticates, starts tracing, and opens start URLs.
            self._tracing_started = True
            self._task.setup(self._page)
            if self.args.startup_page_gate:
                self._page = desktop.select_healthy_start_page(
                    self._context,
                    list(self._task.config.get("start_urls") or []),
                    navigation_responses,
                    self.output / "startup_receipt.json",
                )
            else:
                self._page = self._context.pages[-1]
        except BaseException as error:
            self.startup_failure = True
            if not (self.output / "startup_receipt.json").exists():
                desktop._atomic_json(
                    self.output / "startup_receipt.json",
                    {"result": "failed", "phase": "task.setup", "error": str(error)},
                )
            raise
        finally:
            if self.args.startup_page_gate:
                self._context.remove_listener("response", record_navigation)
        schema = FinalAgentResponse.model_json_schema()
        instruction = desktop.build_public_instruction(self._task.config, schema)
        desktop._atomic_text(self.output / "instruction.txt", instruction + "\n")
        desktop._atomic_json(
            self.output / "instruction.json",
            {
                "intent": self._task.config["intent"],
                "start_urls": self._task.config.get("start_urls") or [],
                "final_response_schema": schema,
            },
        )
        self._page.bring_to_front()
        self._page.wait_for_timeout(int(desktop.RESET_SETTLE_SECONDS * 1000))
        self._opened = True
        self._image = self._capture()
        return LiteEnvObservation(image=self._image, text=instruction)

    def _capture(self) -> bytes:
        image = desktop.screenshot_bytes(self._pyautogui, self._cursor_class)
        desktop._atomic_bytes(self.output / "desktop" / f"step_{self._step_count:03d}.png", image)
        return image

    def _dispatch_gui(self, name: str, arguments: dict[str, Any]) -> None:
        # Durations and keys are checked/projected by their existing owners.
        if name in {"wait", "hold_key"}:
            duration = coerce_model_duration(arguments.get("duration", 1), action_name=name)
            keys = (
                project_model_keys(arguments.get("keys", []), action_name=name, backend="pyautogui")
                if name == "hold_key"
                else []
            )
            for key in keys:
                self._pyautogui.keyDown(key)
            try:
                self._page.wait_for_timeout(math.ceil(duration * 1000))
            finally:
                for key in reversed(keys):
                    self._pyautogui.keyUp(key)
            return
        if name == "scroll":
            arguments = {
                **arguments,
                "amount": coerce_model_numeric(
                    arguments.get("amount", 3),
                    field="amount",
                    action_name="scroll",
                    min_value=0,
                    integer=True,
                ),
            }
        if name == "click" and arguments.get("coordinate") is None:
            command = (
                f"pyautogui.click(button={arguments.get('button', 'left')!r}, "
                f"clicks={arguments.get('clicks', 1)!r}, interval=0.1)"
            )
        else:
            command = to_pyautogui(name, arguments, desktop.SCREEN_WIDTH, desktop.SCREEN_HEIGHT)
        if command is None and name not in {"screenshot", "type"}:
            raise ValueError(f"unsupported desktop action: {name}")
        # The canonical projector may emit several literal PyAutoGUI calls.
        # Execute each through the same AST allowlist, never model source/exec.
        if command:
            for expression in ast.parse(command, mode="exec").body:
                desktop.execute_safe_pyautogui(ast.unparse(expression), self._pyautogui)
        if name == "type" and arguments.get("press_enter"):
            self._pyautogui.press("enter")

    async def step(self, actions: list[dict[str, Any]]) -> LiteEnvStepResult:
        """Execute native calls, or return one honest final-answer repair turn."""
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._step_sync, actions
        )

    def _step_sync(self, actions: list[dict[str, Any]]) -> LiteEnvStepResult:
        if not self._opened or self._finished or self._closed:
            raise RuntimeError("reset must precede step; terminal episodes cannot continue")
        remaining = self.args.max_steps - self._step_count
        call_ids = ordered_tool_call_ids(actions)
        projected, ingress_errors = prepare_env_tool_calls(
            actions, self.metadata, validate_top_level_action=True
        )
        errors = {call_id: feedback.message for call_id, feedback in ingress_errors.items()}
        executed = []
        feedback_text = None
        terminated = False
        # A reserve-window proposal is one model turn, even if it contains
        # multiple calls/child actions. No GUI operation from it is executed.
        reserve_blocked = remaining <= self.args.final_response_reserve_steps and any(
            action["name"] != "response" for action, _ in projected
        )
        has_response = any(action["name"] == "response" for action, _ in projected)
        mixed_final = has_response and any(action["name"] != "response" for action, _ in projected)
        if has_response:
            self.final_content = next(
                action["arguments"]["text"]
                for action, _ in projected
                if action["name"] == "response"
            )
        self._step_count += 1
        if self._repair_pending and (
            not projected or any(action["name"] != "response" for action, _ in projected)
        ):
            self._repair_pending = False
            self.model_protocol_error = "final repair did not return a final JSON response"
            self.response_parse_error = self.model_protocol_error
            errors.update(
                {call_id: self.model_protocol_error for call_id in call_ids if call_id is not None}
            )
            terminated = True
        elif mixed_final:
            self.response_parse_error = (
                "A final response must not share a model turn with GUI actions; "
                "no action from this turn was executed"
            )
            if (
                self.args.repair_invalid_response
                and not self.response_repair_used
                and remaining > 1
            ):
                self.response_repair_used = self._repair_pending = True
                feedback_text = (
                    f"{self.response_parse_error}. This is the one repair attempt. "
                    "Return only a fenced JSON object matching final_response_schema, "
                    "using facts already verified from the visible UI."
                )
            else:
                self.model_protocol_error = self.response_parse_error
                terminated = True
            errors.update(
                {call_id: self.response_parse_error for call_id in call_ids if call_id is not None}
            )
        elif reserve_blocked:
            self.finalization_attempts += 1
            feedback_text = (
                "The GUI action budget is exhausted; no action from this turn was executed. "
                "Submit the final fenced JSON now, using only facts established in the visible UI. "
                "Report an honest failure/not-found outcome if the task is incomplete."
            )
            errors.update({call_id: feedback_text for call_id in call_ids if call_id is not None})
            if self.finalization_attempts >= 2:
                self.truncated = True
                self.response_parse_error = (
                    "final response was not submitted after one reserve-window retry"
                )
        else:
            for action, call_id in projected:
                if terminated:
                    if call_id is not None:
                        errors[call_id] = "Not executed: an earlier tool call ended the episode"
                    continue
                name, arguments = action["name"], action["arguments"]
                if rejection := runtime_rejected_reason(action):
                    if call_id is not None:
                        errors[call_id] = rejection
                    continue
                if name == "response":
                    self.final_content = arguments["text"]
                    try:
                        desktop.normalize_agent_response(self.final_content, self._response_model)
                    except (json.JSONDecodeError, TypeError, ValueError) as error:
                        self.response_parse_error = str(error)
                        if (
                            self.args.repair_invalid_response
                            and not self.response_repair_used
                            and remaining > 1
                        ):
                            self.response_repair_used = True
                            self._repair_pending = True
                            feedback_text = (
                                f"Your final response was invalid: {error}. This is the one repair "
                                "attempt. Return one fenced JSON object matching "
                                "final_response_schema. "
                                "Do not call computer or invent facts; report an honest outcome."
                            )
                            if call_id is not None:
                                errors[call_id] = feedback_text
                        else:
                            self._repair_pending = False
                            self.model_protocol_error = self.response_parse_error
                            terminated = True
                    else:
                        self._repair_pending = False
                        self.response_parse_error = None
                        self.terminal_action = "FINAL_JSON"
                        executed.append({"call": "response", "args": arguments})
                        terminated = True
                    continue
                try:
                    self._dispatch_gui(name, arguments)
                except MODEL_ACTION_ERROR_TYPES as error:
                    if call_id is not None:
                        previous = errors.get(call_id)
                        errors[call_id] = f"{previous}\n{error}" if previous else str(error)
                    executed.append({"call": "noop", "args": {"name": name, "reason": str(error)}})
                    continue
                executed.append({"call": name, "args": arguments})
                # Pump Playwright while settling so network evidence is flushed.
                self._page.wait_for_timeout(int(desktop.POST_ACTION_DELAY_SECONDS * 1000))
        self.truncated = self.truncated or (
            not terminated and self._step_count >= self.args.max_steps
        )
        self._finished = terminated or self.truncated
        if terminated and self.model_protocol_error:
            self.model_protocol_failure_phase = "final_response"
        if self._finished:
            if self.truncated:
                self.final_content = ""
            self._finish_sync()
        else:
            self._image = self._capture()
            next_remaining = self.args.max_steps - self._step_count
            if next_remaining <= self.args.final_response_reserve_steps:
                budget = (
                    f"Budget: {next_remaining} model turns remain, including the next turn. "
                    "The GUI budget is exhausted. Submit an honest final JSON response now."
                )
                feedback_text = f"{feedback_text}\n\n{budget}" if feedback_text else budget
        info = {
            EXECUTED_ACTIONS_INFO_KEY: executed,
            "model_turn": self._step_count,
            "reserve_blocked": reserve_blocked,
            "response_repair_used": self.response_repair_used,
        }
        results = [
            make_tool_result(
                call_id,
                images=[self._image] if not self._finished and index == len(call_ids) - 1 else [],
                text=feedback_text,
                error=errors.get(call_id) if call_id is not None else self.response_parse_error,
            )
            for index, call_id in enumerate(call_ids)
        ]
        return LiteEnvStepResult(
            results=results,
            terminated=terminated,
            truncated=self.truncated,
            reward=float(self.evaluation["score"]) if self.evaluation is not None else None,
            info=info,
        )

    def _finish_sync(self) -> None:
        if self._browser_finalized:
            return
        # Raw content, rather than a normalized/reconstructed answer, is scored.
        desktop._atomic_text(self.output / "agent_response_raw.txt", self.final_content)
        response = {
            "raw_content": self.final_content,
            "valid": self.response_parse_error is None and not self.truncated,
        }
        if self.response_parse_error:
            response["error"] = self.response_parse_error
        if response["valid"]:
            response["validated_projection"], _ = desktop.normalize_agent_response(
                self.final_content, self._response_model
            )
        desktop._atomic_json(self.output / "agent_response.json", response)
        self._browser_finalized = True
        error = desktop.finalize_browser_artifacts(
            self._context,
            self._browser,
            self.output / "trace.zip",
            tracing_started=self._tracing_started,
        )
        if error:
            raise error
        if self._task is not None and not self.startup_failure:
            self.evaluation = desktop.score_artifacts(
                self._task,
                self.final_content,
                self.output / "trace.zip",
                self.output / "network.har",
            )
            self.evaluation["input_receipt"] = {
                "agent_response_raw": desktop.content_identity(self.final_content),
                "network_har": desktop.artifact_identity(self.output / "network.har"),
                "playwright_trace": desktop.artifact_identity(self.output / "trace.zip"),
            }
            if self.args.startup_page_gate:
                self.evaluation["input_receipt"]["startup_receipt"] = desktop.artifact_identity(
                    self.output / "startup_receipt.json"
                )
            desktop._atomic_json(self.output / "evaluation.json", self.evaluation)

    async def close(self) -> None:
        """Flush artifacts and release only the processes owned by this case."""
        if self._closed:
            return
        try:
            await asyncio.get_running_loop().run_in_executor(self._executor, self._close_sync)
        finally:
            self._executor.shutdown(wait=True)

    def _close_sync(self) -> None:
        error = None
        try:
            if self._opened and not self._finished:
                # A provider failure may occur before env.step. Preserve the
                # real empty/unsubmitted answer, flush both network inputs, and
                # evaluate diagnostically before the native loop closes us.
                # The entry later distinguishes typed model-output failure from
                # transport failure; only complete model evidence is terminal.
                self.final_content = ""
                self.response_parse_error = "episode aborted before a final response"
                self._finished = True
                self._finish_sync()
            elif not self._browser_finalized and self._context is not None:
                self._browser_finalized = True
                error = desktop.finalize_browser_artifacts(
                    self._context,
                    self._browser,
                    self.output / "trace.zip",
                    tracing_started=self._tracing_started,
                )
        except BaseException as caught:
            error = caught
        finally:
            for resource, method in (
                (self._playwright, "stop"),
                (self._cwd, "__exit__"),
                (self._xvfb, "__exit__"),
            ):
                if resource is not None:
                    try:
                        getattr(resource, method)(*([] if method == "stop" else [None, None, None]))
                    except BaseException as caught:
                        error = error or caught
            self._closed = True
            self.close_error = error
        if error:
            raise error


class CaseTrajectoryLogger(SampleHook):
    """Keep decision-oriented campaign JSONL beside the full native trajectory."""

    def __init__(self, output: Path, secrets: tuple[str, ...]):
        self.output, self.secrets = output, secrets

    def on_step(self, data: SampleStepData) -> None:
        screenshot = f"desktop/step_{data.step_idx:03d}.png"
        if data.image is not None:
            desktop._atomic_bytes(self.output / screenshot, encode_png(data.image))
        message = data.predict_result.lite_message
        desktop.append_trajectory(
            self.output / "trajectory.jsonl",
            redact(
                {
                    "step": data.step_idx,
                    "screenshot": screenshot,
                    "provider_content": "\n".join(
                        part["text"]
                        for part in message.get("content", [])
                        if part.get("type") == "text"
                    ),
                    "actions": [canonical_lite_tool_call(action) for action in data.actions],
                    "executed_actions": data.step_result.info.get(EXECUTED_ACTIONS_INFO_KEY, []),
                    "feedback": [
                        {
                            "tool_call_id": result.tool_call_id,
                            "text": result.text,
                            "error": result.error,
                        }
                        for result in data.step_result.results
                    ],
                    "timings": data.timings,
                    "elapsed_seconds": sum((data.timings or {}).values()),
                    "terminated": data.step_result.terminated,
                    "truncated": data.step_result.truncated,
                    "reserve_blocked": data.step_result.info.get("reserve_blocked", False),
                },
                self.secrets,
            ),
        )


async def run_case(args: argparse.Namespace) -> dict[str, Any]:
    """Bind the baseline, run native sampling, and publish scoring-bound status."""
    from lite.gym.envs.browsergym import webarena_verified

    records = webarena_verified.task_records()
    if not 0 <= args.task_id < len(records):
        raise ValueError(f"task ID must be in 0..{len(records) - 1}")
    receipt = desktop.validate_baseline_receipt(args.baseline_receipt, args.task_id)
    args.osworld_root = desktop.validate_osworld_checkout(args.osworld_root)
    if not args.api_key:
        raise ValueError("OPENAI_API_KEY or the selected model test key is required")
    base = desktop.normalize_api_base(args.base_url)
    with desktop.sandbox_lock(args.lock_dir.resolve(), receipt["sandbox_id"]):
        output = desktop.prepare_output_directory(args.output_dir)
        desktop._atomic_json(output / "baseline_receipt.json", receipt)
        metadata = desktop.runtime_metadata(webarena_verified.DATASET_SHA256)
        metadata.update(
            script=str(Path(__file__).resolve()),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            harness="native_chat_gateway",
            loop="lite.agents.core.agent.base.AdapterBasedAgent.sample",
        )
        dependencies = {
            "agent": Path(inspect.getfile(ChatGatewayAgent)),
            "loop": Path(inspect.getfile(AdapterBasedAgent)),
            "prompt": Path(chat_prompt.__file__),
            "action_space": Path(inspect.getfile(LiteDesktopActionSet)),
            "env_ingress": Path(inspect.getfile(prepare_env_tool_calls)),
            "desktop_action_projection": Path(inspect.getfile(to_pyautogui)),
            "model_input_projection": Path(inspect.getfile(coerce_model_duration)),
            "desktop_scoring_helpers": Path(desktop.__file__),
            "catalog_trace_loader": Path(webarena_verified.__file__),
            "cursor": args.osworld_root / "desktop_env/server/pyxcursor.py",
        }
        metadata["dependencies"] = {
            owner: {"module_file": str(path.resolve()), **desktop.artifact_identity(path)}
            for owner, path in dependencies.items()
        }
        desktop._atomic_json(output / "run_metadata.json", metadata)
        desktop._atomic_json(output / "status.json", {"task_id": args.task_id, "status": "running"})
        env = WebArenaFullDesktopEnv(args, output)
        secrets = (args.api_key, base)
        error = None
        sample = None
        try:
            agent = ChatGatewayAgent(
                model_id=args.model_id,
                api_key=args.api_key,
                api_base=base,
                metadata=env.metadata,
                request_params={
                    "max_tokens": desktop.MAX_TOKENS,
                    "enable_thinking": True,
                    "preserve_thinking": True,
                    "reasoning_effort": args.reasoning_effort,
                },
                history_turns=desktop.IMAGE_MAX - 1,
                image_fold_size=desktop.FOLD_SIZE,
                request_timeout=httpx.Timeout(desktop.MODEL_TIMEOUT_SECONDS, read=600.0),
                api_retry_max=desktop.MODEL_ATTEMPTS - 1,
                request_log_dir=output / "requests",
                request_image_dir=output / "images",
            )
            hooks = [
                JsonlTrajectoryLogger(
                    output, model_id=args.model_id, secrets=secrets, get_tools=agent._build_tools
                ),
                TrajectoryLogger(output / "native", save_data=False, debug_artifacts=True),
                CaseTrajectoryLogger(output, secrets),
            ]
            # No custom model loop, XML parser, or QwenAgent is invoked here.
            sample = await agent.sample(env, max_steps=args.max_steps, hooks=hooks)
        except ChatGatewayProtocolError as caught:
            env.model_protocol_error = redact(str(caught), secrets)
            env.response_parse_error = env.model_protocol_error
            env.model_protocol_failure_phase = "provider_projection"
            env.model_protocol_error_type = type(caught).__name__
            prefix = f"{agent._request_count:05d}"
            # The canonical projection failed, so no assistant call or GUI
            # action can be invented. The unmodified successful wire body is
            # already archived by ChatGateway; this row binds that decision to
            # the otherwise-complete native case evidence without retrying it.
            desktop.append_trajectory(
                output / "trajectory.jsonl",
                {
                    "step": env._step_count,
                    "screenshot": f"desktop/step_{env._step_count:03d}.png",
                    "provider_content": None,
                    "prediction_failed": True,
                    "raw_response": f"requests/{prefix}.response.json",
                    "actions": [],
                    "executed_actions": [],
                    "feedback": [],
                    "timings": {},
                    "reserve_blocked": False,
                    "error": {"type": type(caught).__name__, "message": env.model_protocol_error},
                    "terminated": True,
                    "truncated": False,
                    "failure_phase": "provider_projection",
                },
            )
            desktop._atomic_json(
                output / "agent_response.json",
                {
                    "raw_content": "",
                    "valid": False,
                    "protocol_failure": True,
                    "error": env.model_protocol_error,
                    "raw_response": f"requests/{prefix}.response.json",
                },
            )
        except BaseException as caught:
            error = caught
        finally:
            try:
                await env.close()
            except BaseException as caught:
                error = error or caught
            error = error or env.close_error
        evaluation = env.evaluation
        status = {
            "task_id": args.task_id,
            "status": (
                "failed"
                if error
                else "evaluator_error"
                if evaluation and evaluation["integrity_status"] == "evaluator_error"
                else "evidence_invalid"
                if not evaluation or not evaluation["publishable"]
                else "model_protocol_failed"
                if env.model_protocol_error
                else "completed"
            ),
            "model": args.model_id,
            "model_profile": args.model_profile,
            "thinking": True,
            "harness": "native_chat_gateway",
            "score": evaluation["score"] if evaluation else None,
            "evaluation_integrity": evaluation["integrity_status"] if evaluation else None,
            "truncated": env.truncated or bool(sample and sample.truncated),
            "terminal_action": env.terminal_action,
            "response_parse_error": env.response_parse_error,
            "response_repair_used": env.response_repair_used,
            "structured_repair_used": False,
            "model_protocol_error": env.model_protocol_error,
            "model_protocol_failure_phase": env.model_protocol_failure_phase,
            "model_protocol_error_type": env.model_protocol_error_type,
            "fixed_parameters": {
                "max_steps": args.max_steps,
                "base_max_steps": args.max_steps,
                "final_response_reserve_steps": args.final_response_reserve_steps,
                "repair_invalid_response": args.repair_invalid_response,
                "history_n": None,
                "text_history": "all",
                "image_max": desktop.IMAGE_MAX,
                "fold_size": desktop.FOLD_SIZE,
                "coordinate_type": "canonical_normalized_0_1000",
                "screen": [desktop.SCREEN_WIDTH, desktop.SCREEN_HEIGHT],
                "max_tokens": desktop.MAX_TOKENS,
                "timeout_seconds": desktop.MODEL_TIMEOUT_SECONDS,
                "read_timeout_seconds": 600.0,
                "model_attempts": desktop.MODEL_ATTEMPTS,
                "sdk_max_retries": 0,
                "post_action_delay_seconds": desktop.POST_ACTION_DELAY_SECONDS,
                "startup_page_gate": args.startup_page_gate,
                "reasoning_effort": args.reasoning_effort,
            },
        }
        if env.startup_failure:
            status["failure_phase"] = "startup"
        if error:
            status["error"] = {"type": type(error).__name__, "message": redact(str(error), secrets)}
        desktop._atomic_json(output / "status.json", status)
        desktop.secure_result_tree(output)
        if error:
            raise error
        return status


def main(argv: list[str] | None = None) -> int:
    """Resolve the public CLI and execute one isolated native episode."""
    asyncio.run(run_case(_parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
