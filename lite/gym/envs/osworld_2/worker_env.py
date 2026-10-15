"""Pinned OSWorld tasks and phase-bound environments through the Worker protocol.

Standalone single-phase environments own their Worker episode. The OSWorld
task runner supplies run_id and phase_index to bind a phase of its own session;
closing that phase releases the handle, while the runner closes the whole task.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from lite.core.messages.final import MAX_STEPS_STOP_REASON, STOP_REASON_INFO_KEY
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER
from lite.core.tools.extra_tools import LiteFinishToolSet, LiteUserInteractionToolSet
from lite.core.tools.results import make_tool_result
from lite.gym.base import LiteBaseEnv
from lite.gym.envs.osworld.actions import OsworldTools, to_pyautogui
from lite.gym.errors import TrueInfraFailure
from lite.gym.types import EXECUTED_ACTIONS_INFO_KEY, LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
    append_feedback,
    error_only_feedback,
    record_model_action_error,
    unsupported_action_message,
)
from lite.gym.utils.feedback.ingress import (
    invalid_action_message,
    prepare_env_tool_calls,
    standalone_tool_call_feedback_with_reason,
)
from lite.gym.utils.feedback.results import build_tool_results_from_decisions, ordered_tool_call_ids
from lite.gym.utils.feedback.surface import resolve_valid_actions
from lite.utils.image import decode_image

from .artifacts import Experiment, redact, task_identifier
from .catalog import DEFAULT_RELEASE, Release, load_release
from .worker_client import READ_ERRORS, WorkerClient

GUI_ACTIONS = [
    name for name in LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER if name != "cursor_position"
]
DEFAULT_EXTRA_TOOLS = ["terminate", "response", "report_infeasible"]
_RPC_ERRORS = (OSError, RuntimeError, ValueError) + READ_ERRORS


class OSWorldWorkerTools(OsworldTools):
    """Worker-supported user interaction in addition to OSWorld GUI extras."""

    _SCHEMAS = {
        **OsworldTools._SCHEMAS,
        "ask_user": LiteUserInteractionToolSet.get_tool_schema("ask_user"),
    }


class OSWorldWorkerEnv(LiteBaseEnv):
    """Run one Agent episode against one native OSWorld task phase."""

    EXTRA_TOOLS = OSWorldWorkerTools

    def __init__(
        self,
        *,
        task_id: str,
        release: str = DEFAULT_RELEASE,
        run_id: str | None = None,
        phase_index: int | None = None,
        connection_file: str | None = None,
        max_steps: int = 200,
        post_action_delay: float = 0.2,
        valid_actions: list[str] | None = None,
        extra_tools: list[str] | None = DEFAULT_EXTRA_TOOLS,
        experiment: str | None = None,
        results_root: str = ".sandbox/environment-results",
    ):
        self.release = load_release(release)
        self._task_id = task_identifier(task_id)
        self.release.select(tasks=[self._task_id])
        self._run_id = run_id
        if phase_index is not None and (
            type(phase_index) is not int or phase_index < 0 or not run_id
        ):
            raise ValueError("phase_index requires a run_id and a nonnegative integer")
        self._bound_phase = phase_index
        self._phase_index = phase_index if phase_index is not None else 0
        self._has_user_simulator = False
        self._reset_lock = asyncio.Lock()
        self._initial_observation = None
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not 0 <= post_action_delay <= 10:
            raise ValueError("post_action_delay must be between 0 and 10 seconds")
        # Construction and task metadata discovery do not read credentials or
        # contact infrastructure. The episode binds its Worker during reset.
        self._connection_file = connection_file
        self._admin = None
        self._worker_id = None
        self._agent = None
        self._max_steps, self._pause = max_steps, post_action_delay
        self._valid_actions = resolve_valid_actions(
            GUI_ACTIONS if valid_actions is None else valid_actions,
            env_name=self.release.env_id,
            platform="desktop",
        )
        if "cursor_position" in self._valid_actions:
            raise ValueError(
                "The Worker has no cursor-position query; use screenshot and mouse_move"
            )
        self._extra_schemas = type(self).extra_tool_schemas(extra_tools)
        self._experiment, self._results_root = experiment, results_root
        self.artifacts = None
        self._pending = None
        self._opened = self._finished = self._failed = False
        self._reset_started = False
        self._step_count = 0
        self._image = None
        self._evaluation = None

    @staticmethod
    def _task_metadata(task_id: str, release: Release | None = None) -> LiteCUAMetadata:
        release = release or load_release()
        others = {
            "benchmark_release": release.name,
            "release_manifest_sha256": release.manifest_sha256,
            "task_sets": [name for name, ids in release.task_sets.items() if task_id in ids],
            "task_revision": release.task_revision,
            "supported_protocol": "gui_user_interaction_and_task_phases",
        }
        if reason := release.exclusions.get(task_id):
            others["exclude_reason"] = reason
        return LiteCUAMetadata(
            dims=("desktop", "use"),
            valid_actions=GUI_ACTIONS,
            extra_tool_schemas=OSWorldWorkerEnv.extra_tool_schemas(DEFAULT_EXTRA_TOOLS),
            others=others,
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return replace(
            self._task_metadata(self._task_id, self.release),
            valid_actions=self._valid_actions,
            extra_tool_schemas=self._extra_schemas,
        )

    async def _rpc(self, method, *args, cleanup=False, **kwargs):
        # Shield the synchronous client's idempotent operation poll. close() can
        # await it after an outer timeout, then clean up only this episode.
        task = asyncio.create_task(asyncio.to_thread(method, *args, **kwargs))
        self._pending = (method.__name__, task, cleanup)
        try:
            return await asyncio.shield(task)
        except _RPC_ERRORS as error:
            self._failed = True
            message = redact(
                str(error), self.artifacts.secrets if self.artifacts else [self._admin.token]
            )
            if self.artifacts:
                if cleanup:
                    self.artifacts.finish("failed", cleanup_error=message, closed=False)
                else:
                    self.artifacts.finish("failed", error=message)
            raise TrueInfraFailure(f"OSWorld Worker: {message}") from None

    def _adopt_reset(self, result):
        self._opened = True
        self.artifacts.secrets.append(result["agent_token"])
        self._agent = WorkerClient(self._admin.endpoint, result["agent_token"], self._admin.timeout)

    async def _observation(self, reference):
        # Retry the immutable saved image through WorkerClient, never replay its action.
        observation = await self._rpc(
            self._agent.observe, self.artifacts.run_id, reference["observation_id"]
        )
        self._image = observation["screenshot"]
        self._size = decode_image(self._image).size
        self.artifacts.observation(observation)
        return observation

    async def reset(self) -> LiteEnvObservation:
        # Native env-server may retry a lost reset response. A runner-assigned
        # run ID identifies one episode; replay its first observation only.
        async with self._reset_lock:
            if self._run_id and self._initial_observation is not None:
                if self._step_count or not self._opened or self._failed:
                    raise RuntimeError("Create a new env and run_id for a new episode")
                return self._initial_observation
            observation = await self._reset_episode()
            self._initial_observation = observation
            return observation

    async def _reset_episode(self) -> LiteEnvObservation:
        await self.close()
        connection_file = self._connection_file or os.environ.get("OSWORLD_WORKER_CONNECTION")
        if not connection_file:
            raise ValueError(
                "Pass connection_file or set OSWORLD_WORKER_CONNECTION "
                "to the private Worker connection JSON"
            )
        connection = json.loads(Path(connection_file).expanduser().read_text())
        self._admin = WorkerClient(connection["endpoint"], connection["token"])
        self._worker_id = connection["worker_id"]
        self.artifacts = None
        health = await self._rpc(self._admin.request, "GET", "/healthz")
        if (
            health["release"] != self.release.name
            or health["worker_id"] != self._worker_id
            or health["state"] != ("idle" if self._bound_phase is None else "active")
            or health.get("busy")
        ):
            raise TrueInfraFailure(
                f"Expected the matching available {self.release.name} Worker phase"
            )
        experiment = Experiment(
            self._experiment,
            self._results_root,
            release=self.release.name,
            config={"kind": "environment", "env_id": self.release.env_id},
        )
        self.artifacts = experiment.new_run(
            self._task_id, run_id=self._run_id, secrets=[self._admin.token],
            existing=self._bound_phase is not None and self._bound_phase > 0,
        )
        if prefix := connection.get("results_prefix"):
            self.artifacts.update(remote_results=f"{prefix}/runs/{self.artifacts.run_id}")
        self._step_count = 0
        self._finished = self._failed = False
        self._evaluation = None
        self._image = None
        self.artifacts.event("reset", status="started")
        self._reset_started = True
        if self._bound_phase is None:
            result = await self._rpc(self._admin.reset, self._task_id, self.artifacts.run_id)
        else:
            result = await self._rpc(self._admin.task, self.artifacts.run_id)
            if result["phase_index"] != self._bound_phase or result["task_id"] != self._task_id:
                raise TrueInfraFailure("Worker task/phase differs from this environment handle")
        self._adopt_reset(result)
        self._has_user_simulator = result["has_user_simulator"]
        self.artifacts.event("reset", status="succeeded", result=result)
        self.artifacts.update(
            max_steps=self._max_steps, runtime=result.get("runtime"),
            phase_index=self._phase_index, phase_count=result["phase_count"],
            task_current_date=result["task_current_date"],
        )
        if self._bound_phase is None and result["phase_count"] != 1:
            self._failed = True
            reason = "Multi-phase tasks require the OSWorld task runner (osworld_2.rollout)"
            self.artifacts.finish("failed", error=reason)
            await self.close()
            raise ValueError(reason)
        observation = await self._observation(result["observation"])
        instruction = observation["instruction"]
        if result["task_current_date"] is not None:
            instruction += f"\n\nCurrent date for this task: {result['task_current_date']}"
        return LiteEnvObservation(
            image=self._image,
            text=instruction,
            metadata={"benchmark_release": self.release.name, "run_id": self.artifacts.run_id,
                      "phase_index": self._phase_index, "phase_count": result["phase_count"],
                      "task_current_date": result["task_current_date"]},
        )

    async def step(self, actions) -> LiteEnvStepResult:
        if not self._opened or self._finished or self._failed:
            raise RuntimeError(
                "reset() is required before step(), and a completed episode cannot continue"
            )
        call_ids = ordered_tool_call_ids(actions)
        metadata = self.metadata
        projected, errors = prepare_env_tool_calls(actions, metadata)
        images, executed, terminal_ids = [], [], set()
        explicit_results = {}
        terminated, stop_reason = False, None
        for action, result_id in projected:
            name, args = action["name"], action["arguments"]
            if terminated:
                if result_id:
                    append_feedback(
                        errors,
                        result_id,
                        error_only_feedback("Not executed: an earlier tool call ended the episode"),
                    )
                continue
            feedback, reason = standalone_tool_call_feedback_with_reason(
                action, self.known_standalone_tool_names(), metadata.extra_tool_schemas
            )
            if feedback is not None:
                if result_id:
                    append_feedback(errors, result_id, feedback)
                executed.append({"call": "noop", "args": {"name": name, "reason": reason}})
                if feedback.carrier == "current":
                    refreshed = await self._rpc(
                        self._agent.refresh, self.artifacts.run_id, phase_index=self._phase_index
                    )
                    await self._observation(refreshed["observation"])
                    images.append(self._image)
                continue
            if name == "ask_user":
                if not self._has_user_simulator:
                    if result_id:
                        errors[result_id] = error_only_feedback(
                            "ask_user is not available in this task."
                        )
                    continue
                result = await self._rpc(
                    self._agent.ask_user, self.artifacts.run_id, args["text"],
                    phase_index=self._phase_index,
                )
                await self._observation(result["observation"])
                images.append(self._image)
                executed.append({"call": "ask_user", "args": {"text": args["text"]}})
                if result_id:
                    explicit_results[result_id] = make_tool_result(
                        result_id, images=[self._image], text=result["answer"],
                        metadata={"phase_index": self._phase_index},
                    )
                self.artifacts.event("ask_user", phase_index=self._phase_index,
                                     question=args["text"], answer=result["answer"])
                continue
            if name in LiteFinishToolSet.get_tool_names() or name == "report_infeasible":
                failure = (
                    name == "report_infeasible"
                    or args.get("status") == "failure"
                    or name == "response"
                    and "[infeasible]" in args.get("text", "").lower()
                )
                command = "FAIL" if failure else "DONE"
                self.artifacts.event(
                    "action", status="started", action=command, tool_call_id=result_id
                )
                result = await self._rpc(
                    self._agent.action, self.artifacts.run_id, command, pause=0,
                    phase_index=self._phase_index,
                )
                await self._observation(result["observation"])
                images.append(self._image)
                executed.append({"call": command})
                self.artifacts.event("action", status="succeeded", result=result)
                terminated, stop_reason = True, name
                if name in LiteFinishToolSet.get_tool_names() and action.get("call_id"):
                    terminal_ids.add(action["call_id"])
                continue
            invalid = invalid_action_message(action, metadata.valid_actions)
            if invalid:
                if result_id:
                    errors[result_id] = error_only_feedback(invalid)
                executed.append({"call": "noop", "args": {"name": name, "reason": invalid}})
                continue
            try:
                # These canonical parameters are not implemented by the pinned
                # upstream OSWorld converter. Keep their Worker lowering here.
                if name == "click" and args.get("coordinate") is None:
                    command = (
                        f"pyautogui.click(button={args.get('button', 'left')!r}, "
                        f"clicks={args.get('clicks', 1)}, interval=0.1)"
                    )
                else:
                    command = to_pyautogui(name, args, *self._size)
                if name == "type" and args.get("press_enter"):
                    command = (command + "; " if command else "") + "pyautogui.press('enter')"
            except MODEL_ACTION_ERROR_TYPES as error:
                record_model_action_error(errors, result_id, error, action_name=name)
                executed.append({"call": "noop", "args": {"name": name, "reason": str(error)}})
                refreshed = await self._rpc(
                    self._agent.refresh, self.artifacts.run_id, phase_index=self._phase_index
                )
                await self._observation(refreshed["observation"])
                images.append(self._image)
                continue
            if command is None and name not in ("screenshot", "type"):
                if result_id:
                    errors[result_id] = error_only_feedback(unsupported_action_message(name))
                executed.append({"call": "noop", "args": {"name": name}})
                continue
            self.artifacts.event("action", status="started", action=command, tool_call_id=result_id)
            if command:
                result = await self._rpc(
                    self._agent.action, self.artifacts.run_id, command, pause=self._pause,
                    phase_index=self._phase_index,
                )
            else:
                result = await self._rpc(
                    self._agent.refresh, self.artifacts.run_id, phase_index=self._phase_index
                )
            await self._observation(result["observation"])
            images.append(self._image)
            executed.append({"call": command or name})
            self.artifacts.event("action", status="succeeded", result=result)

        self._step_count += 1
        truncated = not terminated and self._step_count >= self._max_steps
        info: dict[str, Any] = {
            EXECUTED_ACTIONS_INFO_KEY: executed, "phase_index": self._phase_index,
        }
        reward = None
        if truncated:
            stop_reason = MAX_STEPS_STOP_REASON
        if stop_reason is not None:
            info[STOP_REASON_INFO_KEY] = stop_reason
        if not images and not terminated:
            refreshed = await self._rpc(
                self._agent.refresh, self.artifacts.run_id, phase_index=self._phase_index
            )
            await self._observation(refreshed["observation"])
            images.append(self._image)
        progress = {"env_steps": self._step_count}
        if terminated or truncated:
            self._evaluation = await self._rpc(
                self._admin.evaluate, self.artifacts.run_id, phase_index=self._phase_index
            )
            if self._bound_phase is None and not self._evaluation["complete"]:
                self._failed = True
                self.artifacts.finish(
                    "failed", error="Worker returned an incomplete phase evaluation"
                )
                raise TrueInfraFailure("Worker returned an incomplete phase evaluation")
            reward = float(self._evaluation["phase_score"])
            info["evaluate_payload"] = self._evaluation
            info["run_id"] = self.artifacts.run_id
            info["benchmark_release"] = self.release.name
            info["worker_id"] = self._worker_id
            self._finished = True
            self.artifacts.save_json("result.json", self._evaluation)
            progress.update(score=reward, stop_reason=stop_reason)
        self.artifacts.update(**progress)
        return build_tool_results_from_decisions(
            LiteEnvStepResult(reward=reward, terminated=terminated, truncated=truncated, info=info),
            ordered_call_ids=call_ids,
            continue_call_ids=[c for c in call_ids if c not in terminal_ids],
            images=images or [self._image],
            feedback=errors,
            explicit_results=explicit_results,
        )

    async def close(self) -> None:
        if self._pending is not None:
            method, task, cleanup = self._pending
            try:
                pending_result = await asyncio.shield(task)
            except Exception as error:
                pending_result = None
                self._failed = True
                message = redact(
                    str(error), self.artifacts.secrets if self.artifacts else [self._admin.token]
                )
                if self.artifacts:
                    if cleanup:
                        self.artifacts.finish("failed", cleanup_error=message, closed=False)
                    else:
                        self.artifacts.finish("failed", error=message)
            self._pending = None
            if method == "reset" and pending_result is not None and not self._opened:
                self._adopt_reset(pending_result)
        if not self._reset_started:
            return
        if self._bound_phase is not None:
            # Native DELETE can finish after the next phase is bound. This
            # handle owns no task resource and must never close that task.
            self._reset_started = self._opened = False
            self._agent = None
            self.artifacts.event("phase_close", phase_index=self._phase_index)
            self.artifacts.update(
                phase_closed=True, closed_phase_index=self._phase_index,
                phase_status=(
                    "failed" if self._failed else "completed" if self._finished else "interrupted"
                ),
            )
            return
        health = await self._rpc(self._admin.request, "GET", "/healthz", cleanup=True)
        deadline = asyncio.get_running_loop().time() + self._admin.timeout
        # A lost submit response can exhaust HTTP retries before the server's
        # operation finishes. Wait for that operation instead of racing close
        # against the Worker's busy gate or replaying reset/action.
        while health.get("busy") and health["state"] != "broken":
            if asyncio.get_running_loop().time() >= deadline:
                self._failed = True
                self.artifacts.finish(
                    "failed", cleanup_error="Cleanup timed out waiting for the Worker", closed=False
                )
                raise TrueInfraFailure(
                    "Worker remains busy; its owner must inspect or replace the Sandbox"
                )
            await asyncio.sleep(1)
            health = await self._rpc(self._admin.request, "GET", "/healthz", cleanup=True)
        if health["state"] == "broken":
            self._failed = True
            self.artifacts.finish(
                "failed",
                cleanup_error="Worker is broken; its owner must replace the Sandbox",
                closed=False,
            )
            raise TrueInfraFailure(
                "Cannot close a broken Worker; its owner must replace the Sandbox"
            )
        if health["state"] != "idle":
            # Even when a reset response was lost, the server checks this exact
            # run ID before accepting close. Never close an unidentified run.
            await self._rpc(self._admin.close, self.artifacts.run_id, cleanup=True)
        self._reset_started = False
        self._opened = False
        self._agent = None
        self.artifacts.event("close", status="succeeded")
        status = "failed" if self._failed else "completed" if self._finished else "interrupted"
        self.artifacts.finish(status, closed=True)
