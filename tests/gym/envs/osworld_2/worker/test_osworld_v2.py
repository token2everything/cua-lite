"""CUA-Lite/Worker boundary tests; no VM or model credentials are needed."""

import asyncio
import http.client
import io
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import lite.gym as gym
from lite.core.messages.final import LOOP_DETECT_TERMINATE_REASON
from lite.core.tools.calls import make_tool_call
from lite.gym.envs.osworld_2 import worker_env as adapter
from lite.gym.envs.osworld_2.catalog import load_release, release_names
from lite.gym.errors import TrueInfraFailure
from lite.gym.utils.feedback.ingress import make_internal_terminate_action


class FakeWorker:
    def __init__(self):
        self.release = adapter.DEFAULT_RELEASE
        self.state = "idle"
        self.run_id = None
        self.actions, self.reads, self.closed = [], [], []
        self.resets, self.questions = [], []
        self.task_reads = 0
        self.task_id = None
        self.observations = {}
        self.latest_observation = None
        self.evaluations = 0
        self.phase_count = 1
        self.phase_index = 0
        self.phase_scores = [0.75]
        self.has_user_simulator = False
        self.task_current_date = None
        self.reset_started = threading.Event()
        self.reset_release = None
        self.reset_error = None
        self.action_error = None
        self.evaluate_error = None
        self.busy_reads = 0

    def observation(self):
        index = len(self.observations)
        stream = io.BytesIO()
        Image.new("RGB", (1600, 900), (index, 20, 30)).save(stream, format="PNG")
        obs_id = f"observation-{index}"
        self.observations[obs_id] = {
            "observation_id": obs_id,
            "run_id": self.run_id,
            "screenshot": stream.getvalue(),
            "instruction": (
                "Edit the document." if self.phase_count == 1 else f"Phase {self.phase_index}"
            ),
            "step": index,
            "phase_index": self.phase_index,
        }
        self.latest_observation = {"observation_id": obs_id}
        return self.latest_observation

    def check_phase(self, phase_index):
        if phase_index != self.phase_index:
            raise RuntimeError("Request phase_index does not match the current phase")

    def snapshot(self):
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "state": self.state,
            "busy": False,
            "agent_token": "test-agent-secret",
            "observation": self.latest_observation,
            "phase_count": self.phase_count,
            "phase_index": self.phase_index,
            "has_user_simulator": self.has_user_simulator,
            "task_current_date": self.task_current_date,
        }

    def client(self, endpoint, token, timeout=2100):
        worker = self

        class Client:
            def __init__(self):
                self.endpoint, self.token, self.timeout = endpoint, token, timeout

            def request(self, method, path):
                assert (method, path) == ("GET", "/healthz")
                busy = worker.busy_reads > 0
                worker.busy_reads = max(0, worker.busy_reads - 1)
                return {
                    "worker_id": "test-worker",
                    "release": worker.release,
                    "state": worker.state,
                    "busy": busy,
                    "run_id": worker.run_id,
                    "phase_index": worker.phase_index,
                }

            def reset(self, task_id, run_id):
                assert self.token == "test-admin-secret"
                worker.run_id, worker.state = run_id, "active"
                worker.task_id, worker.phase_index = task_id, 0
                worker.resets.append((task_id, run_id))
                worker.reset_started.set()
                if worker.reset_release is not None:
                    assert worker.reset_release.wait(5)
                if worker.reset_error:
                    raise worker.reset_error
                worker.observation()
                return worker.snapshot()

            def task(self, run_id):
                assert self.token == "test-admin-secret"
                if run_id != worker.run_id or worker.state == "idle":
                    raise RuntimeError("No matching active episode")
                worker.task_reads += 1
                return worker.snapshot()

            def observe(self, run_id, observation_id=None):
                assert self.token == "test-agent-secret"
                assert run_id == worker.run_id
                worker.reads.append(observation_id)
                return worker.observations[observation_id]

            def action(self, run_id, action, pause=0.2, *, phase_index=0):
                assert self.token == "test-agent-secret"
                assert run_id == worker.run_id
                worker.check_phase(phase_index)
                worker.actions.append(action)
                if worker.action_error:
                    raise worker.action_error
                return {
                    "observation": worker.observation(),
                    "reward": 0,
                    "done": action in ("DONE", "FAIL"),
                    "info": {},
                }

            def refresh(self, run_id, *, phase_index=0):
                assert self.token == "test-agent-secret"
                assert run_id == worker.run_id
                worker.check_phase(phase_index)
                return {"observation": worker.observation()}

            def ask_user(self, run_id, question, *, phase_index=0):
                assert self.token == "test-agent-secret"
                assert run_id == worker.run_id
                worker.check_phase(phase_index)
                assert worker.has_user_simulator
                worker.questions.append(question)
                return {
                    "answer": 'Use “Budget 2026.xlsx”; keep the original.\n谢谢。',
                    "observation": worker.observation(),
                }

            def evaluate(self, run_id, *, phase_index=0):
                assert self.token == "test-admin-secret"
                assert run_id == worker.run_id
                worker.check_phase(phase_index)
                worker.evaluations += 1
                if worker.evaluate_error:
                    raise worker.evaluate_error
                complete = phase_index == worker.phase_count - 1
                worker.state = "evaluated" if complete else "phase_evaluated"
                value = worker.phase_scores[phase_index]
                return {
                    "complete": complete,
                    "score": sum(worker.phase_scores[:phase_index + 1]),
                    "phase_score": value,
                    "phase_index": phase_index,
                    "phase_count": worker.phase_count,
                    "result": {"partial": value},
                }

            def advance(self, run_id, *, phase_index):
                assert self.token == "test-admin-secret"
                assert run_id == worker.run_id
                worker.check_phase(phase_index)
                assert worker.state == "phase_evaluated"
                worker.phase_index += 1
                worker.state = "active"
                worker.observation()
                return worker.snapshot()

            def close(self, run_id):
                assert self.token == "test-admin-secret"
                assert run_id == worker.run_id
                assert worker.busy_reads == 0
                worker.closed.append(run_id)
                worker.state = "idle"
                return {"closed": True}

        return Client()


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.connection = self.root / "connection.json"
        self.connection.write_text(
            json.dumps(
                {
                    "endpoint": "http://test-worker/proxy/8800",
                    "token": "test-admin-secret",
                    "worker_id": "test-worker",
                }
            )
        )
        self.worker = FakeWorker()
        self.patcher = patch.object(adapter, "WorkerClient", side_effect=self.worker.client)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.env = self.make_env()

    def make_env(self, **kwargs):
        return adapter.OSWorldWorkerEnv(
            task_id=kwargs.pop("task_id", "054"),
            connection_file=str(self.connection),
            results_root=str(self.root / "results"),
            **kwargs,
        )

    async def asyncTearDown(self):
        if self.worker.reset_release is not None:
            self.worker.reset_release.set()
        if self.worker.state == "broken":
            self.worker.state = "active"
        await self.env.close()

    async def test_native_gym_registration_and_pre_reset_metadata(self):
        for name in release_names():
            with self.subTest(release=name):
                self.worker.release = name
                env = gym.make(
                    f"{load_release(name).env_id}@054",
                    connection_file=str(self.connection),
                    results_root=str(self.root / "results"),
                )
                self.assertEqual(env.metadata.others["benchmark_release"], name)
                self.assertEqual(env.metadata.others["task_id"], "054")
                self.assertNotIn("cursor_position", env.metadata.valid_actions)
                self.assertEqual(self.worker.state, "idle")
                try:
                    obs = await env.reset()
                    self.assertEqual(obs.text, "Edit the document.")
                    self.assertTrue(obs.image.startswith(b"\x89PNG"))
                finally:
                    await env.close()
        self.assertEqual(len(self.worker.closed), len(release_names()))

    async def test_coordinates_and_batch_frames_use_real_image_dimensions(self):
        await self.env.reset()
        result = await self.env.step(
            [
                make_tool_call("mouse_move", {"coordinate": [500, 500]}, call_id="move"),
                make_tool_call("click", {"coordinate": [250, 100]}, call_id="click"),
            ]
        )
        self.assertEqual(
            self.worker.actions,
            ["pyautogui.moveTo(800, 450)", "pyautogui.click(400, 90, button='left')"],
        )
        self.assertEqual(self.worker.reads, ["observation-0", "observation-1", "observation-2"])
        self.assertEqual([r.tool_call_id for r in result.results], ["move", "click"])
        self.assertEqual(len(result.results[0].images), 2)
        self.assertIsNone(result.reward)
        self.assertEqual(self.worker.evaluations, 0)

    async def test_out_of_range_computer_action_returns_feedback_without_execution(self):
        await self.env.reset()
        result = await self.env.step(
            [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "click", "coordinate": [1069, 605]},
                            {"action": "mouse_move", "coordinate": [500, 500]},
                        ]
                    },
                    call_id="computer-1",
                )
            ]
        )
        self.assertEqual(self.worker.actions, ["pyautogui.moveTo(800, 450)"])
        self.assertEqual(len(result.results), 1)
        feedback = result.results[0]
        self.assertEqual(feedback.tool_call_id, "computer-1")
        self.assertIn("normalized to [0, 1000]", feedback.error)
        self.assertIn("1069", feedback.error)
        self.assertIn("x * 1000 / 1600", feedback.error)
        self.assertIn("y * 1000 / 900", feedback.error)
        self.assertEqual(len(feedback.images), 2)
        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(self.worker.evaluations, 0)

        corrected = await self.env.step(
            [make_tool_call("click", {"coordinate": [1000, 1000]}, call_id="corrected")]
        )
        self.assertEqual(self.worker.actions[-1], "pyautogui.click(1599, 899, button='left')")
        self.assertEqual(corrected.results[0].tool_call_id, "corrected")
        self.assertIsNone(corrected.results[0].error)

    async def test_out_of_range_drag_start_is_rejected_before_worker_execution(self):
        await self.env.reset()
        result = await self.env.step(
            [
                make_tool_call(
                    "drag",
                    {"coordinate": [500, 500], "start_coordinate": [-1, 500]},
                    call_id="bad-drag",
                )
            ]
        )
        self.assertEqual(self.worker.actions, [])
        self.assertEqual(result.results[0].tool_call_id, "bad-drag")
        self.assertIn("normalized to [0, 1000]", result.results[0].error)
        self.assertTrue(result.results[0].images)
        self.assertFalse(result.terminated)

    async def test_task_date_is_visible_in_initial_instruction_and_artifacts(self):
        self.worker.task_current_date = "2026-08-10"
        self.env = self.make_env(task_id="024")
        observation = await self.env.reset()
        self.assertEqual(
            observation.text,
            "Edit the document.\n\nCurrent date for this task: 2026-08-10",
        )
        self.assertEqual(observation.metadata["task_current_date"], "2026-08-10")
        run = json.loads((self.env.artifacts.path / "run.json").read_text())
        self.assertEqual(run["task_current_date"], "2026-08-10")

    async def test_type_enter_and_click_at_current_cursor(self):
        await self.env.reset()
        await self.env.step(
            [
                make_tool_call("type", {"text": "", "press_enter": True}, call_id="type"),
                make_tool_call("click", {"button": "right", "clicks": 2}, call_id="click"),
            ]
        )
        self.assertEqual(
            self.worker.actions,
            ["pyautogui.press('enter')", "pyautogui.click(button='right', clicks=2, interval=0.1)"],
        )

    async def test_native_computer_batch_is_one_paired_result(self):
        await self.env.reset()
        result = await self.env.step(
            [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "mouse_move", "coordinate": [500, 500]},
                            {"action": "click", "button": "left"},
                        ]
                    },
                    call_id="computer-1",
                )
            ]
        )
        self.assertEqual(len(self.worker.actions), 2)
        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0].tool_call_id, "computer-1")
        self.assertEqual(len(result.results[0].images), 2)

    async def test_evaluator_failure_does_not_become_zero_score(self):
        await self.env.reset()
        self.worker.evaluate_error = RuntimeError("evaluator failed")
        with self.assertRaisesRegex(TrueInfraFailure, "evaluator failed"):
            await self.env.step(
                [make_tool_call("terminate", {"status": "success"}, call_id="done")]
            )
        await self.env.close()
        status = json.loads((self.env.artifacts.path / "run.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertTrue(status["closed"])
        self.assertNotIn("score", status)
        self.assertFalse((self.env.artifacts.path / "result.json").exists())
        summary = json.loads((self.env.artifacts.experiment.path / "summary.json").read_text())
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertTrue(summary["runs"][0]["closed"])
        self.assertIn("evaluator failed", summary["runs"][0]["error"])

    async def test_bad_call_and_shell_rejection_do_not_block_valid_sibling(self):
        await self.env.reset()
        result = await self.env.step(
            [
                make_tool_call("click", {"coordinate": [None, None]}, call_id="invalid"),
                make_tool_call("bash", {"command": "echo hello"}, call_id="shell"),
                make_tool_call("key", {"keys": ["esc"]}, call_id="valid"),
            ]
        )
        self.assertEqual(self.worker.actions, ["pyautogui.hotkey('esc')"])
        by_id = {r.tool_call_id: r for r in result.results}
        self.assertTrue(by_id["invalid"].error)
        self.assertTrue(by_id["shell"].error)
        self.assertIsNone(by_id["valid"].error)
        self.assertFalse(result.terminated)

    async def test_finish_scores_once_and_does_not_run_later_sibling(self):
        await self.env.reset()
        result = await self.env.step(
            [
                make_tool_call("terminate", {"status": "success"}, call_id="done"),
                make_tool_call("click", {"coordinate": [500, 500]}, call_id="later"),
            ]
        )
        self.assertTrue(result.terminated)
        self.assertEqual(result.reward, 0.75)
        self.assertEqual(self.worker.actions, ["DONE"])
        self.assertEqual([r.tool_call_id for r in result.results], ["later"])
        self.assertIn("ended", result.results[0].error)
        with self.assertRaises(RuntimeError):
            await self.env.step([])
        self.assertEqual(self.worker.evaluations, 1)
        await self.env.close()
        self.assertEqual(
            json.loads((self.env.artifacts.path / "run.json").read_text())["status"], "completed"
        )

    async def test_report_infeasible_is_fail_with_paired_feedback(self):
        await self.env.reset()
        result = await self.env.step(
            [make_tool_call("report_infeasible", {"reason": "Unavailable"}, call_id="fail")]
        )
        self.assertEqual(self.worker.actions, ["FAIL"])
        self.assertTrue(result.terminated)
        self.assertEqual(result.results[0].tool_call_id, "fail")

    async def test_internal_finish_answers_intercepted_call(self):
        await self.env.reset()
        result = await self.env.step(
            [
                make_internal_terminate_action(
                    reason=LOOP_DETECT_TERMINATE_REASON, result_call_id="loop"
                )
            ]
        )
        self.assertTrue(result.terminated)
        self.assertEqual(result.results[0].tool_call_id, "loop")

    async def test_max_steps_counts_batches_and_refreshes_empty_turn(self):
        self.env = self.make_env(max_steps=1)
        initial = await self.env.reset()
        result = await self.env.step([])
        self.assertTrue(result.truncated)
        self.assertFalse(result.terminated)
        self.assertEqual(result.reward, 0.75)
        self.assertEqual(self.worker.actions, [])
        self.assertNotEqual(initial.image, self.env._image)

    async def test_events_preserve_metadata_and_results_without_duplicate_images(self):
        initial = await self.env.reset()
        self.assertEqual(initial.image, self.worker.observations["observation-0"]["screenshot"])
        for i in range(2):
            result = await self.env.step([make_tool_call("screenshot", {}, call_id=f"shot-{i}")])
            self.assertEqual(result.results[0].tool_call_id, f"shot-{i}")
            self.assertEqual(
                result.results[0].images,
                [self.worker.observations[f"observation-{i + 1}"]["screenshot"]],
            )
        result = await self.env.step(
            [make_tool_call("terminate", {"status": "success"}, call_id="finish")]
        )
        await self.env.close()

        output = self.env.artifacts.path
        events_text = (output / "events.jsonl").read_text()
        events = [json.loads(line) for line in events_text.splitlines()]
        observations = [
            event["observation"] for event in events if event["method"] == "observation"
        ]
        self.assertEqual(
            [observation["observation_id"] for observation in observations],
            list(self.worker.observations),
        )
        self.assertEqual([observation["step"] for observation in observations], [0, 1, 2, 3])
        self.assertTrue(all(observation["run_id"] == output.name for observation in observations))
        self.assertTrue(all("screenshot" not in observation for observation in observations))
        self.assertEqual(events[-1]["method"], "close")
        self.assertNotIn("test-admin-secret", events_text)
        self.assertNotIn("test-agent-secret", events_text)
        self.assertEqual(
            {path.name for path in output.iterdir()}, {"run.json", "events.jsonl", "result.json"}
        )
        evaluation = json.loads((output / "result.json").read_text())
        self.assertEqual(evaluation, {
            "complete": True, "score": 0.75, "phase_score": 0.75,
            "phase_index": 0, "phase_count": 1, "result": {"partial": 0.75},
        })
        self.assertEqual(result.reward, evaluation["score"])
        state = json.loads((output / "run.json").read_text())
        self.assertEqual(state["status"], "completed")
        self.assertTrue(state["closed"])

    async def test_transport_error_is_infra_failure_and_action_not_replayed(self):
        await self.env.reset()
        self.worker.action_error = http.client.IncompleteRead(b"{", 20)
        with self.assertRaises(TrueInfraFailure):
            await self.env.step([make_tool_call("key", {"keys": ["esc"]}, call_id="key")])
        self.assertEqual(len(self.worker.actions), 1)
        status = json.loads((self.env.artifacts.path / "run.json").read_text())
        self.assertEqual(status["status"], "failed")
        events = [
            json.loads(line)
            for line in (self.env.artifacts.path / "events.jsonl").read_text().splitlines()
        ]
        self.assertEqual(events[-1]["method"], "action")
        self.assertEqual(events[-1]["status"], "started")
        self.assertEqual(events[-1]["tool_call_id"], "key")
        with self.assertRaises(RuntimeError):
            await self.env.step([])

    async def test_cancelled_reset_is_awaited_then_closed(self):
        self.worker.reset_release = threading.Event()
        resetting = asyncio.create_task(self.env.reset())
        self.assertTrue(await asyncio.to_thread(self.worker.reset_started.wait, 5))
        resetting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await resetting
        self.worker.reset_release.set()
        await self.env.close()
        self.assertEqual(self.worker.closed, [self.env.artifacts.run_id])
        self.assertEqual(self.worker.state, "idle")

    async def test_lost_reset_response_still_closes_only_its_run(self):
        self.worker.reset_error = http.client.IncompleteRead(b"{", 20)
        with self.assertRaises(TrueInfraFailure):
            await self.env.reset()
        await self.env.close()
        self.assertEqual(self.worker.closed, [self.env.artifacts.run_id])

    async def test_lost_reset_response_waits_for_server_operation_before_close(self):
        self.worker.reset_error = http.client.IncompleteRead(b"{", 20)
        with self.assertRaises(TrueInfraFailure):
            await self.env.reset()
        self.worker.busy_reads = 2
        await self.env.close()
        self.assertEqual(self.worker.closed, [self.env.artifacts.run_id])

    async def test_broken_worker_requires_replacement_instead_of_zero_score(self):
        await self.env.reset()
        self.worker.action_error = RuntimeError("broken test-agent-secret")
        with self.assertRaises(TrueInfraFailure) as error:
            await self.env.step(
                [
                    make_tool_call("screenshot", {}, call_id="shot"),
                    make_tool_call("key", {"keys": ["esc"]}, call_id="key"),
                ]
            )
        self.assertNotIn("test-agent-secret", str(error.exception))
        self.worker.state = "broken"
        with self.assertRaisesRegex(TrueInfraFailure, "replace"):
            await self.env.close()
        self.assertEqual(self.worker.closed, [])
        self.assertEqual(self.worker.evaluations, 0)
        status = json.loads((self.env.artifacts.path / "run.json").read_text())
        self.assertIn("broken", status["error"])
        self.assertNotIn("test-agent-secret", status["error"])
        self.assertIn("replace", status["cleanup_error"])
        self.assertFalse(status["closed"])

    async def test_cleanup_rpc_failure_preserves_primary_error_across_repeated_close(self):
        await self.env.reset()
        self.worker.action_error = RuntimeError("primary_action_failure")
        with self.assertRaises(TrueInfraFailure):
            await self.env.step([make_tool_call("key", {"keys": ["esc"]}, call_id="key")])

        def failed_health(*args, **kwargs):
            raise ConnectionError("secondary_cleanup_failure")

        with patch.object(self.env._admin, "request", new=failed_health):
            for _ in range(2):
                with self.assertRaises(TrueInfraFailure):
                    await self.env.close()
                status = json.loads((self.env.artifacts.path / "run.json").read_text())
                self.assertIn("primary_action_failure", status["error"])
                self.assertIn("secondary_cleanup_failure", status["cleanup_error"])
                self.assertFalse(status["closed"])

    async def test_preexisting_busy_worker_is_not_touched(self):
        self.worker.state, self.worker.run_id = "active", "somebody-elses-run"
        with self.assertRaises(TrueInfraFailure):
            await self.env.reset()
        await self.env.close()
        self.assertEqual(self.worker.closed, [])

    async def test_unqualified_tasks_remain_marked_but_candidate_env_can_reset(self):
        release = replace(
            adapter.load_release(),
            exclusions={
                "007": "Awaiting interaction qualification",
                "069": "Awaiting phase qualification",
            },
        )
        # Qualification state belongs to the test, independently of live releases.
        self.enterContext(patch.object(adapter, "load_release", return_value=release))
        for task_id in ("007", "069"):
            self.assertIn(
                "exclude_reason", adapter.OSWorldWorkerEnv._task_metadata(task_id).others
            )
        self.worker.has_user_simulator = True
        self.env = self.make_env(task_id="007", extra_tools=["ask_user"])
        observation = await self.env.reset()
        self.assertEqual(observation.text, "Edit the document.")
        self.assertEqual(self.worker.resets[0][0], "007")
        self.assertIn("exclude_reason", self.env.metadata.others)
        self.assertNotIn("exclude_reason", adapter.OSWorldWorkerEnv._task_metadata("054").others)

    async def test_runtime_protocol_check_closes_unexpected_multiphase_task(self):
        self.worker.phase_count = 2
        with self.assertRaisesRegex(ValueError, "OSWorld task runner"):
            await self.env.reset()
        self.assertEqual(self.worker.closed, [self.env.artifacts.run_id])

    async def test_phase_bound_reset_replays_initial_observation_and_close_keeps_session(self):
        self.worker.phase_count, self.worker.phase_scores = 2, [0.25, 0.5]
        admin = self.worker.client("http://test-worker", "test-admin-secret")
        admin.reset("069", "managed-reset")
        self.env = self.make_env(
            task_id="069", run_id="managed-reset", phase_index=0, experiment="phases"
        )
        initial = await self.env.reset()
        repeated = await asyncio.gather(self.env.reset(), self.env.reset())
        self.assertEqual(repeated, [initial, initial])
        self.assertEqual(initial.text, "Phase 0")
        self.assertEqual(initial.metadata["phase_index"], 0)
        self.assertEqual(initial.metadata["phase_count"], 2)
        self.assertEqual(self.worker.resets, [("069", "managed-reset")])
        self.assertEqual(self.worker.task_reads, 1)
        await self.env.close()
        await self.env.close()
        self.assertEqual(self.worker.closed, [])
        self.assertEqual(self.worker.state, "active")
        state = json.loads((self.env.artifacts.path / "run.json").read_text())
        self.assertEqual(state["status"], "running")
        self.assertFalse(state.get("closed", False))
        with self.assertRaises(RuntimeError):
            await self.env.step([])

    async def test_two_phase_handles_share_run_and_report_only_each_phase_reward(self):
        self.worker.phase_count, self.worker.phase_scores = 2, [0.25, 0.5]
        admin = self.worker.client("http://test-worker", "test-admin-secret")
        admin.reset("069", "managed-two-phase")
        self.env = self.make_env(
            task_id="069", run_id="managed-two-phase", phase_index=0, experiment="phases"
        )
        await self.env.reset()
        first = await self.env.step([
            make_tool_call("terminate", {"status": "success"}, call_id="first-done")
        ])
        self.assertTrue(first.terminated)
        self.assertEqual(first.reward, 0.25)
        self.assertFalse(first.info["evaluate_payload"]["complete"])
        previous = self.env
        admin.advance("managed-two-phase", phase_index=0)
        self.env = self.make_env(
            task_id="069", run_id="managed-two-phase", phase_index=1, experiment="phases"
        )
        second_initial = await self.env.reset()
        self.assertEqual(second_initial.text, "Phase 1")
        self.assertEqual(self.env.artifacts.path, previous.artifacts.path)
        # A detached close from the old native env-server handle can finish late.
        await previous.close()
        self.assertEqual(self.worker.state, "active")
        second = await self.env.step([
            make_tool_call("terminate", {"status": "success"}, call_id="second-done")
        ])
        self.assertEqual(second.reward, 0.5)
        self.assertEqual(second.info["evaluate_payload"]["score"], 0.75)
        self.assertTrue(second.info["evaluate_payload"]["complete"])
        await self.env.close()
        self.assertEqual(self.worker.closed, [])
        self.assertEqual(self.worker.resets, [("069", "managed-two-phase")])
        state = json.loads((self.env.artifacts.path / "run.json").read_text())
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["phase_status"], "completed")
        self.assertEqual(state["closed_phase_index"], 1)
        self.assertFalse(state.get("closed", False))
        admin.close("managed-two-phase")
        self.assertEqual(self.worker.closed, ["managed-two-phase"])

    async def test_stale_phase_handle_cannot_execute_current_phase_action(self):
        self.worker.phase_count = 2
        admin = self.worker.client("http://test-worker", "test-admin-secret")
        admin.reset("069", "managed-stale")
        self.env = self.make_env(
            task_id="069", run_id="managed-stale", phase_index=0, experiment="phases"
        )
        await self.env.reset()
        self.worker.phase_index = 1
        with self.assertRaisesRegex(TrueInfraFailure, "phase_index"):
            await self.env.step([make_tool_call("screenshot", {}, call_id="stale")])
        await self.env.close()
        self.assertEqual(self.worker.actions, [])
        self.assertEqual(self.worker.closed, [])

    async def test_phase_attach_rejects_wrong_task_without_closing_worker(self):
        admin = self.worker.client("http://test-worker", "test-admin-secret")
        admin.reset("069", "managed-wrong")
        self.env = self.make_env(run_id="managed-wrong", phase_index=0)
        with self.assertRaisesRegex(TrueInfraFailure, "task/phase differs"):
            await self.env.reset()
        await self.env.close()
        self.assertEqual(self.worker.closed, [])
        self.assertEqual(self.worker.state, "active")

    async def test_phase_attach_rejects_old_phase_without_closing_current_phase(self):
        self.worker.phase_count = 2
        admin = self.worker.client("http://test-worker", "test-admin-secret")
        admin.reset("069", "managed-old-phase")
        self.worker.phase_index = 1
        self.env = self.make_env(
            task_id="069", run_id="managed-old-phase", phase_index=0
        )
        with self.assertRaisesRegex(TrueInfraFailure, "task/phase differs"):
            await self.env.reset()
        await self.env.close()
        self.assertEqual(self.worker.closed, [])
        self.assertEqual(self.worker.phase_index, 1)
        self.assertEqual(self.worker.state, "active")

    async def test_ask_user_preserves_exact_reply_and_call_pairing_without_termination(self):
        self.worker.has_user_simulator = True
        self.env = self.make_env(task_id="007", extra_tools=["ask_user", "terminate"])
        await self.env.reset()
        result = await self.env.step([
            make_tool_call("ask_user", {"text": "Which file?"}, call_id="ask-1"),
            make_tool_call("screenshot", {}, call_id="screen-after-ask"),
        ])
        self.assertEqual(self.worker.questions, ["Which file?"])
        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)
        self.assertIsNone(result.reward)
        self.assertEqual(self.worker.evaluations, 0)
        self.assertEqual(self.worker.actions, [])
        self.assertEqual(
            [item.tool_call_id for item in result.results], ["ask-1", "screen-after-ask"]
        )
        answer = result.results[0]
        self.assertEqual(answer.text, 'Use “Budget 2026.xlsx”; keep the original.\n谢谢。')
        self.assertIsNone(answer.error)
        self.assertEqual(answer.images, [self.worker.observations["observation-1"]["screenshot"]])
        self.assertEqual(answer.metadata["phase_index"], 0)

    async def test_ask_user_without_simulator_returns_paired_error_and_continues(self):
        self.env = self.make_env(extra_tools=["ask_user"])
        await self.env.reset()
        result = await self.env.step([
            make_tool_call("ask_user", {"text": "Which file?"}, call_id="no-simulator"),
            make_tool_call("key", {"keys": ["esc"]}, call_id="next-action"),
        ])
        by_id = {item.tool_call_id: item for item in result.results}
        self.assertEqual(
            by_id["no-simulator"].error, "ask_user is not available in this task."
        )
        self.assertIsNone(by_id["next-action"].error)
        self.assertFalse(result.terminated)
        self.assertIsNone(result.reward)
        self.assertEqual(self.worker.questions, [])
        self.assertEqual(self.worker.evaluations, 0)
        self.assertEqual(self.worker.actions, ["pyautogui.hotkey('esc')"])


if __name__ == "__main__":
    unittest.main()
