"""Worker transport retries and close receipts need no live VM or platform."""

from __future__ import annotations

import base64
import hashlib
import http.client
import importlib.util
import io
import json
import tempfile
import unittest
import urllib.request
import urllib.response
from pathlib import Path
from unittest.mock import Mock, patch

from lite.gym.envs.osworld_2 import worker_client
from lite.gym.envs.osworld_2.worker_client import WorkerClient


class _Clock:
    def __init__(self):
        self.now = 0.0

    def sleep(self, delay):
        self.now += delay

    def monotonic(self):
        return self.now


class WorkerTransportTests(unittest.TestCase):
    def test_truncated_observation_retries_the_same_get(self):
        client = WorkerClient("http://worker.example", "test-token")
        payload = {"screenshot": base64.b64encode(b"saved-image").decode(), "step": 53}
        broken = io.BytesIO()
        broken.read = Mock(side_effect=http.client.IncompleteRead(b'{"screenshot":', 100))
        client.opener = Mock()
        client.opener.open.side_effect = [broken, io.BytesIO(json.dumps(payload).encode())]
        with patch("lite.gym.envs.osworld_2.worker_client.time.sleep"):
            result = client.observe("run", "saved-observation-53")
        self.assertEqual(result["screenshot"], b"saved-image")
        requests = [call.args[0] for call in client.opener.open.call_args_list]
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(r.get_method() == "GET" for r in requests))
        self.assertEqual(requests[0].full_url, requests[1].full_url)
        self.assertIn("observation_id=saved-observation-53", requests[1].full_url)

    def test_truncated_submission_reuses_operation_request_id(self):
        client = WorkerClient("http://worker.example", "test-token")
        client.request = Mock(
            side_effect=[
                http.client.IncompleteRead(b"{", 20),
                {"operation_id": "one-operation"},
                {"status": "succeeded", "result": {"step": 1}},
            ]
        )
        with patch("lite.gym.envs.osworld_2.worker_client.time.sleep"):
            result = client.action("run", "pyautogui.click(10, 10)")
        self.assertEqual(result, {"step": 1})
        calls = client.request.call_args_list
        self.assertEqual(calls[0].args, calls[1].args)
        self.assertEqual(calls[2].args, ("GET", "/operations/one-operation"))

    def test_redirect_does_not_forward_worker_credentials(self):
        requests = []

        class RedirectTransport(urllib.request.HTTPHandler):
            def http_open(self, request):
                requests.append(request)
                response = urllib.response.addinfourl(
                    io.BytesIO(b"redirect"),
                    {"Location": "http://another-host.example/credentials"},
                    request.full_url,
                    302,
                )
                response.msg = "Found"
                return response

        client = WorkerClient("http://worker.example", "private-token")
        redirect_handler = next(
            h for h in client.opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler)
        )
        client.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), redirect_handler, RedirectTransport()
        )
        with self.assertRaisesRegex(RuntimeError, "Worker HTTP 302"):
            client.request("GET", "/healthz")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].full_url, "http://worker.example/healthz")

    def test_confirmation_request_deadline_bounds_read_retries(self):
        client = WorkerClient("http://worker.example", "test-token")
        client.opener = Mock()
        client.opener.open.side_effect = http.client.IncompleteRead(b"{", 20)
        clock = _Clock()
        with (
            patch.object(worker_client.time, "monotonic", clock.monotonic),
            patch.object(worker_client.time, "sleep", clock.sleep),
            self.assertRaisesRegex(TimeoutError, "confirmation deadline"),
        ):
            client.request("GET", "/healthz", timeout=0.2)
        self.assertEqual(client.opener.open.call_count, 1)
        self.assertEqual(client.opener.open.call_args.kwargs["timeout"], 0.2)
        self.assertEqual(clock.now, 0)


class WorkerCloseConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.client = WorkerClient("http://worker.example", "test-token")
        self.health = {
            "worker_id": "worker-1",
            "release": "osworld-v2.1",
            "run_id": "run-1",
            "last_closed_run_id": "run-1",
            "state": "idle",
            "busy": False,
        }
        self.client.request = Mock(return_value=self.health)
        self.clock = _Clock()
        for name in ("monotonic", "sleep"):
            patcher = patch.object(worker_client.time, name, getattr(self.clock, name))
            patcher.start()
            self.addCleanup(patcher.stop)

    def wait(self, timeout=2):
        return self.client.wait_closed("run-1", "worker-1", "osworld-v2.1", timeout=timeout)

    def test_waits_for_matching_durable_receipt_and_idle(self):
        self.client.request.side_effect = [
            {**self.health, "run_id": None, "last_closed_run_id": None},
            {**self.health, "state": "active", "busy": True, "last_closed_run_id": None},
            {**self.health, "busy": True, "last_closed_run_id": None},
            self.health,
        ]
        self.assertEqual(self.wait(), self.health)
        self.assertEqual(self.client.request.call_count, 4)
        self.assertEqual(self.clock.now, 1.5)
        for call in self.client.request.call_args_list:
            self.assertEqual(call.args, ("GET", "/healthz"))
            self.assertGreater(call.kwargs["timeout"], 0)

    def test_false_idle_and_old_receipts_do_not_confirm_cleanup(self):
        for overrides in (
            {"run_id": None, "last_closed_run_id": None},
            {"last_closed_run_id": "older-run"},
            {"run_id": "older-run"},
            {"busy": True},
        ):
            with self.subTest(overrides=overrides):
                self.client.request.return_value = {**self.health, **overrides}
                with self.assertRaisesRegex(TimeoutError, "did not confirm cleanup"):
                    self.wait(timeout=0.25)

    def test_broken_worker_and_foreign_active_run_fail_immediately(self):
        for overrides, error in (
            ({"state": "broken"}, "Worker is broken"),
            ({"run_id": "foreign-run", "state": "active"}, "another run"),
            ({"run_id": "foreign-run", "busy": True}, "another run"),
            ({"worker_id": "different-worker"}, "different Worker or release"),
            ({"release": "different-release"}, "different Worker or release"),
        ):
            with self.subTest(overrides=overrides):
                self.client.request.return_value = {**self.health, **overrides}
                with self.assertRaisesRegex(RuntimeError, error):
                    self.wait()
        self.assertEqual(self.clock.now, 0)

    def test_worker_without_receipt_protocol_requires_upgrade(self):
        for missing in ("run_id", "last_closed_run_id"):
            with self.subTest(missing=missing):
                self.client.request.return_value = {
                    key: value for key, value in self.health.items() if key != missing
                }
                with self.assertRaisesRegex(RuntimeError, "upgrade the Worker"):
                    self.wait()

    def test_invalid_timeout_never_contacts_worker(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.wait(timeout=timeout)
        self.client.request.assert_not_called()


class WorkerReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        worker_path = Path(worker_client.__file__).parent / "worker"
        runtime_spec = importlib.util.spec_from_file_location(
            "worker_runtime", worker_path / "worker_runtime.py"
        )
        runtime = importlib.util.module_from_spec(runtime_spec)
        runtime_spec.loader.exec_module(runtime)
        service_spec = importlib.util.spec_from_file_location(
            "worker_service_for_transport_test", worker_path / "worker_service.py"
        )
        cls.service = importlib.util.module_from_spec(service_spec)
        with patch.dict("sys.modules", {"worker_runtime": runtime}):
            service_spec.loader.exec_module(cls.service)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        release_dir = Path(worker_client.__file__).parent / "releases/osworld-v2.1"
        release_env = patch.dict("os.environ", {"OSWORLD_RELEASE_DIR": str(release_dir)})
        release_env.start()
        self.addCleanup(release_env.stop)
        self.worker = self.service.Worker({"results": temporary.name, "worker_id": "worker-1"})
        self.addCleanup(self.worker.executor.shutdown)
        self.worker._rpc = Mock(return_value={"closed": True})
        self.worker._stop_process = Mock()

    def release_files(self, name="test-release"):
        path = self.worker.output / name
        path.mkdir()
        content = json.dumps(
            {"files": {"task_054.py": {"sha256": hashlib.sha256(b"task").hexdigest(), "size": 4}}}
        ).encode()
        (path / "task_hashes.json").write_bytes(content)
        (path / "release.json").write_text(
            json.dumps(
                {
                    "release": name,
                    "task_hash_manifest": {
                        "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
                        "task_count": 1,
                    },
                }
            )
        )
        return path

    def prepare_close(self):
        self.worker.run_id = "run-1"
        self.worker.task_id = "054"
        self.worker.run_dir = self.worker.output / "runs" / "run-1"
        self.worker.run_dir.mkdir(parents=True)
        self.worker.state = "evaluated"
        self.worker.busy = True
        self.worker.operations["close-1"] = {
            "method": "close",
            "run_id": "run-1",
            "status": "running",
        }

    def close(self):
        self.worker._execute("close-1", "close", {"request_id": "request-1", "run_id": "run-1"})

    def test_initial_idle_has_no_episode_close_receipt(self):
        health = self.worker.health()
        self.assertEqual(health["state"], "idle")
        self.assertIsNone(health["run_id"])
        self.assertIsNone(health["last_closed_run_id"])

    def test_release_manifest_owns_health_control_runtime_config_and_episode(self):
        original_worker = self.worker
        for name in ("osworld-v2-2026.08.08", "osworld-v2.1"):
            with self.subTest(release=name):
                self.worker = original_worker
                release_dir = self.release_files(name)
                with patch.dict("os.environ", {"OSWORLD_RELEASE_DIR": str(release_dir)}):
                    self.worker = self.service.Worker({"results": str(release_dir / "results")})
                self.addCleanup(self.worker.executor.shutdown)
                self.worker._rpc = Mock(return_value={"closed": True})
                self.worker._stop_process = Mock()
                self.assertEqual(self.worker.health()["release"], name)
                control = json.loads((self.worker.control / "control.json").read_text())
                self.assertEqual(control["release"], name)
                self.assertEqual(self.worker.config["release"], name)
                self.assertEqual(
                    self.worker.config["task_hashes"], str(release_dir / "task_hashes.json")
                )
                self.prepare_close()
                self.close()
                episode = json.loads((self.worker.run_dir / "episode.json").read_text())
                self.assertEqual(episode["release"], name)

    def test_bad_hash_or_task_count_fails_before_creating_worker_state(self):
        for failure in ("hash", "count"):
            with self.subTest(failure=failure):
                release_dir = self.release_files(failure)
                if failure == "hash":
                    with (release_dir / "task_hashes.json").open("a") as stream:
                        stream.write("\n")
                else:
                    path = release_dir / "release.json"
                    manifest = json.loads(path.read_text())
                    manifest["task_hash_manifest"]["task_count"] = 2
                    path.write_text(json.dumps(manifest))
                output = release_dir / "results"
                with (
                    patch.dict("os.environ", {"OSWORLD_RELEASE_DIR": str(release_dir)}),
                    self.assertRaisesRegex(ValueError, "does not match the Worker release"),
                ):
                    self.service.Worker({"results": str(output)})
                self.assertFalse(output.exists())

    def test_reset_accepts_only_manifest_tasks_without_a_hardcoded_task_range(self):
        release_dir = self.release_files()
        with patch.dict("os.environ", {"OSWORLD_RELEASE_DIR": str(release_dir)}):
            worker = self.service.Worker({"results": str(release_dir / "results")})
        self.addCleanup(worker.executor.shutdown)
        for task_id in ("001", "109", "../054", ["054"]):
            with self.subTest(task_id=task_id), self.assertRaises(self.service.APIError) as error:
                worker.submit(
                    "reset",
                    {"request_id": "rejected", "task_id": task_id, "run_id": "run-rejected"},
                    "admin",
                )
            self.assertEqual(error.exception.status, 400)
        self.assertEqual(worker.health()["state"], "idle")
        self.assertFalse((worker.output / "runs").exists())
        with patch.object(worker.executor, "submit") as execute:
            accepted = worker.submit(
                "reset",
                {"request_id": "accepted", "task_id": "054", "run_id": "run-accepted"},
                "admin",
            )
        execute.assert_called_once()
        self.assertIn("operation_id", accepted)
        self.assertEqual(worker.task_id, "054")

    def prepare_active(self, *, has_user_simulator=True):
        self.worker.run_id = "run-1"
        self.worker.task_id = "069"
        self.worker.run_dir = self.worker.output / "runs" / "run-1"
        self.worker.run_dir.mkdir(parents=True)
        self.worker.state = "active"
        self.worker.agent_token = "episode-token"
        self.worker.phase_index = 0
        self.worker.phase_count = 2
        self.worker.has_user_simulator = has_user_simulator
        self.worker.runtime = {"process_id": 1, "vm_container_id": "fake-vm"}

    def submit_and_execute(self, method, body, role="admin"):
        with patch.object(self.worker.executor, "submit") as scheduled:
            receipt = self.worker.submit(method, body, role)
        scheduled.assert_called_once()
        self.worker._execute(receipt["operation_id"], method, body)
        return receipt

    def test_every_phase_mutation_requires_a_current_integer_phase(self):
        self.prepare_active()
        for method in ("action", "observe", "ask_user", "evaluate", "advance"):
            for phase in (None, True, -1, "0", 1):
                body = {
                    "request_id": "invalid",
                    "run_id": "run-1",
                    "action": "DONE",
                    "question": "Could you clarify?",
                }
                if phase is not None:
                    body["phase_index"] = phase
                with self.subTest(method=method, phase=phase):
                    with self.assertRaises(self.service.APIError) as raised:
                        self.worker.submit(method, body, "admin")
                    self.assertEqual(
                        raised.exception.status, 409 if phase == 1 and type(phase) is int else 400
                    )
        self.assertFalse(self.worker.busy)
        self.assertEqual(self.worker.operations, {})
        self.worker._rpc.assert_not_called()

    def test_ask_user_is_idempotent_and_preserves_structured_evidence(self):
        self.prepare_active()
        self.worker._rpc.return_value = {
            "question": "Could you clarify?",
            "answer": "A simulated answer",
            "observation": {
                "screenshot": b"\x89PNG\r\n\x1a\nfixture",
                "instruction": "current phase",
                "user_response": "A simulated answer",
            },
        }
        body = {
            "request_id": "ask-1",
            "run_id": "run-1",
            "phase_index": 0,
            "question": "Could you clarify?",
        }
        receipt = self.submit_and_execute("ask_user", body, "agent")
        self.assertEqual(self.worker.submit("ask_user", body, "agent"), receipt)
        self.worker._rpc.assert_called_once_with("ask_user", {"question": "Could you clarify?"})
        self.assertEqual(self.worker.step, 0)
        operation = self.worker.operation(receipt["operation_id"], "agent")
        self.assertEqual(operation["status"], "succeeded")
        result = operation["result"]
        observed = self.worker.observation("run-1", result["observation"]["observation_id"])
        self.assertEqual(observed["user_response"], result["answer"])
        self.assertEqual(observed["phase_index"], 0)
        persisted = json.loads(
            (self.worker.run_dir / (receipt["operation_id"] + ".operation.json")).read_text()
        )
        self.assertEqual(persisted["result"]["question"], body["question"])
        self.assertEqual(persisted["result"]["answer"], "A simulated answer")
        self.assertNotIn("episode-token", json.dumps(persisted))
        self.worker.phase_index = 1
        self.assertEqual(self.worker.submit("ask_user", body, "agent"), receipt)
        with self.assertRaisesRegex(self.service.APIError, "current phase"):
            self.worker.submit("ask_user", {**body, "request_id": "late-ask"}, "agent")
        self.assertEqual(self.worker._rpc.call_count, 1)

    def test_missing_simulator_rejects_before_any_operation_is_created(self):
        self.prepare_active(has_user_simulator=False)
        with self.assertRaisesRegex(self.service.APIError, "no user simulator"):
            self.worker.submit(
                "ask_user",
                {
                    "request_id": "ask-1",
                    "run_id": "run-1",
                    "phase_index": 0,
                    "question": "Could you clarify?",
                },
                "agent",
            )
        self.assertEqual(self.worker.operations, {})
        self.worker._rpc.assert_not_called()
        self.assertEqual(self.worker.state, "active")

    def test_failed_simulator_is_an_operation_error_not_a_zero_score(self):
        self.prepare_active()
        self.worker._rpc.side_effect = RuntimeError("simulator backend unavailable")
        body = {
            "request_id": "ask-1",
            "run_id": "run-1",
            "phase_index": 0,
            "question": "Could you clarify?",
        }
        receipt = self.submit_and_execute("ask_user", body, "agent")
        operation = self.worker.operation(receipt["operation_id"], "agent")
        self.assertEqual(operation["status"], "failed")
        self.assertNotIn("result", operation)
        self.assertIsNone(self.worker.evaluation)
        self.assertEqual(self.worker.state, "broken")
        self.assertEqual(self.worker.submit("ask_user", body, "agent"), receipt)
        self.worker._rpc.assert_called_once()

    def test_evaluate_and_advance_retries_do_not_cross_phase_boundaries(self):
        self.prepare_active()
        evaluation = {
            "score": 0.35,
            "phase_score": 0.35,
            "phase_index": 0,
            "phase_count": 2,
            "complete": False,
            "gate_failed": False,
            "phase_results": [{"phase_index": 0, "score": 0.35}],
        }
        self.worker._rpc.return_value = evaluation
        evaluate_body = {"request_id": "evaluate-1", "run_id": "run-1", "phase_index": 0}
        evaluated = self.submit_and_execute("evaluate", evaluate_body)
        self.assertEqual(self.worker.submit("evaluate", evaluate_body, "admin"), evaluated)
        self.assertEqual(self.worker.state, "phase_evaluated")
        self.assertEqual(self.worker.task_snapshot("run-1", "admin")["evaluation"], evaluation)
        self.worker._rpc.return_value = {
            "phase_index": 1,
            "phase_count": 2,
            "has_user_simulator": True,
            "task_current_date": "2030-01-02",
            "observation": {
                "screenshot": b"\x89PNG\r\n\x1a\nfixture",
                "instruction": "second phase",
            },
        }
        advance_body = {"request_id": "advance-1", "run_id": "run-1", "phase_index": 0}
        advanced = self.submit_and_execute("advance", advance_body)
        self.assertEqual(self.worker.submit("advance", advance_body, "admin"), advanced)
        self.assertEqual(self.worker.submit("evaluate", evaluate_body, "admin"), evaluated)
        self.assertEqual(self.worker._rpc.call_count, 2)
        self.assertEqual(self.worker.phase_index, 1)
        self.assertEqual(self.worker.health()["phase_count"], 2)
        self.assertEqual(self.worker.latest_observation_ref["phase_index"], 1)
        self.assertEqual(
            self.worker.task_snapshot("run-1", "admin")["task_current_date"], "2030-01-02"
        )
        self.assertIsNone(self.worker.evaluation)
        self.assertEqual(self.worker.agent_token, "episode-token")
        with self.assertRaisesRegex(self.service.APIError, "current phase"):
            self.worker.submit("advance", {**advance_body, "request_id": "late-advance"}, "admin")
        with self.assertRaisesRegex(self.service.APIError, "different arguments"):
            self.worker.submit("advance", {**advance_body, "phase_index": 1}, "admin")

    def test_gate_failure_is_final_and_cannot_advance(self):
        self.prepare_active()
        self.worker._rpc.return_value = {
            "score": 0.1,
            "phase_score": 0.1,
            "complete": True,
            "gate_failed": True,
        }
        self.submit_and_execute(
            "evaluate", {"request_id": "eval", "run_id": "run-1", "phase_index": 0}
        )
        self.assertEqual(self.worker.state, "evaluated")
        with self.assertRaisesRegex(self.service.APIError, "non-final phase"):
            self.worker.submit(
                "advance", {"request_id": "advance", "run_id": "run-1", "phase_index": 0}, "admin"
            )
        self.worker._rpc.assert_called_once()

    def test_task_snapshot_is_admin_only_run_scoped_and_has_no_task_config(self):
        self.prepare_active()
        self.worker.config["user_simulator"] = {"knowledge": "private fixture knowledge"}
        with self.assertRaisesRegex(self.service.APIError, "Coordinator token"):
            self.worker.task_snapshot("run-1", "agent")
        with self.assertRaisesRegex(self.service.APIError, "matching active episode"):
            self.worker.task_snapshot("foreign-run", "admin")
        snapshot = self.worker.task_snapshot("run-1", "admin")
        self.assertEqual(snapshot["agent_token"], "episode-token")
        self.assertEqual(snapshot["phase_index"], 0)
        self.assertEqual(snapshot["runtime"], self.worker.runtime)
        self.assertNotIn("knowledge", json.dumps(snapshot))
        self.assertNotIn("agent_token", self.worker.health())

    def test_receipt_is_published_after_episode_write_finishes(self):
        self.prepare_close()
        original = self.service.write_json
        observed = []

        def write_and_inspect(path, value):
            original(path, value)
            if path.name == "episode.json":
                observed.append(self.worker.health())
                self.assertEqual(json.loads(path.read_text())["state"], "idle")

        with patch.object(self.service, "write_json", side_effect=write_and_inspect):
            self.close()
        self.assertEqual(len(observed), 1)
        self.assertTrue(observed[0]["busy"])
        self.assertIsNone(observed[0]["last_closed_run_id"])
        health = self.worker.health()
        self.assertEqual(health["run_id"], "run-1")
        self.assertEqual(health["last_closed_run_id"], "run-1")
        self.assertFalse(health["busy"])
        self.assertEqual(self.worker.operation("close-1", "admin")["status"], "succeeded")
        self.worker._rpc.assert_called_once_with("close", {})
        self.worker._stop_process.assert_called_once()

    def test_runtime_close_failure_never_publishes_receipt(self):
        self.prepare_close()
        self.worker._rpc.side_effect = RuntimeError("runtime close failed")
        self.close()
        health = self.worker.health()
        self.assertEqual(health["state"], "broken")
        self.assertIsNone(health["last_closed_run_id"])
        self.assertFalse(health["busy"])
        self.assertEqual(self.worker.operation("close-1", "admin")["status"], "failed")

    def test_episode_write_failure_is_not_a_successful_close(self):
        self.prepare_close()
        original = self.service.write_json

        def fail_episode(path, value):
            if path.name == "episode.json":
                raise OSError("storage unavailable")
            original(path, value)

        with patch.object(self.service, "write_json", side_effect=fail_episode):
            self.close()
        health = self.worker.health()
        self.assertEqual(health["state"], "broken")
        self.assertIsNone(health["last_closed_run_id"])
        self.assertFalse(health["busy"])
        operation = self.worker.operation("close-1", "admin")
        self.assertEqual(operation["status"], "failed")
        self.assertIn("could not be persisted", operation["error"])


if __name__ == "__main__":
    unittest.main()
