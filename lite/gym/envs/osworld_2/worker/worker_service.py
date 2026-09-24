"""Authenticated, asynchronous OSWorld service with one active episode per worker."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import multiprocessing
import os
import re
import secrets
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from worker_runtime import OSWorldRuntime, episode_process

TOKEN_HEADER = "X-OSWorld-Session-Token"


class APIError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


def component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise APIError(400, "Expected a simple identifier, not a path")
    return value


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class Worker:
    def __init__(self, config, runtime_factory=OSWorldRuntime):
        # Validate the image's release before publishing any Worker state.
        release_dir = Path(os.environ.get("OSWORLD_RELEASE_DIR", "/opt/osworld-release")).resolve()
        manifest = json.loads((release_dir / "release.json").read_text())
        task_hashes = release_dir / "task_hashes.json"
        content = task_hashes.read_bytes()
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        if digest != manifest["task_hash_manifest"]["sha256"]:
            raise ValueError("Task hash manifest does not match the Worker release")
        files = json.loads(content)["files"]
        self.task_ids = {
            match[1] for name in files if (match := re.fullmatch(r"task_([0-9]{3})\.py", name))
        }
        if (len(files) != manifest["task_hash_manifest"]["task_count"]
                or len(self.task_ids) != len(files)):
            raise ValueError("Task catalog does not match the Worker release task count")
        self.config = {**config, "release": manifest["release"], "task_hashes": str(task_hashes)}
        self.runtime_factory = runtime_factory
        self.worker_id = component(config.get("worker_id") or "worker-" + uuid.uuid4().hex)
        self.output = Path(config["results"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.control = self.output / "_workers" / self.worker_id
        self.control.mkdir(parents=True, exist_ok=True)
        self.admin_token = secrets.token_urlsafe(32)
        self.agent_token = None
        self.run_id = None
        self.last_closed_run_id = None
        self.run_dir = None
        self.task_id = None
        self.state = "idle"
        self.busy = False
        self.operations = {}
        self.requests = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.process = None
        self.connection = None
        self.step = 0
        self.latest_observation = None
        self.latest_observation_ref = None
        self.phase_index = 0
        self.phase_count = 0
        self.has_user_simulator = False
        self.task_current_date = None
        self.runtime = None
        self.evaluation = None
        write_json(self.control / "control.json", {"worker_id": self.worker_id,
                   "token": self.admin_token, "release": self.config["release"],
                   "pid": os.getpid()})

    def role(self, token):
        if token and secrets.compare_digest(token, self.admin_token):
            return "admin"
        if token and self.agent_token and secrets.compare_digest(token, self.agent_token):
            return "agent"
        raise APIError(401, "Invalid worker or episode token")

    def health(self):
        with self.lock:
            return {"worker_id": self.worker_id, "release": self.config["release"],
                    "state": self.state,
                    "busy": self.busy, "ready": self.state != "broken",
                    "phase_index": self.phase_index, "phase_count": self.phase_count,
                    "has_user_simulator": self.has_user_simulator,
                    "run_id": self.run_id, "last_closed_run_id": self.last_closed_run_id}

    def task_snapshot(self, run_id, role):
        with self.lock:
            if role != "admin":
                raise APIError(403, "Coordinator token required")
            if run_id != self.run_id or self.state == "idle":
                raise APIError(409, "No matching active episode")
            return {"run_id": self.run_id, "task_id": self.task_id,
                    "state": self.state, "busy": self.busy,
                    "phase_index": self.phase_index, "phase_count": self.phase_count,
                    "has_user_simulator": self.has_user_simulator,
                    "runtime": self.runtime, "observation": self.latest_observation_ref,
                    "task_current_date": self.task_current_date,
                    "agent_token": self.agent_token, "evaluation": self.evaluation}

    def _rpc(self, method, arguments):
        self.connection.send((method, arguments))
        timeout = (
            self.config.get("setup_timeout", 1800)
            if method in ("reset", "advance")
            else self.config.get("operation_timeout", 300)
        )
        if not self.connection.poll(timeout):
            raise TimeoutError(f"{method} exceeded {timeout}s; replace this sandbox")
        response = self.connection.recv()
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def _stop_process(self):
        if self.process is not None:
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=3)
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None

    def _store_observation(self, observation, operation_id):
        screenshot = observation.pop("screenshot", None)
        if not isinstance(screenshot, bytes) or not screenshot.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("VM did not return a PNG screenshot")
        name = f"{self.step:05d}-{operation_id}"
        (self.run_dir / (name + ".png")).write_bytes(screenshot)
        value = {**observation, "run_id": self.run_id, "step": self.step,
                 "phase_index": self.phase_index, "phase_count": self.phase_count,
                 "screenshot_file": name + ".png"}
        write_json(self.run_dir / (name + ".json"), value)
        self.latest_observation = name
        self.latest_observation_ref = {
            "run_id": self.run_id, "step": self.step, "observation_id": name,
            "phase_index": self.phase_index, "phase_count": self.phase_count,
            "instruction": observation.get("instruction")}
        return self.latest_observation_ref

    def submit(self, method, body, role):
        if method not in ("reset", "action", "observe", "ask_user", "evaluate", "advance", "close"):
            raise APIError(404, "Unknown operation")
        if role != "admin" and method not in ("action", "observe", "ask_user"):
            raise APIError(403, "Coordinator token required")
        request_id = component(body.get("request_id"))
        fingerprint = hashlib.sha256(
            json.dumps([method, body], sort_keys=True).encode()
        ).hexdigest()
        with self.lock:
            if method in ("action", "observe", "ask_user", "evaluate", "advance"):
                if type(body.get("phase_index")) is not int or body["phase_index"] < 0:
                    raise APIError(400, "phase_index must be a nonnegative integer")
            if request_id in self.requests:
                previous, operation_id = self.requests[request_id]
                if previous != fingerprint:
                    raise APIError(409, "request_id was already used with different arguments")
                # A retry only reads its original receipt, including an advance
                # whose successful result has already changed the current phase.
                return {"operation_id": operation_id}
            if len(self.operations) >= 32768:
                raise APIError(409, "Worker operation limit reached; replace the sandbox")
            if self.busy:
                raise APIError(409, "Worker is busy; poll the current operation")
            if self.state == "broken":
                raise APIError(409, "Worker requires replacement after an environment error")
            if method == "reset":
                if self.state != "idle":
                    raise APIError(409, "Close the current episode before resetting")
                task_id = body.get("task_id")
                if not isinstance(task_id, str) or task_id not in self.task_ids:
                    raise APIError(400, "task_id is not in this Worker's release catalog")
                run_id = component(body.get("run_id"))
                run_dir = self.output / "runs" / run_id
                if run_dir.exists():
                    raise APIError(
                        409, "run_id already exists; existing results are never overwritten"
                    )
                run_dir.mkdir(parents=True, mode=0o700)
                self.run_id, self.task_id, self.run_dir = run_id, task_id, run_dir
                self.agent_token = secrets.token_urlsafe(32)
                self.step, self.latest_observation = 0, None
                self.latest_observation_ref = None
                self.phase_index, self.phase_count = 0, 0
                self.has_user_simulator = False
                self.task_current_date = None
                self.runtime, self.evaluation = None, None
                self.state = "starting"
            else:
                if body.get("run_id") != self.run_id or self.state == "idle":
                    raise APIError(409, "No matching active episode")
                if method != "close" and body["phase_index"] != self.phase_index:
                    raise APIError(409, "Request phase_index does not match the current phase")
                if method in ("action", "ask_user"):
                    if self.state != "active":
                        raise APIError(409, "Actions require an unevaluated active phase")
                if method == "ask_user":
                    if not self.has_user_simulator:
                        raise APIError(409, "This task has no user simulator")
                    if not isinstance(body.get("question"), str) or not 0 < len(body["question"]) <= 100000:
                        raise APIError(400, "question must be a nonempty string")
                if method == "action":
                    if (
                        not isinstance(body.get("action"), str)
                        or not 0 < len(body["action"]) <= 100000
                    ):
                        raise APIError(400, "action must be a nonempty PyAutoGUI code string")
                    pause = body.get("pause", 0.2)
                    if not isinstance(pause, (int, float)) or not 0 <= pause <= 10:
                        raise APIError(400, "pause must be between 0 and 10 seconds")
                if method == "advance" and self.state != "phase_evaluated":
                    raise APIError(409, "Evaluate the current non-final phase first")
            operation_id = uuid.uuid4().hex
            self.operations[operation_id] = {"operation_id": operation_id, "method": method,
                "run_id": self.run_id, "status": "running"}
            if method in ("action", "observe", "ask_user", "evaluate", "advance"):
                self.operations[operation_id]["phase_index"] = body["phase_index"]
            self.requests[request_id] = fingerprint, operation_id
            self.busy = True
            self.executor.submit(self._execute, operation_id, method, body)
            return {"operation_id": operation_id}

    def _execute(self, operation_id, method, body):
        try:
            if method == "reset":
                context = multiprocessing.get_context("spawn")
                self.connection, child = context.Pipe()
                config = {**self.config, "output": str(self.run_dir)}
                self.process = context.Process(
                    target=episode_process, args=(child, config, self.runtime_factory)
                )
                self.process.start()
                child.close()
                result = self._rpc("reset", {"task_id": self.task_id})
            elif method == "action":
                result = self._rpc(
                    "action", {"action": body["action"], "pause": body.get("pause", 0.2)}
                )
            elif method == "ask_user":
                result = self._rpc("ask_user", {"question": body["question"]})
            else:
                result = self._rpc(method, {})
            if method == "close":
                self._stop_process()
            with self.lock:
                if method in ("reset", "advance"):
                    self.phase_index = result["phase_index"]
                    self.phase_count = result["phase_count"]
                    self.has_user_simulator = result["has_user_simulator"]
                    self.task_current_date = result["task_current_date"]
                    self.evaluation = None
                    self.state = "active"
                    if method == "reset":
                        self.runtime = result["runtime"]
                elif method == "action":
                    self.step += 1
                elif method == "evaluate":
                    self.evaluation = result
                    self.state = "evaluated" if result["complete"] else "phase_evaluated"
                    write_json(self.run_dir / "result.json", result)
                    if result["complete"]:
                        (self.run_dir / "result.txt").write_text(str(result["score"]) + "\n")
                elif method == "close":
                    self.state, self.agent_token = "idle", None
                if "observation" in result:
                    result["observation"] = self._store_observation(result["observation"], operation_id)
                self.operations[operation_id].update(status="finishing", result=result)
            write_json(self.run_dir / (operation_id + ".operation.json"),
                       {**self.operations[operation_id], "status": "succeeded"})
            # Credentials never enter result artifacts or trajectories.
            if method == "reset":
                with self.lock:
                    self.operations[operation_id]["result"]["agent_token"] = self.agent_token
        except Exception as error:
            self.state = "broken"
            self._stop_process()
            message = type(error).__name__ + ": " + str(error)
            # Task setup may report URLs containing runtime credentials.
            for key, value in os.environ.items():
                if value and any(
                    word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")
                ):
                    message = message.replace(value, "[redacted]")
            with self.lock:
                self.operations[operation_id].update(status="failed", error=message)
            write_json(
                self.run_dir / (operation_id + ".operation.json"), self.operations[operation_id]
            )
        finally:
            with self.lock:
                operation = self.operations[operation_id]
                status = "succeeded" if operation["status"] == "finishing" else operation["status"]
                try:
                    with (self.run_dir / "trajectory.jsonl").open("a") as stream:
                        stream.write(json.dumps({"method": method, "request_id": body["request_id"],
                            "action": body.get("action"), "operation_id": operation_id,
                            "phase_index": body.get("phase_index", self.phase_index),
                            "status": status, "time": time.time()}) + "\n")
                    write_json(self.run_dir / "episode.json", {"run_id": self.run_id,
                        "task_id": self.task_id, "release": self.config["release"],
                        "worker_id": self.worker_id,
                        "phase_index": self.phase_index, "phase_count": self.phase_count,
                        "has_user_simulator": self.has_user_simulator,
                        "state": self.state, "steps": self.step})
                except Exception as error:
                    self.state = "broken"
                    operation.update(status="failed", error=(
                        f"Episode outcome could not be persisted ({type(error).__name__}); "
                        "replace this Worker"
                    ))
                else:
                    operation["status"] = status
                    if method == "close" and status == "succeeded":
                        # Publish only after runtime cleanup and durable episode output.
                        self.last_closed_run_id = self.run_id
                finally:
                    self.busy = False

    def operation(self, operation_id, role):
        with self.lock:
            value = self.operations.get(operation_id)
            if value is None:
                raise APIError(404, "Unknown operation")
            if role != "admin" and (
                value["method"] not in ("action", "observe", "ask_user") or value["run_id"] != self.run_id
            ):
                raise APIError(403, "Operation belongs to the coordinator or another episode")
            return value

    def observation(self, run_id, observation_id=None):
        with self.lock:
            if run_id != self.run_id or self.state in ("idle", "starting", "broken"):
                raise APIError(409, "No matching observable episode")
            name = component(observation_id or self.latest_observation)
            if not re.fullmatch(r"\d{5,}-[a-f0-9]{32}", name):
                raise APIError(400, "Invalid observation_id")
            path = self.run_dir / (name + ".json")
            if not path.is_file():
                raise APIError(404, "Unknown observation")
            value = json.loads(path.read_text())
            value["screenshot"] = base64.b64encode(
                (self.run_dir / value.pop("screenshot_file")).read_bytes()
            ).decode()
            value["screenshot_encoding"] = "base64/png"
            return value

    def shutdown(self):
        # Platform deletion terminates the enclosing daemon and remaining VMs.
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=3)


def make_server(worker, address):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _reply(self, status, body):
            content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _handle(self):
            parsed = urlsplit(self.path)
            if self.command == "GET" and parsed.path == "/healthz":
                return 200, worker.health()
            role = worker.role(self.headers.get(TOKEN_HEADER, ""))
            if self.command == "GET":
                if parsed.path.startswith("/operations/"):
                    return 200, worker.operation(component(parsed.path.rsplit("/", 1)[-1]), role)
                if parsed.path == "/observation":
                    query = parse_qs(parsed.query)
                    return 200, worker.observation(
                        query.get("run_id", [None])[0], query.get("observation_id", [None])[0]
                    )
                if parsed.path == "/task":
                    query = parse_qs(parsed.query)
                    return 200, worker.task_snapshot(query.get("run_id", [None])[0], role)
            if self.command == "POST":
                paths = {
                    "/task/reset": "reset",
                    "/action": "action",
                    "/ask_user": "ask_user",
                    "/observation": "observe",
                    "/task/evaluate": "evaluate",
                    "/task/advance": "advance",
                    "/task/close": "close",
                }
                if parsed.path not in paths:
                    raise APIError(404, "Unknown endpoint")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1048576:
                    raise APIError(413, "Expected a JSON body no larger than 1 MiB")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise APIError(400, "Expected a JSON object")
                return 202, worker.submit(paths[parsed.path], body, role)
            raise APIError(404, "Unknown endpoint")

        def do_GET(self):
            try:
                self.connection.settimeout(30)
                self._reply(*self._handle())
            except APIError as error:
                self._reply(error.status, {"error": error.message})
            except (ValueError, KeyError) as error:
                self._reply(400, {"error": str(error)})

        do_POST = do_GET

    return ThreadingHTTPServer(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()
    worker = Worker({"repo": "/opt/osworld", "assets": "/opt/osworld-data/assets",
                     "vm": "/opt/osworld-data/vm/osworld-v2-ubuntu-x86.qcow2",
                     "results": "/results", "worker_id": os.environ.get("OSWORLD_WORKER_ID"),
                     "setup_timeout": int(os.environ.get("OSWORLD_SETUP_TIMEOUT", "1800")),
                     "operation_timeout": int(os.environ.get("OSWORLD_OPERATION_TIMEOUT", "300"))})
    server = make_server(worker, (args.host, args.port))
    def stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        worker.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
