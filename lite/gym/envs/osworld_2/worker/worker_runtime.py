"""OSWorld adapter. Imported only inside a fresh process for one task episode."""

import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

logger = logging.getLogger(__name__)


def score_of(result):
    value = result.get("score") if isinstance(result, dict) else result
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Evaluator returned a non-finite or out-of-range score")
    return value


class OSWorldRuntime:
    def __init__(self, config):
        self.config = config
        self.env = None
        self.task = None
        self.task_current_date = None
        self.phases = []
        self.phase_index = 0
        self.phase_results = []
        self.evaluation = None

    def _wait_for_desktop(self, timeout=120):
        # The guest HTTP server can return a PNG while Xorg is still blank.
        # Wait for rendered desktop content before task setup or observation.
        from PIL import Image, ImageStat
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            observation = self.env._get_obs()
            with Image.open(io.BytesIO(observation["screenshot"])) as screenshot:
                variation = ImageStat.Stat(screenshot.convert("RGB").resize((64, 36))).stddev
            if max(variation) > 5:
                return observation
            time.sleep(2)
        raise TimeoutError("Guest screenshot remained blank while waiting for the desktop")

    def reset(self, task_id):
        repo = Path(self.config["repo"])
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        os.environ["OSWORLD_BENCHMARK_RELEASE"] = self.config["release"]
        os.environ["OSWORLD_FILE_BASE_URL"] = self.config["assets"]
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(
            filter(None, [os.environ.get("NO_PROXY"), os.environ.get("no_proxy"),
                          "localhost,127.0.0.1,::1"])
        )
        manifest = json.loads(Path(self.config["task_hashes"]).read_text())
        filename = f"task_{task_id}.py"
        source = repo / "evaluation_examples/task_class" / filename
        expected = manifest["files"][filename]
        content = source.read_bytes()
        if len(content) != expected["size"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise RuntimeError(f"Task {task_id} does not match the pinned release")
        from desktop_env.desktop_env import DesktopEnv
        from task_loader import load_task_from_file
        self.task = load_task_from_file(str(source))
        self.task_current_date = self.task.get("task_current_date")
        if self.task_current_date is None:
            self.task_current_date = getattr(self.task, "task_current_date", None)
        phase_factory = getattr(self.task, "get_phases", None)
        self.phases = (phase_factory() or []) if callable(phase_factory) else []
        if not isinstance(self.phases, list):
            raise TypeError("get_phases() must return a list")
        self.env = DesktopEnv(
            provider_name="docker", path_to_vm=self.config["vm"], headless=True,
            action_space="pyautogui", require_a11y_tree=False,
            cache_dir=str(Path(self.config["output"]) / "cache"),
            client_password=os.environ.get("OSWORLD_CLIENT_PASSWORD", "osworld-public-evaluation"),
            enable_proxy=self.config.get("enable_proxy", False),
            volume_size=self.task.get("volume_size"),
        )
        container_id = self.env.provider.container.id
        logger.info("Episode process=%s VM container=%s", os.getpid(), container_id)
        self._wait_for_desktop()
        self.env.reset(task_config=self.task)
        observation = self._wait_for_desktop()
        if self.phases:
            # Official Task069.setup already performs phase 1 setup.
            self.env.instruction = self.phases[0]["instruction"]
            observation["instruction"] = self.env.instruction
        return {"observation": observation, "phase_count": max(1, len(self.phases)),
                "phase_index": 0, "has_user_simulator": self.env.user_simulator is not None,
                "task_current_date": self.task_current_date,
                "runtime": {"process_id": os.getpid(), "vm_container_id": self.env.provider.container.id}}

    def action(self, action, pause=0.2):
        if self.evaluation is not None:
            raise ValueError("This phase was evaluated; advance the phase or close the episode")
        observation, reward, done, info = self.env.step(action, pause=pause)
        return {"observation": observation, "reward": reward, "done": done, "info": info}

    def observe(self):
        return {"observation": self.env._get_obs()}

    def ask_user(self, question):
        if self.evaluation is not None:
            raise ValueError("This phase was evaluated; advance the phase or close the episode")
        if self.env.user_simulator is None:
            raise ValueError("This task has no user simulator")
        answer = self.env.user_simulator.respond(question)
        observation = self.env._get_obs()
        observation["user_response"] = answer
        return {"question": question, "answer": answer, "observation": observation}

    def evaluate(self):
        if self.evaluation is not None:
            return self.evaluation
        if not self.phases:
            result = self.env.evaluate()
            value = score_of(result)
            self.evaluation = {"score": value, "phase_score": value, "result": result,
                               "phase_index": 0, "phase_count": 1, "complete": True}
        else:
            phase = self.phases[self.phase_index]
            result = phase["evaluate"](self.env)
            value = float(result)
            if not math.isfinite(value):
                raise ValueError("Evaluator returned a non-finite phase score")
            self.phase_results.append({"phase_index": self.phase_index,
                "name": phase.get("name", f"Phase {self.phase_index + 1}"),
                "score": value, "result": result})
            threshold = phase.get("gate_min_score")
            gate_failed = ((threshold is not None and value < float(threshold))
                           or (phase.get("gate", False) and value <= 0))
            self.evaluation = {
                "score": round(max(0.0, min(1.0, sum(p["score"] for p in self.phase_results))), 4),
                "phase_score": value, "phase_count": len(self.phases),
                "phase_index": self.phase_index, "phase_results": list(self.phase_results),
                "gate_failed": gate_failed,
                "complete": gate_failed or self.phase_index == len(self.phases) - 1,
            }
        return self.evaluation

    def advance(self):
        if not self.phases or self.evaluation is None or self.evaluation["complete"]:
            raise ValueError("No evaluated phase is available to advance")
        self.phase_index += 1
        phase = self.phases[self.phase_index]
        self.env._step_no = 0
        self.env.action_history.clear()
        self.env._traj_no += 1
        phase["setup"](self.env.setup_controller, use_proxy=self.env.current_use_proxy)
        self.env.is_environment_used = True
        pause = phase.get("pause_after_setup_seconds", 5)
        if pause:
            time.sleep(pause)
        self.env.instruction = phase["instruction"]
        self.evaluation = None
        return {"phase_index": self.phase_index, "phase_count": len(self.phases),
                "has_user_simulator": self.env.user_simulator is not None,
                "task_current_date": self.task_current_date,
                "observation": self.env._get_obs()}

    def close(self):
        if self.env is None:
            return {"closed": True}
        # DockerProvider.remove() omits anonymous volumes. Remember only this
        # episode's anonymous volumes so repeated runs do not exhaust the worker.
        provider = self.env.provider
        container = getattr(provider, "container", None)
        volumes = []
        if container is not None:
            container.reload()
            import re
            volumes = [m["Name"] for m in container.attrs.get("Mounts", [])
                       if m.get("Type") == "volume" and re.fullmatch(r"[a-f0-9]{64}", m.get("Name", ""))]
        self.env.close()
        if container is not None:
            from docker.errors import NotFound
            try:
                container.reload()
            except NotFound:
                pass
            else:
                raise RuntimeError("OSWorld VM container remained after close; replace the worker")
            for name in volumes:
                try:
                    provider.client.volumes.get(name).remove()
                except NotFound:
                    pass
        self.env = None
        logger.info("VM container removed and episode volumes cleaned")
        return {"closed": True}


def episode_process(connection, config, factory=OSWorldRuntime):
    """Process isolation also resets imported task globals between episodes."""
    log = open(Path(config["output"]) / "runtime.log", "a", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
    runtime = factory(config)
    try:
        while True:
            method, arguments = connection.recv()
            try:
                result = getattr(runtime, method)(**arguments)
                connection.send({"ok": True, "result": result})
            except Exception as error:
                import traceback
                traceback.print_exc()
                connection.send({"ok": False, "error": type(error).__name__ + ": " + str(error)})
            if method == "close":
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        try:
            runtime.close()
        finally:
            connection.close()
            log.close()
