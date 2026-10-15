"""Adapter: run gym-anything (CUAWorld) tasks on cua-lite's sandbox runtime.

lite.cuaworld.* images build on the shared sandbox Dockerfile, built with the
desktop unix user ``ga`` (``USER=ga`` build-arg) so the upstream gym-anything
assets — which hardcode ``ga`` / ``/home/ga`` in task prompts, verifiers, and
setup scripts — do not need a ga->user transform. The cached materials remain
pristine; before execution, hook bodies receive deterministic CUA-Lite engine
guards and setup-only integration fixups. Unlike lite.osworld (which runs
directly as its desktop user), the exec-stdio session runs as ``root``
(``SandboxBaseEnv.EXEC_USER``): the fork's hooks launch GUI apps via
``su - ga -c`` (must not run as root), which is passwordless only from root.

Two responsibilities only:

* **run a hook** — write it to a file and ``bash <file>`` (NOT inline). The
  exec-stdio server runs commands as ``bash -c "<text>"``, so an inline hook's
  body lands in argv; a hook's ``pkill -f <app>`` would then match — and kill —
  the setup shell itself. From a file, argv is just the path.
* **run the verifier** — the original sync ``verifier.py`` runs host-side in a
  dedicated child process. The process is bounded by a hard timeout, and
  infrastructure failures (crash, import error, timeout, non-dict result, or
  non-finite score) raise ``CuaWorldVerifierError`` instead of being scored as
  agent failures. Its ``env_info`` handles (``copy_from_env`` etc.) bridge back
  to the parent's async exec-stdio interface. Final reward follows the task's
  upstream ``reward_type`` contract.

**Trust boundary:** ``verifier.py`` is upstream Python executed on the HOST (not
in the container), with host env vars / filesystem / network. The lock pins a
reviewed CUA-Lite materials commit; do not point this env at an untrusted
materials source.
"""

from __future__ import annotations

import asyncio
import ctypes
import functools
import importlib.util
import inspect
import json
import logging
import math
import multiprocessing
import os
import pickle
import re
import select
import shlex
import shutil
import signal
import stat
import struct
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from lite.gym.errors import CuaWorldVerifierError

logger = logging.getLogger(__name__)
_PROCESS_START_RESERVE = 0.05
_MAX_PROCESS_MESSAGE_BYTES = 256 * 1024 * 1024
_DEFAULT_VERIFIER_TIMEOUT = 600.0
_DEFAULT_PREPARATION_TIMEOUT = 180.0
_SESSION_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# cuaworld's server runs as root (`_CUAWorldEnv.EXEC_USER = "root"`) so the upstream
# gym-anything hooks/verifiers can self-drop via `su - ga` (passwordless ONLY from
# root). No per-command uid routing — the whole session is root.
#
# Keep the shell text in `_POST_START_COMMAND` free of software names and original
# hook paths. It is sent to exec-stdio as a single `bash -c` argument, so an
# upstream hook ending in `pkill -f <software>` can match that outer command if our
# comments mention the same string. The actual hook body is copied to a neutral
# temp file before execution for the same reason.

_POST_START_COMMAND = r"""
set -o pipefail
hook="$(
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path

path = Path("/workspace/env.json")
if path.is_file():
    value = (json.loads(path.read_text()).get("hooks") or {}).get("post_start", "")
    if value:
        print(value)
PY
)"
[ -z "$hook" ] || {
  [ ! -f "$hook" ] || chmod 755 "$hook"
  /usr/bin/python3 - "$hook" /tmp/cuaworld_ps_body.sh <<'PY'
import re, sys

__HELPER_GUARD_SRC__

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8", errors="surrogateescape").read()
try:
    library = open(
        "/workspace/scripts/task_utils.sh", encoding="utf-8", errors="surrogateescape"
    ).read()
except OSError:
    library = ""
text = re.sub(
    r'''__COUNT_RE__''',
    lambda m: "$( { " + m.group(1).strip() + "; } | awk 'NR==1' )",
    text,
)
text = _guard_helper_calls(text, _guardable_helper_names(library))
text = re.sub(r'''__SLOT_RE__''', r"\1${\2:-null}\3", text)
text = re.sub(
    r'''__SEARCH_RE__''',
    lambda m: "$( " + m.group(1).strip() + " || true )",
    text,
)
open(dst, "w", encoding="utf-8", errors="surrogateescape").write(text)
PY
  chmod 755 /tmp/cuaworld_ps_body.sh
  printf '%s\n' \
    'export DISPLAY=:1' \
    'export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
    'bash /tmp/cuaworld_ps_body.sh' > /tmp/cuaworld_post_start.sh
  # Do NOT redirect into the container: `run_command` captures stdout/stderr, and
  # swallowing them here made the rc!=0 warning log an EMPTY detail — a hook could
  # fail on every episode and leave no diagnostic anywhere.
  bash /tmp/cuaworld_post_start.sh 2>&1
}
"""


def _post_start_command() -> str:
    """`_POST_START_COMMAND` with the three hook-normalization patterns and the
    helper guard's own source injected, so the container-side rewrite and
    `_guard_post_start_body` can never drift apart."""
    return (
        _POST_START_COMMAND.replace("__COUNT_RE__", _COUNT_FALLBACK_RE.pattern)
        .replace("__HELPER_GUARD_SRC__", _helper_guard_source())
        .replace("__SLOT_RE__", _EMPTY_SLOT_RE.pattern)
        .replace("__SEARCH_RE__", _SEARCH_NOMATCH_RE.pattern)
    )


def create_episode_dir() -> Path:
    """Create the host-side episode directory consumed by upstream verifiers."""
    return Path(tempfile.mkdtemp(prefix="lite_cuaworld_episode_"))


def remove_episode_dir(episode_dir: Path | None) -> None:
    if episode_dir is not None:
        shutil.rmtree(episode_dir, ignore_errors=True)


async def run_cuaworld_post_start(
    computer: Any,
    *,
    timeout: float = 1800.0,
) -> None:
    """Run the environment-level post_start hook once per booted container.

    Upstream treats hook failures as warnings and continues reset, so this
    bridge preserves that behavior while moving the hook to the correct
    lifecycle phase where DISPLAY and boot-time services exist.
    """
    try:
        # cuaworld's server runs as root (EXEC_USER="root") so its `su - ga -c …`
        # hooks self-drop passwordlessly.
        result = await computer.interface.run_command(
            _post_start_command(),
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - upstream logs and continues
        logger.warning(
            "lite.cuaworld post_start hook could not run: %s: %s",
            type(exc).__name__,
            exc,
        )
        return
    returncode = getattr(result, "returncode", 0)
    if returncode:
        detail = (
            getattr(result, "stderr", "")
            or getattr(result, "stdout", "")
            or ""
        ).strip()
        logger.warning(
            "lite.cuaworld post_start hook failed (rc=%s): %s",
            returncode,
            detail[-500:],
        )


def _resolved_verifier_timeout(timeout: float | None) -> float:
    if timeout is not None:
        return max(0.0, float(timeout))
    override = os.environ.get("LITE_CUAWORLD_VERIFIER_TIMEOUT")
    if override is not None:
        try:
            return max(0.0, float(override))
        except ValueError:
            logger.warning(
                "invalid LITE_CUAWORLD_VERIFIER_TIMEOUT=%r; using derived default",
                override,
            )
    from lite.gym.envs.lite.cuaworld.src.vlm import get_vlm_config

    config = get_vlm_config()
    request_timeout = config.get("timeout")
    if request_timeout is None:
        return _DEFAULT_VERIFIER_TIMEOUT
    # `max_retries` is the compatibility key exposed by vlm.py; its value is
    # total provider attempts in this shim, so timeout budgeting multiplies by
    # attempts directly and only budgets backoff between attempts.
    attempts = max(1, int(config.get("max_retries", 3)))
    retry_backoff = sum(2 ** attempt for attempt in range(attempts - 1))
    return max(
        _DEFAULT_VERIFIER_TIMEOUT,
        float(request_timeout) * attempts + retry_backoff + 30.0,
    )


def _append_trajectory_event(episode_dir: Path, event: dict[str, Any]) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    with (episode_dir / "traj.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def initialize_episode(
    episode_dir: Path,
    *,
    env_id: str,
    lite_env_id: str,
    task_id: str,
    container_name: str | None,
    screenshot: bytes,
) -> None:
    """Write the upstream reset/session records and initial frame."""
    started_at = time.time()
    _append_trajectory_event(
        episode_dir,
        {
            "event": "reset",
            "ts": started_at,
            "seed": None,
            "env": env_id,
            "task": task_id,
            "use_cache": False,
            "cache_level": None,
            "use_savevm": None,
            "checkpoint_loaded": False,
        },
    )
    _append_trajectory_event(
        episode_dir,
        {
            "event": "session",
            "env_id": env_id,
            "lite_env_id": lite_env_id,
            "task_id": task_id,
            "container_name": container_name,
            "artifacts_dir": str(episode_dir),
        },
    )
    write_episode_frame(episode_dir, 0, screenshot)


def record_episode_step(
    episode_dir: Path,
    *,
    index: int,
    actions: list[dict[str, Any]],
) -> None:
    _append_trajectory_event(
        episode_dir,
        {
            "event": "step",
            "ts": time.time(),
            "idx": index,
            "action": actions,
        },
    )


def write_episode_frame(
    episode_dir: Path,
    index: int,
    screenshot: bytes,
) -> Path:
    destination = episode_dir / f"frame_{index:05d}.png"
    destination.write_bytes(screenshot)
    return destination


async def _capture_verification_screenshot(
    computer: Any,
    destination: Path,
    *,
    deadline: float,
    task: str,
    label: str,
) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CuaWorldVerifierError(
            f"{label} screenshot could not be captured before deadline",
            phase="screenshot",
            kind="timeout",
            task=task,
        )
    try:
        png = await asyncio.wait_for(
            computer.interface.screenshot(), timeout=remaining
        )
        from lite.gym.envs.lite.cuaworld.src.vlm import validated_image_mime

        await _call_in_process_before_deadline(
            validated_image_mime,
            (png,),
            deadline,
        )
        await _write_bytes_in_process_before_deadline(
            destination,
            png,
            deadline,
        )
    except CuaWorldVerifierError:
        raise
    except Exception as exc:  # noqa: BLE001 - no silent stale visual evidence
        raise CuaWorldVerifierError(
            f"{label} screenshot could not be captured: "
            f"{type(exc).__name__}: {exc}",
            phase="screenshot",
            kind="timeout" if isinstance(exc, TimeoutError) else "screenshot_failed",
            task=task,
        ) from exc


def _load_trajectory(episode_dir: Path) -> dict[str, Any]:
    """Mirror gym-anything VerificationRunner._load_traj."""
    steps: list[dict[str, Any]] = []
    traj_path = episode_dir / "traj.jsonl"
    if traj_path.exists():
        for line in traj_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                steps.append(item)

    frame_paths = [str(path) for path in sorted(episode_dir.glob("frame_*.png"))]
    final_png = episode_dir / "final.png"
    post_verification_png = episode_dir / "post_verification.png"
    step_frames: dict[int, str] = {}
    for frame_path in frame_paths:
        try:
            index = int(Path(frame_path).stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        step_frames[index] = frame_path
    return {
        "steps": steps,
        "episode_dir": str(episode_dir),
        "frames": frame_paths,
        "final_screenshot": str(final_png) if final_png.exists() else None,
        "post_verification_screenshot": (
            str(post_verification_png) if post_verification_png.exists() else None
        ),
        "first_frame": frame_paths[0] if frame_paths else None,
        "last_frame": frame_paths[-1] if frame_paths else None,
        "step_frames": step_frames,
    }


def _gated_call(
    gate_reader: Any,
    func: Callable[..., Any],
    args: tuple[Any, ...],
) -> None:
    try:
        gate_reader.recv()
    except (EOFError, OSError):
        return
    finally:
        gate_reader.close()
    func(*args)


class _MessagePipe:
    """Blocking child-process endpoint with explicit bounded framing."""

    def __init__(self, endpoint: Any):
        self._endpoint = endpoint

    def fileno(self) -> int:
        return self._endpoint.fileno()

    def close(self) -> None:
        self._endpoint.close()

    def send(self, value: Any) -> None:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > _MAX_PROCESS_MESSAGE_BYTES:
            raise ValueError("process message exceeds size limit")
        data = memoryview(struct.pack("!i", len(payload)) + payload)
        offset = 0
        while offset < len(data):
            written = os.write(self.fileno(), data[offset:])
            if written <= 0:
                raise BrokenPipeError
            offset += written

    def recv(self) -> Any:
        header = self._recv_exact(4)
        size = struct.unpack("!i", header)[0]
        if size < 0 or size > _MAX_PROCESS_MESSAGE_BYTES:
            raise ValueError(f"invalid process message size: {size}")
        return pickle.loads(self._recv_exact(size))

    def _recv_exact(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = os.read(self.fileno(), size - len(chunks))
            if not chunk:
                raise EOFError
            chunks.extend(chunk)
        return bytes(chunks)


def _cleanup_process(
    process: multiprocessing.Process,
    connections: tuple[Any, ...],
    *,
    tree: bool,
) -> None:
    try:
        if process.pid is not None:
            if tree and process.is_alive():
                _terminate_verifier_tree(process)
            elif process.is_alive():
                process.kill()
            process.join(0.2)
            if process.is_alive():
                process.kill()
                process.join(0.2)
            if not process.is_alive():
                process.close()
    finally:
        for connection in connections:
            if connection is not None:
                try:
                    connection.close()
                except (OSError, ValueError):
                    pass


def _cleanup_process_in_thread(
    process: multiprocessing.Process,
    connections: tuple[Any, ...] = (),
    *,
    tree: bool = False,
) -> None:
    threading.Thread(
        target=_cleanup_process,
        args=(process, connections),
        kwargs={"tree": tree},
        name="cuaworld-process-cleanup",
        daemon=True,
    ).start()


def _connection_ready(endpoint: Any) -> bool:
    try:
        return bool(select.select((endpoint.fileno(),), (), (), 0)[0])
    except (OSError, ValueError):
        return False


async def _start_before_deadline(
    process: multiprocessing.Process,
    deadline: float,
    connections: tuple[Any, ...],
    *,
    tree: bool,
    gate_reader: Any,
    gate_writer: Any,
) -> bool:
    remaining = deadline - time.monotonic()
    if remaining <= _PROCESS_START_RESERVE:
        for connection in connections:
            if connection is not None:
                connection.close()
        return False
    started = threading.Event()
    decided = threading.Event()
    state: dict[str, Any] = {"abandoned": False, "error": None}

    def start_and_own() -> None:
        try:
            process.start()
        except BaseException as exc:  # noqa: BLE001 - relayed to async caller
            state["error"] = exc
        finally:
            started.set()
        decided.wait()
        if state["abandoned"]:
            _cleanup_process(process, connections, tree=tree)

    threading.Thread(
        target=start_and_own,
        name="cuaworld-process-start",
        daemon=True,
    ).start()
    try:
        while not started.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                state["abandoned"] = True
                decided.set()
                return False
            await asyncio.sleep(min(0.002, remaining))
    except asyncio.CancelledError:
        state["abandoned"] = True
        decided.set()
        raise
    if state["error"] is not None:
        state["abandoned"] = True
        decided.set()
        raise state["error"]
    gate_reader.close()
    try:
        gate_writer.send(True)
    except BaseException:
        state["abandoned"] = True
        decided.set()
        raise
    finally:
        gate_writer.close()
    decided.set()
    return True


def _sync_worker(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result_writer: Any,
) -> None:
    try:
        result_writer.send((True, func(*args, **kwargs)))
    except BaseException as exc:  # noqa: BLE001 - isolated sync operation
        result_writer.send((False, type(exc).__name__, str(exc)))
    finally:
        result_writer.close()


async def _recv_before_deadline(
    endpoint: Any,
    deadline: float,
) -> Any:
    async def recv_exact(size: int) -> bytes:
        chunks = bytearray()
        loop = asyncio.get_running_loop()
        fd = endpoint.fileno()
        while len(chunks) < size:
            ready = loop.create_future()

            def read_ready() -> None:
                try:
                    chunk = os.read(fd, size - len(chunks))
                except BlockingIOError:
                    return
                except BaseException as exc:  # noqa: BLE001 - async fd relay
                    loop.remove_reader(fd)
                    if not ready.done():
                        ready.set_exception(exc)
                else:
                    loop.remove_reader(fd)
                    if not ready.done():
                        ready.set_result(chunk)

            loop.add_reader(fd, read_ready)
            try:
                chunk = await ready
            finally:
                loop.remove_reader(fd)
            if not chunk:
                raise EOFError
            chunks.extend(chunk)
        return bytes(chunks)

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError

    async def receive() -> Any:
        header = await recv_exact(4)
        size = struct.unpack("!i", header)[0]
        if size < 0 or size > _MAX_PROCESS_MESSAGE_BYTES:
            raise ValueError(f"invalid process message size: {size}")
        return pickle.loads(await recv_exact(size))

    os.set_blocking(endpoint.fileno(), False)
    try:
        return await asyncio.wait_for(receive(), timeout=remaining)
    except TimeoutError:
        endpoint.close()
        raise


async def _send_before_deadline(
    endpoint: Any,
    value: Any,
    deadline: float,
) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_PROCESS_MESSAGE_BYTES:
        raise ValueError("process message exceeds size limit")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    fd = endpoint.fileno()
    os.set_blocking(fd, False)

    async def send_all(data: bytes) -> None:
        view = memoryview(data)
        offset = 0
        loop = asyncio.get_running_loop()
        while offset < len(view):
            ready = loop.create_future()

            def write_ready() -> None:
                nonlocal offset
                try:
                    written = os.write(fd, view[offset:])
                except BlockingIOError:
                    return
                except BaseException as exc:  # noqa: BLE001 - async fd relay
                    loop.remove_writer(fd)
                    if not ready.done():
                        ready.set_exception(exc)
                else:
                    loop.remove_writer(fd)
                    if written <= 0:
                        if not ready.done():
                            ready.set_exception(BrokenPipeError())
                        return
                    offset += written
                    if not ready.done():
                        ready.set_result(None)

            loop.add_writer(fd, write_ready)
            try:
                await ready
            finally:
                loop.remove_writer(fd)

    try:
        await asyncio.wait_for(
            send_all(struct.pack("!i", len(payload)) + payload),
            timeout=remaining,
        )
    except TimeoutError:
        endpoint.close()
        raise


async def _call_in_daemon_thread_before_deadline(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    deadline: float,
) -> Any:
    """Run small blocking work without letting a stuck thread block shutdown."""
    finished = threading.Event()
    outcome: list[tuple[bool, Any]] = []

    def run() -> None:
        try:
            value = func(*args)
        except BaseException as exc:  # noqa: BLE001 - relayed to async caller
            outcome.append((False, exc))
        else:
            outcome.append((True, value))
        finally:
            finished.set()

    threading.Thread(
        target=run,
        name="cuaworld-deadline-call",
        daemon=True,
    ).start()
    while not finished.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        await asyncio.sleep(min(0.01, remaining))
    ok, payload = outcome[0]
    if ok:
        return payload
    raise payload


async def _call_in_process_before_deadline(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    deadline: float,
    *,
    kwargs: dict[str, Any] | None = None,
) -> Any:
    """Run potentially blocking sync work in a process that can be killed."""
    remaining = deadline - time.monotonic()
    if remaining <= _PROCESS_START_RESERVE:
        raise TimeoutError
    context = multiprocessing.get_context("spawn")
    result_reader, result_writer_connection = context.Pipe(duplex=False)
    result_writer = _MessagePipe(result_writer_connection)
    gate_reader, gate_writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_gated_call,
        args=(
            gate_reader,
            _sync_worker,
            (func, args, kwargs or {}, result_writer),
        ),
        daemon=True,
    )
    connections = (
        result_reader,
        result_writer,
        gate_reader,
        gate_writer,
    )
    locally_owned = True
    try:
        started = await _start_before_deadline(
            process,
            deadline,
            connections,
            tree=False,
            gate_reader=gate_reader,
            gate_writer=gate_writer,
        )
    except BaseException:
        locally_owned = False
        raise
    if not started:
        locally_owned = False
        raise TimeoutError
    result_writer.close()
    try:
        payload = await _recv_before_deadline(result_reader, deadline)
    finally:
        if locally_owned:
            result_reader.close()
            if process.is_alive():
                process.kill()
            _cleanup_process_in_thread(process)
    if payload[0]:
        return payload[1]
    _, error_type, message = payload
    raise RuntimeError(f"{error_type}: {message}")


def _read_optional_regular_text(path: Path) -> str | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"expected a regular file: {path}")
    return path.read_text()


def _read_regular_bytes(path: Path) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"expected a regular file: {path}")
    return path.read_bytes()


def _final_reward(
    task_spec: dict | None,
    *,
    raw_score: float,
    passed: bool,
) -> tuple[float, str]:
    """Apply gym-anything's final reward-type contract."""
    reward_type = (
        ((task_spec.get("init") or {}).get("reward_type", "sparse"))
        if task_spec is not None
        else "continuous"
    )
    bounded_score = max(0.0, min(100.0, raw_score))
    if reward_type == "sparse":
        reward = 1.0 if passed else 0.0
    elif reward_type in {"partial", "rubric"}:
        reward = bounded_score
    elif reward_type == "continuous":
        reward = bounded_score / 100.0
    elif passed:
        # A dense/weighted verifier that PASSED is worth 1.0, like sparse — its
        # raw_score is NOT a 0-100 percentage. EVERY registered, non-excluded
        # dense/weighted verifier gates `passed` on a threshold BELOW 100
        # (17/18/26/29/55/60/65/70/75; mean 63), so under `score/100` a run the
        # verifier certifies as solved would be worth 0.17-0.75 — while the sparse
        # tasks that make up ~95% of the corpus, whose verifiers apply the very same
        # internal partial criteria, pay 1.0 for the same outcome in the same
        # training batch. Worse, three
        # live verifiers return a raw count on a sub-100 scale — vlc/
        # foia_video_redaction (max_score 32), sports_coaching_video_analysis (46),
        # dailies_contact_sheet_pipeline (52) — so a FLAWLESS run would cap at
        # 0.32/0.46/0.52 forever; that is a unit error, not partial credit. This
        # also matches what the adapter already assumes elsewhere: the missing-score
        # fallback below is `100.0 if passed else 0.0`, so under `score/100` a
        # verifier that OMITS `score` would out-earn one that honestly reports 60.
        reward = 1.0
    else:
        # `dense` (113 tasks) and `weighted` (1) land here — REGISTERED basis, and
        # identical on the on-disk/parseable basis; 92 + 1 = 93 of them are also
        # non-excluded. Dense reward would be
        # accumulated per step by upstream `reward_shaping` — but the locked
        # CUAWorld materials declare NO reward_shaping, so there is nothing to
        # accumulate and returning 0.0 threw the verifier's verdict away: every one
        # of those 114 tasks scored 0 no matter what the agent did. Observed on
        # `openrocket/transonic_drag_optimization`, whose verifier returned
        # `passed: true, raw_score: 85` while the rollout recorded reward 0.0.
        # A FAILING run keeps the verifier's score as shaped partial credit.
        reward = bounded_score / 100.0
    return reward, reward_type


# --------------------------------------------------------------------------- #
# Verifier loading.
# --------------------------------------------------------------------------- #
def _install_vlm_shim() -> None:
    """Install both VLM modules exposed by pinned gym-anything."""
    import types

    from lite.gym.envs.lite.cuaworld.src import vlm, vlm_utils

    pkg = sys.modules.get("gym_anything") or types.ModuleType("gym_anything")
    pkg.vlm = vlm  # type: ignore[attr-defined]
    sys.modules["gym_anything"] = pkg
    sys.modules["gym_anything.vlm"] = vlm
    sys.modules["vlm_utils"] = vlm_utils


def _resolve_verifier_target(task_dir: Path, target: str) -> tuple[Path, str]:
    if not isinstance(target, str) or not target:
        raise ValueError("empty verifier target")
    if "::" in target:
        rel_text, func_name = target.split("::", 1)
    else:
        rel_text, func_name = "verifier.py", target
    if not rel_text or not func_name:
        raise ValueError(f"invalid verifier target: {target!r}")
    rel = PurePosixPath(rel_text)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe verifier target: {target!r}")
    root = task_dir.resolve()
    verifier_path = (root / Path(*rel.parts)).resolve()
    try:
        verifier_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"unsafe verifier target: {target!r}") from exc
    if not verifier_path.is_file():
        raise FileNotFoundError(f"verifier target does not exist: {rel_text}")
    return verifier_path, func_name


def _load_verifier(verifier_path: Path, func_name: str) -> Callable:
    """Load a host-side verifier callable from a task-local Python file.

    gym-anything verifiers are self-contained modules loaded by file location;
    they get their utilities through ``env_info`` + the ``gym_anything.vlm`` shim.
    """
    _install_vlm_shim()
    # Some verifiers import env-local helpers — either same-dir siblings, or a
    # shared package at the env root (e.g. `from calc_verification_utils import …`
    # living in <env>/utils/, or `from utils… import …`). Put the task dir, the
    # env root, and env-root/utils on sys.path so those imports resolve (the
    # caller restores sys.path/sys.modules afterward — see run_cuaworld_verify).
    tdir = verifier_path.parent            # .cache/<software>/<env>/tasks/<task>
    env_root = tdir.parent.parent          # .cache/<software>/<env>
    for p in (str(env_root / "utils"), str(env_root), str(tdir)):
        if Path(p).is_dir() and p not in sys.path:
            sys.path.insert(0, p)
    mod_name = f"cuaworld_verifier_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, verifier_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load verifier {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def _verifier_worker(
    verifier_path: str,
    verifier_target: str,
    traj: dict[str, Any],
    task_info: dict[str, Any],
    workdir: str,
    requests: Any,
    responses: Any,
    results: Any,
) -> None:
    """Child-process entry point. Container operations are RPCed to the parent."""
    if hasattr(os, "setsid"):
        os.setsid()
    try:
        func = _load_verifier(Path(verifier_path), verifier_target)
    except Exception as exc:  # noqa: BLE001 - upstream verifier import
        results.send({"error": f"load: {type(exc).__name__}: {exc}"})
        return

    request_id = 0
    rpc_lock = threading.Lock()

    def rpc(operation: str, *args: Any) -> Any:
        nonlocal request_id
        with rpc_lock:
            request_id += 1
            current_id = request_id
            requests.send((current_id, operation, args))
            response_id, ok, payload = responses.recv()
            if response_id != current_id:
                raise RuntimeError(
                    f"verifier RPC response mismatch: {response_id} != {current_id}"
                )
            if not ok:
                raise RuntimeError(payload)
            return payload

    def copy_from_env(container_src: str, host_dst: str | None = None) -> str:
        """Serve BOTH upstream call shapes, and always return the host path.

        Measured over the pinned materials by AST (every ``*.py`` under the 3216
        on-disk task dirs; all 67 are also registered AND non-excluded, so the
        figure is basis-invariant here): **67 call sites in 67 tasks, all
        sweet_home_3d**, do ``result_path = copy_from_env(src)`` and then open the
        returned path. The two-argument-only shim raised ``TypeError:
        copy_from_env() missing 1 required positional argument`` for every one of
        them, so the verifier died before scoring and a task the agent had visibly
        completed reported reward 0.

        Every other site passes an explicit ``host_dst`` (~3.9k on the same on-disk
        basis; the exact total moves with the scan definition — 6 ``.py`` files under
        the task dirs do not ``ast.parse`` — so it is deliberately not pinned here).
        Almost all discard the return, but
        exactly 2 TRUTHINESS-TEST it (`sumo/evaluate_phased_evacuation:53`
        ``success = copy_from_env(src, tmp); if success: return tmp``, and
        `astroimagej/extract_linear_shock_profile:105`), so returning ``None`` there
        made sumo's ``get_env_file`` unable to ever return a file — a guaranteed
        reward 0 on a registered, non-excluded task. Upstream's contract is
        truthy-on-success, so return the path unconditionally: correct for the two
        testers, invisible to every caller that discards it.

        The 1-arg path allocates that host file inside ``workdir`` (the per-episode
        dir), NOT in the process-wide temp dir. ``mkstemp`` has no finalizer and the
        67 sweet_home_3d callers never unlink what they are handed, so a bare
        ``tempfile.mkstemp()`` leaked one host file per call, per episode, for the
        whole lifetime of an env-server. Scoped to ``workdir`` the file dies with the
        episode (``remove_episode_dir``) while the returned path stays just as valid
        for the caller that opens it.
        """
        if host_dst is None:
            fd, host_dst = tempfile.mkstemp(
                dir=workdir, suffix=f"-{Path(container_src).name}"
            )
            os.close(fd)
        rpc("copy_from_env", container_src, host_dst)
        return host_dst

    def copy_to_env(host_src: str, container_dst: str) -> None:
        rpc("copy_to_env", host_src, container_dst)

    def exec_capture(command: str) -> str:
        return str(rpc("run_command", command))

    from lite.gym.envs.lite.cuaworld.src.vlm import (
        clear_vlm_provider_failures,
        get_final_screenshot,
        get_first_screenshot,
        pop_vlm_provider_failures,
        query_vlm,
        sample_trajectory_frames,
    )

    env_info: dict[str, Any] = {
        "env_id": task_info.pop("_env_id", None),
        "lite_env_id": task_info.pop("_lite_env_id", None),
        "copy_from_env": copy_from_env,
        "copy_to_env": copy_to_env,
        "exec_capture": exec_capture,
        "exec_in_env": exec_capture,
        "run_in_env": exec_capture,
        # Four spellings of the SAME judge. A verifier reads its handle with
        # `.get()`, so a missing alias is None, the branch is skipped, and the task
        # silently loses those points with no error anywhere. Census by AST over
        # every `.py` under the 3216 on-disk task dirs (a plain-text grep for
        # `env_info[...]`/`env_info.get(...)` returns the same key set, so the list
        # below is COMPLETE, not a sample): `query_vlm` 471, `vlm_query` 2, `vlm` 1,
        # `call_vlm` 1.
        # The three aliases cost 1 registered ∧ non-excluded task its whole gate and
        # 1 more its VLM points:
        #  * `vlc_media_player/corporate_training_localization_dubbing` (`vlm`) is
        #    STRUCTURALLY UNPASSABLE without this — only +20 and +15 are reachable,
        #    +25/+20/+20 all sit behind `if … and vlm`, and it needs `score >= 70`.
        #  * `slicer3d/split_segment_scissors` (`vlm_query`) is capped at 90/100
        #    against a `>= 65` gate.
        #  * `slicer3d/calculate_evans_index` (`vlm_query`, -15/100) and
        #    `measure_vertebral_compression_ratio` (`call_vlm`, -10/100) are already
        #    in `validation_excludes.json`, so they add no live tasks today — the
        #    aliases keep them correct if they are ever un-excluded.
        # Both call shapes in use (`vlm(prompt=…, image=path)` and
        # `vlm(prompt=…, images=frames)`) are already accepted by `query_vlm`, and
        # all four callers read `.get("success")`/`.get("parsed")` off its envelope.
        "query_vlm": query_vlm,
        "vlm": query_vlm,
        "vlm_query": query_vlm,
        "call_vlm": query_vlm,
        "sample_trajectory_frames": sample_trajectory_frames,
        "get_final_screenshot": get_final_screenshot,
        "get_first_screenshot": get_first_screenshot,
        "container": task_info.pop("_container", None),
        "episode_dir": workdir,
    }
    clear_vlm_provider_failures()
    try:
        result = func(traj, env_info, task_info)
        vlm_failures = pop_vlm_provider_failures()
        if vlm_failures:
            payload = {"error": f"VLMProviderError: {vlm_failures[-1]}"}
        else:
            payload = {"result": result}
    except BaseException as exc:  # noqa: BLE001 - isolated verifier execution
        # VLMProviderError intentionally inherits BaseException so upstream
        # verifier-local `except Exception` blocks cannot downgrade judge outages
        # into normal scores. A few upstream files use bare `except:`, which does
        # catch BaseException; the provider-failure side-channel above catches
        # that case after the verifier returns.
        pop_vlm_provider_failures()
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    results.send(payload)


def _verifier_supervisor(
    worker: Callable[..., None],
    verifier_path: str,
    verifier_target: str,
    traj: dict[str, Any],
    task_info: dict[str, Any],
    workdir: str,
    requests: Any,
    responses: Any,
    results: Any,
) -> None:
    """Run verifier code below a subreaper and clean it before forwarding."""
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            error = os.strerror(ctypes.get_errno())
            results.send({"error": f"could not isolate verifier descendants: {error}"})
            return
    context = multiprocessing.get_context("spawn")
    inner_reader, inner_writer = context.Pipe(duplex=False)
    verifier = context.Process(
        target=worker,
        args=(
            verifier_path,
            verifier_target,
            traj,
            task_info,
            workdir,
            requests,
            responses,
            inner_writer,
        ),
    )
    started = False
    try:
        verifier.start()
        started = True
        requests.close()
        responses.close()
        inner_writer.close()
        try:
            payload = inner_reader.recv()
        except (EOFError, OSError):
            payload = {
                "error": (
                    "verifier result pipe closed unexpectedly"
                    if verifier.exitcode is None
                    else f"verifier process exited with code {verifier.exitcode}"
                )
            }
        try:
            os.kill(verifier.pid, signal.SIGSTOP)
        except (ProcessLookupError, TypeError):
            pass
        _terminate_descendants(os.getpid())
        verifier.join(0.1)
        results.send(payload)
    finally:
        inner_reader.close()
        inner_writer.close()
        if started:
            if verifier.is_alive():
                verifier.kill()
            verifier.join(0.1)
            if not verifier.is_alive():
                verifier.close()


def _terminate_descendants(root_pid: int) -> None:
    """Freeze and kill only descendants listed by Linux procfs."""
    descendants: set[int] = set()
    # Freeze descendants before killing them so none can fork into the gap
    # between discovery and cleanup. A second scan catches children created
    # just before their parent received SIGSTOP.
    for _ in range(2):
        pending = [root_pid]
        seen = {root_pid}
        while pending:
            parent_pid = pending.pop()
            try:
                task_dirs = tuple(Path(f"/proc/{parent_pid}/task").iterdir())
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            for task_dir in task_dirs:
                try:
                    child_text = (task_dir / "children").read_text()
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
                for value in child_text.split():
                    child_pid = int(value)
                    if child_pid in seen:
                        continue
                    seen.add(child_pid)
                    descendants.add(child_pid)
                    try:
                        os.kill(child_pid, signal.SIGSTOP)
                    except ProcessLookupError:
                        continue
                    pending.append(child_pid)
    for child_pid in reversed(tuple(descendants)):
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _terminate_verifier_tree(process: multiprocessing.Process) -> None:
    """Freeze the verifier, terminate its descendants, then terminate it."""
    root_pid = process.pid
    if root_pid is None:
        return
    try:
        os.kill(root_pid, signal.SIGSTOP)
    except ProcessLookupError:
        return
    _terminate_descendants(root_pid)
    try:
        os.kill(root_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _write_bytes_before_deadline(
    destination: Path,
    data: bytes,
    deadline: float,
    *,
    temporary_path: Path | None = None,
) -> None:
    """Atomically write to a regular file without opening pipes or devices."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = 0o666
    try:
        existing = destination.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISREG(existing.st_mode):
            raise ValueError(f"copy destination is not a regular file: {destination}")
        mode = stat.S_IMODE(existing.st_mode)
    if temporary_path is None:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temporary_path = Path(temporary)
    else:
        fd = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    try:
        os.fchmod(fd, mode)
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            if time.monotonic() >= deadline:
                raise TimeoutError
            written = os.write(fd, view[offset:offset + 1024 * 1024])
            if written <= 0:
                raise OSError(f"short write to {destination}")
            offset += written
        if time.monotonic() >= deadline:
            raise TimeoutError
        os.close(fd)
        fd = -1
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        if fd >= 0:
            os.close(fd)


async def _write_bytes_in_process_before_deadline(
    destination: Path,
    data: bytes,
    deadline: float,
) -> None:
    temporary_path = destination.parent / (
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        await _call_in_process_before_deadline(
            _write_bytes_before_deadline,
            (destination, data, deadline),
            deadline,
            kwargs={
                "temporary_path": temporary_path,
            },
        )
    finally:
        temporary_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Hook execution, driven over the exec-stdio interface.
# --------------------------------------------------------------------------- #
#: Injected into every hook's python via PYTHONPATH. The upstream export scripts
#: interpolate SHELL booleans into PYTHON heredocs — `PHOTO_FOUND="false"` in bash
#: becomes `photo_found = false` in the generated program, which is a NameError, so
#: the script dies BEFORE writing /tmp/<task>_result.json and the verifier reports
#: "Could not retrieve result JSON" on a task the agent actually did. Naming the
#: three JSON spellings in builtins makes those programs parse as intended.
#: Affected hooks are export scripts spread across roughly a dozen software (the
#: exact count is detector-sensitive — independent scans of this corpus returned 8,
#: 38 and 170 depending on how "bare bool in a python heredoc" is matched, so treat
#: any single figure with suspicion; sweet_home_3d, blender3d, freecad, vscode and
#: eclipse dominate every reading). The ones that happen to assign "False" were
#: already fine and stay fine. All of them invoke python via an UNQUOTED heredoc, so
#: PYTHONPATH is inherited and the shim applies; note it does NOT survive a
#: `su - ga` login shell, which strips PYTHONPATH — no affected hook does that today.
#: CAVEAT: this converts a crash into a verdict, and the verdict is not always
#: right. `blender3d/radial_array_modifier_setup` writes
#: `"exists": $FILE_EXISTS == "true" if "$FILE_EXISTS" == "true" else False`; with
#: `true` bound, that evaluates `True == "true"` → False even when the file exists.
#: Materials are a pinned, gitignored HF snapshot — they cannot be patched in
#: place, so normalization belongs here, like the `setsid -w` wrapper below.
_HOOK_SITECUSTOMIZE = (
    "import builtins\n"
    "builtins.true, builtins.false, builtins.null = True, False, None\n"
)

#: `$(grep -c PATTERN FILE || echo "0")` is the corpus's idiom for "count, or 0 if
#: the file is missing" — and it is wrong: `grep -c` prints `0` AND exits 1 when it
#: matches nothing, so `||` fires too and the substitution yields `0\n0`. Spliced
#: into a JSON heredoc that produces `"extrude_count": 0\n0,` → the verifier's
#: `json.load` dies with `Expecting ',' delimiter`, discarding a real score.
#: 157 sites in 106 files across 16 software (measured over every `.sh` in the
#: pinned caches; ardour 63 and pycharm 16 dominate). Wrapping the whole substitution in
#: `{ …; } | awk 'NR==1'` keeps both intents exactly: grep's count wins when it
#: printed one, and the `echo` fallback still wins when grep printed nothing (file
#: absent). The leading `[^()]*?` matters — without it the equally-broken piped form
#: `$(echo "$OUT" | grep -c " PASSED" || echo "0")` is missed (23 such sites in
#: pycharm/slicer3d/dbeaver/vlc/vscode/eclipse/openemr). Measured strict superset:
#: on all 92 files the anchored form already covered, the rewrite is byte-identical,
#: and `bash -n` passes on every rewritten file.
#: Every matched site targets exactly ONE file, so `NR==1` truncates nothing — a
#: multi-file/recursive `grep -c` prints `path:N` per line, and none occur here.
_COUNT_FALLBACK_RE = re.compile(
    r"\$\(\s*([^()]*?(?:grep|egrep|fgrep|pgrep)\s+-[a-zA-Z]*c\b[^()]*?\|\|\s*echo\s+[^()]*?)\s*\)"
)


#: `WID=$(xdotool search --name "GMAT" 2>/dev/null | head -1)` under `set -euo
#: pipefail`: `xdotool search` exits 1 when it finds NO window, the pipeline inherits
#: that, the assignment inherits the pipeline, and `set -e` aborts the whole setup.
#: So these scripts fail hardest when they are WORKING — coppeliasim's probe looks
#: for a Welcome dialog its own config suppresses, and gmat's poll loop can never
#: reach its second iteration; the abort lands BEFORE the cleanup tail, so a warm-up
#: instance of the app survives into the agent's episode.
#: This applies to POST_START ONLY, and that is the whole point. Of the 222 xdotool
#: `search` sites in task-dir hooks (on-disk basis: every `.sh` under the 3216 task
#: dirs in the pinned caches; 217 files), ZERO set `pipefail` and 218 set no `-e` at
#: all — and the 4 that do set `-e` still have no `pipefail`, so the pipeline already
#: takes `head`'s rc=0 and rewriting any of them is a pure no-op. The only two files
#: in the entire corpus where `xdotool search` sits under both `-e` and `pipefail`
#: are `coppeliasim_env/scripts/setup_coppeliasim.sh` (4 sites) and
#: `gmat_env/scripts/setup_gmat.sh` (1) — image-baked `/workspace/scripts` files that
#: `_run_task_hook` never walks, reached only as env.json `post_start`.
#: Blast radius, ON-DISK basis: coppeliasim 81 + gmat 75 = 156 task dirs whose setup
#: ran past line 90 for the first time. On the REGISTERED ∧ NON-EXCLUDED basis (what
#: a rollout actually samples) the same fix is worth coppeliasim 33 + gmat 57 = 90
#: tasks. Both figures describe this one change; they differ only in denominator.
_SEARCH_NOMATCH_RE = re.compile(
    r"\$\(\s*((?:[A-Z_]+=\S+\s+)*xdotool\s+search\b[^()]*?)\s*\)"
)


#: ── The `task_utils.sh` non-fatal `return 1` class ───────────────────────────
#: 35 of the 40 cuaworld software ship a `scripts/task_utils.sh` shell library. It is
#: image-baked at `/workspace/scripts/` and `source`d by 5304 of the 6375 task-dir
#: `.sh` hooks (ON-DISK basis: every `.sh` under the 3216 task dirs in the pinned
#: caches); 1166 of those sourcing hooks run under `set -e`. Because `set -o` state
#: belongs to the CALLER, a helper that ends a failure path with `return 1` kills its
#: caller mid-setup — even when the author's own message says the condition is not an
#: error. `wait_for_slew_complete` (kstars_sim, prints `WARNING: Slew did not complete
#: within Ns`) was the first instance found; it is one of **56 (software, helper)
#: pairs** with the same defect, so the guard is derived from the LIBRARY instead of
#: naming one function — `_guardable_helper_names` re-reads the pinned
#: `task_utils.sh` and the set can never fall behind a materials bump.
#:
#: SCOPE — deliberately WARNING ∪ SILENT, never ERROR. Guarding an ERROR-intent
#: `return 1` would convert a crash into a silently wrong verdict, which is strictly
#: worse than the crash. So a helper is guarded only when EVERY one of its non-zero
#: exits is announced as a `WARNING`/`⚠` or taken silently (`echo ""; return 1`, or no
#: print at all); one `ERROR`/`❌`/`Failed` message anywhere disqualifies the whole
#: helper. That drops e.g. `imagej/launch_fiji` ("ERROR: Fiji not found"), and the
#: plain-`Timeout:` helpers of knime/odoo/openemr/qgis/sumo, which state a fact
#: without ruling on severity (13 more live tasks; left out on purpose).
#:
#: BLAST RADIUS of the guard as scoped (measured by `_guard_helper_calls` over the
#: pinned materials): 781 sites in 512 files → **507 on-disk / 448 registered / 354
#: registered ∧ non-excluded** task dirs, against 50 live for the single-name rule it
#: replaces. Aborting there loses, on the on-disk basis, the initial screenshot in
#: 502 tasks, focus/maximize in 503, dialog dismissal in 232 and a fixture write in
#: 112 (live: 349 / 353 / 159 / 91).
#:
#: TWO SHAPES, because only one of them is a bare call. REPRODUCED in real bash
#: against the real library on the exemplar
#: `openvsp/openvsp_fuselage_lofting/setup_task.sh:58` — `WID=$(wait_for_openvsp 60)`
#: under `set -e`: as shipped the assignment inherits the substitution's rc=1 and the
#: hook dies before its first screenshot; with `|| true` the author's whole `else`
#: branch ("WARNING: OpenVSP did not appear …" + screenshot) runs and rc is 0. A
#: whole-line bare-call anchor cannot see that line at all, which is why 121 of the
#: 354 live tasks are reachable ONLY through the assignment shape. `local`/`export`
#: assignments are NOT matched and must not be: the builtin's own rc masks the
#: substitution's, so `local x=$(f)` never aborts (verified in bash).
#:
#: Both patterns are anchored to a WHOLE LINE: `|` `&` `;` `(` `)` `<` `>` `#` are
#: excluded from the argument run, so an already-guarded call, a backgrounded one, an
#: `if wait_for_slew_complete …; then` and a trailing-comment line are left alone.
#: Applied ONLY to hooks that actually set `-e` (`set -e` 1325 / `set -euo pipefail`
#: 239 are the only errexit spellings in the corpus; `set -o pipefail` 79 is not one)
#: — elsewhere the rewrite would be a no-op except for the script's final rc, which
#: `_run_hook`'s caller reads. Measured: of the 781 guarded sites, ZERO are the last
#: statement of their script or of an enclosing function, so no script exit status
#: and no shell function's return value changes.
def _guardable_helper_names(library: str) -> tuple[str, ...]:
    """Helpers in one `task_utils.sh` whose every failure exit the AUTHOR marked
    non-fatal: announced as a WARNING, or taken silently.

    Kept dependency-free (only `re`), self-contained and pure ASCII: this function's
    own SOURCE is shipped into the container by `_helper_guard_source`, so the
    post_start rewrite runs the identical rule. The two pictographs the corpus uses
    as severity markers are spelled as escapes for that reason.
    """
    opener = re.compile(
        r"^(?:function[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)[ \t]*\([ \t]*\)[ \t]*\{[ \t]*$"
    )
    returns = re.compile(r"^[ \t]*return[ \t]+([0-9]+)[ \t]*$")
    prints = re.compile(r"^[ \t]*(?:echo|printf)[ \t]+(.*?)[ \t]*$")
    lines = library.splitlines()
    names: list[str] = []
    index = 0
    while index < len(lines):
        opened = opener.match(lines[index])
        index += 1
        if opened is None:
            continue
        # Every library defines its helpers at column 0, so the body ends at the
        # first unindented `}`; a nested block's closer is always indented.
        body: list[str] = []
        while index < len(lines) and lines[index] != "}":
            body.append(lines[index])
            index += 1
        spoken: list[str] = []
        failures: list[str] = []
        for line in body:
            said = prints.match(line)
            if said is not None:
                spoken.append(said.group(1))
                continue
            exited = returns.match(line)
            if exited is None:
                continue
            if exited.group(1) != "0":
                # Intent is whatever the helper printed since its previous `return`,
                # not just the line above: the message and the exit are often
                # separated by a log dump (`tail -20 /tmp/x.log; return 1`).
                announced = " ".join(spoken)
                failures.append(
                    "warning"
                    if re.search("WARNING|Warning|\u26a0", announced)
                    else "error"
                    if re.search("ERROR|Error|\u274c|Fail", announced)
                    else "silent"
                    if not re.search(r"[^\s\"']", announced)
                    else "other"
                )
            spoken = []
        if failures and set(failures) <= {"warning", "silent"}:
            names.append(opened.group(1))
    return tuple(sorted(set(names)))


def _helper_guard_patterns(names: tuple[str, ...]) -> tuple[str, str]:
    """`(bare-call, assignment)` regexes for the two whole-line shapes in which a
    call to one of `names` propagates rc=1 to a `set -e` caller."""
    call = (
        "(?:"
        + "|".join(sorted(map(re.escape, names), key=len, reverse=True))
        + r")(?![A-Za-z0-9_])(?:[ \t]+[^\r\n|&;()<>#]*?)?"
    )
    # A trailing REDIRECTION must not defeat the anchor. The argument run above
    # excludes `<>` so a pipeline or a process substitution cannot be swallowed,
    # but that also killed the match on `helper > /dev/null`, and
    # `openlca/waste_treatment_linkage_setup:20` (`ensure_uslci_database >
    # /dev/null`) went unguarded and still aborts its setup. A redirection is
    # neither a pipeline nor a logical operator, so `cmd > /dev/null || true` keeps
    # the intended semantics. Covers `>`, `>>`, `2>&1`, `&>`; still refuses `|`,
    # `&&`, `;`, an existing `|| true`, and the `if cmd; then` form.
    redirect = r"(?:[ \t]*(?:&>>?|[0-9]?>>?|[0-9]?>&[0-9-]?)[ \t]*[^\s|&;()<>#]*)*"
    return (
        r"(?m)^([ \t]*" + call + redirect + r")[ \t]*$",
        r"(?m)^([ \t]*[A-Za-z_][A-Za-z0-9_]*=\"?\$\([ \t]*"
        + call
        + r"[ \t]*\)\"?" + redirect + r")[ \t]*$",
    )


def _guard_helper_calls(text: str, names: tuple[str, ...]) -> str:
    """Append `|| true` to every unguarded call to `names`, in a hook that runs
    under `set -e`: the only place their non-fatal `return 1` aborts anything."""
    if not names or not re.search(
        r"(?m)^[ \t]*set[ \t]+-(?:[a-zA-Z]*e[a-zA-Z]*|o[ \t]+errexit)\b", text
    ):
        return text
    for pattern in _helper_guard_patterns(names):
        text = re.sub(pattern, r"\1 || true", text)
    return text


@functools.lru_cache(maxsize=None)
def _library_helpers(library_path: str) -> tuple[str, ...]:
    try:
        library = Path(library_path).read_text(
            encoding="utf-8", errors="surrogateescape"
        )
    except OSError:
        return ()
    return _guardable_helper_names(library)


def _hook_helpers(task_dir: Path) -> tuple[str, ...]:
    """Guardable helper names of the `task_utils.sh` this task's env bakes.

    `task_dir` is `.cache/<software>/<env>/tasks/<task>`, so the library the
    container has at `/workspace/scripts/task_utils.sh` is a sibling two levels up.
    Software without one (5 of 40) simply guard nothing."""
    return _library_helpers(str(task_dir.parents[1] / "scripts" / "task_utils.sh"))


def _helper_guard_source() -> str:
    """The three helper-guard functions as SOURCE, for the container-side post_start
    rewrite. Shipping the code (not a rendered pattern) is what lets that path derive
    its own names from the baked `/workspace/scripts/task_utils.sh` while staying
    provably the same rule as the host-side guard."""
    return "\n".join(
        inspect.getsource(function)
        for function in (
            _guardable_helper_names,
            _helper_guard_patterns,
            _guard_helper_calls,
        )
    )


#: The residual of the `_HOOK_SITECUSTOMIZE` class that no builtins binding can
#: reach. The shim fixes a slot whose shell value is the WORD `true`/`false`/`null`;
#: it cannot fix a slot whose value is the EMPTY STRING, because `"key": ,` is a
#: SyntaxError (python heredoc) / a JSONDecodeError (the far more common `cat >
#: "$RESULT_JSON" << EOF` heredoc) before any name is ever looked up. Same symptom
#: as the shim's class — "Could not retrieve result JSON" on a task the agent did.
#: REPRODUCED, real bash + real python3 in `cua-lite/lite.cuaworld.base`, on
#: `slicer3d/convert_volume_nifti/export_result.sh`: `NIFTI_DIMENSIONS=""` is only
#: overwritten inside `if [ -f /tmp/nifti_validation.json ]`, so when the agent
#: produces no NIfTI the final heredoc emits `"nifti_dimensions": ,` (5 such keys)
#: and `json.load` dies with `Expecting value: line 13 column 25`.
#: `${VAR:-null}` is the corpus's OWN default for these slots — that same file
#: writes `|| echo "null"` on the guarded path and initializes `SOURCE_DIMENSIONS=
#: "null"` — so this restores the author's intent exactly where their guard missed.
#: BASIS (every `.sh` under the 3216 task dirs). This shape — a whole line whose
#: value is exactly one bare `$VAR` after a double-quoted key and a colon — occurs
#: 13939 times: 10598 in `cat > <var-named file> << EOF` heredocs, 2751 in
#: `cat > *.json`, 566 in python heredocs, 24 in `cat > *.py`. On-disk 2090 tasks →
#: 1988 registered → **1510 registered ∧ non-excluded**. That is the population the
#: rule makes empty-safe, NOT a defect count: a slot only misfires when its producer
#: actually expands empty. Detector sensitivity is the finding here — scans that
#: counted "bare bool in a python heredoc" instead returned 8 / 38 / 170, and the
#: python-heredoc-only slice of THIS shape is 566 sites / 138 reg ∧ non-excluded.
#: Shape restricted to the `"key": $VAR` member form ON PURPOSE. The sibling
#: `name = $VAR` form (61 in python heredocs) is NOT rewritten: it is ambiguous
#: outside python — 51 of its occurrences are not in a heredoc at all, and the one
#: hit in an image-baked `post_start` script is `Exec=$VSPBIN` in a `.desktop`
#: heredoc (openvsp `scripts/setup_openvsp.sh:57`), where `null` would be wrong.
#: The `"key": $VAR` shape has ZERO occurrences in any `.cache/*/*/scripts/*.sh`, so
#: applying this in the shared shell-idiom guard (task hooks and `post_start`,
#: which does NOT export the shim's PYTHONPATH) cannot emit a `null` that no
#: builtin defines.
#: Strictly non-behavioural when the value is non-empty: `${VAR:-null}` expands to
#: exactly `$VAR`. Verified over the whole corpus — `bash -n` passes on every
#: rewritten file, and re-running the 185-hook reproduction corpus with the rewrite
#: applied produced zero new python/JSON errors.
_EMPTY_SLOT_RE = re.compile(
    r'(?m)^([ \t]*"[A-Za-z_][A-Za-z0-9_]*"[ \t]*:[ \t]*)'
    r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?([ \t]*,?[ \t]*)$"
)

_SLICER_MODAL_DISMISSAL = """\
# CUA-Lite Slicer startup modal normalization.
for key in Return Escape space; do
    DISPLAY=:1 xdotool key "$key" 2>/dev/null || true
    sleep 0.3
done
"""
_SLICER_WAIT_RE = re.compile(
    r"(?m)^([ \t]*wait_for_slicer[ \t]+[0-9]+[ \t]*(?:#.*)?\n)"
    r"(?![ \t]*# CUA-Lite Slicer startup modal normalization\.)"
)
_SLICER_WMCTRL_RE = re.compile(
    r'(?m)^([ \t]*(?:DISPLAY=:1[ \t]+)?wmctrl[ \t]+-a[ \t]+"Slicer"[^\n]*\n)'
    r"(?![ \t]*# CUA-Lite Slicer startup modal normalization\.)"
)


def _literal_block_pattern_ignore_blank_line_ws(text: str) -> str:
    pieces: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.endswith("\n"):
            body = line[:-1]
            pieces.append(r"[ \t]*\n" if not body.strip() else re.escape(line))
        else:
            pieces.append(r"[ \t]*" if not line.strip() else re.escape(line))
    return "".join(pieces)


_ASTRO_ALIGN_SLOW_EXTRACT_BLOCK = """import os
import glob
import tarfile
import shutil
import numpy as np
from astropy.io import fits

WASP12_CACHE = "/opt/fits_samples/WASP-12b_calibrated.tar.gz"
RAW_DIR = "/home/ga/AstroImages/raw_sequence"

if not os.path.exists(WASP12_CACHE):
    print(f"ERROR: Cached data not found at {WASP12_CACHE}")
    exit(1)

# Extract first 20 FITS files to a temporary directory
TMP_EXTRACT = "/tmp/wasp_extract"
os.makedirs(TMP_EXTRACT, exist_ok=True)

print("Extracting frames from archive...")
with tarfile.open(WASP12_CACHE, 'r:gz') as tar:
    members = [m for m in tar.getmembers() if m.name.endswith('.fits')]
    members = sorted(members, key=lambda x: x.name)[:20]
    tar.extractall(path=TMP_EXTRACT, members=members)

extracted_files = sorted(glob.glob(f"{TMP_EXTRACT}/**/*.fits", recursive=True))

print(f"Injecting severe tracking drift into {len(extracted_files)} frames...")
for i, fpath in enumerate(extracted_files):
    # Read data
    with fits.open(fpath) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header

        # Inject artificial drift (dx = 3px/frame, dy = 2px/frame)
        dy, dx = i * 2, i * 3

        shifted = np.zeros_like(data)
        if dy > 0 and dx > 0:
            shifted[dy:, dx:] = data[:-dy, :-dx]
        else:
            shifted = data.copy()

        # Write to raw_sequence directory
        out_name = f"drifting_frame_{i:02d}.fits"
        out_path = os.path.join(RAW_DIR, out_name)
        fits.writeto(out_path, shifted, hdr, overwrite=True)

# Cleanup temp
shutil.rmtree(TMP_EXTRACT)
print("Drifting sequence prepared successfully.")
"""
_ASTRO_ALIGN_SLOW_EXTRACT_RE = re.compile(
    _literal_block_pattern_ignore_blank_line_ws(_ASTRO_ALIGN_SLOW_EXTRACT_BLOCK)
)
_ASTRO_ALIGN_STREAMING_EXTRACT_BLOCK = """import os
import io
import tarfile
import numpy as np
from astropy.io import fits

WASP12_CACHE = "/opt/fits_samples/WASP-12b_calibrated.tar.gz"
RAW_DIR = "/home/ga/AstroImages/raw_sequence"

if not os.path.exists(WASP12_CACHE):
    print(f"ERROR: Cached data not found at {WASP12_CACHE}")
    exit(1)

print("Extracting and drifting the first 20 frames from archive...")
count = 0
with tarfile.open(WASP12_CACHE, 'r|gz') as tar:
    for member in tar:
        if count >= 20:
            break
        if not member.isfile() or not member.name.endswith('.fits'):
            continue

        src = tar.extractfile(member)
        if src is None:
            continue

        # CUA-Lite fixup: streaming mode avoids indexing and random-seeking
        # through a 4.4GB gzip tarball; Astropy needs a seekable object.
        with fits.open(io.BytesIO(src.read()), memmap=False) as hdul:
            data = np.asarray(hdul[0].data)
            hdr = hdul[0].header

            # Inject artificial drift (dx = 3px/frame, dy = 2px/frame)
            dy, dx = count * 2, count * 3

            shifted = np.zeros_like(data)
            if dy > 0 and dx > 0:
                shifted[dy:, dx:] = data[:-dy, :-dx]
            else:
                shifted = data.copy()

            # Write to raw_sequence directory
            out_name = f"drifting_frame_{count:02d}.fits"
            out_path = os.path.join(RAW_DIR, out_name)
            fits.writeto(out_path, shifted, hdr, overwrite=True)
            count += 1

if count != 20:
    print(f"ERROR: Expected 20 FITS frames, found {count}")
    exit(1)

print("Drifting sequence prepared successfully.")
"""


def _apply_material_hook_fixups(text: str) -> str:
    """Apply narrow CUA-Lite migration fixups to pinned hook bodies.

    These are not upstream task rewrites. They normalize places where the pinned
    materials assume the upstream compose/image layout but CUA-Lite deliberately
    runs a native service or regenerated fixture path.
    """
    fixed = text.replace(
        "\ncd /home/ga/openemr\n",
        "\n"
        "# CUA-Lite native OpenEMR lives under Apache's document root.\n"
        "if [ -d /var/www/html/openemr ]; then\n"
        "    cd /var/www/html/openemr\n"
        "else\n"
        "    cd /home/ga/openemr\n"
        "fi\n",
    )

    fixed = fixed.replace(
        'if [ -f "$TARGET_DIR/ct_volume.nii.gz" ]; then\n'
        '    CT_FILE="$TARGET_DIR/ct_volume.nii.gz"\n'
        'elif [ -d "$TARGET_DIR/PATIENT_DICOM" ]',
        'if [ -f "$TARGET_DIR/ct_volume.nii.gz" ]; then\n'
        '    CT_FILE="$TARGET_DIR/ct_volume.nii.gz"\n'
        'elif [ -f "$TARGET_DIR/patient_${PATIENT_NUM}_ct.nii.gz" ]; then\n'
        '    CT_FILE="$TARGET_DIR/patient_${PATIENT_NUM}_ct.nii.gz"\n'
        'elif [ -d "$TARGET_DIR/PATIENT_DICOM" ]',
    )

    fixed = fixed.replace(
        'CT_FILE="$IRCADB_DIR/ircadb_patient${PATIENT_NUM}.nii.gz"\n'
        'SEG_FILE="$IRCADB_DIR/ircadb_patient${PATIENT_NUM}_seg.nrrd"\n',
        'CT_FILE="$IRCADB_DIR/ircadb_patient${PATIENT_NUM}.nii.gz"\n'
        'SEG_FILE="$IRCADB_DIR/ircadb_patient${PATIENT_NUM}_seg.nrrd"\n'
        'SYNTHETIC_CT_FILE="$PATIENT_DIR/patient_${PATIENT_NUM}_ct.nii.gz"\n'
        'if [ ! -f "$CT_FILE" ] && [ -f "$SYNTHETIC_CT_FILE" ]; then\n'
        '    ln -sf "$SYNTHETIC_CT_FILE" "$CT_FILE"\n'
        'fi\n',
    )

    fixed = _SLICER_WAIT_RE.sub(r"\1" + _SLICER_MODAL_DISMISSAL, fixed)
    fixed = _SLICER_WMCTRL_RE.sub(r"\1" + _SLICER_MODAL_DISMISSAL, fixed)
    fixed = _ASTRO_ALIGN_SLOW_EXTRACT_RE.sub(
        lambda _: _ASTRO_ALIGN_STREAMING_EXTRACT_BLOCK,
        fixed,
    )
    return fixed


def _guard_shell_idioms(text: str, helpers: tuple[str, ...]) -> str:
    # `awk 'NR==1'`, NOT `head -1`: head exits after the first line and the writer
    # then takes SIGPIPE, which under the hooks' `set -euo pipefail` turns the whole
    # substitution into rc=141 and aborts the script — measured ~19/20 (it is a race
    # with head's exit), and strictly worse than the malformed JSON it was meant to
    # fix (no result file at all). awk drains its input, so the pipeline exits 0 in
    # all three cases (match, no-match, missing file).
    guarded = _COUNT_FALLBACK_RE.sub(
        lambda m: "$( { " + m.group(1).strip() + "; } | awk 'NR==1' )", text
    )
    guarded = _guard_helper_calls(guarded, helpers)
    return _EMPTY_SLOT_RE.sub(r"\1${\2:-null}\3", guarded)


def _guard_hook_body(
    text: str,
    helpers: tuple[str, ...],
    *,
    material_fixups: bool = True,
) -> str:
    """Normalize a task hook body before upload.

    This includes the shared shell idioms that corrupt or abort a hook body
    (`_COUNT_FALLBACK_RE`, `_guard_helper_calls`, `_EMPTY_SLOT_RE`). Narrow
    CUA-Lite material fixups are setup-only: export/post_task hooks are reward
    evidence collection, and setup UI/material rewrites must not mutate them.
    Env-level `post_start` bodies use `_guard_shell_idioms` directly.

    `helpers` is REQUIRED, not defaulted: this guard was already shipped once as
    dead code (it rewrote the hook PATH instead of the hook CONTENT), and an
    optional argument is exactly how it would silently go dead again."""
    guarded = _COUNT_FALLBACK_RE.sub(
        lambda m: "$( { " + m.group(1).strip() + "; } | awk 'NR==1' )", text
    )
    if material_fixups:
        guarded = _apply_material_hook_fixups(guarded)
    guarded = _guard_helper_calls(guarded, helpers)
    return _EMPTY_SLOT_RE.sub(r"\1${\2:-null}\3", guarded)


def _is_export_hook_dependency(relative: str) -> bool:
    """Return true for task scripts that collect reward evidence after the episode."""
    name = PurePosixPath(relative).name.lower()
    return name == "export_result.sh" or name.startswith("post_task")


def _apply_hook_execution_fixups(text: str, task_name: str, relative: str) -> str:
    """Apply narrow interpreter/runtime fixups that preserve hook semantics.

    These are distinct from setup material fixups: export hooks still do not get
    UI/material rewrites, but they may need a corrected interpreter when the
    pinned hook already says that interpreter is valid and the alternative would
    be an environment false negative.
    """
    if task_name == "create_oring_groove" and relative == "export_result.sh":
        text = text.replace(
            "if python3 /tmp/analyze_geometry.py > /tmp/geo_out.json 2>/dev/null; then\n",
            "if freecadcmd /tmp/analyze_geometry.py > /tmp/geo_out.json 2>/dev/null; then\n",
        )
    return text


def _guard_post_start_body(text: str, helpers: tuple[str, ...]) -> str:
    """Normalize an env-level `post_start` body: shared shell-idiom guards plus
    the `xdotool search` abort (see `_SEARCH_NOMATCH_RE` — post_start is the only
    place the latter can fire, and the only place its two real files are reachable).

    Task-material fixups are intentionally excluded here; post_start bodies are
    image-level scripts, not per-task OpenEMR/Slicer hooks.
    """
    return _SEARCH_NOMATCH_RE.sub(
        lambda m: "$( " + m.group(1).strip() + " || true )",
        _guard_shell_idioms(text, helpers),
    )


async def _run_hook(
    computer: Any,
    text: str,
    *,
    name: str,
    timeout: float,
    helpers: tuple[str, ...],
    material_fixups: bool = True,
) -> Any:
    """Write a hook to a container file and run ``bash <file>`` (see module doc:
    keeps a hook's ``pkill -f <app>`` from matching the setup shell). Exports
    ``DISPLAY=:1`` up front for hooks that launch the GUI without an explicit
    ``DISPLAY=`` prefix. Use the desktop session PATH so a hook's bare
    ``python3`` resolves to system Python, where the upstream materials install
    their document libraries, rather than exec-stdio's env-only venv."""
    shim_dir = "/tmp/cuaworld_hookshim"
    await computer.interface.write_bytes(
        f"{shim_dir}/sitecustomize.py", _HOOK_SITECUSTOMIZE.encode()
    )
    body = (
        f"export DISPLAY=:1\nexport PATH={_SESSION_PATH}\n"
        f"export PYTHONPATH={shim_dir}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        f"{_guard_hook_body(text, helpers, material_fixups=material_fixups)}"
    )
    path = f"/tmp/cuaworld_{name}.sh"
    await computer.interface.write_bytes(path, body.encode())
    # cuaworld hooks self-drop via `su - ga` (passwordless only from root), so
    # the whole session runs as root (`EXEC_USER="root"`, #156). And run the whole hook
    # in a new session (`setsid`) so any GUI it backgrounds (`su - ga -c "… &"`)
    # is detached from this shell's process group and survives the hook returning
    # — upstream's task setups hand-inserted `setsid` into each `su - ga -c "…"`
    # for exactly this; doing it once here keeps the mirrored `tasks/` byte-
    # identical to upstream instead of patching 230 task-dir hook files (on-disk
    # basis: 248 `setsid` lines under the 3216 task dirs).
    #   -w: wait for the session leader and propagate ITS exit code — a bare
    #   `setsid` returns 0 regardless, which would hide a hook's `exit 1` (kstars/
    #   openemr rely on rc). -w still returns as soon as the FOREGROUND body ends;
    #   a backgrounded `&` GUI does not hold it (verified: 0.003s with `sleep 10 &`).
    return await computer.interface.run_command(
        f"setsid -w bash {path}", timeout=timeout
    )


def _task_hook(task_spec: dict | None, name: str) -> str | None:
    if task_spec is None:
        return None
    hooks = task_spec.get("hooks") or {}
    hook = hooks.get(name) if isinstance(hooks, dict) else None
    return hook if isinstance(hook, str) and hook.strip() else None


async def _run_task_hook(
    computer: Any,
    task_dir: Path,
    hook: str,
    *,
    name: str,
    timeout: float,
    material_fixups: bool = True,
) -> Any:
    """Run the hook declared by task.json using its host-side task sources."""
    source_root = f"/workspace/tasks/{task_dir.name}"
    remote_root = f"/tmp/cuaworld_task_sources/{task_dir.name}"
    helpers = _hook_helpers(task_dir)
    uploaded = getattr(computer, "_cuaworld_uploaded_task_dirs", None)
    if not isinstance(uploaded, set):
        uploaded = set()
        setattr(computer, "_cuaworld_uploaded_task_dirs", uploaded)
    if str(task_dir) not in uploaded:
        executable_paths: list[str] = []
        for source in sorted(task_dir.rglob("*")):
            metadata = source.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    f"task hook dependency is not a regular file: {source}"
                )
            relative = source.relative_to(task_dir).as_posix()
            destination = f"{remote_root}/{relative}"
            data = source.read_bytes()
            # Normalize the shell idioms that abort or corrupt a hook. This has to
            # happen HERE, on the file CONTENT: task.json declares hooks as bare
            # PATHS (3197 of 3197 parseable task dirs), so rewriting the hook string
            # in `_run_hook` rewrote a path and touched no script — the guard was
            # dead code.
            # The SAME `source_root` → `remote_root` rewrite the hook path gets below
            # also has to reach the content: `/workspace/tasks/` is hardcoded inside
            # some hooks and DOES NOT EXIST in any image (verified: `ls -d
            # /workspace/tasks` fails in lite.cuaworld.{diagrams_net,imagej,webots};
            # /workspace holds only assets/config/data/env.json/scripts) — task
            # materials live at `remote_root`. ON-DISK basis: 11 `.sh` files in 11
            # task dirs (diagrams_net 6, webots 3, imagej 2) → 11 registered → 10
            # registered ∧ non-excluded (diagrams_net/microservices_outage_rca is
            # already excluded). None sets `-e`, so they do not abort; they fail
            # SILENTLY WORSE — webots `exit 1`s on "Task world file not found",
            # diagrams_net's `cp` never lands the fixture on the desktop, and
            # imagej's `python3 …/parse_results.py` is missing so export_result.sh
            # writes its all-zeros FALLBACK json and the verifier scores a real run 0.
            # Replacing `source_root` (task-SCOPED, not the bare `/workspace/tasks`
            # prefix) cannot corrupt a hook that legitimately mentions the string:
            # every occurrence in the corpus names its own task dir, and the two
            # non-`.sh` files that mention it (libreoffice_calc `validated_pi.json`,
            # an openemr plan transcript) are inert and never rewritten.
            if source.suffix == ".sh":
                try:
                    text = data.decode()
                except UnicodeDecodeError:
                    pass
                else:
                    text = _apply_hook_execution_fixups(text, task_dir.name, relative)
                    apply_material_fixups = (
                        material_fixups
                        and not _is_export_hook_dependency(relative)
                    )
                    guarded = _guard_hook_body(
                        text,
                        helpers,
                        material_fixups=apply_material_fixups,
                    )
                    data = guarded.replace(source_root, remote_root).encode()
            await computer.interface.write_bytes(destination, data)
            # Hugging Face snapshots do not preserve executable bits. Restore
            # scripts from their content as well as from a local checkout mode.
            if metadata.st_mode & 0o111 or data.startswith(b"#!"):
                executable_paths.append(destination)
        for offset in range(0, len(executable_paths), 50):
            paths = executable_paths[offset:offset + 50]
            result = await computer.interface.run_command(
                "chmod 755 " + " ".join(map(shlex.quote, paths)),
                timeout=min(timeout, 30.0),
            )
            if getattr(result, "returncode", 0):
                raise RuntimeError(
                    f"could not preserve executable task hook dependencies: "
                    f"{getattr(result, 'stderr', '')}"
                )
        uploaded.add(str(task_dir))
    rewritten = hook.replace(source_root, remote_root)
    return await _run_hook(
        computer,
        rewritten,
        name=name,
        timeout=timeout,
        helpers=helpers,
        material_fixups=material_fixups,
    )


def _post_task_settle_seconds(task_spec: dict | None) -> float:
    if _task_hook(task_spec, "post_task") is None:
        return 0.0
    override = os.environ.get("GYM_ANYTHING_POST_TASK_SETTLE_SEC")
    if override is not None:
        try:
            return max(0.0, float(override))
        except ValueError:
            return 15.0
    if task_spec is not None:
        for source in (task_spec.get("extras"), task_spec.get("metadata")):
            value = (
                source.get("post_task_settle_sec")
                if isinstance(source, dict)
                else None
            )
            if value is not None:
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    return 15.0
    return 15.0


_ASTROIMAGEJ_UPDATER_CLEANUP_COMMAND = r"""
export DISPLAY=:1
quiet=0
min_polls="${CUA_LITE_AIJ_ADAPTER_MIN_POLLS:-8}"
max_polls="${CUA_LITE_AIJ_ADAPTER_MAX_POLLS:-40}"
quiet_polls="${CUA_LITE_AIJ_ADAPTER_QUIET_POLLS:-4}"
for poll in $(seq 1 "$max_polls"); do
  ids=$(DISPLAY=:1 sudo -u ga xdotool search --name "AstroImageJ Updater" 2>/dev/null || true)
  if [ -n "$ids" ]; then
    quiet=0
    for wid in $ids; do
      DISPLAY=:1 sudo -u ga xdotool windowactivate --sync "$wid" 2>/dev/null || true
      DISPLAY=:1 sudo -u ga xdotool key Escape 2>/dev/null || true
      DISPLAY=:1 sudo -u ga xdotool windowclose "$wid" 2>/dev/null || true
    done
  elif [ "$poll" -ge "$min_polls" ]; then
    quiet=$((quiet + 1))
    [ "$quiet" -ge "$quiet_polls" ] && exit 0
  else
    quiet=0
  fi
  sleep 0.5
done
"""


def _is_astroimagej_task(task_spec: dict | None) -> bool:
    if not isinstance(task_spec, dict):
        return False
    env_id = task_spec.get("env_id")
    return isinstance(env_id, str) and env_id.startswith("astroimagej_env")


async def _cleanup_astroimagej_updater_after_setup(
    computer: Any,
    task_spec: dict | None,
    task_name: str,
    *,
    timeout: float,
) -> None:
    """Close late AstroImageJ updater dialogs before the first observation.

    Some pinned task hooks launch AstroImageJ directly instead of going through
    the patched task utility or `/usr/local/bin/aij` wrapper. Run this after the
    task's pre_task returns so late updater windows cannot contaminate turn_00.
    """
    if not _is_astroimagej_task(task_spec):
        return
    try:
        await computer.interface.run_command(
            _ASTROIMAGEJ_UPDATER_CLEANUP_COMMAND,
            timeout=min(max(timeout, 1.0), 30.0),
        )
    except Exception as exc:  # noqa: BLE001 - non-fatal visual cleanup
        logger.warning(
            "lite.cuaworld AstroImageJ updater cleanup failed for %s: %s: %s",
            task_name,
            type(exc).__name__,
            exc,
        )


async def run_cuaworld_setup(
    computer: Any,
    task_dir: Path,
    *,
    task_spec: dict | None = None,
    timeout: float = 120.0,
) -> None:
    """Run the task.json ``pre_task`` hook inside the container.

    A failed setup is not an agent failure: the first screenshot is no longer the
    declared task state, so continuing would produce a false-negative training
    sample. Environment-level ``post_start`` keeps upstream's warning behavior;
    task-level ``pre_task`` must fail the reset loudly.
    """
    try:
        hook = _task_hook(task_spec, "pre_task")
        if hook is None:
            legacy = _read_optional_regular_text(task_dir / "setup_task.sh")
            if task_spec is not None or legacy is None:
                return
            res = await _run_hook(
                computer,
                legacy,
                name="setup",
                timeout=timeout,
                helpers=_hook_helpers(task_dir),
            )
        else:
            res = await _run_task_hook(
                computer,
                task_dir,
                hook,
                name="setup",
                timeout=timeout,
            )
    except CuaWorldVerifierError:
        raise
    except Exception as exc:  # noqa: BLE001 - transport/read/setup failure
        raise CuaWorldVerifierError(
            "pre_task hook could not run: "
            f"{type(exc).__name__}: {exc}",
            phase="setup",
            kind="timeout" if isinstance(exc, TimeoutError) else "spawn",
            task=task_dir.name,
        ) from exc
    rc = getattr(res, "returncode", 0)
    if rc:
        # These hooks trace to stdout and route their probes' stderr to
        # /dev/null, so the reason a hook `exit 1`s is almost always on
        # stdout and stderr is empty. Report both (stderr first) — reading
        # stderr alone renders the failure as a bare rc with no message.
        detail = "; ".join(
            f"{name}: {text.strip()[-500:]}"
            for name, text in (
                ("stderr", getattr(res, "stderr", "") or ""),
                ("stdout", getattr(res, "stdout", "") or ""),
            )
            if text.strip()
        ) or "<no output captured>"
        raise CuaWorldVerifierError(
            f"pre_task hook failed with rc={rc}: {detail}",
            phase="setup",
            kind="command_failed",
            task=task_dir.name,
        )
    # run_command runs as root for cuaworld (`EXEC_USER="root"`), so
    # any file a hook created directly (before its own `su - ga` drop) is
    # root-owned. Restore user ownership of the agent home so the ownership
    # invariant (find /home/ga -not -user ga == empty) holds during the episode.
    try:
        await computer.interface.run_command(
            "chown -R ga:ga /home/ga",
            timeout=min(timeout, 60.0),
        )
    except Exception as exc:  # noqa: BLE001 - non-fatal ownership restore
        logger.warning(
            "lite.cuaworld chown after setup failed for %s: %s: %s",
            task_dir.name,
            type(exc).__name__,
            exc,
        )
    await _cleanup_astroimagej_updater_after_setup(
        computer, task_spec, task_dir.name, timeout=timeout
    )


async def run_cuaworld_verify(
    computer: Any,
    task_dir: Path,
    verifier_target: str,
    *,
    task_metadata: dict | None = None,
    task_spec: dict | None = None,
    env_id: str | None = None,
    lite_env_id: str | None = None,
    episode_dir: Path | None = None,
    timeout: float | None = None,
    preparation_timeout: float | None = None,
) -> tuple[float, dict]:
    """Run the task's declared ``post_task`` hook (if any), then ``verifier.py``.

    Returns ``(reward, info)`` using gym-anything's task ``reward_type``.
    """
    # Run inside a per-episode temp dir + a sys.path/sys.modules snapshot, both
    # torn down in `finally`. The verifier itself runs in a child process, so
    # generic imports (`utils`, `helpers`, ...) cannot leak across tasks.
    owns_episode_dir = episode_dir is None
    workdir = episode_dir or create_episode_dir()
    workdir.mkdir(parents=True, exist_ok=True)
    verifier_timeout = _resolved_verifier_timeout(timeout)
    resolved_preparation_timeout = (
        _DEFAULT_PREPARATION_TIMEOUT
        if preparation_timeout is None and timeout is None
        else verifier_timeout
        if preparation_timeout is None
        else max(0.0, float(preparation_timeout))
    )
    deadline = time.monotonic() + resolved_preparation_timeout
    process = None
    request_reader = request_writer = None
    response_reader = response_writer = None
    result_reader = result_writer = None
    try:
        # 1) post_task export hook (some envs have none — their verifier reads
        #    container files directly via env_info).
        post_task_hook = _task_hook(task_spec, "post_task")
        if task_spec is None and post_task_hook is None:
            try:
                post_task_hook = await _call_in_daemon_thread_before_deadline(
                    _read_optional_regular_text,
                    (task_dir / "export_result.sh",),
                    deadline,
                )
            except Exception as exc:  # noqa: BLE001 - transport/read failure
                raise CuaWorldVerifierError(
                    "legacy post_task hook could not be read: "
                    f"{type(exc).__name__}: {exc}",
                    phase="post_task",
                    kind="timeout" if isinstance(exc, TimeoutError) else "spawn",
                    task=task_dir.name,
                ) from exc
        if post_task_hook is not None:
            try:
                remaining = max(deadline - time.monotonic(), 0)
                if task_spec is None:
                    export_result = await asyncio.wait_for(
                        _run_hook(
                            computer,
                            post_task_hook,
                            name="export",
                            timeout=remaining,
                            helpers=_hook_helpers(task_dir),
                            material_fixups=False,
                        ),
                        timeout=remaining,
                    )
                else:
                    export_result = await asyncio.wait_for(
                        _run_task_hook(
                            computer,
                            task_dir,
                            post_task_hook,
                            name="export",
                            timeout=remaining,
                            material_fixups=False,
                        ),
                        timeout=remaining,
                    )
            except Exception as exc:  # noqa: BLE001 - transport/read failure
                raise CuaWorldVerifierError(
                    "post_task hook could not run: "
                    f"{type(exc).__name__}: {exc}",
                    phase="post_task",
                    kind="timeout" if isinstance(exc, TimeoutError) else "spawn",
                    task=task_dir.name,
                ) from exc
            else:
                # Log on rc != 0 OR on any stderr. These hooks routinely end with a
                # trailing `echo "=== ... complete ==="`, so a python heredoc that
                # died mid-script still exits 0 — the traceback was being discarded
                # and the only symptom reaching the artifacts was the verifier's
                # "Could not retrieve result JSON", with no way to tell a crashed
                # exporter from a task the agent genuinely failed. With stderr
                # captured, "result JSON missing" triage is a log read.
                rc = getattr(export_result, "returncode", 0)
                stderr = (getattr(export_result, "stderr", "") or "").strip()
                if rc:
                    detail = (
                        stderr or (getattr(export_result, "stdout", "") or "")
                    ).strip() or "<no output captured>"
                    raise CuaWorldVerifierError(
                        f"post_task hook failed with rc={rc}: {detail[-500:]}",
                        phase="post_task",
                        kind="command_failed",
                        task=task_dir.name,
                    )
                if stderr:
                    logger.warning(
                        "lite.cuaworld post_task hook stderr for %s: %s",
                        task_dir.name,
                        stderr[-500:],
                    )
            settle_seconds = _post_task_settle_seconds(task_spec)
            if settle_seconds > 0:
                remaining = deadline - time.monotonic()
                if settle_seconds > remaining:
                    raise CuaWorldVerifierError(
                        f"post_task settle ({settle_seconds:.0f}s) exceeds the "
                        f"remaining preparation budget ({remaining:.0f}s)",
                        phase="post_task",
                        kind="timeout",
                        task=task_dir.name,
                    )
                await asyncio.sleep(settle_seconds)

        # 2) Capture post-task and final screenshots as separate observations,
        # matching gym-anything's complete/finalize sequence.
        await _capture_verification_screenshot(
            computer,
            workdir / "post_verification.png",
            deadline=deadline,
            task=task_dir.name,
            label="post_verification",
        )
        await _capture_verification_screenshot(
            computer,
            workdir / "final.png",
            deadline=deadline,
            task=task_dir.name,
            label="final",
        )
        traj = _load_trajectory(workdir)
        task_info = {
            "task_id": (
                task_spec.get("id", task_dir.name)
                if isinstance(task_spec, dict)
                else task_dir.name
            ),
            "metadata": task_metadata or {},
            "task_spec": task_spec or {},
            "_env_id": env_id,
            "_lite_env_id": lite_env_id,
            "_container": (
                getattr(computer, "container_name", None)
                or getattr(computer, "name", None)
            ),
        }

        # 4) Run the verifier in a killable process. The parent services its
        # container RPC requests under a separate wall-clock deadline. Export,
        # settle, and screenshot work above must not consume the VLM/verifier
        # budget.
        deadline = time.monotonic() + verifier_timeout
        try:
            resolved_verifier_path, verifier_func = _resolve_verifier_target(
                task_dir, verifier_target
            )
        except Exception as exc:  # noqa: BLE001 - bad task declaration
            raise CuaWorldVerifierError(
                f"verifier target is invalid: {type(exc).__name__}: {exc}",
                phase="verify",
                kind="spawn",
                task=task_dir.name,
            ) from exc
        context = multiprocessing.get_context("spawn")
        request_reader, request_writer_connection = context.Pipe(duplex=False)
        request_writer = _MessagePipe(request_writer_connection)
        response_reader_connection, response_writer = context.Pipe(duplex=False)
        response_reader = _MessagePipe(response_reader_connection)
        result_reader, result_writer_connection = context.Pipe(duplex=False)
        result_writer = _MessagePipe(result_writer_connection)
        gate_reader, gate_writer = context.Pipe(duplex=False)
        candidate_process = context.Process(
            target=_gated_call,
            args=(
                gate_reader,
                _verifier_supervisor,
                (
                    _verifier_worker,
                    str(resolved_verifier_path),
                    verifier_func,
                    traj,
                    task_info,
                    str(workdir),
                    request_writer,
                    response_reader,
                    result_writer,
                ),
            ),
        )
        start_connections = (
            request_reader,
            request_writer,
            response_reader,
            response_writer,
            result_reader,
            result_writer,
            gate_reader,
            gate_writer,
        )
        try:
            started = await _start_before_deadline(
                candidate_process,
                deadline,
                start_connections,
                tree=True,
                gate_reader=gate_reader,
                gate_writer=gate_writer,
            )
        except asyncio.CancelledError:
            request_reader = request_writer = None
            response_reader = response_writer = None
            result_reader = result_writer = None
            raise
        except Exception as exc:  # noqa: BLE001 - process startup failure
            request_reader = request_writer = None
            response_reader = response_writer = None
            result_reader = result_writer = None
            raise CuaWorldVerifierError(
                f"verifier process could not start: {type(exc).__name__}: {exc}",
                phase="verify",
                kind="spawn",
                task=task_dir.name,
            ) from exc
        if not started:
            request_reader = request_writer = None
            response_reader = response_writer = None
            result_reader = result_writer = None
            raise CuaWorldVerifierError(
                f"verifier process did not come up within {verifier_timeout}s",
                phase="verify",
                kind="timeout",
                task=task_dir.name,
            )
        process = candidate_process
        if time.monotonic() >= deadline:
            raise CuaWorldVerifierError(
                f"verifier budget ({verifier_timeout}s) was already spent before the "
                "check could run",
                phase="verify",
                kind="timeout",
                task=task_dir.name,
            )
        request_writer.close()
        request_writer = None
        response_reader.close()
        response_reader = None
        result_writer.close()
        result_writer = None
        payload = None
        request_open = True
        while time.monotonic() < deadline:
            while request_open and _connection_ready(request_reader):
                try:
                    request_id, operation, args = await _recv_before_deadline(
                        request_reader, deadline
                    )
                except (EOFError, OSError):
                    request_open = False
                    break
                except TimeoutError:
                    deadline = 0
                    break
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    if operation == "copy_from_env":
                        container_src, host_dst = args
                        data = await asyncio.wait_for(
                            computer.interface.read_bytes(container_src),
                            timeout=remaining,
                        )
                        destination = Path(host_dst)
                        await _write_bytes_in_process_before_deadline(
                            destination, data, deadline
                        )
                        value = None
                    elif operation == "copy_to_env":
                        host_src, container_dst = args
                        data = await _call_in_process_before_deadline(
                            _read_regular_bytes,
                            (Path(host_src),),
                            deadline,
                        )
                        await asyncio.wait_for(
                            computer.interface.write_bytes(container_dst, data),
                            timeout=remaining,
                        )
                        value = None
                    elif operation == "run_command":
                        # Verifier exec_capture: runs as root
                        # (`EXEC_USER="root"`) so upstream verifier commands that
                        # `su - ga` stay passwordless.
                        command_result = await asyncio.wait_for(
                            computer.interface.run_command(
                                *args
                            ),
                            timeout=remaining,
                        )
                        value = (
                            (getattr(command_result, "stdout", "") or "")
                            + (getattr(command_result, "stderr", "") or "")
                        )
                    else:
                        raise RuntimeError(f"unknown verifier RPC operation: {operation}")
                    response = (request_id, True, value)
                except TimeoutError:
                    deadline = 0
                    break
                except Exception as exc:  # noqa: BLE001 - bridge error
                    response = (
                        request_id,
                        False,
                        f"{type(exc).__name__}: {exc}",
                    )
                try:
                    await _send_before_deadline(
                        response_writer, response, deadline
                    )
                except (BrokenPipeError, EOFError, OSError):
                    request_open = False
                    break
                except TimeoutError:
                    deadline = 0
                    break
            if _connection_ready(result_reader):
                try:
                    payload = await _recv_before_deadline(
                        result_reader, deadline
                    )
                except (EOFError, OSError):
                    payload = {"error": "verifier result pipe closed unexpectedly"}
                except TimeoutError:
                    payload = {
                        "error": (
                            "TimeoutError: verifier exceeded "
                            f"{verifier_timeout}s"
                        )
                    }
                break
            if not process.is_alive():
                if _connection_ready(result_reader):
                    try:
                        payload = await _recv_before_deadline(
                            result_reader, deadline
                        )
                    except (EOFError, OSError):
                        payload = {
                            "error": "verifier result pipe closed unexpectedly"
                        }
                    except TimeoutError:
                        payload = {
                            "error": (
                                "TimeoutError: verifier exceeded "
                                f"{verifier_timeout}s"
                            )
                        }
                else:
                    payload = {
                        "error": f"verifier process exited with code {process.exitcode}"
                    }
                break
            await asyncio.sleep(0.02)
        if payload is None:
            raise CuaWorldVerifierError(
                f"verifier exceeded its {verifier_timeout}s budget",
                phase="verify",
                kind="timeout",
                task=task_dir.name,
            )
        if "error" in payload:
            # The verifier RAISED — a missing dependency, bad task data, or its own
            # bug. It never judged the agent, so there is no score to report; scoring
            # this 0 is what made `pycharm/fix_excel_sales_report`'s unguarded
            # `import openpyxl` look like an honest task failure.
            raise CuaWorldVerifierError(
                f"verifier raised: {payload['error']}",
                phase="verify",
                kind="verifier_raised",
                task=task_dir.name,
            )
        res = payload.get("result")
        if not isinstance(res, dict):
            raise CuaWorldVerifierError(
                f"verifier returned {type(res).__name__}, not a result dict",
                phase="verify",
                kind="bad_result",
                task=task_dir.name,
            )
        passed_value = res.get("passed", False)
        if not isinstance(passed_value, bool):
            raise CuaWorldVerifierError(
                f"verifier returned non-bool passed={passed_value!r}",
                phase="verify",
                kind="bad_result",
                task=task_dir.name,
            )
        passed = passed_value
        fallback_score = 100.0 if passed else 0.0
        score_value = res.get("score", fallback_score)
        try:
            raw = float(score_value)
        except (TypeError, ValueError) as exc:
            raise CuaWorldVerifierError(
                f"verifier returned nonnumeric score={score_value!r}",
                phase="verify",
                kind="bad_result",
                task=task_dir.name,
            ) from exc
        if not math.isfinite(raw):
            raise CuaWorldVerifierError(
                f"verifier scored {raw!r}, which is not a finite number",
                phase="verify",
                kind="bad_result",
                task=task_dir.name,
            )
        reward, reward_type = _final_reward(
            task_spec,
            raw_score=raw,
            passed=passed,
        )
        return reward, {
            "passed": passed,
            "raw_score": raw,
            "reward_type": reward_type,
            "feedback": res.get("feedback", ""),
        }
    finally:
        if process is not None and process.pid is not None:
            _cleanup_process_in_thread(process, tree=True)
        for connection in (
            request_reader,
            request_writer,
            response_reader,
            response_writer,
            result_reader,
            result_writer,
        ):
            if connection is not None:
                connection.close()
        if owns_episode_dir:
            remove_episode_dir(workdir)
