"""osworld — CUA-Lite gym wrapper for the OFFICIAL OSWorld benchmark (v1).

369 real-world desktop tasks run on a locally-managed VM-in-Docker container
(`cua-lite/osworld`; QEMU/KVM booting Ubuntu.qcow2). DEDICATED, one container per trajectory.

EVAL-IN-CONTAINER (androidworld pattern): OSWorld's `desktop_env` + `osworld_evaluation_examples`
are baked into the image and driven by an in-container HTTP server (`docker/server.py`); the
cua-lite HOST imports ZERO `desktop_env` and just talks to the container over one JSON-RPC
port. So the single host uv carries no OSWorld dep at all (osworld, osworld_2, and lite.osworld
each keep their eval stack inside their own image — no shared-dist coupling). The host still
owns the container (naming + reaping) and translates LiteDesktopActionSpace → pyautogui
host-side (a pure function, no `desktop_env`).

Prerequisites:
  - uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh   (docker build + provision qcow2)
  - KVM (/dev/kvm), /dev/net/tun, the built image, and the qcow2 (all checked by ensure).

Usage:
    uv run python -c "import lite.gym as gym; print(len(gym.registry.task_ids('osworld')))"
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from lite.core.messages.final import (
    MAX_STEPS_STOP_REASON,
    STOP_REASON_INFO_KEY,
)
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.calls import RuntimeEnvAction
from lite.core.tools.schemas import BaseTools
from lite.gym.base import LiteBaseEnv
from lite.gym.container import spawn_background_destroy
from lite.gym.envs.osworld.actions import OsworldTools, to_pyautogui
from lite.gym.envs.osworld.container import OSWorldContainer, OSWorldContainerFactory
from lite.gym.errors import TrueInfraFailure, EnvDepsMissingError
from lite.gym.registry import register, registry
from lite.gym.services import EnvServerResource
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvObservation,
    LiteEnvStepResult,
    LiteExecutedAction,
)
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.rpc import json_rpc
from lite.gym.utils.config.identity import EnvIdentity
from lite.gym.utils.feedback.errors import (
    append_feedback,
    MODEL_ACTION_ERROR_TYPES,
    ToolErrorFeedback,
    error_only_feedback,
    record_batch_abort,
    record_model_action_error,
    record_tool_execution_error,
    unsupported_action_message,
)
from lite.gym.utils.feedback.ingress import (
    invalid_action_message,
    prepare_env_tool_calls,
    standalone_tool_call_feedback_with_reason,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)
from lite.gym.utils.feedback.surface import (
    resolve_extra_tools,
    resolve_valid_actions,
)
from lite.utils.image import png_from_b64_async

logger = logging.getLogger(__name__)

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)
_RELEASE = json.loads((Path(ENV_DIR) / "data" / "release.json").read_text(encoding="utf-8"))

# ============================================================================
# Config defaults — read once from configs/default.yaml. Named constants.
# ============================================================================
_MAX_STEPS = CFG.env_kwargs["max_steps"]
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]
_IMAGE = CFG.server_kwargs["image"]
_RAM = CFG.server_kwargs["ram_size"]
_CPU = CFG.server_kwargs["cpu_cores"]
_DISK = CFG.server_kwargs["disk_size"]
_BOOT_TO = float(CFG.server_kwargs["boot_timeout"])
_SCREEN_W = CFG.server_kwargs["screen_width"]
_SCREEN_H = CFG.server_kwargs["screen_height"]
_MAX_WORKERS = CFG.server_kwargs["max_workers"]

_CLIENT_PASSWORD = "password"   # v1 image sudo password (passed to the in-container server)

_QCOW2_CFG = CFG.server_kwargs["qcow2_path"]
_QCOW2 = _QCOW2_CFG if os.path.isabs(_QCOW2_CFG) else str((Path(ENV_DIR) / _QCOW2_CFG).resolve())
_QCOW2_SIZE = int(_RELEASE["qcow2"]["size"])
_DATA_DIR = Path(ENV_DIR) / "data"   # vendored task index (host registration needs no desktop_env)

_README = "lite/gym/envs/osworld/README.md"
_INSTALL = "uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh"

_RESET_RPC_TIMEOUT = 300.0
_STEP_RPC_TIMEOUT = 180.0
_EVAL_RPC_TIMEOUT = 180.0

_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="osworld-exec")
atexit.register(lambda: _EXECUTOR.shutdown(wait=False, cancel_futures=True))


#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = OsworldTools.get_tool_names() | LiteFinishToolSet.get_tool_names()


# ---------------------------------------------------------------------------
# In-container eval-server RPC (JSON; the host never imports desktop_env)
# ---------------------------------------------------------------------------
def _rpc(base_url: str, path: str, body: dict | None = None, timeout: float = _STEP_RPC_TIMEOUT) -> dict:
    """Thin label/timeout binding over the shared JSON RPC — the two
    osworld envs carried byte-identical copies of this client)."""
    return json_rpc(base_url, path, body, timeout=timeout, label="osworld")


# ---------------------------------------------------------------------------
# Prerequisite checks (ensure — terminal, 501). No desktop_env check (it's in the image).
# ---------------------------------------------------------------------------
def _dep_error(what: str) -> EnvDepsMissingError:
    return EnvDepsMissingError(what=what, install=_INSTALL, see=_README)


def _check_kvm() -> None:
    if not os.path.exists("/dev/kvm"):
        raise _dep_error("/dev/kvm not present — osworld needs hardware virtualization (KVM); no software fallback")


def _check_tun() -> None:
    if not os.path.exists("/dev/net/tun"):
        raise _dep_error("/dev/net/tun not present — the VM-in-Docker needs it for guest networking")


def _check_image() -> None:
    # Freshness gate (image_for → ensure_runnable): the image must exist AND its lite.src_hash label
    # must match the current docker/ sources — so editing docker/Dockerfile without a rebuild fails
    # HERE instead of silently running the stale image (docs/envs.md#image-build-and-freshness).
    from lite.gym.utils.backend.docker import require_image_present
    from lite.gym.utils.backend.freshness import image_for
    require_image_present(image_for("osworld", tag=_IMAGE))


def _check_qcow2() -> None:
    if not os.path.exists(_QCOW2):
        raise _dep_error("VM disk cache is missing (Ubuntu.qcow2); provision it via install.sh")
    size = os.path.getsize(_QCOW2)
    if size != _QCOW2_SIZE:
        raise _dep_error(
            f"VM disk cache has size {size}, expected {_QCOW2_SIZE}; "
            "re-run install.sh provision"
        )


def _ensure_services(env_id: str) -> None:
    _check_kvm()
    _check_tun()
    _check_image()
    _check_qcow2()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class OSWorldConfig:
    """Per-task config: domain + task_id (the in-container server loads the JSON + runs
    setup/eval); metadata for the registry."""

    domain: str = ""
    task_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment (thin RPC client — no desktop_env on the host)
# ---------------------------------------------------------------------------
class OSWorldEnv(LiteBaseEnv, EnvServerResource):
    """CUA-Lite wrapper: owns one VM-in-Docker container per trajectory + an in-container eval
    server it drives over JSON-RPC. Non-poolable DEDICATED; cancellation-safe leak handling."""

    EXTRA_TOOLS: ClassVar[type[BaseTools]] = OsworldTools

    def __init__(
        self,
        *,
        config: OSWorldConfig,
        max_steps: int = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        valid_actions: list[str] | None = None,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        **kwargs: Any,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"OSWorldEnv got unexpected env kwargs: {unknown}")
        # Copy metadata, don't alias: `config` is the shared registration-side
        # object (one per task, all instances) — an instance-side metadata
        # write must never reach it.
        self._config = dataclasses.replace(config, metadata=dict(config.metadata))
        self._max_steps = max_steps
        self._post_action_delay = post_action_delay
        # ``valid_actions`` used to land in ``**kwargs`` and be silently
        # ignored. Resolved here so a typo fails at the config boundary and a
        # real subset reaches ``metadata.valid_actions``. ``None`` = full GUI
        # surface, which is this env's registered default. Runtime
        # invalid/unsupported feedback is owned by ``step``.
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="osworld", platform="desktop",
        )
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        self._step_count = 0
        self._container: OSWorldContainer | None = None
        self._pending: OSWorldContainer | None = None
        self._pending_cf_future: Any = None

    @staticmethod
    def _task_metadata(config) -> LiteCUAMetadata:
        """Same-source metadata builder.
        extra_tool_schemas mirrors bind()'s default resolution, so the yaml
        default flows to BOTH sides."""
        return LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=resolve_extra_tools(
                _EXTRA_TOOLS, tools=OsworldTools, env_name="osworld",
            ),
            others={**config.metadata},
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # env_kwargs amendments: valid_actions / extra_tool_schemas resolved
        # at construction.
        return dataclasses.replace(
            self._task_metadata(self._config),
            valid_actions=self._valid_actions,
            extra_tool_schemas=self._extra_tool_schemas,
        )

    @property
    def external_resource_id(self) -> str | None:
        c = self._container or self._pending
        return c.name if c is not None else None

    async def reset(self) -> LiteEnvObservation:
        loop = asyncio.get_event_loop()
        _ensure_services("osworld")

        if self._pending_cf_future is not None:
            self._spawn_background_destroy(self._pending_cf_future, self._pending)
            self._pending_cf_future = None
            self._pending = None
        await self._teardown_existing()

        self._step_count = 0

        identity = getattr(self, "identity", None) or EnvIdentity()
        factory = OSWorldContainerFactory(
            qcow2_path=_QCOW2, image=_IMAGE, ram_size=_RAM, cpu_cores=_CPU, disk_size=_DISK,
            boot_timeout=_BOOT_TO, screen_width=_SCREEN_W, screen_height=_SCREEN_H,
            client_password=_CLIENT_PASSWORD, session_id=identity.session_id,
            token_hash=identity.token_hash, server_port=identity.server_port,
            task_id=self._config.task_id,
        )
        build_cf = _EXECUTOR.submit(factory.build)
        self._pending_cf_future = build_cf
        try:
            container = await asyncio.wrap_future(build_cf, loop=loop)
        except BaseException:
            self._pending_cf_future = None
            self._spawn_background_destroy(build_cf, None)
            raise
        self._pending = container
        self._pending_cf_future = None

        start_cf = _EXECUTOR.submit(container.start)
        self._pending_cf_future = start_cf
        try:
            await asyncio.wrap_future(start_cf, loop=loop)
        except BaseException:
            self._pending_cf_future = None
            self._pending = None
            if start_cf.done():
                await loop.run_in_executor(_EXECUTOR, container.destroy)
            else:
                self._spawn_background_destroy(start_cf, container)
            raise
        self._pending_cf_future = None
        self._container = container
        self._pending = None

        res = await loop.run_in_executor(
            _EXECUTOR, _rpc, container.base_url, "/reset",
            {"domain": self._config.domain, "task_id": self._config.task_id}, _RESET_RPC_TIMEOUT)
        return LiteEnvObservation(image=await png_from_b64_async(res.get("screenshot_b64")),
            text=res.get("instruction", ""),
        )

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        metadata = self.metadata
        actions, ingress_errors = prepare_env_tool_calls(actions, metadata)
        loop = asyncio.get_event_loop()
        terminated = False
        # Model-emitted calls that ENDED the episode. They get no continuation
        # observation: devs/migration/verify.py forbids a tool result for a
        # terminal call. Keyed on the env-internal projected ``action["call_id"]``
        # and NOT on ``result_call_id``, because an INTERNAL finish call has no
        # model-call id and the loop-detect wrapper's injected ``terminate``
        # carries the intercepted NON-finish model call's id as
        # ``_result_call_id`` -- that call must still be answered.
        terminal_call_ids: set[str] = set()
        stop_reason: str | None = None
        executed: list[LiteExecutedAction] = []
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        base = self._container.base_url if self._container is not None else None
        # One frame PER EXECUTED ACTION, in action order (the contract the agent
        # layer consumes). Filled inside the loop below.
        step_screenshots: list[bytes] = []

        async def capture_frame() -> bytes | None:
            """The guest's current frame; ``None`` when the server returns none."""
            sr = await loop.run_in_executor(
                _EXECUTOR, _rpc, base, "/screenshot", {}, _STEP_RPC_TIMEOUT
            )
            return await png_from_b64_async(sr.get("screenshot_b64"))

        for index, (action, result_call_id) in enumerate(actions):
            name = action["name"]
            args = action["arguments"]

            tool_feedback, tool_reason = standalone_tool_call_feedback_with_reason(
                action,
                _KNOWN_STANDALONE_TOOL_NAMES,
                metadata.extra_tool_schemas,
            )
            if tool_feedback is not None:
                if result_call_id:
                    append_feedback(action_errors, result_call_id, tool_feedback)
                executed.append({
                    "call": "noop",
                    "args": {
                        "name": name,
                        "reason": tool_reason,
                    },
                })
                # The shared layer already decided the SURFACE: ``current`` is a
                # GUI slot the model got wrong, which owes a frame even though
                # nothing ran (R2a + R3); ``error_only`` is a text tool, which
                # owes none. Read that decision rather than re-deriving it.
                if tool_feedback.carrier == "current" and base is not None:
                    if (frame := await capture_frame()) is not None:
                        step_screenshots.append(frame)
                continue

            if name == "report_infeasible" or (
                name == "terminate" and args.get("status") == "failure"
            ) or (name == "response" and "[infeasible]" in str(args.get("text", "")).lower()):
                terminated = True
                stop_reason = name
                # ``report_infeasible`` shares this branch but is NOT a canonical
                # finish tool: ``lite/data/utils/rows.py``'s row contract keys
                # finish calls on ``LiteFinishToolSet`` alone, and sibling envs pair a
                # result to it on purpose. Only the canonical finish calls are
                # terminal for this rule.
                if name in LiteFinishToolSet.get_tool_names() and action.get("call_id"):
                    terminal_call_ids.add(action["call_id"])
                if base is not None:
                    try:
                        await loop.run_in_executor(
                            _EXECUTOR,
                            _rpc,
                            base,
                            "/step",
                            {"cmd": "FAIL", "pause": 0},
                            _STEP_RPC_TIMEOUT,
                        )
                    except Exception as e:
                        record_tool_execution_error(
                            action_errors, result_call_id, e, action_name=name
                        )
                        executed.append({
                            "call": "noop",
                            "args": {"name": name, "reason": str(e)},
                        })
                break
            if name in LiteFinishToolSet.get_tool_names():
                terminated = True
                stop_reason = name
                if action.get("call_id"):
                    terminal_call_ids.add(action["call_id"])
                break

            invalid_action = invalid_action_message(action, metadata.valid_actions)
            if invalid_action:
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(invalid_action)
                executed.append({
                    "call": "noop",
                    "args": {"name": name, "reason": invalid_action},
                })
                continue

            try:
                pyautogui_str = to_pyautogui(name, args, _SCREEN_W, _SCREEN_H)
            except MODEL_ACTION_ERROR_TYPES as e:
                record_model_action_error(
                    action_errors, result_call_id, e, action_name=name
                )
                executed.append({"call": "noop", "args": {"name": name, "reason": str(e)}})
                # POLICY: a MODEL fault costs this action and nothing more. It
                # never reached the screen, so the state the tail was chosen
                # against is exactly the state still on it -- there is nothing to
                # protect the tail from. The model gets the reason as text plus a
                # frame for this slot, then the batch continues. Contrast the
                # interface-failure arm: THERE the action may have half executed
                # and the screen is unknown, which is a different fault.
                if base is not None and (frame := await capture_frame()) is not None:
                    step_screenshots.append(frame)
                continue
            if pyautogui_str and base is not None:
                try:
                    await loop.run_in_executor(
                        _EXECUTOR,
                        _rpc,
                        base,
                        "/step",
                        {"cmd": pyautogui_str, "pause": 0},
                        _STEP_RPC_TIMEOUT,
                    )
                except (ConnectionResetError, ConnectionRefusedError) as e:
                    # R1: the BACKEND is gone (connect refused, reset, retries
                    # exhausted). The model cannot cause this and must never be
                    # shown it -- degrading it to feedback trains the model
                    # against a screen that stopped updating. Raise so the
                    # trajectory is discarded and retried against a fresh env.
                    raise TrueInfraFailure(
                        f"osworld backend failed during {name}: {e}"
                    ) from e
                except Exception as e:
                    # A non-200 is NOT separable here: the guest crashing on a
                    # pyautogui string the MODEL supplied looks the same as a
                    # server fault. Keep it model-visible rather than discard a
                    # trajectory that may be the model's own mistake.
                    record_tool_execution_error(
                        action_errors, result_call_id, e, action_name=name
                    )
                    executed.append({
                        "call": "noop",
                        "args": {"name": name, "reason": str(e)},
                    })
                    record_batch_abort(action_errors, result_call_id, actions[index + 1:])
                    break
                # Settle AFTER EACH action — mirrors upstream's per-action `time.sleep(pause)` (step()
                # sleeps at its end, default 2s). In a batched turn this lets action N's UI update before
                # action N+1 fires (else e.g. a `type` runs before a preceding `click` focuses the field).
                # Done host-side + async so the RPC executor thread isn't held for the delay (concurrency).
                if self._post_action_delay > 0:
                    await asyncio.sleep(self._post_action_delay)
                executed.append({"call": pyautogui_str})
                # Capture AFTER the settle delay, or the frame records the
                # pre-settle screen.
                if (frame := await capture_frame()) is not None:
                    step_screenshots.append(frame)
            elif name in ("screenshot", "cursor_position"):
                # Read-only: no pyautogui command to send, but the action DID run
                # and owes its frame like any other. No settle delay — it cannot
                # have changed the screen.
                if base is not None and (frame := await capture_frame()) is not None:
                    step_screenshots.append(frame)
            else:
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(
                        unsupported_action_message(name)
                    )
                executed.append({"call": "noop", "args": {"name": name}})

        self._step_count += 1
        truncated = not terminated and self._step_count >= self._max_steps
        if truncated:
            stop_reason = MAX_STEPS_STOP_REASON

        if base is not None and not step_screenshots:
            # Nothing ran (empty batch, a terminal-only turn, or every call
            # rejected). The turn still owes the model one current observation.
            if (frame := await capture_frame()) is not None:
                step_screenshots.append(frame)

        reward = None
        if (terminated or truncated) and base is not None:
            er = await loop.run_in_executor(_EXECUTOR, _rpc, base, "/evaluate", {}, _EVAL_RPC_TIMEOUT)
            reward = er.get("reward")
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward, terminated=terminated, truncated=truncated,
                info={
                    EXECUTED_ACTIONS_INFO_KEY: executed,
                    **({STOP_REASON_INFO_KEY: stop_reason} if stop_reason else {}),
                },
            ),
            ordered_call_ids=result_call_ids,
            continue_call_ids=[
                call_id for call_id in result_call_ids
                if call_id not in terminal_call_ids
            ],
            images=step_screenshots,
            text=None,
            feedback=action_errors,
        )

    async def close(self) -> None:
        loop = asyncio.get_event_loop()
        if self._container is not None:
            try:
                await loop.run_in_executor(_EXECUTOR, _rpc, self._container.base_url, "/close", {}, 15.0)
            except Exception:
                pass
        if self._pending_cf_future is not None:
            self._spawn_background_destroy(self._pending_cf_future, self._pending)
            self._pending_cf_future = None
            self._pending = None
        await self._teardown_existing()

    async def _teardown_existing(self) -> None:
        loop = asyncio.get_event_loop()
        for attr in ("_container", "_pending"):
            c = getattr(self, attr)
            if c is not None:
                try:
                    await loop.run_in_executor(_EXECUTOR, c.destroy)
                except Exception as e:
                    logger.warning("destroy %s failed: %s", c.name, e)
                finally:
                    # Null AFTER destroy returns, and only if still ours —
                    # an eager null blanks external_resource_id while the
                    # container is still alive in docker, inviting the drift
                    # reaper to race our own teardown; same fix as
                    # mobileworld's _destroy_container.
                    if getattr(self, attr) is c:
                        setattr(self, attr, None)

    def _spawn_background_destroy(self, cf: Any, container: OSWorldContainer | None) -> None:
        reap_pending_build(cf, container, thread_name="osworld-bg-destroy")


def reap_pending_build(cf: Any, container: Any, *, thread_name: str) -> None:
    """Reap an in-flight build()/start() future's container via the shared
    background cleanup shell. ``container`` may be None when ``cf`` is a build() future
    that RETURNS the container — we reap whatever it produced. The dying
    handle is captured as a LOCAL (never read back off shared state).
    osworld_2 imports this verbatim (same single-source pattern as
    the shared action projection)."""
    def _destroy(c: Any) -> None:
        if c is not None:
            c.destroy()   # idempotent, never raises (LiteContainerBase)

    spawn_background_destroy(
        cf,
        on_result=lambda res: _destroy(container if container is not None else res),
        on_failure=lambda _e: _destroy(container),
        thread_name=thread_name,
    )


# ---------------------------------------------------------------------------
# Task registration — from the vendored data/tasks.json (host needs no desktop_env)
# ---------------------------------------------------------------------------
# Regeneration reference (used to build data/tasks.json from osworld_evaluation_examples):
_GOOGLE_AUTH_TASK_IDS: frozenset[str] = frozenset({
    "0c825995-5b70-4526-b663-113f4c999dd2", "22a4636f-8179-4357-8e87-d1743ece1f81",
    "78aed49a-a710-4321-a793-b611a7c5b56b", "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc",
    "a0b9dc9c-fc07-4a88-8c5d-5e3ecad91bcb", "46407397-a7d5-4c6b-92c6-dbe038b1457b",
    "897e3b53-5d4d-444b-85cb-2cdc8a97d903", "b52b40a5-ad70-4c53-b5b0-5650a8387052",
})
_BLOCKED_TASK_IDS: frozenset[str] = frozenset({
    "121ba48f-9e17-48ce-9bc6-a4fb17a7ebba", "2888b4e6-5b47-4b57-8bf5-c73827890774",
    "47543840-672a-467d-80df-8f7c3b9788c9", "9f3f70fc-5afc-4958-a7b7-3bb4fcb01805",
    "b4f95342-463e-4179-8c3f-193cd7241fb2", "e1e75309-3ddb-4d09-92ec-de869c928143",
    "f8cfa149-d1c1-4215-8dac-4a0932bad3c2",
})


def _is_infeasible(task_config: dict[str, Any]) -> bool:
    func = (task_config.get("evaluator") or {}).get("func")
    return func == "infeasible" or (isinstance(func, list) and "infeasible" in func)


def _load_tasks() -> None:
    index_file = _DATA_DIR / "tasks.json"
    if not index_file.exists():
        logger.warning("osworld data/tasks.json missing at %s", index_file)
        return
    for rec in json.loads(index_file.read_text())["tasks"]:
        domain, task_id = rec["domain"], rec["task_id"]
        # No "task_id" here: identity is framework-injected (registry.register
        # for the spec; the LiteBaseEnv.metadata property for live envs).
        others: dict[str, Any] = {"domain": domain}
        if rec.get("exclude_reason"):
            others["exclude_reason"] = rec["exclude_reason"]
        config = OSWorldConfig(domain=domain, task_id=task_id, metadata=dict(others))
        register(
            key=f"osworld@{task_id}",
            entry_point=lambda *, cfg=config, **kw: OSWorldEnv(config=cfg, **kw),
            split="eval",
            # Same-source contract: registered copy == the env's
            # builder output; the two sides cannot drift.
            metadata=OSWorldEnv._task_metadata(config),
        )


_load_tasks()
registry.set_env_make_kwargs("osworld", CFG.make_kwargs)

# ---------------------------------------------------------------------------
# Services + backend family
# ---------------------------------------------------------------------------
from lite.gym.services import BackendFamily, register_family, register_services  # noqa: E402
from lite.gym.remote.reaper import ContainerServices  # noqa: E402


class OSWorldServices(ContainerServices):
    """DEDICATED container services — image/KVM/tun/qcow2 presence check on `ensure` (NO host
    desktop_env — it's in the image); `live_ids`/`reap` inherited (scoped by server_port)."""

    def ensure(self, env_id: str) -> None:
        _ensure_services(env_id)


register_services("osworld", OSWorldServices())
register_family("osworld", BackendFamily.DEDICATED)
