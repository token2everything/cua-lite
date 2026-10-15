"""WindowsAgentArena benchmark backed by a local, disposable QEMU VM.

Original Microsoft WindowsAgentArena tasks on a locally prepared Windows 11 VM
(one fresh QEMU-in-Docker container per episode; legacy metadata shape).

Usage::

    import asyncio, lite.gym as gym
    env = gym.make("waa@" + gym.registry.task_ids("waa", split="eval")[0], max_steps=15)
    asyncio.run(env.reset())

Prerequisites: run ``lite/gym/envs/waa/scripts/install.sh`` once
(builds/pulls the runner + prepared qcow2, KVM required). See the env README.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from lite.core.messages.final import (
    MAX_STEPS_STOP_REASON,
    STOP_REASON_INFO_KEY,
)
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import LITE_DESKTOP_KEY_ACTION_NAMES
from lite.core.tools.calls import RuntimeEnvAction
from lite.core.tools.extra_tools import LiteFinishToolSet, make_report_infeasible_tool
from lite.core.tools.schemas import BaseTools
from lite.gym.errors import CapacityExhausted, TrueInfraFailure
from lite.gym.registry import register, registry
from lite.gym.remote.reaper import ContainerServices
from lite.gym.services import EnvServerPoolable, EnvServerResource

if TYPE_CHECKING:
    from lite.gym.remote.scope import ServerScope
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvObservation,
    LiteEnvStepResult,
    LiteExecutedAction,
)
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.coordinate import norm_to_pixel
from lite.gym.utils.backend.model_inputs import (
    coerce_model_duration,
    project_model_keys,
)
from lite.gym.utils.config.identity import EnvIdentity
from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
    ToolErrorFeedback,
    append_feedback,
    error_only_feedback,
    record_batch_abort,
    record_model_action_error,
    record_tool_execution_error,
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
    copy_valid_actions,
    resolve_extra_tools,
    resolve_valid_actions,
)
from lite.gym.wrappers import overlay_cursor_px
from lite.utils.image import png_from_b64_async

from .qemu import QemuConfig, QemuInstance, reap_runtime_slots

logger = logging.getLogger(__name__)

ENV_ID = "waa"
ENV_DIR = Path(__file__).parent
CFG = env_config.load(str(ENV_DIR))
DATA_PATH = ENV_DIR / "data" / "tasks.json"
ASSET_MANIFEST_PATH = ENV_DIR / "data" / "assets.json"

_MAX_STEPS = CFG.env_kwargs["max_steps"]
_SEED = CFG.env_kwargs["seed"]
_VALID_ACTIONS_CONFIG = CFG.env_kwargs.get("valid_actions")
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
_BASE_DISK = CFG.env_kwargs["base_disk"]
_ASSETS_DIR = CFG.env_kwargs["assets_dir"]
_RUNNER_IMAGE = CFG.env_kwargs["runner_image"]
_ACTION_TIMEOUT_S = CFG.server_kwargs["action_timeout_s"]
_CONTROL_TIMEOUT_S = CFG.server_kwargs["control_timeout_s"]
_EVALUATE_TIMEOUT_S = CFG.server_kwargs["evaluate_timeout_s"]
_CLOSE_TIMEOUT_S = CFG.server_kwargs["close_timeout_s"]

_RUNTIME_ROOT = CFG.server_kwargs["runtime_root"]
_SNAPSHOT_DIR = CFG.server_kwargs["snapshot_dir"]
_VCPUS = CFG.server_kwargs["vcpus"]
_MEMORY_GB = CFG.server_kwargs["memory_gb"]
_SHM_SIZE = CFG.server_kwargs["shm_size"]
_BIND_ADDRESS = CFG.server_kwargs["bind_address"]
_READY_TIMEOUT_S = CFG.server_kwargs["ready_timeout_s"]
_READINESS_POLL_INTERVAL_S = CFG.server_kwargs["readiness_poll_interval_s"]

_VALID_ACTIONS = resolve_valid_actions(
    _VALID_ACTIONS_CONFIG,
    env_name="waa",
    platform="desktop",
)

class WaaTools(BaseTools):
    """What waa declares beyond the GUI surface: ``report_infeasible``."""

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "report_infeasible": make_report_infeasible_tool(
            description="Report that the task cannot be completed in the current VM.",
        ),
    }


#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = WaaTools.get_tool_names() | LiteFinishToolSet.get_tool_names()


# Variants that receive full reward with NO agent action: the prepared image or
# the evaluator's accepted-success set already satisfies the check on the setup
# state. Excluded so the reported score reflects real agent work (mirrors
# lite.osworld's manual ``block:`` reasons, which are not derivable from the
# upstream evaluator func). Keyed by (upstream config id, variant) because the
# defect is split-specific — variant "standard" is the ``eval`` split,
# "no_context" is ``eval_noctxt``.
_TRIVIAL_REWARD_EXCLUSIONS: dict[tuple[str, str], str] = {
    ("34a4fee9-e52e-4a4a-96d2-68d35091504a-WOS", "standard"):
        "block: File Explorer already opens in Details view — reward 1 with no action",
    ("4d34ff3b-5cc8-44b2-a272-fb07927e996e-WOS", "standard"):
        "block: absence of the target Amazon cookie is accepted as success",
    ("4d34ff3b-5cc8-44b2-a272-fb07927e996e-WOS", "no_context"):
        "block: absence of the target Amazon cookie is accepted as success",
    ("9504989a-0d6e-4017-aefb-d359f6c752aa-wos", "standard"):
        "block: prepared image already uses the requested time zone",
    ("9504989a-0d6e-4017-aefb-d359f6c752aa-wos", "no_context"):
        "block: prepared image already uses the requested time zone",
    ("7c70e16b-e14f-4baa-b046-3e022b2d0305-WOS", "no_context"):
        "block: setup state already satisfies the accepted file ordering",
}


def _task_exclude_reason(task_config: dict[str, Any], variant: str) -> str | None:
    """Return the registry filter label — upstream ``infeasible`` contracts plus
    the hand-curated split-specific trivial-reward blocks."""
    func = task_config.get("evaluator", {}).get("func")
    if func == "infeasible" or (isinstance(func, list) and "infeasible" in func):
        return "infeasible"
    return _TRIVIAL_REWARD_EXCLUSIONS.get((task_config["id"], variant))


def _task_metadata_others(
    task_config: dict[str, Any],
    *,
    domain: str,
    variant: str,
) -> dict[str, Any]:
    others: dict[str, Any] = {
        "domain": domain,
        "variant": variant,
        "benchmark": "WindowsAgentArena",
        "upstream_task_id": task_config["id"],
        "os": "windows",
    }
    exclude_reason = _task_exclude_reason(task_config, variant)
    if exclude_reason:
        others["exclude_reason"] = exclude_reason
    return others


class WindowsAgentArenaBridgeClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=None)

    async def post(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        response = await self._client.post(
            f"{self._base_url}{path}",
            json=payload or {},
            timeout=timeout,
        )
        # The WAA guest bridge returns the screenshot as inline `screenshot_b64`
        # in the JSON body, so this parse walks a multi-MB base64 string. Offload
        # it off the single env-server event loop (mirrors the png_from_b64_async
        # decode that follows) so a co-tenant env's step/reset isn't stalled.
        try:
            data = await asyncio.to_thread(response.json)
        except ValueError:
            data = {}
        if response.is_error:
            detail = data.get("error") or response.text
            raise RuntimeError(f"WAA bridge {path} returned HTTP {response.status_code}: {detail}")
        if not data.get("ok", False):
            raise RuntimeError(data.get("error") or f"WAA bridge call failed: {path}")
        return data

    async def close(self) -> None:
        await self._client.aclose()


def _px(coord: list[int] | None, screen_w: int, screen_h: int) -> tuple[int, int]:
    # Canonical round+clamp.
    return norm_to_pixel(coord, screen_w, screen_h, on_malformed="raise")


def _to_pyautogui(
    name: str,
    args: dict[str, Any],
    screen_w: int,
    screen_h: int,
) -> str | None:
    """Translate one Lite desktop action into code executed by WAA's guest server."""

    def _keys_to_pyautogui() -> list[str]:
        return project_model_keys(
            args.get("keys", []),
            action_name=name,
            backend="pyautogui",
        )

    if name == "click":
        x, y = _px(args.get("coordinate"), screen_w, screen_h)
        button = args.get("button", "left")
        clicks = max(1, int(args.get("clicks", 1)))
        return f"pyautogui.click({x}, {y}, clicks={clicks}, button={button!r})"
    if name == "mouse_move":
        x, y = _px(args.get("coordinate"), screen_w, screen_h)
        return f"pyautogui.moveTo({x}, {y})"
    if name in {"mouse_down", "mouse_up"}:
        method = "mouseDown" if name == "mouse_down" else "mouseUp"
        return f"pyautogui.{method}(button={args.get('button', 'left')!r})"
    if name == "drag":
        end_x, end_y = _px(args.get("coordinate"), screen_w, screen_h)
        button = args.get("button", "left")
        duration = coerce_model_duration(
            args.get("duration", 0.5),
            action_name="drag",
        )
        start = args.get("start_coordinate")
        prefix = ""
        if start:
            start_x, start_y = _px(start, screen_w, screen_h)
            prefix = f"pyautogui.moveTo({start_x}, {start_y}); "
        return f"{prefix}pyautogui.dragTo({end_x}, {end_y}, duration={duration}, button={button!r})"
    if name == "scroll":
        coord = args.get("coordinate")
        prefix = ""
        if coord:
            x, y = _px(coord, screen_w, screen_h)
            prefix = f"pyautogui.moveTo({x}, {y}); "
        amount = int(args.get("amount", 3))
        direction = args.get("direction", "down")
        if direction in {"left", "right"}:
            delta = -amount if direction == "left" else amount
            return f"{prefix}pyautogui.hscroll({delta})"
        delta = amount if direction == "up" else -amount
        return f"{prefix}pyautogui.scroll({delta})"
    if name == "type":
        return f"pyautogui.write({str(args.get('text', ''))!r}, interval=0.02)"
    if name in LITE_DESKTOP_KEY_ACTION_NAMES:
        keys = _keys_to_pyautogui()
        if name == "key":
            return f"pyautogui.hotkey({', '.join(repr(key) for key in keys)})"
        if name == "key_down":
            return "; ".join(f"pyautogui.keyDown({key!r})" for key in keys)
        if name == "key_up":
            return "; ".join(f"pyautogui.keyUp({key!r})" for key in reversed(keys))
        duration = coerce_model_duration(
            args.get("duration", 1.0),
            action_name="hold_key",
        )
        down = "; ".join(f"pyautogui.keyDown({key!r})" for key in keys)
        up = "; ".join(f"pyautogui.keyUp({key!r})" for key in reversed(keys))
        return f"{down}; time.sleep({duration}); {up}"
    if name == "wait":
        duration = coerce_model_duration(
            args.get("duration", 1.0),
            action_name="wait",
        )
        return f"time.sleep({duration})"
    if name in {"screenshot", "cursor_position"}:
        return None
    raise ValueError(f"unsupported WindowsAgentArena action: {name}")


def _cursor_px_from_action(
    name: str,
    args: dict[str, Any],
    screen_w: int,
    screen_h: int,
) -> tuple[int, int] | None:
    """Return the cursor position after a WAA action when the action moves it."""
    if name in {"click", "mouse_move"}:
        return _px(args.get("coordinate"), screen_w, screen_h)
    if name == "drag":
        return _px(args.get("coordinate"), screen_w, screen_h)
    if name == "scroll" and args.get("coordinate"):
        return _px(args.get("coordinate"), screen_w, screen_h)
    return None


class WindowsAgentArenaEnv(EnvServerPoolable, EnvServerResource):

    EXTRA_TOOLS: ClassVar[type[BaseTools]] = WaaTools

    def __init__(
        self,
        *,
        base_disk: str = _BASE_DISK,
        assets_dir: str = _ASSETS_DIR,
        runner_image: str = _RUNNER_IMAGE,
        task: Any = None,
        task_config: dict[str, Any] | None = None,
        domain: str | None = None,
        variant: str | None = None,
        max_steps: int = _MAX_STEPS,
        seed: int | None = _SEED,
        valid_actions: list[str] | None = _VALID_ACTIONS_CONFIG,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        post_action_delay: float = _POST_ACTION_DELAY,
        cursor: bool = True,
    ) -> None:
        # Immutable per env lifetime: everything that defines the VM. Server-wide
        # knobs (vcpus/memory/runtime_root/snapshot_dir/...) come from module-level
        # server_kwargs, identical for every instance, so they are not per-instance
        # constructor kwargs.
        self._qemu_config = QemuConfig(
            base_disk=Path(base_disk).expanduser().resolve(),
            runner_image=runner_image,
            runtime_root=Path(_RUNTIME_ROOT).expanduser().resolve(),
            assets_dir=Path(assets_dir).expanduser().resolve(),
            snapshot_dir=Path(_SNAPSHOT_DIR).expanduser().resolve(),
            vcpus=_VCPUS,
            memory_gb=_MEMORY_GB,
            shm_size=_SHM_SIZE,
            bind_address=_BIND_ADDRESS,
            ready_timeout_s=_READY_TIMEOUT_S,
            readiness_poll_interval_s=_READINESS_POLL_INTERVAL_S,
        )
        self._instance: QemuInstance | None = None
        self._bridge: WindowsAgentArenaBridgeClient | None = None
        self._vm_used = False
        self._screen_w = 1440
        self._screen_h = 900
        self._cursor_px = (self._screen_w // 2, self._screen_h // 2)
        self._cursor = True
        self._last_screenshot: bytes | None = None
        # Safe defaults for task-less construction before a real task is
        # bound; bind() overwrites these.
        self._task_config: dict[str, Any] | None = None
        self._domain: str | None = None
        self._variant: str | None = None
        self._extra_tool_schemas: list[dict[str, Any]] = []
        self.bind(
            task,
            task_config=task_config,
            domain=domain,
            variant=variant,
            max_steps=max_steps,
            seed=seed,
            valid_actions=valid_actions,
            extra_tools=extra_tools,
            post_action_delay=post_action_delay,
            cursor=cursor,
        )

    def _reset_cursor(self) -> None:
        self._cursor_px = (self._screen_w // 2, self._screen_h // 2)

    async def _park_cursor(self) -> dict[str, Any]:
        """Park the REAL guest pointer at screen centre and return the frame
        captured after it landed.

        The bridge exposes no cursor-position read (its six routes are health /
        reset / step / evaluate / screenshot / close — see ``docker/bridge.py``),
        so the centre cannot be *queried*. It can, however, be *established*:
        ``/step`` executes arbitrary pyautogui in the guest, so a real
        ``moveTo`` puts the pointer where we claim it is, and the response
        screenshot is grabbed after the move — capture and cursor from the same
        instant, as in the lite.* sandbox path. Every later action that moves the
        pointer is likewise one we issued at a known coordinate
        (``_cursor_px_from_action``), so the tracked value stays true.

        (A queryable coordinate does exist one hop deeper: the prepared guest
        image patches WAA's server to return ``pyautogui.position()`` as JSON —
        ``docker/prep/Dockerfile``. Exposing it would need a new bridge route and
        a runner-image rebuild; parking the pointer needs neither.)
        """
        assert self._bridge
        x, y = self._cursor_px
        return await self._bridge.post(
            "/step",
            {"action": f"pyautogui.moveTo({x}, {y})", "pause": 0.0},
            timeout=_CONTROL_TIMEOUT_S,
        )

    async def _set_last_screenshot(self, screenshot_b64: Any) -> bytes | None:
        """Store and return the frame carried by a bridge response.

        Returns ``None`` — leaving the previous frame in place — when the
        response carried no screenshot, so a caller collecting per-action frames
        records nothing rather than a copy of the previous frame.

        INVARIANT: the composited coordinate is the real guest pointer
        position — established by ``_park_cursor`` at reset and thereafter moved
        only by pyautogui commands we emitted at coordinates we computed."""
        image = await png_from_b64_async(screenshot_b64)
        if image is None:
            return None
        x, y = self._cursor_px
        if self._cursor:
            try:
                self._last_screenshot = await asyncio.to_thread(overlay_cursor_px, image, x, y)
            except Exception:
                logger.debug("failed to overlay WindowsAgentArena cursor", exc_info=True)
                self._last_screenshot = image
        else:
            self._last_screenshot = image
        return self._last_screenshot

    def bind(
        self,
        task: Any = None,
        *,
        task_config: dict[str, Any] | None = None,
        domain: str | None = None,
        variant: str | None = None,
        max_steps: int = _MAX_STEPS,
        seed: int | None = _SEED,
        valid_actions: list[str] | None = _VALID_ACTIONS_CONFIG,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        post_action_delay: float = _POST_ACTION_DELAY,
        cursor: bool = True,
    ) -> None:
        # Compatibility callers may pass the task_id string positionally; resolve
        # it to the registered task. Direct construction passes the full task_config.
        if isinstance(task, str) and task:
            resolved = _task_by_id(task)
            task_config = resolved["config"]
            domain = resolved["domain"]
            variant = resolved["variant"]
        # dict(...): the resolved config is the shared registration-side
        # record — copy, don't alias (None = legacy no-task seed).
        self._task_config = dict(task_config) if task_config is not None else None
        self._domain = domain
        self._variant = variant
        self._seed = seed
        self._max_steps = max_steps
        # Soft env_kwarg, resolved through the shared helper so every env
        # answers ``valid_actions`` identically (None = full GUI surface,
        # [] = deliberately none, unknown name = raise at the config
        # boundary). Runtime invalid/unsupported feedback is owned by ``step``.
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="waa", platform="desktop",
        )
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        self._post_action_delay = post_action_delay
        self._cursor = cursor
        self._step_count = 0
        self._finished = False

    @staticmethod
    def _task_metadata(
        task_config: dict[str, Any] | None,
        *,
        domain: str | None = None,
        variant: str | None = None,
    ) -> LiteCUAMetadata:
        """Same-source metadata builder.
        ``task_config=None`` is the legacy no-task seed; extra_tool_schemas
        mirrors bind()'s default resolution."""
        others = (
            {"benchmark": "WindowsAgentArena", "os": "windows"}
            if task_config is None
            else _task_metadata_others(task_config, domain=domain, variant=variant)
        )
        return LiteCUAMetadata(
            dims=("desktop", "use"),
            valid_actions=copy_valid_actions(_VALID_ACTIONS),
            extra_tool_schemas=resolve_extra_tools(
                _EXTRA_TOOLS, tools=WaaTools, env_name="waa",
            ),
            others=others,
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # env_kwargs amendment: extra_tool_schemas resolved at bind.
        return dataclasses.replace(
            self._task_metadata(
                self._task_config, domain=self._domain, variant=self._variant,
            ),
            valid_actions=self._valid_actions,
            extra_tool_schemas=list(self._extra_tool_schemas),
        )

    @property
    def external_resource_id(self) -> str | None:
        return self._instance.name if self._instance else None

    async def boot(self) -> None:
        """Acquire the VM, run once and task-independent.
        Idempotent. With a ready snapshot present this is a ~15-50s restore.
        Only VM / bridge boot is retryable "warming"."""
        if self._instance is not None and self._bridge is not None:
            return
        await self.close()
        if self._instance is not None:
            # close() preserved the slot (docker rm timed out / failed) — don't
            # overwrite a still-running container reference; surface it for cleanup.
            raise RuntimeError(
                "previous WindowsAgentArena VM could not be removed; "
                "run scripts/cleanup.sh before retrying"
            )
        _check_runtime_dependencies(
            base_disk=self._qemu_config.base_disk,
            runner_image=self._qemu_config.runner_image,
            assets_dir=self._qemu_config.assets_dir,
        )
        identity = getattr(self, "identity", None) or EnvIdentity()
        instance = QemuInstance(config=self._qemu_config, task_id="pool", identity=identity)
        self._instance = instance
        try:
            await instance.start()
        except CapacityExhausted:
            # qemu.py raises this for "still warming" boot failures (container
            # exited / bridge not ready yet); surface it unchanged.
            await self.close()
            raise
        except (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.TransportError) as exc:
            # Transport-level hiccup while the VM/bridge stabilizes — retryable.
            # Classified by exception TYPE (not string matching), like _setup_task.
            await self.close()
            raise CapacityExhausted.warming(
                what=(
                    "WindowsAgentArena VM or bridge was not ready during boot: "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )
            ) from exc
        except BaseException:
            await self.close()
            raise
        assert instance.bridge_url
        self._bridge = WindowsAgentArenaBridgeClient(instance.bridge_url)
        self._vm_used = False

    async def reset(self) -> LiteEnvObservation:
        if self._task_config is None:
            raise RuntimeError("WindowsAgentArena reset() before a task was bound; bind() first")
        # A VM that already ran an episode is disposed so the next episode gets a
        # clean one; WAA has no in-guest revert.
        if self._vm_used:
            await self.close()
        await self.boot()
        self._step_count = 0
        self._finished = False
        result = await self._setup_task()
        self._vm_used = True
        return result

    async def _setup_task(self) -> LiteEnvObservation:
        # /reset runs WAA task setup (asset fetch + guest setup) inside the booted
        # guest. A genuine setup/config fault (the bridge raises RuntimeError) is
        # terminal and must surface — never masked as "warming". But a bare
        # transport-level hiccup while the just-booted guest stabilizes is retryable,
        # so those exception TYPES (not string matches) become a warming 503.
        assert self._bridge and self._instance
        try:
            data = await self._bridge.post(
                "/reset",
                {"task_config": self._task_config},
                timeout=self._qemu_config.ready_timeout_s,
            )
            screen = data.get("screen_size") or {}
            self._screen_w = int(screen.get("width") or self._screen_w)
            self._screen_h = int(screen.get("height") or self._screen_h)
            self._reset_cursor()
            # Only when we actually paint a cursor: otherwise this would perturb
            # the guest (hover / tooltips under the centre) for no benefit.
            if self._cursor:
                data = await self._park_cursor()
            await self._set_last_screenshot(data.get("screenshot_b64"))
            return LiteEnvObservation(image=self._last_screenshot,
                text=self._task_config["instruction"],
                metadata={"novnc_url": self._instance.novnc_url},
            )
        except (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.TransportError) as exc:
            await self.close()
            raise CapacityExhausted.warming(
                what=(
                    "WindowsAgentArena guest was not ready during task setup: "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )
            ) from exc
        except BaseException:
            await self.close()
            raise

    async def _finish(
        self,
        *,
        final_action: str | None,
        truncated: bool,
        stop_reason: str,
        step_screenshots: list[bytes],
        info: dict[str, Any] | None = None,
    ) -> LiteEnvStepResult:
        """End the episode, score it, and place its closing frame in ``step_screenshots``.

        The batch returns one frame per executed action, so where the closing
        frame belongs depends on whether an action produced it:

        * ``final_action`` set — a terminal call (report_infeasible / terminate /
          response) ran its own guest command and this frame was grabbed after
          it, so it is that action's frame and is appended.
        * ``final_action`` ``None`` — the step budget ended the episode, not an
          action. The frame supersedes the last executed action's frame instead
          of adding one, keeping the count equal to the actions that ran.
        """
        assert self._bridge
        if final_action:
            await self._bridge.post(
                "/step",
                {"action": final_action, "pause": 0.0},
                timeout=_CONTROL_TIMEOUT_S,
            )
        data = await self._bridge.post("/evaluate", timeout=_EVALUATE_TIMEOUT_S)
        shot = await self._bridge.post("/screenshot", timeout=_CONTROL_TIMEOUT_S)
        closing_frame = await self._set_last_screenshot(shot.get("screenshot_b64"))
        if closing_frame is not None:
            if final_action is None and step_screenshots:
                step_screenshots[-1] = closing_frame
            else:
                step_screenshots.append(closing_frame)
        self._finished = True
        result_info = dict(info or {})
        result_info[STOP_REASON_INFO_KEY] = stop_reason
        return LiteEnvStepResult(
            reward=float(data["reward"]),
            terminated=not truncated,
            truncated=truncated,
            info=result_info,
        )

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        metadata = self.metadata
        actions, ingress_errors = prepare_env_tool_calls(actions, metadata)
        if not self._bridge:
            raise RuntimeError("WindowsAgentArenaEnv.reset() must be called before step()")
        if self._finished:
            raise RuntimeError("WindowsAgentArena episode is already finished")

        executed: list[LiteExecutedAction] = []
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        result_call_ids = ordered_tool_call_ids(input_actions)
        # One frame PER EXECUTED ACTION, in action order (the contract the agent
        # layer consumes). The bridge already returns the frame it grabbed after
        # each action, so this costs no extra round-trip.
        step_screenshots: list[bytes] = []
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
                if tool_feedback.carrier == "current":
                    shot = await self._bridge.post(
                        "/screenshot", timeout=_CONTROL_TIMEOUT_S,
                    )
                    frame = await self._set_last_screenshot(
                        shot.get("screenshot_b64"),
                    )
                    if frame is not None:
                        step_screenshots.append(frame)
                continue
            # Model-emitted calls that ENDED the episode get no continuation
            # observation -- the three terminal returns below drop
            # env-internal projected ``action["call_id"]`` from
            # ``continue_call_ids``:
            # devs/migration/verify.py forbids a tool result for a terminal
            # call. NOT ``result_call_id``, because an INTERNAL finish call has
            # no model-call id and the loop-detect wrapper's injected
            # ``terminate`` carries the intercepted NON-finish model call's id
            # as ``_result_call_id`` -- that call must still be answered.
            if name == "report_infeasible":
                executed.append({"call": name, "args": {"command": None}})
                self._step_count += 1
                result = await self._finish(
                    final_action="FAIL",
                    truncated=False,
                    stop_reason="report_infeasible",
                    step_screenshots=step_screenshots,
                    info={EXECUTED_ACTIONS_INFO_KEY: executed},
                )
                return build_tool_results_from_decisions(
                    result,
                    ordered_call_ids=result_call_ids,
                    # ``report_infeasible`` is NOT a canonical finish tool, so it
                    # keeps its result: ``lite/data/utils/rows.py``'s row
                    # contract keys finish calls on ``LiteFinishToolSet`` alone.
                    continue_call_ids=result_call_ids,
                    images=step_screenshots,
                    feedback=action_errors,
                )
            if name == "terminate":
                executed.append({"call": name, "args": {"command": None}})
                self._step_count += 1
                result = await self._finish(
                    final_action=("FAIL" if args.get("status") == "failure" else "DONE"),
                    truncated=False,
                    stop_reason="terminate",
                    step_screenshots=step_screenshots,
                    info={EXECUTED_ACTIONS_INFO_KEY: executed},
                )
                return build_tool_results_from_decisions(
                    result,
                    ordered_call_ids=result_call_ids,
                    continue_call_ids=[
                        call_id for call_id in result_call_ids
                        if call_id != action.get("call_id")
                    ],
                    images=step_screenshots,
                    feedback=action_errors,
                )
            if name == "response":
                failed = "[INFEASIBLE]" in str(args.get("text", "")).upper()
                executed.append({"call": name, "args": {"command": None}})
                self._step_count += 1
                result = await self._finish(
                    final_action="FAIL" if failed else "DONE",
                    truncated=False,
                    stop_reason="response",
                    step_screenshots=step_screenshots,
                    info={EXECUTED_ACTIONS_INFO_KEY: executed},
                )
                return build_tool_results_from_decisions(
                    result,
                    ordered_call_ids=result_call_ids,
                    continue_call_ids=[
                        call_id for call_id in result_call_ids
                        if call_id != action.get("call_id")
                    ],
                    images=step_screenshots,
                    feedback=action_errors,
                )

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
                command = _to_pyautogui(name, args, self._screen_w, self._screen_h)
                cursor_after = _cursor_px_from_action(
                    name, args, self._screen_w, self._screen_h
                )
            except MODEL_ACTION_ERROR_TYPES as e:
                record_model_action_error(
                    action_errors, result_call_id, e, action_name=name
                )
                executed.append({
                    "call": name,
                    "args": {"command": None, "reason": str(e)},
                })
                # POLICY: a MODEL fault costs this action and nothing more. It
                # never reached the screen, so the state the tail was chosen
                # against is exactly the state still on it -- there is nothing to
                # protect the tail from. The model gets the reason as text plus a
                # frame for this slot, then the batch continues. Contrast the
                # interface-failure arm: THERE the action may have half executed
                # and the screen is unknown, which is a different fault.
                shot = await self._bridge.post("/screenshot", timeout=_CONTROL_TIMEOUT_S)
                if (frame := await self._set_last_screenshot(shot.get("screenshot_b64"))) is not None:
                    step_screenshots.append(frame)
                continue
            if command is None:
                # Read-only (screenshot / cursor_position): no guest command to
                # send, but the action DID run and owes its frame like any other.
                data = await self._bridge.post("/screenshot", timeout=_CONTROL_TIMEOUT_S)
                frame = await self._set_last_screenshot(data.get("screenshot_b64"))
                if frame is not None:
                    step_screenshots.append(frame)
                executed.append({"call": name, "args": {"command": command}})
            else:
                try:
                    data = await self._bridge.post(
                        "/step",
                        {"action": command, "pause": self._post_action_delay},
                        timeout=_ACTION_TIMEOUT_S,
                    )
                except httpx.TransportError as e:
                    # R1: the guest bridge is unreachable (connect refused, read
                    # timeout, socket reset). The model cannot cause this and
                    # must never be shown it -- degrading it to feedback trains
                    # against a screen that stopped updating. Raise so the
                    # trajectory is discarded and retried against a fresh env.
                    raise TrueInfraFailure(
                        f"waa bridge failed during {name}: {e}"
                    ) from e
                except Exception as e:
                    # A bridge HTTP error is NOT separable here: the guest
                    # rejecting a command the MODEL shaped looks the same as a
                    # server fault. Keep it model-visible rather than discard a
                    # trajectory that may be the model's own mistake.
                    record_tool_execution_error(
                        action_errors, result_call_id, e, action_name=name
                    )
                    executed.append({
                        "call": name,
                        "args": {"command": command, "reason": str(e)},
                    })
                    record_batch_abort(action_errors, result_call_id, actions[index + 1:])
                    break
                if cursor_after is not None:
                    self._cursor_px = cursor_after
                # ``pause`` settled the guest BEFORE the bridge grabbed this
                # frame, so it is this action's post-settle frame.
                frame = await self._set_last_screenshot(data.get("screenshot_b64"))
                if frame is not None:
                    step_screenshots.append(frame)
                executed.append({"call": name, "args": {"command": command}})

        if not step_screenshots and self._last_screenshot is not None:
            # Nothing ran (empty batch, or every call rejected). The turn still
            # owes the model one current observation, and no action changed the
            # guest, so the frame from the last one that did is the current one.
            # ``_finish`` supersedes it below when the step budget ends here.
            step_screenshots.append(self._last_screenshot)
        self._step_count += 1
        if self._step_count >= self._max_steps:
            result = await self._finish(
                final_action=None,
                truncated=True,
                stop_reason=MAX_STEPS_STOP_REASON,
                step_screenshots=step_screenshots,
                info={EXECUTED_ACTIONS_INFO_KEY: executed},
            )
            return build_tool_results_from_decisions(
                result,
                ordered_call_ids=result_call_ids,
                continue_call_ids=result_call_ids,
                images=step_screenshots,
                # Must match the non-truncating return below. `action_errors` is
                # written by `record_model_action_error` above, then `break`s
                # into this budget check -- so a model whose LAST turn had a bad
                # argument would otherwise be told nothing about why.
                feedback=action_errors,
            )
        return build_tool_results_from_decisions(
            LiteEnvStepResult(info={EXECUTED_ACTIONS_INFO_KEY: executed}),
            ordered_call_ids=result_call_ids,
            continue_call_ids=result_call_ids,
            images=step_screenshots,
            feedback=action_errors,
        )

    async def close(self) -> None:
        bridge, instance = self._bridge, self._instance
        self._bridge = None
        if bridge:
            try:
                await asyncio.wait_for(
                    bridge.post("/close", timeout=_CLOSE_TIMEOUT_S),
                    timeout=_CLOSE_TIMEOUT_S,
                )
            except Exception:
                pass
            await bridge.close()
        if instance:
            await instance.close()
            if instance.name is None and self._instance is instance:
                self._instance = None


def _load_tasks() -> list[dict[str, Any]]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"WindowsAgentArena task catalog missing: {DATA_PATH}. "
            "Run scripts/utils/sync_tasks.py from the repository checkout."
        )
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


_TASKS_BY_ID: dict[str, dict[str, Any]] | None = None


def _task_by_id(task_id: str) -> dict[str, Any]:
    """Resolve a task_id to its registered task (config/domain/variant).

    Used by bind() compatibility paths that receive a task_id string.
    """
    global _TASKS_BY_ID
    if _TASKS_BY_ID is None:
        _TASKS_BY_ID = {task["task_id"]: task for task in _load_tasks()}
    return _TASKS_BY_ID[task_id]


registry.set_env_make_kwargs(ENV_ID, CFG.make_kwargs)
for _task in _load_tasks():
    register(
        f"{ENV_ID}@{_task['task_id']}",
        WindowsAgentArenaEnv,
        split=_task["split"],
        # Same-source contract: registered copy == the env's builder
        # output (incl. the no-override extra_tool_schemas default — the full
        # catalog had zero registered-side consumers).
        metadata=WindowsAgentArenaEnv._task_metadata(
            _task["config"],
            domain=_task["domain"],
            variant=_task["variant"],
        ),
        task_config=_task["config"],
        domain=_task["domain"],
        variant=_task["variant"],
    )


def _check_runner_image(runner_image: str) -> None:
    from lite.gym.utils.backend.docker import require_image_present
    from lite.gym.utils.backend.freshness import image_for

    require_image_present(image_for(ENV_ID, tag=runner_image))


def _asset_manifest_digest() -> str:
    manifest = json.loads(ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _check_asset_cache(assets_dir: Path) -> None:
    marker_path = assets_dir / ".complete.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        marker = {}
    ok = marker.get("manifest_sha256") == _asset_manifest_digest()
    if ok:
        # The marker only proves the manifest matched at install time. Confirm each
        # content-addressed blob is actually present at the expected size, so a
        # truncated / bit-rotted / partially-deleted cache fails here instead of
        # opaquely inside the guest during task setup. Size-only to stay cheap
        # (install.sh status runs the full per-blob sha256 verification).
        manifest = json.loads(ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))
        for entry in manifest["assets"]:
            blob = assets_dir / entry["sha256"]
            if not blob.is_file() or blob.stat().st_size != entry["size"]:
                ok = False
                break
    if not ok:
        from lite.gym.errors import EnvDepsMissingError

        raise EnvDepsMissingError(
            what=f"WindowsAgentArena asset cache is missing or stale: {assets_dir}",
            install=(
                "uv run --no-sync bash "
                "lite/gym/envs/waa/scripts/install.sh"
            ),
            see="lite/gym/envs/waa/README.md",
        )


def _check_runtime_dependencies(
    *,
    base_disk: Path,
    runner_image: str,
    assets_dir: Path = Path(_ASSETS_DIR).expanduser(),
) -> None:
    _check_runner_image(runner_image)
    if not base_disk.is_file() or base_disk.stat().st_size == 0:
        from lite.gym.errors import EnvDepsMissingError

        raise EnvDepsMissingError(
            what=f"WindowsAgentArena base disk is missing or empty: {base_disk}",
            install=(
                "uv run --no-sync bash "
                "lite/gym/envs/waa/scripts/install.sh"
            ),
            see="lite/gym/envs/waa/README.md",
        )
    _check_asset_cache(assets_dir)


def _ensure_services(env_id: str) -> None:
    # The factory invokes service checks before it applies per-call kwargs, so
    # validating the default disk here would reject valid base_disk overrides.
    # The selected disk is checked by WindowsAgentArenaEnv.reset().
    _check_runner_image(_RUNNER_IMAGE)


class WindowsAgentArenaServices(ContainerServices):
    def ensure(self, env_id: str) -> None:
        _ensure_services(env_id)

    def reap(
        self, env_id: str, scope: ServerScope, in_use: set[str], *, boot: bool = False
    ) -> int:
        reaped = super().reap(env_id, scope, in_use, boot=boot)
        try:
            reaped += reap_runtime_slots(
                Path(_RUNTIME_ROOT),
                server_port=scope.server_port,
                boot=boot,
            )
        except Exception:
            logger.exception("failed to reap WindowsAgentArena runtime slots")
        return reaped


from lite.gym.services import BackendFamily, register_family, register_services  # noqa: E402

register_services(ENV_ID, WindowsAgentArenaServices())
register_family(ENV_ID, BackendFamily.DEDICATED)
