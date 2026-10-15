"""cua.bench.<mode>.<dataset> — task-oriented envs wrapping cua-bench.

cua-bench is a 3-level suite: **dataset** (``kicad``, ``basic``, …) → **environment**
(a task dir, e.g. ``154d0750``) → **variant**. Mapping onto cua-lite's ``env_id@task_id``
model (like ``browsergym.webarena@21``: collection @ task), **each dataset is one env_id**
— the collection — and the **task_id** is a specific task: the environment (single-variant,
e.g. ``154d0750``) or ``environment/variant`` (e.g. ``click-button/0``). The env_id also
names the backend mode: ``cua.bench.webtop.<dataset>`` (in-process, PURE) or
``cua.bench.local.<dataset>`` (local cua-xfce container, DEDICATED + drift-reaped), by each
environment's declared backend — so a uniform dataset yields one env_id, a mixed one yields both.

Registration scans a dataset root (``$CUA_BENCH_DATASET_ROOT`` if set, else the install.sh
``.cache/datasets`` default) and registers one key per task. Discovery is cheap —
``cb.make(path)`` + ``tasks_config_fn()`` only import + call a function; no sandbox is
spawned (that happens in ``reset()``).

Resolution: ``gym.make("cua.bench.local.kicad@154d0750")`` → registry ``_import_env`` finds
no ``envs/cua/bench/local/kicad/`` → walks up to ``cua.bench`` → imports THIS module →
registration runs.

Deps: ``cua-bench`` (lazy). Import raises ``EnvDepsMissingError`` when absent
(caught by the registry's ``_import_all``); the pure translation helpers live in
``translate.py`` (no cua-bench needed). Install:
``uv run --no-sync bash lite/gym/envs/cua/scripts/install.sh``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import EnvAction, RuntimeEnvAction
from lite.gym.base import LiteBaseEnv
from lite.gym.envs.cua.bench.translate import normalize_reward, to_cb_action
from lite.gym.envs.cua.utils import INSTALL as _INSTALL
from lite.gym.envs.cua.utils import SEE as _SEE
from lite.gym.envs.cua.utils import to_px, unpack
from lite.gym.errors import EnvDepsMissingError, TrueInfraFailure
from lite.gym.registry import register, registry
from lite.gym.remote.reaper import ContainerServices
from lite.gym.services import EnvServerResource
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.ports import allocate_ports, release_ports, touch_ports
from lite.gym.utils.config.identity import EnvIdentity
from lite.gym.utils.config.naming import format_container_name
from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
    ToolErrorFeedback,
    append_feedback,
    current_feedback,
    record_model_action_error,
)
from lite.gym.utils.feedback.ingress import (
    prepare_env_tool_calls,
    standalone_tool_call_feedback,
    unsupported_env_action_message,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)
from lite.gym.utils.feedback.surface import (
    copy_valid_actions,
    resolve_valid_actions,
)
from lite.gym.wrappers import overlay_cursor_px

logger = logging.getLogger(__name__)

_DATASET_ROOT_ENV = "CUA_BENCH_DATASET_ROOT"
# All registration defaults live in configs/default.yaml (cua-lite convention) — read via
# env_config.load, not hardcoded. ENV_DIR = this bench/ pkg (configs/ is co-located here,
# since the config is cua.bench-only). Override the whole file with $CUA_BENCH_CONFIG.
ENV_DIR = str(Path(__file__).resolve().parent)
CFG = env_config.load(ENV_DIR)
_DEFAULT_MAX_STEPS = CFG.env_kwargs["max_steps"]
_DEFAULT_EXTRA_TOOLS = CFG.env_kwargs.get("extra_tools", [])
_DEFAULT_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
# Native desktop resolution — cua-lite convention (1920×1080), forced on every native task's
# setup_config (the [0,1000] click space is derived from the live screenshot, so coords still map).
_SETUP_W = CFG.env_kwargs["width"]
_SETUP_H = CFG.env_kwargs["height"]
_DEFAULT_CURSOR = CFG.make_kwargs.get("cursor", True)
_VALID_ACTIONS = resolve_valid_actions(
    CFG.env_kwargs.get("valid_actions"),
    env_name="cua.bench",
    platform="desktop",
)
_SUPPORTED_CUA_BENCH_ACTIONS = frozenset({
    "click",
    "type",
    "key",
    "mouse_move",
    "drag",
    "scroll",
    "wait",
})
_CUA_LOCAL_PORT_RANGE = (24000, 24999)


def _port_aware_remote_session_class() -> type:
    from cua_bench.computers.remote import RemoteDesktopSession

    class CuaLiteRemoteDesktopSession(RemoteDesktopSession):
        """RemoteDesktopSession variant that preserves the Lite-assigned API port.

        cua-bench 0.2.11 forwards ``noVNC_port`` to ``Computer`` but drops
        ``api_port`` in full-lifecycle native mode. Lite owns the local
        container allocation here, so the wrapper supplies both ports at this
        boundary instead of letting every CUA container bind host ``8000``.
        """

        def __init__(
            self,
            *args: Any,
            api_port: int | None = None,
            noVNC_port: int | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            if api_port is not None:
                self._api_port = api_port
            if noVNC_port is not None:
                self._vnc_port = noVNC_port

        async def _ensure_computer(self) -> None:
            if self._client_only_mode:
                await super()._ensure_computer()
                return
            if self._initialized and self._computer is not None:
                return

            from computer import Computer

            image = self._image
            if not image:
                if self._os_type in ("windows", "win11", "win10"):
                    image = "trycua/cua-qemu-windows:latest"
                else:
                    image = "trycua/cua-xfce:latest"

            ports_to_touch = [p for p in (self._api_port, self._vnc_port) if p is not None]
            touch_ports(*ports_to_touch)
            self._computer = Computer(
                os_type=self._os_type,
                provider_type=self._provider_type,
                image=image,
                display=f"{self._width}x{self._height}",
                memory=self._memory,
                cpu=self._cpu,
                name=self._name or None,
                storage=self._storage or None,
                ephemeral=self._ephemeral,
                noVNC_port=self._vnc_port if self._vnc_port != 8006 else None,
                api_port=self._api_port,
            )
            await self._computer.run()
            self._initialized = True

            if hasattr(self._computer, "noVNC_port"):
                self._vnc_port = self._computer.noVNC_port
                self._vnc_url = f"http://localhost:{self._vnc_port}"

    return CuaLiteRemoteDesktopSession


def _install_port_aware_create_sandbox(cb_env: Any) -> None:
    original_create_sandbox = cb_env.create_sandbox

    async def create_sandbox(
        provider: str,
        provider_config: dict[str, Any] | None = None,
        setup_config: dict[str, Any] | None = None,
    ) -> None:
        if provider not in ("native", "computer"):
            await original_create_sandbox(
                provider=provider,
                provider_config=provider_config,
                setup_config=setup_config,
            )
            return

        from cua_bench.bot import Bot
        from cua_bench.computers import DesktopSetupConfig

        setup = dict(setup_config or {})
        session_config = dict(provider_config or {})
        for key in ("api_port", "noVNC_port"):
            value = setup.pop(key, None)
            if value is not None:
                session_config[key] = value

        cb_env.session_name = provider
        cb_env.session_config = session_config
        cb_env.setup_config = DesktopSetupConfig(**setup)
        cb_env.session = _port_aware_remote_session_class()(**session_config)
        cb_env.session.env = cb_env
        await cb_env.session.start(config=cb_env.setup_config, headless=cb_env.headless)
        cb_env.page = cb_env.session.page
        cb_env.bot = Bot(cb_env)

    cb_env.create_sandbox = create_sandbox


# cua-bench tasks are desktop GUI (webtop / native linux|windows) → desktop
# action space. (android cua-bench would want mobile — open question.)
def _task_metadata(
    extra_tool_schemas: list[dict[str, Any]] | None = None,
    valid_actions: list[str] | None = _VALID_ACTIONS,
) -> LiteCUAMetadata:
    """Same-source metadata builder, shared by
    CuaBenchEnv and CuaBenchLocalEnv. Fresh object per call; no per-task
    others."""
    return LiteCUAMetadata(
        dims=(
            LiteCUAMetadata.Platform.DESKTOP.value,
            LiteCUAMetadata.TaskType.USE.value,
        ),
        extra_tool_schemas=list(extra_tool_schemas or []),
        valid_actions=copy_valid_actions(valid_actions),
    )


def _require_cb() -> Any:
    try:
        import cua_bench as cb
    except ImportError as e:
        raise EnvDepsMissingError(
            what="`cua-bench` not installed", install=_INSTALL, see=_SEE
        ) from e
    return cb


class CuaBenchEnv(LiteBaseEnv):
    """One cua-bench (environment, variant) as a cua-lite env. reward = cua-bench evaluator."""

    # Backend that consumes our KeyAction/HotkeyAction: this base IS the in-process
    # webtop (Playwright) env; CuaBenchLocalEnv overrides to the native computer-server
    # (pynput). Drives key-token projection in `to_cb_action` (see translate.py).
    _KEY_BACKEND = "playwright"

    def __init__(self, env_path: str, variant_idx: int = 0, *, max_steps: int = _DEFAULT_MAX_STEPS,
                 extra_tools: list[str] | None = _DEFAULT_EXTRA_TOOLS,
                 valid_actions: list[str] | None = _VALID_ACTIONS,
                 post_action_delay: float = _DEFAULT_POST_ACTION_DELAY,
                 cursor: bool = _DEFAULT_CURSOR,
                 seed: int | None = None) -> None:
        # `seed` is the one framework kwarg the factory/rollout.py forwards to every env
        # (scripts/rollout.py --seed); accepted but unused — cua-bench task state is fixed
        # by the task's own setup. Accepted EXPLICITLY (no **kwargs) so a typo'd env_kwarg
        # still raises, matching browsergym's env.
        _ = seed
        self._env_path = env_path
        self._variant_idx = variant_idx
        self._max_steps = max_steps
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="cua.bench", platform="desktop",
        )
        self._post_action_delay = post_action_delay
        self._cursor = cursor
        # This base IS the in-process webtop (PURE) env: it uses each task's own declared
        # computer untouched (see _before_cb_reset). The local-container (DEDICATED) mode
        # is CuaBenchLocalEnv. Backend is fixed by the env_id, never a kwarg — so a PURE
        # webtop id can't be flipped into spawning an untracked container.
        self._step_count = 0
        self._cb: Any = None
        self._wh: tuple[int, int] = (_SETUP_W, _SETUP_H)
        self._cursor_px: tuple[int, int] = (_SETUP_W // 2, _SETUP_H // 2)
        # Placeholder until reset() PARKS the real pointer there — no session
        # exists yet, so no pointer is anywhere. Gates the composite.
        self._cursor_px_known = False
        self._last_png: bytes | None = None

    # The builder is module-level (shared by every CuaBench env_id this file
    # registers); expose it on the class for the family-wide convention.
    _task_metadata = staticmethod(_task_metadata)

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return self._task_metadata(self._extra_tool_schemas, self._valid_actions)

    async def reset(self) -> LiteEnvObservation:
        cb = _require_cb()
        self._cb = cb.make(self._env_path)
        self._before_cb_reset(self._cb)   # webtop: no-op; local: inject native provider + image
        png, task = await self._cb.reset(task_id=self._variant_idx)   # task_id = int variant index
        self._step_count = 0
        computer = getattr(task, "computer", None) or {}
        setup = computer.get("setup_config", {}) if isinstance(computer, dict) else {}
        # The [0,1000]→px coordinate space is the screenshot's OWN pixel size —
        # that's exactly what the agent sees and clicks on (cua-bench feeds x,y
        # straight into page.mouse.click). Prefer it over setup_config (which is
        # only what the task requested); fall back to setup_config, then default.
        self._wh = _png_wh(png) or (setup.get("width", _SETUP_W), setup.get("height", _SETUP_H))
        self._cursor_px = (self._wh[0] // 2, self._wh[1] // 2)
        # Make the centre a FACT, not an assumption. cua-bench's session protocol
        # (DesktopSession: screenshot / execute_action / ...) has no cursor read
        # in either backend — webtop is Playwright (write-only mouse) and the
        # native backend's get_cursor_position lives on an interface cua-bench
        # does not surface — so the position cannot be QUERIED. It can be
        # ESTABLISHED: execute a real MoveToAction, then re-grab the frame so the
        # capture and the cursor are the same instant (the lite.* semantics).
        # execute_action, not step(): parking the pointer is env setup, not one
        # of the agent's scored actions.
        if self._cursor:
            await self._cb.session.execute_action(
                cb.MoveToAction(x=self._cursor_px[0], y=self._cursor_px[1])
            )
            png = await self._cb.session.screenshot()
            self._cursor_px_known = True
        self._last_png = await self._capture_png(png)
        return LiteEnvObservation(image=self._last_png, text=task.description)

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        import cua_bench as cb  # present by now (reset already imported it)
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        metadata = self.metadata
        actions, ingress_errors = prepare_env_tool_calls(
            actions,
            metadata,
            validate_top_level_action=True,
        )
        done = False
        # Model-emitted calls that ENDED the episode. They get no continuation
        # observation: devs/migration/verify.py forbids a tool result for a
        # terminal call. Keyed on the env-internal projected ``action["call_id"]``
        # and NOT on ``result_call_id``, because an INTERNAL finish call has no
        # model-call id and the loop-detect wrapper's injected ``terminate``
        # carries the intercepted NON-finish model call's id as
        # ``_result_call_id`` -- that call must still be answered.
        terminal_call_ids: set[str] = set()
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        # One frame PER EXECUTED ACTION, in action order (the contract the agent
        # layer consumes). cua-bench's ``step()`` returns the frame it grabbed
        # right after the action, so this costs no extra round-trip.
        step_screenshots: list[bytes] = []
        known_standalone_tools = type(self).known_standalone_tool_names()
        for index, (a, result_call_id) in enumerate(actions):
            tool_feedback = standalone_tool_call_feedback(
                a, known_standalone_tools, metadata.extra_tool_schemas,
            )
            if tool_feedback is not None:
                if result_call_id:
                    append_feedback(action_errors, result_call_id, tool_feedback)
                # The shared layer already decided the SURFACE: ``current`` is a
                # GUI slot the model got wrong, which owes a frame even though
                # nothing ran (R2a + R3); ``error_only`` is a text tool, which
                # owes none. Read that decision rather than re-deriving it.
                if tool_feedback.carrier == "current" and self._last_png is not None:
                    step_screenshots.append(self._last_png)
                continue
            unsupported_action = unsupported_env_action_message(
                a["name"], _SUPPORTED_CUA_BENCH_ACTIONS,
            )
            if unsupported_action:
                if result_call_id:
                    append_feedback(
                        action_errors, result_call_id,
                        current_feedback(unsupported_action),
                    )
                # A GUI slot naming an action this env does not carry: same
                # surface as above, so it owes its frame too.
                if self._last_png is not None:
                    step_screenshots.append(self._last_png)
                continue
            try:
                cb_action, done = to_cb_action(
                    a, self._wh, cb, self._KEY_BACKEND, cursor_px=self._cursor_px
                )
            except MODEL_ACTION_ERROR_TYPES as e:
                record_model_action_error(
                    action_errors, result_call_id, e, action_name=a["name"]
                )
                # POLICY: a MODEL fault costs this action and nothing more. It
                # never reached the screen, so the state the tail was chosen
                # against is exactly the state still on it -- there is nothing to
                # protect the tail from. The model gets the reason as text plus a
                # frame for this slot, then the batch continues. Contrast the
                # interface-failure arm: THERE the action may have half executed
                # and the screen is unknown, which is a different fault.
                step_screenshots.append(self._last_png)
                continue
            try:
                png = await self._cb.step(cb_action)
            except Exception as e:
                if isinstance(e, TrueInfraFailure):
                    raise
                raise TrueInfraFailure(
                    f"cua.bench {a['name']} backend failed: {e}"
                ) from e
            self._update_cursor(a)
            self._last_png = await self._capture_png(png)
            step_screenshots.append(self._last_png)
            if done:
                if a.get("call_id"):
                    terminal_call_ids.add(a["call_id"])
                break
        # cua-bench's step() screenshots immediately (no settle); re-grab a settled frame
        # after a desktop repaint delay, so both evaluate() and the obs see the final state.
        # Only the LAST frame is re-grabbed: the earlier ones record the state each action
        # produced at the time it ran and cannot be re-taken later.
        if actions and self._post_action_delay > 0 and self._cb is not None:
            await asyncio.sleep(self._post_action_delay)
            self._last_png = await self._capture_png(
                await self._cb.session.screenshot()
            )
            if step_screenshots:
                step_screenshots[-1] = self._last_png
        if not step_screenshots and self._last_png is not None:
            # Nothing ran (empty batch, or every call rejected). The turn still
            # owes the model one current observation.
            step_screenshots.append(self._last_png)
        self._step_count += 1
        truncated = not done and self._step_count >= self._max_steps
        # Guard evaluate() like cua-bench's own runner (runners.py:135): a task
        # without an evaluator makes cb.evaluate() raise, not return no-reward.
        # Evaluate on truncation too (not just terminate) — cua-bench's runner
        # (runners.py:138) evaluates unconditionally after the step loop, so a
        # goal-reached-but-never-terminated rollout still gets scored (the session
        # is still open at max_steps). Only a mid-episode step yields reward=None.
        has_eval = getattr(self._cb, "evaluate_task_fn", None) is not None
        reward = (
            normalize_reward(await self._cb.evaluate())
            if ((done or truncated) and has_eval)
            else None
        )
        # Always emit a screenshot (fall back to the last one if actions was empty),
        # like cua.sandbox — otherwise the agent's post-step obs guard would raise.
        return build_tool_results_from_decisions(
            LiteEnvStepResult(reward=reward, terminated=done, truncated=truncated),
            ordered_call_ids=result_call_ids,
            continue_call_ids=[
                call_id for call_id in result_call_ids
                if call_id not in terminal_call_ids
            ],
            images=step_screenshots,
            feedback=action_errors,
        )

    async def close(self) -> None:
        if self._cb is not None:
            await self._cb.close()
            self._cb = None

    def _before_cb_reset(self, cb_env: Any) -> None:
        """Hook run after ``cb.make()``, before ``cb.reset()``. No-op for the webtop base
        (each task's own declared computer is used untouched); :class:`CuaBenchLocalEnv`
        overrides it to inject the native provider + our conformant image."""

    def _update_cursor(self, a: EnvAction) -> None:
        """Track the cursor position needed for Lite drag(start_coordinate=None)."""
        name, args = unpack(a)
        if name in ("click", "mouse_move", "drag", "scroll") and args.get("coordinate"):
            self._cursor_px = to_px(args["coordinate"], self._wh)

    async def _capture_png(self, png: bytes) -> bytes:
        """Return a screenshot PNG, adding the env-tracked cursor if configured.

        INVARIANT: the composited coordinate is the real pointer position, or
        nothing is composited. ``_cursor_px`` is only ever a coordinate we moved
        the pointer to — reset's parking MoveToAction, or a coordinate action
        (``_update_cursor``) — and ``_cursor_px_known`` is False until that
        parking move lands.
        """
        if not self._cursor or not self._cursor_px_known:
            return png
        return await asyncio.to_thread(
            overlay_cursor_px, png, self._cursor_px[0], self._cursor_px[1]
        )


# base image is per-dataset: CuaBenchLocalEnv/Services read
# CFG.for_override(dataset).server_kwargs["base_image"]


async def _docker(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace")


class CuaBenchLocalEnv(CuaBenchEnv, EnvServerResource):
    """cua.bench.local.<dataset> — a cua-bench task on a LOCAL cua-xfce container (DEDICATED).

    Reclaimed by the env-server's STOCK container drift-reaper with no cua-specific reaper
    code: cua's Computer SDK derives the container name from the image tag, so reset() tags
    the base image to a cua-lite-conformant name (format_container_name) and points the task
    at it — the container name then matches the reaper's ``^lite-env-<port>-…-<env_id>-``
    scope. Lite assigns each container its own host API/noVNC ports. Backend
    is fixed by the env_id, never a kwarg."""

    _KEY_BACKEND = "pynput"   # native container runs on the cua computer-server (pynput)

    def __init__(self, env_path: str, variant_idx: int = 0, *, env_id: str,
                 max_steps: int = _DEFAULT_MAX_STEPS,
                 extra_tools: list[str] | None = _DEFAULT_EXTRA_TOOLS,
                 valid_actions: list[str] | None = _VALID_ACTIONS,
                 post_action_delay: float = _DEFAULT_POST_ACTION_DELAY,
                 cursor: bool = _DEFAULT_CURSOR,
                 seed: int | None = None) -> None:
        super().__init__(env_path, variant_idx, max_steps=max_steps,
                         extra_tools=extra_tools, valid_actions=valid_actions,
                         post_action_delay=post_action_delay,
                         cursor=cursor, seed=seed)
        self._env_id = env_id            # registered id; must match the reaper's env_id
        self._cfg = CFG.for_override(env_id.rsplit(".", 1)[-1])   # per-dataset res + image
        self._provider_setup_config: dict = {}    # carries the conformant image into the computer
        self._container_name: str | None = None   # what `docker ps` shows (reaper key)
        self._image_tag: str | None = None        # the per-instance ref alias we create
        self._ports_owned: tuple[int, ...] = ()

    @property
    def external_resource_id(self) -> str | None:
        """The container name the drift reaper reconciles against (``None`` until reset)."""
        return self._container_name

    async def reset(self) -> LiteEnvObservation:
        # identity is set by the factory under an env-server; absent in direct mode (name then
        # omits the server_port segment; the reaper is env-server-only, so never sees it).
        identity = getattr(self, "identity", None) or EnvIdentity()
        # env_id is the dataset (e.g. cua.bench.local.kicad); include the environment name
        # + variant so containers under the same env_id are distinct and readable.
        repo = format_container_name(
            env_id=self._env_id, task_id=f"{Path(self._env_path).name}-{self._variant_idx}",
            suffix=uuid.uuid4().hex[:6], session_id=identity.resolved_session_id(),
            token_hash=identity.token_hash, server_port=identity.server_port,
        ).lower()   # docker image repos must be lowercase (a user's SESSION_ID may not be)
        image_tag = f"{repo}:latest"
        container_name = f"{repo}_latest"   # cua: image.replace(":","_")
        api_port, novnc_port = allocate_ports(
            n=2,
            range_start=_CUA_LOCAL_PORT_RANGE[0],
            range_end=_CUA_LOCAL_PORT_RANGE[1],
        )
        self._ports_owned = (api_port, novnc_port)
        base_image = self._cfg.server_kwargs["base_image"]
        rc, out = await _docker("tag", base_image, image_tag)
        if rc != 0:
            release_ports(*self._ports_owned)
            self._ports_owned = ()
            raise RuntimeError(f"docker tag {base_image} -> {repo}:latest failed: {out.strip()}")
        self._image_tag = image_tag
        self._container_name = container_name
        self._provider_setup_config = {
            "image": self._image_tag,
            "api_port": api_port,
            "noVNC_port": novnc_port,
        }
        try:
            return await super().reset()
        except Exception:
            await self.close()
            raise

    def _before_cb_reset(self, cb_env: Any) -> None:
        """Inject provider="native" + our conformant image before cb.reset() creates the
        session. For a computer-less task (KiCad) this supplies the provider (else cua-bench
        never creates a session → setup hits ``session.apps`` on None); for a native-declared
        task it overwrites the computer with our image.

        ⚠️ cua-bench INTERNAL CONTRACT (verified against cua-bench 0.2.11): reset()
        populates ``self.tasks`` from ``tasks_config_fn()`` iff ``self.tasks is None``.
        We pre-seed ``tasks``; assert the precondition so a version drift fails loud,
        not as reward-0."""
        assert getattr(cb_env, "tasks", None) is None, (
            "cua-bench contract changed: env.tasks already populated before reset() — "
            "provider injection would be ignored (see the version note above)")
        _install_port_aware_create_sandbox(cb_env)
        # Force the cua-lite desktop resolution (configs/default.yaml → 1920×1080) + os_type=linux
        # + our conformant image on EVERY native task, overriding upstream's per-task setup_config.
        # cua-bench declares mixed resolutions (basic 1024×768, workflows 1920×1080); cua-lite
        # normalizes all desktop tasks to one convention (matches osworld / lite.osworld).
        setup_config = {
            "width": self._cfg.env_kwargs["width"], "height": self._cfg.env_kwargs["height"],
            "os_type": "linux", **self._provider_setup_config,
        }
        tasks = cb_env.tasks_config_fn()
        for t in tasks:
            t.computer = {"provider": "native", "setup_config": setup_config}
        cb_env.tasks = tasks

    async def close(self) -> None:
        try:
            await super().close()   # cua-bench session.close() removes the container
        finally:
            if self._image_tag is not None:
                await _docker("rmi", self._image_tag)   # drop the ref alias (layers stay via base)
            if self._ports_owned:
                release_ports(*self._ports_owned)
            self._image_tag = self._container_name = None
            self._provider_setup_config = {}
            self._ports_owned = ()


class CuaBenchLocalServices(ContainerServices):
    """Stock container reconcile view (``live_ids``/``reap`` inherited) + a base-image check.
    Everything reaper-side is inherited; we only verify the base image is present so a
    mis-installed host fails fast instead of at first container boot."""

    def ensure(self, env_id: str) -> None:
        base_image = CFG.for_override(env_id.rsplit(".", 1)[-1]).server_kwargs["base_image"]
        try:
            r = subprocess.run(["docker", "image", "inspect", base_image],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except (OSError, subprocess.SubprocessError) as e:
            raise EnvDepsMissingError(
                what=f"docker not available for {env_id}", install=_INSTALL, see=_SEE) from e
        if r.returncode != 0:
            raise EnvDepsMissingError(
                what=f"base image `{base_image}` not present (needed by {env_id})",
                install=_INSTALL, see=_SEE)


#: Default dataset location — install.sh downloads the cua-bench datasets here, so a
#: fresh install "just works" with no ``$CUA_BENCH_DATASET_ROOT`` export (mirrors
#: osworld_g's ``.cache``). The env var overrides it (point at any single dataset dir).
_CACHE_DATASETS = Path(__file__).resolve().parent.parent / ".cache" / "datasets"


def _dataset_root() -> Path | None:
    root = os.environ.get(_DATASET_ROOT_ENV)
    if root:
        return Path(root)
    return _CACHE_DATASETS if _CACHE_DATASETS.is_dir() else None


def _dataset_name(dataset_dir: Path) -> str:
    """Short dataset label for the env_id — strip the conventional ``cua-bench-`` prefix
    (``cua-bench-kicad`` → ``kicad``). Lowercased so the env_id matches the (lowercased)
    docker container name the reaper scopes by (docker image repos must be lowercase)."""
    n = dataset_dir.name
    return (n[len("cua-bench-"):] if n.startswith("cua-bench-") else n).lower()


def _is_contained_path(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _iter_datasets(root: Path):
    """Yield ``(dataset_name, [environment_dirs])`` under ``root``. ``root`` may be a single
    dataset dir (environment dirs — each has ``main.py`` — directly inside) OR the install.sh
    ``.cache`` of several datasets (``cua-bench-basic/``, … — environments one level deeper)."""
    children = sorted(
        p for p in root.iterdir() if p.is_dir() and _is_contained_path(p, root)
    )
    direct = [c for c in children if (c / "main.py").is_file()]
    if direct:                                    # root is a single dataset dir
        yield _dataset_name(root), direct
        return
    for ds in children:                           # root is a cache of datasets
        envs = [
            gc for gc in sorted(ds.iterdir())
            if gc.is_dir()
            and (gc / "main.py").is_file()
            and _is_contained_path(gc, root)
        ]
        if envs:
            yield _dataset_name(ds), envs


_LOCAL_SVC: Any = None   # one shared (stateless) services instance for every local env_id


def _register_local_backend(env_id: str) -> None:
    """Declare a local-container env_id as a DEDICATED backend so the env-server's
    stock container drift-reaper reconciles it. Env-server mode cold-spawns a fresh
    container per claim, and Cua ports them apart."""
    global _LOCAL_SVC
    from lite.gym.services import BackendFamily, register_family, register_services
    if _LOCAL_SVC is None:
        _LOCAL_SVC = CuaBenchLocalServices()
    register_family(env_id, BackendFamily.DEDICATED)
    register_services(env_id, _LOCAL_SVC)


def _register_tasks() -> None:
    """Register cua-bench tasks with the cua-lite collection model: **env_id =
    ``cua.bench.<mode>.<dataset>``** (the collection, e.g. ``cua.bench.local.kicad``),
    **task_id = ``<environment>``** (single-variant) or ``<environment>/<variant>``. ``mode``
    = ``webtop`` (in-process, PURE) | ``local`` (cua-xfce container, DEDICATED) by each
    environment's declared backend — so a uniform dataset yields one env_id and a mixed one
    (e.g. dev ``example_tasks``) yields both. Dataset root = ``$CUA_BENCH_DATASET_ROOT`` if
    set, else the install.sh ``.cache/datasets`` default. No-op if absent; raises
    ``EnvDepsMissingError`` if cua-bench isn't installed."""
    cb = _require_cb()
    root = _dataset_root()
    if root is None or not root.is_dir():
        logger.debug("cua.bench: no dataset root (%s unset + no %s) — nothing registered",
                     _DATASET_ROOT_ENV, _CACHE_DATASETS)
        return
    from lite.gym.services import BackendFamily, register_family
    pure_eids: set[str] = set()
    local_eids: set[str] = set()
    for dataset, env_dirs in _iter_datasets(root):
        for env_dir in env_dirs:
            try:
                tasks = cb.make(str(env_dir)).tasks_config_fn()
                n = max(len(tasks), 1)
                # Backend is read from variant 0 — cua-bench environments share one computer
                # config across their variants (verified for the shipped datasets).
                comp = getattr(tasks[0], "computer", None) if tasks else None
                declared = comp.get("provider") if isinstance(comp, dict) else None
            except Exception as e:   # a broken environment shouldn't abort the whole sweep
                logger.warning(
                    "cua.bench: could not read %s (%s); assuming 1 variant, webtop",
                    env_dir,
                    e,
                )
                n, declared = 1, "webtop"
            # A task is in-process ONLY if it explicitly declares webtop/simulated; a
            # computer-LESS task (declared is None, e.g. KiCad) needs a native provider
            # injected, so it routes to `local` (webtop would crash on session.apps).
            in_process = declared in ("webtop", "simulated")
            eid = f"cua.bench.{'webtop' if in_process else 'local'}.{dataset}"
            # Per-dataset step budget from configs/default.yaml (base + overrides[dataset],
            # merged by CFG.for_override) so a bare gym.make / --env-id is sensible without a
            # rollout config; the rollout yaml / CLI still overrides per run.
            dcfg = CFG.for_override(dataset).env_kwargs
            max_steps = dcfg["max_steps"]
            post_action_delay = dcfg["post_action_delay"]
            for v in range(n):
                task_id = env_dir.name if n == 1 else f"{env_dir.name}/{v}"
                if in_process:
                    register(
                        f"{eid}@{task_id}",
                        entry_point=CuaBenchEnv,
                        split="eval",
                        metadata=_task_metadata(),
                        env_path=str(env_dir),
                        variant_idx=v,
                        max_steps=max_steps,
                        post_action_delay=post_action_delay,
                    )
                else:
                    register(
                        f"{eid}@{task_id}",
                        entry_point=CuaBenchLocalEnv,
                        split="eval",
                        metadata=_task_metadata(),
                        env_path=str(env_dir),
                        variant_idx=v,
                        env_id=eid,
                        max_steps=max_steps,
                        post_action_delay=post_action_delay,
                    )
            (pure_eids if in_process else local_eids).add(eid)
    for eid in sorted(pure_eids):
        register_family(eid, BackendFamily.PURE)
        registry.set_env_make_kwargs(eid, CFG.make_kwargs)
    for eid in sorted(local_eids):
        _register_local_backend(eid)               # DEDICATED + stock container drift-reaper
        registry.set_env_make_kwargs(eid, CFG.make_kwargs)


def _png_wh(png: bytes) -> tuple[int, int] | None:
    """(width, height) from a PNG's IHDR (big-endian uint32 at bytes 16:24)."""
    if len(png) >= 24 and png[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    return None


_register_tasks()
