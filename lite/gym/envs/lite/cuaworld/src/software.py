"""Shared registration core for gym-anything (CUAWorld) software environments.

Every ``lite.cuaworld.<software>`` sub-env is a thin config over this module: it picks the
upstream CUAWorld env, fetches assets, and registers tasks that run the
gym-anything ``pre_task`` hook (setup) + original ``verifier.py`` (eval) over
cua-lite's ``cua-lite/sandbox.linux`` exec-stdio runtime. Keeping all envs on this one
core is what guarantees identical trajectory/standard shape across softwares.

Sub-envs are registered (one line each) in
``lite/gym/envs/lite/cuaworld/main.py``::

    register_cuaworld_software("pymol", "pymol_env")

Materials (env.json + scripts + tasks + ``registered.json``) come from the
materials repo via ``scripts/install.sh``; images build on ``cua-lite/lite.cuaworld.base``
(the shared sandbox Dockerfile built with the ``ga`` desktop user), so fetched
hooks/verifiers that hardcode ``ga``/``/home/ga`` need no runtime rewrite (see
``adapter``).
"""

from __future__ import annotations

import json
import logging
import math
import stat
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from lite.core.tools.calls import runtime_rejected_reason
from lite.core.tools.extra_tools import BASH_TOOL_NAME
from lite.gym.envs.lite.cuaworld.src import image_spec
from lite.gym.envs.lite.cuaworld.src.adapter import (
    _DEFAULT_PREPARATION_TIMEOUT,
    _resolved_verifier_timeout,
    create_episode_dir,
    initialize_episode,
    record_episode_step,
    remove_episode_dir,
    run_cuaworld_post_start,
    run_cuaworld_setup,
    run_cuaworld_verify,
    write_episode_frame,
)
from lite.gym.errors import CuaWorldVerifierError, EnvDepsMissingError
from lite.gym.registry import registry
from lite.gym.remote.reaper import ContainerServices
from lite.gym.sandbox import SandboxTaskConfig, register_tasks
from lite.gym.sandbox.base import SandboxBaseEnv
from lite.gym.services import BackendFamily, register_family, register_services
from lite.gym.utils import config as env_config
from lite.gym.utils.feedback.ingress import prepare_env_tool_calls

logger = logging.getLogger(__name__)

_CUAWORLD_DIR = Path(__file__).resolve().parents[1]


def _materials_dir() -> Path:
    return _CUAWORLD_DIR / ".cache"

# Registration defaults come from configs/default.yaml (swap wholesale with
# LITE_CUAWORLD_CONFIG=<name>). Per-software resource overrides stay at
# register_cuaworld_software(...) in main.py.
CFG = env_config.load(str(_CUAWORLD_DIR))
_EK = CFG.env_kwargs
#: gym-anything contract: fresh runner per episode = Layer-B cap 0;
#: canonical telling in configs/default.yaml.
_MAX_RESETS_PER_CONTAINER = CFG.server_kwargs["max_resets_per_container"]
_COMPUTER_BASE: dict[str, Any] = _EK["computer"]

#: Upstream gym-anything default when a task declares no ``hooks.pre_task_timeout``.
_UPSTREAM_PRE_TASK_TIMEOUT = 600.0

#: Ceiling (seconds) on a task's declared ``hooks.pre_task_timeout``. THE single
#: number that couples the setup budget to the reset deadline:
#:   * :func:`_make_setup_fn` clamps any declaration above it (and WARNs), so no
#:     task can ask for more setup time than ``reset()`` is allowed to take;
#:   * :data:`MAKE_KWARGS` derives ``reset_timeout`` from it, so ``reset()`` is
#:     always allowed to take that long.
#: 1800 s is the largest value the corpus declares today (slicer3d's heavy
#: volume/segmentation setups: 1800 s x63, 1200 s x24, 900 s x43 on disk — every
#: >600 s declaration in lite.cuaworld lives in slicer3d). Raising this one
#: constant raises BOTH halves together; they cannot drift.
MAX_PRE_TASK_TIMEOUT = 1800.0
_STEP_TIMEOUT_RESERVE = 30.0

#: Env-wide make defaults registered via ``registry.set_env_make_kwargs`` —
#: configs/default.yaml's ``make_kwargs`` with a DERIVED ``reset_timeout``.
#:
#: ``SandboxBaseEnv.reset()`` runs boot + ``run_cuaworld_post_start`` + the task's
#: ``setup_fn``, and ``StepTimeoutWrapper`` bounds that whole sequence. The yaml's
#: ``reset_timeout`` is the boot/post_start budget (and is what
#: ``run_cuaworld_post_start`` is given verbatim below); the setup budget stacks on
#: top of it, so the wrapper deadline must be their SUM. Hand-writing the total in
#: the yaml is exactly the drift this expression removes: before, a task declaring
#: 1800 s of setup was killed by a 600 s ``reset_timeout`` and surfaced as an
#: ``EnvTimeoutError`` indistinguishable from a hung boot.
MAKE_KWARGS: dict[str, Any] = {
    **CFG.make_kwargs,
    "reset_timeout": float(CFG.make_kwargs["reset_timeout"]) + MAX_PRE_TASK_TIMEOUT,
}

_DEFAULT_POST_ACTION_DELAY: float = _EK["post_action_delay"]
_DEFAULT_MEMORY: str = _COMPUTER_BASE["memory"]
_DEFAULT_CPU: str = _COMPUTER_BASE["cpu"]
_DEFAULT_TIMEOUT: int = _COMPUTER_BASE["timeout"]
_DISPLAY: str = (
    f"{_COMPUTER_BASE['display_resolution'][0]}"
    f"x{_COMPUTER_BASE['display_resolution'][1]}"
)


def _verification_budgets_for_step(step_timeout: float) -> tuple[float, float]:
    """Return ``(verifier_timeout, preparation_timeout)`` below step timeout.

    ``StepTimeoutWrapper`` is the outer wall-clock deadline for the whole final
    step. If the host-side verifier/VLM budget exceeds it, the wrapper cancels
    first and the episode can finalize without a structured verifier reason.
    Clamp the inner budget so verifier timeout wins before the wrapper does.
    """
    if not math.isfinite(step_timeout) or step_timeout <= _STEP_TIMEOUT_RESERVE:
        raise ValueError("step_timeout is too small for terminal verification")
    step_budget = step_timeout - _STEP_TIMEOUT_RESERVE
    preparation_timeout = min(_DEFAULT_PREPARATION_TIMEOUT, step_budget)
    verifier_timeout = _resolved_verifier_timeout(None)
    if preparation_timeout + verifier_timeout <= step_budget:
        return verifier_timeout, preparation_timeout
    verifier_timeout = step_budget - preparation_timeout
    if verifier_timeout <= 0:
        raise ValueError("step_timeout leaves no verifier budget")
    return verifier_timeout, preparation_timeout


def _upstream_trajectory_actions(
    actions: list[dict[str, Any]], display: str
) -> list[dict[str, Any]]:
    """Translate accepted env-internal desktop actions to gym-anything actions."""
    width_text, height_text = display.lower().split("x", 1)
    width, height = int(width_text), int(height_text)
    translated: list[dict[str, Any]] = []

    def pixel(value: Any) -> list[int]:
        coordinate = value if isinstance(value, list) else []
        if len(coordinate) < 2:
            return [width // 2, height // 2]
        return [
            int(coordinate[0] / 1000.0 * width),
            int(coordinate[1] / 1000.0 * height),
        ]

    for action in actions:
        name = action["name"]
        args = action["arguments"]
        if name in {"terminate", "response"}:
            break
        if name == "cursor_position":
            continue
        if name == "click":
            point = pixel(args.get("coordinate"))
            button = args.get("button", "left")
            clicks = max(1, int(args.get("clicks", 1)))
            # ``double_click`` is LEFT-only in this vocabulary, so only a left
            # multi-click may collapse into it; a right/middle multi-click repeats
            # that button instead.
            key = {"right": "right_click", "middle": "middle_click"}.get(
                button, "left_click"
            )
            if key == "left_click" and clicks >= 2:
                translated.append({"mouse": {"double_click": point}})
                clicks -= 2
            for _ in range(clicks):
                translated.append({"mouse": {key: point}})
        elif name == "mouse_move":
            translated.append({"mouse": {"move": pixel(args.get("coordinate"))}})
        elif name in {"mouse_down", "mouse_up"}:
            coordinate = args.get("coordinate")
            if coordinate:
                translated.append({"mouse": {"move": pixel(coordinate)}})
            button = args.get("button", "left")
            state = name.removeprefix("mouse_")
            translated.append({
                "mouse": {"buttons": {f"{button}_{state}": True}}
            })
        elif name == "drag" and args.get("coordinate"):
            start = args.get("start_coordinate")
            if start:
                translated.append({"mouse": {"move": pixel(start)}})
            button = args.get("button", "left")
            translated.append({"mouse": {"buttons": {f"{button}_down": True}}})
            translated.append({"mouse": {"move": pixel(args["coordinate"])}})
            translated.append({"mouse": {"buttons": {f"{button}_up": True}}})
        elif name == "scroll":
            coordinate = args.get("coordinate")
            if coordinate:
                translated.append({"mouse": {"move": pixel(coordinate)}})
            amount = int(args.get("amount", 3))
            direction = args.get("direction", "down")
            if direction in {"up", "down"}:
                translated.append({
                    "mouse": {"scroll": amount if direction == "down" else -amount}
                })
            elif direction in {"left", "right"}:
                translated.append({
                    "mouse": {
                        "horizontal_scroll": (
                            -amount if direction == "left" else amount
                        )
                    }
                })
        elif name == "type":
            translated.append({"keyboard": {"text": args.get("text", "")}})
        elif name == "key":
            translated.append({"keyboard": {"keys": _upstream_record_keys(args.get("keys", []))}})
        elif name == "key_down":
            translated.append({
                "keyboard": {"keys_down": _upstream_record_keys(args.get("keys", []))}
            })
        elif name == "key_up":
            translated.append({
                "keyboard": {"keys_up": _upstream_record_keys(args.get("keys", []))}
            })
        elif name == "hold_key":
            keys = _upstream_record_keys(args.get("keys", []))
            translated.extend([
                {"keyboard": {"keys_down": keys}},
                {"action": "wait", "time": float(args.get("duration", 1.0))},
                {"keyboard": {"keys_up": keys}},
            ])
        elif name == "wait":
            translated.append({
                "action": "wait",
                "time": float(args.get("duration", 5.0)),
            })
        elif name == "screenshot":
            translated.append({"action": "screenshot"})
    return translated


def _upstream_record_keys(keys: Any) -> Any:
    """Project Lite canonical glyphs to upstream CUAWorld trajectory spelling."""
    if not isinstance(keys, (list, tuple)):
        return keys
    return ["plus" if key == "+" else key for key in keys]


def _accepted_actions_for_final_eval(
    actions: list[dict[str, Any]],
    metadata,
) -> list[dict[str, Any]]:
    accepted, _errors = prepare_env_tool_calls(
        actions,
        metadata,
        defer_extra_schema_validation_for={BASH_TOOL_NAME},
    )
    # Ingress now FORWARDS a rejected child so the env can answer it per action
    # and still frame its slot. This function feeds the reward evaluator, which
    # must only ever see actions that were actually performed -- a rejected one
    # counted here is an action the agent never took, scored as if it had.
    return [
        action
        for action, _result_id in accepted
        if runtime_rejected_reason(action) is None
    ]


def _materials_identity(upstream_env: str) -> str:
    return image_spec.materials_identity(upstream_env)


def _make_env_class(env_id: str, computer_config: dict) -> type[SandboxBaseEnv]:
    """Build a per-software SandboxBaseEnv subclass with fixed constructor
    container shape (image/display/resources). Mirrors lite.osworld's
    LiteOsworldEnv.

    Every task of one ``lite.cuaworld.<sw>`` shares one image, so the
    constructor surface is constant per env_id. Direct mode is unaffected — the
    constructor only pre-stamps the same computer_config the tasks already carry.
    """
    class _CUAWorldEnv(SandboxBaseEnv):
        # lite.cuaworld.* images build on the shared sandbox Dockerfile built with
        # the desktop unix user `ga` (USER=ga build-arg) — the upstream gym-anything
        # assets hardcode `ga` (task prompts, verifiers, setup scripts), so we match
        # the container to the assets. The exec session runs as `root`: the upstream
        # hooks drop to the desktop user via `su - ga -c` (GUI apps must not run as
        # root), passwordless only FROM root — a `ga`→`su - ga` would prompt for a
        # password and hang. The base default is now `user`, so cuaworld OVERRIDES
        # EXEC_USER back to `root` for these upstream `su - ga` hooks.
        EXEC_USER: ClassVar[str] = "root"
        # The AGENT-facing shell must NOT inherit that root exec identity: it is
        # the agent's own terminal on its own desktop, so it drops to the
        # unprivileged desktop account. `ga` is real in these images — the shared
        # sandbox Dockerfile is built `--build-arg USER=ga`
        # (scripts/install.sh: `docker build -t "${BASE_IMAGE}" --build-arg USER=ga`)
        # and does `useradd -m -s /bin/bash -G sudo "${USER}"` with HOME=/home/ga.
        # Unlike osworld/cuagym (agent uid == exec uid, a control lever only),
        # here `ga` != root is a genuine isolation boundary.
        AGENT_USER: ClassVar[str] = "ga"
        _CUAWORLD_COMPUTER_CONFIG: ClassVar[dict] = computer_config
        _CUAWORLD_ENV_ID: ClassVar[str] = env_id

        def __init__(
            self,
            task: Any = None,
            *,
            display_resolution: tuple[int, int] = (1920, 1080),
            image: str | None = None,
            vnc_port: int | None = None,
            env_id: str | None = None,
            setup_fn: Any = None,
            evaluate_step_fn: Any = None,
            evaluate_final_fn: Any = None,
            container: str | None = None,
            computer_config: dict[str, Any] | None = None,
            post_action_delay: float = 1.0,
            debug: bool = False,
            max_steps: int | None = None,
            seed: int | None = None,
            valid_actions: list[str] | None = None,
            extra_tools: list[str] | None = None,
        ) -> None:
            if computer_config is None:
                computer_config = type(self)._CUAWORLD_COMPUTER_CONFIG
            if env_id is None:
                env_id = type(self)._CUAWORLD_ENV_ID
            self._set_constructor_state(
                display_resolution=display_resolution,
                image=image,
                vnc_port=vnc_port,
                env_id=env_id,
                setup_fn=setup_fn,
                evaluate_step_fn=evaluate_step_fn,
                evaluate_final_fn=evaluate_final_fn,
                container=container,
                computer_config=computer_config,
                post_action_delay=post_action_delay,
                debug=debug,
            )
            self._cuaworld_episode_dir: Path | None = None
            self._cuaworld_finalized = False
            #: Reuse cap consumed by ``reset_with_recycle`` (0 — see default.yaml).
            self._max_resets_per_container = _MAX_RESETS_PER_CONTAINER
            self._cuaworld_post_start_computer: Any = None
            self._cuaworld_episode_task: Any = None
            self._cuaworld_episode_evaluate_final_fn: Callable | None = None
            self._cuaworld_episode_started_at: float | None = None
            self._cuaworld_timeout_sec: float | None = None
            self.bind(
                task,
                max_steps=max_steps,
                seed=seed,
                valid_actions=valid_actions,
                extra_tools=extra_tools,
            )

        def bind(self, task: Any = None, *, max_steps: int | None = None,
                 seed: int | None = None,
                 valid_actions: list[str] | None = None,
                 extra_tools: list[str] | None = None) -> None:
            # Compatibility callers may pass a task_id STRING instead of the task
            # object. Resolve it, then route the base soft kwargs (max_steps/seed)
            # through — signature matches SandboxBaseEnv.bind so an unknown soft
            # kwarg still fails fast, per the base contract.
            if isinstance(task, str) and task:
                from lite.gym.sandbox import lookup_task
                task = lookup_task(type(self)._CUAWORLD_ENV_ID, task)
            # ``extra_tools`` is forwarded verbatim: lite.cuaworld owns no
            # env-local catalog, so the base resolver serves it (``bash`` +
            # finish tools).
            super().bind(task, max_steps=max_steps, seed=seed,
                         valid_actions=valid_actions, extra_tools=extra_tools)
            # CUAWorld creates task-specific closures from each task bundle. A
            # reused env instance can be rebound across tasks, so refresh those
            # closures here without changing the shared sandbox contract.
            self._setup_fn = task.setup_fn if task is not None else None
            self._evaluate_step_fn = (
                task.evaluate_step_fn if task is not None else None
            )
            self._evaluate_final_fn = (
                task.evaluate_final_fn if task is not None else None
            )
            self._cuaworld_timeout_sec = (
                float(task.metadata.get("timeout_sec", 600))
                if task is not None
                else None
            )

        async def boot(self) -> None:
            await super().boot()
            computer = self._computer
            assert computer is not None
            if computer is not self._cuaworld_post_start_computer:
                await run_cuaworld_post_start(
                    computer,
                    timeout=float(CFG.make_kwargs["reset_timeout"]),
                )
                self._cuaworld_post_start_computer = computer

        async def _finalize_unfinished_episode(self) -> None:
            if (
                self._cuaworld_episode_dir is None
                or self._cuaworld_finalized
                or self._cuaworld_episode_task is None
                or self._computer is None
                or self._cuaworld_episode_evaluate_final_fn is None
            ):
                return
            try:
                await self._cuaworld_episode_evaluate_final_fn(
                    self._cuaworld_episode_task,
                    self._computer,
                    [],
                    self._debug,
                )
            except Exception:
                logger.debug(
                    "lite.cuaworld unfinished-episode finalization failed",
                    exc_info=True,
                )
            self._cuaworld_finalized = True

        def _clear_episode(self) -> None:
            remove_episode_dir(self._cuaworld_episode_dir)
            self._cuaworld_episode_dir = None
            self._cuaworld_episode_task = None
            self._cuaworld_episode_evaluate_final_fn = None
            self._cuaworld_episode_started_at = None

        def backend_alive(self) -> bool:
            return self._computer is not None

        async def reset_to_pristine(self) -> bool:
            # Exactly the old ``had_episode`` criterion, consulted only on
            # the first reset (cap 0): a never-started runner (no episode dir)
            # is kept; a runner that served is never rolled back → destroy.
            return self._cuaworld_episode_dir is None

        async def destroy_backend(self) -> None:
            # Layer-B destroy hook: finalize the previous episode
            # (guarded no-op when nothing to finalize), then drop the runner.
            try:
                await self._finalize_unfinished_episode()
            finally:
                self._clear_episode()
                await SandboxBaseEnv.close(self)

        async def reset(self):
            # Framework-owned fresh-runner recycle (cap 0):
            # finalize+teardown live in destroy_backend; the first-reset gate
            # keeps a freshly booted, never-served runner (the invariant the old
            # had_episode branch protected by hand).
            await self.reset_with_recycle()
            self._cuaworld_episode_dir = create_episode_dir()
            self._cuaworld_finalized = False
            try:
                result = await super().reset()
                # Container startup and task setup stay outside the episode
                # budget. super().reset() runs the upstream pre_task hook, so
                # starting before it can make the first agent step truncate.
                self._cuaworld_episode_started_at = time.monotonic()
                assert self._task is not None
                assert self._computer is not None
                setattr(
                    self._computer,
                    "_cuaworld_episode_dir",
                    self._cuaworld_episode_dir,
                )
                initialize_episode(
                    self._cuaworld_episode_dir,
                    env_id=self._task.metadata["others"]["upstream_env_id"],
                    lite_env_id=type(self)._CUAWORLD_ENV_ID,
                    task_id=self._task.metadata["others"]["upstream_task_id"],
                    container_name=self.external_resource_id,
                    screenshot=result.image,
                )
                self._cuaworld_episode_task = self._task
                self._cuaworld_episode_evaluate_final_fn = self._evaluate_final_fn
                return result
            except BaseException:
                self._clear_episode()
                raise

        async def step(self, actions):
            if self._cuaworld_episode_dir is None or self._computer is None:
                raise RuntimeError("lite.cuaworld step called before reset")
            input_actions = actions
            final_eval_actions = _accepted_actions_for_final_eval(
                input_actions,
                self.metadata,
            )
            step_index = self._step_count
            # Upstream reset and step 0 both write frame_00000; the first step
            # intentionally replaces the reset screenshot.
            frame_index = step_index
            trajectory_actions = _upstream_trajectory_actions(
                final_eval_actions, type(self)._CUAWORLD_COMPUTER_CONFIG["display"]
            )
            setattr(self._computer, "_cuaworld_pending_step_index", step_index)
            setattr(self._computer, "_cuaworld_pending_frame_index", frame_index)
            setattr(
                self._computer,
                "_cuaworld_pending_actions",
                trajectory_actions,
            )
            evaluate_final_fn = self._evaluate_final_fn

            async def evaluate_final_once(
                task: SandboxTaskConfig,
                computer: Any,
                actions: Any = None,
                debug: bool = False,
            ):
                assert evaluate_final_fn is not None
                try:
                    return await evaluate_final_fn(task, computer, actions, debug)
                finally:
                    # A terminal verifier may be cancelled by StepTimeoutWrapper.
                    # Either way, do not repeat upstream post_task hooks or VLM
                    # calls during close().
                    self._cuaworld_finalized = True

            if evaluate_final_fn is not None:
                self._evaluate_final_fn = evaluate_final_once
            try:
                result = await super().step(input_actions)
                timed_out = (
                    not result.terminated
                    and not result.truncated
                    and self._cuaworld_timeout_sec is not None
                    and self._cuaworld_episode_started_at is not None
                    and time.monotonic() - self._cuaworld_episode_started_at
                    >= self._cuaworld_timeout_sec
                )
                if timed_out:
                    if evaluate_final_fn is not None and not self._cuaworld_finalized:
                        evaluated = await evaluate_final_once(
                            self._task,
                            self._computer,
                            final_eval_actions,
                            self._debug,
                        )
                        if isinstance(evaluated, tuple):
                            result.reward, eval_info = evaluated
                            if eval_info:
                                result.info["eval"] = eval_info
                        else:
                            result.reward = evaluated
                    result.truncated = True
                    result.info["termination_reason"] = "timeout"
                if (
                    (result.terminated or result.truncated)
                    and evaluate_final_fn is not None
                    and not self._cuaworld_finalized
                ):
                    evaluated = await evaluate_final_once(
                        self._task,
                        self._computer,
                        final_eval_actions,
                        self._debug,
                    )
                    if isinstance(evaluated, tuple):
                        result.reward, eval_info = evaluated
                        if eval_info:
                            result.info["eval"] = eval_info
                    else:
                        result.reward = evaluated
                if result.terminated or result.truncated:
                    self._cuaworld_finalized = True
                if not result.terminated and not result.truncated:
                    frame_image = next(
                        (
                            tool_result.images[-1]
                            for tool_result in reversed(result.results)
                            if tool_result.images
                        ),
                        None,
                    )
                    record_episode_step(
                        self._cuaworld_episode_dir,
                        index=step_index,
                        actions=trajectory_actions,
                    )
                    if frame_image is not None:
                        write_episode_frame(
                            self._cuaworld_episode_dir,
                            frame_index,
                            frame_image,
                        )
                return result
            finally:
                self._evaluate_final_fn = evaluate_final_fn
                if self._computer is not None:
                    setattr(self._computer, "_cuaworld_pending_step_index", None)
                    setattr(self._computer, "_cuaworld_pending_frame_index", None)
                    setattr(self._computer, "_cuaworld_pending_actions", None)

        async def close(self) -> None:
            try:
                await self._finalize_unfinished_episode()
            finally:
                try:
                    await SandboxBaseEnv.close(self)
                finally:
                    self._clear_episode()

    _CUAWorldEnv.__name__ = f"CUAWorldEnv_{env_id.replace('.', '_')}"
    _CUAWorldEnv.__qualname__ = _CUAWorldEnv.__name__
    return _CUAWorldEnv


class CUAWorldServices(ContainerServices):
    """Env-server services for a lite.cuaworld.<sw>: image freshness check
    (``ensure``) + per-container reconciliation (``live_ids``/``reap`` inherited).
    lite.cuaworld images are built by ``scripts/install.sh`` and carry the same
    ``lite.src_hash`` label as other container envs; ``image_freshness`` folds in
    the locked materials revision."""

    def __init__(
        self,
        image: str,
        build_cmd: str,
        register_tasks_fn: Callable[[], None],
    ) -> None:
        super().__init__()
        self._image = image
        self._build_cmd = build_cmd
        self._register_tasks_fn = register_tasks_fn

    def register_tasks(self, env_id: str) -> None:
        self._register_tasks_fn()

    def ensure(self, env_id: str) -> None:
        from lite.gym.utils.backend.docker import require_image_present
        from lite.gym.utils.backend.freshness import image_for

        try:
            image = replace(
                image_for(env_id, tag=self._image),
                install=self._build_cmd,
            )
        except (OSError, ValueError) as exc:
            raise EnvDepsMissingError(
                what=f"{env_id} materials source is unavailable or invalid: {exc}",
                install=self._build_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            ) from exc
        require_image_present(image)


def _declared_mem_gb(cache_root: Path) -> int | None:
    """``resources.mem_gb`` from the materials' ``env.json`` — the memory the upstream
    gym-anything env declares it needs. ``None`` when the materials are not installed
    yet; a present-but-malformed declaration is an actionable materials/install
    error.

    File-I/O boundary, so the parse is guarded; every internal caller below trusts the
    returned value.
    """
    env_spec_path = cache_root / "env.json"
    if not env_spec_path.is_file():
        return None
    try:
        declared = int(json.loads(env_spec_path.read_text())["resources"]["mem_gb"])
    except (OSError, json.JSONDecodeError, UnicodeDecodeError,
            KeyError, TypeError, ValueError) as exc:
        raise EnvDepsMissingError(
            what=f"lite.cuaworld env.json has invalid resources.mem_gb in {env_spec_path}",
            install="uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh build <software>",
            see="lite/gym/envs/lite/cuaworld/README.md",
        ) from exc
    if declared <= 0:
        raise EnvDepsMissingError(
            what=f"lite.cuaworld env.json resources.mem_gb must be positive in {env_spec_path}",
            install="uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh build <software>",
            see="lite/gym/envs/lite/cuaworld/README.md",
        )
    return declared


def _resolve_memory(software: str, cache_root: Path, requested: str) -> str:
    """Container memory for one software: the larger of the caller's ``memory=`` and the
    upstream ``env.json`` ``resources.mem_gb`` declaration.

    The declaration is a REQUIREMENT (what the software's own maintainers say it needs);
    ``register_cuaworld_software(memory=...)`` is a lite-side knob that may raise it but
    must never silently lower it — booting under the requirement just moves the failure
    to a cgroup OOM-kill mid-task, scored as a plain reward-0.

    Before this resolver nothing in the cuaworld python read ``mem_gb`` at all (all 40
    registered softwares declare it; ``grep -rn mem_gb`` over the package found only
    env.json). It was written when the shared baseline was 4GB and most
    ``register_cuaworld_software`` lines carried their own 4/6/8GB ``memory=``, which
    left seven softwares booting below their declaration. The baseline is now 8GB and
    no registration passes ``memory=`` at all, so the floor only still fires for a
    software declaring MORE than 8 — blender3d (16) is the one such case. Keep the
    resolver: it is what makes a future baseline change safe, and it is the only thing
    reading the upstream requirement.

    ``--memory`` is a cgroup ceiling, not a reservation, so raising it to the
    upstream declaration does not consume RAM until a container actually needs
    the extra pages. Fleet concurrency still belongs in rollout orchestration.

    ``memory`` on this surface is always spelled ``"<N>GB"`` (default.yaml, and any
    ``register_cuaworld_software`` line that reintroduces one); no other spelling
    is parsed.
    """
    declared = _declared_mem_gb(cache_root)
    if declared is None or declared <= float(requested.removesuffix("GB")):
        return requested
    logger.info(
        "lite.cuaworld.%s: honouring env.json resources.mem_gb=%dGB over the "
        "registered memory=%s", software, declared, requested,
    )
    return f"{declared}GB"


def _task_dir(tasks_dir: Path, task_id: str) -> Path | None:
    if Path(task_id).name != task_id or task_id in {".", ".."}:
        return None
    task_dir = tasks_dir / task_id
    try:
        metadata = task_dir.lstat()
    except OSError:
        return None
    return task_dir if stat.S_ISDIR(metadata.st_mode) else None


def _make_setup_fn(task_dir: Path, task_spec: dict):
    async def setup_fn(task: SandboxTaskConfig, computer) -> None:
        hooks = task_spec.get("hooks") or {}
        declared = float(
            hooks.get("pre_task_timeout", _UPSTREAM_PRE_TASK_TIMEOUT)
            if isinstance(hooks, dict)
            else _UPSTREAM_PRE_TASK_TIMEOUT
        )
        # Clamp to the budget MAKE_KWARGS["reset_timeout"] was sized for. A task
        # asking for more would otherwise be cut off mid-setup by the reset
        # wrapper with a generic EnvTimeoutError; warn instead so the mismatch is
        # visible and the fix (raise MAX_PRE_TASK_TIMEOUT — which raises the
        # reset deadline in the same breath) is obvious.
        timeout = min(declared, MAX_PRE_TASK_TIMEOUT)
        if declared > MAX_PRE_TASK_TIMEOUT:
            logger.warning(
                "lite.cuaworld: task %s declares pre_task_timeout=%.0fs > "
                "MAX_PRE_TASK_TIMEOUT=%.0fs; clamping (reset_timeout=%.0fs is "
                "sized for the ceiling, not the declaration)",
                task_dir.name, declared, MAX_PRE_TASK_TIMEOUT,
                MAKE_KWARGS["reset_timeout"],
            )
        await run_cuaworld_setup(
            computer,
            task_dir,
            task_spec=task_spec,
            timeout=timeout,
        )

    return setup_fn


def _make_step_eval_fn():
    async def evaluate_step_fn(
        task: SandboxTaskConfig, computer, actions=None, debug=False,
    ):
        return 0.0

    return evaluate_step_fn


def _make_eval_fn(
    task_dir: Path,
    verifier_target: str,
    task_meta: dict,
    task_spec: dict,
    env_id: str,
    upstream_env_id: str,
):
    async def evaluate_final_fn(
        task: SandboxTaskConfig, computer, actions=None, debug=False,
    ):
        episode_dir = getattr(computer, "_cuaworld_episode_dir", None)
        pending_step_index = getattr(
            computer, "_cuaworld_pending_step_index", None
        )
        pending_frame_index = getattr(
            computer, "_cuaworld_pending_frame_index", None
        )
        pending_actions = getattr(computer, "_cuaworld_pending_actions", None)
        if (
            isinstance(episode_dir, Path)
            and isinstance(pending_step_index, int)
            and isinstance(pending_frame_index, int)
            and isinstance(pending_actions, list)
        ):
            record_episode_step(
                episode_dir,
                index=pending_step_index,
                actions=pending_actions,
            )
            try:
                screenshot = await computer.interface.screenshot()
            except Exception as exc:  # noqa: BLE001 - surface as eval failure
                raise CuaWorldVerifierError(
                    f"failed to capture final screenshot: {type(exc).__name__}",
                    phase="verify",
                    kind="screenshot_failed",
                    task=task_dir.name,
                ) from exc
            if isinstance(screenshot, bytes):
                write_episode_frame(
                    episode_dir,
                    pending_frame_index,
                    screenshot,
                )
        try:
            verifier_timeout, preparation_timeout = _verification_budgets_for_step(
                float(MAKE_KWARGS["step_timeout"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CuaWorldVerifierError(
                f"invalid terminal verification budget: {exc}",
                phase="verify",
                kind="timeout",
                task=task_dir.name,
            ) from exc
        return await run_cuaworld_verify(
            computer,
            task_dir,
            verifier_target,
            task_metadata=task_meta,
            task_spec=task_spec,
            env_id=upstream_env_id,
            lite_env_id=env_id,
            episode_dir=episode_dir,
            timeout=verifier_timeout,
            preparation_timeout=preparation_timeout,
        )

    return evaluate_final_fn


_EXCLUDE_REASONS: dict[str, dict[str, str]] | None = None

#: Path of the empirical no-LLM validation sweep's output. Exported so every reader
#: (this module, tests) opens the SAME file by the same name.
VALIDATION_EXCLUDES_PATH = _CUAWORLD_DIR / "data" / "validation_excludes.json"


def is_excludes_metadata_key(key: str) -> bool:
    """True for a top-level key of ``data/validation_excludes.json`` that is FILE
    METADATA rather than a software name.

    The file's contract is the ``_`` prefix: ``_meta`` (holding ``_comment``,
    ``_splits_covered`` and the ``_total`` self-checksum) is the only such key today,
    and any block added later must be ``_``-prefixed too. THE single definition of that
    rule — :func:`_exclude_reasons` (which must skip metadata) and the ``_total``
    checksum test (which must count exactly the non-metadata keys) both call this, so
    they cannot disagree about what counts as a software entry. The two used to spell
    it differently (``not startswith("_")`` here, ``!= "_meta"`` in the test): same
    answer while ``_meta`` is alone, silently divergent the moment it is not.
    """
    return key.startswith("_")


def _exclude_reasons() -> dict[str, dict[str, str]]:
    """``software -> {task_id: reason}`` from ``data/validation_excludes.json`` (the empirical
    no-LLM validation sweep). Baked into each task's ``metadata.others['exclude_reason']`` so a
    rollout drops flagged tasks with the SAME standard filter lite.osworld uses:
    ``--filter "lambda m: not m.others.get('exclude_reason')"``."""
    global _EXCLUDE_REASONS
    if _EXCLUDE_REASONS is None:
        def fail(message: str, exc: Exception | None = None) -> None:
            err = EnvDepsMissingError(
                what=f"lite.cuaworld validation_excludes.json is invalid: {message}",
                install="uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh build <software>",
                see="lite/gym/envs/lite/cuaworld/README.md",
            )
            if exc is not None:
                raise err from exc
            raise err

        try:
            doc = json.loads(VALIDATION_EXCLUDES_PATH.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            fail(str(exc), exc)
        if not isinstance(doc, dict):
            fail("top-level document must be an object")
        meta = doc.get("_meta")
        if not isinstance(meta, dict) or not isinstance(meta.get("_total"), int):
            fail("_meta._total integer checksum is required")
        table: dict[str, dict[str, str]] = {}
        for sw, tasks in doc.items():
            if is_excludes_metadata_key(sw):
                continue
            if not isinstance(sw, str) or not sw:
                fail(f"software key must be a non-empty string: {sw!r}")
            if not isinstance(tasks, dict):
                fail(f"{sw} entry must be an object")
            table[sw] = {}
            for task_id, reason in tasks.items():
                if not isinstance(task_id, str) or not task_id:
                    fail(f"{sw} task id must be a non-empty string: {task_id!r}")
                if not isinstance(reason, str) or not reason:
                    fail(f"{sw}/{task_id} reason must be a non-empty string")
                table[sw][task_id] = reason
        entries = sum(len(tasks) for tasks in table.values())
        if entries != meta["_total"]:
            fail(f"_meta._total={meta['_total']} but file contains {entries} entries")
        _EXCLUDE_REASONS = table
    return _EXCLUDE_REASONS


def _build_configs(
    tasks_dir: Path,
    task_ids: list[str],
    computer_config: dict,
    env_id: str,
    upstream_env: str,
    upstream_env_id: str,
) -> list[SandboxTaskConfig]:
    configs: list[SandboxTaskConfig] = []
    _excl = _exclude_reasons().get(env_id.rsplit(".", 1)[-1], {})
    for tid in task_ids:
        tdir = _task_dir(tasks_dir, tid)
        if tdir is None:
            logger.warning(
                "lite.cuaworld: skipping task %s/%s (unsafe task directory)",
                upstream_env,
                tid,
            )
            continue
        tj = tdir / "task.json"
        # Upstream task.json files are occasionally malformed (raw newlines in a
        # JSON string, missing `success.spec.program`, or a null `init`/`metadata`),
        # or a listed task may be absent. Parse AND extract every field inside one
        # guard so a single bad task is skipped rather than crashing the whole env's
        # registration (which would take down every `lite.cuaworld.*` import).
        # The lite.cuaworld.base home is /home/ga, so upstream paths stay as-is.
        try:
            spec = json.loads(tj.read_text())
            instruction = (
                spec.get("description")
                or (spec.get("natural_language") or {}).get("prompt", "")
            )
            verifier_target = spec["success"]["spec"]["program"]
            if "::" in verifier_target:
                verifier_file, verifier_func = verifier_target.split("::", 1)
            else:
                verifier_file, verifier_func = "verifier.py", verifier_target
            verifier_path = Path(verifier_file)
            if (
                not verifier_target
                or not verifier_file
                or not verifier_func
                or verifier_path.is_absolute()
                or ".." in verifier_path.parts
            ):
                raise ValueError("unsafe verifier target")
            task_meta = spec.get("metadata") or {}
            init = spec.get("init") or {}
            max_steps = int(init.get("max_steps", 2000))
            timeout_sec = float(init.get("timeout_sec", 600))
            if max_steps <= 0:
                raise ValueError("max_steps must be positive")
            if not math.isfinite(timeout_sec) or timeout_sec <= 0:
                raise ValueError("timeout_sec must be positive and finite")
            reward_type = init.get("reward_type", "sparse")
        except (OSError, json.JSONDecodeError, UnicodeDecodeError,
                KeyError, TypeError, AttributeError, ValueError) as e:
            logger.warning("lite.cuaworld: skipping task %s/%s (bad task.json: %s)",
                           upstream_env, tid, type(e).__name__)
            continue
        configs.append(
            SandboxTaskConfig(
                task_id=tid,
                instruction=instruction,
                computer=computer_config,
                max_steps=max_steps,
                setup_fn=_make_setup_fn(tdir, spec),
                evaluate_step_fn=_make_step_eval_fn(),
                evaluate_final_fn=_make_eval_fn(
                    tdir,
                    verifier_target,
                    task_meta,
                    spec,
                    env_id,
                    upstream_env_id,
                ),
                metadata={
                    "timeout_sec": timeout_sec,
                    "others": {
                        "source": f"gym-anything/{upstream_env}",
                        "upstream_task_id": spec.get("id", tid),
                        "upstream_env_id": upstream_env_id,
                        "reward_type": reward_type,
                        # empirical validation flagged this task as broken/gameable →
                        # rollout drops it via the standard exclude_reason filter.
                        **({"exclude_reason": _excl[tid]} if tid in _excl else {}),
                    }
                },
            )
        )
    return configs


def register_cuaworld_software(
    software: str,
    upstream_env: str,
    *,
    memory: str = _DEFAULT_MEMORY,
    cpu: str = _DEFAULT_CPU,
    gpus: int = 0,
    timeout: int = _DEFAULT_TIMEOUT,
    build_cmd: str | None = None,
) -> None:
    """Register ``lite.cuaworld.<software>`` tasks, baked-shape metadata, and
    env-server services.

    Splits come from the materials' ``registered.json`` (e.g. ``eval`` =
    CUAWorld-Test, ``train``, any ``additional_splits``) — ALL are registered;
    testing just samples a few via ``--head``. ``gpus>0`` requests GPU passthrough.
    ``memory`` is a FLOOR-checked request: the
    materials' ``env.json`` ``resources.mem_gb`` wins whenever it is larger (see
    :func:`_resolve_memory`).
    """
    image_spec.register_software(software, upstream_env)
    env_id = f"lite.cuaworld.{software}"
    image = f"cua-lite/lite.cuaworld.{software}:latest"
    cache_root = _materials_dir() / software / upstream_env
    tasks_dir = cache_root / "tasks"
    # Standard lifecycle entry: the software name selects its fixed materials
    # subtree; callers cannot override that mapping.
    install_cmd = build_cmd or (
        f"uv run --no-sync bash "
        f"lite/gym/envs/lite/cuaworld/scripts/install.sh build {software}"
    )

    computer_config: dict[str, Any] = {
        "image": image,
        "display": _DISPLAY,
        # env.json's declared requirement is a FLOOR on the caller's request — see
        # _resolve_memory for why, and for the host-headroom arithmetic.
        "memory": _resolve_memory(software, cache_root, memory),
        "cpu": cpu,
        "timeout": timeout,
    }
    if gpus:
        computer_config["gpus"] = gpus

    env_class = _make_env_class(env_id, computer_config)
    catalog_registered = False

    def register_catalog() -> None:
        nonlocal catalog_registered
        if catalog_registered:
            return
        if not tasks_dir.is_dir() or not (cache_root / ".complete").is_file():
            raise EnvDepsMissingError(
                what=(
                    f"{env_id} task materials are missing or incomplete. "
                    "Install or pull the environment first."
                ),
                install=install_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            )
        try:
            expected_identity = _materials_identity(upstream_env)
        except (OSError, ValueError) as exc:
            raise EnvDepsMissingError(
                what=f"{env_id} materials source is unavailable or invalid: {exc}",
                install=install_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            ) from exc
        identity_stamp = cache_root / ".materials_revision"
        try:
            actual_identity = (
                identity_stamp.read_text().strip() if identity_stamp.is_file() else ""
            )
        except OSError as exc:
            raise EnvDepsMissingError(
                what=f"{env_id} could not read its materials revision: {exc}",
                install=install_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            ) from exc
        if actual_identity != expected_identity:
            raise EnvDepsMissingError(
                what=(
                    f"{env_id} task materials are stale or from a different asset lock "
                    "in its runtime materials cache."
                ),
                install=install_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            )

        reg = cache_root / "registered.json"
        if not reg.is_file():
            raise EnvDepsMissingError(
                what=f"{env_id} materials are missing registered.json",
                install=install_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            )

        digest_stamp = cache_root / ".materials_digest"
        try:
            expected_digest = image_spec.cache_materials_digest(cache_root)
            actual_digest = (
                digest_stamp.read_text().strip() if digest_stamp.is_file() else ""
            )
        except (OSError, ValueError) as exc:
            raise EnvDepsMissingError(
                what=(
                    f"{env_id} could not validate its materials digest "
                    f"({type(exc).__name__})"
                ),
                install=install_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            ) from exc
        if actual_digest != expected_digest:
            raise EnvDepsMissingError(
                what=(
                    f"{env_id} task materials content digest is stale or invalid "
                    "in its runtime materials cache."
                ),
                install=install_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            )

        # All top-level list splits plus nested additional_splits.
        try:
            raw = json.loads(reg.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EnvDepsMissingError(
                what=f"{env_id} has an invalid registered.json: {exc}",
                install=install_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            ) from exc
        if not isinstance(raw, dict):
            raise EnvDepsMissingError(
                what=f"{env_id} registered.json must contain an object",
                install=install_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            )
        splits: dict[str, list[str]] = {}
        for split, task_ids in raw.items():
            if split == "additional_splits" or not isinstance(task_ids, list):
                continue
            if not isinstance(split, str) or not all(
                isinstance(task_id, str) for task_id in task_ids
            ):
                raise EnvDepsMissingError(
                    what=f"{env_id} registered.json has an invalid {split!r} split",
                    install=install_cmd,
                    see="lite/gym/envs/lite/cuaworld/README.md",
                )
            splits[split] = task_ids
        extra = raw.get("additional_splits")
        if extra is not None and not isinstance(extra, dict):
            raise EnvDepsMissingError(
                what=f"{env_id} registered.json additional_splits must be an object",
                install=install_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            )
        if isinstance(extra, dict):
            for k, v in extra.items():
                if not isinstance(k, str) or not isinstance(v, list) or not all(
                    isinstance(task_id, str) for task_id in v
                ):
                    raise EnvDepsMissingError(
                        what=(
                            f"{env_id} registered.json has an invalid "
                            f"additional split {k!r}"
                        ),
                        install=install_cmd,
                        see="lite/gym/envs/lite/cuaworld/README.md",
                    )
                splits.setdefault(k, v)
        upstream_env_id = upstream_env
        env_spec_path = cache_root / "env.json"
        if env_spec_path.is_file():
            try:
                env_spec = json.loads(env_spec_path.read_text())
                candidate_env_id = env_spec["id"]
                if not isinstance(candidate_env_id, str) or not candidate_env_id:
                    raise ValueError("id must be a non-empty string")
                upstream_env_id = candidate_env_id
            except (
                OSError,
                json.JSONDecodeError,
                UnicodeDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise EnvDepsMissingError(
                    what=f"{env_id} has an invalid env.json: {exc}",
                    install=install_cmd,
                    see="lite/gym/envs/lite/cuaworld/README.md",
                ) from exc
        configs = {
            split: _build_configs(
                tasks_dir,
                ids,
                computer_config,
                env_id,
                upstream_env,
                upstream_env_id,
            )
            for split, ids in splits.items()
        }
        if not any(configs.values()):
            raise EnvDepsMissingError(
                what=f"{env_id} materials contain no valid registered tasks",
                install=install_cmd,
                see="lite/gym/envs/lite/cuaworld/README.md",
            )
        register_tasks(
            env_id,
            configs,
            env_class=env_class,
            factory_kwargs={"post_action_delay": _DEFAULT_POST_ACTION_DELAY},
        )
        catalog_registered = True

    # Env-server services: image check + container reaping.
    register_services(
        env_id,
        CUAWorldServices(image, install_cmd, register_catalog),
    )
    # Backend family: one fresh container per trajectory, with DEDICATED
    # lifecycle/reconcile behavior like lite.osworld.
    register_family(env_id, BackendFamily.DEDICATED)
    registry.declare_env_id(env_id)
    registry.set_env_make_kwargs(env_id, MAKE_KWARGS)
    # Preserve eager registration when materials already exist, while keeping the
    # service/family discoverable before installation. If materials land later in
    # the same process, EnvServices.register_tasks retries on the next catalog probe.
    if tasks_dir.is_dir() and (cache_root / ".complete").is_file():
        try:
            register_catalog()
        except EnvDepsMissingError as exc:
            logger.warning("lite.cuaworld: deferred catalog for %s: %s", env_id, exc.what)
