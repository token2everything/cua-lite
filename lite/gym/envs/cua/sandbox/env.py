"""cua.sandbox — open-ended instruction env over the Cua ``Sandbox`` SDK.

A **direct wrapper** (NOT registered / not ``gym.make``-able): drive a
Cua-managed sandbox (local QEMU/Docker by default, or ``local=False`` for Cua cloud) from a free-form
instruction, on any OS. Inference-only (``reward=None``); the episode ends on
the agent's ``terminate``/``response`` or ``max_steps``.

Usage (see /docs/examples/cua.md):

    import asyncio
    import lite.agents as agents
    from lite.gym.envs.cua.sandbox import CuaSandboxEnv

    env = CuaSandboxEnv(instruction="open Firefox and search today's weather")
    agent = agents.make("gpt-5.5", env=env)
    result = asyncio.run(agent.sample(env))

Deps: the ``cua-sandbox`` package is imported LAZILY (only ``reset()`` needs it), so
importing this module / constructing the env / reading ``metadata`` works
without ``cua`` installed. Install with
``uv run --no-sync bash lite/gym/envs/cua/scripts/install.sh``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LiteDesktopActionSet,
    LiteMobileActionSet,
)
from lite.core.tools.calls import EnvAction, RuntimeEnvAction
from lite.gym.base import LiteBaseEnv
from lite.gym.envs.cua.utils import INSTALL as _INSTALL
from lite.gym.envs.cua.utils import SEE as _SEE
from lite.gym.envs.cua.utils import is_terminal, to_px, unpack
from lite.gym.errors import EnvDepsMissingError, TrueInfraFailure
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.model_inputs import (
    coerce_model_duration,
    project_model_keys,
)
from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
    ToolErrorFeedback,
    append_feedback,
    record_model_action_error,
)
from lite.gym.utils.feedback.ingress import (
    prepare_env_tool_calls,
    standalone_tool_call_feedback,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)
from lite.gym.wrappers import overlay_cursor_px

# Construction defaults live in configs/default.yaml (cua-lite convention) — read via
# env_config.load, not hardcoded. ENV_DIR = this sandbox/ pkg (configs/ co-located).
# Override the whole file with $CUA_SANDBOX_CONFIG. (No width/height key: cua.sandbox
# resolution is image-fixed — see the yaml note.)
ENV_DIR = str(Path(__file__).resolve().parent)
CFG = env_config.load(ENV_DIR)
_DEFAULT_MAX_STEPS = CFG.env_kwargs["max_steps"]
_DEFAULT_EXTRA_TOOLS = CFG.env_kwargs.get("extra_tools", [])
_DEFAULT_PLATFORM = CFG.env_kwargs["platform"]
_DEFAULT_LOCAL = CFG.env_kwargs["local"]
_DEFAULT_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
# cua.sandbox is a direct env, so it reads its own capture knob and composites
# the cursor inside _screenshot using the env-tracked pixel pointer. The Cua
# screenshot carries no host cursor, so this is the only cursor the agent sees.
_DEFAULT_CURSOR = CFG.make_kwargs.get("cursor", True)

# The ONLY catalog actions ``_dispatch`` deliberately makes no Sandbox call
# for: read-only reads with no Cua counterpart (``screenshot`` is redundant —
# step() captures a frame after every action — and cua exposes no cursor read).
# Everything else in the platform's DERIVED vocabulary
# (``LiteDesktopActionSet`` / ``LiteMobileActionSet``'s ``get_action_names()``) must
# have a ladder branch, and a name outside that vocabulary means ingress leaked.
# Both are dispatcher bugs, so the ladders RAISE instead of falling through to a
# silent ``return False`` that the step loop would screenshot as if it had run
# (same derive-then-raise contract as
# ``lite/gym/utils/feedback/surface.py:resolve_valid_actions``).
_DESKTOP_NO_OP_ACTIONS = frozenset({"screenshot", "cursor_position"})
_MOBILE_NO_OP_ACTIONS = frozenset({"screenshot"})


def _make_image(platform: str, image_kwargs: dict[str, Any], Image: Any) -> Any:
    """platform → ``Image.<platform>()``. Linux defaults to the lightweight
    ``container`` (``Image.linux`` itself defaults to ``vm``); overridable via
    ``image_kwargs``."""
    if platform == "linux":
        return Image.linux(**{"kind": "container", **image_kwargs})
    if platform == "macos":
        return Image.macos(**image_kwargs)
    if platform == "windows":
        return Image.windows(**image_kwargs)
    if platform == "android":
        return Image.android(**image_kwargs)
    raise ValueError(f"unknown platform {platform!r} (linux|macos|windows|android)")


class CuaSandboxEnv(LiteBaseEnv):
    """Open-ended Cua sandbox driven by a free-form instruction (no task_id, no reward)."""

    def __init__(
        self,
        instruction: str,
        *,
        max_steps: int = _DEFAULT_MAX_STEPS,
        extra_tools: list[str] | None = _DEFAULT_EXTRA_TOOLS,
        platform: str = _DEFAULT_PLATFORM,   # linux | macos | windows | android
        local: bool = _DEFAULT_LOCAL,        # True → local QEMU/Docker (no key); False → Cua cloud (CUA_API_KEY)
        post_action_delay: float = _DEFAULT_POST_ACTION_DELAY,   # asyncio.sleep before each post-step screenshot
        name: str | None = None,
        image_kwargs: dict[str, Any] | None = None,     # → Image.<platform>()  (e.g. {"kind": "vm"})
        sandbox_kwargs: dict[str, Any] | None = None,   # → Sandbox.ephemeral() (runtime/region/cpu/...)
        cursor: bool = _DEFAULT_CURSOR,                  # composite the tracked cursor onto each screenshot
        seed: int | None = None,                        # framework kwarg; unused (inference-only, nothing to seed)
    ) -> None:
        # `seed` accepted EXPLICITLY (no **kwargs) to mirror cua.bench + browsergym:
        # a typo'd env_kwarg raises rather than silently no-op'ing. (This env isn't
        # registered/factory-constructed, so seed is purely defensive here.)
        _ = seed
        self._instruction = instruction
        self._max_steps = max_steps
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        self._step_count = 0
        self._platform = platform
        self._local = local
        self._post_action_delay = post_action_delay
        self._name = name
        self._image_kwargs = dict(image_kwargs or {})
        self._sandbox_kwargs = dict(sandbox_kwargs or {})
        self._cursor = cursor
        self._cm: Any = None   # the ephemeral() async context manager (held for teardown)
        self._sb: Any = None   # the live Sandbox
        self._wh: tuple[int, int] | None = None   # (W, H) for [0,1000]→px
        self._cursor_px: tuple[int, int] = (0, 0)    # last cursor px (for mouse_down/up)
        # Placeholder until reset() PARKS the real pointer. (0, 0) is the "never
        # moved" sentinel, not an observed position — gates the composite so it
        # is never painted.
        self._cursor_px_known = False

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # Pure function of ctor args (reset() not required). Platform selects the
        # agent's action space (agents/factory.py): android → mobile, else desktop.
        # No ``_task_metadata`` builder: this env is never registered, so there
        # is no registered copy to same-source — the family's only
        # documented exemption.
        plat = (
            LiteCUAMetadata.Platform.MOBILE if self._platform == "android"
            else LiteCUAMetadata.Platform.DESKTOP
        )
        return LiteCUAMetadata(
            dims=(plat.value, LiteCUAMetadata.TaskType.USE.value),
            extra_tool_schemas=list(self._extra_tool_schemas),
        )

    async def _screenshot(self) -> bytes:
        """Grab the sandbox screenshot, compositing the tracked cursor at its exact
        pixel position when ``cursor`` is on (Cua frames carry no host
        cursor). Runs the decode/paste/re-encode off the event loop.

        INVARIANT: the composited coordinate is the real pointer position, or
        nothing is composited. ``_cursor_px`` only ever holds a coordinate we
        moved the pointer to, and ``_cursor_px_known`` is False until reset's
        parking ``mouse.move`` lands — and stays False forever on android, which
        is a touch surface with no pointer to draw.
        """
        png = await self._sb.screenshot()
        if self._cursor and self._cursor_px_known:
            png = await asyncio.to_thread(
                overlay_cursor_px, png, self._cursor_px[0], self._cursor_px[1]
            )
        return png

    async def reset(self) -> LiteEnvObservation:
        try:
            from cua_sandbox import (  # the SDK package; NOT the `cua` meta-pkg (drags in cua-agent)
                Image,
                Sandbox,
            )
        except ImportError as e:
            raise EnvDepsMissingError(
                what="`cua-sandbox` (Sandbox SDK) not installed", install=_INSTALL, see=_SEE
            ) from e
        image = _make_image(self._platform, self._image_kwargs, Image)
        # .ephemeral() auto-provisions a sandbox (cloud or local per self._local);
        # hold the context manager so close() can tear it down.
        self._cm = Sandbox.ephemeral(image, local=self._local, name=self._name, **self._sandbox_kwargs)
        self._sb = await self._cm.__aenter__()
        self._wh = await self._sb.get_dimensions()   # (W, H), absolute pixels
        self._cursor_px = (self._wh[0] // 2, self._wh[1] // 2)   # cua mouse_down/up need x,y; Lite's don't carry one
        # Make the centre a FACT, not an assumption. The cua-sandbox mouse
        # interface is write-only (click / move / scroll / down / up / drag — no
        # position read; env.py's `cursor_position` action is a documented
        # no-op), so the pointer cannot be QUERIED — but it can be ESTABLISHED:
        # move it, then screenshot, so capture and cursor share one instant.
        # Android is a touch surface with no pointer, so nothing is parked and
        # nothing is ever composited there.
        if self._cursor and self._platform != "android":
            await self._sb.mouse.move(*self._cursor_px)
            self._cursor_px_known = True
        self._step_count = 0
        png = await self._screenshot()               # PNG bytes (+ cursor if enabled)
        # observation.text on reset == the instruction (agent reads it as the goal).
        return LiteEnvObservation(image=png, text=self._instruction)

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        metadata = self.metadata
        actions, ingress_errors = prepare_env_tool_calls(
            actions,
            metadata,
            validate_top_level_action=True,
        )
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        known_standalone_tools = type(self).known_standalone_tool_names()
        terminated = False
        # Model-emitted calls that ENDED the episode. They get no continuation
        # observation: devs/migration/verify.py forbids a tool result for a
        # terminal call. Keyed on the env-internal projected ``action["call_id"]``
        # and NOT on ``result_call_id``, because an INTERNAL finish call has no
        # model-call id and the loop-detect wrapper's injected ``terminate``
        # carries the intercepted NON-finish model call's id as
        # ``_result_call_id`` -- that call must still be answered.
        terminal_call_ids: set[str] = set()
        # One frame PER EXECUTED ACTION, in action order (the contract the agent
        # layer consumes).
        step_screenshots: list[bytes] = []
        for a, result_call_id in actions:
            tool_feedback = standalone_tool_call_feedback(
                a, known_standalone_tools, metadata.extra_tool_schemas,
            )
            if tool_feedback is not None:
                if result_call_id:
                    append_feedback(action_errors, result_call_id, tool_feedback)
                # The shared layer already decided the SURFACE: ``current`` is a
                # GUI slot the model got wrong, which owes a frame even though
                # nothing ran (R2a + R3); ``error_only`` is a text tool, which
                # owes none. Read that decision rather than re-deriving it. The
                # check lives in THIS loop so the frame lands in slot order.
                if tool_feedback.carrier == "current":
                    step_screenshots.append(await self._screenshot())
                continue
            try:
                dispatched_terminal = await self._dispatch(a)
            except MODEL_ACTION_ERROR_TYPES as e:
                record_model_action_error(
                    action_errors, result_call_id, e, action_name=a["name"]
                )
                # POLICY: a MODEL fault costs this action and nothing more. It
                # never reached the screen, so the state the tail was chosen
                # against is exactly the state still on it -- there is nothing to
                # protect the tail from. The model gets the reason as text and a
                # frame for this slot, then the batch continues. Contrast the
                # interface-failure arm: THERE the action may have half executed
                # and the screen is unknown, which is a different fault.
                step_screenshots.append(await self._screenshot())
                continue
            except Exception as e:
                if isinstance(e, TrueInfraFailure):
                    raise
                raise TrueInfraFailure(
                    f"cua.sandbox {a['name']} backend failed: {e}"
                ) from e
            if dispatched_terminal:
                terminated = True
                if a.get("call_id"):
                    terminal_call_ids.add(a["call_id"])
                break
            # Settle before capturing — desktop apps repaint slowly (else the
            # frame is stale) — then record this action's own frame.
            if self._post_action_delay > 0:
                await asyncio.sleep(self._post_action_delay)
            step_screenshots.append(await self._screenshot())
        self._step_count += 1
        truncated = not terminated and self._step_count >= self._max_steps
        if not step_screenshots:
            # Nothing ran (empty batch, a terminal-only turn, or every call
            # rejected). The turn still owes the model one current observation,
            # and nothing moved, so no settle delay is owed.
            step_screenshots.append(await self._screenshot())
        # Inference env: reward stays None; episode ends on terminate/response or max_steps.
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=None, terminated=terminated, truncated=truncated,
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
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)   # ephemeral → destroy
            self._cm = self._sb = None

    async def _dispatch(self, a: EnvAction) -> bool:
        """Translate one Lite action → Cua ``Sandbox`` calls (desktop or mobile).

        Returns True iff the action ends the episode (terminate/response)."""
        name, args = unpack(a)
        if is_terminal(name):
            return True

        def px(coord: list[int]) -> tuple[int, int]:
            return to_px(coord, self._wh)

        # android → mobile action space (LiteMobileActionSpace → sb.mobile.*)
        if self._platform == "android":
            return await self._dispatch_mobile(name, args, px)

        m, k = self._sb.mouse, self._sb.keyboard
        if name == "click":
            x, y = self._cursor_px = px(args["coordinate"])
            if args.get("clicks") == 2:
                await m.double_click(x, y)
            elif args.get("button") == "right":
                await m.right_click(x, y)
            else:
                await m.click(x, y, args.get("button", "left"))
        elif name == "type":
            await k.type(args["text"])
        elif name == "key":
            # backend="pynput" projects canonical tokens → cua computer-server
            # spelling, like every other lite env does (else pageup/meta fail).
            await k.keypress(project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="pynput",
            ))
        elif name == "key_down":                      # Lite gives a list; cua takes one key at a time
            for key in project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="pynput",
            ):
                await k.key_down(key)
        elif name == "key_up":
            for key in reversed(project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="pynput",
            )):
                await k.key_up(key)
        elif name == "hold_key":                      # press all, hold, release in reverse
            proj = project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="pynput",
            )
            duration = coerce_model_duration(
                args.get("duration", 1),
                action_name="hold_key",
            )
            for key in proj:
                await k.key_down(key)
            await asyncio.sleep(duration)
            for key in reversed(proj):
                await k.key_up(key)
        elif name == "mouse_move":
            self._cursor_px = px(args["coordinate"])
            await m.move(*self._cursor_px)
        elif name == "mouse_down":                    # Lite carries no coordinate → act at current cursor
            await m.mouse_down(*self._cursor_px, args.get("button", "left"))
        elif name == "mouse_up":
            await m.mouse_up(*self._cursor_px, args.get("button", "left"))
        elif name == "drag":                          # start_coordinate optional → drag from current cursor
            end = args.get("coordinate")
            if end:
                start = px(args["start_coordinate"]) if args.get("start_coordinate") else self._cursor_px
                end = px(end)
                self._cursor_px = end
                await m.drag(*start, *end)
        elif name == "scroll":
            amt = int(args.get("amount", 3))
            d = args.get("direction", "down")
            sx = amt if d == "right" else (-amt if d == "left" else 0)
            sy = amt if d == "up" else (-amt if d == "down" else 0)
            coord = args.get("coordinate")
            scroll_px = px(coord if coord is not None else [500, 500])
            if coord is not None:
                self._cursor_px = scroll_px
            await m.scroll(*scroll_px, scroll_x=sx, scroll_y=sy)
        elif name == "wait":
            duration = coerce_model_duration(
                args.get("duration", 1),
                action_name="wait",
            )
            await asyncio.sleep(duration)
        elif name not in _DESKTOP_NO_OP_ACTIONS:
            # No silent fallthrough: ingress (prepare_env_tool_calls with
            # validate_top_level_action=True) admits only this platform's catalog
            # actions, so an unhandled name here is a dispatcher bug, not a
            # model error — raise rather than screenshot an unchanged desktop.
            raise TrueInfraFailure(
                f"cua.sandbox desktop dispatch has no branch for {name!r}; "
                f"this surface admits {sorted(LiteDesktopActionSet.get_action_names())}"
            )
        return False

    async def _dispatch_mobile(self, name: str, args: dict, px) -> bool:
        """Translate one LiteMobileActionSpace action → Cua ``sb.mobile.*`` calls.

        Android has no pointer to park at reset, so the marker only becomes
        drawable once a touch has actually put a finger somewhere — from then on
        ``_cursor_px`` is a real contact point, not a guess.
        """
        mob = self._sb.mobile
        if args.get("coordinate") is not None:
            self._cursor_px_known = True
        if name == "tap":
            x, y = self._cursor_px = px(args["coordinate"])
            await (mob.double_tap(x, y) if args.get("clicks") == 2 else mob.tap(x, y))
        elif name == "long_press":
            raw_duration = args.get("duration")
            duration = coerce_model_duration(
                1.0 if raw_duration is None else raw_duration,
                action_name="long_press",
            )
            x, y = self._cursor_px = px(args["coordinate"])
            await mob.long_press(x, y, duration_ms=int(duration * 1000))
        elif name == "type":
            await mob.type_text(args["text"])
        elif name in ("swipe", "drag"):   # cua mobile has no drag; swipe covers both
            start = px(args["start_coordinate"])
            end = px(args["coordinate"])
            self._cursor_px = end
            await mob.swipe(*start, *end)
        elif name == "pinch":
            cx, cy = self._cursor_px = px(args["coordinate"])
            spread = int(args.get("amount", 25) / 100 * min(self._wh))   # Lite amount is %-of-screen (default 25) → px spread
            await (mob.pinch_in(cx, cy, spread) if args.get("direction") == "in" else mob.pinch_out(cx, cy, spread))
        elif name == "system_button":
            b = args.get("button", "Home")
            if b == "Home":
                await mob.home()
            elif b == "Back":
                await mob.back()
            elif b == "Recent":
                await mob.recents()
            elif b == "Enter":
                await mob.key(66)    # Android KEYCODE_ENTER
            elif b == "Menu":
                await mob.key(82)    # Android KEYCODE_MENU
        elif name == "wait":
            duration = coerce_model_duration(
                args.get("duration", 1),
                action_name="wait",
            )
            await asyncio.sleep(duration)
        elif name not in _MOBILE_NO_OP_ACTIONS:
            # Same contract as the desktop ladder: no silent fallthrough.
            raise TrueInfraFailure(
                f"cua.sandbox mobile dispatch has no branch for {name!r}; "
                f"this surface admits {sorted(LiteMobileActionSet.get_action_names())}"
            )
        return False
