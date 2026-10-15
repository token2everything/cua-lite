"""
Environment wrappers for composable behavior.

EnvWrapper auto-delegates attribute access to the inner env via
``__getattribute__`` so that **any** env method passes through
without per-method forwarding — including methods that only exist on
:class:`EnvServerPoolable` subclasses (``bind`` and env-specific helpers). The
MRO walk stops at :class:`LiteBaseEnv` exclusive so the wrapper
class's overrides win over base defaults; subclasses override only
what they want to intercept.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image

from lite.core.messages.final import (
    LOOP_DETECT_TERMINATE_REASON,
    STOP_REASON_INFO_KEY,
)
from lite.core.metadata import LiteBaseMetadata
from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    validate_lite_action_batch_structure,
)
from lite.core.tools.calls import (
    RUNTIME_SIDECAR_KEYS,
    EnvAction,
    LiteToolCall,
    RuntimeEnvAction,
    make_tool_call,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
)
from lite.gym.base import LiteBaseEnv
from lite.gym.types import (
    LiteEnvObservation,
    LiteEnvStepResult,
)
from lite.gym.utils.feedback.ingress import (
    is_internal_finish_tool_call,
    make_internal_terminate_action,
)
from lite.gym.utils.feedback.results import (
    invalid_env_tool_call_envelope_message,
)

ASSETS_DIR = Path(__file__).parent / "assets"

logger = logging.getLogger(__name__)

class EnvWrapper(LiteBaseEnv, allow_metadata_override=True):
    """Delegation base class. Attribute access auto-forwarded to inner env.

    Why ``__getattribute__`` and not ``__getattr__``: ``__getattr__`` only
    fires when normal attribute lookup *fails*. Methods declared on
    :class:`LiteBaseEnv` are *found* via normal MRO lookup, so
    ``__getattr__`` would route those to the wrapper's own (possibly
    abstract) implementations instead of delegating to the inner env.
    ``__getattribute__`` runs on every access, letting us route around
    the inheritance cleanly.

    Routing rules per access:

      1. ``self.env`` and dunders bypass routing (no recursion).
      2. If THIS wrapper class or a subclass above EnvWrapper in MRO
         defines the attribute, use that override (e.g.
         ``StepTimeoutWrapper.step``).
      3. If the instance ``__dict__`` has it (instance attrs set in
         ``__init__``, e.g. ``StepTimeoutWrapper.step_timeout``), use that.
      4. Otherwise delegate to ``self.env`` — catches every method on
         the inner env (including EnvServerPoolable-only methods like
         ``bind``) without per-method forwarding.

    Subclasses just override what they want to intercept (``reset`` to
    composite, ``step`` to time-bound, ...); everything else passes
    through automatically.
    """

    def __init__(self, env: LiteBaseEnv):
        self.env = env

    def __getattribute__(self, name: str) -> Any:
        # 1. Internal access (avoid recursion).
        if name == "env" or name.startswith("__"):
            return object.__getattribute__(self, name)

        # 2. Subclass override (anywhere in MRO above LiteBaseEnv).
        # Walk from most-specific class down; stop EXCLUSIVELY at
        # :class:`LiteBaseEnv` so the wrapper's own overrides win
        # over the abstract base.
        for cls in type(self).__mro__:
            if cls is LiteBaseEnv:
                break
            if name in cls.__dict__:
                return object.__getattribute__(self, name)

        # 3. Instance attribute (set in subclass __init__).
        instance_dict = object.__getattribute__(self, "__dict__")
        if name in instance_dict:
            return instance_dict[name]

        # 4. Delegate to inner env. Catches every inherited method
        #    (including EnvServerPoolable-only members like ``bind``)
        #    without per-method forwarding.
        return getattr(object.__getattribute__(self, "env"), name)

    # NOTE: explicit @property / async-method definitions BELOW are
    # intentionally kept even though __getattribute__ would delegate
    # anyway. They satisfy LiteBaseEnv's abstract-method contract (so
    # ``isinstance`` + ``ABC``-style introspection work) AND serve as
    # documentation of the wrapper surface. Subclasses override these
    # to intercept; the rule-2 MRO walk finds the subclass version
    # before reaching EnvWrapper.

    @property
    def unwrapped(self) -> LiteBaseEnv:
        """Strip all wrapper layers down to the bare env."""
        return self.env.unwrapped

    @property
    def metadata(self) -> LiteBaseMetadata:
        return self.env.metadata

    def _runtime_metadata(self) -> LiteBaseMetadata:
        # Satisfies the base ABC; unreachable via the base ``metadata``
        # property (this class's own pass-through shadows it) — delegate
        # for consistency if ever called directly.
        return self.env.metadata

    async def reset(self) -> LiteEnvObservation:
        return await self.env.reset()

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        return await self.env.step(actions)

    async def close(self) -> None:
        await self.env.close()

class EnvTimeoutError(Exception):
    """Raised when env.reset() or env.step() exceeds its timeout."""

    def __init__(self, phase: str, timeout: float):
        self.phase = phase
        self.timeout = timeout
        super().__init__(f"{phase}() timed out after {timeout:.1f}s")


class StepTimeoutWrapper(EnvWrapper):
    """Raises EnvTimeoutError if reset or a single step exceeds a timeout."""

    def __init__(
        self,
        env: LiteBaseEnv,
        step_timeout: float = 120.0,
        reset_timeout: float = 600.0,
    ):
        super().__init__(env)
        self.step_timeout = step_timeout
        self.reset_timeout = reset_timeout

    async def reset(self) -> LiteEnvObservation:
        try:
            return await asyncio.wait_for(self.env.reset(), timeout=self.reset_timeout)
        except TimeoutError:
            raise EnvTimeoutError("reset", self.reset_timeout)

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        try:
            return await asyncio.wait_for(self.env.step(actions), timeout=self.step_timeout)
        except TimeoutError:
            raise EnvTimeoutError("step", self.step_timeout)


def _annotate_loop_trigger(
    result: LiteEnvStepResult,
    fingerprint: str,
    period_length: int,
) -> None:
    """Stamp the loop-detect observability keys onto a step result.

    One writer, one key spelling. The key itself is owned by
    ``lite.core.messages.final`` — every layer that reports why an episode
    ended writes it, so the wrapper's value simply lands last and wins.
    """
    result.info[STOP_REASON_INFO_KEY] = LOOP_DETECT_TERMINATE_REASON
    result.info["loop_action"] = fingerprint
    result.info["loop_period_length"] = period_length


_CURSOR_SPRITE_CACHE: dict[int, Image.Image] = {}


def load_cursor_sprite(height: int = 24) -> Image.Image:
    """Cached load + LANCZOS-resize of the shared 40×60 cursor asset to ``height``px
    (straight alpha, tip at top-left). Env screenshot/capture code uses this
    helper when it knows its native screenshot lacks a visible cursor."""
    sprite = _CURSOR_SPRITE_CACHE.get(height)
    if sprite is None:
        raw = Image.open(ASSETS_DIR / "cursor.png").convert("RGBA")
        aspect = raw.width / raw.height
        sprite = raw.resize((int(height * aspect), height), Image.LANCZOS)
        _CURSOR_SPRITE_CACHE[height] = sprite
    return sprite


def overlay_cursor_px(png: bytes, x_px: int, y_px: int, *, height: int = 24) -> bytes:
    """Composite the cursor sprite onto a PNG at absolute pixel ``(x_px, y_px)`` and
    return re-encoded PNG bytes. For envs that already track the visible pointer
    in pixels. Straight alpha; PIL paste clips gracefully when the tip sits at the
    right/bottom edge."""
    sprite = load_cursor_sprite(height)
    raw = Image.open(io.BytesIO(png))
    img = raw.convert("RGBA")
    # Clamp the tip into the frame so an edge coordinate still shows the cursor
    # instead of pasting it one pixel off-frame.
    w, h = img.size
    x_px = max(0, min(x_px, w - 1))
    y_px = max(0, min(y_px, h - 1))
    img.paste(sprite, (x_px, y_px), sprite)
    img = img.convert(raw.mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class LoopDetectWrapper(EnvWrapper):
    """Replace repeated assistant actions with an internal ``terminate``.

    Detection is action-only: trigger when the trailing fingerprints contain N
    repetitions of a period-L block, for some L in ``1..max_period``. On trigger,
    the action that closed the loop is replaced with
    ``terminate(status=terminal_status, reason="REPETITIVE_LOOP")`` and later
    actions in the same batch are dropped. The env owns scoring; this wrapper
    only chooses the injected terminal call shape and annotates ``result.info``.

    ``terminal_status`` is explicit scoring policy. The default ``"success"``
    preserves current rollout behavior, while env-specific callers can pass a
    different status without changing loop detection.

    Fingerprint: ``f"{name}|{json.dumps(args, sort_keys=True)}"`` over canonical
    Lite tool-call objects after dropping runtime-only fields such as
    ``id``. ``click(100,100)`` ≠ ``click(101,100)``. Runtime-only finish
    calls are not fingerprinted, because they are framework control flow rather
    than assistant actions.

    Limitation: the fingerprint is action-only, never observation-aware, so
    repeated navigation on a changing screen can still match. The registry
    default stays off; opt in with ``--env-kwargs '{"loop_detect": N}'``.

    Granularity: canonical Lite tool calls, with compatibility expansion for
    valid ``computer`` / ``mobile`` batches. A loop wholly inside one ``step()``
    call also triggers; any pre-trigger actions in the same batch are
    forwarded as one truncated canonical batch before the runtime ``terminate``
    call. The wrapper deliberately does not read env metadata or decide whether
    a name is active GUI, inactive GUI, or an extra tool; concrete env ingress
    owns that routing and feedback.

    Performance: ``_detect_loop`` is O(max_period² × loop_threshold) per
    action — for the default (5, 5) that's ~75 fingerprint comparisons,
    negligible vs. any env step. Buffer is bounded at
    ``max_period × loop_threshold`` (default 25) so memory is fixed.

    Exception safety: ``env.step()`` may raise (timeout, network). The
    wrapper snapshots the deque before the inner step and restores on
    any raised exception, so a failed step doesn't pollute the
    fingerprint history with actions the env never observed.

    Observability:
      * ``result.info["stop_reason"] = "REPETITIVE_LOOP"`` — primary signal.
      * ``result.info["loop_action"]`` — fingerprint of the action that
        closed the loop (the one being replaced with terminate).
      * ``result.info["loop_period_length"]`` — L, the detected period
        (1 = pure repeat, 2 = ABAB ping-pong, etc.).
    Lets the rollout caller distinguish wrapper-triggered terminate from
    agent-emitted terminate in metrics / dataset filters without touching
    the env protocol.
    """

    def __init__(
        self,
        env: LiteBaseEnv,
        loop_threshold: int,
        max_period: int | None = None,
        terminal_status: str = "success",
    ):
        if loop_threshold < 2:
            raise ValueError(
                f"loop_threshold must be >= 2; got {loop_threshold}"
            )
        if max_period is None:
            max_period = loop_threshold
        if max_period < 1:
            raise ValueError(f"max_period must be >= 1; got {max_period}")
        super().__init__(env)
        self._loop_threshold = loop_threshold
        self._max_period = max_period
        self._terminal_status = terminal_status
        # Buffer fits the largest detectable pattern (max_period × threshold).
        # Older fingerprints fall off the back — they can't be part of the
        # current trailing-repetition window anyway.
        self._recent_fps: deque[str] = deque(maxlen=max_period * loop_threshold)

    @staticmethod
    def _fingerprint(action: EnvAction) -> str:
        name = action["name"]
        args = action["arguments"]
        # Fingerprints only need to be stable and comparable, so non-JSON
        # argument values fall back to their string form instead of crashing.
        return (
            f"{name}|"
            f"{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
        )

    @staticmethod
    def _fingerprint_actions(action: LiteToolCall) -> list[EnvAction]:
        name = tool_call_name(action)
        arguments = tool_call_arguments(action)
        if name not in LITE_ACTION_BATCH_TOOL_NAMES:
            return [{
                "name": name,
                "arguments": {k: v for k, v in arguments.items() if v is not None},
            }]
        child_actions, batch_error = validate_lite_action_batch_structure(name, arguments)
        if batch_error is not None:
            return []
        return child_actions

    @staticmethod
    def _make_terminate_action(
        result_call_id: str | None = None,
        *,
        status: str = "success",
    ) -> RuntimeEnvAction:
        return make_internal_terminate_action(
            status=status,
            reason=LOOP_DETECT_TERMINATE_REASON,
            result_call_id=result_call_id,
        )

    def _detect_loop(self) -> int | None:
        """Return the period length L if the buffer's tail is N consecutive
        repetitions of an L-length block (for some 1 ≤ L ≤ max_period);
        otherwise None.

        Periods are tried in ascending order, so L=1 (single-action stuck
        loop) takes priority over L=2 (ping-pong) when both could match —
        matches the original single-action wrapper's semantics on AAAAA.
        """
        n = self._loop_threshold
        # deque doesn't slice; materialize once. Bounded by maxlen so cheap.
        buf = list(self._recent_fps)
        nb = len(buf)
        for L in range(1, self._max_period + 1):
            needed = L * n
            if nb < needed:
                # Larger L need even more — none of them can match either.
                break
            tail_start = nb - needed
            block = buf[tail_start : tail_start + L]
            matched = True
            for rep in range(1, n):
                rep_start = tail_start + rep * L
                if buf[rep_start : rep_start + L] != block:
                    matched = False
                    break
            if matched:
                return L
        return None

    async def reset(self) -> LiteEnvObservation:
        self._recent_fps.clear()
        return await self.env.reset()

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        # Snapshot before mutating: if env.step raises (timeout, network,
        # cancellation), restore so the deque only reflects actions the
        # env actually observed. Otherwise a retried step would see stale
        # fingerprints from a never-executed batch.
        snapshot = list(self._recent_fps)
        forwarded: list[RuntimeEnvAction] = []
        triggered_fp: str | None = None
        triggered_L: int | None = None
        for action in actions:
            # A wrapper sits above the env, so nothing here can be paired to a
            # call: a raise kills the rollout. Forward bad envelopes or reserved
            # runtime keys and let env ingress own feedback/raise semantics.
            if invalid_env_tool_call_envelope_message(action):
                forwarded.append(action)
                continue
            # A dict by now: the envelope check above rejects everything else.
            if RUNTIME_SIDECAR_KEYS & action.keys():
                forwarded.append(action)
                continue
            if is_internal_finish_tool_call(action):
                forwarded.append(action)
                continue
            child_actions = self._fingerprint_actions(action)
            prefix_len = 0
            for child_action in child_actions:
                fp = self._fingerprint(child_action)
                self._recent_fps.append(fp)
                L = self._detect_loop()
                if L is not None:
                    result_call_id = tool_call_id(action)
                    if len(child_actions) > 1 and prefix_len > 0:
                        prefix_action = make_tool_call(
                            tool_call_name(action),
                            {
                                **tool_call_arguments(action),
                                "actions": tool_call_arguments(action)["actions"][:prefix_len],
                            },
                        )
                        forwarded.append(prefix_action)
                    forwarded.append(
                        self._make_terminate_action(
                            result_call_id,
                            status=self._terminal_status,
                        )
                    )
                    triggered_fp = fp
                    triggered_L = L
                    break
                prefix_len += 1
            if triggered_fp is not None:
                break
            forwarded.append(action)
        try:
            result = await self.env.step(forwarded)
        except BaseException:
            # BaseException covers asyncio.CancelledError and
            # KeyboardInterrupt too — we want the deque restored before
            # re-raising in every case.
            self._recent_fps.clear()
            self._recent_fps.extend(snapshot)
            raise
        if triggered_fp is not None:
            _annotate_loop_trigger(result, triggered_fp, triggered_L)
        return result
