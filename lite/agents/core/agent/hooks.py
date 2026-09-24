"""Sample hook protocol and step data.

Usage:
    from lite.agents.core.agent.hooks import SampleHook, SampleStepData

    class MyHook(SampleHook):
        def on_step(self, data: SampleStepData) -> None:
            print(f"step {data.step_idx}: reward={data.step_result.reward}")

    lite_rl_sample = await agent.sample(env, hooks=[MyHook()])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    # Avoid a runtime import cycle: ``base.agent`` imports ``SampleHook`` /
    # ``SampleStepData`` from this module, and we in turn reference types
    # that live in ``base.agent``. ``from __future__ import annotations``
    # above makes all annotations lazy strings, so the runtime import is
    # unnecessary.
    from lite.agents.types import PredictResult
    from lite.core import (
        LiteRLSample,
        LiteSample,
    )
    from lite.gym.types import LiteEnvStepResult


@dataclass
class SampleStepData:
    """Per-step data passed to ``SampleHook.on_step``."""
    step_idx: int
    image: Image.Image | None  # None when no image carrier is available
    predict_result: PredictResult
    step_result: LiteEnvStepResult
    # Actual actions passed to env.step(). For no-tool finals this is the
    # transient response/terminate action synthesized by the agent loop, not a
    # persisted assistant tool_call.
    actions: list[dict[str, Any]]
    current_image_index: int | None = None
    # Per-step phase durations in seconds, e.g. {"predict": <LLM-call s>, "act":
    # <env.step s>, "observe": <obs-build s>}. None when the agent loop does not
    # record timings. The logger persists this as turn_NNNN/05_timing.json and
    # aggregates it into summary.json.
    timings: dict[str, float] | None = None


class SampleHook:
    """Lifecycle hook for agent.sample().

    Subclass and override the methods you need. Default implementations are no-ops.
    """

    def on_step(self, data: SampleStepData) -> None:
        """Called after each env.step."""

    def on_trajectory(self, trajectory: LiteSample) -> None:
        """Observe cumulative messages after feedback is stored; do not mutate.

        The desktop Responses and adapter-backed loops publish each turn;
        on_complete remains the final sample hook.
        """

    def on_complete(self, lite_rl_sample: LiteRLSample | None) -> None:
        """Called after sampling finishes (or fails). ``lite_rl_sample`` is None on error."""
