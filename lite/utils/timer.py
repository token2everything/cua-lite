"""Lightweight timer for profiling async pipelines.

Usage::

    from lite.utils.timer import Timer

    t = Timer()
    do_something()
    t.mark("step_a")
    do_more()
    t.mark("step_b")
    logger.info("perf: %s", t)
    # => "step_a=0.12 step_b=0.34 total=0.46"
"""

from __future__ import annotations

import time
from typing import Awaitable, TypeVar

_T = TypeVar("_T")


async def timed(coro: Awaitable[_T]) -> tuple[_T, float]:
    """Await ``coro`` and return ``(result, elapsed_seconds)`` (monotonic).

    Times exactly the awaited operation: the elapsed reading is taken right after
    the await resolves, so for the SEQUENTIAL awaits inside one agent turn (LLM
    call, then env.step) each measures that phase's real wall duration. Other
    coroutines running while this one is suspended are NOT mis-attributed — the
    await's wall span is the I/O time. (Under a saturated loop a small
    resume-scheduling delay folds in; that's a real cost, not a bug.)
    """
    t = time.perf_counter()
    result = await coro
    return result, time.perf_counter() - t


class Timer:
    """Records elapsed time between successive ``mark()`` calls."""

    __slots__ = ("_start", "_last", "_marks")

    def __init__(self):
        now = time.monotonic()
        self._start = now
        self._last = now
        self._marks: list[tuple[str, float]] = []

    def mark(self, name: str) -> float:
        """Record elapsed time since last mark (or creation). Returns the interval."""
        now = time.monotonic()
        dt = now - self._last
        self._marks.append((name, dt))
        self._last = now
        return dt

    @property
    def total(self) -> float:
        return time.monotonic() - self._start

    def __str__(self) -> str:
        parts = [f"{k}={v:.2f}" for k, v in self._marks]
        parts.append(f"total={self.total:.2f}")
        return " ".join(parts)
