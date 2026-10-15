"""Shared wall-clock readiness poller for env adapters.

Run: not directly. Imported by ``ensure_services`` boot paths that wait for a
just-spawned container's in-container server to come up.

The shape that recurs verbatim in the shared-backend browser envs
(mobilegym / webgym) is::

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if <healthy>:
            return  # success
        time.sleep(interval)
    raise ...        # timeout

:func:`wait_until_ready` is the loop only; the caller keeps its own
success-side-effects and its own timeout exception (the message/exception
type differs per env). It deliberately does NOT swallow predicate exceptions
or do per-iteration log-tailing — envs whose poll loops carry bespoke
per-iteration logic (captcha's proc-died check, the android KVM fast-fail
probes) keep their hand-rolled loops.
"""

from __future__ import annotations

import time
from collections.abc import Callable


def wait_until_ready(
    predicate: Callable[[], bool],
    *,
    timeout: float,
    interval: float,
) -> bool:
    """Poll ``predicate`` until it returns truthy or ``timeout`` elapses.

    Args:
        predicate: Cheap readiness check, called once per poll. Its exceptions
            propagate (callers here pass total functions that return bool).
        timeout: Wall-clock budget in seconds (``time.monotonic``).
        interval: Sleep between polls in seconds.

    Returns:
        True if ``predicate()`` became truthy within ``timeout``; False if the
        deadline passed first. The final poll is taken at loop entry, so a
        predicate that is already truthy returns immediately without sleeping.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False
