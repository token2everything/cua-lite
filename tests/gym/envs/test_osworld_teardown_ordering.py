"""The local OSWorld container remains visible throughout teardown.

``_teardown_existing`` must keep ``_container`` / ``_pending`` pointing at the
dying container UNTIL destroy() returns — an eager null blanks the
``external_resource_id`` while the container is still alive in docker,
inviting the drift reaper to race the env's own teardown (same fix as
mobileworld's ``_destroy_container``). Stub containers record whether the
attr still pointed at them AT destroy time; no docker.

Run: uv run pytest tests/gym/envs/test_osworld_teardown_ordering.py
"""
from __future__ import annotations

import asyncio

from lite.gym.envs.osworld.main import OSWorldEnv


class _StubContainer:
    """Records whether ``env.<attr>`` still pointed at this stub when
    destroy() ran (the section 7.4a ordering fact under test)."""

    name = "stub-container"

    def __init__(self, env, attr: str, *, raise_on_destroy: bool = False):
        self._env = env
        self._attr = attr
        self._raise = raise_on_destroy
        self.pointed_at_me_at_destroy: bool | None = None

    def destroy(self) -> None:
        self.pointed_at_me_at_destroy = getattr(self._env, self._attr) is self
        if self._raise:
            raise RuntimeError("synthetic destroy failure")


def test_teardown_nulls_attr_only_after_destroy_returns():
    env = OSWorldEnv.__new__(OSWorldEnv)
    stub = _StubContainer(env, "_container")
    env._container = stub
    env._pending = None

    asyncio.run(env._teardown_existing())

    assert stub.pointed_at_me_at_destroy is True, (
        "attr must still point at the container AT destroy time (no eager null)"
    )
    assert env._container is None, "attr must be nulled after destroy returns"
    assert env._pending is None


def test_teardown_nulls_attr_even_when_destroy_raises():
    env = OSWorldEnv.__new__(OSWorldEnv)
    stub = _StubContainer(env, "_container", raise_on_destroy=True)
    env._container = stub
    env._pending = None

    asyncio.run(env._teardown_existing())   # must not raise

    assert stub.pointed_at_me_at_destroy is True
    assert env._container is None, (
        "a failed destroy must still release the handle (finally-nulled)"
    )
