"""Dispatcher-level test for the drift reaper.

Exercises ``_reap_drift_one_cycle`` against a real ``State`` object
with a mocked ``reconcile_all`` so we can focus on dispatcher
correctness (build_state_view, cross-env validation, ghost-pop under
lock, admission_notify) without re-testing the helper.

Helper-level correctness is covered by ``test_reaper.py`` and
``test_reaper_quarantine.py``.

Run:
    uv run pytest tests/gym/remote/test_server_state_view.py -v -s
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from gym.remote.conftest import make_test_admission as _make_test_admission

from lite.gym.remote.scope import ServerScope
from lite.gym.remote.server import (
    State,
    _EnvSession,
    _reap_drift_one_cycle,
    build_state_view,
)
from lite.gym.types import ReapReport

_SCOPE = ServerScope(server_port=30200, token_hash="tok")


def _fake_state() -> State:
    """Bare State for the dispatcher; admission/policy fields are not
    touched by ``_reap_drift_one_cycle``."""
    state = State(admission=_make_test_admission(max_live_envs=100),
        idle_ttl_sec=3600.0,
    )
    state.server_port = 30200
    return state


def _fake_session(*, env_id: str, session_id: str, task_id: str,
                  token_hash: str = "deadbe",
                  created_at: float | None = None) -> _EnvSession:
    """Minimal _EnvSession; .env is mocked because the dispatcher only
    reads metadata fields."""
    if created_at is None:
        created_at = time.time()
    return _EnvSession(
        env=MagicMock(),
        env_key=f"{env_id}@{task_id}",
        env_id=env_id,
        session_id=session_id,
        token_hash=token_hash,
        created_at=created_at,
        last_active_at=created_at,
    )


# =============================================================================
# build_state_view
# =============================================================================

def test_build_state_view_groups_by_env_id():
    state = _fake_state()
    state.envs["a1"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t1",
    )
    state.envs["a2"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t2",
    )
    state.envs["d1"] = _fake_session(
        env_id="lite.demo", session_id="S", task_id="t",
    )

    sv = build_state_view(state)
    assert set(sv.by_env_id.keys()) == {"androidworld", "lite.demo"}
    assert {i.id for i in sv.by_env_id["androidworld"]} == {"a1", "a2"}
    assert {i.id for i in sv.by_env_id["lite.demo"]} == {"d1"}
    assert sv.snapshot_at > 0


def test_build_state_view_is_frozen():
    state = _fake_state()
    state.envs["a"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t",
    )
    sv = build_state_view(state)
    with pytest.raises((AttributeError, Exception)):
        sv.by_env_id = {}        # frozen dataclass
    inst = sv.by_env_id["androidworld"][0]
    with pytest.raises((AttributeError, Exception)):
        inst.id = "x"             # frozen dataclass


# =============================================================================
# Dispatcher behaviour (cross-env validation + ghost pop)
# =============================================================================

def _patch_reconcile_all(reports: list[ReapReport]):
    """Monkey-patch the module-level ``reconcile_all`` so the
    dispatcher sees controlled returns. Returns a restore function."""
    # ``recovery`` is imported lazily by the dispatcher, so it may not be in
    # sys.modules yet when this test runs first (KeyError under -n auto / in
    # isolation). import_module is idempotent and guarantees it's loaded.
    import importlib

    reg_mod = importlib.import_module("lite.gym.remote.recovery")
    orig = reg_mod.reconcile_all

    def fake_reconcile_all(state_view, scope, *, allowed_env_ids=None):
        return reports
    reg_mod.reconcile_all = fake_reconcile_all
    return lambda: setattr(reg_mod, "reconcile_all", orig)


@pytest.mark.asyncio
async def test_dispatcher_pops_legit_ghost_ids():
    state = _fake_state()
    state.envs["gA"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t",
    )
    state.envs["gB"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t",
    )

    restore = _patch_reconcile_all([
        ReapReport(orphans_reaped=2, ghost_ids=("gA",)),
    ])
    try:
        await _reap_drift_one_cycle(state, _SCOPE)
    finally:
        restore()

    assert "gA" not in state.envs
    assert "gB" in state.envs
    assert state.dead_envs_total == 1


@pytest.mark.asyncio
async def test_dispatcher_rejects_unknown_ghost_ids():
    """A hook returning an id NOT in the snapshot must NOT cause that
    id to leak past the cross-env validation. (state.envs.pop with
    None default is idempotent so this is a belt-and-suspenders check
    — the real value is no spurious dead_envs_total bump.)"""
    state = _fake_state()
    state.envs["real"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t",
    )

    restore = _patch_reconcile_all([
        ReapReport(
            orphans_reaped=0,
            ghost_ids=("real", "forged-id-from-nowhere"),
        ),
    ])
    try:
        await _reap_drift_one_cycle(state, _SCOPE)
    finally:
        restore()

    assert "real" not in state.envs              # legit ghost popped
    assert state.dead_envs_total == 1            # exactly one increment


@pytest.mark.asyncio
async def test_dispatcher_never_reaps_a_ghost_with_a_call_in_flight():
    """The drift reaper must honour ``in_flight`` like the idle reaper ("never
    close an env mid-call").

    A ghost really can be busy: ``/reset``'s outer retry loop destroys and
    re-creates the container INSIDE the held sema slot, so a reconcile snapshot
    taken in that window sees a live session with no container. Reaping it would
    release the admission slot and ``close()`` the env under a reset that was
    about to succeed.
    """
    state = _fake_state()
    live = _fake_session(env_id="androidworld", session_id="S", task_id="t")
    live.in_flight = 1                      # a /reset is awaited inside it
    live.last_active_at = time.time()       # ... and it started just now
    state.envs["busy"] = live
    state.envs["idle"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t",
    )
    state.admission.admit()
    state.admission.admit()

    restore = _patch_reconcile_all([
        ReapReport(orphans_reaped=0, ghost_ids=("busy", "idle")),
    ])
    try:
        await _reap_drift_one_cycle(state, _SCOPE)
    finally:
        restore()

    assert "busy" in state.envs, "reaped an env with a call in flight"
    assert "idle" not in state.envs, "the quiescent ghost must still be reaped"
    # Only the quiescent one released its slot / counted as dead.
    assert state.admission.in_flight == 1
    assert state.dead_envs_total == 1
    live.env.close.assert_not_called()


@pytest.mark.asyncio
async def test_dispatcher_reaps_a_deferred_ghost_once_its_call_drains():
    """Deferral is not abandonment: the reaper re-offers the ghost every cycle, so
    the instance is reaped as soon as ``in_flight`` returns to 0 — otherwise a
    busy ghost leaks its admission slot forever.
    """
    state = _fake_state()
    busy = _fake_session(env_id="androidworld", session_id="S", task_id="t")
    busy.in_flight = 1
    state.envs["g"] = busy
    state.admission.admit()

    restore = _patch_reconcile_all([
        ReapReport(orphans_reaped=0, ghost_ids=("g",)),
    ])
    try:
        await _reap_drift_one_cycle(state, _SCOPE)
        assert "g" in state.envs and state.admission.in_flight == 1

        busy.in_flight = 0          # the handler's ``finally`` ran
        await _reap_drift_one_cycle(state, _SCOPE)
    finally:
        restore()

    assert "g" not in state.envs
    assert state.admission.in_flight == 0
    assert state.dead_envs_total == 1


@pytest.mark.asyncio
async def test_dispatcher_no_ghosts_is_quiet():
    """Empty cycle (no orphans, no ghosts) does not touch state.envs
    and does not bump dead_envs_total."""
    state = _fake_state()
    state.envs["a"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t",
    )

    restore = _patch_reconcile_all([
        ReapReport(orphans_reaped=0, ghost_ids=()),
    ])
    try:
        await _reap_drift_one_cycle(state, _SCOPE)
    finally:
        restore()

    assert "a" in state.envs
    assert state.dead_envs_total == 0


@pytest.mark.asyncio
async def test_dispatcher_isolates_concurrent_modifications():
    """Snapshot is taken under the lock; subsequent dispatch runs over
    the FROZEN snapshot. An entry added AFTER the snapshot can't be
    classified as orphan/ghost this cycle (it's not in state_view)."""
    state = _fake_state()
    state.allowed_env_ids = {"androidworld"}
    state.envs["before-snapshot"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t",
    )

    # Track what reconcile_all gets — verify "after-snapshot" is NOT in it.
    captured = {}

    def fake_reap(state_view, scope, *, allowed_env_ids=None):
        captured["allowed_env_ids"] = allowed_env_ids
        captured["view_ids"] = {
            i.id for v in state_view.by_env_id.values() for i in v
        }
        return [ReapReport(orphans_reaped=0)]

    reg_mod = __import__("importlib").import_module("lite.gym.remote.recovery")
    orig = reg_mod.reconcile_all
    reg_mod.reconcile_all = fake_reap
    try:
        # Add an entry between when this coroutine yields control and
        # when the dispatcher actually reads state.envs. We can't truly
        # interleave with the lock-held snapshot in this single-thread
        # asyncio test, but we can verify the snapshot ONLY contains
        # what was present at snapshot time.
        await _reap_drift_one_cycle(state, _SCOPE)
    finally:
        reg_mod.reconcile_all = orig

    assert captured["view_ids"] == {"before-snapshot"}
    assert captured["allowed_env_ids"] == {"androidworld"}


@pytest.mark.asyncio
async def test_dispatcher_releases_l2_slot_on_ghost_pop():
    """When ghosts are popped, the L2 slot they were holding must be
    released so concurrent callers can admit."""
    state = _fake_state()
    state.envs["x"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t",
    )
    # Pre-admit a slot to match the live env; the reaper should release it.
    state.admission.admit()
    assert state.admission.in_flight == 1

    restore = _patch_reconcile_all([
        ReapReport(orphans_reaped=0, ghost_ids=("x",)),
    ])
    try:
        await _reap_drift_one_cycle(state, _SCOPE)
    finally:
        restore()

    assert state.admission.in_flight == 0
    assert "x" not in state.envs


@pytest.mark.asyncio
async def test_dispatcher_no_release_when_nothing_popped():
    """If only orphans were reaped (no ghosts), in_flight is unchanged."""
    state = _fake_state()
    state.envs["x"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t",
    )
    state.admission.admit()
    assert state.admission.in_flight == 1

    restore = _patch_reconcile_all([
        ReapReport(orphans_reaped=5, ghost_ids=()),
    ])
    try:
        await _reap_drift_one_cycle(state, _SCOPE)
    finally:
        restore()

    # Ghost-less reap: env stays in state.envs, slot stays held.
    assert state.admission.in_flight == 1
    assert "x" in state.envs


@pytest.mark.asyncio
async def test_dispatcher_aggregates_multiple_env_reports():
    """When multiple envs return reports, dispatcher sums orphans and
    unions ghost_ids correctly."""
    state = _fake_state()
    state.envs["aw1"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t",
    )
    state.envs["aw2"] = _fake_session(
        env_id="androidworld", session_id="S", task_id="t",
    )
    state.envs["d1"] = _fake_session(
        env_id="lite.demo", session_id="S", task_id="t",
    )

    restore = _patch_reconcile_all([
        ReapReport(orphans_reaped=3, ghost_ids=("aw1",)),
        ReapReport(orphans_reaped=2, ghost_ids=("d1",)),
    ])
    try:
        await _reap_drift_one_cycle(state, _SCOPE)
    finally:
        restore()

    # Both legit ghosts dropped:
    assert "aw1" not in state.envs
    assert "d1" not in state.envs
    # The non-ghost survives:
    assert "aw2" in state.envs
    assert state.dead_envs_total == 2


@pytest.mark.asyncio
async def test_dispatcher_survives_inner_exception():
    """If reconcile_all raises, the dispatcher's one-cycle helper
    propagates the exception (caller's loop catches it). We don't
    want the cycle to silently swallow programming errors — only
    transient I/O failures inside per-env hooks are swallowed by the
    hook itself."""
    state = _fake_state()
    reg_mod = __import__("importlib").import_module("lite.gym.remote.recovery")
    orig = reg_mod.reconcile_all
    reg_mod.reconcile_all = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("inner bug")
    )
    try:
        with pytest.raises(RuntimeError, match="inner bug"):
            await _reap_drift_one_cycle(state, _SCOPE)
    finally:
        reg_mod.reconcile_all = orig
