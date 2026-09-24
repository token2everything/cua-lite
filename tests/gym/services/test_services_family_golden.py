"""GOLDEN characterization of every env's CURRENTLY-INFERRED server props
("characterize before changing anything").

This is the no-regression safety net for the Stage-2 refactor that migrates the
env-server's ~10 capability-inference sites from incidental signals
(``isinstance(svc, EnvServices)``, ``live_ids() -> None`` overload) to the
declared ``family`` + its pure derivations. It pins
the exact properties those sites read today, so a refactor that
    changes any observable wiring fails HERE rather than silently in production
(container reaping / availability).

The companion parity assertions document that PURE and REMOTE envs may
register catalog ``EnvServices``
→ ``isinstance(svc, EnvServices)`` is True but ``has_lifecycle(family)`` is False.
Recovery's isinstance sites therefore parity-match *has-a-services-object*, which is
``reconcile_mode(family) is not NONE`` ONLY up to those harmless no-ops (the
services object's live_ids/reap are base no-ops, so including or excluding it reaps
nothing).

Run: uv run pytest tests/gym/services/test_services_family_golden.py -q
"""

from __future__ import annotations

import pytest
from gym.conftest import _production_cuaworld_leaves, import_env_main_or_skip, importable_env_ids

from lite.gym.services import (
    BackendFamily,
    EnvServices,
    ReconcileMode,
    WarmStrategy,
    family_of,
    has_lifecycle,
    reconcile_mode,
    services_for,
    warm_strategy,
)

# GOLDEN snapshot — captured by introspecting the real env main modules
# (2026-06, pre Stage-2). Columns:
#   family       — declared BackendFamily
#   has_svc      — services_for(env) isinstance EnvServices (today's reconcile/health gate)
#   ensure/health/live_ids/reap — does the services object OVERRIDE the EnvServices base?
GOLDEN: dict[str, dict] = {
    "screenspot_pro": dict(
        family=BackendFamily.PURE,
        has_svc=True,
        ensure=False,
        health=False,
        live_ids=False,
        reap=False,
    ),
    "osworld_g": dict(
        family=BackendFamily.PURE,
        has_svc=True,
        ensure=False,
        health=False,
        live_ids=False,
        reap=False,
    ),
    "osworld": dict(
        family=BackendFamily.DEDICATED,
        has_svc=True,
        ensure=True,
        health=False,
        live_ids=True,
        reap=True,
    ),
    "osworld_2": dict(
        family=BackendFamily.REMOTE,
        has_svc=True,
        ensure=False,
        health=False,
        live_ids=False,
        reap=False,
    ),
    "androidworld": dict(
        family=BackendFamily.DEDICATED,
        has_svc=True,
        ensure=True,
        health=True,
        live_ids=True,
        reap=True,
    ),
    "androidlab": dict(
        family=BackendFamily.DEDICATED,
        has_svc=True,
        ensure=True,
        health=True,
        live_ids=True,
        reap=True,
    ),
    "captcha": dict(
        family=BackendFamily.DEDICATED,
        has_svc=True,
        ensure=False,
        health=True,
        live_ids=True,
        reap=True,
    ),
    "lite.demo": dict(
        family=BackendFamily.DEDICATED,
        has_svc=True,
        ensure=True,
        health=False,
        live_ids=True,
        reap=True,
    ),
    "waa": dict(
        family=BackendFamily.DEDICATED,
        has_svc=True,
        ensure=True,
        health=False,
        live_ids=True,
        reap=True,
    ),
    "lite.osworld": dict(
        family=BackendFamily.DEDICATED,
        has_svc=True,
        ensure=True,
        health=False,
        live_ids=True,
        reap=True,
    ),
    "lite.cuagym": dict(
        family=BackendFamily.DEDICATED,
        has_svc=True,
        ensure=True,
        health=False,
        live_ids=True,
        reap=True,
    ),
    "lite.scalecua": dict(
        family=BackendFamily.DEDICATED,
        has_svc=True,
        ensure=True,
        health=False,
        live_ids=True,
        reap=True,
    ),
    "lite.cuaworld.pymol": dict(
        family=BackendFamily.DEDICATED,
        has_svc=True,
        ensure=True,
        health=False,
        live_ids=True,
        reap=True,
    ),
    "mobileworld": dict(
        family=BackendFamily.DEDICATED,
        has_svc=True,
        ensure=True,
        health=True,
        live_ids=True,
        reap=True,
    ),
    "webgym": dict(
        family=BackendFamily.SINGLETON,
        has_svc=True,
        ensure=True,
        health=True,
        live_ids=False,
        reap=True,
    ),
    "mobilegym": dict(
        family=BackendFamily.SINGLETON,
        has_svc=True,
        ensure=True,
        health=True,
        live_ids=False,
        reap=True,
    ),
    "webharbor.webvoyager": dict(
        family=BackendFamily.SINGLETON,
        has_svc=True,
        ensure=True,
        health=True,
        live_ids=False,
        reap=True,
    ),
    "online_mind2web": dict(
        family=BackendFamily.SINGLETON,
        has_svc=True,
        ensure=True,
        health=True,
        live_ids=False,
        reap=True,
    ),
    "browsergym": dict(
        family=BackendFamily.SINGLETON,
        has_svc=True,
        ensure=True,
        health=True,
        live_ids=False,
        reap=True,
    ),
}


def test_golden_coverage_floor():
    """At least 10 of the 18 golden envs must be importable on ANY host —
    FAILS (not skips) below the floor: silent zero-coverage is the failure
    mode the per-env skips must not reintroduce."""
    importable = importable_env_ids(GOLDEN)
    assert len(importable) >= 10, (
        f"only {len(importable)}/18 golden envs importable (vacuous coverage)"
    )


def _overrides(svc, name: str) -> bool:
    return getattr(type(svc), name, None) is not getattr(EnvServices, name)


@pytest.mark.parametrize("env_id", list(GOLDEN))
def test_inferred_props_match_golden(env_id):
    """Every env's CURRENT inferred props == the captured golden. Any refactor
    that changes observable wiring (which envs reconcile / report health)
    trips here."""
    import_env_main_or_skip(env_id)
    g = GOLDEN[env_id]
    svc = services_for(env_id)
    has_svc = isinstance(svc, EnvServices)
    assert has_svc == g["has_svc"], f"{env_id}: has_svc {has_svc} != {g['has_svc']}"
    if has_svc:
        for m in ("ensure", "health", "live_ids", "reap"):
            assert _overrides(svc, m) == g[m], f"{env_id}: {m} override {not g[m]}"


@pytest.mark.parametrize("env_id", list(GOLDEN))
def test_family_matches_golden(env_id):
    import_env_main_or_skip(env_id)
    assert family_of(env_id) is GOLDEN[env_id]["family"]


def test_all_cuaworld_leaves_match_pymol_golden():
    # own import — do NOT rely on a sibling item on the same xdist worker
    __import__("lite.gym.envs.lite.cuaworld.main")
    import importlib

    registry = importlib.import_module("lite.gym.registry")

    leaves = _production_cuaworld_leaves()
    assert len(leaves) == 40
    assert set(leaves) <= registry._declared_env_ids
    expected = GOLDEN["lite.cuaworld.pymol"]
    for env_id in leaves:
        svc = services_for(env_id)
        assert family_of(env_id) is expected["family"]
        assert isinstance(svc, EnvServices) is expected["has_svc"]
        for method in ("ensure", "health", "live_ids", "reap"):
            assert _overrides(svc, method) is expected[method]


@pytest.mark.parametrize("env_id", list(GOLDEN))
def test_derivations_consistent_with_family(env_id):
    """The section 2 pure derivations agree with the declared family for every env."""
    import_env_main_or_skip(env_id)
    fam = family_of(env_id)
    assert (
        reconcile_mode(fam)
        == {
            BackendFamily.DEDICATED: ReconcileMode.PER_INSTANCE,
            BackendFamily.SINGLETON: ReconcileMode.BOOT_ONLY,
            BackendFamily.PURE: ReconcileMode.NONE,
            BackendFamily.REMOTE: ReconcileMode.NONE,
        }[fam]
    )
    assert has_lifecycle(fam) == (fam in (BackendFamily.DEDICATED, BackendFamily.SINGLETON))
    if GOLDEN[env_id]["family"] is BackendFamily.SINGLETON:
        assert warm_strategy(env_id) is WarmStrategy.EAGER
    else:
        assert warm_strategy(env_id) is WarmStrategy.NONE


@pytest.mark.parametrize("env_id", list(GOLDEN))
def test_catalog_services_do_not_imply_owned_backend_lifecycle(env_id):
    """PURE and REMOTE catalog services have no local resources to reconcile."""
    import_env_main_or_skip(env_id)
    g = GOLDEN[env_id]
    if g["family"] in (BackendFamily.PURE, BackendFamily.REMOTE):
        assert g["has_svc"] is True
        assert has_lifecycle(g["family"]) is False
        # ...but its reconcile is a pure no-op: live_ids/reap NOT overridden.
        assert g["live_ids"] is False and g["reap"] is False


def test_cuaworld_pymol_importable_not_silently_skipped():
    """A conftest ``_env_main_module`` resolver revert degrades the
    ``lite.cuaworld.*`` rows to named SKIPs (ModuleNotFoundError) that the
    coverage floors are too coarse to notice — this must-import assert makes
    the revert loud (the U17 pattern from test_metadata_invariant)."""
    assert "lite.cuaworld.pymol" in importable_env_ids(GOLDEN)
